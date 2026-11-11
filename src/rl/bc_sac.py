"""Behavior-Cloning Auxiliary SAC + Configurable Entropy Floor (Phase 1 / 15.2.5B.2)."""

from __future__ import annotations

import pickle
from typing import Any, Dict, Optional

import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class BCDecayingSAC(SAC):
    """SAC subclass adding decaying auxiliary BC loss and a configurable entropy floor.

    Total actor objective:
        L_total = L_SAC + lambda_BC(t) * L_BC
    where lambda_BC(t) decays linearly from bc_lambda_init toward bc_lambda_final
    over bc_decay_steps, and log_ent_coef is clamped to >= log(min_ent_coef).
    Saved policy checkpoints remain 100% standard SAC MultiInputPolicy archives.
    """

    def __init__(
        self,
        *args: Any,
        bc_demos_path: Optional[str] = None,
        bc_lambda_init: float = 0.305,
        bc_lambda_final: float = 0.02,
        bc_decay_steps: int = 5000,
        bc_batch_size: int = 128,
        min_ent_coef: float = 0.02,
        actor_update_start_step: int = 0,
        bc_normalize_q: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bc_lambda_init = float(bc_lambda_init)
        self.bc_lambda_final = float(bc_lambda_final)
        self.bc_decay_steps = int(max(1, bc_decay_steps))
        self.bc_batch_size = int(max(1, bc_batch_size))
        self.min_ent_coef = float(max(0.0, min_ent_coef))
        self.actor_update_start_step = int(max(0, actor_update_start_step))
        self.bc_normalize_q = bool(bc_normalize_q)

        self._demo_obs: Optional[Dict[str, th.Tensor]] = None
        self._demo_act: Optional[th.Tensor] = None
        self._demo_n: int = 0
        self._bc_rng = np.random.default_rng(int(kwargs.get("seed", 42) or 42))

        if bc_demos_path:
            self.load_bc_demonstrations(bc_demos_path)

    def load_bc_demonstrations(self, demos_path: str) -> int:
        """Load validated demonstration (obs, action) pairs onto model device."""
        with open(demos_path, "rb") as f:
            demos = pickle.load(f)
        if not demos:
            raise ValueError(f"Empty demonstration dataset at {demos_path}")

        obs_keys = list(demos[0]["observations"][0].keys())
        obs_lists: Dict[str, list[np.ndarray]] = {k: [] for k in obs_keys}
        act_list: list[np.ndarray] = []

        for ep in demos:
            for t in range(int(ep["length"])):
                act = np.asarray(ep["actions"][t], dtype=np.float32)
                if not np.all(np.isfinite(act)) or np.any(np.abs(act) > 1.0001):
                    raise ValueError(f"Invalid demo action outside [-1,1]: {act}")
                for k in obs_keys:
                    arr = np.asarray(ep["observations"][t][k], dtype=np.float32)
                    if not np.all(np.isfinite(arr)):
                        raise ValueError(f"Non-finite demo observation key={k}")
                    obs_lists[k].append(arr)
                act_list.append(np.clip(act, -1.0, 1.0))

        self._demo_obs = {
            k: th.as_tensor(np.stack(v, axis=0), device=self.device, dtype=th.float32)
            for k, v in obs_lists.items()
        }
        self._demo_act = th.as_tensor(np.stack(act_list, axis=0), device=self.device, dtype=th.float32)
        self._demo_n = int(self._demo_act.shape[0])
        return self._demo_n

    def current_bc_lambda(self) -> float:
        progress = min(1.0, float(self.num_timesteps) / float(self.bc_decay_steps))
        return float(self.bc_lambda_final + (self.bc_lambda_init - self.bc_lambda_final) * (1.0 - progress))

    def _sample_bc_batch(self, batch_size: int) -> tuple[Dict[str, th.Tensor], th.Tensor]:
        assert self._demo_obs is not None and self._demo_act is not None and self._demo_n > 0
        idx = self._bc_rng.integers(0, self._demo_n, size=min(batch_size, self._demo_n))
        idx_t = th.as_tensor(idx, device=self.device, dtype=th.long)
        obs_b = {k: v[idx_t] for k, v in self._demo_obs.items()}
        act_b = self._demo_act[idx_t]
        return obs_b, act_b

    def run_bc_warmstart(self, steps: int = 800, lr: float = 0.001, batch_size: int = 128) -> Dict[str, float]:
        """Short supervised BC warm-start on actor weights before SAC+HER updates."""
        assert self._demo_obs is not None and self._demo_act is not None and self._demo_n > 0
        self.policy.set_training_mode(True)
        with th.no_grad():
            init_pred = self.actor(self._demo_obs, deterministic=True)
            init_mse = float(F.mse_loss(init_pred, self._demo_act).item())

        opt = th.optim.Adam(self.actor.parameters(), lr=float(lr))
        last_mse = init_mse
        for _ in range(int(steps)):
            obs_b, act_b = self._sample_bc_batch(batch_size)
            pred = self.actor(obs_b, deterministic=True)
            loss = F.mse_loss(pred, act_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_mse = float(loss.item())

        with th.no_grad():
            if hasattr(self.actor, "log_std") and hasattr(self.actor.log_std, "bias") and self.actor.log_std.bias is not None:
                self.actor.log_std.bias.fill_(-1.2)
            final_pred = self.actor(self._demo_obs, deterministic=True)
            final_mse = float(F.mse_loss(final_pred, self._demo_act).item())
            open_mask = self._demo_act[:, 3] > 0.0
            prem_rate = float(th.mean((final_pred[open_mask, 3] <= 0.0).float()).item()) if th.any(open_mask) else 0.0

        return {
            "bc_pretrain_steps": int(steps),
            "initial_mse": round(init_mse, 6),
            "final_mse": round(final_mse, 6),
            "last_batch_mse": round(last_mse, 6),
            "open_premature_close_rate": round(prem_rate, 6),
        }

    def run_critic_warmup(self, steps: int = 1000, batch_size: int = 128) -> Dict[str, float]:
        """Pre-train critic on seeded HER replay buffer (actor frozen) so Q-values are aligned before RL."""
        self.policy.set_training_mode(True)
        losses: list[float] = []
        for step_i in range(int(steps)):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                ent_coef = th.exp(self.log_ent_coef.detach()) if self.log_ent_coef is not None else self.ent_coef_tensor
                if self.min_ent_coef > 0.0:
                    ent_coef = th.clamp(ent_coef, min=self.min_ent_coef)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            losses.append(float(critic_loss.item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            if step_i % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
        return {
            "critic_warmup_steps": int(steps),
            "initial_critic_loss": round(losses[0], 6) if losses else 0.0,
            "final_critic_loss": round(float(np.mean(losses[-50:])), 6) if losses else 0.0,
        }

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        sac_actor_losses, bc_losses, total_actor_losses, critic_losses = [], [], [], []
        imitation_mses, prem_close_rates = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                if self.min_ent_coef > 0.0:
                    ent_coef = th.clamp(ent_coef, min=self.min_ent_coef)
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(float(ent_coef_loss.item()))
            else:
                ent_coef = self.ent_coef_tensor
                if self.min_ent_coef > 0.0:
                    ent_coef = th.clamp(ent_coef, min=self.min_ent_coef)

            ent_coefs.append(float(ent_coef.item()))

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                if self.min_ent_coef > 0.0 and self.log_ent_coef is not None:
                    with th.no_grad():
                        self.log_ent_coef.clamp_(min=float(np.log(self.min_ent_coef)))

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(float(critic_loss.item()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # SAC actor objective
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            sac_actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            # Auxiliary BC objective (L_total = L_SAC + lambda_BC * L_BC, optional TD3+BC |Q| scaling)
            lam_bc = self.current_bc_lambda()
            if self._demo_obs is not None and self._demo_act is not None and lam_bc > 0.0:
                obs_bc, act_bc = self._sample_bc_batch(self.bc_batch_size)
                pred_bc = self.actor(obs_bc, deterministic=True)
                bc_loss = F.mse_loss(pred_bc, act_bc)
                q_scale = th.clamp(th.abs(min_qf_pi).mean().detach(), min=1.0) if self.bc_normalize_q else 1.0
                total_actor_loss = sac_actor_loss + (lam_bc * q_scale) * bc_loss
                bc_val = float(bc_loss.item())
                # Track imitation MSE & premature closure on open-demo samples (act_bc[:,3] > 0)
                with th.no_grad():
                    open_mask = act_bc[:, 3] > 0.0
                    if th.any(open_mask):
                        prem_rate = float(th.mean((pred_bc[open_mask, 3] <= 0.0).float()).item())
                    else:
                        prem_rate = 0.0
                imitation_mses.append(bc_val)
                prem_close_rates.append(prem_rate)
            else:
                bc_loss = th.zeros((), device=self.device)
                total_actor_loss = sac_actor_loss
                bc_val = 0.0

            sac_actor_losses.append(float(sac_actor_loss.item()))
            bc_losses.append(bc_val)
            total_actor_losses.append(float(total_actor_loss.item()))

            self.actor.optimizer.zero_grad()
            total_actor_loss.backward()
            if self.num_timesteps >= self.actor_update_start_step:
                self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(total_actor_losses))
        self.logger.record("train/sac_actor_loss", np.mean(sac_actor_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/total_actor_loss", np.mean(total_actor_losses))
        self.logger.record("train/bc_lambda", self.current_bc_lambda())
        if imitation_mses:
            self.logger.record("train/imitation_mse", np.mean(imitation_mses))
        if prem_close_rates:
            self.logger.record("train/bc_open_premature_close_rate", np.mean(prem_close_rates))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

