"""Policy transfer utilities for Phase 15.2.5B -> 15.2.5C (Method B: Shared Feature Mapping).

Maps learned weights from a single-object SAC policy (22D obs, 3D ag, 3D dg -> 28D concatenated input)
into a two-object SAC policy (31D obs, 6D ag, 6D dg -> 43D concatenated input) with exact 1-to-1
column alignment for shared semantics and zero-initialized target-only slots so initial two-object
Object-1 behavior reproduces the source policy without random interference from Object 2.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from stable_baselines3 import SAC


# Source 28D actor feature vector ordering (CombinedExtractor sorts Dict keys alphabetically:
# achieved_goal(0:3), desired_goal(3:6), observation(6:28))!
# WAIT — VERIFY CombinedExtractor KEY ORDERING IN CODE BEFORE USING!
def get_dict_concat_slices(observation_space) -> Dict[str, Tuple[int, int]]:
    """Compute exact start:end slices produced by SB3 CombinedExtractor (iterates over space.spaces)."""
    offset = 0
    slices: Dict[str, Tuple[int, int]] = {}
    for key, subspace in observation_space.spaces.items():
        dim = int(subspace.shape[0])
        slices[key] = (offset, offset + dim)
        offset += dim
    return slices


def transfer_single_to_multi_sac(source_model: SAC, target_model: SAC) -> Dict[str, Any]:
    """Transfer shared actor and critic parameters from 1-object B to 2-object C cleanly."""
    src_slices = get_dict_concat_slices(source_model.observation_space)
    tgt_slices = get_dict_concat_slices(target_model.observation_space)

    # Verify expected dimensions
    assert src_slices["observation"][1] - src_slices["observation"][0] == 22
    assert src_slices["achieved_goal"][1] - src_slices["achieved_goal"][0] == 3
    assert src_slices["desired_goal"][1] - src_slices["desired_goal"][0] == 3
    assert tgt_slices["observation"][1] - tgt_slices["observation"][0] == 31
    assert tgt_slices["achieved_goal"][1] - tgt_slices["achieved_goal"][0] == 6
    assert tgt_slices["desired_goal"][1] - tgt_slices["desired_goal"][0] == 6

    # Build 1-to-1 column mapping from source input indices to target input indices
    col_map: list[Tuple[int, int]] = []
    # achieved_goal[0:3] -> tgt achieved_goal[0:3]
    for i in range(3):
        col_map.append((src_slices["achieved_goal"][0] + i, tgt_slices["achieved_goal"][0] + i))
    # desired_goal[0:3] -> tgt desired_goal[0:3]
    for i in range(3):
        col_map.append((src_slices["desired_goal"][0] + i, tgt_slices["desired_goal"][0] + i))
    # obs[0:4] (ee + gw) -> tgt obs[0:4]
    for i in range(4):
        col_map.append((src_slices["observation"][0] + i, tgt_slices["observation"][0] + i))
    # obs[4:13] (obj0) -> tgt obs[4:13]
    for i in range(9):
        col_map.append((src_slices["observation"][0] + 4 + i, tgt_slices["observation"][0] + 4 + i))
    # obs[13:22] (bins) -> tgt obs[22:31]
    for i in range(9):
        col_map.append((src_slices["observation"][0] + 13 + i, tgt_slices["observation"][0] + 22 + i))

    source_params = sum(p.numel() for p in source_model.policy.parameters())
    target_params = sum(p.numel() for p in target_model.policy.parameters())
    transferred_params = 0

    with torch.no_grad():
        # 1. Actor first linear layer (28 -> 43 input features; latent_pi[0].weight is (256, in_dim))
        src_w0 = source_model.policy.actor.latent_pi[0].weight.data
        src_b0 = source_model.policy.actor.latent_pi[0].bias.data
        tgt_w0 = target_model.policy.actor.latent_pi[0].weight.data
        tgt_b0 = target_model.policy.actor.latent_pi[0].bias.data

        # Zero new object-2 / goal-2 columns so Object 1 policy starts cleanly undisturbed
        tgt_w0.zero_()
        for s_col, t_col in col_map:
            tgt_w0[:, t_col].copy_(src_w0[:, s_col])
            transferred_params += src_w0.shape[0]
        tgt_b0.copy_(src_b0)
        transferred_params += src_b0.numel()

        # 2. Remaining actor layers (latent_pi[2], mu, log_std) - exact shape match
        for attr_path in [
            ("latent_pi", 2),
            ("mu", None),
            ("log_std", None),
        ]:
            src_mod = getattr(source_model.policy.actor, attr_path[0])
            tgt_mod = getattr(target_model.policy.actor, attr_path[0])
            if attr_path[1] is not None:
                src_mod = src_mod[attr_path[1]]
                tgt_mod = tgt_mod[attr_path[1]]
            for sp, tp in zip(src_mod.parameters(), tgt_mod.parameters()):
                assert sp.shape == tp.shape
                tp.copy_(sp)
                transferred_params += sp.numel()

        # 3. Critic & Critic-target shared layers (qf0, qf1: 28+4=32 -> 43+4=47 input features)
        act_dim = int(source_model.action_space.shape[0])
        src_in_total = src_w0.shape[1]
        tgt_in_total = tgt_w0.shape[1]
        for critic_name in ("critic", "critic_target"):
            src_c = getattr(source_model.policy, critic_name)
            tgt_c = getattr(target_model.policy, critic_name)
            for q_name in ("qf0", "qf1"):
                sq = getattr(src_c, q_name)
                tq = getattr(tgt_c, q_name)
                # Layer 0: [state_concat, action(4)]
                tq[0].weight.data.zero_()
                for s_col, t_col in col_map:
                    tq[0].weight.data[:, t_col].copy_(sq[0].weight.data[:, s_col])
                    transferred_params += sq[0].weight.data.shape[0]
                for a_i in range(act_dim):
                    tq[0].weight.data[:, tgt_in_total + a_i].copy_(sq[0].weight.data[:, src_in_total + a_i])
                    transferred_params += sq[0].weight.data.shape[0]
                tq[0].bias.data.copy_(sq[0].bias.data)
                transferred_params += sq[0].bias.data.numel()
                # Remaining linear layers (idx 2, 4)
                for idx in (2, 4):
                    for sp, tp in zip(sq[idx].parameters(), tq[idx].parameters()):
                        assert sp.shape == tp.shape
                        tp.copy_(sp)
                        transferred_params += sp.numel()

        # Transfer log_ent_coef cleanly
        if hasattr(source_model, "log_ent_coef") and hasattr(target_model, "log_ent_coef"):
            target_model.log_ent_coef.data.copy_(source_model.log_ent_coef.data)

    newly_init_or_unmapped = target_params - transferred_params
    return {
        "strategy": "Method_B_Shared_Encoder_And_Actor_Critic_Slice_Mapping",
        "source_total_params": int(source_params),
        "target_total_params": int(target_params),
        "transferred_params": int(transferred_params),
        "newly_initialized_params": int(newly_init_or_unmapped),
        "mapped_input_cols": len(col_map),
        "unmapped_new_input_cols": tgt_in_total - len(col_map),
        "frozen_layers": [],
        "unfrozen_layers": ["actor", "critic", "critic_target"],
    }

