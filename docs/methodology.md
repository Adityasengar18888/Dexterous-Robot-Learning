# Methodology & Reinforcement Learning Formulation

## Core Algorithmic Framework

### 1. Goal-Conditioned RL
Goal-conditioned RL extends standard Markov Decision Processes to include a goal space $\mathcal{G}$.
Policies $\pi(a_t \mid s_t, g)$ maximize the expected discounted return:
$$R_t = \sum_{k=0}^{\infty} \gamma^k r(s_{t+k}, a_{t+k}, g)$$

### 2. Observation Specification
In accordance with Gymnasium conventions, environments emit observations as a structured dictionary:
- `observation`: Proprioceptive state of the manipulator (EE coordinates, velocities, gripper state).
- `achieved_goal`: State of the object or end-effector corresponding to goal coordinates.
- `desired_goal`: Stationary or dynamic goal target sampled uniformly from the task workspace.

### 3. Reward Formulation
The primary research formulation is **Sparse Binary Reward**:
$$r(s, a, g) = \begin{cases} 0 & \text{if } \|s_{\text{achieved}} - g\|_2 \le \epsilon \\ -1 & \text{otherwise} \end{cases}$$
where $\epsilon = 0.05\text{ m}$ ($5\text{ cm}$) by default.
Dense shaping rewards ($r = -\|s_{\text{achieved}} - g\|_2$) are maintained strictly as diagnostic tools to verify gradient flow.

### 4. Hindsight Experience Replay (HER)
Sparse rewards cause severe sample inefficiency in naive off-policy RL.
HER stores transitions $(s_t, a_t, r_t, s_{t+1}, g)$ and creates additional virtual replay transitions by substituting the originally intended goal $g$ with an achieved goal $g' = s_{t+k}$ encountered later in the trajectory ("future" strategy with $k=4$ virtual goals per real transition).

### 5. Kinematics & Actuation Control
Cartesian delta commands $a \in [-1, 1]^4$ scale to physical displacements:
$$\Delta x = a_{0:3} \cdot \delta_{\text{max}}$$
Targets are clamped to safety bounds $[x_{\text{min}}, x_{\text{max}}] \times [y_{\text{min}}, y_{\text{max}}] \times [z_{\text{min}}, z_{\text{max}}]$ before solving Inverse Kinematics (IK) with null-space biasing toward neutral rest poses.
