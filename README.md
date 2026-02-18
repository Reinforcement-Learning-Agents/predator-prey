#  Predator-Prey Reinforcement Learning

**Multi-Agent DQN with Partial Observability in a Custom OpenAI Gym Environment**

---

## Problem Description and Environment

This project implements a **multi-agent predator-prey system** in a custom discrete 2D grid environment. Two types of autonomous agents — predators and prey — operate simultaneously, each learning independent policies in an adversarial setting under **partial observability**.

The grid is a configurable 10×10 discrete space where agents spawn at random non-overlapping positions. At any step, a learning agent can only perceive opponents within a **Manhattan distance of 3 cells** (vision radius), making the task genuinely challenging: agents must explore when no opponent is visible, and learn spatial reasoning rather than simple reactive behavior.

Default agent configuration:
- 2 predators (1 learning, 1 random scripted)
- 2 prey (1 learning, 1 random scripted)

**Observation Space** — each agent receives a 4-dimensional continuous vector:

```
obs = [x_norm, y_norm, rel_dx, rel_dy]
```

- `x_norm`, `y_norm` — ego position normalized to `[0, 1]`
- `rel_dx`, `rel_dy` — relative position to nearest opponent, normalized by vision radius; set to `0.0` when no opponent is in range

**Action Space** — 5 discrete actions: Stay, Up, Down, Left, Right. Boundary clipping prevents out-of-bounds movement.

**Episode Termination** occurs when: (1) `steps >= 200`, (2) all prey are eliminated, or (3) the learning prey is captured.

---

## Reinforcement Learning Formulation

The reward functions are intentionally asymmetric to reflect the asymmetric nature of the task.

**Predator (sparse rewards):**
```
R_pred(t) = 10.0 × n_captured − 0.01
```
Capture events yield a large reward (10.0 per prey caught), while every step incurs a small penalty (−0.01). The 1000:1 capture-to-step ratio ensures the agent prioritizes hunting over idling.

**Prey (dense rewards):**
```
R_prey(t) = 0.1 − 10.0 × I_caught
```
The prey earns a small survival bonus (+0.1) every step it remains alive, and suffers a large penalty (−10.0) when caught. This dense positive signal provides stable gradients and faster convergence compared to the predator's sparse reward structure.

---

## Algorithmic Approach

Two algorithms are implemented and compared:

**Tabular Q-Learning (baseline):** A `defaultdict`-based Q-table maps discrete state tuples to action-value arrays. Continuous observations are cast to tuples for exact-match lookup. This approach has a fundamental limitation: it cannot generalize between similar states, and the table grows unboundedly with new unique observations. It serves as a meaningful lower bound for comparison.

**Deep Q-Network (DQN):** The primary algorithm, implemented via Stable-Baselines3. A two-layer MLP (64 units, ReLU) maps the continuous 4-dimensional observation directly to Q-values for each of the 5 actions. Key stabilization mechanisms include:

- **Experience replay** — 50,000-step buffer with random mini-batches (size 64), breaking temporal correlations
- **Target network** — separate frozen network updated every 1,000 steps, preventing divergence
- **ε-greedy exploration** — linear decay from 1.0 → 0.02 over the first 20% of training

DQN's neural network generalizes across the continuous observation space — something tabular methods fundamentally cannot do.

---

## Implementation Details

The codebase is split into two focused modules:

**`predator_prey_env.py`** — Custom Gym environment (~205 lines)

```python
class PredatorPreyEnv(gym.Env):
    # Core methods: reset(), step(), seed(), render()
    # Agent dataclass: fields x (int), y (int), alive (bool)
    # _nearest_relative(): O(n) Manhattan distance to all opponents,
    #   selects minimum, normalizes if within vision_radius
    # make_env(): factory function for SB3 DummyVecEnv wrapping
```

**`train_and_eval.py`** — Training and evaluation pipeline (~174 lines)

```python
# q_learning_baseline()  — tabular Q-learning with ε-greedy exploration
# train_dqn()            — DQN training via Stable-Baselines3
# rollout_stats()        — deterministic evaluation (100 episodes, ε=0)
# plot_learning_curves() — Q-learning vs. DQN reward comparison
# plot_hist()            — step distribution histograms
# main()                 — CLI with argparse (role, hyperparameters)
```

**Key compatibility challenge solved:** Stable-Baselines3 requires Gymnasium, while this environment is built on OpenAI Gym. This was resolved by installing `shimmy 2.0.0`, which acts as a translation bridge between the two APIs (handling `reset()` return format differences, `seed()` method signatures, etc.). SB3 was pinned to `2.7.1` for full Gymnasium 1.2.3 support.

---

## Experimental Setup and Evaluation

| Parameter | Value |
|---|---|
| Grid size | 10 × 10 |
| Vision radius | 3 (Manhattan) |
| Max steps/episode | 200 |
| DQN timesteps | 50,000 |
| Q-learning episodes | 400 |
| Learning rate | 1e-3 (DQN), 0.1 (Q-table) |
| Discount γ | 0.99 (DQN), 0.95 (Q-table) |
| Batch size | 64 |
| Replay buffer | 50,000 |
| Target update interval | 1,000 steps |
| Evaluation episodes | 100 (deterministic, ε=0) |
| Random seeds | Q-learning: 42 · DQN: 0 · Eval: 123/1234 |

**System:** Python 3.11.9, Windows 10, CPU-only (no GPU required)

---

## Results and Discussion

### Predator Agent

After 50,000 training timesteps (~456 episodes):

| Metric | Value |
|---|---|
| Mean evaluation reward | 505.60 ± 639.06 |
| Mean capture step | 120.1 / 200 (60%) |
| Training speed | ~165–168 FPS |

The high mean reward confirms that the predator successfully learned capture behavior. The large variance (±639.06) is expected and structurally caused by the sparse reward: episodes with multiple captures produce very high returns, while episodes with no captures yield near-zero or negative rewards from the step penalty. This is a known challenge in sparse-reward RL — not a failure of the algorithm.

### Prey Agent

After 50,000 training timesteps (~324 episodes):

| Metric | Value |
|---|---|
| Mean evaluation reward | 13.78 ± 10.40 |
| Mean survival steps | 165.8 / 200 (83%) |
| Training speed | ~162–174 FPS |

A mean survival of 165.8 steps (83% of the maximum episode length) shows the prey learned effective evasion. The much lower variance (±10.40) compared to the predator reflects the advantage of dense reward shaping: consistent per-step feedback produces more stable Q-value estimates and more consistent policies.

### DQN vs. Tabular Q-Learning

DQN outperforms tabular Q-learning for both agent types. The core reason is generalization: the neural network learns a continuous spatial representation of the environment, while Q-learning treats every unique observation tuple as an entirely distinct state. In a partially observable, continuous observation space, this makes Q-learning fundamentally ill-suited — it cannot transfer knowledge between nearby observations. DQN also uses a fixed parameter count (~10K weights) regardless of how many unique states are encountered, while the Q-table grows unboundedly.

---

## How to Run the Code

### 1. Clone and install dependencies

```bash
git clone https://github.com/iheb457/Predator-Prey
cd predator-prey-rl
pip install -r requirements.txt
```

**`requirements.txt`**
```
gym==0.26.2
stable-baselines3==2.7.1
shimmy==2.0.0
numpy==1.26.4
matplotlib==3.8.2
tensorboard==2.20.0
```

### 2. Train the predator agent

```bash
python train_and_eval.py --role predator
```

### 3. Train the prey agent

```bash
python train_and_eval.py --role prey
```

### 4. Monitor training with TensorBoard

```bash
tensorboard --logdir logs/
```

Logs are saved to `logs/DQN_{run_id}/` automatically during training.

### 5. Evaluate a trained model

The evaluation loop runs 100 deterministic episodes (ε=0) after training and prints mean reward and step statistics alongside learning curve plots.

---

## References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR*, 22(268).
- Watkins, C. J. C. H. (1989). Learning from delayed rewards. PhD Thesis, University of Cambridge.
- Brockman, G., et al. (2016). OpenAI Gym. *arXiv:1606.01540*.
- Lowe, R., et al. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *NeurIPS*, 30.

---

> A detailed technical Word report covering extended analysis, figures, and appendices is available in the repository under `Predator_Prey_RL_Report.docx`.

---

## Author
Project developed by **Iheb Bousselmi** (Student ID: 5870020) as part of a university reinforcement learning project.
