# Predator-Prey RL

2D grid predator-prey environment with partial observability. Q-learning baseline + DQN (Stable-Baselines3). One agent learns; others use random policies.

## Setup

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python train_and_eval.py --role predator --timesteps 50000
python train_and_eval.py --role prey --timesteps 50000
```

Flags: `--role` (predator/prey), `--timesteps`, `--episodes`, `--seed`.

## Environment

- Grid: 10x10, vision radius 3
- Obs: [x_norm, y_norm, rel_dx, rel_dy]
- Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
- Predator: +10 capture, -0.01/step
- Prey: +0.1/step, -10 if caught

## Output

`{role}_learning.png`, `{role}_survival_hist.png`, `logs/` for TensorBoard.
