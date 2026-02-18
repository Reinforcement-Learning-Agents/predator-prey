"""
Training and evaluation: Q-learning baseline + DQN (Stable-Baselines3).
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from predator_prey_env import make_env


def q_learning_baseline(
    role: str,
    episodes: int = 300,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
) -> Tuple[Dict[Tuple, np.ndarray], List[float]]:
    """Tabular Q-learning baseline. States discretized by tuple-casting obs."""
    env = make_env(train_role=role, seed=42)
    q_table: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(env.action_space.n))
    episode_rewards: List[float] = []
    for ep in range(episodes):
        state = tuple(env.reset())
        done = False
        total_reward = 0.0
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))
            obs, reward, done, _ = env.step(action)
            next_state = tuple(obs)
            total_reward += reward
            best_next = np.max(q_table[next_state])
            td_target = reward + gamma * best_next * (not done)
            q_table[state][action] += alpha * (td_target - q_table[state][action])
            state = next_state
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
    
    return q_table, episode_rewards


@dataclass
class TrainConfig:
    """DQN training hyperparameters."""
    role: str
    total_timesteps: int = 50_000
    learning_rate: float = 1e-3
    buffer_size: int = 50_000
    batch_size: int = 64
    gamma: float = 0.99
    exploration_fraction: float = 0.2
    exploration_final_eps: float = 0.02
    train_freq: int = 1
    target_update_interval: int = 1_000
    seed: int = 0


def train_dqn(cfg: TrainConfig, log_dir: str = "logs") -> DQN:
    env = DummyVecEnv([lambda: make_env(train_role=cfg.role, seed=cfg.seed)])
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        exploration_fraction=cfg.exploration_fraction,
        exploration_final_eps=cfg.exploration_final_eps,
        train_freq=cfg.train_freq,
        target_update_interval=cfg.target_update_interval,
        verbose=1,
        seed=cfg.seed,
        tensorboard_log=log_dir,
    )
    model.learn(total_timesteps=cfg.total_timesteps)
    return model


def evaluate(model: DQN, role: str, n_episodes: int = 50):
    env = DummyVecEnv([lambda: make_env(train_role=role, seed=123)])
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    return mean_reward, std_reward


def rollout_stats(model: DQN, role: str, n_episodes: int = 50) -> Tuple[List[float], List[int]]:
    env = make_env(train_role=role, seed=1234)
    episode_rewards: List[float] = []
    survival_steps: List[int] = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action))
            total_r += reward
        episode_rewards.append(total_r)
        survival_steps.append(info.get("steps", 0))
    
    return episode_rewards, survival_steps


def plot_learning_curves(q_rewards: List[float], dqn_rewards: List[float], role: str, out_path: str):
    plt.figure(figsize=(8, 4))
    plt.plot(q_rewards, label="Q-learning")
    plt.plot(dqn_rewards, label="DQN (episode rewards)", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Learning curves for {role}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_hist(values: List[int], title: str, xlabel: str, out_path: str):
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Predator-Prey RL training")
    parser.add_argument("--role", choices=["predator", "prey"], default="predator",
                        help="Agent role to train (default: predator)")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="DQN training timesteps (default: 50000)")
    parser.add_argument("--episodes", type=int, default=400,
                        help="Q-learning baseline episodes (default: 400)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility (default: 0)")
    args = parser.parse_args()

    print(f"Running Q-learning baseline for {args.role}...")
    _, q_rewards = q_learning_baseline(role=args.role, episodes=args.episodes)
    print(f"Training DQN for {args.role}...")
    cfg = TrainConfig(role=args.role, total_timesteps=args.timesteps, seed=args.seed)
    model = train_dqn(cfg)
    print(f"Evaluating DQN for {args.role}...")
    dqn_rewards, survival_steps = rollout_stats(model, role=args.role, n_episodes=100)
    mean_r, std_r = np.mean(dqn_rewards), np.std(dqn_rewards)
    print(f"[{args.role}] DQN mean reward: {mean_r:.2f} ± {std_r:.2f}")
    print(f"[{args.role}] Mean survival/capture steps: {np.mean(survival_steps):.1f}")
    print(f"Generating plots for {args.role}...")
    plot_learning_curves(q_rewards, dqn_rewards, args.role, 
                        out_path=f"{args.role}_learning.png")
    hist_title = "Prey survival steps" if args.role == "prey" else "Predator capture steps"
    plot_hist(
        survival_steps,
        title=hist_title,
        xlabel="Steps",
        out_path=f"{args.role}_survival_hist.png",
    )
    print(f"Training complete! Check {args.role}_learning.png and {args.role}_survival_hist.png")


if __name__ == "__main__":
    main()
