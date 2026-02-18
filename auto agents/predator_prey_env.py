"""
Predator-prey environment for RL. 2D grid, partial observability (vision radius).
One agent learns; others use random policies.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple

import gym
import numpy as np
from gym import spaces


@dataclass
class Agent:
    """Agent state: grid position and alive flag."""
    x: int
    y: int
    alive: bool = True


class PredatorPreyEnv(gym.Env):
    """
    Grid predator-prey. obs=[x_norm, y_norm, rel_dx, rel_dy]. Actions: 0=stay, 1-4=dir.
    Predator: +10 capture, -0.01/step. Prey: +0.1/step, -10 if caught.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        grid_size: int = 10,
        n_predators: int = 2,
        n_prey: int = 2,
        vision_radius: int = 3,
        max_steps: int = 200,
        train_role: str = "predator",
        seed: int = 1,
    ):
        """Initialize environment. train_role: which agent learns (predator/prey)."""
        super().__init__()
        assert train_role in ("predator", "prey")
        self.grid_size = grid_size
        self.n_predators = n_predators
        self.n_prey = n_prey
        self.vision_radius = vision_radius
        self.max_steps = max_steps
        self.train_role = train_role
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.predators: List[Agent] = []
        self.prey: List[Agent] = []
        self.steps = 0
    
    def seed(self, seed: int = None):
        """Set random seed (Gym compat). Returns [seed] or []."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return [seed] if seed is not None else []

    def reset(self):
        self.steps = 0
        self.predators = self._spawn(self.n_predators)
        self.prey = self._spawn(self.n_prey)
        return self._get_obs()

    def step(self, action: int):
        self.steps += 1
        if self.train_role == "predator":
            self._move_agent(self.predators[0], action)
            self._move_scripted_prey()
            self._move_scripted_predators(skip_first=True)
        else:
            self._move_agent(self.prey[0], action)
            self._move_scripted_predators()
            self._move_scripted_prey(skip_first=True)
        reward = self._compute_reward()
        done = self._is_done()
        obs = self._get_obs()
        info = {"steps": self.steps, "prey_alive": sum(p.alive for p in self.prey)}
        return obs, reward, done, info

    def _spawn(self, count: int) -> List[Agent]:
        coords = set()
        agents: List[Agent] = []
        while len(agents) < count:
            x, y = self.rng.integers(0, self.grid_size, size=2)
            if (x, y) in coords:
                continue
            coords.add((x, y))
            agents.append(Agent(int(x), int(y)))
        return agents

    def _move_agent(self, agent: Agent, action: int):
        if not agent.alive:
            return
        dx, dy = 0, 0
        if action == 1:
            dy = -1
        elif action == 2:
            dy = 1
        elif action == 3:
            dx = -1
        elif action == 4:
            dx = 1
        agent.x = int(np.clip(agent.x + dx, 0, self.grid_size - 1))
        agent.y = int(np.clip(agent.y + dy, 0, self.grid_size - 1))

    def _move_scripted_predators(self, skip_first: bool = False):
        for idx, predator in enumerate(self.predators):
            if skip_first and idx == 0:
                continue
            self._move_agent(predator, self.rng.integers(0, self.action_space.n))

    def _move_scripted_prey(self, skip_first: bool = False):
        for idx, prey in enumerate(self.prey):
            if skip_first and idx == 0:
                continue
            if not prey.alive:
                continue
            self._move_agent(prey, self.rng.integers(0, self.action_space.n))

    def _compute_reward(self) -> float:
        for predator in self.predators:
            for prey in self.prey:
                if prey.alive and predator.x == prey.x and predator.y == prey.y:
                    prey.alive = False
        prey_caught = sum(not p.alive for p in self.prey)
        if self.train_role == "predator":
            reward = 10.0 * prey_caught - 0.01
        else:
            reward = 0.1
            if prey_caught > 0 and not self.prey[0].alive:
                reward -= 10.0
        return reward

    def _is_done(self) -> bool:
        return (self.steps >= self.max_steps or 
                all(not p.alive for p in self.prey) or 
                not self.prey[0].alive)

    def _get_obs(self) -> np.ndarray:
        if self.train_role == "predator":
            ego = self.predators[0]
            opponents = [p for p in self.prey if p.alive]
        else:
            ego = self.prey[0]
            opponents = [p for p in self.predators if p.alive]
        dx, dy = self._nearest_relative(ego, opponents)
        obs = np.array(
            [
                ego.x / (self.grid_size - 1),
                ego.y / (self.grid_size - 1),
                dx,
                dy,
            ],
            dtype=np.float32,
        )
        return obs

    def _nearest_relative(self, ego: Agent, others: List[Agent]) -> Tuple[float, float]:
        if not others:
            return 0.0, 0.0
        min_dist = float("inf")
        rel = (0.0, 0.0)
        for agent in others:
            dx = agent.x - ego.x
            dy = agent.y - ego.y
            manhattan = abs(dx) + abs(dy)
            if manhattan < min_dist:
                min_dist = manhattan
                rel = (dx / self.vision_radius, dy / self.vision_radius) if manhattan <= self.vision_radius else (0.0, 0.0)
        
        return rel

    def render(self, mode="human"):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)
        for prey in self.prey:
            if prey.alive:
                grid[prey.y, prey.x] = "p"
        for predator in self.predators:
            symbol = "A" if (self.train_role == "predator" and predator is self.predators[0]) else "P"
            grid[predator.y, predator.x] = symbol
        print("\n".join(" ".join(row) for row in grid))


def make_env(train_role: str, seed: int = 0) -> gym.Env:
    """Create predator-prey env. train_role: predator or prey."""
    return PredatorPreyEnv(train_role=train_role, seed=seed)
