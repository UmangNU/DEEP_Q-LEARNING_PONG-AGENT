"""
Model Evaluation Script

ATTRIBUTION & LICENSE:
- Evaluation methodology: Standard RL evaluation
- Implementation: Original work by Umang Mistry

Author: Umang Mistry  
Date: November 2024
License: MIT
"""

import gymnasium as gym
import ale_py
import numpy as np
import time
import sys

from agent import DQNAgent
from config import Config
from utils import stack_frames

gym.register_envs(ale_py)


def evaluate(model_path, n_episodes=10, render=True, use_boltzmann=False):
    """
    Evaluate a trained agent
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to run
        render: Whether to display the game
        use_boltzmann: Use Boltzmann exploration instead of ε-greedy
    """
    # Create environment
    render_mode = 'human' if render else 'rgb_array'
    env = gym.make(Config.ENV_NAME, render_mode=render_mode)
    
    print("=" * 70)
    print("🎮 EVALUATING DQN AGENT")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Policy: {'Boltzmann' if use_boltzmann else 'ε-greedy'}")
    print("=" * 70)
    
    # Create and load agent
    agent = DQNAgent(env.action_space.n, Config())
    agent.load(model_path)
    agent.epsilon = 0.01  # Minimal exploration for evaluation
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        stacked_state, frame_stack = stack_frames(None, state, True)
        
        episode_reward = 0
        step = 0
        
        while True:
            # Select action
            if use_boltzmann:
                action = agent.select_action_boltzmann(stacked_state, temperature=0.5)
            else:
                action = agent.select_action(stacked_state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process state
            next_stacked_state, frame_stack = stack_frames(
                frame_stack, next_state, False
            )
            
            episode_reward += reward
            stacked_state = next_stacked_state
            step += 1
            
            if render:
                time.sleep(0.01)  # Slow down for viewing
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        print(f"  Reward: {episode_reward:.1f}")
        print(f"  Length: {step} steps")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Best Episode: {max(episode_rewards):.1f}")
    print(f"Worst Episode: {min(episode_rewards):.1f}")
    print(f"Win Rate: {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards) * 100:.1f}%")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    # Default to final model
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/baseline_final.pth'
    
    evaluate(model_path, n_episodes=10, render=True)