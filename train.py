"""
Training Script with Checkpointing

ATTRIBUTION & LICENSE:
- Training loop structure: Standard RL training pattern
- Checkpoint system: Original implementation by Umang Mistry
- All code: Original work by Umang Mistry

Author: Umang Mistry
Date: November 2024
License: MIT
"""

import gymnasium as gym
import ale_py
import numpy as np
import os
import json
from datetime import datetime

from agent import DQNAgent
from config import Config
from utils import stack_frames, set_random_seeds

# Register ALE environments
gym.register_envs(ale_py)


def train(config=None, experiment_name="baseline"):
    """
    Main training function
    
    Args:
        config: Configuration object (default: Config())
        experiment_name: Name for this experiment run
    """
    if config is None:
        config = Config()
    
    # Set random seeds for reproducibility
    set_random_seeds(config.RANDOM_SEED)
    
    # Create environment
    env = gym.make(config.ENV_NAME)
    
    print("=" * 70)
    print("🎮 DQN PONG AGENT TRAINING")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Environment: {config.ENV_NAME}")
    print(f"Total Episodes: {config.TOTAL_EPISODES}")
    print(f"Learning Rate (α): {config.LEARNING_RATE}")
    print(f"Discount Factor (γ): {config.GAMMA}")
    print(f"Epsilon Start: {config.EPSILON_START}")
    print(f"Epsilon Min: {config.EPSILON_MIN}")
    print(f"Epsilon Decay: {config.EPSILON_DECAY}")
    print(f"Random Seed: {config.RANDOM_SEED}")
    print("=" * 70)
    
    # Create agent
    agent = DQNAgent(env.action_space.n, config)
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    
    # Check for checkpoint to resume from
    start_episode = 0
    checkpoint_path = f'checkpoints/{experiment_name}_checkpoint.pth'
    
    if config.RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_path):
        print(f"\n📂 Resuming from checkpoint: {checkpoint_path}")
        start_episode, episode_rewards = agent.load_checkpoint(checkpoint_path)
        print(f"Resuming from episode {start_episode + 1}")
        print("=" * 70)
    
    # Training loop
    for episode in range(start_episode, config.TOTAL_EPISODES):
        # Reset environment
        state, _ = env.reset()
        stacked_state, frame_stack = stack_frames(None, state, True)
        
        episode_reward = 0
        episode_loss_list = []
        step = 0
        
        # Episode loop
        while True:
            # Select action
            action = agent.select_action(stacked_state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process next state
            next_stacked_state, frame_stack = stack_frames(
                frame_stack, next_state, False
            )
            
            # Store experience
            agent.store(stacked_state, action, reward, next_stacked_state, done)
            
            # Train agent
            loss = agent.train()
            if loss is not None:
                episode_loss_list.append(loss)
            
            # Update for next iteration
            episode_reward += reward
            stacked_state = next_stacked_state
            step += 1
            
            if done:
                break
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        avg_loss = np.mean(episode_loss_list) if episode_loss_list else 0
        episode_losses.append(avg_loss)
        
        # Logging
        if (episode + 1) % config.LOG_EVERY == 0:
            # Calculate averages
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            recent_lengths = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
            recent_losses = episode_losses[-100:] if len(episode_losses) >= 100 else episode_losses
            
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            
            print(f"\n📊 Episode {episode + 1}/{config.TOTAL_EPISODES}")
            print(f"   Reward: {episode_reward:.1f} | Avg(100): {avg_reward:.2f}")
            print(f"   Length: {step} steps | Avg(100): {avg_length:.1f}")
            print(f"   Loss: {avg_loss:.6f} | Epsilon: {agent.epsilon:.4f}")
            print(f"   Memory: {len(agent.memory)}/{config.MEMORY_SIZE} | Steps: {agent.steps}")
        
        # Save checkpoint
        if (episode + 1) % config.CHECKPOINT_EVERY == 0:
            agent.save_checkpoint(
                episode + 1, 
                episode_rewards, 
                checkpoint_path
            )
            print(f"💾 Checkpoint saved at episode {episode + 1}")
        
        # Save model periodically
        if (episode + 1) % config.SAVE_EVERY == 0:
            model_path = f'models/{experiment_name}_ep{episode + 1}.pth'
            agent.save(model_path)
            print(f"💾 Model saved: {model_path}")
    
    # Final save
    final_model_path = f'models/{experiment_name}_final.pth'
    agent.save(final_model_path)
    
    # Save training statistics
    stats = {
        'experiment_name': experiment_name,
        'total_episodes': config.TOTAL_EPISODES,
        'learning_rate': config.LEARNING_RATE,
        'gamma': config.GAMMA,
        'epsilon_start': config.EPSILON_START,
        'epsilon_min': config.EPSILON_MIN,
        'epsilon_decay': config.EPSILON_DECAY,
        'final_epsilon': agent.epsilon,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_reward_last_100': float(np.mean(episode_rewards[-100:])),
        'avg_length_last_100': float(np.mean(episode_lengths[-100:])),
        'best_episode_reward': float(max(episode_rewards)),
        'worst_episode_reward': float(min(episode_rewards)),
        'total_steps': agent.steps,
        'timestamp': datetime.now().isoformat()
    }
    
    stats_path = f'logs/{experiment_name}_stats.json'
    os.makedirs('logs', exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print final statistics
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Final Avg Reward (100ep): {stats['avg_reward_last_100']:.2f}")
    print(f"Final Avg Length (100ep): {stats['avg_length_last_100']:.1f}")
    print(f"Best Episode: {stats['best_episode_reward']:.1f}")
    print(f"Worst Episode: {stats['worst_episode_reward']:.1f}")
    print(f"Total Steps: {stats['total_steps']}")
    print(f"Final Epsilon: {stats['final_epsilon']:.4f}")
    print(f"\nModel saved: {final_model_path}")
    print(f"Stats saved: {stats_path}")
    print("=" * 70)
    
    env.close()
    return stats


if __name__ == "__main__":
    # Run baseline training
    train(Config(), "baseline")