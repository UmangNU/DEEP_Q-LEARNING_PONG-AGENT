"""
Hyperparameter Experiment Runner

ATTRIBUTION & LICENSE:
- Experiment framework: Original implementation by Umang Mistry
- Compares different gamma, alpha, epsilon configurations

Author: Umang Mistry
Date: November 2024
License: MIT
"""

import os
from train import train
from config import Config, ExperimentConfigs


def run_all_experiments():
    """
    Run all required experiments for the assignment
    """
    print("\n" + "=" * 70)
    print("🧪 RUNNING ALL EXPERIMENTS")
    print("=" * 70)
    
    experiments = [
        ("baseline", ExperimentConfigs.baseline()),
        ("high_gamma", ExperimentConfigs.high_gamma()),
        ("low_gamma", ExperimentConfigs.low_gamma()),
        ("high_lr", ExperimentConfigs.high_learning_rate()),
        ("low_lr", ExperimentConfigs.low_learning_rate()),
        ("fast_epsilon", ExperimentConfigs.fast_epsilon_decay()),
        ("slow_epsilon", ExperimentConfigs.slow_epsilon_decay()),
    ]
    
    results = {}
    
    for name, config in experiments:
        print(f"\n\n{'='*70}")
        print(f"Starting Experiment: {name}")
        print(f"{'='*70}")
        
        # Reduce episodes for experiments (to save time)
        config.TOTAL_EPISODES = 500  # Shorter runs for comparison
        
        stats = train(config, name)
        results[name] = stats
        
        print(f"\n✅ Completed: {name}")
        print(f"   Avg Reward: {stats['avg_reward_last_100']:.2f}")
    
    # Print comparison
    print("\n\n" + "=" * 70)
    print("📊 EXPERIMENT COMPARISON")
    print("=" * 70)
    print(f"{'Experiment':<20} {'Avg Reward':<15} {'Best Reward':<15}")
    print("-" * 70)
    
    for name, stats in results.items():
        print(f"{name:<20} {stats['avg_reward_last_100']:<15.2f} {stats['best_episode_reward']:<15.1f}")
    
    print("=" * 70)
    print("\n✅ All experiments complete!")
    print("Results saved in logs/ directory")


if __name__ == "__main__":
    run_all_experiments()