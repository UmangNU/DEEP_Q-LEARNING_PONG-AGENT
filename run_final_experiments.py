"""
Final experiments for assignment - 50 episodes each
Tests different gamma, alpha, and epsilon values
"""

import os
from train import train
from config import Config

print("="*70)
print("RUNNING FINAL EXPERIMENTS - 50 EPISODES EACH")
print("="*70)

# Experiment 1: High Gamma
print("\n" + "="*70)
print("EXPERIMENT 1: HIGH GAMMA (γ=0.995)")
print("Testing if higher discount factor improves performance")
print("="*70)
config1 = Config()
config1.GAMMA = 0.995
config1.TOTAL_EPISODES = 50
config1.RANDOM_SEED = 200
config1.RESUME_FROM_CHECKPOINT = False
train(config1, "exp_high_gamma")

# Experiment 2: Low Gamma
print("\n" + "="*70)
print("EXPERIMENT 2: LOW GAMMA (γ=0.95)")
print("Testing if lower discount factor affects performance")
print("="*70)
config2 = Config()
config2.GAMMA = 0.95
config2.TOTAL_EPISODES = 50
config2.RANDOM_SEED = 201
config2.RESUME_FROM_CHECKPOINT = False
train(config2, "exp_low_gamma")

# Experiment 3: High Learning Rate
print("\n" + "="*70)
print("EXPERIMENT 3: HIGH LEARNING RATE (α=0.0005)")
print("Testing if faster learning rate improves efficiency")
print("="*70)
config3 = Config()
config3.LEARNING_RATE = 0.0005
config3.TOTAL_EPISODES = 50
config3.RANDOM_SEED = 202
config3.RESUME_FROM_CHECKPOINT = False
train(config3, "exp_high_lr")

# Experiment 4: Fast Epsilon Decay
print("\n" + "="*70)
print("EXPERIMENT 4: FAST EPSILON DECAY (0.999)")
print("Testing faster exploration decay")
print("="*70)
config4 = Config()
config4.EPSILON_DECAY = 0.999
config4.TOTAL_EPISODES = 50
config4.RANDOM_SEED = 203
config4.RESUME_FROM_CHECKPOINT = False
train(config4, "exp_fast_epsilon")

print("\n" + "="*70)
print("✅ ALL EXPERIMENTS COMPLETE!")
print("="*70)
print("\nResults saved in logs/ directory:")
print("  - logs/exp_high_gamma_stats.json")
print("  - logs/exp_low_gamma_stats.json")
print("  - logs/exp_high_lr_stats.json")
print("  - logs/exp_fast_epsilon_stats.json")
print("\nUse these results to update documentation Section 4, 5, 6!")