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
from config import Config

# Experiment 1: Baseline (already done, just document it)
print("\n" + "="*70)
print("Experiment 1: BASELINE - Already completed (2000 episodes)")
print("Using existing results from logs/baseline_stats.json")
print("="*70)

# Experiment 2: High Gamma
print("\n" + "="*70)
print("Experiment 2: HIGH GAMMA (γ=0.995)")
print("="*70)
config_high_gamma = Config()
config_high_gamma.GAMMA = 0.995
config_high_gamma.TOTAL_EPISODES = 300
config_high_gamma.RANDOM_SEED = 100
config_high_gamma.RESUME_FROM_CHECKPOINT = False
train(config_high_gamma, "high_gamma")

# Experiment 3: Low Gamma
print("\n" + "="*70)
print("Experiment 3: LOW GAMMA (γ=0.95)")
print("="*70)
config_low_gamma = Config()
config_low_gamma.GAMMA = 0.95
config_low_gamma.TOTAL_EPISODES = 300
config_low_gamma.RANDOM_SEED = 101
config_low_gamma.RESUME_FROM_CHECKPOINT = False
train(config_low_gamma, "low_gamma")

# Experiment 4: High Learning Rate
print("\n" + "="*70)
print("Experiment 4: HIGH LEARNING RATE (α=0.0005)")
print("="*70)
config_high_lr = Config()
config_high_lr.LEARNING_RATE = 0.0005
config_high_lr.TOTAL_EPISODES = 300
config_high_lr.RANDOM_SEED = 102
config_high_lr.RESUME_FROM_CHECKPOINT = False
train(config_high_lr, "high_lr")

# Experiment 5: Low Learning Rate
print("\n" + "="*70)
print("Experiment 5: LOW LEARNING RATE (α=0.00005)")
print("="*70)
config_low_lr = Config()
config_low_lr.LEARNING_RATE = 0.00005
config_low_lr.TOTAL_EPISODES = 300
config_low_lr.RANDOM_SEED = 103
config_low_lr.RESUME_FROM_CHECKPOINT = False
train(config_low_lr, "low_lr")

# Experiment 6: Fast Epsilon Decay
print("\n" + "="*70)
print("Experiment 6: FAST EPSILON DECAY (0.999)")
print("="*70)
config_fast_epsilon = Config()
config_fast_epsilon.EPSILON_DECAY = 0.999
config_fast_epsilon.TOTAL_EPISODES = 300
config_fast_epsilon.RANDOM_SEED = 104
config_fast_epsilon.RESUME_FROM_CHECKPOINT = False
train(config_fast_epsilon, "fast_epsilon")

print("\n" + "="*70)
print("✅ ALL EXPERIMENTS COMPLETE!")
print("="*70)
print("\nResults saved in logs/ directory")
print("Check these files:")
print("  - logs/high_gamma_stats.json")
print("  - logs/low_gamma_stats.json")
print("  - logs/high_lr_stats.json")
print("  - logs/low_lr_stats.json")
print("  - logs/fast_epsilon_stats.json")