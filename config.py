"""
Configuration for DQN Pong Agent

ATTRIBUTION & LICENSE:
- Based on DQN algorithm: Mnih et al. (2015) "Human-level control through deep RL" Nature
- Hyperparameters adapted from Atari DQN standards
- All implementation code: Original work by Umang Mistry

Author: Umang Mistry
Course: AI/ML & Prompt Engineering, Northeastern University
Professor: Nick Brown
Date: November 2024
License: MIT (see LICENSE file)
"""

class Config:
    # Environment
    ENV_NAME = 'ALE/Pong-v5'
    
    # Training
    TOTAL_EPISODES = 2000
    RANDOM_SEED = 43  # CHANGED from 42 for new run
    
    # Network Architecture
    LEARNING_RATE = 0.0001  # Alpha in Bellman equation
    GAMMA = 0.99  # Discount factor in Bellman equation
    
    # Exploration (ε-greedy policy) - IMPROVED
    EPSILON_START = 1.0
    EPSILON_MIN = 0.02           # CHANGED from 0.1
    EPSILON_DECAY = 0.999995     # CHANGED from 0.9995 (much slower!)
    
    # Memory
    MEMORY_SIZE = 10000
    MIN_MEMORY = 1000  # Start training after this many samples
    BATCH_SIZE = 32
    
    # Target Network
    TARGET_UPDATE = 1000  # Update target network every N steps
    
    # Checkpointing
    CHECKPOINT_EVERY = 50  # Save checkpoint every N episodes
    RESUME_FROM_CHECKPOINT = True  # CHANGED to False for fresh start
    
    # Logging
    LOG_EVERY = 10
    SAVE_EVERY = 100
    
    # Device
    DEVICE = 'cpu'  # Use 'mps' for M3 GPU if stable


class ExperimentConfigs:
    """
    Configurations for hyperparameter experiments
    Required by assignment to test different alpha/gamma values
    """
    
    @staticmethod
    def baseline():
        """Baseline configuration"""
        return Config()
    
    @staticmethod
    def high_gamma():
        """Higher discount factor - values future rewards more"""
        config = Config()
        config.GAMMA = 0.995
        return config
    
    @staticmethod
    def low_gamma():
        """Lower discount factor - values immediate rewards more"""
        config = Config()
        config.GAMMA = 0.95
        return config
    
    @staticmethod
    def high_learning_rate():
        """Higher learning rate - faster but less stable learning"""
        config = Config()
        config.LEARNING_RATE = 0.0005
        return config
    
    @staticmethod
    def low_learning_rate():
        """Lower learning rate - slower but more stable learning"""
        config = Config()
        config.LEARNING_RATE = 0.00005
        return config
    
    @staticmethod
    def fast_epsilon_decay():
        """Faster exploration decay"""
        config = Config()
        config.EPSILON_DECAY = 0.999
        return config
    
    @staticmethod
    def slow_epsilon_decay():
        """Slower exploration decay"""
        config = Config()
        config.EPSILON_DECAY = 0.9999
        return config