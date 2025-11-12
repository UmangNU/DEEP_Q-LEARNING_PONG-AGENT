"""
Experience Replay Buffer

ATTRIBUTION & LICENSE:
- Experience replay concept: Mnih et al. (2015)
- Implementation: Original work by Umang Mistry

Author: Umang Mistry
Date: November 2024
License: MIT
"""

import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples
    """
    
    def __init__(self, capacity):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample random batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )
    
    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)