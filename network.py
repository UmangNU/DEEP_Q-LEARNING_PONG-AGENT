"""
Deep Q-Network Architecture

ATTRIBUTION & LICENSE:
- Architecture specification: Mnih et al. (2015) Nature paper
  (3 conv layers: 32→64→64 filters, 2 FC layers: 512→6)
- PyTorch implementation: Original work by Umang Mistry

Author: Umang Mistry
Date: November 2024
License: MIT
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network for Atari games
    
    Input: 4 stacked grayscale frames (4, 84, 84)
    Output: Q-values for each action
    """
    
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        
        # Convolutional layers - extract visual features
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully connected layers - compute Q-values
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def forward(self, x):
        """
        Forward pass through network
        
        Args:
            x: Input tensor (batch, 4, 84, 84)
        
        Returns:
            Q-values for each action (batch, n_actions)
        """
        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0
        
        # Convolutional layers with ReLU activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        return self.fc2(x)