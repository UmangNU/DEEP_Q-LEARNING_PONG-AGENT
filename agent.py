"""
DQN Agent with Epsilon-Greedy and Boltzmann Exploration

ATTRIBUTION & LICENSE:
- DQN algorithm: Mnih et al. (2015)
- Experience replay concept: Mnih et al. (2015)
- Target network technique: Mnih et al. (2015)
- Boltzmann exploration: Standard RL textbook method
- All implementation: Original work by Umang Mistry

Author: Umang Mistry
Date: November 2024
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os

from network import DQN
from replay_buffer import ReplayBuffer
from config import Config


class DQNAgent:
    """
    DQN Agent with ε-greedy exploration
    """
    
    def __init__(self, n_actions, config=None):
        """
        Initialize DQN Agent
        
        Args:
            n_actions: Number of possible actions
            config: Configuration object (default: Config())
        """
        if config is None:
            config = Config()
        
        self.n_actions = n_actions
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        print(f"🖥️  Using device: {self.device}")
        
        # Policy network - the one we train
        self.policy_net = DQN(n_actions).to(self.device)
        
        # Target network - for stable Q-value targets
        self.target_net = DQN(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer - Adam with learning rate (alpha)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config.LEARNING_RATE
        )
        
        # Experience replay buffer
        self.memory = ReplayBuffer(config.MEMORY_SIZE)
        
        # Exploration parameters
        self.epsilon = config.EPSILON_START
        self.steps = 0
        self.episode = 0
        
    def select_action(self, state):
        """
        Select action using ε-greedy policy
        
        With probability ε: explore (random action)
        With probability 1-ε: exploit (best Q-value action)
        
        Args:
            state: Current state
        
        Returns:
            Selected action
        """
        # Explore: random action
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        # Exploit: action with highest Q-value
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()
    
    def select_action_boltzmann(self, state, temperature=1.0):
        """
        Alternative exploration policy: Boltzmann (Softmax) exploration
        Required by assignment to try policy other than ε-greedy
        
        Args:
            state: Current state
            temperature: Controls randomness (higher = more random)
        
        Returns:
            Selected action
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).cpu().numpy()[0]
            
            # Apply softmax with temperature
            exp_q = np.exp(q_values / temperature)
            probs = exp_q / np.sum(exp_q)
            
            return np.random.choice(self.n_actions, p=probs)
    
    def store(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """
        Train on batch using Bellman equation
        
        Bellman Equation:
        Q(s,a) = R + γ * max_a' Q(s', a')
        
        Where:
        - Q(s,a): Q-value of state-action pair
        - R: Immediate reward
        - γ (gamma): Discount factor
        - s': Next state
        - a': Action in next state
        
        Returns:
            Loss value or None if not enough samples
        """
        # Don't train until we have enough experiences
        if len(self.memory) < self.config.MIN_MEMORY:
            return None
        
        # Sample random batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.BATCH_SIZE
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values: Q(s, a)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values: R + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            # If episode ended (done=1), next Q-value is 0
            target_q = rewards + (1 - dones) * self.config.GAMMA * next_q
        
        # Compute loss (Mean Squared Error)
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon (reduce exploration over time)
        self.epsilon = max(
            self.config.EPSILON_MIN, 
            self.epsilon * self.config.EPSILON_DECAY
        )
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.config.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"🎯 Target network updated at step {self.steps}")
        
        return loss.item()
    
    def save_checkpoint(self, episode, episode_rewards, filepath):
        """
        Save training checkpoint
        
        Args:
            episode: Current episode number
            episode_rewards: List of all episode rewards
            filepath: Where to save checkpoint
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'episode': episode,
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episode_rewards': episode_rewards
        }, filepath)
    
    def load_checkpoint(self, filepath):
        """
        Load training checkpoint
        
        Args:
            filepath: Path to checkpoint file
        
        Returns:
            Tuple of (start_episode, episode_rewards)
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        
        return checkpoint['episode'], checkpoint['episode_rewards']
    
    def save(self, filepath):
        """Save final model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
    
    def load(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']