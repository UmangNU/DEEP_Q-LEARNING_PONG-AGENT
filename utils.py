"""
Utility Functions for Preprocessing

ATTRIBUTION & LICENSE:
- Preprocessing approach: Standard Atari preprocessing (Mnih et al., 2015)
- Implementation: Original work by Umang Mistry

Author: Umang Mistry
Date: November 2024
License: MIT
"""

import cv2
import numpy as np


def preprocess_frame(frame):
    """
    Preprocess Atari frame:
    1. Convert RGB to grayscale
    2. Resize to 84x84
    
    Args:
        frame: Raw frame from environment (210, 160, 3)
    
    Returns:
        Preprocessed frame (84, 84)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
    return resized


def stack_frames(frames, state, is_new):
    """
    Stack 4 consecutive frames to give agent motion information
    
    Args:
        frames: List of previous frames
        state: New frame from environment
        is_new: Whether this is start of new episode
    
    Returns:
        Tuple of (stacked_array, frames_list)
    """
    # Preprocess new frame
    frame = preprocess_frame(state)
    
    if is_new:
        # Start of episode - stack same frame 4 times
        frames = [frame] * 4
    else:
        # Add new frame, remove oldest
        frames = frames[1:] + [frame]
    
    return np.array(frames), frames


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)