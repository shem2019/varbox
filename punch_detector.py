# punch_detector.py

import numpy as np
from config import PUNCH_DISTANCE_THRESHOLD

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def detect_punch(fighter, opponent_head):
    """
    fighter: dict with 'left_wrist', 'right_wrist'
    opponent_head: [x, y]
    """
    left_hit = calculate_distance(fighter['left_wrist'], opponent_head) < PUNCH_DISTANCE_THRESHOLD
    right_hit = calculate_distance(fighter['right_wrist'], opponent_head) < PUNCH_DISTANCE_THRESHOLD
    return left_hit or right_hit
