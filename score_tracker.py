# score_tracker.py

from config import COOLDOWN_FRAMES

class ScoreTracker:
    def __init__(self):
        self.scores = {}          # {fighter_id: int}
        self.last_punch = {}      # {fighter_id: frame_number}
        self.punch_log = []       # [(Fighter ID, Timestamp, Score)]

    def update(self, frame_num, fighter_id, timestamp):
        last = self.last_punch.get(fighter_id, -COOLDOWN_FRAMES)
        if frame_num - last > COOLDOWN_FRAMES:
            self.scores[fighter_id] = self.scores.get(fighter_id, 0) + 1
            self.last_punch[fighter_id] = frame_num
            self.punch_log.append((f"Fighter {fighter_id}", timestamp, self.scores[fighter_id]))

    def get_score(self, fighter_id):
        return self.scores.get(fighter_id, 0)
