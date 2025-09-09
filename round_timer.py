# round_timer.py
class RoundTimer:
    """
    Frame-based round/interval timing.
    """
    def __init__(self, fps: int, round_secs: int = 180, rest_secs: int = 60, total_rounds: int = 12):
        self.fps = max(fps, 1)
        self.round_frames = round_secs * self.fps
        self.rest_frames = rest_secs * self.fps
        self.total_rounds = total_rounds

        self.frame = 0
        self.round_no = 1
        self.in_round = True  # True=active round, False=rest
        self._just_ended_round = False

    def step(self):
        """
        Advance a frame. Returns (round_no, in_round, just_ended_round, bout_over)
        """
        self._just_ended_round = False
        self.frame += 1

        if self.in_round:
            if self.frame >= self.round_frames:
                self.in_round = False
                self._just_ended_round = True
                self.frame = 0
        else:
            # rest
            if self.frame >= self.rest_frames:
                self.frame = 0
                self.in_round = True
                self.round_no += 1

        bout_over = (self.round_no > self.total_rounds)
        return self.round_no if not bout_over else self.total_rounds, self.in_round, self._just_ended_round, bout_over

    def time_in_phase(self):
        """Frames elapsed in current phase."""
        return self.frame
