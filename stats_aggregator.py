# stats_aggregator.py
class StatsAggregator:
    """
    Keeps per-round punch stats for RED/BLUE.
    Extend later with 'power' tags or accuracy if your punch detector exposes them.
    """
    def __init__(self, total_rounds: int = 12):
        self.total_rounds = total_rounds
        self.round_stats = {r: {"RED": {"landed": 0}, "BLUE": {"landed": 0}}
                            for r in range(1, total_rounds + 1)}
        # Optional: KD counts and fouls per round (can be set externally)
        self.kd = {r: {"RED": 0, "BLUE": 0} for r in range(1, total_rounds + 1)}
        self.deductions = {r: {"RED": 0, "BLUE": 0} for r in range(1, total_rounds + 1)}

    def add_punch(self, role: str, round_no: int):
        if role in ("RED", "BLUE") and 1 <= round_no <= self.total_rounds:
            self.round_stats[round_no][role]["landed"] += 1

    def add_knockdown(self, role_down: str, round_no: int, count: int = 1):
        # role_down is the fighter who was knocked down
        if role_down in ("RED", "BLUE") and 1 <= round_no <= self.total_rounds:
            self.kd[round_no][role_down] += max(1, int(count))

    def add_deduction(self, role: str, round_no: int, points: int = 1):
        if role in ("RED", "BLUE") and 1 <= round_no <= self.total_rounds:
            self.deductions[round_no][role] += max(1, int(points))

    def get_round(self, round_no: int):
        return self.round_stats.get(round_no, {"RED": {"landed": 0}, "BLUE": {"landed": 0}})
