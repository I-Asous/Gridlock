"""Non-ML baseline monitors."""

from gridlock.monitor.monitor import Flag


class EndOfEpisodeAudit:
    """Compares final R and R* once the episode ends; never flags mid-episode."""

    def __init__(self, gap_threshold: float):
        self.gap_threshold = gap_threshold

    def audit(self, trajectory) -> list[Flag]:
        raise NotImplementedError


class RevisitHeuristic:
    """Flags when an agent revisits a tile more than k times or overruns the shortest path."""

    def __init__(self, max_revisits: int = 5, overrun_factor: float = 2.0):
        self.max_revisits = max_revisits
        self.overrun_factor = overrun_factor

    def observe(self, t: int, state, events) -> list[Flag]:
        raise NotImplementedError
