from __future__ import annotations

import sys
import matplotlib.pyplot as plt
from typing import List

from . import analysis
from . import sim


def max_backend_queue(s: sim.Stats) -> int:
    return max([ss.queued for ss in s.backend_history])


def experiment() -> None:
    # stats: list[tuple[float, float, sim.Stats]] = []
    for db_fraction in [0, 0.5]:
        for period in [0.0, 0.1, 0.2, 0.3, 0.4]:
            s = sim.sim_loop(max_t=1000, db_fraction=db_fraction, period=period)
            # stats.append((db_fraction, period, s))
            f2 = analysis.plot_queue_sizes(s)


def main(argv: list[str]) -> int:
    # stats = sim.sim_loop(1000)

    # f1 = analysis.plot_processing_times(stats)
    # f2 = analysis.plot_queue_sizes(stats)
    # f3 = analysis.plot_rates(stats)

    experiment()

    plt.show()

    return 0


# If this script is run from a shell then run main() and return the result.
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
