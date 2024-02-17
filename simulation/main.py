from __future__ import annotations

import sys
import matplotlib.pyplot as plt
from typing import Any

from . import analysis
from . import sim


def max_backend_queue(s: sim.Stats) -> int:
    return max([ss.queued for ss in s.backend_history])


def experiment() -> None:
    i = 0
    stats: list[tuple[float, float, sim.Stats]] = []

    db_fractions = [0, 0.5]
    rpses = [1, 5, 10]

    results: Any = {}
    for db_fraction in db_fractions:
        results[db_fraction] = {}
        for rps in rpses:
            s = sim.sim_loop(
                max_t=1000,
                db_fraction=db_fraction,
                requests_per_second=rps,
            )
            results[db_fraction][rps] = s

    fig, ax = plt.subplots(len(db_fractions), len(rpses))
    for i, db_fraction in enumerate(db_fractions):
        for j, rps in enumerate(rpses):
            analysis.plot_queue_sizes(results[db_fraction][rps], ax[i][j])
            ax[i][j].set_title(f"db_fraction = {db_fraction}, rps = {rps}")

    fig.legend()


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
