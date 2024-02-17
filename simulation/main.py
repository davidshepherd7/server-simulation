from __future__ import annotations

import sys
import matplotlib.pyplot as plt
from typing import List

from . import analysis
from . import sim


def main(argv: list[str]) -> int:
    stats = sim.sim_loop(1000)

    # f1 = analysis.plot_processing_times(stats)
    f2 = analysis.plot_queue_sizes(stats)
    f3 = analysis.plot_rates(stats)

    plt.show()

    return 0


# If this script is run from a shell then run main() and return the result.
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
