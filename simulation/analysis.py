import enum
import sys
import heapq
import argparse
import random
import string
from typing import Any, Callable

import dataclasses
from dataclasses import dataclass

from typing import NewType, Tuple
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats

from .request import Request
from .sim import Stats


# chatgpt generated
def events_per_second(timestamps: list[float]) -> Tuple[Any, Any]:
    # Calculate the time span
    start_time = min(timestamps)
    end_time = max(timestamps)
    bins = np.arange(start_time, end_time + 1, 20.0)

    # Calculate the events per second using binned_statistic
    counts, _, _ = scipy.stats.binned_statistic(
        timestamps, timestamps, statistic="count", bins=bins
    )

    return bins[:-1], counts.astype(int)


def plot_processing_times(stats: Stats) -> Any:
    fig, ax = plt.subplots()
    processing_times = [r.processing_time for r in stats.all_requests]
    ax.hist(processing_times, 30)
    fig.suptitle("Distribution of generated processing times")
    return fig


def plot_rates(stats: Stats) -> Any:
    fig, ax = plt.subplots()
    ax.plot(
        *events_per_second(
            [
                r.when_completed
                for r in stats.all_requests
                if r.when_completed is not None
            ]
        ),
        label="Completions",
    )
    ax.plot(
        *events_per_second(
            [
                r.when_timed_out
                for r in stats.all_requests
                if r.when_timed_out is not None
            ]
        ),
        label="Timeouts",
    )
    fig.suptitle("Event rates")
    fig.legend()
    return fig


def plot_queue_sizes(stats: Stats) -> Any:
    fig, ax = plt.subplots()
    ax.plot(
        [s.t for s in stats.history],
        [s.queued for s in stats.history],
        label="Queue size",
    )
    ax.plot(
        [s.t for s in stats.history],
        [s.processing for s in stats.history],
        label="Processing count",
    )
    fig.suptitle("Queue sizes")
    fig.legend()
    return fig
