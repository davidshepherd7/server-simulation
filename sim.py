#! /usr/bin/env python3

import enum
import sys
import heapq
import argparse
import random
import string
from typing import Any, Callable, List, Set, Optional

import dataclasses
from dataclasses import dataclass

from typing import NewType
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import scipy  # type: ignore[import]
import scipy.stats  # type: ignore[import]


@dataclass(frozen=True)
class Event:
    t: float
    func: Callable[[float], List["Event"]]

    def call(self, t: float) -> List["Event"]:
        return self.func(t)

    def __lt__(self, other: "Event") -> bool:
        return self.t < other.t


@dataclass
class Request:
    processing_time: float
    when_sent: float
    when_completed: Optional[float] = None
    when_timed_out: Optional[float] = None

    id: str = dataclasses.field(
        default_factory=lambda: "".join(
            random.choice(string.ascii_uppercase + string.digits) for i in range(10)
        )
    )

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class StatSnapshot:
    t: float
    queued: int
    processing: int
    completed: int
    timed_out: int


class Stats:
    def __init__(self, t_start: float) -> None:
        self.history: List[StatSnapshot] = [StatSnapshot(t_start, 0, 0, 0, 0)]
        self.all_requests: List[Request] = []

    def snapshot(self, t: float, backend: "Backend") -> None:
        self.history.append(
            StatSnapshot(
                t,
                len(backend.queue),
                len(backend.processing),
                len(backend.completed),
                len(backend.timed_out),
            )
        )

    def record_request(self, t: float, r: Request) -> None:
        self.all_requests.append(r)


class Backend:
    def __init__(self, stats: Stats, parallelism: int, timeout: float) -> None:
        self.queue: List[Request] = []
        self.processing: Set[Request] = set()
        self.completed: Set[Request] = set()
        self.timed_out: Set[Request] = set()

        self.stats = stats
        self.parallelism = parallelism
        self.timeout = timeout

    def append_request(self, t: float, r: Request) -> List[Event]:
        self.queue.append(r)
        out = self.try_start_process_request(t)
        self.stats.snapshot(t, self)
        return out

    def try_start_process_request(self, t: float) -> List[Event]:
        assert len(self.processing) <= self.parallelism

        # All cores are full
        if len(self.processing) >= self.parallelism:
            return []

        # No work to do
        if len(self.queue) == 0:
            return []

        r = self.queue.pop()
        self.processing.add(r)
        self.stats.snapshot(t, self)
        return [
            Event(t + self.timeout, lambda t: self.maybe_timeout(t, r)),
            Event(
                t + r.processing_time,
                lambda t: self.complete_process(t, r),
            ),
        ]

    def maybe_timeout(self, t: float, r: Request) -> List[Event]:
        # Already completed
        if r not in self.processing:
            return []

        self.processing.remove(r)
        self.timed_out.add(r)
        r.when_timed_out = t
        self.stats.snapshot(t, self)
        return self.try_start_process_request(t)

    def complete_process(self, t: float, r: Request) -> List[Event]:
        # Already completed
        if r not in self.processing:
            return []

        self.processing.remove(r)
        self.completed.add(r)
        r.when_completed = t
        self.stats.snapshot(t, self)
        return self.try_start_process_request(t)


class Client:
    def __init__(
        self,
        backend: Backend,
        stats: Stats,
        period: Callable[[], float],
        processing_time: Callable[[], float],
    ) -> None:
        self.backend = backend
        self.stats = stats

        self.period = period
        self.processing_time = processing_time

    def fire_request(self, t: float) -> List[Event]:
        r = Request(self.processing_time(), t)
        self.stats.record_request(t, r)
        ev = self.backend.append_request(t, r)

        next_request = t + self.period()
        return [Event(next_request, self.fire_request)] + ev


# Run a single simulation.
def sim_loop(max_t: float) -> Stats:
    t = 0.0
    stats = Stats(t_start=t)
    backend = Backend(stats, parallelism=20, timeout=10.0)
    client = Client(
        backend,
        stats,
        period=lambda: max(0.0, random.lognormvariate(0, 1.0)),
        processing_time=lambda: max(0.0, random.lognormvariate(1, 1.0)),
    )

    q: List[Event] = [Event(0, client.fire_request)]

    # This is the core simulation loop. Until we've reached the maximum
    #  simulation time, pull the next event off from a heap of events, fire
    #  whichever callback is associated with that event, and add any events it
    #  generates back to the heap.
    heapq.heapify(q)
    while len(q) > 0 and t < max_t:
        e = heapq.heappop(q)
        t = e.t
        new_events = e.call(t)
        for new_event in new_events:
            heapq.heappush(q, new_event)

    return stats


#### Analysis


# chatgpt generated
def events_per_second(timestamps: List[float]):
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


def main(argv):
    stats = sim_loop(1000)

    # f1 = plot_processing_times(stats)
    f2 = plot_queue_sizes(stats)
    f3 = plot_rates(stats)

    plt.show()

    return 0


# If this script is run from a shell then run main() and return the result.
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
