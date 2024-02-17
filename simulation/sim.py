from __future__ import annotations
import heapq
import random
from typing import Any, Callable

import dataclasses
from dataclasses import dataclass

from .request import Request


@dataclass(frozen=True)
class StatSnapshot:
    t: float
    queued: int
    processing: int
    completed: int
    timed_out: int


class Stats:
    def __init__(self, t_start: float) -> None:
        self.history: list[StatSnapshot] = [StatSnapshot(t_start, 0, 0, 0, 0)]
        self.all_requests: list[Request] = []

    def snapshot(self, t: float, backend: "Backend") -> None:
        self.history.append(
            StatSnapshot(
                t,
                len(backend.queue),
                len(backend.processing),
                len(backend.completed),
                len(backend.worker_timed_out),
            )
        )

    def record_request(self, t: float, r: Request) -> None:
        self.all_requests.append(r)


@dataclass(frozen=True)
class Event:
    t: float
    func: Callable[[float], list["Event"]]

    def call(self, t: float) -> list["Event"]:
        return self.func(t)

    def __lt__(self, other: "Event") -> bool:
        return self.t < other.t


class Backend:
    def __init__(self, stats: Stats, parallelism: int, worker_timeout: float) -> None:
        self.queue: list[Request] = []
        self.processing: set[Request] = set()
        self.completed: set[Request] = set()
        self.worker_timed_out: set[Request] = set()

        self.stats = stats
        self.parallelism = parallelism
        self.worker_timeout = worker_timeout

    def append_request(self, t: float, r: Request) -> list[Event]:
        self.queue.append(r)
        out = self.try_start_process_request(t)
        self.stats.snapshot(t, self)
        return out

    def try_start_process_request(self, t: float) -> list[Event]:
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
            Event(t + self.worker_timeout, lambda t: self.maybe_timeout(t, r)),
            Event(
                t + r.processing_time,
                lambda t: self.complete_process(t, r),
            ),
        ]

    def maybe_timeout(self, t: float, r: Request) -> list[Event]:
        # Already completed
        if r not in self.processing:
            return []

        self.processing.remove(r)
        self.worker_timed_out.add(r)
        r.when_timed_out = t
        self.stats.snapshot(t, self)
        return self.try_start_process_request(t)

    def complete_process(self, t: float, r: Request) -> list[Event]:
        # Already timed out or completed
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

    def fire_request(self, t: float) -> list[Event]:
        r = Request(self.processing_time(), t)
        self.stats.record_request(t, r)
        ev = self.backend.append_request(t, r)

        next_request = t + self.period()
        return [Event(next_request, self.fire_request)] + ev


# Run a single simulation.
def sim_loop(max_t: float) -> Stats:
    t = 0.0
    stats = Stats(t_start=t)
    backend = Backend(stats, parallelism=20, worker_timeout=10.0)
    client = Client(
        backend,
        stats,
        period=lambda: max(0.0, random.lognormvariate(0, 1.0)),
        processing_time=lambda: max(0.0, random.lognormvariate(1, 1.0)),
    )

    q: list[Event] = [Event(0, client.fire_request)]

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
