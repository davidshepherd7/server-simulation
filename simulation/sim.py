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


@dataclass(frozen=True)
class Event:
    t: float
    func: Callable[[float], list["Event"]]

    def call(self, t: float) -> list["Event"]:
        return self.func(t)

    def __lt__(self, other: "Event") -> bool:
        return self.t < other.t


class Database:
    backend: Backend

    def __init__(self, connections: int, connection_timeout: float | None) -> None:
        self.queue: list[Request] = []
        self.processing: set[Request] = set()
        self.completed: set[Request] = set()
        self.worker_timed_out: set[Request] = set()

        self.connections = connections
        self.connection_timeout = connection_timeout

        self.stats: list[StatSnapshot] = []

    def _snapshot(self, t: float) -> None:
        self.stats.append(
            StatSnapshot(
                t,
                len(self.queue),
                len(self.processing),
                len(self.completed),
                len(self.worker_timed_out),
            )
        )

    def append_request(self, t: float, r: Request) -> list[Event]:
        self.queue.append(r)
        out = self.try_start_process_request(t)
        self._snapshot(t)
        return out

    def try_start_process_request(self, t: float) -> list[Event]:
        assert len(self.processing) <= self.connections

        # All cores are full
        if len(self.processing) >= self.connections:
            return []

        # No work to do
        if len(self.queue) == 0:
            return []

        r = self.queue.pop()
        self.processing.add(r)
        self._snapshot(t)

        # if self.connection_timeout is not None:
        #     timeout_events = [
        #         Event(t + self.connection_timeout, lambda t: self.maybe_timeout(t, r))
        #     ]
        # else:
        #     timeout_events = []

        assert r.database_time is not None

        return [
            # *timeout_events,
            Event(
                t + r.database_time,
                lambda t: self.complete_process(t, r),
            ),
        ]

    # def maybe_timeout(self, t: float, r: Request) -> list[Event]:
    #     # Already completed
    #     if r not in self.processing:
    #         return []

    #     self.processing.remove(r)
    #     self.worker_timed_out.add(r)
    #     # r.when_timed_out = t
    #     self._snapshot(t)
    #     return [
    #         *self.try_start_process_request(t),
    #         Event(t, lambda t: self.backend.database_timeout(t, r)),
    #     ]

    def complete_process(self, t: float, r: Request) -> list[Event]:
        # Already timed out or completed
        if r not in self.processing:
            return []

        self.processing.remove(r)
        self.completed.add(r)
        # r.when_completed = t
        self._snapshot(t)
        return [
            *self.backend.return_from_db(t, r),
            *self.try_start_process_request(t),
        ]


class Backend:
    def __init__(
        self, database: Database, workers: int, worker_timeout: float | None
    ) -> None:
        self.database = database

        self.queue: list[Request] = []
        self.processing: set[Request] = set()
        self.completed: set[Request] = set()
        self.worker_timed_out: set[Request] = set()

        self.workers = workers
        self.worker_timeout = worker_timeout

        self.stats: list[StatSnapshot] = []

    def _snapshot(self, t: float) -> None:
        self.stats.append(
            StatSnapshot(
                t,
                len(self.queue),
                len(self.processing),
                len(self.completed),
                len(self.worker_timed_out),
            )
        )

    def append_request(self, t: float, r: Request) -> list[Event]:
        self.queue.append(r)
        out = self.try_start_process_request(t)
        self._snapshot(t)
        return out

    def try_start_process_request(self, t: float) -> list[Event]:
        assert len(self.processing) <= self.workers

        # All cores are full
        if len(self.processing) >= self.workers:
            return []

        # No work to do
        if len(self.queue) == 0:
            return []

        r = self.queue.pop()
        self.processing.add(r)
        self._snapshot(t)

        if self.worker_timeout is not None:
            timeout_events = [
                Event(t + self.worker_timeout, lambda t: self.maybe_timeout(t, r))
            ]
        else:
            timeout_events = []

        if r.database_time is None:
            processing_events = [
                Event(
                    t + r.processing_time,
                    lambda t: self.complete_process(t, r),
                ),
            ]
        else:
            processing_events = self.database.append_request(t, r)

        return [*timeout_events, *processing_events]

    def maybe_timeout(self, t: float, r: Request) -> list[Event]:
        # Already completed
        if r not in self.processing:
            return []

        self.processing.remove(r)
        self.worker_timed_out.add(r)
        r.when_timed_out = t
        self._snapshot(t)
        return self.try_start_process_request(t)

    def return_from_db(self, t: float, r: Request) -> list[Event]:
        return self.complete_process(t, r)

    def complete_process(self, t: float, r: Request) -> list[Event]:
        # Already timed out or completed
        if r not in self.processing:
            return []

        self.processing.remove(r)
        self.completed.add(r)
        r.when_completed = t
        self._snapshot(t)
        return self.try_start_process_request(t)


class Client:
    def __init__(
        self,
        backend: Backend,
        period: Callable[[], float],
        processing_time: Callable[[], float],
        database_time: Callable[[], float | None],
    ) -> None:
        self.backend = backend

        self.period = period
        self.processing_time = processing_time
        self.database_time = database_time

        self.all_requests: list[Request] = []

    def fire_request(self, t: float) -> list[Event]:
        r = Request(self.processing_time(), self.database_time(), t)
        self.all_requests.append(r)
        ev = self.backend.append_request(t, r)

        next_request = t + self.period()
        return [Event(next_request, self.fire_request)] + ev


@dataclass(frozen=True)
class Stats:
    t_start: float
    all_requests: list[Request]
    backend_history: list[StatSnapshot]
    db_history: list[StatSnapshot]


# Run a single simulation.
def sim_loop(
    *,
    max_t: float,
    db_fraction: float = 0.5,
    requests_per_second: float = 5,
) -> Stats:
    t = 0.0
    cpu_fraction = 1 - db_fraction
    assert 0.0 <= cpu_fraction <= 1.0

    database = Database(connections=4, connection_timeout=None)
    backend = Backend(database, workers=20, worker_timeout=10.0)
    database.backend = backend  # Yuck
    client = Client(
        backend,
        period=lambda: 1 / requests_per_second,
        processing_time=lambda: max(0.0, random.lognormvariate(1, 1.0)) * cpu_fraction,
        database_time=lambda: random.choices(
            [None, max(0.0, random.lognormvariate(1, 1.0)) * db_fraction],
            weights=[cpu_fraction, db_fraction],
            k=1,
        )[0],
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

    return Stats(
        t_start=t,
        all_requests=client.all_requests,
        backend_history=backend.stats,
        db_history=database.stats,
    )
