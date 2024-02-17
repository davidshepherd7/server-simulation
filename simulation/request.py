from __future__ import annotations
import enum
import sys
import heapq
import argparse
import random
import string
from typing import Any, Callable

import dataclasses
from dataclasses import dataclass

from typing import NewType
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats


@dataclass
class Request:
    processing_time: float
    database_time: float | None
    when_sent: float
    _when_completed: float | None = None
    _when_timed_out: float | None = None

    id: str = dataclasses.field(
        default_factory=lambda: "".join(
            random.choice(string.ascii_uppercase + string.digits) for i in range(10)
        )
    )

    @property
    def when_completed(self) -> float | None:
        return self._when_completed

    @when_completed.setter
    def when_completed(self, val: float) -> None:
        assert not self._when_timed_out
        assert val >= self.when_sent
        self._when_completed = val

    @property
    def when_timed_out(self) -> float | None:
        return self._when_timed_out

    @when_timed_out.setter
    def when_timed_out(self, val: float) -> None:
        assert not self._when_completed
        assert val >= self.when_sent
        self._when_timed_out = val

    def __hash__(self) -> int:
        return hash(self.id)
