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
    when_sent: float
    when_completed: float | None = None
    when_timed_out: float | None = None

    id: str = dataclasses.field(
        default_factory=lambda: "".join(
            random.choice(string.ascii_uppercase + string.digits) for i in range(10)
        )
    )

    def __hash__(self) -> int:
        return hash(self.id)
