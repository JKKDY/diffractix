from __future__ import annotations

from typing import Any


class Solution:
    """Result of solving a Diffractix inverse-design problem."""

    @property
    def x(self):
        raise NotImplementedError

    @property
    def parameter_info(self):
        raise NotImplementedError

    @property
    def parameters(self):
        raise NotImplementedError

    @property
    def success(self):
        raise NotImplementedError

    @property
    def cost(self):
        raise NotImplementedError

    @property
    def message(self):
        raise NotImplementedError

    @property
    def targets(self):
        raise NotImplementedError

    @property
    def constraints(self):
        raise NotImplementedError

    @property
    def violations(self):
        raise NotImplementedError

    @property
    def feasible(self):
        raise NotImplementedError

    def value(self, node):
        raise NotImplementedError

    def __getitem__(self, node):
        return self.value(node)

    def run(self):
        raise NotImplementedError

    @property
    def system(self):
        raise NotImplementedError