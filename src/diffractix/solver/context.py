from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from diffractix.graph import Symbol


class SolverSymbolKind(Enum):
    AT = auto()
    AFTER = auto()
    Z_AT = auto()
    Z_AFTER = auto()
    Z = auto()
    STATE = auto()


@dataclass(frozen=True)
class SolverSymbolKey:
    kind: SolverSymbolKind
    element_id: int | None = None
    occurrence: int | None = None
    index: int | None = None
    name: str | None = None


class SolverContext:
    """Numerical context available while evaluating solver expressions."""

    def __init__(self, theta, run):
        self.theta = theta
        self._run = run
        self._result = None

    def _get_result(self):
        if self._result is None:
            self._result = self._run(self.theta)
        return self._result

    @property
    def z(self):
        return self._get_result().z

    @property
    def states(self):
        return self._get_result().states

    def at(self, location, occurrence=None):
        return self._get_result().at(location, occurrence)

    def after(self, element, occurrence=None):
        return self._get_result().after(element, occurrence)

    def z_at(self, element, occurrence=None):
        return self._get_result().z_at(element, occurrence)

    def z_after(self, element, occurrence=None):
        return self._get_result().z_after(element, occurrence)


class SolverCompileContext:
    class SymbolGenerator:
        def __init__(
            self,
            target_elements,
            kind: SolverSymbolKind,
            target=None,
            occurrence=None,
            index=None,
        ):
            self._target_elements = target_elements
            self._kind = kind
            self._target = target
            self._occurrence = occurrence
            self._index = index

        def __getattr__(self, name):
            if self._target is not None:
                self._target_elements[id(self._target)] = self._target
                key = SolverSymbolKey(
                    kind=self._kind,
                    element_id=id(self._target),
                    occurrence=self._occurrence,
                    name=name,
                )
            else:
                key = SolverSymbolKey(
                    kind=self._kind,
                    index=self._index,
                    name=name,
                )

            return Symbol(key)

    class ZGenerator:
        def __getitem__(self, index):
            return Symbol(
                SolverSymbolKey(
                    kind=SolverSymbolKind.Z,
                    index=index,
                )
            )

    class StatesGenerator:
        def __init__(self, context):
            self._context = context

        def __getitem__(self, index):
            return self._context.SymbolGenerator(
                self._context._target_elements,
                SolverSymbolKind.STATE,
                index=index,
            )

    def __init__(self):
        self._target_elements = {}

    @property
    def z(self):
        return self.ZGenerator()

    @property
    def states(self):
        return self.StatesGenerator(self)

    def at(self, element, occurrence=None):
        return self.SymbolGenerator(
            self._target_elements,
            SolverSymbolKind.AT,
            element,
            occurrence,
        )

    def after(self, element, occurrence=None):
        return self.SymbolGenerator(
            self._target_elements,
            SolverSymbolKind.AFTER,
            element,
            occurrence,
        )

    def z_at(self, element, occurrence=None):
        self._target_elements[id(element)] = element
        return Symbol(
            SolverSymbolKey(
                kind=SolverSymbolKind.Z_AT,
                element_id=id(element),
                occurrence=occurrence,
            )
        )

    def z_after(self, element, occurrence=None):
        self._target_elements[id(element)] = element
        return Symbol(
            SolverSymbolKey(
                kind=SolverSymbolKind.Z_AFTER,
                element_id=id(element),
                occurrence=occurrence,
            )
        )
