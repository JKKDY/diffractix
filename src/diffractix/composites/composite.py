from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import get_type_hints

from ..elements.base import ElementBase, annotation_contains_node
from ..elements.element import OpticalElement


class CompositeElement(ElementBase, Sequence):
    """
    Ordered collection of named optical elements.

    Node-annotated attributes define public composite parameters. Public
    OpticalElement or CompositeElement attributes present when __init__ runs
    define the ordered contents of the composite.

    Composite elements may contain other composites. Normal iteration exposes
    the direct hierarchy, while walk() and leaf_elements recursively flatten it.
    """

    # --------------
    # INITIALIZATION
    # --------------
    def __init__(self):
        names, elements = [], []
        for name, value in vars(self).items():
            if not name.startswith("_") and self._is_element(value):
                names.append(name)
                elements.append(value)
        self._element_names = tuple(names)
        self._elements = tuple(elements)

    # -------------------
    # PARAMETER DISCOVERY
    # -------------------
    @classmethod
    def _get_parameter_names(cls) -> tuple[str, ...]:
        cached = cls.__dict__.get("_parameter_names")
        if cached is not None:
            return cached
        type_hints = get_type_hints(cls)
        parameter_names = []

        for base in reversed(cls.mro()): # Method Resolution Order
            for name in getattr(base, "__annotations__", {}):
                annotation = type_hints.get(name)
                if (
                    annotation is not None
                    and annotation_contains_node(annotation)
                    and name not in parameter_names
                ):
                    parameter_names.append(name)

        cls._parameter_names = tuple(parameter_names)
        return cls._parameter_names

    # --------------
    # ELEMENT ACCESS
    # --------------
    @staticmethod
    def _is_element(value) -> bool:
        return isinstance(value, (OpticalElement, CompositeElement))

    @property
    def elements(self) -> tuple[OpticalElement | CompositeElement, ...]:
        """Direct child elements in propagation order."""
        return self._elements

    @property
    def element_names(self) -> tuple[str, ...]:
        """Names of direct child elements in propagation order."""
        return self._element_names

    @property
    def named_elements(self) -> tuple[tuple[str, OpticalElement | CompositeElement], ...]:
        """Direct child elements paired with their attribute names."""
        return tuple(zip(self.element_names, self.elements))

    # -----------------
    # NESTED COMPOSITES
    # -----------------
    def walk(self, prefix: str = "") -> Iterator[tuple[str, OpticalElement]]:
        """
        Recursively yield concrete optical elements as (path, element).

        Nested composite names are joined with dots, for example:
        "relay.first_lens".
        """
        for name, element in self.named_elements:
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(element, CompositeElement):
                yield from element.walk(path)
            else:
                yield path, element

    @property
    def leaf_elements(self) -> tuple[OpticalElement, ...]:
        """All concrete optical elements recursively in propagation order."""
        return tuple(element for _, element in self.walk())

    # ------------------
    # SEQUENCE INTERFACE
    # ------------------
    def __len__(self):
        return len(self._elements)

    def __iter__(self) -> Iterator[OpticalElement | CompositeElement]:
        return iter(self._elements)

    def __getitem__(self, index):
        return self._elements[index]

    # -------
    # DISPLAY
    # -------
    def __repr__(self):
        return f"{type(self).__name__}(len={len(self)})"
