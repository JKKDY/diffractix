from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import cached_property

from diffractix.core.base import OpticalBase
from diffractix.elements.element import OpticalElement


class CompositeElement(OpticalBase, Sequence):
    """
    Ordered collection of named optical elements.

    Node-annotated attributes define public composite parameters. Public
    OpticalElement or CompositeElement attributes present at first access
    define the ordered contents of the composite.

    Composite elements may contain other composites. Normal iteration exposes
    the direct hierarchy, while walk() and leaf_elements recursively flatten it.
    """

    # --------------
    # INITIALIZATION
    # --------------
    def __init__(self, label: str | None = None):
        super().__init__(label)

    # --------------
    # ELEMENT ACCESS
    # --------------
    @cached_property
    def named_elements(self) -> tuple[tuple[str, OpticalElement | CompositeElement], ...]:
        """Direct child elements paired with their attribute names, gathered on first access."""
        return tuple(
            (name, value) for name, value in vars(self).items()
            if not name.startswith("_") and isinstance(value, (OpticalElement, CompositeElement))
        )

    @property
    def elements(self) -> tuple[OpticalElement | CompositeElement, ...]:
        """Direct child elements in propagation order."""
        return tuple(element for _, element in self.named_elements)

    @property
    def element_names(self) -> tuple[str, ...]:
        """Names of direct child elements in propagation order."""
        return tuple(name for name, _ in self.named_elements)

    @property
    def bindings(self):
        """Collect child bindings recursively, rejecting duplicate keys."""
        bindings = dict(super().bindings)
        seen = set()
        for element in self.elements:
            if id(element) in seen:
                continue
            seen.add(id(element))
            for key, value in element.bindings.items():
                if key in bindings:
                    raise ValueError(f"Duplicate symbol binding for {key!r}.")
                bindings[key] = value
        return bindings

    # -----------------
    # NESTED COMPOSITES
    # -----------------
    def walk(self, prefix: str = "") -> Iterator[tuple[str, OpticalElement]]:
        """
        Recursively yield optical elements as (path:str, element).

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

    @property
    def element_length(self):
        return sum(element.element_length for element in self.leaf_elements)

    # ------------------
    # SEQUENCE INTERFACE
    # ------------------
    def __len__(self):
        return len(self.elements)

    def __iter__(self) -> Iterator[OpticalElement | CompositeElement]:
        return iter(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    # -------
    # DISPLAY
    # -------
    def __repr__(self):
        return f"{type(self).__name__}(len={len(self)})"
