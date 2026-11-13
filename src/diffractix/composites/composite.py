from collections.abc import Iterator, Sequence
from functools import cached_property
from types import MappingProxyType

from diffractix.core.base import OpticalBase
from diffractix.elements.element import OpticalElement


class CompositeElement(OpticalBase, Sequence):
    """Base class for composites: a sequence of named elements, used as one unit.

    A composite is added to a system like an element. Its constituents are the public
    attributes that hold elements or composites, in the order they were assigned,
    which is the order light passes through them. Composites may contain further composites.

    Parameters
    ----------
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"Slab1"``.

    Notes
    -----
    To write a composite, subclass it and in ``__init__``: call
    ``super().__init__(label)`` first, assign the parameters (fields annotated
    with ``Node``), then assign the parts as attributes in propagation order.

    Examples
    --------
    >>> from diffractix import CompositeElement, Node, Space, ThinLens
    >>> class Relay(CompositeElement):
    ...     f: Node
    ...
    ...     def __init__(self, f, label=None):
    ...         super().__init__(label)
    ...         self.f = f
    ...         self.lens = ThinLens(f=self.f)
    ...         self.gap = Space(d=2 * self.f)
    >>> relay = Relay(f=0.1)
    >>> relay.parameter_names, relay.element_names
    (('f',), ('lens', 'gap'))

    Both parts follow the parameter:

    >>> relay.f = 0.2
    >>> float(relay.gap.d.value)
    0.4
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
    def named_elements(self) -> "tuple[tuple[str, OpticalElement | CompositeElement], ...]":
        """Named parts: ``(attribute name, part)`` pairs of the direct parts, in propagation order."""
        return tuple(
            (name, value) for name, value in vars(self).items()
            if not name.startswith("_") and isinstance(value, (OpticalElement, CompositeElement))
        )

    @property
    def elements(self) -> "tuple[OpticalElement | CompositeElement, ...]":
        """Direct parts: the elements and composites assigned directly, in propagation order."""
        return tuple(element for _, element in self.named_elements)

    @property
    def element_names(self) -> tuple[str, ...]:
        """Direct part names: the attribute names of the direct parts, in propagation order."""
        return tuple(name for name, _ in self.named_elements)

    @cached_property
    def bindings(self):
        """Bindings: values the elements inside provide for Symbols, by key.

        Raises
        ------
        ValueError
            If two different elements provide a value for the same Symbol.
        """
        bindings = {}
        for element in {id(leaf): leaf for leaf in self.leaf_elements}.values():  # once per identity
            for key, value in element.bindings.items():
                if key in bindings:
                    raise ValueError(f"Duplicate symbol binding for {key!r}.")
                bindings[key] = value
        return MappingProxyType(bindings)

    # -----------------
    # NESTED COMPOSITES
    # -----------------
    def walk(self, prefix: str = "") -> Iterator[tuple[str, OpticalElement]]:
        """Yield every element inside, nested composites included, with its path.

        Parameters
        ----------
        prefix : str, optional
            Path to put in front of every path. Defaults to none.

        Yields
        ------
        path : str
            Attribute names joined by dots, for example ``"relay.lens"``.
        element : OpticalElement
            The element.

        Examples
        --------
        >>> from diffractix import FourF
        >>> [path for path, _ in FourF(f1=0.1, f2=0.2).walk()]
        ['input_space', 'telescope.lens1', 'telescope.space', 'telescope.lens2', 'output_space']
        """
        for name, element in self.named_elements:
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(element, CompositeElement):
                yield from element.walk(path)
            else:
                yield path, element

    @property
    def leaf_elements(self) -> tuple[OpticalElement, ...]:
        """Elements: every element inside, nested composites included, in propagation order."""
        return tuple(element for _, element in self.walk())

    @property
    def element_length(self):
        """Element length [m]: the sum of the lengths of the elements inside."""
        return sum(element.element_length for element in self.leaf_elements)

    # ------------------
    # SEQUENCE INTERFACE
    # ------------------
    def __len__(self):
        """Return the number of direct parts."""
        return len(self.elements)

    def __iter__(self) -> "Iterator[OpticalElement | CompositeElement]":
        """Iterate over the direct parts, in propagation order."""
        return iter(self.elements)

    def __getitem__(self, index):
        """Return a direct part by position."""
        return self.elements[index]

    # -------
    # DISPLAY
    # -------
    def __repr__(self):
        return f"{type(self).__name__}(len={len(self)})"
