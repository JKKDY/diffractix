from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number, Real
from typing import ClassVar

from ..core.base import OpticalBase, parameter
from ..graph import InputNode, Node, Parameter, walk_ast

@dataclass(eq=False, kw_only=True)
class OpticalElement(OpticalBase, ABC):
    """
    Base class for optical elements.

    Dataclass subclasses inherit the label constructor argument. Subclasses
    overriding __post_init__ should call super().__post_init__().
    """

    label: str | None = None

    # Static members
    _instance_counts: ClassVar[dict[type, int]] = {}

    def __post_init__(self):
        # Create a label if not explicitly set
        if self.label is None:
            cls = type(self)
            num_instances = OpticalElement._instance_counts.get(cls, 0)
            self.label = f"{cls.__name__}{num_instances + 1}"
            OpticalElement._instance_counts[cls] = num_instances + 1


    # ------------------
    # ELEMENT PROPERTIES
    # ------------------
    @property
    @abstractmethod
    def matrix(self):
        """
        Declarative 2x2 ABCD matrix.

        Entries may be graph Nodes or numerical constants. System.build()
        compiles these expressions into the numerical simulation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def element_length(self):
        """
        Declarative physical length of the element.

        May be a graph Node or a numerical constant.
        """
        raise NotImplementedError

    # ----------
    # VALIDATION
    # ----------
    def validate(self):
        """Validate this element's ABCD representation before system compilation."""
        matrix = self.matrix
        length = self.element_length
        refractive_index = self.refractive_index

        # Validate matrix shape.
        try:
            rows = tuple(matrix)
        except TypeError:
            raise TypeError("matrix must be a 2x2 structure of Nodes or numeric scalars.") from None

        if len(rows) != 2:
            raise ValueError(f"matrix must have exactly 2 rows, got {len(rows)}.")

        matrix_entries = []
        for i, row in enumerate(rows):
            try:
                entries = tuple(row)
            except TypeError:
                raise TypeError(f"matrix row {i} must contain exactly 2 entries.") from None

            if len(entries) != 2:
                raise ValueError(f"matrix row {i} must contain exactly 2 entries, got {len(entries)}.")

            for j, value in enumerate(entries):
                if not isinstance(value, Node) and (
                    isinstance(value, bool) or not isinstance(value, Number)
                ):
                    raise TypeError(
                        f"matrix[{i}][{j}] must be a Node or numeric scalar, "
                        f"got {type(value).__name__}."
                    )
                matrix_entries.append(value)

        # Validate physical length.
        if not isinstance(length, Node) and (
            isinstance(length, bool) or not isinstance(length, Real)
        ):
            raise TypeError(
                "element_length must be a Node or real numeric scalar, got {type(length).__name__}."
            )

        # Validate refractive index.
        if refractive_index is not None and not isinstance(refractive_index, Node) and (
            isinstance(refractive_index, bool) or not isinstance(refractive_index, Real)
        ):
            raise TypeError(
                "refractive_index must be a Node, real numeric scalar, "
                f"or None; got {type(refractive_index).__name__}."
            )

        # Validate every reachable symbolic dependency.
        roots = [v for v in (*matrix_entries, length) if isinstance(v, Node)]
        if isinstance(refractive_index, Node) and not (
            isinstance(refractive_index, InputNode) and refractive_index.node is None
        ):
            roots.append(refractive_index)
        tuple(walk_ast(roots))


    # -------
    # DISPLAY
    # -------
    def __str__(self):
        def current_value(value):
            if isinstance(value, Node):
                try:
                    return value.value
                except (AttributeError, RuntimeError):
                    return None
            if isinstance(value, Real):
                return value
            return None

        length = current_value(self.element_length)
        length_string = "?" if length is None else f"{length:.4g}"
        parameter_details = []

        for name, handle in zip(self.parameter_names, self.parameters):
            value = current_value(handle)
            value_string = "?" if value is None else f"{value:.4g}"
            node = handle.node
            if isinstance(node, Parameter):
                status = "VAR" if node.is_variable else "FIX"
            else:
                status = "EXPR"
            parameter_details.append(f"{name}={value_string} [{status}]")

        details = " | ".join(parameter_details)
        return (
            f"{type(self).__name__} '{self.label}' "
            f"L={length_string}"
            + (f" | {details}" if details else "")
        )
