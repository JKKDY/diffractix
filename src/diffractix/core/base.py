from __future__ import annotations

from dataclasses import field, fields, is_dataclass
from numbers import Real
from typing import ClassVar, get_args, get_origin, get_type_hints

from diffractix.graph import InputNode, Node, Parameter


_GRAPH_PARAMETER = object()


def parameter(default=None):
    """Declare a dataclass field as a parameter.

    Fields annotated with ``Node`` are parameters automatically. Use this for a
    parameter whose annotation is something else.

    Parameters
    ----------
    default : float or Node, optional
        Default value. Defaults to ``None``, meaning no value.

    Returns
    -------
    dataclasses.Field
        The field, marked as a parameter.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from diffractix import OpticalElement, parameter
    >>> @dataclass(eq=False, kw_only=True)
    ... class Lens(OpticalElement):
    ...     f: float = parameter(0.1)
    ...
    ...     @property
    ...     def matrix(self):
    ...         return ((1, 0), (-1 / self.f, 1))
    ...
    ...     @property
    ...     def element_length(self):
    ...         return 0.0
    >>> Lens().f.value
    0.1
    """
    return field(default=default, metadata={_GRAPH_PARAMETER: True})


class OpticalBase:
    """Base class for elements, composites and light states, providing their parameters, labels and constraints.

    A field annotated with ``Node`` (or declared with :func:`parameter`) is a
    parameter. Parameters can be made variable and optimized in inverse
    design. By default they are fixed and act as constants Assigning a number creates a `Parameter` named after the field;
    assigning a graph expression uses it as is. The attribute is a handle to that
    value, so ``lens.f.variable()`` and ``lens.f.value`` work, and assigning
    ``lens.f = 0.2`` later replaces what the handle points to.

    Parameters
    ----------
    label : str, optional
        Name used in results and messages. Defaults to an automatic name such as
        ``"Slab1"``.
    """

    _parameter_names: ClassVar[tuple[str, ...] | None] = None
    _instance_counts: ClassVar[dict[type, int]] = {}
    
    def __new__(cls, *args, **kwargs):
        # use __new__ instead of __init__ so subclasses to have to call super().__init__ explicitely 
        instance = super().__new__(cls)
        object.__setattr__(instance, "_constraints", [])
        object.__setattr__(instance, "label", None)
        return instance

    def __init__(self, label: str | None = None):
        self._init_label(label)

    def _init_label(self, label: str | None = None):
        """Set the label, auto-numbering per class ('Slab1', 'Slab2', ...) when none is given.

        A label already assigned to the instance is kept. Numerical copies
        (internal states) are never numbered.
        """
        if label is None:
            label = getattr(self, "label", None)
        if label is None and not getattr(type(self), "_is_numerical", False):
            count = OpticalBase._instance_counts.get(type(self), 0) + 1
            OpticalBase._instance_counts[type(self)] = count
            label = f"{type(self).__name__}{count}"
        self.label = label

    # -------------------
    # PARAMETER DISCOVERY
    # -------------------
    @property
    def bindings(self):
        """Bindings: values for Symbols that this object provides to the system. Empty by default."""
        return {}

    @staticmethod
    def _annotation_contains_node(annotation) -> bool:
        """Return whether an annotation contains Node or a Node subclass."""
        if isinstance(annotation, type):
            try:
                return issubclass(annotation, Node)
            except TypeError:
                return False
        return any(OpticalBase._annotation_contains_node(arg) for arg in get_args(annotation))


    @property
    def refractive_index(self):
        """Refractive index: the medium the light is in after this object, or ``None`` if the medium does not change.

        For an element with a length, such as a `Space`, this is also the medium inside it.
        """
        return None

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Parameter names: the parameter fields of this class, in declaration order."""
        cls = type(self)
        cached = cls.__dict__.get("_parameter_names")

        if cached is not None:
            return cached

        # find all fields explicitely marked with parameter() (e.g.  f: float = parameter(0.2))
        explicitly_marked_parameters = set()
        if is_dataclass(cls):
            for item in fields(cls):
                if item.metadata.get(_GRAPH_PARAMETER, False):
                    explicitly_marked_parameters.add(item.name)

        # get parameter names in base classes 
        # traverse in method resolution order (e.g. object -> OpticalBase -> OpticalElement -> ThinLens)
        # we dont use fields() because base classes needn't be dataclasses
        names = []
        type_hints = get_type_hints(cls)

        for base in reversed(cls.mro()):
            for name in getattr(base, "__annotations__", {}):
                annotation = type_hints.get(name)
                if name in names or get_origin(annotation) is ClassVar:
                    continue

                if name in explicitly_marked_parameters or cls._annotation_contains_node(annotation):
                    names.append(name)

        cls._parameter_names = tuple(names)
        return cls._parameter_names

    @property
    def parameters(self) -> tuple[InputNode, ...]:
        """Parameters: the handle of each parameter, in the order of `parameter_names`."""
        return tuple(getattr(self, name) for name in self.parameter_names)

    @property
    def values(self) -> tuple[float | None, ...]:
        """Values: the current value of each parameter, or ``None`` where it has none (for example an unbound Symbol)."""
        values = []
        for handle in self.parameters:
            try:
                values.append(handle.value)
            except (AttributeError, RuntimeError):
                values.append(None)
        return tuple(values)


    # ------------
    # CONSTRAINTS
    # ------------
    @property
    def constraints(self) -> tuple:
        """Constraints: those attached with :meth:`require`."""
        return tuple(self._constraints)

    def require(self, *constraints, label: str | None = None):
        """Attach constraints that must hold in any optimization of a system containing this object.

        Parameters
        ----------
        *constraints : Comparison, Constraint or callable
            Comparisons such as ``lens.f > 0``, `Constraint` objects, or callables
            without arguments that return a comparison.
        label : str, optional
            Name for a single comparison, used in reports.

        Returns
        -------
        OpticalBase
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If `label` is given with several constraints or with a `Constraint`.

        Notes
        -----
        Callables must be differentiable by the numerical backend; diffractix does
        not check this.
        """
        from diffractix.solver.constraint import label_constraints
        self._constraints.extend(label_constraints(constraints, label))
        return self


    # --------------------
    # PARAMETER ASSIGNMENT
    # --------------------
    def __setattr__(self, name, value):
        def convert_value_to_node(name: str) -> Node | None:
            if isinstance(value, Node):
                return value
            if isinstance(value, Real) and not isinstance(value, bool):
                return Parameter(value=value, name=name, owner=self)
            if value is None:
                return None
            raise TypeError(
                f"Parameter {self._get_parameter_label(name)!r} must be a Node, numeric scalar, or None; "
                f"got {type(value).__name__}."
            )

        if name in self.parameter_names:
            handle = self.__dict__.get(name)
            if isinstance(handle, InputNode):
                handle.node = convert_value_to_node(name)
                return
            value = InputNode(convert_value_to_node(name,))
        object.__setattr__(self, name, value)

    def _get_parameter_label(self, name: str) -> str:
        label = getattr(self, "label", None)
        if label is not None:
            return f"{label}.{name}"
        return f"{type(self).__name__}.{name}"

    def _get_parameter_node(self, name: str) -> Parameter | None:
        handle = getattr(self, name)
        if not isinstance(handle, InputNode):
            raise TypeError( f"Parameter {self._get_parameter_label(name)!r} is not an InputNode.")
        if isinstance(handle.node, Parameter):
            return handle.node
        return None


    # -----------
    # VARIABILITY
    # -----------
    def _validate_parameter_names(self, names: tuple[str, ...]) -> tuple[str, ...]:
        if not names:
            return self.parameter_names

        unknown = [name for name in names if name not in self.parameter_names]
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) {unknown} for {type(self).__name__}. "
                f"Available: {self.parameter_names}"
            )

        return names

    def variable(self, *parameter_names: str, **bounds):
        """Mark parameters as optimization variables, optionally with new bounds.

        Parameters are fixed until marked variable. Call with no arguments to mark
        every parameter of this object as variable.

        Parameters
        ----------
        *parameter_names : str
            Names of parameters to mark variable. Their bounds stay as they are (default: (-inf, +inf)).
        **bounds : tuple of float
            Parameters to mark variable with new bounds, as ``name=(lower, upper)``.
            A ``None`` endpoint keeps the current one.

        Returns
        -------
        OpticalBase
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If a name is unknown, given both plainly and with bounds, or a bounds pair
            is invalid or excludes the current value.
        TypeError
            If a named parameter holds an expression rather than a `Parameter`.

        Notes
        -----
        The call is atomic: if anything is invalid, no parameter changes.

        Examples
        --------
        >>> from diffractix import ThickLens
        >>> from diffractix.units import mm
        >>> lens = ThickLens(d=6 * mm, n=1.5, R1=100 * mm, R2=-100 * mm)
        >>> lens = lens.variable(R1=(50 * mm, 200 * mm), d=(3 * mm, 10 * mm))
        >>> lens.R1.node.is_variable, lens.R2.node.is_variable
        (True, False)
        """
        duplicated = set(parameter_names) & set(bounds)
        if duplicated:
            raise ValueError(f"Parameter(s) {sorted(duplicated)} given both by name and with bounds.")
        requested = (*parameter_names, *bounds)
        strict = bool(requested)
        names = self._validate_parameter_names(requested)

        parameters = []
        for name in names:
            parameter = self._get_parameter_node(name)
            if parameter is not None:
                parameters.append((parameter, bounds.get(name)))
            elif strict:
                raise TypeError(
                    f"Parameter {self._get_parameter_label(name)!r} does not directly contain "
                    "a Parameter and cannot itself be marked variable."
                )

        previous = [(parameter, parameter.bounds, parameter.is_variable) for parameter, _ in parameters]
        try:
            for parameter, pair in parameters:
                parameter.variable(bounds=pair)
        except Exception:
            for parameter, old_bounds, was_variable in previous:
                parameter.lower_bound, parameter.upper_bound = old_bounds
                if not was_variable:
                    parameter.fixed()
            raise

        return self

    def fixed(self, *parameter_names: str):
        """Mark parameters as fixed, so the optimizer does not change them.

        Parameters are fixed by default; use this to undo :meth:`variable`. Call with
        no arguments to fix every parameter of this object that holds a `Parameter`.

        Parameters
        ----------
        *parameter_names : str
            Names of parameters to fix.

        Returns
        -------
        OpticalBase
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If a name is unknown.
        TypeError
            If a named parameter holds an expression rather than a `Parameter`.
        """
        strict = bool(parameter_names)
        names = self._validate_parameter_names(parameter_names)

        for name in names:
            parameter = self._get_parameter_node(name)
            if parameter is not None:
                parameter.fixed()
            elif strict:
                raise TypeError(
                    f"Parameter {self._get_parameter_label(name)!r} does not directly contain "
                    "a Parameter and cannot itself be marked fixed."
                )

        return self
