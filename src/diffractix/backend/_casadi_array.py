"""Private NumPy-shaped values used while tracing with CasADi SX."""

from dataclasses import dataclass
from functools import wraps
from math import prod

import casadi as ca
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class TracingError(TypeError):
    """An operation needs a concrete value or is unsupported during SX tracing."""


class _Array(NDArrayOperatorsMixin):
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return prod(self.shape)

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of a traced scalar")
        return self.shape[0]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __array__(self, dtype=None, copy=None):
        raise TracingError("Cannot convert an SX value to a NumPy array. Use backend.xp.")

    def __bool__(self):
        raise TracingError(
            "Cannot branch in Python on an SX value. Use backend.xp.where. "
            "CasADi traces derivatives even with jit=False."
        )

    def __float__(self):
        raise TracingError("Cannot convert an SX value to a Python float.")

    def __int__(self):
        raise TracingError("Cannot convert an SX value to a Python integer.")

    __index__ = __int__

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != "__call__" or kwargs:
            raise TracingError(f"Unsupported SX operation: {ufunc.__name__}.{method}.")
        return _elementwise(ufunc.__name__, *args)


@dataclass(eq=False)
class SXArray(_Array):
    # Keep all data in a column in NumPy's row-major order.
    data: ca.SX
    shape: tuple[int, ...] = ()
    dtype: np.dtype = np.dtype(float)

    def __getitem__(self, key):
        indices = np.arange(self.size).reshape(self.shape)[key]
        return SXArray(ca.vec(self.data[np.ravel(indices).tolist()]), np.shape(indices), self.dtype)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        size = self.size
        shape = np.empty(size, dtype=np.uint8).reshape(shape).shape
        return SXArray(self.data, shape, self.dtype)

    def ravel(self):
        return SXArray(self.data, (self.size,), self.dtype)


@dataclass(eq=False)
class _Complex(_Array):
    real: object
    imag: object

    @property
    def dtype(self):
        return np.result_type(_dtype_operand(self.real), _dtype_operand(self.imag), np.complex64)

    @property
    def shape(self):
        return np.broadcast_shapes(np.shape(self.real), np.shape(self.imag))

    def __getitem__(self, key):
        return _Complex(_as_sx(self.real, self.shape)[key], _as_sx(self.imag, self.shape)[key])

    def ravel(self):
        return _Complex(_as_sx(self.real, self.shape).ravel(), _as_sx(self.imag, self.shape).ravel())

    def reshape(self, shape):
        return _Complex(_as_sx(self.real, self.shape).reshape(shape),
                        _as_sx(self.imag, self.shape).reshape(shape))


def _as_sx(value, shape):
    if isinstance(value, SXArray):
        if value.shape == shape:
            return value
        indices = np.broadcast_to(np.arange(value.size).reshape(value.shape), shape)
        return SXArray(ca.vec(value.data[indices.ravel().tolist()]), shape, value.dtype)
    array = np.broadcast_to(np.asarray(value), shape)
    if array.dtype.kind not in "bifu":
        raise TracingError(f"SX needs real numerical values, got dtype {array.dtype}.")
    _check_integer_range(array)
    return SXArray(ca.SX(array.astype(float).reshape(-1, 1)), shape, array.dtype)


def _check_integer_range(array):
    # SX stores doubles. Beyond this bound consecutive integers lose precision.
    if array.dtype.kind in "iu" and array.dtype.itemsize >= 8:
        if np.any(array > 2**53) or (array.dtype.kind == "i" and np.any(array < -(2**53))):
            raise TracingError("SX cannot represent integers outside the exact double-precision range.")


def _parts(value):
    if isinstance(value, _Complex):
        return value.real, value.imag
    if isinstance(value, SXArray):
        return value, 0
    return np.real(value), np.imag(value)


def _is_complex(value):
    return isinstance(value, _Complex) or (not isinstance(value, SXArray) and np.iscomplexobj(value))


def _dtype_operand(value):
    if isinstance(value, _Array):
        return value.dtype
    return np.asarray(value) if isinstance(value, (tuple, list)) else value


def _complex_op(name, *args):
    a, b = _parts(args[0])
    if name == "negative":
        return _Complex(-a, -b)
    if name == "absolute":
        return np.sqrt(a*a + b*b)
    if name == "square" or (name == "power" and not isinstance(args[1], _Array)
                            and np.shape(args[1]) == () and args[1] == 2):
        return _Complex(a*a - b*b, 2*a*b)
    if name == "conjugate":
        return _Complex(a, -b)
    if name == "sin":
        return _Complex(np.sin(a)*np.cosh(b), np.cos(a)*np.sinh(b))
    if name in ("isfinite", "isinf", "isnan"):
        op = getattr(np, name)
        return op(a) & op(b) if name == "isfinite" else op(a) | op(b)
    if len(args) == 2:
        c, d = _parts(args[1])
        if name == "add":
            return _Complex(a+c, b+d)
        if name == "subtract":
            return _Complex(a-c, b-d)
        if name == "multiply":
            if not _is_complex(args[1]):
                return _Complex(a*c, b*c)
            if not _is_complex(args[0]):
                return _Complex(a*c, a*d)
            return _Complex(a*c - b*d, a*d + b*c)
        if name == "divide" or name == "true_divide":
            if not _is_complex(args[1]):
                return _Complex(a/c, b/c)
            norm = c*c + d*d
            return _Complex((a*c + b*d)/norm, (b*c - a*d)/norm)
    raise TracingError(f"SX does not support {name} on traced complex values.")


def _divide(a, b):
    # SX simplifies 0 / x and x / x even at x == 0.
    result = ca.rdivide(a, b)
    if b.is_constant() and np.all(np.asarray(ca.DM(b)) != 0):
        return result
    reciprocal = ca.rdivide(1, b)
    invalid = ca.if_else(a < 0, -reciprocal, ca.if_else(a > 0, reciprocal, np.nan))
    return ca.if_else(b == 0, invalid, result)


def _extreme(operation, a, b):
    # CasADi's fmin/fmax select the non-NaN operand. NumPy's minimum/maximum propagate NaN.
    valid = ca.logic_and(ca.fabs(a) <= np.inf, ca.fabs(b) <= np.inf)
    return ca.if_else(valid, operation(a, b), np.nan)


_REAL_OPS = {
    "add": ca.plus, "subtract": ca.minus, "multiply": ca.times,
    "divide": _divide, "true_divide": _divide, "power": ca.power,
    "negative": ca.SX.__neg__, "positive": lambda x: x,
    "square": lambda x: x*x, "absolute": ca.fabs, "conjugate": lambda x: x,
    "sqrt": ca.sqrt, "exp": ca.exp, "log": ca.log,
    "sin": ca.sin, "cos": ca.cos, "tan": ca.tan,
    "sinh": ca.sinh, "cosh": ca.cosh, "tanh": ca.tanh,
    "arctan2": ca.atan2,
    "maximum": lambda a, b: _extreme(ca.fmax, a, b),
    "minimum": lambda a, b: _extreme(ca.fmin, a, b),
    "less": ca.SX.__lt__, "less_equal": ca.SX.__le__,
    "greater": ca.SX.__gt__, "greater_equal": ca.SX.__ge__,
    "equal": ca.SX.__eq__, "not_equal": ca.SX.__ne__,
    "logical_and": ca.logic_and, "logical_or": ca.logic_or, "logical_not": ca.logic_not,
    "bitwise_and": ca.logic_and, "bitwise_or": ca.logic_or, "invert": ca.logic_not,
    "isfinite": lambda x: ca.fabs(x) < np.inf,
    "isinf": lambda x: ca.fabs(x) == np.inf,
    "isnan": lambda x: ca.logic_not(ca.fabs(x) <= np.inf),
}
_PREDICATES = frozenset({
    "less", "less_equal", "greater", "greater_equal", "equal", "not_equal",
    "logical_and", "logical_or", "logical_not", "bitwise_and", "bitwise_or", "invert",
    "isfinite", "isinf", "isnan",
})


def _elementwise(name, *args):
    if any(_is_complex(arg) for arg in args):
        return _complex_op(name, *args)
    shape = np.broadcast_shapes(*(np.shape(arg) for arg in args))
    if name not in _REAL_OPS:
        raise TracingError(f"SX does not support {name} on traced values.")
    values = tuple(_as_sx(arg, shape) for arg in args)
    if name in ("bitwise_and", "bitwise_or", "invert") and any(value.dtype.kind != "b" for value in values):
        raise TracingError(f"SX supports {name} only on Boolean predicates.")
    dtype = np.dtype(bool) if name in _PREDICATES else np.result_type(*map(_dtype_operand, args))
    if dtype.kind in "biu" and name not in _PREDICATES:
        dtype = np.dtype(float)
    return SXArray(_REAL_OPS[name](*(value.data for value in values)), shape, dtype)


def _assemble(values, operation, axis):
    if any(_is_complex(value) for value in values):
        real, imag = zip(*(_parts(value) for value in values))
        shapes = [np.shape(value) for value in values]
        return _Complex(
            _assemble([_as_sx(v, s) for v, s in zip(real, shapes)], operation, axis),
            _assemble([_as_sx(v, s) for v, s in zip(imag, shapes)], operation, axis),
        )
    arrays = [_as_sx(value, np.shape(value)) for value in values]
    offset = 0
    indices = []
    for array in arrays:
        indices.append(np.arange(offset, offset + array.size).reshape(array.shape))
        offset += array.size
    order = operation(indices, axis=axis)
    data = ca.vertcat(*(array.data for array in arrays))[order.ravel().tolist()]
    return SXArray(ca.vec(data), order.shape, np.result_type(*(array.dtype for array in arrays)))


def _contains_sx(value):
    if isinstance(value, _Array):
        return True
    return isinstance(value, (tuple, list)) and any(_contains_sx(item) for item in value)


class NumPy:
    """Delegate concrete NumPy calls and implement the traced operations explicitly."""

    def __getattr__(self, name):
        standard = getattr(np, name)
        if isinstance(standard, type) or not callable(standard):
            return standard

        @wraps(standard)
        def call(*args, **kwargs):
            if not any(_contains_sx(arg) for arg in (*args, *kwargs.values())):
                return standard(*args, **kwargs)
            if not kwargs:
                return _elementwise(standard.__name__, *args)
            raise TracingError(f"SX does not support the supplied arguments to xp.{name}.")
        return call

    def array(self, value, *args, **kwargs):
        if not _contains_sx(value):
            return np.array(value, *args, **kwargs)
        if args or kwargs:
            raise TracingError("SX array construction does not accept dtype or copy options.")
        if isinstance(value, _Array):
            return value
        return self.stack(tuple(self.array(item) for item in value))

    asarray = array

    @staticmethod
    def real(value):
        if isinstance(value, _Complex):
            return _as_sx(value.real, value.shape)
        return _parts(value)[0]

    @staticmethod
    def imag(value):
        if isinstance(value, _Complex):
            return _as_sx(value.imag, value.shape)
        if isinstance(value, SXArray):
            return SXArray(ca.SX.zeros(value.size, 1), value.shape)
        return _parts(value)[1]

    @staticmethod
    def stack(values, axis=0):
        return _assemble(values, np.stack, axis) if _contains_sx(values) else np.stack(values, axis=axis)

    @staticmethod
    def concatenate(values, axis=0):
        if _contains_sx(values):
            return _assemble(values, np.concatenate, axis)
        return np.concatenate(values, axis=axis)

    @staticmethod
    def ravel(value):
        return value.ravel() if isinstance(value, _Array) else np.ravel(value)

    @staticmethod
    def reshape(value, shape):
        return value.reshape(shape) if isinstance(value, _Array) else np.reshape(value, shape)

    @staticmethod
    def sum(value):
        if isinstance(value, SXArray):
            dtype = np.sum(np.empty(0, dtype=value.dtype)).dtype
            return SXArray(ca.sum1(value.data), dtype=dtype)
        if isinstance(value, _Complex):
            return _Complex(xp.sum(xp.real(value)), xp.sum(xp.imag(value)))
        return np.sum(value)

    @staticmethod
    def all(value):
        if isinstance(value, SXArray):
            return SXArray(ca.sum1(value.data != 0) == value.size, dtype=np.dtype(bool))
        return np.all(value)

    @staticmethod
    def mean(value):
        return xp.sum(value) / value.size if isinstance(value, _Array) else np.mean(value)

    @staticmethod
    def std(value):
        if isinstance(value, _Array):
            return xp.sqrt(xp.mean(xp.abs(value - xp.mean(value))**2))
        return np.std(value)

    @staticmethod
    def dot(a, b):
        if not _contains_sx((a, b)):
            return np.dot(a, b)
        if isinstance(a, _Complex) or isinstance(b, _Complex):
            raise TracingError("SX dot requires real values.")
        a_shape, b_shape = np.shape(a), np.shape(b)
        if not a_shape or not b_shape:
            return a*b
        if len(a_shape) > 2 or len(b_shape) > 2 or a_shape[-1] != b_shape[0]:
            raise TracingError("SX dot supports aligned vectors and matrices.")
        shape = a_shape[:-1] + b_shape[1:]
        if not (prod(a_shape) and prod(b_shape)):
            return _as_sx(np.zeros(shape), shape)
        a_data = ca.reshape(_as_sx(a, a_shape).data, a_shape[-1], -1).T
        b_data = ca.reshape(_as_sx(b, b_shape).data, b_shape[-1] if len(b_shape) == 2 else 1, -1).T
        return SXArray(ca.vec((a_data @ b_data).T), shape)

    @staticmethod
    def where(condition, yes, no):
        if not _contains_sx((condition, yes, no)):
            return np.where(condition, yes, no)
        if _is_complex(yes) or _is_complex(no):
            a, b = _parts(yes)
            c, d = _parts(no)
            return _Complex(xp.where(condition, a, c), xp.where(condition, b, d))
        shape = np.broadcast_shapes(np.shape(condition), np.shape(yes), np.shape(no))
        dtype = np.result_type(_dtype_operand(yes), _dtype_operand(no))
        condition, yes, no = (_as_sx(value, shape) for value in (condition, yes, no))
        return SXArray(ca.if_else(condition.data, yes.data, no.data), shape, dtype)

    @staticmethod
    def full(shape, value):
        if isinstance(value, SXArray):
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            return _as_sx(value, shape)
        return np.full(shape, value)

    @staticmethod
    def linspace(start, stop, num=50):
        if _contains_sx((start, stop)):
            if np.shape(start) or np.shape(stop):
                raise TracingError("SX linspace needs scalar endpoints.")
            if num <= 1:
                return xp.full(num, start)
            weights = np.linspace(0., 1., num)
            return (1. - weights)*start + weights*stop
        return np.linspace(start, stop, num)


xp = NumPy()
