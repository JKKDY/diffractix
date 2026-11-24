"""SX tracing and compiled numerical functions for the CasADi backend."""

from functools import partial, wraps
from math import prod
from threading import RLock
from weakref import WeakKeyDictionary

import numpy as np

from . import _Backend, _SolverCallbackCompiler

try:
    import casadi as ca
except ImportError as exc:
    raise ImportError(
        "Diffractix cannot import CasADi. Install it with "
        "'pip install \"diffractix[casadi]\"', or select another backend."
    ) from exc

from ._casadi_array import SXArray, TracingError, _Complex, _as_sx, _check_integer_range, _parts, xp


def require(condition, message):
    if isinstance(condition, SXArray):
        if condition.shape:
            raise ValueError("require() needs a scalar condition.")
        if not condition.data.is_constant():
            return
        condition = bool(float(condition.data))
    if not bool(condition):
        raise ValueError(message)


def _signature(arg):
    if isinstance(arg, (SXArray, _Complex)):
        return arg.shape, arg.dtype
    array = np.asarray(arg)
    if array.dtype.kind not in "bifuc":
        raise TracingError(f"SX inputs must be numerical, got dtype {array.dtype}.")
    return array.shape, array.dtype


def _symbol(index, shape, dtype):
    symbol = ca.SX.sym(f"arg{index}", prod(shape))
    if dtype.kind == "c":
        imag = ca.SX.sym(f"arg{index}_imag", prod(shape))
        real_dtype = np.empty((), dtype=dtype).real.dtype
        return (symbol, imag), _Complex(SXArray(symbol, shape, real_dtype), SXArray(imag, shape, real_dtype))
    return (symbol,), SXArray(symbol, shape, dtype)


def _output(value):
    shape, dtype = _signature(value)
    if dtype.kind == "c":
        real, imag = _parts(value)
        return [_as_sx(real, shape).data, _as_sx(imag, shape).data], shape, dtype
    return [_as_sx(value, shape).data], shape, dtype


def _input(value, shape, dtype):
    if dtype.kind == "c":
        return [entry for part in _parts(value) for entry in _input(part, shape, np.dtype(float))]
    if isinstance(value, SXArray):
        return [_as_sx(value, shape).data]
    if dtype.kind in "iu":
        _check_integer_range(np.asarray(value))
    if shape == ():
        return [float(value)]
    array = np.asarray(value)
    if array.shape != shape:
        array = np.broadcast_to(array, shape)
    return [array.reshape(-1, 1)]


def _result(values, shape, dtype, symbolic):
    if symbolic:
        if dtype.kind == "c":
            real_dtype = np.empty((), dtype=dtype).real.dtype
            return _Complex(SXArray(values[0], shape, real_dtype), SXArray(values[1], shape, real_dtype))
        return SXArray(values[0], shape, dtype)
    if dtype.kind == "c":
        real, imag = (value.full().reshape(shape) for value in values)
        if not shape:
            return complex(real.item(), imag.item())
        result = np.empty(shape, dtype=dtype)
        result.real, result.imag = real, imag
        return result
    if not shape:
        scalar = float(values[0])
        return scalar if dtype == np.dtype(float) else dtype.type(scalar)
    return values[0].full().reshape(shape).astype(dtype, copy=False)


class _Transforms:
    def __init__(self, theta=None):
        self.theta = theta
        self.traces = WeakKeyDictionary()
        self.lock = RLock()

    def trace(self, function, signature, static):
        try:
            traces = self.traces.setdefault(function, {})
        except TypeError:
            traces = {}
        key = signature, _static_key(static)
        if key not in traces:
            symbols, args = [], []
            static_args = dict(static)
            for index, entry in enumerate(signature):
                if entry is None:
                    args.append(static_args[index])
                elif index == 0 and self.theta is not None and entry == _signature(self.theta):
                    symbols.append(self.theta.data)
                    args.append(self.theta)
                else:
                    inputs, arg = _symbol(index, *entry)
                    symbols.extend(inputs)
                    args.append(arg)
            output = function(*args)
            traces[key] = symbols, args, _output(output)
        return traces[key]

    def compile(self, function, kind, argnum, signature, static):
        symbols, inputs, (outputs, shape, dtype) = self.trace(function, signature, static)
        if kind != "jit":
            arg_shape, arg_dtype = signature[argnum]
            if arg_dtype.kind != "f" or dtype.kind not in "fiu":
                raise TracingError("CasADi derivatives require a real floating input and real output.")
            output, = outputs
            if kind in ("grad", "hessian") and shape:
                raise TracingError(f"CasADi {kind} requires a scalar output.")
            x = inputs[argnum].data
            if kind == "grad":
                outputs, shape = [ca.gradient(output, x)], arg_shape
            elif kind == "jacobian":
                outputs, shape = [ca.vec(ca.jacobian(output, x).T)], shape + arg_shape
            else:
                outputs, shape = [ca.vec(ca.hessian(output, x)[0].T)], arg_shape + arg_shape
            dtype = arg_dtype
        callback = ca.Function("diffractix", symbols, outputs, {"allow_free": True})
        return callback, shape, dtype, callback.has_free()

    def transform(self, function, kind, argnum=0, static_argnums=()):
        if kind == "jit" and not static_argnums and getattr(function, "_casadi_compiled", False):
            return function
        if isinstance(static_argnums, int):
            static_argnums = (static_argnums,)
        if kind != "jit" and not isinstance(argnum, int):
            raise TypeError("CasADi derivatives support one integer argnum.")
        functions = {}

        @wraps(function)
        def compiled(*args):
            if static_argnums:
                static = tuple((i, args[i]) for i in range(len(args))
                               if i in static_argnums or i-len(args) in static_argnums)
                static_indices = {i for i, _ in static}
                signature = tuple(None if i in static_indices else _signature(arg)
                                  for i, arg in enumerate(args))
            else:
                static = ()
                signature = tuple(_signature(arg) for arg in args)
            key = signature, _static_key(static)
            try:
                entry = functions.get(key)
            except TypeError as error:
                raise TracingError("CasADi static arguments must be hashable.") from error
            if entry is None:
                with self.lock:
                    entry = functions.get(key)
                    if entry is None:
                        entry = self.compile(function, kind, argnum, signature, static)
                        functions[key] = entry
            callback, shape, dtype, has_free = entry
            symbolic = any(isinstance(arg, (SXArray, _Complex)) for arg in args)
            if not symbolic and has_free:
                raise TracingError("Cannot evaluate an SX function that captures a traced value.")
            values = [value for arg, spec in zip(args, signature) if spec is not None
                      for value in _input(arg, *spec)]
            return _result(callback.call(values), shape, dtype, symbolic)

        compiled._casadi_compiled = True
        return compiled


class _OpticalResultTrace:
    """Trace propagation once using symbolic theta and cache requested result fields."""

    def __init__(self, initial_values):
        _, self.theta = _symbol(0, *_signature(initial_values))
        self._result = None
        self._quantities = {}
        self._lock = RLock()

    def bindings(self, keys, run, resolve):
        with self._lock:
            if self._result is None:
                self._result = run(self.theta)
            for key in keys:
                if key not in self._quantities:
                    # The callback compiles this expression with its own goals.
                    self._quantities[key] = resolve(key, self._result)
            return {key: self._quantities[key] for key in keys}


def _optical_callback_compiler(cache, initial_values, run, resolve):
    trace = cache.get(_OpticalResultTrace)
    if trace is None:
        trace = cache.setdefault(_OpticalResultTrace, _OpticalResultTrace(initial_values))
    transforms = _Transforms(trace.theta)

    def bind_traced_result(theta, keys):
        if theta is not trace.theta:
            return None
        return trace.bindings(keys, run, resolve)

    return _SolverCallbackCompiler(
        jit=partial(transforms.transform, kind="jit"),
        grad=partial(transforms.transform, kind="grad"),
        jacobian=partial(transforms.transform, kind="jacobian"),
        hessian=partial(transforms.transform, kind="hessian"),
        bind_traced_result=bind_traced_result,
    )


def _static_key(static):
    return tuple((index, type(value), value) for index, value in static)


_transforms = _Transforms()


def grad(function, argnum=0):
    return _transforms.transform(function, "grad", argnum)


def jacobian(function, argnum=0):
    return _transforms.transform(function, "jacobian", argnum)


def hessian(function, argnum=0):
    return _transforms.transform(function, "hessian", argnum)


def jit(function, *, static_argnums=()):
    return _transforms.transform(function, "jit", static_argnums=static_argnums)


def _load_casadi() -> _Backend:
    return _Backend(
        name="casadi",
        xp=xp,
        grad=grad,
        jacobian=jacobian,
        hessian=hessian,
        jit=jit,
        require=require,
        tracing_errors=(TracingError,),
        optical_callback_compiler=_optical_callback_compiler,
    )
