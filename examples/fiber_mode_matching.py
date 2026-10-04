# debug_fiber_mode_matching.py

import importlib

import numpy as np

import diffractix as dfx


# ============================================================
# Configuration
# ============================================================

WAVELENGTH = 1550e-9
PACKAGE_LENGTH = 25e-3
MIN_GAP = 4e-3

FD_STEP = 1e-7
HESSIAN_FD_STEP = 1e-6


# ============================================================
# Finite-difference diagnostics
# ============================================================

def finite_difference_gradient(func, x, h=FD_STEP):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)

    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()

        xp[i] += h
        xm[i] -= h

        result[i] = (
            func(xp) - func(xm)
        ) / (2.0 * h)

    return result


def finite_difference_jacobian(func, x, h=FD_STEP):
    x = np.asarray(x, dtype=float)
    y = np.atleast_1d(func(x))

    result = np.zeros((len(y), len(x)))

    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()

        xp[i] += h
        xm[i] -= h

        result[:, i] = (
            np.atleast_1d(func(xp))
            - np.atleast_1d(func(xm))
        ) / (2.0 * h)

    return result


def finite_difference_hessian(problem, x, h=HESSIAN_FD_STEP):
    x = np.asarray(x, dtype=float)
    n = len(x)

    result = np.zeros((n, n))

    for i in range(n):
        xp = x.copy()
        xm = x.copy()

        xp[i] += h
        xm[i] -= h

        result[:, i] = (
            np.asarray(problem.gradient(xp))
            - np.asarray(problem.gradient(xm))
        ) / (2.0 * h)

    return result


def inspect_problem(problem, x, label):
    x = np.asarray(x, dtype=float)

    objective = float(problem.objective(x))

    gradient = np.asarray(
        problem.gradient(x),
        dtype=float,
    )

    gradient_fd = finite_difference_gradient(
        problem.objective,
        x,
    )

    hessian = np.asarray(
        problem.objective_hessian(x),
        dtype=float,
    )

    hessian_fd = finite_difference_hessian(
        problem,
        x,
    )

    constraints = np.asarray(
        problem.constraints(x),
        dtype=float,
    )

    jacobian = np.asarray(
        problem.jacobian(x),
        dtype=float,
    )

    jacobian_fd = finite_difference_jacobian(
        problem.constraints,
        x,
    )

    print()
    print("=" * 78)
    print(label)
    print("=" * 78)

    print()
    print("x")
    print(x)

    print()
    print("variable bounds")
    for i, (value, lower, upper) in enumerate(
        zip(
            x,
            problem.x_lower,
            problem.x_upper,
        )
    ):
        print(
            f"  x[{i}] = {value:.12g}    "
            f"[{lower:.12g}, {upper:.12g}]"
        )

    print()
    print(f"objective = {objective:.12g}")

    print()
    print("AUTOGRAD GRADIENT")
    print(gradient)

    print()
    print("FINITE-DIFFERENCE GRADIENT")
    print(gradient_fd)

    print()
    print("GRADIENT DIFFERENCE")
    print(gradient - gradient_fd)

    print()
    print("GRADIENT RELATIVE ERROR")
    denominator = np.maximum(
        1.0,
        np.maximum(
            np.abs(gradient),
            np.abs(gradient_fd),
        ),
    )
    print((gradient - gradient_fd) / denominator)

    print()
    print("AUTOGRAD OBJECTIVE HESSIAN")
    print(hessian)

    print()
    print("FINITE-DIFFERENCE OBJECTIVE HESSIAN")
    print(hessian_fd)

    print()
    print("HESSIAN DIFFERENCE")
    print(hessian - hessian_fd)

    print()
    print("CONSTRAINT VALUES")
    for i, value in enumerate(constraints):
        print(
            f"  c[{i}] = {value:.12g}    "
            f"[{problem.constraint_lower[i]:.12g}, "
            f"{problem.constraint_upper[i]:.12g}]"
        )

    print()
    print("AUTOGRAD CONSTRAINT JACOBIAN")
    print(jacobian)

    print()
    print("FINITE-DIFFERENCE CONSTRAINT JACOBIAN")
    print(jacobian_fd)

    print()
    print("JACOBIAN DIFFERENCE")
    print(jacobian - jacobian_fd)

    print()
    print("BOX-BOUND FIRST-ORDER SANITY")

    for i, (value, lower, upper, g) in enumerate(
        zip(
            x,
            problem.x_lower,
            problem.x_upper,
            gradient,
        )
    ):
        if np.isclose(value, lower, rtol=0.0, atol=1e-10):
            position = "LOWER BOUND"
        elif np.isclose(value, upper, rtol=0.0, atol=1e-10):
            position = "UPPER BOUND"
        else:
            position = "INTERIOR"

        print(
            f"  x[{i}] {position:11s}   "
            f"gradient = {g:+.12g}"
        )


# ============================================================
# Monkey-patch the backend entry point
#
# Solver.solve() constructs the complete Problem and then calls
# solve_scipy(problem, ...). We intercept exactly there.
# ============================================================

solver_module = importlib.import_module(
    "diffractix.solver.solver"
)

_original_solve_scipy = solver_module.solve_scipy


def debug_solve_scipy(
    problem,
    method=None,
    options=None,
):
    method_name = method or "trust-constr"

    inspect_problem(
        problem,
        problem.x0,
        f"{method_name}: INITIAL POINT",
    )

    result = _original_solve_scipy(
        problem,
        method=method,
        options=options,
    )

    inspect_problem(
        problem,
        result.x,
        f"{method_name}: RETURNED POINT",
    )

    print()
    print("=" * 78)
    print(f"{method_name}: RAW SCIPY INFORMATION")
    print("=" * 78)

    raw = result.raw

    print("success:          ", result.success)
    print("cost:             ", result.cost)
    print("message:          ", result.message)
    print("iterations:       ", result.iterations)
    print("x:                ", result.x)

    for name in (
        "optimality",
        "constr_violation",
        "execution_time",
        "tr_radius",
        "barrier_parameter",
        "barrier_tolerance",
        "cg_niter",
        "cg_stop_cond",
        "grad",
        "lagrangian_grad",
        "v",
    ):
        value = getattr(raw, name, None)

        if value is not None:
            print(f"{name + ':':18s}", value)

    return result


solver_module.solve_scipy = debug_solve_scipy


# ============================================================
# Optical problem
# ============================================================

source = dfx.GaussianBeam.from_fiber_tip(
    wavelength=WAVELENGTH,
    MFD=10.4e-6,
)

target_mode = dfx.GaussianBeam.from_fiber_tip(
    wavelength=WAVELENGTH,
    MFD=9.2e-6,
)


input_gap = dfx.Space(
    d=5e-3,
    label="Input gap",
)

input_gap.d.variable(
    lower_bound=4e-3,
    upper_bound=10e-3,
)


lens1 = dfx.ThinLens(
    f=8e-3,
    label="Lens 1",
)


middle_gap = dfx.Space(
    d=8e-3,
    label="Lens spacing",
)

middle_gap.d.variable(
    lower_bound=5e-3,
    upper_bound=14e-3,
)


lens2 = dfx.ThinLens(
    f=7e-3,
    label="Lens 2",
)


output_gap = dfx.Space(
    d=(
        PACKAGE_LENGTH
        - input_gap.d
        - middle_gap.d
    ),
    label="Output gap",
)


target = dfx.Plane(
    label="Target fiber",
)


system = dfx.System()

system.add_input_beam(source)
system.add(input_gap)
system.add(lens1)
system.add(middle_gap)
system.add(lens2)
system.add(output_gap)
system.add(target)

system.require(
    output_gap.d >= MIN_GAP
)


# ============================================================
# Compile and inspect initial design
# ============================================================

simulation = system.build()
initial = simulation.run()

initial_beam = initial.at(target)
initial_overlap = initial_beam.overlap_with(
    target_mode
)

print()
print("=" * 78)
print("COMPILED SIMULATION")
print("=" * 78)
print()
print(simulation)

print()
print(
    f"Initial fiber overlap: "
    f"{100.0 * initial_overlap:.6f} %"
)


# ============================================================
# Inverse-design problem
# ============================================================

solver = dfx.Solver(simulation)

solver.target(
    dfx.Objective(
        lambda ctx: (
            ctx.at(target).w / target_mode.w
            - 1.0
        ),
        label="Target mode radius",
    ),

    dfx.Objective(
        lambda ctx: (
            ctx.at(target).z0
            / target_mode.z_r
        ),
        label="Target waist position",
    ),
)


# ============================================================
# Solve with trust-constr
# ============================================================

print()
print()
print("#" * 78)
print("TRUST-CONSTR")
print("#" * 78)

trust_solution = solver.solve(
    method="trust-constr",
    options={
        "factorization_method": "SVDFactorization",
    },
)


# ============================================================
# Report trust-constr optical result
# ============================================================

trust_result = trust_solution.run()
trust_beam = trust_result.at(target)

trust_overlap = trust_beam.overlap_with(
    target_mode
)

print()
print("=" * 78)
print("TRUST-CONSTR OPTICAL RESULT")
print("=" * 78)

print(
    f"Input gap:     "
    f"{1e3 * trust_solution[input_gap.d]:.6f} mm"
)

print(
    f"Lens spacing:  "
    f"{1e3 * trust_solution[middle_gap.d]:.6f} mm"
)

print(
    f"Output gap:    "
    f"{1e3 * trust_solution[output_gap.d]:.6f} mm"
)

print(
    f"Target radius: "
    f"{1e6 * target_mode.w:.6f} um"
)

print(
    f"Matched radius:"
    f" {1e6 * trust_beam.w:.6f} um"
)

print(
    f"Waist offset:  "
    f"{1e6 * trust_beam.z0:.6f} um"
)

print(
    f"Fiber overlap: "
    f"{100.0 * trust_overlap:.8f} %"
)

print()
print("Targets")

for target_result in trust_solution.targets:
    print(
        f"  {target_result.objective.label}: "
        f"value={target_result.value}, "
        f"cost={target_result.cost}"
    )


# ============================================================
# Solve the EXACT SAME problem with SLSQP
#
# This is an important diagnostic:
#
# trust-constr:
#     gradient + Jacobian + Hessians
#
# SLSQP:
#     gradient + Jacobian
#
# If SLSQP works while trust-constr fails, the Hessian path
# becomes the leading suspect.
# ============================================================

print()
print()
print("#" * 78)
print("SLSQP")
print("#" * 78)

slsqp_solution = solver.solve(
    method="SLSQP",
)


# ============================================================
# Report SLSQP optical result
# ============================================================

slsqp_result = slsqp_solution.run()
slsqp_beam = slsqp_result.at(target)

slsqp_overlap = slsqp_beam.overlap_with(
    target_mode
)

print()
print("=" * 78)
print("SLSQP OPTICAL RESULT")
print("=" * 78)

print(
    f"Input gap:     "
    f"{1e3 * slsqp_solution[input_gap.d]:.6f} mm"
)

print(
    f"Lens spacing:  "
    f"{1e3 * slsqp_solution[middle_gap.d]:.6f} mm"
)

print(
    f"Output gap:    "
    f"{1e3 * slsqp_solution[output_gap.d]:.6f} mm"
)

print(
    f"Target radius: "
    f"{1e6 * target_mode.w:.6f} um"
)

print(
    f"Matched radius:"
    f" {1e6 * slsqp_beam.w:.6f} um"
)

print(
    f"Waist offset:  "
    f"{1e6 * slsqp_beam.z0:.6f} um"
)

print(
    f"Fiber overlap: "
    f"{100.0 * slsqp_overlap:.8f} %"
)

print()
print("Targets")

for target_result in slsqp_solution.targets:
    print(
        f"  {target_result.objective.label}: "
        f"value={target_result.value}, "
        f"cost={target_result.cost}"
    )