"""Post Processing - Classical Shadow - All Observable Calculation - Container/Purity Value Kind
(:mod:`qurrium_pqp.process.classical_shadow.all_observable.container_kind`)

"""

from typing import Literal, TypedDict

from qurry.process.utils import FloatType
from qurry.process.classical_shadow import (
    RhoMethod,
    RhoMethodType,
    TraceMethod,
    PurityValueKind,
    verify_purity_value_kind,
    default_method_on_value_kind,
)

from ..trace_process import handle_trace_method_extend, TraceMethodExtendType, BitWiseTraceMethod


PurityValueKindExtend = PurityValueKind | Literal["bitwise"]
"""The kind of purity value calculation.
This will depend on the rho_method and trace_method.

- "multi_shots":
    The *rho_method is one of the multi_shots methods* **and** *trace_method is one of the
    matrix operation methods*.

    .. code-block:: python

        (
            rho_method in [
                "multi_shots", 
                "multi_shots_vectorized",
            ]
        ) and (
            trace_method in [
                "trace_of_matmul", 
                "einsum_ij_ji", 
                "quick_trace_of_matmul",
                "einsum_aij_bji_to_ab_numpy", 
                "einsum_aij_bji_to_ab_jax",
            ]
        )

- "single_shots":
    The *rho_method is one of the single_shots methods* **and** *trace_method is one of the
    matrix operation methods*, or the *trace_method is one of the non-matrix operation methods
    except "bitwise_py"*.

    .. code-block:: python

        (
            trace_method in [
                "nomatmul_trace_py", 
                "nomatmul_trace_rust",
            ]
        ) or (
            (
                rho_method in [
                    "single_shots", 
                    "single_shots_vectorized",
                ]
            ) and (
                trace_method in [
                    "trace_of_matmul", 
                    "einsum_ij_ji", 
                    "quick_trace_of_matmul",
                    "einsum_aij_bji_to_ab_numpy", 
                    "einsum_aij_bji_to_ab_jax",
                ]
            )
        )

- "bitwise":
    The *trace_method is "bitwise_py"* no matter what the rho_method is.

    .. code-block:: python

        (trace_method in ["bitwise_py"])
"""


def verify_purity_value_kind_extend(
    rho_method: RhoMethodType, trace_method: TraceMethodExtendType
) -> PurityValueKindExtend:
    """Verify the kind of purity value calculation.

    Args:
        rho_method (RhoMethodType, optional):
            It can be either "multi_shots", "multi_shots_vectorized",
            "single_shots", or "single_shots_vectorized".

            For the "multi_shots_*" methods, the counts and random basis are used as is.
            For the "single_shots_*" methods, the counts and random basis are
            converted to single shot per snapshot for classical shadow post-processing.

            **Warning: Althought larger snapshots number means more accurate values.**
            **But if your shots number is large,**
            **this may significantly increase memory usage**
            **and require a lot of computing resource.**
            **In worst scenrio, this will break your computer.**
            **Please reconsider for performance.**

            - "multi_shots": Use Numpy to calculate the rho_m with precomputed values.
            - "multi_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow.

            - "single_shots": Use Numpy to calculate the rho_m
                with precomputed values with converted single shot counts.
            - "single_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow with converted single shot counts.

            Currently, "multi_shots" is the best option for performance.
            Default to DEFAULT_RHO_METHOD, which is "multi_shots".
        trace_method (TraceMethodType, optional):
            The method to calculate the trace of rho.

            - Matrix operation methods:
                For the matrix operation methods, it will require rho has been calculated first.
                - "trace_of_matmul": Use `np.trace(np.matmul(rho_m1, rho_m2))`
                    to calculate the each summation item in `rho_m_list`.
                - "einsum_ij_ji": Use `np.einsum("ij,ji", rho_m1, rho_m2)`
                    to calculate the each summation item in `rho_m_list`.
                - "einsum_aij_bji_to_ab_numpy": Use
                    `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                    This is the fastest implementation to calculate the trace of Rho
                    if JAX is not available.
                - "einsum_aij_bji_to_ab_jax": Use
                    `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                    This is the fastest implementation to calculate the trace of Rho
                    if JAX is available.

            - Non-matrix operation methods:
                - "nomatmul_trace_py": Use pure Python implementation without multiprocessing.
                - "nomatmul_trace_rust": Use Rust implementation via PyO3.

            - Skip calculation of trace:
                - "skip_trace": Skip the trace calculation and return NaN.

            - The extended BitWise methods:
                - "bitwise_py": Use pure Python implementation of BitWise method.

            For the non-matrix operation methods, it will directly calculate the trace from
            the counts and random basis.

            Default to DEFAULT_TRACE_METHOD.

    Returns:
        PurityValueKindExtend: The kind of purity value calculation.
    """
    trace_method = handle_trace_method_extend(trace_method)
    if isinstance(trace_method, BitWiseTraceMethod):
        return "bitwise"

    return verify_purity_value_kind(rho_method, trace_method)


def default_method_on_value_kind_extend(
    value_kind: PurityValueKindExtend,
) -> tuple[RhoMethod, TraceMethod | BitWiseTraceMethod]:
    """Get the default method on each kind of purity value calculation.

    Args:
        purity_value_kind (PurityValueKind):
            The kind of purity value calculation.

    Raises:
        ValueError: If the purity value kind is not recognized.

    Returns:
        tuple[RhoMethod, TraceMethod | BitWiseTraceMethod]:
            The default (rho_method, trace_method).
    """
    if value_kind == "bitwise":
        return RhoMethod.get_default(), BitWiseTraceMethod.get_default()
    return default_method_on_value_kind(value_kind)


class ClassicalShadowPurityExtend(TypedDict):
    """The expectation value of Rho."""

    purity: FloatType
    """The purity calculated by classical shadow."""
    entropy: FloatType
    """The entropy calculated by classical shadow."""
    purity_value_kind: PurityValueKind | str
    """The kind of purity value calculation.
    This will depend on the rho_method and trace_method.

    If it is not one of the defined kinds, it will be "unknown".
    """
    taking_time: float
    """The time taken for the calculation."""
    trace_method: TraceMethodExtendType
    """The method to calculate the trace of rho."""
