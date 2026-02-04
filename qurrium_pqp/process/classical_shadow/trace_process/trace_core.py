"""Qurrium PQP Crossroads - Post Processing - Classical Shadow - Trace Process Extended
(:mod:`qurrium_pqp.process.classical_shadow.trace_process.trace_core`)

"""

import time
from typing import Literal
import numpy as np
import numpy.typing as npt

from qurry.process.utils import FloatType
from qurry.process.classical_shadow.utils import convert_to_basis_spin
from qurry.process.classical_shadow.trace_process import (
    TraceMethod,
    TraceMethodType,
    DEFAULT_TRACE_METHOD,
    all_trace_core,
)

from .bitwise import BitWiseTraceMethod, bitwise_py_core

TraceMethodExtendType = BitWiseTraceMethod | TraceMethodType
"""The method to calculate the trace of rho.

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
"""


def handle_trace_method_extend(
    trace_method: TraceMethodExtendType,
) -> TraceMethod | BitWiseTraceMethod:
    """Handle the trace method extend type.

    Args:
        trace_method (TraceMethodExtendType):
            The trace method extend type.

    Returns:
        TraceMethod | BitWiseTraceMethod:
            The trace method.
    """
    if isinstance(trace_method, str):
        try:
            return BitWiseTraceMethod.from_string(trace_method)
        except ValueError:
            return TraceMethod.from_string(trace_method)

    if not isinstance(trace_method, (TraceMethod, BitWiseTraceMethod)):
        raise ValueError(
            f"Invalid trace method: {trace_method}. Supported methods are: "
            + ", ".join(TraceMethod.get_all_methods() + BitWiseTraceMethod.get_all_methods())
        )

    return trace_method


def all_trace_core_extend(
    shots: int,
    counts: list[dict[str, int]],
    random_basis_array: list[list[Literal[0, 1, 2] | int]],
    rho_m_list: list[npt.NDArray[np.complex128]],
    selected_classical_registers_sorted: list[int],
    trace_method: TraceMethodExtendType = DEFAULT_TRACE_METHOD,
) -> tuple[FloatType, FloatType, float]:
    """Calculate the trace by all given methods.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis_array (list[list[Literal[0, 1, 2] | int]]):
            The random basis for classical shadow.
        rho_m_list (list[npt.NDArray[np.complex128]]):
            The list of Rho M.
            It should be a list of 2-dimensional arrays.
        selected_classical_registers_sorted (list[int]):
            The **sorted** list of the index of the selected classical registers.

        trace_method (TraceMethodExtendType, optional):
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
        tuple[FloatType, FloatType, float]:
            The purity, the second Renyi entropy, and the time taken (in seconds).
    """

    trace_method = handle_trace_method_extend(trace_method)

    if isinstance(trace_method, TraceMethod):
        return all_trace_core(
            shots=shots,
            counts=counts,
            random_basis_array=random_basis_array,
            rho_m_list=rho_m_list,
            selected_classical_registers_sorted=selected_classical_registers_sorted,
            trace_method=trace_method,
        )

    if not trace_method == BitWiseTraceMethod.BITWISE_PY:
        raise BitWiseTraceMethod.value_error()

    begin = time.time()
    selected_clreg_sorted = sorted(selected_classical_registers_sorted)
    pauli_basis, spin_outcome = convert_to_basis_spin(shots, counts, random_basis_array)

    purity = bitwise_py_core(
        pauli_basis=pauli_basis,
        spin_outcome=spin_outcome,
        subsystem=selected_clreg_sorted,
    )
    return purity, -np.log2(purity), time.time() - begin
