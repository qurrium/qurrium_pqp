"""ShadowUnveilMore - Analysis (:mod:`qurrium_pqp.qurries.classical_shadow_more.analysis`)"""

from typing import Any, Literal, overload, TypeVar, Generic
from collections.abc import Iterable
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from qurry.qurries.classical_shadow.arguments import SUArguments
from qurry.qurries.classical_shadow.analysis import SUMiddleware, SUBasicResult, SUEstimationResult
from qurry.qurries.classical_shadow.utils import get_random_basis_array
from qurry.qurrium import (
    Commonparams,
    AnalysisPrototype,
    AnalyzeArgs,
    ProcessEntriesPrototype,
    AnalysisResultsPrototype,
)
from qurry.qurrium.utils import bitstring_mapping_getter
from qurry.process.utils import counts_list_recount_pyrust, FloatType
from qurry.process.classical_shadow import (
    RhoMethod,
    RhoMethodType,
    DEFAULT_RHO_METHOD,
    ShadowBasisType,
    ShadowRandomBasis,
    TraceMethod,
    DEFAULT_TRACE_METHOD,
    ListTraceMethod,
    ListTraceMethodType,
    DEFAULT_LIST_TRACE_METHOD,
    PurityValueKind,
    ClassicalShadowBasic,
    ClassicalShadowPurity,
    EstimationOfObservable,
)

from ...process.classical_shadow import (
    BitWiseTraceMethod,
    TraceMethodExtendType,
    handle_trace_method_extend,
    classical_shadow_complex_extend,
)


class SUMAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis` and
    :meth:`~qurrium_pqp.qurries.classical_shadow_more.experiment.SUExperiment.analyze`.
    """

    selected_qubits: list[int] | None
    """The selected qubits."""
    # estimation of given operators
    given_operators: list[npt.NDArray] | None
    """The list of the operators to estimate."""
    accuracy_prob_comp_delta: FloatType
    """The accuracy probability for computing delta."""
    max_shadow_norm: FloatType | None
    """The maximum shadow norm of the given operators."""
    # other config
    rho_method: RhoMethodType
    """The method to reconstruct the density matrix."""
    trace_method: TraceMethodExtendType
    """The method to compute the trace."""
    estimate_trace_method: ListTraceMethodType
    """The method to estimate the trace."""
    counts_used: Iterable[int] | None
    """The index of the counts used."""


@dataclass(frozen=True)
class SUMProcessEntries(ProcessEntriesPrototype):
    """The entries for post-processing."""

    __name__ = "SUMProcessEntries"

    random_basis_array: list[list[Literal[0, 1, 2] | int]]
    """The random basis for classical shadow."""
    selected_classical_registers: list[int] | None
    """The list of **the index of the selected_classical_registers**."""

    # esitimation of given operators
    given_operators: list[npt.NDArray] | None
    """The list of the operators to estimate."""
    accuracy_predict_epsilon: FloatType
    r"""The prediction of accuracy, which used the notation :math:`\epsilon`
    and mentioned in Theorem S1 in the supplementary material,
    the equation (S13) in the supplementary material.

    We can calculate the prediction of accuracy :math:`\epsilon` from the equation (S13)
    in the supplementary material, the equation (S13) is as follows,

    .. math::
        N = \frac{34}{\epsilon^2} \max_{1 \leq i \leq M} 
        || O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2

    where :math:`\epsilon` is the prediction of accuracy,
    and :math:`M` is the number of given operatorsm
    and :math:`N` is the number of classical snapshots.
    The :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` is maximum shadow norm,
    which is defined in the supplementary material with value between 0 and 1.
    """
    maximum_shadow_norm: FloatType | None
    r"""The maximum shadow norm, which is defined in the supplementary material 
    with value between 0 and 1.
    The maximum shadow norm is used to calculate the prediction of accuracy :math:`\epsilon`
    from the equation (S13) in the supplementary material.

    We can calculate the prediction of accuracy :math:`\epsilon` from the equation (S13)
    in the supplementary material, the equation (S13) is as follows,

    .. math::
        N = \frac{34}{\epsilon^2} \max_{1 \leq i \leq M} 
        || O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2

    where :math:`\epsilon` is the prediction of accuracy,
    and :math:`M` is the number of given operatorsm
    and :math:`N` is the number of classical snapshots.
    The :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` is maximum shadow norm,
    which is defined in the supplementary material with value between 0 and 1.

    Due to maximum shadow norm is complex and it is a norm,
    we suppose we have the worst case scenario,
    where the maximum shadow norm is 1 as default.
    Thus, we can simplify the equation to:

    .. math::
        N = \frac{34}{\epsilon^2}
    """

    rho_method: RhoMethodType
    """The method to reconstruct the density matrix."""
    shadow_basis: ShadowRandomBasis
    """The shadow basis used for classical shadow."""
    trace_method: TraceMethodExtendType
    """The method to calculate the trace of Rho."""
    estimate_trace_method: ListTraceMethodType
    """The method to calculate the trace of Rho."""

    def export(self) -> dict[str, Any]:
        """Export the results for file writing.

        Returns:
            dict[str, Any]: The data to be exported.
        """
        rho_method = (
            self.rho_method
            if isinstance(self.rho_method, RhoMethod)
            else RhoMethod.from_string(self.rho_method)
        )
        trace_method = (
            self.trace_method
            if isinstance(self.trace_method, (TraceMethod, BitWiseTraceMethod))
            else handle_trace_method_extend(self.trace_method)
        )
        estimate_trace_method = (
            self.estimate_trace_method
            if isinstance(self.estimate_trace_method, ListTraceMethod)
            else ListTraceMethod.from_string(self.estimate_trace_method)
        )

        return {
            "shots": self.shots,
            "random_basis_array": self.random_basis_array,
            "selected_classical_registers": (
                list(self.selected_classical_registers)
                if self.selected_classical_registers is not None
                else None
            ),
            "given_operators": (
                [
                    np.array(np.array(op, dtype=np.complex128), dtype=str).tolist()
                    for op in self.given_operators
                ]
                if self.given_operators is not None
                else None
            ),
            "accuracy_predict_epsilon": float(self.accuracy_predict_epsilon),
            "maximum_shadow_norm": (
                None if self.maximum_shadow_norm is None else float(self.maximum_shadow_norm)
            ),
            "rho_method": rho_method.value,
            "shadow_basis": self.shadow_basis.export(),
            "trace_method": trace_method.value,
            "estimate_trace_method": estimate_trace_method.value,
        }

    @classmethod
    def ingest(cls, raw_dict: dict[str, Any]):
        """Ingest from a serialized dictionary.

        Args:
            raw_dict (dict[str, Any]): The raw read dictionary.
        """
        missing_fields = set(cls.dataclass_fields()) - set(raw_dict.keys())
        if missing_fields:
            raise ValueError(f"Missing fields for {cls.__name__}: {', '.join(missing_fields)}")
        given_operators = (
            [np.array(op, dtype=complex) for op in raw_dict["given_operators"]]
            if raw_dict["given_operators"] is not None
            else None
        )

        return cls(
            shots=raw_dict["shots"],
            random_basis_array=raw_dict["random_basis_array"],
            selected_classical_registers=raw_dict["selected_classical_registers"],
            given_operators=given_operators,
            accuracy_predict_epsilon=float(raw_dict["accuracy_predict_epsilon"]),
            maximum_shadow_norm=(
                None
                if raw_dict["maximum_shadow_norm"] is None
                else float(raw_dict["maximum_shadow_norm"])
            ),
            rho_method=RhoMethod.from_string(raw_dict["rho_method"]),
            shadow_basis=ShadowRandomBasis.ingest(raw_dict["shadow_basis"]),
            trace_method=handle_trace_method_extend(raw_dict["trace_method"]),
            estimate_trace_method=ListTraceMethod.from_string(raw_dict["estimate_trace_method"]),
        )

    def __repr__(self) -> str:
        """The representation of the process entries."""
        entries_str_dict = {
            field: f"{field}={getattr(self, field)!r}"
            for field in self.fields
            if field not in ["random_basis_array", "given_operators"]
        }

        entries_str_dict["random_basis_array"] = (
            (f"random_basis_array=[...{len(self.random_basis_array)} items...]")
            if self.random_basis_array is not None
            else "random_basis_array=None"
        )
        entries_str_dict["given_operators"] = (
            (f"given_operators=[...{len(self.given_operators)} items...]")
            if self.given_operators is not None
            else "given_operators=None"
        )

        field_strs = [entries_str_dict[field] for field in self.fields]
        return f"{self.__class__.__name__}({', '.join(field_strs)})"


@dataclass(frozen=True)
class SUMPurityResult(AnalysisResultsPrototype):
    """The purity result of :class:`~qurrium_pqp.qurries.classical_shadow_more.analysis.SUAnalysis`."""

    __name__ = "SUMPurityResult"

    purity: FloatType
    """The purity of the density matrix."""
    entropy: FloatType
    """The second Renyi entropy of the density matrix."""
    purity_value_kind: PurityValueKind | str
    """The kind of purity value."""
    taking_time: float
    """The time taken for the calculation."""
    trace_method: TraceMethodExtendType
    """The method to calculate the trace of Rho."""

    def export(self) -> dict[str, Any]:
        """Export the results for file writing.

        Returns:
            dict[str, Any]: The data to be exported.
        """
        return {
            "purity": float(self.purity),
            "entropy": float(self.entropy),
            "purity_value_kind": self.purity_value_kind,
            "taking_time": float(self.taking_time),
            "trace_method": (
                handle_trace_method_extend(self.trace_method).value
                if isinstance(self.trace_method, str)
                else self.trace_method.value
            ),
        }

    @classmethod
    def ingest(cls, raw_dict: dict[str, Any]):
        """Ingest from a serialized dictionary.

        Args:
            raw_dict (dict[str, Any]): The raw read dictionary.
        """
        missing_fields = set(cls.dataclass_fields()) - set(raw_dict.keys())
        if missing_fields:
            raise ValueError(f"Missing fields for {cls.__name__}: {', '.join(missing_fields)}")

        return cls(
            purity=float(raw_dict["purity"]),
            entropy=float(raw_dict["entropy"]),
            purity_value_kind=raw_dict["purity_value_kind"],
            taking_time=float(raw_dict["taking_time"]),
            trace_method=handle_trace_method_extend(raw_dict["trace_method"]),
        )


_K_Inst = TypeVar("_K_Inst", bound=str)
"""The key type variable for SUResultsType and SUResults."""
_V_Inst = TypeVar("_V_Inst", bound=AnalysisResultsPrototype)
"""The value type variable for SUResultsType and SUResults."""


class SUMResultsType(
    Generic[_K_Inst, _V_Inst],
    dict[
        Literal["basic", "purity", "estimation"] | _K_Inst | str,
        type[SUBasicResult] | type[SUMPurityResult] | type[SUEstimationResult] | type[_V_Inst],
    ],
):
    """The results of :class:`~qurrium_pqp.qurries.classical_shadow_more.analysis.SUAnalysis`."""

    @overload
    def __getitem__(self, key: Literal["basic"]) -> type[SUBasicResult]: ...
    @overload
    def __getitem__(self, key: Literal["purity"]) -> type[SUMPurityResult]: ...
    @overload
    def __getitem__(self, key: Literal["estimation"]) -> type[SUEstimationResult]: ...
    @overload
    def __getitem__(self, key: _K_Inst) -> type[_V_Inst]: ...
    @overload
    def __getitem__(
        self, key: str
    ) -> type[SUBasicResult] | type[SUMPurityResult] | type[SUEstimationResult] | type[_V_Inst]: ...

    def __getitem__(self, key):
        return super().__getitem__(key)


class SUMResults(
    Generic[_K_Inst, _V_Inst],
    dict[
        Literal["basic", "purity", "estimation"] | _K_Inst | str,
        SUBasicResult | SUMPurityResult | SUEstimationResult | _V_Inst,
    ],
):
    """The results of :class:`~qurrium_pqp.qurries.classical_shadow_more.analysis.SUAnalysis`."""

    @overload
    def __getitem__(self, key: Literal["basic"]) -> SUBasicResult: ...
    @overload
    def __getitem__(self, key: Literal["purity"]) -> SUMPurityResult: ...
    @overload
    def __getitem__(self, key: Literal["estimation"]) -> SUEstimationResult: ...
    @overload
    def __getitem__(self, key: _K_Inst) -> _V_Inst: ...
    @overload
    def __getitem__(
        self, key: str
    ) -> SUBasicResult | SUMPurityResult | SUEstimationResult | _V_Inst: ...

    def __getitem__(self, key):
        return super().__getitem__(key)


class SUMAnalysis(
    Generic[_K_Inst, _V_Inst],
    AnalysisPrototype[
        SUArguments,
        SUMAnalyzeArgs,
        SUMiddleware,
        SUMProcessEntries,
        SUMResultsType[_K_Inst, _V_Inst],
        SUMResults[_K_Inst, _V_Inst],
    ],
):
    """The container for the analysis of
    :class:`~qurrium_pqp.qurries.classical_shadow_more.experiment.SUExperiment`."""

    __name__ = "SUMAnalysis"

    results: SUMResults[_K_Inst, _V_Inst]
    """The results of the analysis."""

    @classmethod
    def analyze_arguments_type(cls) -> type[SUMAnalyzeArgs]:
        """The type of analyze arguments."""
        return SUMAnalyzeArgs

    @classmethod
    def middleware_entries_type(cls) -> type[SUMiddleware]:
        """The middleware entries type for this analysis."""
        return SUMiddleware

    @classmethod
    def postprocess_entries_type(cls) -> type[SUMProcessEntries]:
        """The post-process entries type for this analysis."""
        return SUMProcessEntries

    @classmethod
    def available_results_types(cls) -> SUMResultsType[_K_Inst, _V_Inst]:
        """The available result types for this analysis."""
        return SUMResultsType(
            {"basic": SUBasicResult, "purity": SUMPurityResult, "estimation": SUEstimationResult}
        )

    @classmethod
    def quantities(
        cls,
        shots: int,
        counts: list[dict[str, int]],
        random_basis_array: list[list[Literal[0, 1, 2] | int]],
        selected_classical_registers: Iterable[int] | None,
        # estimation of given operators
        given_operators: list[npt.NDArray] | None,
        accuracy_prob_comp_delta: FloatType,
        max_shadow_norm: FloatType | None,
        # other config
        rho_method: RhoMethodType,
        shadow_basis: ShadowBasisType,
        trace_method: TraceMethodExtendType,
        estimate_trace_method: ListTraceMethodType,
    ) -> tuple[ClassicalShadowBasic, ClassicalShadowPurity | None, EstimationOfObservable | None]:
        r"""Calculate the classical shadow quantities.

        Args:
            shots (int):
                The number of shots.
            counts (list[dict[str, int]]):
                The list of the counts.
            random_basis_array (list[list[Literal[0, 1, 2] | int]]):
                The random basis for classical shadow.
            selected_classical_registers (Iterable[int]):
                The list of **the index of the selected_classical_registers**.

            given_operators (list[npt.NDArray] | None, optional):
                The list of the operators to estimate. Defaults to None.
            accuracy_prob_comp_delta (FloatType, optional):
                The accuracy probability component delta. Defaults to 0.01.
            max_shadow_norm (FloatType | None, optional):
                The maximum shadow norm. Defaults to None.
                If it is None, it will be calculated by the largest shadow norm upper bound.
                If it is not None, it must be a positive float number.
                It is :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` in equation.

            rho_method (RhoMethodType, optional):
                It can be either "multi_shots_proto", "multi_shots", "multi_shots_vectorized",
                "single_shots_proto", "single_shots", or "single_shots_vectorized".

                For the "multi_shots_*" methods, the counts and random basis are used as is.
                For the "single_shots_*" methods, the counts and random basis are
                converted to single shot per snapshot for classical shadow post-processing.

                **Warning: Althought larger snapshots number means more accurate values.**
                **But if your shots number is large,**
                **this may significantly increase memory usage**
                **and require a lot of computing resource.**
                **In worst scenrio, this will break your computer.**
                **Please reconsider for performance.**

                - "multi_shots_proto": Use Numpy to calculate the rho_m.
                - "multi_shots": Use Numpy to calculate the rho_m with precomputed values.
                - "multi_shots_vectorized": Use Numpy to calculate the rho_m
                    with a vectorized workflow.

                - "single_shots_proto": Use Numpy to calculate the rho_m
                    with converted single shot counts.
                - "single_shots": Use Numpy to calculate the rho_m
                    with precomputed values with converted single shot counts.
                - "single_shots_vectorized": Use Numpy to calculate the rho_m
                    with a vectorized workflow with converted single shot counts.

                Currently, "multi_shots" is the best option for performance.
                Default to DEFAULT_RHO_METHOD, which is "multi_shots".
            shadow_basis (ShadowBasisType, optional):
                The shadow basis to use. Defaults to :data:`DEFAULT_SHADOW_BASIS`.

                Here are the built-in basis sets:
                - `RX_RY_RZ`:
                    Uses :math:`R_X(\frac{\pi}{2})`,
                    :math:`R_Y(-\frac{\pi}{2})`, and :math:`R_Z(0)` gates.
                - `H_H-Sdg_I`:
                    Uses :math:`H`, :math:`H` followed by :math:`S^\dagger`, and Identity gates.
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
            estimate_trace_method (ListTraceMethodType, optional):
                The method to use for the calculation.

                - "einsum_aij_bji_to_ab_numpy":
                    Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                    This is the fastest implementation to calculate the trace of Rho
                    if JAX is not available.
                - "einsum_aij_bji_to_ab_jax":
                    Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                    This is the fastest implementation to calculate the trace of Rho.

                Defaults to DEFAULT_LIST_TRACE_METHOD.

        Returns:
            ClassicalShadowComplex: The result of the classical shadow.
        """

        return classical_shadow_complex_extend(
            shots=shots,
            counts=counts,
            random_basis_array=random_basis_array,
            selected_classical_registers=selected_classical_registers,
            # estimation of given operators
            given_operators=given_operators,
            accuracy_prob_comp_delta=accuracy_prob_comp_delta,
            max_shadow_norm=max_shadow_norm,
            # other config
            rho_method=rho_method,
            shadow_basis=shadow_basis,
            trace_method=trace_method,
            estimate_trace_method=estimate_trace_method,
        )

    @classmethod
    def generate_entries(
        cls,
        arguments: SUArguments,
        commonparams: Commonparams,
        counts: list[dict[str, int]],
        analyze_arguments: SUMAnalyzeArgs,
        random_basis: dict[int, dict[int, int]] | None = None,
    ) -> tuple[SUMAnalyzeArgs, SUMiddleware, SUMProcessEntries, list[dict[str, int]]]:
        """Generate the entries for analysis.

        Args:
            arguments (SUArguments): The arguments for the experiment.
            commonparams (Commonparams): The common parameters for the experiment.
            counts (list[dict[str, int]]): The counts from the experiment.
            analyze_arguments (SUAnalyzeArgs): The analyze arguments.
            random_basis (dict[int, dict[int, int]] | None, optional):
                The random basis for classical shadow. Defaults to None.

        Returns:
            The generated entries for analysis and the possibly filtered counts.
        """

        if random_basis is None:
            raise ValueError("random_basis should be specified.")
        if len(random_basis) != arguments.snapshots:
            raise ValueError(
                f"The number of random basis should be {arguments.snapshots}, "
                + f"but got {len(random_basis)}."
            )

        counts_used = analyze_arguments.get("counts_used", None)
        if isinstance(counts_used, Iterable):
            if max(counts_used) >= len(counts):
                raise ValueError(
                    f"counts_used should be less than {len(counts)}, but get {max(counts_used)}."
                )
            counts = [counts[i] for i in counts_used]
        elif counts_used is not None:
            raise TypeError(
                f"counts_used should be Iterable[int] or None, but got {type(counts_used)}."
            )

        bitstring_mapping, final_mapping = bitstring_mapping_getter(
            counts, arguments.registers_mapping
        )

        selected_qubits = analyze_arguments.get("selected_qubits", None)
        selected_qubits = (
            [qi % arguments.actual_num_qubits for qi in selected_qubits]
            if selected_qubits
            else list(arguments.registers_mapping.keys())
        )
        if len(set(selected_qubits)) != len(selected_qubits):
            raise ValueError(
                f"selected_qubits should not have duplicated elements, but got {selected_qubits}."
            )

        counts_of_last_clreg = counts_list_recount_pyrust(
            counts,
            len(next(iter(counts[0]))),
            list(final_mapping.values()),
        )

        # random basis follow normal register mapping
        # for it does not need to consider extra classical registers
        # but effect by count_used
        random_basis_array = get_random_basis_array(
            arguments.registers_mapping, random_basis, counts_used
        )

        return (
            analyze_arguments,
            SUMiddleware(
                num_qubits=arguments.actual_num_qubits,
                selected_qubits=selected_qubits,
                registers_mapping=arguments.registers_mapping,
                bitstring_mapping=bitstring_mapping,
                final_mapping=final_mapping,
                unitary_located=arguments.unitary_located,
                counts_used=counts_used,
            ),
            SUMProcessEntries(
                shots=commonparams.shots,
                random_basis_array=random_basis_array,
                selected_classical_registers=[
                    arguments.registers_mapping[qi] for qi in selected_qubits
                ],
                # estimation of given operators
                given_operators=analyze_arguments.get("given_operators", None),
                accuracy_predict_epsilon=analyze_arguments.get("accuracy_prob_comp_delta", 0.01),
                maximum_shadow_norm=analyze_arguments.get("max_shadow_norm", None),
                # other config
                rho_method=analyze_arguments.get("rho_method", DEFAULT_RHO_METHOD),
                shadow_basis=arguments.shadow_basis,
                trace_method=analyze_arguments.get("trace_method", DEFAULT_TRACE_METHOD),
                estimate_trace_method=analyze_arguments.get(
                    "estimate_trace_method", DEFAULT_LIST_TRACE_METHOD
                ),
            ),
            counts_of_last_clreg,
        )

    @classmethod
    def perform_analysis(
        cls,
        arguments: SUArguments,
        commonparams: Commonparams,
        counts: list[dict[str, int]],
        analyze_arguments: SUMAnalyzeArgs,
        serial: int,
        outfields: dict[str, Any] | None = None,
        datetime: str | None = None,
        random_basis: dict[int, dict[int, int]] | None = None,
    ):
        """Perform the analysis for the experiment.

        Args:
            arguments (SUArguments): The arguments for the experiment.
            commonparams (Commonparams): The common parameters for the experiment.
            counts (list[dict[str, int]]): The counts from the experiment.
            analyze_arguments (SUMAnalyzeArgs): The analyze arguments.
            serial (int): The serial number of the analysis.
            outfields (dict[str, Any] | None, optional): The output fields. Defaults to None.
            datetime (str | None, optional): The datetime string. Defaults to None.
            random_basis (dict[int, dict[int, int]] | None, optional):
                The random basis for classical shadow. Defaults to None.

        Returns:
            The analysis result.
        """

        analyze_arguments, middleware_entries, postprocess_entries, selected_counts = (
            cls.generate_entries(arguments, commonparams, counts, analyze_arguments, random_basis)
        )

        cs_basic_obj, cs_trace_obj, cs_estimation_obj = cls.quantities(
            shots=commonparams.shots,
            counts=selected_counts,
            random_basis_array=postprocess_entries.random_basis_array,
            selected_classical_registers=postprocess_entries.selected_classical_registers,
            # estimation of given operators
            given_operators=postprocess_entries.given_operators,
            accuracy_prob_comp_delta=postprocess_entries.accuracy_predict_epsilon,
            max_shadow_norm=postprocess_entries.maximum_shadow_norm,
            # other config
            rho_method=postprocess_entries.rho_method,
            shadow_basis=arguments.shadow_basis,
            trace_method=postprocess_entries.trace_method,
            estimate_trace_method=postprocess_entries.estimate_trace_method,
        )

        results = SUMResults[_K_Inst, _V_Inst]({"basic": SUBasicResult(**cs_basic_obj)})
        if cs_trace_obj is not None:
            results["purity"] = SUMPurityResult(**cs_trace_obj)
        if cs_estimation_obj is not None:
            results["estimation"] = SUEstimationResult(**cs_estimation_obj)

        return cls(
            analyze_arguments=analyze_arguments,
            middleware_entries=middleware_entries,
            postprocess_entries=postprocess_entries,
            results=results,
            serial=serial,
            outfields=outfields,
            datetime=datetime,
        )
