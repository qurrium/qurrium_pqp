"""Qurrium PQP Crossroads - Qurrium Implementation Conversion
(:mod:`qurrium_pqp.qurries.conversion`)

This module provides functions to transform
the output of Qurrium to the text file format in
Predicting Properties of Quantum Many-Body Systems,
which the example is in following:

.. code-block:: text

    [system size]
    [subsystem 1 size] [position of qubit 1] [position of qubit 2] ...
    [subsystem 2 size] [position of qubit 1] [position of qubit 2] ...

"""

from typing import Literal, Any, overload
from pathlib import Path

from qiskit import QuantumCircuit

from qurry.tools import DatetimeDict
from qurry.qurrium import WCKeyable
from qurry.qurrium.experiment.utils import exp_id_process, make_qasm_strings
from qurry.qurries.classical_shadow import SUExperiment

from .classical_shadow_more.experiment import SUMExperiment


# qurrium to PGP transformation
def check_classical_shadow_exp(
    classical_shadow_exp: SUExperiment | SUMExperiment,
) -> tuple[dict[int, dict[int, int]], dict[int, int]]:
    """Check if the classical shadow experiment is valid for conversion,
    then return its random basis and registers mapping.

    Args:
        classical_shadow_exp (SUExperiment | SUMExperiment):
            The Qurrium experiment to convert.

    Raises:
        TypeError: If the input is not a SUExperiment or SUMExperiment.
        ValueError: If required attributes are missing or invalid.

    Returns:
        tuple[dict[int, dict[int, int]], dict[int, int]]:
            random_unitary_ids mapping from the experiment and registers mapping.
    """
    if not isinstance(classical_shadow_exp, (SUExperiment, SUMExperiment)):
        raise TypeError(
            "The input must be an instance of SUExperiment or SUMExperiment "
            + "from qurry.qurries.classical_shadow or "
            + "qurrium_pqp.qurries.classical_shadow_more."
        )
    if classical_shadow_exp.args.unitary_located is None:
        raise ValueError("The unitary located must be specified in the experiment.")
    if classical_shadow_exp.args.qubits_measured is None:
        raise ValueError("The qubits measured must be specified in the experiment.")
    if classical_shadow_exp.args.registers_mapping is None:
        raise ValueError("The registers mapping must be specified in the experiment.")
    measured_qubits = set(classical_shadow_exp.args.qubits_measured)
    unitary_located = set(classical_shadow_exp.args.unitary_located)
    missing_qubits = measured_qubits - unitary_located
    if missing_qubits:
        raise ValueError(
            "All measured qubits must be part of the unitary located. "
            + f"Missing qubits: {sorted(missing_qubits)}"
        )
    if classical_shadow_exp.args.random_basis is None:
        raise ValueError("The experiment must have 'random_basis' in args.")

    return (
        classical_shadow_exp.args.random_basis,
        classical_shadow_exp.args.registers_mapping,
    )


def qurrium_to_pqp_result(
    shadow_exp: SUExperiment | SUMExperiment,
) -> tuple[list[list[int]], list[list[int]]]:
    """Convert a Qurrium experiment to Predicting Quantum Properties result.

    Args:
        shadow_exp (SUExperiment | SUMExperiment):
            The Qurrium experiment to convert.

    Returns:
        A tuple containing a list of pauli basis and a list of spin outcomes
    """

    check_classical_shadow_exp(shadow_exp)

    return shadow_exp.get_basis_spin_format()


@overload
def pqp_result_to_qurrium(
    pauli_basis: list[list[int]],
    spin_outcome: list[list[int]],
    qurrium_class: Literal["classical_shadow"],
    # common params and arguments
    exp_name: str = "experiment.pqp",
    target_circuit: QuantumCircuit | None = None,
    backend_name: str | None = None,
    tags: tuple[str, ...] | None = None,
    datetimes: DatetimeDict | dict[str, str] | None = None,
    outfields: dict[str, Any] | None = None,
    multiprocess_build: bool = False,
    # multimanager
    serial: int | None = None,
    summoner_id: str | None = None,
    summoner_name: str | None = None,
    # process tool
    qasm_version: Literal["qasm2", "qasm3"] = "qasm3",
    save_location: str | None = None,
) -> SUExperiment: ...


@overload
def pqp_result_to_qurrium(
    pauli_basis: list[list[int]],
    spin_outcome: list[list[int]],
    qurrium_class: Literal["classical_shadow_more"],
    # common params and arguments
    exp_name: str = "experiment.pqp",
    target_circuit: QuantumCircuit | None = None,
    backend_name: str | None = None,
    tags: tuple[str, ...] | None = None,
    datetimes: DatetimeDict | dict[str, str] | None = None,
    outfields: dict[str, Any] | None = None,
    multiprocess_build: bool = False,
    # multimanager
    serial: int | None = None,
    summoner_id: str | None = None,
    summoner_name: str | None = None,
    # process tool
    qasm_version: Literal["qasm2", "qasm3"] = "qasm3",
    save_location: str | None = None,
) -> SUMExperiment: ...


def pqp_result_to_qurrium(
    pauli_basis,
    spin_outcome,
    qurrium_class="classical_shadow",
    # common params and arguments
    exp_name="experiment.pqp",
    target_circuit=None,
    backend_name=None,
    tags=None,
    datetimes=None,
    outfields=None,
    multiprocess_build=False,
    # multimanager
    serial=None,
    summoner_id=None,
    summoner_name=None,
    # process tool
    qasm_version: Literal["qasm2", "qasm3"] = "qasm3",
    save_location: str | None = None,
):
    """Convert Predicting Quantum Properties result to a Qurrium experiment.

    Args:
        pauli_basis (list[list[int]]):
            The list of pauli basis from PQP result.
        spin_list (list[list[int]]):
            The list of spin outcomes from PQP result.
        qurrium_class (Literal['classical_shadow', 'classical_shadow_more'], optional):
            The Qurrium class to use for the experiment.
            Defaults to 'classical_shadow'.

        exp_name (str, optional):
            The name of the experiment. Defaults to 'experiment.pqp'.
        target_circuit (QuantumCircuit | None, optional):
            The target circuit for the experiment.
        backend_name (str | None, optional):
            The name of the backend used for the experiment.
        tags (tuple[str, ...] | None, optional):
            Tags for the experiment. Defaults to None.
        datetimes (DatetimeDict | dict[str, str] | None, optional):
            Datetime information for the experiment. Defaults to None.
        outfields (dict[str, Any] | None, optional):
            Additional data to include. Defaults to None.
        multiprocess_build (bool, optional):
            Whether to use multiprocessing for building the experiment. Defaults to False.

        serial (int | None, optional):
            Serial number for the experiment in :class:`~qurry.qurrium.multimanager.MultiManager`.
        summoner_id (str | None, optional):
            ID of the summoner for :class:`~qurry.qurrium.multimanager.MultiManager`.
        summoner_name (str | None, optional):
            Name of the summoner for :class:`~qurry.qurrium.multimanager.MultiManager`.

        qasm_version (Literal['qasm2', 'qasm3'], optional):
            The OpenQASM version to use for the circuit. Defaults to 'qasm3'.
        save_location (str | None, optional):
            The location to save the experiment.

    Returns:
        SUExperiment: The converted Qurrium experiment.
    """
    if target_circuit is not None and not isinstance(target_circuit, QuantumCircuit):
        raise TypeError("The target_circuit must be a QuantumCircuit instance.")
    if len(pauli_basis) != len(spin_outcome):
        raise ValueError(
            "The length of pauli_basis must be equal to the length of spin_list. "
            + f"Found {len(pauli_basis)} and {len(spin_outcome)}."
        )
    if len(pauli_basis) == 0:
        raise ValueError("The pauli_basis and spin_list cannot be empty.")
    num_random_basis = len(pauli_basis)
    num_classical_register = len(pauli_basis[0])
    if any(len(pb) != num_classical_register for pb in pauli_basis):
        raise ValueError("All pauli_basis entries must have the same length.")
    if any(len(sl) != num_classical_register for sl in spin_outcome):
        raise ValueError("All spin_list entries must have the same length.")
    if qurrium_class == "classical_shadow":
        ExperimentInstance = SUExperiment
    elif qurrium_class == "classical_shadow_more":
        ExperimentInstance = SUMExperiment
    else:
        raise ValueError(
            "Invalid qurrium_class. Supported classes are 'classical_shadow' and "
            + "'classical_shadow_more'."
        )

    outfields_inner = {} if outfields is None else outfields.copy()
    outfields_inner["denoted"] = "This is a PQP transformed Qurrium experiment."

    counts = []
    random_basis = {}
    for i, (single_pauli_basis, single_spin_outcome) in enumerate(zip(pauli_basis, spin_outcome)):
        bitstring = "".join("1" if spin == 1 else "0" for spin in reversed(single_spin_outcome))
        counts.append({bitstring: 1})
        random_basis[i] = dict(enumerate(single_pauli_basis))

    current_exp = ExperimentInstance(
        arguments={
            "exp_name": exp_name,
            "snapshot": num_random_basis,
            "qubits_measured": list(range(num_classical_register)),
            "registers_mapping": {qi: qi for qi in range(num_random_basis)},
            "actual_num_qubits": num_classical_register,
            "unitary_located": list(range(num_random_basis)),
            "random_basis": random_basis,
        },
        commonparams={
            "exp_id": exp_id_process(None),
            "target_keys": [] if target_circuit is None else [target_circuit.name],
            "shots": 1,
            "backend": "pgp_transformed" + (f"-{backend_name}" if backend_name else ""),
            "run_args": {},
            "transpile_args": {},
            "tags": tags,
            "save_location": Path(save_location) if save_location else Path("./"),
            "serial": serial,
            "summoner_id": summoner_id,
            "summoner_name": summoner_name,
            "datetimes": datetimes,
        },
        outfields=outfields_inner,
    )
    current_exp.commons.datetimes.add_only("transform-from-pqp")
    current_exp.afterwards.counts.extend(counts)

    if target_circuit is not None:
        targets: list[tuple[WCKeyable, QuantumCircuit]] = [
            (target_circuit.name, target_circuit),
        ]

        current_exp.beforewards.target.extend(targets)
        cirqs, side_prodict = current_exp.method(
            targets=targets, arguments=current_exp.args, multiprocess=multiprocess_build
        )
        current_exp.side_products.update(side_prodict)

        circuit_qasm_strings, target_qasm_strings = make_qasm_strings(
            cirqs, targets, qasm_version, multiprocess=multiprocess_build
        )
        current_exp.beforewards.circuit_qasm.extend(circuit_qasm_strings)
        current_exp.beforewards.target_qasm.extend(target_qasm_strings)

    return current_exp
