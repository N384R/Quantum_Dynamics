import itertools
from math import prod
import warnings
import numpy as np
from qsextra import ExcitonicSystem, ChromophoreSystem
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp, DensityMatrix
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.result.result import Result
from qiskit_aer import AerSimulator, AerError


def qevolve(system: ChromophoreSystem | ExcitonicSystem,
            time: float | list[float] | np.ndarray,
            shots: int | None = None,
            initialize_circuit: bool = True,
            Trotter_number: int = 1,
            Trotter_order: int = 1,
            dt: float = None,
            coll_rates: float | list[float] | None = None,
            GPU: bool = False,
            verbose: bool = True,
            ) -> QuantumCircuit | list[QuantumCircuit] | Result:
    verboseprint = print if verbose else lambda *a, **k: None

    if type(system) not in (ChromophoreSystem, ExcitonicSystem):
        raise ValueError(
            'The system must be an instance of ChromophoreSystem or ExcitonicSystem.')
    if not system._validity:
        raise ValueError(
            'The system is not valid.')
    if type(system) is ChromophoreSystem and system.mode_dict is None:
        warnings.warn(
            'The Pseudomodes are not specified. Executing the dynamics as an ExcitonicSystem.')
        return qevolve(system.extract_ExcitonicSystem(),
                       time,
                       shots,
                       initialize_circuit,
                       Trotter_number,
                       Trotter_order,
                       dt,
                       coll_rates,
                       GPU,
                       verbose,
                       )

    if type(system) is ExcitonicSystem:
        if not np.isscalar(coll_rates) and coll_rates is not None:
            raise ValueError(
                'coll_rates must be a scalar for ExcitonicSystem.')
    elif type(system) is ChromophoreSystem:
        if coll_rates is not None:
            coll_rates = np.atleast_1d(coll_rates)
            if len(coll_rates) != len(system.mode_dict['omega_mode']):
                raise ValueError(
                    'The number of collision rates is different from the number of pseudomodes per chromophore.')

    # Constructing the circuit
    qc, register_list = __create_circuit(system, coll_rates is not None)

    # Circuit initialization
    if initialize_circuit:
        qc = __circuit_init(qc, register_list, system)
        qc.barrier(label='Initialization')

    verboseprint('Start creating the circuits...')

    # Propagation
    qc_list = __propagation(qc,
                            register_list,
                            system,
                            time,
                            Trotter_number,
                            Trotter_order,
                            dt,
                            coll_rates,
                            shots,
                            )

    verboseprint('Circuits created...')

    # Return the circuit
    if shots is None:
        return qc_list
    # Execute the circuit
    if shots == 0:
        verboseprint('Extracting Density Matrix...')
        size = range(system.system_size)
        results = []
        for qc in qc_list:
            dm = DensityMatrix(qc)
            probability = dm.probabilities_dict(qargs=size)
            results.append(probability)
    else:
        verboseprint('Start measuring the circuits...')
        simulator = AerSimulator()
        if GPU:
            try:
                simulator.set_options(device='GPU')
            except AerError as e:
                print(e)
        results = simulator.run(qc_list, shots=shots).result()
    return results


def __create_circuit(system: ChromophoreSystem | ExcitonicSystem,
                     bool_ancillae: bool,
                     ):
    register_list = []
    # Exciton system qubits
    qr_e = QuantumRegister(system.system_size, name='sys_e')
    register_list.append(qr_e)
    # Pseudomode qubits
    if type(system) is ChromophoreSystem:
        qr_p_list = []
        for i in range(system.system_size):
            for k, d_k in enumerate(system.mode_dict['lvl_mode']):
                n_qubits = np.log2(d_k).astype(int)
                qr_p_list.append(
                    QuantumRegister(n_qubits, name=f'mode({i},{k})'))
        register_list += qr_p_list
    # Ancillae qubits for collision model
    if bool_ancillae:
        qr_a = AncillaRegister(1, name='ancillae')
        register_list.append(qr_a)
    qc = QuantumCircuit(*register_list)
    return qc, register_list


def __circuit_init(qc: QuantumCircuit,
                   register_list: list[QuantumRegister],
                   system: ChromophoreSystem | ExcitonicSystem,
                   ) -> QuantumCircuit:
    qr_e = register_list[0]
    if system.state_type == 'state':
        qc.initialize(system.state, qr_e)
    elif system.state_type == 'delocalized excitation':
        relevant_qubits = np.where(np.array(system.state) != 0)[0].tolist()
        relevant_state = np.zeros(2**len(relevant_qubits), dtype=complex)
        for ni, i in enumerate(relevant_qubits):
            relevant_state[2**ni] = system.state[i]
        qc_relevant = QuantumCircuit(len(relevant_qubits))
        qc_relevant.initialize(relevant_state)
        qc.compose(qc_relevant,
                   qubits=[qr_e[i] for i in relevant_qubits],
                   inplace=True,
                   )
    elif system.state_type == 'localized excitation':
        qc.x(qr_e[system.state])
    return qc


def __propagation(qc: QuantumCircuit,
                  register_list: list[QuantumRegister],
                  system: ChromophoreSystem | ExcitonicSystem,
                  time: float | list[float] | np.ndarray,
                  Trotter_number: int,
                  Trotter_order: int,
                  dt: float,
                  coll_rates: float | list[float] | None,
                  shots: int | None,
                  ):
    if dt is None:
        if np.isscalar(time):
            dt_Trotter = time / Trotter_number
        else:
            dt_Trotter = (time[1] - time[0]) / Trotter_number
        dt = dt_Trotter
        Trotter_number = 1
    else:
        dt_Trotter = dt / Trotter_number

    time = np.atleast_1d(time)
    tolerance = 1e-10
    if all(abs(t - np.round(t / dt) * dt) < tolerance for t in time):
        number_of_dts = np.round(time / dt).astype(int)
    else:
        raise ValueError(
            'The time steps must be multiples of the time step dt.')

    # Composing the circuit for a Trotter step
    qc_ts = __Trotter_step(system,
                           dt_Trotter,
                           Trotter_order,
                           coll_rates is not None,
                           )

    # Composing the circuit for the collisions
    if coll_rates is not None:
        qc_collision = __collision_circuit(system,
                                           dt,
                                           Trotter_order,
                                           coll_rates,
                                           )

    # Adding a classical register when needed
    if shots != 0 and shots is not None:
        creg = ClassicalRegister(system.system_size)
        qc.add_register(creg)

    # Propagation
    qc_list = []
    for ndt in range(np.max(number_of_dts) + 1):
        if ndt != 0:
            for _ in range(Trotter_number):
                qc.compose(qc_ts, inplace=True)
            qc.barrier(label=f'Trot step {ndt}')
            if coll_rates is not None:
                qc.compose(qc_collision, inplace=True)
                qc.barrier(label=f'Coll step {ndt}')
        if ndt in number_of_dts:
            qc_copy = qc.copy()
            if shots != 0 and shots is not None:
                qc_copy.measure(register_list[0], creg)
            qc_list.append(qc_copy)
    return qc_list


def __Trotter_step(system: ChromophoreSystem | ExcitonicSystem,
                   dt: float,
                   Trotter_order: int,
                   bool_ancillae: bool,
                   ) -> QuantumCircuit:
    qc, register_list = __create_circuit(system, bool_ancillae)
    qr_e = register_list[0]
    qc.compose(__sys_evolution(system, dt, Trotter_order),
               qubits=qr_e,
               inplace=True,
               )
    qc.barrier(label='sys evol')
    if type(system) is ChromophoreSystem:
        N = system.system_size
        W = len(system.mode_dict['omega_mode'])
        qr_p_list = register_list[1:-1] if bool_ancillae else register_list[1:]
        for k in range(W):
            _, _, a_p_a_dag_dict, a_dag_a_dict = __mode_operators(system, k)
            n_qubits_mode_k = len(list(a_dag_a_dict.keys())[0])

            qc_mode_evo_k = __mode_evolution(system.mode_dict['omega_mode'][k],
                                             a_dag_a_dict,
                                             dt,
                                             Trotter_order,
                                             )

            qc_system_int_k = __sys_mode_interaction(system.mode_dict['coupl_ep'][k],
                                                     a_p_a_dag_dict,
                                                     dt,
                                                     Trotter_order,
                                                     )
            for i in range(N):
                qr_p_ik = qr_p_list[i*W + k]
                qc.compose(qc_mode_evo_k, qubits=qr_p_ik, inplace=True)
                qubit_list = [qr_e[i]] + qr_p_ik[:]
                qc.compose(qc_system_int_k, qubits=qubit_list, inplace=True)
                qc.barrier(label=f'sys_{i}-mode_{k} int')
    return qc


def __sys_evolution(system: ChromophoreSystem | ExcitonicSystem,
                    dt: float,
                    Trotter_order: int,
                    ) -> QuantumCircuit:
    energies = system.e_el
    couplings = system.coupl_el
    N = system.system_size
    qc = QuantumCircuit(N)

    # Create the dictionary of Pauli operations
    op_dict = {}
    for i in range(N):
        if energies[i] != 0.:
            op_str = ''.join([*['I']*(N-i-1), 'Z', *['I']*i])
            op_dict[op_str] = - energies[i] / 2
    for i in range(N):
        for j in range(i+1, N):
            if couplings[i, j] != 0.:
                for pauli in ['X', 'Y']:
                    op_str = ''.join(
                        [*['I']*(N-j-1), pauli, *['I']*(j-i-1), pauli, *['I']*i])
                    op_dict[op_str] = couplings[i, j] / 2

    # Selecting the Trotter class
    Trotter = __Trotter_class(Trotter_order)

    # Creating the gates
    pauli_gates = SparsePauliOp(list(op_dict.keys()),
                                coeffs=list(op_dict.values()),
                                )
    evolution_gate = PauliEvolutionGate(pauli_gates,
                                        time=dt,
                                        synthesis=Trotter,
                                        )

    # Appending to qc
    qc.append(evolution_gate, range(N))
    return qc.decompose(reps=2)


def __Trotter_class(Trotter_order: int):
    if Trotter_order == 1:
        return LieTrotter()
    else:
        return SuzukiTrotter(order=Trotter_order)


def __mode_operators(system: ChromophoreSystem,
                     k: int,
                     ):
    def add_to_dict(target_dict: dict,
                    Pauli_ops: list[list[str]],
                    Pauli_coeffs: list[list[complex]],
                    mul: float,
                    ):
        Pauli_strings = [''.join(p) for p in itertools.product(*Pauli_ops)]
        Pauli_coeffs_comb = [mul * prod(c)
                             for c in itertools.product(*Pauli_coeffs)]
        for i, p in enumerate(Pauli_strings):
            target_dict[p] = target_dict.get(p, 0) + Pauli_coeffs_comb[i]
        return target_dict

    d_k = system.mode_dict['lvl_mode'][k]
    n_qubits = np.log2(d_k).astype(int)

    # Constructing a list of Gray codes for the pseudomode
    pseudo_encoded = gray_code_list(n_qubits)

    # a
    a_dict = {}
    for index in range(1, d_k):
        Pauli_ops = []
        Pauli_coeffs = []
        for i in range(n_qubits):
            if pseudo_encoded[index - 1][i] == pseudo_encoded[index][i]:
                Pauli_ops.append(['I', 'Z'])
                Pauli_coeffs.append(
                    [1/2, 1/2] if pseudo_encoded[index-1][i] == '0' else [1/2, -1/2])
            else:
                Pauli_ops.append(['X', 'Y'])
                Pauli_coeffs.append(
                    [1/2, 1.j/2] if pseudo_encoded[index-1][i] == '0' else [1/2, -1.j/2])
        a_dict = add_to_dict(
            a_dict, Pauli_ops, Pauli_coeffs, np.sqrt(index))

    # a^dagger
    a_dag_dict = {}
    for index in range(d_k - 1):
        Pauli_ops = []
        Pauli_coeffs = []
        for i in range(n_qubits):
            if pseudo_encoded[index + 1][i] == pseudo_encoded[index][i]:
                Pauli_ops.append(['I', 'Z'])
                Pauli_coeffs.append(
                    [1/2, 1/2] if pseudo_encoded[index+1][i] == '0' else [1/2, -1/2])
            else:
                Pauli_ops.append(['X', 'Y'])
                Pauli_coeffs.append(
                    [1/2, 1.j/2] if pseudo_encoded[index+1][i] == '0' else [1/2, -1.j/2])
        a_dag_dict = add_to_dict(
            a_dag_dict, Pauli_ops, Pauli_coeffs, np.sqrt(index+1))

    # a + a^dagger
    a_p_a_dag_dict = {}
    for (key, value_a), value_a_dag in zip(a_dict.items(), a_dag_dict.values()):
        if value_a != -value_a_dag:
            a_p_a_dag_dict[key] = value_a + value_a_dag

    # a^dagger * a
    a_dag_a_dict = {}
    for index in range(d_k):
        Pauli_ops = []
        Pauli_coeffs = []
        for i in range(n_qubits):
            Pauli_ops.append(['I', 'Z'])
            Pauli_coeffs.append(
                [1/2, 1/2] if pseudo_encoded[index][i] == '0' else [1/2, -1/2])
        a_dag_a_dict = add_to_dict(
            a_dag_a_dict, Pauli_ops, Pauli_coeffs, index)

    return a_dict, a_dag_dict, a_p_a_dag_dict, a_dag_a_dict


def __mode_evolution(frequency_mode: float,
                     a_dag_a_dict: dict,
                     dt: float,
                     Trotter_order: int,
                     ) -> QuantumCircuit:
    n_qubits = len(list(a_dag_a_dict.keys())[0])

    qr_p_k = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr_p_k)

    if frequency_mode != 0:
        Trotter = __Trotter_class(Trotter_order)

        coeffs = frequency_mode * np.array(list(a_dag_a_dict.values()))
        pauli_gates = SparsePauliOp(list(a_dag_a_dict.keys()),
                                    coeffs=coeffs,
                                    )
        evolution_gate = PauliEvolutionGate(pauli_gates,
                                            time=dt,
                                            synthesis=Trotter,
                                            )
        qc.append(evolution_gate, qr_p_k)
    return qc.decompose(reps=2)


def __sys_mode_interaction(interaction_strength: ChromophoreSystem,
                           a_p_a_dag_dict: dict,
                           dt: float,
                           Trotter_order: int,
                           ) -> QuantumCircuit:
    qr_e = QuantumRegister(1)
    n_qubits = len(list(a_p_a_dag_dict.keys())[0])

    qr_p_k = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr_e, qr_p_k)

    I = SparsePauliOp('I')
    Z = SparsePauliOp('Z')

    if interaction_strength != 0:
        Trotter = __Trotter_class(Trotter_order)

        coeffs = np.array(list(a_p_a_dag_dict.values()))
        a_p_a_dag_gates = SparsePauliOp(list(a_p_a_dag_dict.keys()),
                                        coeffs=coeffs,
                                        )

        H_coll_gate = a_p_a_dag_gates ^ ((0.5 * I) - (0.5 * Z))
        H_coll_gate = interaction_strength * H_coll_gate.simplify()
        evolution_gate = PauliEvolutionGate(H_coll_gate,
                                            time=dt,
                                            synthesis=Trotter,
                                            )
        qubit_list = [qr_e[0]] + qr_p_k[:]
        qc.append(evolution_gate, qubit_list)
    return qc.decompose(reps=2)


def __collision_circuit(system: ChromophoreSystem | ExcitonicSystem,
                        dt: float,
                        Trotter_order: int,
                        coll_rates: float | list[float],
                        ) -> QuantumCircuit:
    if type(system) is ExcitonicSystem:
        return __collisions_exsys(system, dt, coll_rates)
    elif type(system) is ChromophoreSystem:
        return __collisions_cpsys(system, dt, Trotter_order, coll_rates)


def __collisions_exsys(system: ExcitonicSystem,
                       dt: float,
                       coll_rates: float,
                       ) -> QuantumCircuit:
    qc, register_list = __create_circuit(system, coll_rates is not None)
    qr_e = register_list[0]
    qr_a = register_list[-1]
    for i in range(system.system_size):
        qc.rzx(2 * np.sqrt(coll_rates) * np.sqrt(dt), qr_e[i], qr_a[0])
        qc.reset(qr_a[0])
        qc.barrier(label=f'Collision {i}')
    return qc


def __collisions_cpsys(system: ChromophoreSystem,
                       dt: float,
                       Trotter_order: int,
                       coll_rates: list[float],
                       ) -> QuantumCircuit:
    N = system.system_size
    W = len(system.mode_dict['omega_mode'])
    qc, register_list = __create_circuit(system, coll_rates is not None)
    qr_p_list = register_list[1:-1]
    qr_a = register_list[-1]

    Trotter = __Trotter_class(Trotter_order)

    X = SparsePauliOp('X')
    Y = SparsePauliOp('Y')

    for k in range(W):
        if coll_rates[k] != 0:
            a_dict, a_dag_dict, _, _ = __mode_operators(system, k)

            # Creating the gates
            a_pauli_gates = SparsePauliOp(
                list(a_dict.keys()), coeffs=np.array(list(a_dict.values())))
            a_dag_pauli_gates = SparsePauliOp(
                list(a_dag_dict.keys()), coeffs=np.array(list(a_dag_dict.values())))
            H_coll_gate = ((a_pauli_gates ^ ((0.5 * X) - (0.5j * Y))) +
                           (a_dag_pauli_gates ^ ((0.5 * X) + (0.5j * Y))))
            # Sum repeated Pauli strings and remove zeros
            H_coll_gate = np.sqrt(coll_rates[k] / dt) * H_coll_gate.simplify()
            evolution_gate = PauliEvolutionGate(H_coll_gate,
                                                time=dt,
                                                synthesis=Trotter)

            for i in range(N):
                qr_p_ik = qr_p_list[i*W + k]
                qubit_list = [qr_a[0]] + qr_p_ik[:]
                qc.append(evolution_gate, qubit_list)
                qc.reset(qr_a[0])
                qc.barrier(label=f'Collision {i}')
    return qc.decompose(reps=2)


def gray_code_list(n: int) -> list[str]:
    return [f'{i ^ i >> 1:0{n}b}' for i in range(1 << n)]
