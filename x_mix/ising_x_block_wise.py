import cirq
import numpy as np
import sympy

def blockwise_optimize_qaoa(
    p_max, block_size,
    J, h,
    steps_per_block=50,
    eta=1e-2,
    seed=0,
    tol=1e-3
):
    rng = np.random.default_rng(seed)
    params = None

    for p_local in range(block_size, p_max + 1, block_size):
        print(f"\n=== Optimizing p = {p_local} ===")

        circuit, gamma_syms, beta_syms = build_qaoa_circuit(p_local, J, h)

        if params is None:
            params = rng.uniform(0, np.pi, size=2 * p_local)
        else:
            new_params = 0.01 * rng.standard_normal(2 * block_size)
            params = np.concatenate([params, new_params])

        prev_energy = None

        for step in range(steps_per_block):
            grad = gradient_energy(
                params, circuit, gamma_syms, beta_syms, J, h
            )
            params -= eta * grad

            energy = energy_from_params(
                params, circuit, gamma_syms, beta_syms, J, h
            )

            if step % 10 == 0:
                print(f"p={p_local}, step={step:03d}, E={energy:.6f}")

            if prev_energy is not None and abs(prev_energy - energy) < tol:
                break

            prev_energy = energy

    return params, circuit, gamma_syms, beta_syms

def measure_distribution(circuit, gamma_syms, beta_syms, params, shots=2000):
    measurement_circuit = circuit.copy()
    measurement_circuit.append(cirq.measure(*qubits, key="m"))
    sim = cirq.Simulator()
    resolver_dict = {}
    p_local = len(gamma_syms)
    for i in range(p_local):
        resolver_dict[gamma_syms[i]] = params[2 * i]
        resolver_dict[beta_syms[i]] = params[2 * i + 1]
    resolver = cirq.ParamResolver(resolver_dict)
    result = sim.run(measurement_circuit, param_resolver=resolver, repetitions=shots)
    hist = result.histogram(key="m")
    return hist


def energy_TFIM_from_wavefunction(wf, J, h):
    n = n_qubits
    probs = np.abs(wf) ** 2
    states = np.arange(2 ** n)
    bits = ((states[:, None] >> np.arange(n)[None, :]) & 1).astype(int)
    z_vals = 1 - 2 * bits

    # <Z_i Z_{i+1}>
    E_ZZ = sum(np.sum(probs * z_vals[:, i] * z_vals[:, i + 1]) for i in range(n - 1))

    E_ZZ += np.sum(probs * z_vals[:, -1] * z_vals[:, 0])
    # <X_i>
    E_X = 0.0
    for i in range(n):
        flip = 1 << i
        E_X += np.vdot(wf, wf[np.arange(len(wf)) ^ flip]).real

    return -J * E_ZZ - h * E_X

def energy_from_params(params, circuit, gamma_symbols, beta_symbols, J, h):
    sim = cirq.Simulator()
    resolver_dict = {}
    for i in range(len(gamma_symbols)):
        resolver_dict[gamma_symbols[i]] = params[2 * i]
        resolver_dict[beta_symbols[i]] = params[2 * i + 1]
    resolver = cirq.ParamResolver(resolver_dict)
    wf = sim.simulate(circuit, param_resolver=resolver).final_state_vector
    return energy_TFIM_from_wavefunction(wf, J, h)

def gradient_energy(params, circuit, gamma_symbols, beta_symbols, J, h, eps=1e-3):
    grad = np.zeros_like(params)
    for k in range(len(params)):
        shift = np.zeros_like(params)
        shift[k] = eps
        f_plus = energy_from_params(params + shift, circuit, gamma_symbols, beta_symbols, J, h)
        f_minus = energy_from_params(params - shift, circuit, gamma_symbols, beta_symbols, J, h)
        grad[k] = (f_plus - f_minus) / (2 * eps)
    return grad

def gamma_layer(gamma_value, J,h):
    for i in range(n_qubits - 1):
        yield cirq.ZZ(qubits[i], qubits[i + 1]) ** (2 * gamma_value * J / np.pi)
    yield cirq.ZZ(qubits[-1], qubits[0]) ** (2 * gamma_value * J / np.pi) 
    for j in range(n_qubits):
        yield cirq.X(qubits[j]) ** (2 * gamma_value * h / np.pi)

def beta_layer(beta_value):
    for i in range(n_qubits):
        yield cirq.X(qubits[i]) ** (2 * beta_value / np.pi)


def build_qaoa_circuit(p,J,h):
    gamma_symbols = [sympy.Symbol(f"γ_{i}") for i in range(p)]
    beta_symbols = [sympy.Symbol(f"β_{i}") for i in range(p)]
    circuit = cirq.Circuit(cirq.H.on_each(qubits))
    for i in range(p):
        circuit.append(gamma_layer(gamma_symbols[i], J,h))
        circuit.append(beta_layer(beta_symbols[i]))
    return circuit, gamma_symbols, beta_symbols


def optimize_qaoa_for_field(p, J, h, steps=80, eta=1e-2, seed=0, tol=1e-3):

    circuit, gamma_syms, beta_syms = build_qaoa_circuit(p,J,h)
    rng = np.random.default_rng(seed)
    params = rng.uniform(0, np.pi, size=2 * p)

    prev_energy = None

    for step in range(steps + 1):
        # Compute gradient and update
        grad = gradient_energy(params, circuit, gamma_syms, beta_syms, J, h)
        params -= eta * grad

        # Evaluate energy
        energy = energy_from_params(params, circuit, gamma_syms, beta_syms, J, h)

        # Display progress every 20 steps or at convergence
        if not step % 20:
            print(f"h={h:.3f}, Step {step:03d}: E_total = {energy:.6f}")

        # Check convergence
        if prev_energy is not None:
            delta_E = abs(prev_energy - energy)
            if delta_E <= tol:
                print(f"Converged at step {step}: ΔE = {delta_E:.6f}, E = {energy:.6f}")
                break
        prev_energy = energy

    return params, circuit, gamma_syms, beta_syms


if __name__=="__main__":
    n_qubits = 10
    J = 1.0
    h = 0.0
    p = 100
    steps = 1000
    eta = 1e-4
    shots = 2000
    qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]
    results, energies = {}, {}
    sim = cirq.Simulator()
    
    params, circuit, gamma_syms, beta_syms = blockwise_optimize_qaoa(
            p_max=p,
            block_size=25,
            J=J,
            h=h,
            steps_per_block=50,
            eta=eta,
            seed=42)

    hist = measure_distribution(circuit, gamma_syms, beta_syms, params, shots=shots)
    # final energy
    resolver =resolver = cirq.ParamResolver({
    **{gamma_syms[i]: params[2*i] for i in range(p)},
    **{beta_syms[i]: params[2*i+1] for i in range(p)} })

    wf = sim.simulate(circuit, param_resolver=resolver).final_state_vector
    E_final = energy_TFIM_from_wavefunction(wf, J, h)
    results[h] = (params, circuit, gamma_syms, beta_syms, hist, wf)

    print("\nMeasured bitstrings and probabilities:")
    for state_int, count in hist.most_common():
        bitstring = format(state_int, f"0{n_qubits}b")
        prob = count / shots
        print(f"{bitstring} : {prob:.4f}")

