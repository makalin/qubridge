# Performance Notes

## Simulator Cost

- Quantum simulation cost grows exponentially with qubit count.
- Keep `n_qubits` in the 2-8 range for fast iteration on CPUs.

## Recommended Defaults

- `n_qubits=4`
- `n_q_layers=2`
- `batch_size=16-64`

These settings provide reasonable runtime while preserving hybrid behavior.

## Hardware Guidance

- Classical layers can benefit from GPU, but the default PennyLane simulator is typically CPU-bound.
- For local development, prioritize faster CPU cores and moderate batch sizes.

## Practical Tuning

1. Reduce `n_qubits` first when runtime is high.
2. Reduce `n_q_layers` second.
3. Increase batch size only if memory allows and epoch time remains acceptable.
4. Use TensorBoard to monitor convergence before increasing model complexity.
