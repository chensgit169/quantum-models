# Usage

At present ED is integrated into ``hamiltonian`` module, example:

```python
heisenberg = HeisenbergH(11)
energy, psi = heisenberg.compute_eigenstates(k=20, which='SA')
print(energy)
```

See [Exact Diagonalization](./Exact Diagonalization.md) for explanation about the principles.
