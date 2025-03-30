from itertools import product
from time import perf_counter
from typing import Dict

import jax
import pandas as pd
import os.path as osp
from pyscf import dft, scf

from mess import Hamiltonian, basisset, minimise, molecule
from mess.interop import to_pyscf


def benchmark_mess(
    mol_name: str, basis_name: str, xc_method: str, num_runs: int = 10
) -> Dict[str, float]:
    """Run a benchmark calculation with MESS, measuring compilation and execution time.

    Args:
        mol_name: Name of the molecule to benchmark
        basis_name: Name of the basis set to use
        xc_method: Exchange-correlation method
        num_runs: Number of runs to average over (default: 10)

    Returns:
        Dictionary containing timing results and benchmark parameters
    """
    mol = molecule(mol_name)
    basis = basisset(mol, basis_name)
    H = Hamiltonian(basis, xc_method=xc_method)
    tic = perf_counter()
    H = jax.device_put(H)
    transfer_time = perf_counter() - tic

    tic = perf_counter()
    minimise.lower(H).compile()
    compilation_time = perf_counter() - tic

    tic = perf_counter()

    for _ in range(num_runs):
        minimise(H)[0].block_until_ready()

    avg_time = (perf_counter() - tic) / num_runs

    return {
        "transfer_time_s": transfer_time,
        "compilation_time_s": compilation_time,
        "average_runtime_s": avg_time,
        "num_runs": num_runs,
        "device": f"MESS ({jax.devices()[0].device_kind})",
        "basis": basis_name,
        "structure": mol_name,
        "method": xc_method,
    }


def benchmark_pyscf(
    mol_name: str, basis_name: str, xc_method: str, num_runs: int = 10
) -> Dict[str, float]:
    """Run a benchmark calculation with PySCF, measuring execution time.

    Args:
        mol_name: Name of the molecule to benchmark
        basis_name: Name of the basis set to use
        xc_method: Exchange-correlation method
        num_runs: Number of runs to average over (default: 10)

    Returns:
        Dictionary containing timing results and benchmark parameters
    """
    mol = molecule(mol_name)
    scf_mol = to_pyscf(mol, basis_name=basis_name)

    solver = dft.RKS(scf_mol, xc=xc_method) if xc_method != "hfx" else scf.RHF(scf_mol)

    tic = perf_counter()
    for _ in range(num_runs):
        solver.kernel()

    avg_time = (perf_counter() - tic) / num_runs

    return {
        "transfer_time_s": 0,
        "compilation_time_s": 0,
        "average_runtime_s": avg_time,
        "num_runs": num_runs,
        "device": "PySCF (cpu)",
        "basis": basis_name,
        "structure": mol_name,
        "method": xc_method,
    }


def generate_filename(
    directory: str, base_name: str = "results", ext: str = ".parquet"
) -> str:
    """Generate a unique benchmark filename by appending an incrementing number."""

    counter = 0

    while True:
        new_path = osp.join(directory, f"{base_name}_{counter:03d}{ext}")

        if not osp.exists(new_path):
            return new_path

        counter += 1


if __name__ == "__main__":
    num_runs = 10
    basis_names = ["sto-3g", "6-31g"]
    structure = ["h2", "h2o", "ch4", "c6h6"]
    methods = ["hfx", "pbe"]
    results = []

    for basis, mol, xc in product(basis_names, structure, methods):
        results.append(benchmark_mess(mol, basis, xc, num_runs))
        results.append(benchmark_pyscf(mol, basis, xc, num_runs))

    df = pd.DataFrame(results)
    output_path = generate_filename(osp.dirname(osp.abspath(__file__)))
    df.to_parquet(output_path, index=False)
    print(df)
