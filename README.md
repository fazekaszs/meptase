# MePTASE

## Introduction 

MePTASE (_Metadynamics with PyTorch in the Atomic Simulation Environment_) is
a Python package created for metadynamics simulations in ASE.
Using metadynamics, which is a type of non-equilibrium molecular dynamics,
one can explore potential of mean force surfaces, resulting in the
free energy landscape along different reaction coordinates called 
_collective variables_ (or simply CVs).
This is done by placing bump-like functions (hereby; _kernels_) onto a
history-dependent potential energy surface, defined above the applied
CV space.
When the forces acting on atoms are calculated (using a custom chosen
ASE calculator, providing flexibility) additional bias forces are computed
from the negative gradient of the bias potential and these are added to the
calculator-provided (unbiased) forces.
Since I opted for high flexibility and simplicity with this package, the
bias potential and its negative gradient are calculated using PyTorch,
which provides convenient autograd functionalities.
I implemented some basic CVs (e.g. distances, angles, dihedral angles),
some basic kernels (e.g. Gaussian kernel, von Mises kernel, Beta kernel), 
and even some additional potentials (e.g. flat-bottomed harmonic walls).

## Installation

MePTASE can be installed from the PyPI repository using `pip`:

```shell
pip install meptase
```

## Usage from JSON Config Files

### IO Control

The `io_control` block controls molecule construction and the project output
directory. Every run needs exactly one block of this type, and the following
keys can be set within it:

| Key                             | Type                 | Restriction                                                                                                                         | Description                                                                                                                                                                                                                                     |
|---------------------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `output_dir`                    | `string`             | Required.                                                                                                                           | Path of the directory the simulation outputs are written to. It is created if it does not exist; if it does exist, `overwrite_output_dir` decides what happens.                                                                                 |
| `mol_from_smiles`               | `string`             | Optional. Cannot be combined with `mol_from_mol_file`; at least one of the two must be set.                                         | A valid SMILES string used to build the molecule. Hydrogen atoms are added and a 3D structure is generated via RDKit's ETKDGv3 embedding. Atoms marked with atom map numbers (`:1`, `:2`, ...) can be referenced in the CV and PEF definitions. |
| `mol_from_mol_file`             | `string`             | Optional. Cannot be combined with `mol_from_smiles`; at least one of the two must be set. Requires `mol_file_atom_map_num_indices`. | Path to an RDKit `.mol` file used to build the molecule. Since mol files do not carry atom map numbers, they are assigned from `mol_file_atom_map_num_indices` in order.                                                                        |
| `mol_file_atom_map_num_indices` | `array` of `integer` | Required if `mol_from_mol_file` is set. Entries must be unique.                                                                     | List of atom indices (0-based) to which the atom map numbers `1, 2, ..., N` are assigned in the given order, i.e. the first element of the list gets atom map number 1, the second one gets 2, etc.                                             |
| `mmff_optimize`                 | `boolean`            | Optional. Defaults to `false`.                                                                                                      | Whether the generated structure is geometry optimized with the MMFF force field before the run.                                                                                                                                                 |
| `overwrite_output_dir`          | `boolean`            | Optional. Defaults to `false`.                                                                                                      | Whether an already existing `output_dir` is deleted and recreated. If `false` and the directory exists, the run is aborted.                                                                                                                     |
| `etkdvg3_seed`                  | `integer`            | Optional. Defaults to `0`.                                                                                                          | Random seed used by the ETKDGv3 3D embedding. Setting it makes structure generation reproducible.                                                                                                                                               |

The key `mol_from_smiles` selects the marked atoms of the SMILES string to
define the CV atoms, while `mol_from_mol_file` selects them via the atom map
numbers assigned from `mol_file_atom_map_num_indices`.

An example JSON IO Control block:

```json
{
  "io_control": {
    "mol_from_smiles": "C[C:1](=O)[NH:2][C@@H:3](C)[C:4](=O)[NH:5]C",
    "mmff_optimize": true,
    "output_dir": "test_alanine",
    "overwrite_output_dir": true,
    "etkdvg3_seed": 1994
  }
}
```

### Unbiased Calculator

The `unbiased_calculator` block selects the ASE calculator that computes the
unbiased (intrinsic) energy and forces of the system. Its type is looked up in
a calculator registry, and its parameters are passed to the constructor of the
corresponding ASE calculator. The following keys can be set within it:

| Key          | Type     | Restriction                                     | Description                                                                                                                                |
|--------------|----------|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `type`       | `string` | Required. Must be a registered calculator name. | Identifier of the unbiased calculator. Currently the registry contains `tblite` (GFN-xTB) and `mace` (MACE-MP machine learning potential). |
| `parameters` | `object` | Optional.                                       | Keyword arguments forwarded to the ASE calculator constructor. Their allowed keys depend on the selected calculator type.                  |

For example, `tblite` accepts the standard `tblite.ase.TBLite` keyword
arguments such as `max_iterations`, `accuracy` and `verbosity`, while `mace`
accepts the `MACECalculator` keyword arguments (e.g. `model_path`).

An example JSON Unbiased Calculator block:

```json
{
  "unbiased_calculator": {
    "type": "tblite",
    "parameters": {
      "max_iterations": 1000,
      "accuracy": 1.0,
      "verbosity": 0
    }
  }
}
```

### CVs, Kernels, PEFs

The collective variables, bias kernels and additional potential energy
functions (PEFs) are each given as a JSON list. Every element of a list is an
object with a `type` key selecting a registered implementation and a
`parameters` object holding the constructor arguments. CVs and PEFs built from
atom indices always reference atoms by their SMILES atom map numbers (or the
map numbers assigned by `mol_file_atom_map_num_indices`).

#### Collective variables

| `type`     | `parameters`                                                               | Description                                                                                                 |
|------------|----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| `distance` | `name` (`string`), `indices` (`array` of `[i, j]` index pairs)             | Distance between the atom pairs given in `indices`. Each inner list contains two atom map numbers.          |
| `angle`    | `name` (`string`), `indices` (`array` of `[i, j, k]` index triplets)       | Angle centered at the middle atom `j` of each triplet. Each inner list contains three atom map numbers.     |
| `dihedral` | `name` (`string`), `indices` (`array` of `[i, j, k, l]` index quadruplets) | Dihedral angle formed by the four atoms of each quadruplet. Each inner list contains four atom map numbers. |

The `name` cannot contain the dot character (`.`), since the name is used in
`target_cvs` under the `<name>.<index>` scheme, where `<index>` is the 1-based
position of the CV component inside the collective variable.

#### Kernels

Kernels define the shape of the history-dependent bias (the hills). Every
collective variable component must be targeted by at least one kernel, which
is declared through the `target_cvs` list. One kernel can target several CV
components at once, in which case the same kernel instance is applied to all
of them.

| `type`      | `parameters`                                                          | Description                                                                                                                                         |
|-------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `gaussian`  | `width` (`number`)                                                    | Unbounded Gaussian kernel, recommended for distances. The `width` is the standard deviation.                                                        |
| `von_mises` | `width` (`number`), `period` (`number`, optional, defaults to `2*pi`) | Periodic kernel, recommended for angles and dihedrals. The `width` is the distance between the two extrema of the kernel derivative.                |
| `beta`      | `width` (`number`), `domain` (`number`, optional, defaults to `pi`)   | Bounded kernel on the interval `[0, domain]`, recommended for angles. The `width` is the distance between the two extrema of the kernel derivative. |

#### Additional potentials (PEFs)

PEFs are history-independent (in contrast to the bias potential) and are used
e.g. to keep the system away from unphysical regions of the CV space. They are
optional, so the list can be empty. Their `target_cvs` list declares which CV
components they act on, and the units of the potential energy are eV.

| `type`                        | `parameters`                                                                                              | Description                                                                                                    |
|-------------------------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `lower_harmonic_wall`         | `target_cvs` (`array` of `string`), `cv_min` (`number`), `force_constant` (`number`)                      | Harmonic wall that is only active below `cv_min`. `force_constant` is in eV / CV-unit^2.                       |
| `upper_harmonic_wall`         | `target_cvs` (`array` of `string`), `cv_max` (`number`), `force_constant` (`number`)                      | Harmonic wall that is only active above `cv_max`. `force_constant` is in eV / CV-unit^2.                       |
| `flat_bottomed_harmonic_wall` | `target_cvs` (`array` of `string`), `cv_min` (`number`), `cv_max` (`number`), `force_constant` (`number`) | Combination of the lower and upper walls: harmonic outside the interval `[cv_min, cv_max]` and flat within it. |

Example JSON CV, Kernel and PEF blocks:

```json
{
  "collective_variables": [
    {
      "type": "distance",
      "parameters": { "name": "d", "indices": [[1, 4]] }
    },
    {
      "type": "dihedral",
      "parameters": { "name": "theta", "indices": [[1, 2, 3, 4]] }
    }
  ],
  "kernels": [
    {
      "type": "gaussian",
      "target_cvs": ["d.1"],
      "parameters": { "width": 0.2 }
    },
    {
      "type": "von_mises",
      "target_cvs": ["theta.1"],
      "parameters": { "width": 0.15 }
    }
  ],
  "additional_potentials": [
    {
      "type": "flat_bottomed_harmonic_wall",
      "target_cvs": ["d.1"],
      "parameters": { "cv_min": 4.0, "cv_max": 6.0, "force_constant": 5.0 }
    }
  ]
}
```

### Run Control

The `run_control` block sets the simulation parameters of the metadynamics run
(Langevin dynamics with periodic hill deposition). Every run needs exactly one
block of this type, and all of its keys must be strictly positive. The
following keys can be set within it:

| Key                         | Type      | Restriction                 | Description                                                                                                                    |
|-----------------------------|-----------|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `temperature`               | `number`  | Required, must be positive. | Simulation temperature in K. Used both for the Maxwell-Boltzmann initial velocity distribution and by the Langevin thermostat. |
| `timestep`                  | `number`  | Required, must be positive. | Integration time step of the Langevin dynamics in fs.                                                                          |
| `friction`                  | `number`  | Required, must be positive. | Friction coefficient of the Langevin thermostat (in the ASE convention, i.e. inverse time in the unit of the time step).       |
| `kernel_height`             | `number`  | Required, must be positive. | Height of the deposited kernels (hills) in eV.                                                                                 |
| `steps_between_hills`       | `integer` | Required, must be positive. | Number of dynamics steps between two consecutive hill depositions.                                                             |
| `n_hills`                   | `integer` | Required, must be positive. | Total number of hills deposited during the run. The simulation ends after `n_hills * steps_between_hills` dynamics steps.      |
| `trajectory_write_interval` | `integer` | Required, must be positive. | Number of dynamics steps between two consecutive writes to the trajectory file.                                                |

An example JSON Run Control block:

```json
{
  "run_control": {
    "temperature": 310.0,
    "timestep": 0.5,
    "friction": 0.2,
    "steps_between_hills": 50,
    "n_hills": 1000,
    "trajectory_write_interval": 100,
    "kernel_height": 0.1
  }
}
```

## Usage from Python

MePTASE can also be used directly from Python, giving full control over every
stage of the simulation. Instead of describing the run with a JSON file, the
user composes the collective variables, kernels, additional potentials and the
unbiased calculator by hand, wraps them into a metadynamics engine and a
calculator, and then drives the dynamics with an ASE integrator of choice
(e.g. Langevin dynamics).

The components of a run are built from the following classes:

- Collective variables: `DistanceCV`, `AngleCV` and `DihedralCV`, each derived
  from `CVBase`. Multiple CVs can be merged into a single mapper with
  `MergeCV`.
- Kernels: `GaussianKernel`, `VonMisesKernel` and `BetaKernel`, each derived
  from `KernelBase`.
- Additional potentials: `LowerHarmonicWall`, `UpperHarmonicWall` and
  `FlatBottomedHarmonic`, each derived from `PotentialEnergyFunction`. Multiple
  potentials can be merged with `MergedPEF`.
- The unbiased calculator is any ASE calculator (e.g. `tblite.ase.TBLite` or a
  MACE calculator).

These are combined into a `MetaDynamicsEngine`, which manages the CV
calculation, the bias potential and the hill deposition, and a
`MetaDynamicsCalculator` that adds the bias forces to the unbiased ones
calculated by the ASE calculator. Note, that in contrast to the JSON config
files, the atom indices are given directly as zero-based indices of the ASE
Atoms object, so no atom map numbers are needed.

The Python API gives one more degree of flexibility over the JSON config
files: the user is not limited to the built-in implementations. Since the
engine works with any differentiable PyTorch callable, custom collective
variables, kernels and additional potentials can be implemented by deriving
from the respective base classes (`CVBase`, `KernelBase` and
`PotentialEnergyFunction`) and overriding their abstract `run` method. These
custom classes can be used seamlessly alongside the built-in ones, without
touching the engine or the calculator.

A minimal example of running a metadynamics simulation:

```python
import torch
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from tblite.ase import TBLite

from meptase.collective_variables import AngleCV
from meptase.additional_potentials import FlatBottomedHarmonic
from meptase.kernels import BetaKernel
from meptase.metadynamics import MetaDynamicsEngine, MetaDynamicsCalculator

# 1. Build the ASE Atoms object (e.g. from an RDKit conformer).
ase_mol = ...  # ase.Atoms with the initial structure

# 2. Choose the unbiased ASE calculator.
unbiased_calculator = TBLite(max_iterations=1000, accuracy=1.0, verbosity=0)

# 3. Define the collective variable (0-based atom indices), an additional
#    potential, and the kernel used for the bias.
cv = AngleCV(
    name="theta",
    indices=torch.tensor([[0, 1, 2]], dtype=torch.int)
)
wall = FlatBottomedHarmonic(
    indices=torch.tensor([0], dtype=torch.int),
    cv_min=10. * torch.pi / 180.,
    cv_max=170. * torch.pi / 180.,
    force_constant=500.  # eV / rad^2
)
kernel = BetaKernel(width=5. * torch.pi / 180.)

# 4. Combine everything into a metadynamics engine and calculator.
engine = MetaDynamicsEngine(
    mapper=cv,
    additional_potential=wall,
    kernels=[kernel, ],
    kernel_indices=torch.tensor([0, ], dtype=torch.int),
    kernel_height=0.05  # eV
)
ase_mol.calc = MetaDynamicsCalculator(
    unbiased_calculator=unbiased_calculator,
    engine=engine
)

# 5. Initialize the velocities and set up the dynamics. A hill is deposited
#    every 50 steps via the calculator.
MaxwellBoltzmannDistribution(ase_mol, temperature_K=310)
trajectory = Trajectory("output.traj", "w", ase_mol)
dynamics = Langevin(ase_mol, 0.5 * units.fs, temperature_K=310, friction=0.2)
dynamics.attach(trajectory, interval=100)

for _ in range(4000):
    dynamics.run(50)
    ase_mol.calc.deposit_hill()

# 6. Analyze the results: the deposited CV history and the reconstructed FES.
cv_history = engine.cv_history
fes_domain, fes_values = ase_mol.calc.get_fes()
```

The example above deposits a hill at fixed height at every step. For well
tempered metadynamics, one can pass a `well_tempered_temperature` to the
`MetaDynamicsEngine`, in which case the deposited hills get exponentially
down-weighted based on the already accumulated bias potential.

## Implementing Custom Parts

### Custom CV Implementation

... TODO

### Custom Kernel Implementation

... TODO

### Custom PEF Implementation

... TODO

## Run Reproducibility

### Validation Using Published Data

... TODO

### Similarity of Parallel Runs

... TODO

## Highlights

... TODO (example runs, screenshots)
