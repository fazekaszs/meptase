# MePTASE

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
