# Microtubule-Wetting
Code about the microtubule wetting project.

We try to simulate the wetting of microtubule by proteins such as EBs and CLIP-170 proteins.
We use a phase field approach.
The phase separation is introduced via a double well potential (different potentials are tested, but we settled
on a simple (phi-a)²(phi-b)² type of potential).
The wetting is introduced via a surface energy h*phi.

We are simulating different situations considering cylindrical coordinates and assuming axisymmetry:
  -an infinite MT by using periodic boundary conditions along the x-axis.
  -a growing MT with the simulation frame moving with the growing tip.
  -For both cases above, we are simullating in the absence and presence of hydrodynamic flows.
  -We also introduce a chemical reaction term to induce droplet evaporation.

The dynamics of the phase field are governed by a Cahn-Hilliard equation while the dynamics of the hydrodynamic flows are governed by the Navier-Stokes equation.

We solve the Cahn-Hilliard equation with an upwind finite difference scheme and the Navier-Stokes equation with the Lattice Boltzmann Method.
