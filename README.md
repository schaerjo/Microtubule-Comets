# Microtubule-Wetting
Code about the microtubule wetting project.

We try to simulate the wetting of microtubule by proteins such as EBs and CLIP-170 proteins.
We use a phase field approach.
The phase separation is introduced via a double well potential (different potentials are tested, but we settled
on a simple (phi-a)²(phi-b)² type of potential).
The wetting is introduced via a surface energy h*phi.

We are simulating different situations considering cylindrical coordinates and assuming axisymmetry:
  -an infinite MT by using periodic boundary conditions along the x-axis. (Simulation_Static_...)
  -a growing MT with the simulation frame moving with the growing tip. (Simulation_Growing_...)
  -For both cases above, we are simullating in the absence (noFlow) and presence (withFlows) of hydrodynamic flows.
  -We also introduce a chemical reaction term to induce droplet evaporation. (withEvaporation)

The dynamics of the phase field are governed by a Cahn-Hilliard equation while the dynamics of the hydrodynamic flows are governed by the Navier-Stokes equation.

We solve the Cahn-Hilliard equation with an upwind finite difference scheme and the Navier-Stokes equation with the Lattice Boltzmann Method.

Julia version: 1.8.5
Packages version:
  julia> Pkg.status()
  Status `~/.julia/environments/v1.11/Project.toml`
  ⌃ [336ed68f] CSV v0.10.14
  ⌃ [13f3f980] CairoMakie v0.12.13
  ⌅ [5ae59095] Colors v0.12.11
    [a93c6f00] DataFrames v1.7.0
  ⌃ [5789e2e9] FileIO v1.16.3
    [53c48c17] FixedPointNumbers v0.8.5
  ⌃ [6218d12a] ImageMagick v1.3.1
    [41ab1584] InvertedIndices v1.3.0
    [4138dd39] JLD v0.13.5
    [682c06a0] JSON v0.21.4
    [2fda8390] LsqFit v0.15.0
  ⌅ [ee78f7c6] Makie v0.21.13
    [2774e3e8] NLsolve v4.5.1
    [cbe49d4c] RemoteFiles v0.5.0
    [276daf66] SpecialFunctions v2.4.0
    [9bd350c2] OpenSSH_jll v9.9.1+1
  Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`
