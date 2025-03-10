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

The .csv files provide examples of parameter sets used in our simulations.

CUDA version:

  CUDA runtime 12.3, artifact installation
  CUDA driver 12.8
  NVIDIA driver 570.86.15
  
  CUDA libraries:
  - CUBLAS: 12.3.4
  - CURAND: 10.3.4
  - CUFFT: 11.0.12
  - CUSOLVER: 11.5.4
  - CUSPARSE: 12.2.0
  - CUPTI: 21.0.0
  - NVML: 12.0.0+570.86.15
  
  Julia packages:
  - CUDA.jl: 5.2.0
  - CUDA_Driver_jll: 0.7.0+1
  - CUDA_Runtime_jll: 0.11.1+0
  - CUDA_Runtime_Discovery: 0.2.3
  
  Toolchain:
  - Julia: 1.8.5
  - LLVM: 13.0.1
  
  1 device:
    0: Tesla P100-PCIE-12GB (sm_60, 11.899 GiB / 12.000 GiB available)

Package version:
  Status `~/.julia/environments/v1.8/Project.toml`
  ⌃ [336ed68f] CSV v0.10.12
  ⌃ [052768ef] CUDA v5.2.0
  ⌃ [a93c6f00] DataFrames v1.6.1
  ⌃ [31c24e10] Distributions v0.25.107
  ⌃ [7a1cc6ca] FFTW v1.8.0
  ⌃ [5789e2e9] FileIO v1.16.2
  ⌃ [4138dd39] JLD v0.13.4
    [682c06a0] JSON v0.21.4
  ⌃ [ee78f7c6] Makie v0.20.7
    [ade2ca70] Dates
    [8bb1440f] DelimitedFiles
    [37e2e46d] LinearAlgebra
    [d6f4376e] Markdown
    [de0858da] Printf
    [9a3f8284] Random
    [10745b16] Statistics
  Info Packages marked with ⌃ have new versions available and may be upgradable.
