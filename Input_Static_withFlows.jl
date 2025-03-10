dir = "/path/to/data/serie/" 
idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
@show fn = "$idx/" 
file = joinpath(dir, fn) 
mkpath(file) 
println("path_c = ", dir)
println(idx) 
mkpath(dir)

include("/path/to/using.jl")
using_mod(".JulUtils")
using_pkg("CUDA, JLD, Statistics, Printf, LinearAlgebra, Random, DelimitedFiles, CSV, DataFrames, Dates")

Tf = Float64
Ti = Int64
Ta = CuArray

df_file = "DF_Static_withFlows"
dir_df = @__DIR__
df = CSV.read(joinpath(dir_df,df_file*".csv"), DataFrame)[idx,:]

#LBM
#D = 2
Qlbm = 9

# Weigth
w = Ta([4/9 1/9 1/9 1/9 1/9 1/36 1/36 1/36 1/36])
# Lattice Velocities
ξ = Ta([0 0; 1 0; 0 1; -1 0; 0 -1; 1 1; -1 1; -1 -1; 1 -1])

#GPU
Lx = Ti(df[:Lx])
Lr = Ti(df[:Lr]) 
a2D_block = (Lx, Lr)
Bx = Ti(df[:Bx])
Br = Ti(df[:Br])
a2D_grid = (Bx, Br)

Nx = Lx*Bx
Nr = Lr*Br

#Steps
dx = Tf(1.0)
dt = Tf(1.0)
dtd = Tf(df[:dtd])
Ndtd = Ti(dt/dtd)

#Physical quantities.
dx_phy = Tf(df[:dx])
dt_phy = Tf(df[:dt])

R = (Tf(df[:R])/dx_phy)

#LBM
ρ0 = Tf(df[:rho])
τ=Tf(df[:tau])
ω=1/τ
nu=(2*τ-1)/6

#Phase field
ϕa = Tf(df[:phia])
ϕb = Tf(df[:phib])

D = Tf(df[:D])
σst = Tf(df[:surftens])
M = df[:MS]/σst

k = 1.5*D*σst/((ϕb-ϕa)*(ϕb-ϕa))
β = 12*σst/(D*(ϕb-ϕa)*(ϕb-ϕa)*(ϕb-ϕa)*(ϕb-ϕa))

#Wetting.
h = Tf(df[:h])
a = dx*h

#Homogeneous init
ϕ0 = Tf(df[:phi0])

#Film init.
#thickness = Tf(df[:thickness])

NΔt = Tf(df[:Nt])
prin = Tf(df[:prin])
prin0 = 0
