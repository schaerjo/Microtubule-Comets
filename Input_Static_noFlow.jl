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

df_file = "DF_Static_noFlow"
dir_df = @__DIR__
df = CSV.read(joinpath(dir_df,df_file*".csv"), DataFrame)[idx,:]

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
#println("Ndtd = ", Ndtd)

#Physical quantities.
dx_phy = Tf(df[:dx])
dt_phy = Tf(df[:dt])

R = (Tf(df[:R])/dx_phy)

#Wetting.
h = Tf(df[:h])
a = dx*h

#Domain limits.
xl = Ti(1)
xr = Ti(Nx)
yb = Ti(1)
yt = Ti(Nr)

#Phase field
ϕa = Tf(df[:phia])
ϕb = Tf(df[:phib])

D = Tf(df[:D])
σst = Tf(df[:surftens])
M = df[:MS]/σst

k = 1.5*D*σst/((ϕb-ϕa)*(ϕb-ϕa))
β = 12*σst/(D*(ϕb-ϕa)*(ϕb-ϕa)*(ϕb-ϕa)*(ϕb-ϕa))

#Homogeneous init
ϕ0 = Tf(df[:phi0])

#Film init
#thickness = Tf(df[:thickness])

NΔt = Tf(df[:Nt])
prin = Tf(df[:prin])
prin0 = Tf(df[:prin0])
