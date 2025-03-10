dir = "/path/to/data/serie/" 
idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) # job ID
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

df_file = "DF_Growing_noFlow"
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
v_phy = Tf(df[:v])

R = (Tf(df[:R])/dx_phy)

#Wetting.
h = Tf(df[:h])
a = dx*h
h0 = Tf(df[:h0])
a0 = dx*h0
htip = Tf(df[:htip])
atip = dx*htip
lzone = Ti(df[:lzone])
tail = Ti(df[:tail])
lzone0 = Ti(df[:lzone0])

#Domain limits.
xl = Ti(1)
xr = Ti(Nx)
yb = Ti(1)
yt = Ti(Nr)
xm0 = Ti(df[:xm0])
ym = Ti(R)

#MT growth.
k0 = v_phy*1000*dt_phy/(60*dx_phy)

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

#Displacement.
xlim = Ti(df[:xlim])
xdel = Ti(df[:xdel])
Block_Inter = Ti(ceil((xdel)/256))

grav = Tf(0.0)

NΔt = Tf(df[:Nt])
prin = Tf(df[:prin])
prin0 = Tf(df[:prin0])
