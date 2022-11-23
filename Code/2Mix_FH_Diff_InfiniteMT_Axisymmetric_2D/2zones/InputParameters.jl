include("C:/Users/joels/OneDrive/Documents/GitHub/Utilities/julia_utilities.jl")
# include("../../../../Utilities/julia_utilities.jl")
usingpkg("CUDA, JLD, Statistics, Printf, LinearAlgebra, Random, DelimitedFiles, CSV, DataFrames, Dates")

# idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

# dir = "/home/users/s/schaerjo/scratch/Wetting/2Mix_FH_Diff_InfiniteMT_Axisymmetric_2D/1zone/Serie0/" 
dir = "C:/Users/joels/OneDrive/Documents/Work/Wetting/2Mix_FH_Diff_InfiniteMT_Axisymmetric_2D/2zones/Local/Sim_3/"
# @show fn = "$idx/"
# file = joinpath(dir, fn)
#path to save Data on computer
localpath = "C:/Users/joels/OneDrive/Documents/Work/Wetting/2Mix_FH_Diff_InfiniteMT_Axisymmetric_2D/2zones/Local/Sim_3/"
# mkpath(file)

# dir_df = @__DIR__
# df = CSV.read(joinpath(dir_df,"Serie0.csv"), DataFrame)[idx,:]

Tf = Float64

dx = 1.0
dt = Tf(1e-3)

WrapsX = 32
WrapsZ = 8
Bx = 50
Bz = 50
block_dim = (WrapsX, WrapsZ)
grid_dim = (Bx, Bz)

Nx = WrapsX * Bx
Nz = WrapsZ * Bz

M = Tf(1.0)
a = Tf(1.0)#e-1)
χ = Tf(3.0)
k = Tf(5.0)
hmax = Tf(0.1)
hmin = Tf(0.0)
L = Tf(400)
ϕ0 = Tf(0.20)
R = Tf(13)
# M = Tf(df[:M])
# a = Tf(df[:a])
# χ = Tf(df[:chi])
# k = Tf(df[:k])
# hmax = Tf(df[:hmax])
# hmin = Tf(df[:hmin])
# L = Tf(df[:L])
# ϕ0 = Tf(df[:phi0])
# R = Tf(df[:R])

NΔt = 1000000
prin = 10000
# NΔt = Int(floor(1000000/dt))
# prin = Int(floor(10000/dt))
