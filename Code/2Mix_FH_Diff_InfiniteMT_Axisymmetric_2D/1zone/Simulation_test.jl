#Packages and float type
using Pkg
listofpkg = ["CUDA", "JLD", "Statistics", "Printf", "LinearAlgebra", "Random", "DelimitedFiles", "CSV", "DataFrames", "Dates"]
for package in listofpkg
    try
        @eval using $(Symbol(package))
    catch
        println("Installing $package ...")
        Pkg.add(package)
    end
end
using CUDA
using JLD
using Printf

using CSV, DataFrames

# include("InputParameters.jl")
include("InputParametersLocal.jl")

####Simulation######################################################################################################

#Gradient and chemical potential
function kernel_comp_∇_Δ!(Δϕ, ∇ϕ, ϕ, Nx, Nz, h, R, dx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    imm = mod(i-2,1:Nx); ipp = mod(i+2,1:Nx)
    jm = mod(j-1,1:Nz); jp = mod(j+1,1:Nz)
    jmm = mod(j-2,1:Nz); jpp = mod(j+2,1:Nz)

    dxdx = dx*dx

    @inbounds begin
        if j == 2
            ∇ϕ[i,j,1] = (-ϕ[ipp,j] + 8*ϕ[ip,j] - 8*ϕ[i_,j] + ϕ[imm,j])/(12*dx) #∂xϕ
            ∇ϕ[i,j,2] = (ϕ[i,jp] - ϕ[i,jm])/(2*dx) #∂zϕ 1st order

            Δϕ[i,j] = (ϕ[i_,j] + ϕ[ip,j] + ϕ[i,jm] + ϕ[i,jp] - 4*ϕ[i,j])/dxdx + ∇ϕ[i,j,2]/(R + (j - 1.0)*dx)
        elseif j == 1
            ∇ϕ[i,j,1] = (-ϕ[ipp,j] + 8*ϕ[ip,j] - 8*ϕ[i_,j] + ϕ[imm,j])/(12*dx) #∂xϕ
            ∇ϕ[i,j,2] = -h #∂zϕ

            Δϕ[i,j] = (ϕ[i_,j] + ϕ[ip,j] - 2*ϕ[i,j])/dxdx + 2*((ϕ[i,jp] - ϕ[i,j]) + dx*h)/dxdx + ∇ϕ[i,j,2]/(R + (j - 1.0)*dx)
        elseif j == Nz
            ∇ϕ[i,j,1] = (-ϕ[ipp,j] + 8*ϕ[ip,j] - 8*ϕ[i_,j] + ϕ[imm,j])/(12*dx) #∂xϕ
            ∇ϕ[i,j,2] = 0.0 #∂zϕ

            Δϕ[i,j] = (ϕ[i_,j] + ϕ[ip,j] - 2*ϕ[i,j])/dxdx + 2*((ϕ[i,jm] - ϕ[i,j]))/dxdx + ∇ϕ[i,j,2]/(R + (j - 1.0)*dx)
        elseif j == Nz-1
            ∇ϕ[i,j,1] = (-ϕ[ipp,j] + 8*ϕ[ip,j] - 8*ϕ[i_,j] + ϕ[imm,j])/(12*dx) #∂xϕ
            ∇ϕ[i,j,2] = (ϕ[i,jp] - ϕ[i,jm])/(2*dx) #∂zϕ 1st order
            
            Δϕ[i,j] = (ϕ[i_,j] + ϕ[ip,j] + ϕ[i,jm] + ϕ[i,jp] - 4*ϕ[i,j])/dxdx + ∇ϕ[i,j,2]/(R + (j - 1.0)*dx)
        else
            ∇ϕ[i,j,1] = (-ϕ[ipp,j] + 8*ϕ[ip,j] - 8*ϕ[i_,j] + ϕ[imm,j])/(12*dx) #∂xϕ
            ∇ϕ[i,j,2] = (-ϕ[i,jpp] + 8*ϕ[i,jp] - 8*ϕ[i,jm] + ϕ[i,jmm])/(12*dx) #∂zϕ

            Δϕ[i,j] = (ϕ[i_,j] + ϕ[ip,j] + ϕ[i,jm] + ϕ[i,jp] - 4*ϕ[i,j])/dxdx + ∇ϕ[i,j,2]/(R + (j - 1.0)*dx)
        end
    end
    return nothing
end

function comp_∇_Δ!(Δϕ, ∇ϕ, ϕ, Nx, Nz, h, R, dx)
    @cuda threads = block_dim blocks = grid_dim kernel_comp_∇_Δ!(Δϕ, ∇ϕ, ϕ, Nx, Nz, h, R, dx)
end

function kernel_comp_μ!(μ, ϕ_, Δϕ_, χ, k, a)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    # imm = mod(i-2,1:Nx); ipp = mod(i+2,1:Nx)
    # jm = mod(j-1,1:Nz); jp = mod(j+1,1:Nz)
    # jmm = mod(j-2,1:Nz); jpp = mod(j+2,1:Nz)

    @inbounds begin

        ϕ = ϕ_[i,j]; #ϕ2 = ϕ*ϕ
        #∂xϕ = ∇ϕ[i,j,1]; ∂zϕ = ∇ϕ[i,j,2]
        Δϕ = Δϕ_[i,j]

        # μ
        μ[i,j] = a*(CUDA.log(ϕ/(1-ϕ)) + χ*(1-2*ϕ) - k*(Δϕ))

    end
    return nothing
end

function comp_μ!(μ, ϕ_, Δϕ_, χ, k, a)
    @cuda threads = block_dim blocks = grid_dim kernel_comp_μ!(μ, ϕ_, Δϕ_, χ, k, a)
end

function kernel_comp_Δ_μ!(μ, Δμ, Nx, Nz, R, dx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    # imm = mod(i-2,1:Nx); ipp = mod(i+2,1:Nx)
    jm = mod(j-1,1:Nz); jp = mod(j+1,1:Nz)
    jmm = mod(j-2,1:Nz); jpp = mod(j+2,1:Nz)

    dxdx = dx*dx

    @inbounds begin

        if j == 1
            Δμ[i,j] = (μ[i_,j] + μ[ip,j] + 2*μ[i,jp] - 4*μ[i,j])/dxdx
        elseif j == 2
            Δμ[i,j] = (μ[i_,j] + μ[i,jm] + μ[ip,j] + μ[i,jp] - 4*μ[i,j])/dxdx + (μ[i,jp] - μ[i,jm])/(2*dx*(R + (j - 1.0)*dx))
        elseif j == Nz
            Δμ[i,j] = (μ[i_,j] + 2*μ[i,jm] + μ[ip,j] - 4*μ[i,j])/dxdx
        elseif j == Nz-1
            Δμ[i,j] = (μ[i_,j] + μ[i,jm] + μ[ip,j] + μ[i,jp] - 4*μ[i,j])/dxdx + (μ[i,jp] - μ[i,jm])/(2*dx*(R + (j - 1.0)*dx))
        else
           Δμ[i,j] = (μ[i_,j] + μ[i,jm] + μ[ip,j] + μ[i,jp] - 4*μ[i,j])/dxdx + (-μ[i,jpp] + 8*μ[i,jp] - 8*μ[i,jm] + μ[i,jmm])/(12*dx*(R + (j - 1.0)*dx))
        end

    end

    return nothing
end

function comp_Δ_μ!(μ, Δμ, Nx, Nz, R, dx)
    @cuda threads = block_dim blocks = grid_dim kernel_comp_Δ_μ!(μ, Δμ, Nx, Nz, R, dx)
end

#Conservation Equation
function kernel_comp_ϕ!(ϕn, ϕ, Δμ, dt, M, dϕ)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    @inbounds begin

        ϕn[i,j] = ϕ[i,j] + dt*(M*Δμ[i,j])
        dϕ[i,j] = dt*(M*Δμ[i,j])

    end

    return nothing
end

@inline function evolve_phi!(ϕ_temp, ϕ, ∇ϕ, Δϕ, μ, Δμ, Nx, Nz, k, h, R, χ, dt, dx, M, a, dϕ)
    @cuda threads = block_dim blocks = grid_dim kernel_comp_∇_Δ!(Δϕ, ∇ϕ, ϕ, Nx, Nz, h, R, dx)
    @cuda threads = block_dim blocks = grid_dim kernel_comp_μ!(μ, ϕ, Δϕ, χ, k, a)
    @cuda threads = block_dim blocks = grid_dim kernel_comp_Δ_μ!(μ, Δμ, Nx, Nz, R, dx)
    @cuda threads = block_dim blocks = grid_dim kernel_comp_ϕ!(ϕ_temp, ϕ, Δμ, dt, M, dϕ)
    ϕ .= ϕ_temp
end

#Simulation ###############################################################################

#Initialization
ϕ = CUDA.ones(Tf, Nx, Nz)*ϕ0
dϕ = CUDA.zeros(Tf, Nx, Nz)
perta = CUDA.rand(Tf, Nx, Nz)
@. ϕ[:,:] *= (1+(perta-0.5)*1e-5)
ϕ_temp = CUDA.ones(Tf, Nx, Nz)
∇ϕ = CUDA.zeros(Tf, Nx, Nz, 2)
Δϕ = CUDA.zeros(Tf, Nx, Nz)
μ = CUDA.zeros(Tf, Nx, Nz)
Δμ = CUDA.zeros(Tf, Nx, Nz)

global max = Tf(0.0)

dird = string(dir,"Data/")
# dird = string(file,"Data/")
isdir(dird) ? rm(dird, recursive=true) : nothing
mkpath(dir)
mkpath(dird)

println(" M= ", M, " a= ", a, " chi= ", χ, " k= ", k, " h= ", h, " phi0= ", ϕ0, " R= ", R, " dx= ", dx, " dt= ", dt, " Nx= ", Nx, " Nz= ", Nz, " T= ", NΔt, " prin= ", prin)

#Time Loop
for t=0:NΔt
    if t%prin==0 #&& t>=10*1200
        println(t)
        println("Max dϕ = ", maximum(dϕ))
        println("Max ϕ = ", maximum(ϕ))
        println("Min dϕ = ", minimum(dϕ))
        println("Min ϕ = ", minimum(ϕ))
        save(string(dird,"data_",@sprintf("%08i",t),".jld"), "PF", Array(ϕ), "dPF", Array(dϕ))
    end

    evolve_phi!(ϕ_temp, ϕ, ∇ϕ, Δϕ, μ, Δμ, Nx, Nz, k, h, R, χ, dt, dx, M, a, dϕ)

    global max = maximum(dϕ) > max ? maximum(dϕ) : max

    if any(isnan, ϕ)
        println("t=", t, " and ϕ is NaN")
        println("Max dϕ = ", maximum(dϕ))
        println("Max ϕ = ", maximum(ϕ))
        println("Min dϕ = ", minimum(dϕ))
        println("Min ϕ = ", minimum(ϕ))
        println("Job is over, error")
        return 1
    end
    if any(x->x>=1.0, ϕ)
        println("t=", t, " and ϕ >= 1")
        println("Max dϕ = ", maximum(dϕ))
        println("Max ϕ = ", maximum(ϕ))
        println("Min dϕ = ", minimum(dϕ))
        println("Min ϕ = ", minimum(ϕ))
        println("Job is over, error")
        return 1
    end
    if any(x->x<=0.0, ϕ)
        println("t=", t, " and ϕ <= 0")
        println("Max dϕ = ", maximum(dϕ))
        println("Max ϕ = ", maximum(ϕ))
        println("Min dϕ = ", minimum(dϕ))
        println("Min ϕ = ", minimum(ϕ))
        println("Job is over, error")
        return 1
    end
end

println("Max dϕ = ", max)
# println("Job is over, no error")