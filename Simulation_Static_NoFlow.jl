#include("InputParametersLocal2.jl")
#include("InputParametersCluster.jl")
include("InputParametersClusterArray.jl")

println(Dates.now())

####Simulation######################################################################################################

#   Phase Field functions
#   =====================
function kernel_comp_derivative!(Δϕ, ∇ϕ, ϕ, a, Nx, Nr, R, dx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    #jm = mod(j-1,1:Nr); jp = mod(j+1,1:Nr)
    jm = j - 1
    jp = j + 1

    r = (R + j - 0.5) * dx
    dxdx = dx * dx

    @inbounds begin
        ϕr0 = 0.0
        if j == 1 #MT surface.
            ϕr0 = ϕ[i,j] + a
            ∇ϕ[i, j, 1] = (ϕ[ip, j] - ϕ[i_, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, jp] - ϕr0) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i_, j] + ϕ[ip, j] + ϕr0 + ϕ[i, jp] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        elseif j == Nr #Top.
            ∇ϕ[i, j, 1] = (ϕ[ip, j] - ϕ[i_, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, j] - ϕ[i, jm]) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i_, j] + ϕ[ip, j] + ϕ[i, j] + ϕ[i, jm] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        else
            ∇ϕ[i, j, 1] = (ϕ[ip, j] - ϕ[i_, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, jp] - ϕ[i, jm]) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i_, j] + ϕ[ip, j] + ϕ[i, jm] + ϕ[i, jp] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        end

    end
    return nothing
end

function kernel_comp_μ!(μ, ϕ_, Δϕ_, ϕa, ϕb, β, k)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    @inbounds begin

        ϕ = ϕ_[i, j]
        Δϕ = Δϕ_[i, j]

        μ[i, j] = 4 * β * (ϕ - ϕa) * (ϕ - ϕb) * (ϕ - (ϕa + ϕb) * 0.5) - k * Δϕ

    end
    return nothing
end

#   Cahn_Hilliard
#   =============
function kernel_diffusion!(ϕn, ϕ, μ, M, dt, dx, Nx, Nr, R)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    #jm = mod(j-1,1:Nr); jp = mod(j+1,1:Nr)
    jm = j - 1
    jp = j + 1

    r = (R + j - 0.5) * dx
    dxdx = dx * dx
    @inbounds begin
        if j == 1 #MT surface
            Δμ = (μ[i_, j] + μ[i, j] - 4 * μ[i, j] + μ[ip, j] + μ[i, jp]) / dxdx + (μ[i, jp] - μ[i, j]) / (2 * r * dx)
        elseif j == Nr #Top
            Δμ = (μ[i_, j] + μ[i, jm] - 4 * μ[i, j] + μ[ip, j] + μ[i, j]) / dxdx + (μ[i, j] - μ[i, jm]) / (2 * r * dx)
        else
            Δμ = (μ[i_, j] + μ[i, jm] - 4 * μ[i, j] + μ[ip, j] + μ[i, jp]) / dxdx + (μ[i, jp] - μ[i, jm]) / (2 * r * dx)
        end
        ϕn[i, j] = ϕ[i, j] + dt * (M * Δμ)
    end
    return nothing
end

#   Initialize
#   ============
function kernel_phi_init!(ϕ, ϕ0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    @inbounds begin
        ϕ[i, j] = ϕ0
    end

    return nothing
end

function kernel_film_init!(ϕ, ϕa, ϕb, thickness, D, R, pert)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    r = (R + j - 0.5)
    
    @inbounds begin

        ϕ[i,j] = 0.5*(ϕa+ϕb) + 0.5*(ϕb-ϕa)*CUDA.tanh(2*((thickness + (pert[i]-0.5)) - r)/(D))

    end
    
    return nothing
end

#   Interface Detection
#   =====================
function kernel_interface_tracking!(ϕ, interface, Nx, Nr)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    @inbounds begin
        if i<= Nx
            for j=1:Nr
                if ϕ[i,j] <= 0.5
                    if j==1
                        interface[i] = (j-1)
                    else
                        interface[i] = (j-1) + (0.5-ϕ[i,j-1])/(ϕ[i,j]-ϕ[i,j-1])
                    end
                    break
                end
            end
        end
    end
    return nothing
end

#   Main
#   ====

dird = string(file, "Data/")
isdir(dird) ? rm(dird, recursive=true) : nothing
mkpath(dir)
mkpath(dird)

ϕ = CUDA.zeros(Tf, Nx, Nr)
ϕ_temp = CUDA.zeros(Tf, Nx, Nr)
∇ϕ = CUDA.zeros(Tf, Nx, Nr, 2)
Δϕ = CUDA.zeros(Tf, Nx, Nr)

μ = CUDA.zeros(Tf, Nx, Nr)

#CPU arrays for saving to disk.
ϕw = zeros(Tf, Nx, Nr)
# interfacew = zeros(Tf, Nx)
# interfaceM = zeros(Tf, Ntw+1, Nx)
# global twrite = Ti(1)

#Kernels declaration
gpukernel_comp_derivative = @cuda launch = false kernel_comp_derivative!(Δϕ, ∇ϕ, ϕ, a, Nx, Nr, R, dx)
gpukernel_comp_μ = @cuda launch = false kernel_comp_μ!(μ, ϕ, Δϕ, ϕa, ϕb, β, k)
gpukernel_diffusion = @cuda launch = false kernel_diffusion!(ϕ_temp, ϕ, μ, M, dt, dx, Nx, Nr, R)
gpukernel_phi_init = @cuda launch = false kernel_phi_init!(ϕ, ϕ0)

#gpukernel_interface_tracking = @cuda launch = false kernel_interface_tracking!(ϕ, interface, Nx, Nr)

# Initialization
gpukernel_phi_init(ϕ, ϕ0; threads=a2D_block, blocks=a2D_grid)
pert = CUDA.rand(Tf, Nx, Nr)
@. ϕ *= (1 + (pert - 0.5) * 1e-2)
#gpukernel_film_init = @cuda launch = false kernel_film_init!(ϕ, ϕa, ϕb, thickness, D, R, pert)
#gpukernel_film_init(ϕ, ϕa, ϕb, thickness, D, R, pert)

#gpukernel_interface_tracking(ϕ, interface, Nx, Nr; threads=256, blocks=Block_Inter)
gpukernel_comp_derivative(Δϕ, ∇ϕ, ϕ, a, Nx, Nr, R, dx; threads=a2D_block, blocks=a2D_grid)
gpukernel_comp_μ(μ, ϕ, Δϕ, ϕa, ϕb, β, k; threads=a2D_block, blocks=a2D_grid)

println(Dates.now())
println(0)
copyto!(ϕw, ϕ)
# copyto!(interfacew, interface)
# interfaceM[twrite,:] = interfacew
# global twrite += 1
save(string(dird, "data_", @sprintf("%08i", 0), ".jld"), "PF", ϕw)
GC.gc()
CUDA.memory_status()

if any(isnan, ϕ)
    println("t=", 0, " and ϕ is NaN")
    println("Job is over, error")
    return 1
end

#Time Loop
CUDA.@time begin

    for t = 1:NΔt
        for i = 1:Ndtd
            #Diffusion
            gpukernel_comp_derivative(Δϕ, ∇ϕ, ϕ, a, Nx, Nr, R, dx; threads=a2D_block, blocks=a2D_grid)
            gpukernel_comp_μ(μ, ϕ, Δϕ, ϕa, ϕb, β, k; threads=a2D_block, blocks=a2D_grid)
            gpukernel_diffusion(ϕ_temp, ϕ, μ, M, dtd, dx, Nx, Nr, R; threads=a2D_block, blocks=a2D_grid)
            copyto!(ϕ, ϕ_temp)
            if any(isnan, ϕ)
                println("t=", t, " and idt = ", i, " and ϕ is NaN after diffusion")
                println("Job is over, error")
                return 1
            end
        end

        if any(isnan, ϕ)
            println("t=", t, " and ϕ is NaN")
            println("Job is over, error")
            return 1
        end

        if t >= prin0 && (t-prin0) % prin == 0
            println(Dates.now())
            println(t)
            copyto!(ϕw, ϕ)
            # gpukernel_interface_tracking(ϕ, interface, Nx, Nr; threads=256, blocks=Block_Inter)
            # copyto!(interfacew, interface)
            # interfaceM[twrite,:] = interfacew
            # global twrite += 1
            save(string(dird, "data_", @sprintf("%08i", t), ".jld"), "PF", ϕw)
            GC.gc()
            CUDA.memory_status()
        end
    end
end
# save(string(dird, "Interface.jld"), "Interface", interfaceM)
