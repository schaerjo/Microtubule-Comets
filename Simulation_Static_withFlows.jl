#include("InputParametersLocal.jl")
#include("InputParametersCluster_InfMT.jl")
include("InputParametersClusterArray.jl")

println(Dates.now())

####Simulation######################################################################################################

#   Collide and stream
#   ====================
function kernel_collide!(fn, f, F, F_axi, ρ_, v, ω, w, ξ, R)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds begin      
        # Density and velocity
        ρ = ρ_[i,j]
        vx = v[i,j,1]
        vr = v[i,j,2]
        v2= (vx*vx+vr*vr)
        # Source term factor
        ω_ = (1-0.5*ω)
        Fx=ω_*F[i,j,1]
        Fr=ω_*F[i,j,2]
        #cylindrical terms
        θ=-ρ_[i,j]*vr/(9*(R + j - 0.5))
        Fx_axi = F_axi[i,j,1]
        Fr_axi = F_axi[i,j,2]

        for α = 1:9
            fn[i,j,α] = (1.0-ω)*f[i,j,α] + ω*(ρ*w[α]*(1 + 3*(ξ[α,1]*vx + ξ[α,2]*vr) - 1.5*v2 + 4.5*(ξ[α,1]*vx + ξ[α,2]*vr)^2)) + w[α]*( Fx*(3*(ξ[α,1]-vx)+9*((ξ[α,1]*vx + ξ[α,2]*vr))*ξ[α,1]) + Fr*(3*(ξ[α,2]-vr)+9*((ξ[α,1]*vx + ξ[α,2]*vr))*ξ[α,2]) ) + θ + (1/6)*(ξ[α,1]*Fx_axi + ξ[α,2]*Fr_axi)
        end
    end
    return nothing
end


# Stream 2D:
function kernel_stream!(f, fpc, ξ, Nx, Nr)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    for α=1:9
        ii = mod(i + ξ[α,1], 1:Nx)
        jj = mod(j + ξ[α,2], 1:Nr)
        f[ii, jj, α] = fpc[i,j,α]
    end
    
    return nothing
end

function kernel_apply_BB!(f, fpc, z1, z2, Nx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    if j==z1
        f[i,j,3] = fpc[i,j,5]
        f[i,j,7] = fpc[i,j,9]
        f[i,j,6] = fpc[i,j,8]
    elseif j==z2
        f[i,j,5] = fpc[i,j,3]
        f[i,j,8] = fpc[i,j,6]
        f[i,j,9] = fpc[i,j,7]
    end
    return nothing
end

#   Meso to macro
#   ===============
function kernel_comp_ρ_v!(ρ, v, f, ξ, F)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    @inbounds ρ[i,j] = 0
    @inbounds v[i,j,1]=0.5*F[i,j,1]; v[i,j,2]=0.5*F[i,j,2]
    
    @inbounds for α = 1:9
        ρ[i, j] +=  f[i, j, α] 
        v[i, j, 1] +=  f[i, j, α] * ξ[α,1]
        v[i, j, 2] +=  f[i, j, α] * ξ[α,2]
    end
    
    @inbounds v[i,j,1] *= 1/ρ[i,j]
    @inbounds v[i,j,2] *= 1/ρ[i,j]
    
    return nothing
end

#   Phase Field functions
#   =====================
function kernel_comp_derivative!(Δϕ, ∇ϕ, ϕ, a, Nx, Nr, R, dx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    jm = mod(j-1,1:Nr); jp = mod(j+1,1:Nr)

    r = (R + j - 0.5)*dx
    dxdx=dx*dx
    
    @inbounds begin
        ϕr0 = 0.0
        if j==1
            ϕr0 = ϕ[i,j] + a
            ∇ϕ[i,j,1] = (ϕ[ip,j] - ϕ[i_,j])/(2*dx) #∂xϕ
            ∇ϕ[i,j,2] = (ϕ[i,jp] - ϕr0)/(2*dx) #∂zϕ
            Δϕ[i,j] = (ϕ[i_,j] + ϕ[ip,j] + ϕr0 + ϕ[i,jp] - 4*ϕ[i,j])/dxdx + ∇ϕ[i,j,2]/r
        elseif j==Nr
            ∇ϕ[i,j,1] = (ϕ[ip,j] - ϕ[i_,j])/(2*dx) #∂xϕ
            ∇ϕ[i,j,2] = (ϕ[i,j] - ϕ[i,jm])/(2*dx) #∂zϕ
            Δϕ[i,j] = (ϕ[i_,j] + ϕ[ip,j] + ϕ[i,jm] + ϕ[i,j] - 4*ϕ[i,j])/dxdx + ∇ϕ[i,j,2]/r
        else
            ∇ϕ[i,j,1] = (ϕ[ip,j] - ϕ[i_,j])/(2*dx) #∂xϕ
            ∇ϕ[i,j,2] = (ϕ[i,jp] - ϕ[i,jm])/(2*dx) #∂zϕ
            Δϕ[i,j] = (ϕ[i_,j] + ϕ[ip,j] + ϕ[i,jm] + ϕ[i,jp] - 4*ϕ[i,j])/dxdx + ∇ϕ[i,j,2]/r
    end
    
    end
    return nothing
end

function kernel_comp_μ!(μ, ϕ_, Δϕ_, ϕa, ϕb, β, k)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    @inbounds begin
        ϕ = ϕ_[i,j]
        Δϕ = Δϕ_[i,j]
        μ[i,j] = 4*β*(ϕ-ϕa)*(ϕ-ϕb)*(ϕ - (ϕa+ϕb)*0.5) - k*Δϕ
    end
    return nothing
end

#   Force computation
#   =================
function kernel_comp_F!(F, ϕ, μ, Nx, Nr)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    jm = mod(j-1,1:Nr); jp = mod(j+1,1:Nr)

    Fx = 0.0
    Fr = 0.0

    @inbounds begin
        if j==1
            Fx += -ϕ[i,j]*(μ[ip,j] - μ[i_,j])/2
            Fr += -ϕ[i,j]*(μ[i,jp] - μ[i,j])/2
        elseif j==Nr
            Fx += -ϕ[i,j]*(μ[ip,j] - μ[i_,j])/2
            Fr += -ϕ[i,j]*(μ[i,j] - μ[i,jm])/2
        else
            Fx += -ϕ[i,j]*(μ[ip,j] - μ[i_,j])/2
            Fr += -ϕ[i,j]*(μ[i,jp] - μ[i,jm])/2
        end
        F[i,j,1] += Fx
        F[i,j,2] += Fr
    end
    return nothing
end

function kernel_comp_F_axi!(F_axi, v, ρ_, f, ω, nu, Nx, R)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    imm = mod(i-2,1:Nx); ipp = mod(i+2,1:Nx)

    r = (R + j - 0.5)
    Fx = 0.0
    Fr = 0.0

    @inbounds begin
        #Density and velocity
        ρ = ρ_[i,j]; vx = v[i,j,1]; vr = v[i,j,2];
        vrvr=vr*vr; vxvr=vx*vr;

        #Computation with paper trick.
        Fx = -ρ*vxvr/r - ((nu)/r)*(3*ω*(f[i,j,6]-f[i,j,7]+f[i,j,8]-f[i,j,9] - ρ*vxvr) + (v[ip, j, 2]-v[i_, j, 2])/2)
        Fr = -ρ*vrvr/r - ρ*nu*vr/(r*r) - ((nu)/r)*(1.5*ω*(f[i,j,3]-f[i,j,5]+f[i,j,6]+f[i,j,7]-f[i,j,8]-f[i,j,9] - ρ*vr) + 0.5*vr/r)

        F_axi[i,j,1] += Fx
        F_axi[i,j,2] += Fr
    end
    return nothing
end

#   Cahn_Hilliard
#   =============
function kernel_advection!(ϕn, ϕ, v, Nx, Nr, dt, R, dx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    jm = mod(j-1,1:Nr); jp = mod(j+1,1:Nr)
    
    r = R + (j - 0.5)*dx
    ∂xϕvx = 0.0
    ∂rϕvr = 0.0
    @inbounds begin
        ∂xϕvx += -abs(v[i,j,1])*ϕ[i,j]
        ∂xϕvx += v[i_,j,1] > 0 ? ϕ[i_,j]*v[i_,j,1] : 0.0
        ∂xϕvx -= v[ip,j,1] < 0 ? ϕ[ip,j]*v[ip,j,1] : 0.0
        
        if j==1
            ∂rϕvr += v[i,j,2] > 0 ? -abs(v[i,j,2])*ϕ[i,j] - ϕ[i,j]*v[i,j,2]/r : 0.0
            ∂rϕvr -= v[i,jp,2] < 0 ? ϕ[i,jp]*v[i,jp,2] : 0.0
        elseif j==Nr
            ∂rϕvr += v[i,j,2] < 0 ? -abs(v[i,j,2])*ϕ[i,j] - ϕ[i,j]*v[i,j,2]/r : 0.0
            ∂rϕvr += v[i,jm,2] > 0 ? ϕ[i,jm]*v[i,jm,2] : 0.0
        else
            ∂rϕvr += -abs(v[i,j,2])*ϕ[i,j] - ϕ[i,j]*v[i,j,2]/r
            ∂rϕvr += v[i,jm,2] > 0 ? ϕ[i,jm]*v[i,jm,2] : 0.0
            ∂rϕvr -= v[i,jp,2] < 0 ? ϕ[i,jp]*v[i,jp,2] : 0.0
        end
            
        ϕn[i,j] = ϕ[i,j] + dt*(∂xϕvx + ∂rϕvr)
    end
    return nothing
end

function kernel_diffusion!(ϕn, ϕ, μ, M, Nx, Nr, dt, R, dx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    jm = mod(j-1,1:Nr); jp = mod(j+1,1:Nr)
    
    r = R + (j - 0.5)*dx
    dxdx=dx*dx
    @inbounds begin
        if j==1
            Δμ = (μ[i_,j] + μ[i,j] - 4*μ[i,j] + μ[ip,j] + μ[i,jp])/dxdx + (μ[i,jp] - μ[i,j])/(2*r*dx)
        elseif j==Nr
            Δμ = (μ[i_,j] + μ[i,jm] - 4*μ[i,j] + μ[ip,j] + μ[i,j])/dxdx + (μ[i,j] - μ[i,jm])/(2*r*dx)
        else
            Δμ = (μ[i_,j] + μ[i,jm] - 4*μ[i,j] + μ[ip,j] + μ[i,jp])/dxdx + (μ[i,jp] - μ[i,jm])/(2*r*dx)
        end
        ϕn[i,j] = ϕ[i,j] + dt*(M*Δμ)
    end
    return nothing
end

#   Initialize
#   ============
function kernel_initialize_pop!(f, w, ρ0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    for α in eachindex(w)
        f[i,j,α] = w[α]*ρ0
    end
    
    return nothing
end

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
dird = string(file,"Data/")
isdir(dird) ? rm(dird, recursive=true) : nothing
mkpath(dir)
mkpath(dird)

f = CUDA.ones(Tf, Nx, Nr, Qlbm)
f_t = CUDA.zeros(Tf, Nx, Nr, Qlbm)
ρ = CUDA.ones(Tf, Nx, Nr)*ρ0
ρ_temp = CUDA.ones(Tf, Nx, Nr) * ρ0
v = CUDA.zeros(Tf, Nx, Nr, 2)
v_temp = CUDA.zeros(Tf, Nx, Nr, 2)

ϕ = CUDA.zeros(Tf, Nx, Nr)
ϕ_temp = CUDA.zeros(Tf, Nx, Nr)
∇ϕ = CUDA.zeros(Tf, Nx, Nr, 2)
Δϕ = CUDA.zeros(Tf, Nx, Nr)

F = CUDA.zeros(Tf, Nx, Nr, 2)
μ = CUDA.zeros(Tf, Nx, Nr)

F_axi = CUDA.zeros(Tf, Nx, Nr, 2)

interface = CUDA.zeros(Tf, Nx)

#CPU arrays for saving to disk.
interfacew = zeros(Tf, Nx)
interfaceM = zeros(Tf, Ntw, Nx)
global twrite = Ti(1)

#Kernels declaration
gpukernel_collide = @cuda launch = false kernel_collide!(f_t, f, F, F_axi, ρ, v, ω, w, ξ, R)
gpukernel_stream = @cuda launch = false kernel_stream!(f, f_t, ξ, Nx, Nr)
gpukernel_apply_BB = @cuda launch = false kernel_apply_BB!(f, f_t, 1, Nr, Nx)
gpucomp_ρ_v = @cuda launch = false kernel_comp_ρ_v!(ρ, v, f, ξ, F)
gpukernel_comp_derivative = @cuda launch = false kernel_comp_derivative!(Δϕ, ∇ϕ, ϕ, a, Nx, Nr, R, dx)
gpukernel_comp_μ = @cuda launch = false kernel_comp_μ!(μ, ϕ, Δϕ, ϕa, ϕb, β, k)
gpukernel_comp_F = @cuda launch = false kernel_comp_F!(F, ϕ, μ, Nx, Nr)
gpukernel_comp_F_axi = @cuda launch = false kernel_comp_F_axi!(F_axi, v, ρ, f, ω, nu, Nx, R)
gpukernel_advection = @cuda launch = false kernel_advection!(ϕ_temp, ϕ, v, Nx, Nr, dt, R, dx)
gpukernel_diffusion = @cuda launch = false kernel_diffusion!(ϕ_temp, ϕ, μ, M, Nx, Nr, dtd, R, dx)

gpukernel_initialize_pop = @cuda launch = false kernel_initialize_pop!(f, w, ρ0)
gpukernel_phi_init = @cuda launch = false kernel_phi_init!(ϕ, ϕ0)

gpukernel_interface_tracking = @cuda launch = false kernel_interface_tracking!(ϕ, interface, Nx, Nr)

# Initialization
gpukernel_initialize_pop(f, w, ρ0; threads=a2D_block, blocks=a2D_grid)
gpukernel_phi_init(ϕ, ϕ0; threads=a2D_block, blocks=a2D_grid)
pert = CUDA.rand(Tf, Nx, Nr)
@. ϕ *= (1 + (pert - 0.5) * 1e-2)
# gpukernel_film_init = @cuda launch = false kernel_film_init!(ϕ, ϕa, ϕb, thickness, D, R, pert)
# gpukernel_film_init(ϕ, ϕa, ϕb, thickness, D, R, pert; threads=a2D_block, blocks=a2D_grid)

gpukernel_interface_tracking(ϕ, interface, Nx, Nr; threads=256, blocks=Block_Inter)
gpukernel_comp_derivative(Δϕ, ∇ϕ, ϕ, a, Nx, Nr, R, dx; threads=a2D_block, blocks=a2D_grid)
gpukernel_comp_μ(μ, ϕ, Δϕ, ϕa, ϕb, β, k; threads=a2D_block, blocks=a2D_grid)
gpukernel_comp_F(F, ϕ, μ, Nx, Nr; threads=a2D_block, blocks=a2D_grid)
gpukernel_comp_F_axi(F_axi, v, ρ, f, ω, nu, Nx, R; threads=a2D_block, blocks=a2D_grid)

println(Dates.now())
println(0)
copyto!(ϕw, ϕ)
copyto!(interfacew, interface)
interfaceM[twrite,:] = interfacew
global twrite += 1
save(string(dird, "data_", @sprintf("%08i", 0), ".jld"), "PF", ϕw)
GC.gc()
CUDA.memory_status()

if any(isnan, ϕ)
    println("t=", 0, " and ϕ is NaN")
    println("Job is over, error")
    return 1
end
if any(isnan, ρ)
    println("t=", 0, " and ρ is NaN")
    println("Job is over, error")
    return 1
end

#Time Loop
CUDA.@time begin

    for t=1:NΔt
        #Lattice Boltzmann for ρ and v.
        gpukernel_collide(f_t, f, F, F_axi, ρ, v, ω, w, ξ, R; threads=a2D_block, blocks=a2D_grid)
        gpukernel_stream(f, f_t, ξ, Nx, Nr; threads=a2D_block, blocks=a2D_grid)
        gpukernel_apply_BB(f, f_t, 1, Nr, Nx; threads=a2D_block, blocks=a2D_grid)

        #Advection
        gpukernel_advection(ϕ_temp, ϕ, v, Nx, Nr, dt, R, dx; threads=a2D_block, blocks=a2D_grid)
        copyto!(ϕ, ϕ_temp)
        if any(isnan, ϕ)
            println("t=", t, " and ϕ is NaN after advection")
            println("Job is over, error")
            return 1
        end
        #Diffusion
        for i = 1:Ndtd
            gpukernel_comp_derivative(Δϕ, ∇ϕ, ϕ, a, Nx, Nr, R, dx; threads=a2D_block, blocks=a2D_grid)
            gpukernel_comp_μ(μ, ϕ, Δϕ, ϕa, ϕb, β, k; threads=a2D_block, blocks=a2D_grid)
            gpukernel_diffusion(ϕ_temp, ϕ, μ, M, Nx, Nr, dtd, R, dx; threads=a2D_block, blocks=a2D_grid)
            copyto!(ϕ, ϕ_temp)
            if any(isnan, ϕ)
                println("t=", t, " and idt = ", i, " and ϕ is NaN after diffusion")
                println("Job is over, error")
                return 1
            end
        end

        gpucomp_ρ_v(ρ, v, f, ξ, F; threads=a2D_block, blocks=a2D_grid)
        @. F_axi = 0.0
        gpukernel_comp_F_axi(F_axi, v, ρ, f, ω, nu, Nx, R; threads=a2D_block, blocks=a2D_grid)
        @. F = 0.0
        gpukernel_comp_derivative(Δϕ, ∇ϕ, ϕ, a, Nx, Nr, R, dx; threads=a2D_block, blocks=a2D_grid)
        gpukernel_comp_μ(μ, ϕ, Δϕ, ϕa, ϕb, β, k; threads=a2D_block, blocks=a2D_grid)
        gpukernel_comp_F(F, ϕ, μ, Nx, Nr; threads=a2D_block, blocks=a2D_grid)
        
        if any(isnan, ϕ)
            println("t=", t, " and ϕ is NaN")
            println("Job is over, error")
            return 1
        end
        if any(isnan, ρ)
            println("t=", t, " and ρ is NaN")
            println("Job is over, error")
            return 1
        end

        if t >= prin0 && (t-prin0) % prin == 0
            println(Dates.now())
            println(t)
            copyto!(ϕw, ϕ)
            gpukernel_interface_tracking(ϕ, interface, Nx, Nr; threads=256, blocks=Block_Inter)
            copyto!(interfacew, interface)
            interfaceM[twrite,:] = interfacew
            global twrite += 1
            save(string(dird, "data_", @sprintf("%08i", t), ".jld"), "PF", ϕw)
            GC.gc()
            CUDA.memory_status()
        end
    end
end
# save(string(dird, "Interface.jld"), "Interface", interfaceM)
