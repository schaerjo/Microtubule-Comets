include("InputParametersClusterArray.jl")

println(Dates.now())

####Simulation######################################################################################################

#   Phase Field functions
#   =====================
function kernel_comp_derivative!(Δϕ, ∇ϕ, ϕ, ϕ0, a, a0, atip, lzone, tail, lzone0, dx, x1, x2, y1, y2, xm, ym)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    #i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    #jm = mod(j-1,1:Nr); jp = mod(j+1,1:Nr)
    i_ = i - 1
    ip = i + 1
    jm = j - 1
    jp = j + 1

    r = (j - 0.5) * dx
    dxdx = dx * dx

    @inbounds begin
        ϕr0 = 0.0
        distx = 0
        if j == ym + 1 && i == x1 #Bottom left corner.
            if lzone0 != 0
                ϕr0 = ϕ[i, j]
            elseif tail == 0
                ϕr0 = ϕ[i, j] + a0
            else
                ϕr0 = ϕ[i, j] + (a - a0) * CUDA.exp((i * dx - xm + lzone) / tail) + a0
            end
            ∇ϕ[i, j, 1] = (ϕ[ip, j] - ϕ[i, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, jp] - ϕr0) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i, j] + ϕ[ip, j] + ϕr0 + ϕ[i, jp] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        elseif j == y1 && i == x2 #Bottom right corner.
            ∇ϕ[i, j, 1] = (ϕ0 - ϕ[i_, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, jp] - ϕ[i, j]) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i_, j] + ϕ0 + ϕ[i, j] + ϕ[i, jp] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        elseif j == y2 && i == x2 #Top right corner.
            ∇ϕ[i, j, 1] = (ϕ0 - ϕ[i_, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, j] - ϕ[i, jm]) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i_, j] + ϕ0 + ϕ[i, jm] + ϕ[i, j] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        elseif j == y2 && i == x1 #Top left corner.
            ∇ϕ[i, j, 1] = (ϕ[ip, j] - ϕ[i, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, j] - ϕ[i, jm]) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i, j] + ϕ[ip, j] + ϕ[i, jm] + ϕ[i, j] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        elseif j == y1 && i == xm + 1 #Bottom MT corner.
            ∇ϕ[i, j, 1] = (ϕ[ip, j] - ϕ[i, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, jp] - ϕ[i, j]) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i, j] + ϕ[ip, j] + ϕ[i, j] + ϕ[i, jp] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        elseif j == y2 #Top boundary.
            ∇ϕ[i, j, 1] = (ϕ[ip, j] - ϕ[i_, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, j] - ϕ[i, jm]) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i_, j] + ϕ[ip, j] + ϕ[i, jm] + ϕ[i, j] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        elseif i == x1 #Left boundary.
            ∇ϕ[i, j, 1] = (ϕ[ip, j] - ϕ[i, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, jp] - ϕ[i, jm]) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i, j] + ϕ[ip, j] + ϕ[i, jm] + ϕ[i, jp] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        elseif i == xm + 1 && j < ym + 1 #Face of MT.
            ϕr0 = ϕ[i, j] + atip
            ∇ϕ[i, j, 1] = (ϕ[ip, j] - ϕr0) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, jp] - ϕ[i, jm]) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕr0 + ϕ[ip, j] + ϕ[i, jm] + ϕ[i, jp] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        elseif i == x2 #Right boundary.
            ∇ϕ[i, j, 1] = (ϕ0 - ϕ[i_, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, jp] - ϕ[i, jm]) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i_, j] + ϕ0 + ϕ[i, jm] + ϕ[i, jp] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        elseif j == ym + 1 && i < xm + 1 #MT boundary.
            distx = xm - i
            if i <= lzone0
                ϕr0 = ϕ[i, j]
            elseif distx < lzone
                ϕr0 = ϕ[i, j] + a
            else
                if tail == 0
                    ϕr0 = ϕ[i, j] + a0
                else
                    ϕr0 = ϕ[i, j] + (a - a0) * CUDA.exp((i * dx - xm + lzone) / tail) + a0
                end
            end
            ∇ϕ[i, j, 1] = (ϕ[ip, j] - ϕ[i_, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, jp] - ϕr0) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i_, j] + ϕ[ip, j] + ϕr0 + ϕ[i, jp] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
        elseif j == y1 && i > xm + 1 #Bottom boundary.
            ∇ϕ[i, j, 1] = (ϕ[ip, j] - ϕ[i_, j]) / (2 * dx) #∂xϕ
            ∇ϕ[i, j, 2] = (ϕ[i, jp] - ϕ[i, j]) / (2 * dx) #∂zϕ
            Δϕ[i, j] = (ϕ[i_, j] + ϕ[ip, j] + ϕ[i, j] + ϕ[i, jp] - 4 * ϕ[i, j]) / dxdx + ∇ϕ[i, j, 2] / r
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
function kernel_diffusion!(ϕn, ϕ, ϕ0, ϕa, ϕb, μ, β, k, M, dt, dx, x1, x2, y1, y2, xm, ym, is_in)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    #i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    #jm = mod(j-1,1:Nr); jp = mod(j+1,1:Nr)
    i_ = i - 1
    ip = i + 1
    jm = j - 1
    jp = j + 1

    r = (j - 0.5) * dx
    dxdx = dx * dx
    @inbounds begin

        if j == ym + 1 && i == x1 #bottom left corner.
            Δμ = (μ[i, j] + μ[i, j] - 4 * μ[i, j] + μ[ip, j] + μ[i, jp]) / dxdx + (μ[i, jp] - μ[i, j]) / (2 * r * dx)
        elseif j == y1 && i == x2 #bottom right corner.
            μ0 = 4 * β * (ϕ0 - ϕa) * (ϕ0 - ϕb) * (ϕ0 - (ϕa + ϕb) * 0.5) - k * (ϕ[i, j] - ϕ0) #Chemical potential just outside domain.
            Δμ = (μ[i_, j] + μ[i, j] - 4 * μ[i, j] + μ0 + μ[i, jp]) / dxdx + (μ[i, jp] - μ[i, j]) / (2 * r * dx)
        elseif j == y2 && i == x2 #top right corner.
            μ0 = 4 * β * (ϕ0 - ϕa) * (ϕ0 - ϕb) * (ϕ0 - (ϕa + ϕb) * 0.5) - k * (ϕ[i, j] - ϕ0) #Chemical potential just outside domain.
            Δμ = (μ[i_, j] + μ[i, jm] - 4 * μ[i, j] + μ0 + μ[i, j]) / dxdx + (μ[i, j] - μ[i, jm]) / (2 * r * dx)
        elseif j == y2 && i == x1 #top left corner.
            Δμ = (μ[i, j] + μ[i, jm] - 4 * μ[i, j] + μ[ip, j] + μ[i, j]) / dxdx + (μ[i, j] - μ[i, jm]) / (2 * r * dx)
        elseif j == y1 && i == xm + 1 #Bottom MT corner.
            Δμ = (μ[i, j] + μ[i, j] - 4 * μ[i, j] + μ[ip, j] + μ[i, jp]) / dxdx + (μ[i, jp] - μ[i, j]) / (2 * r * dx)
        elseif j == y2 #top.
            Δμ = (μ[i_, j] + μ[i, jm] - 4 * μ[i, j] + μ[ip, j] + μ[i, j]) / dxdx + (μ[i, j] - μ[i, jm]) / (2 * r * dx)
        elseif i == x1 #left
            Δμ = (μ[i, j] + μ[i, jm] - 4 * μ[i, j] + μ[ip, j] + μ[i, jp]) / dxdx + (μ[i, jp] - μ[i, jm]) / (2 * r * dx)
        elseif j < ym + 1 && i == xm + 1 #Face of MT
            Δμ = (μ[i, j] + μ[i, jm] - 4 * μ[i, j] + μ[ip, j] + μ[i, jp]) / dxdx + (μ[i, jp] - μ[i, jm]) / (2 * r * dx)
        elseif i == x2 #right
            μ0 = 4 * β * (ϕ0 - ϕa) * (ϕ0 - ϕb) * (ϕ0 - (ϕa + ϕb) * 0.5) - k * (ϕ[i, j] - ϕ0) #Chemical potential just outside domain.
            Δμ = (μ[i_, j] + μ[i, jm] - 4 * μ[i, j] + μ0 + μ[i, jp]) / dxdx + (μ[i, jp] - μ[i, jm]) / (2 * r * dx)
        elseif j == y1 && i > xm + 1 #bottom.
            Δμ = (μ[i_, j] + μ[i, j] - 4 * μ[i, j] + μ[ip, j] + μ[i, jp]) / dxdx + (μ[i, jp] - μ[i, j]) / (2 * r * dx)
        elseif j == ym + 1 && i < xm + 1 #MT
            Δμ = (μ[i_, j] + μ[i, j] - 4 * μ[i, j] + μ[ip, j] + μ[i, jp]) / dxdx + (μ[i, jp] - μ[i, j]) / (2 * r * dx)
        elseif is_in[i, j]
            Δμ = (μ[i_, j] + μ[i, jm] - 4 * μ[i, j] + μ[ip, j] + μ[i, jp]) / dxdx + (μ[i, jp] - μ[i, jm]) / (2 * r * dx)
        end
        if is_in[i, j]
            ϕn[i, j] = ϕ[i, j] + dt * (M * Δμ)
        end
    end
    return nothing
end

function kernel_displacement!(ϕn, ϕ, ϕ0, is_in, x2, cutpoint)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    ip = i + cutpoint

    @inbounds begin
        if is_in[i, j]
            if x2 - i < cutpoint #right
                ϕn[i, j] = ϕ0
            else
                ϕn[i, j] = ϕ[ip, j]
            end
        end
    end
    return nothing
end

function kernel_interface_tracking!(ϕ, interface, Nr, xdel)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds begin
        if i <= xdel
            for j = 1:Nr
                if ϕ[i, j] <= 0.5
                    interface[i] = j
                    break
                end
            end
        end
    end
    return nothing
end

#   Initialize
#   ============
function kernel_create_boundary!(is_in, xm, ym)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    #= if i > x2 || i < x1 || j > z2 || j < z1 
        is_in[i,j] = false
    end =#
    if i < xm + 1 && j < ym + 1
        is_in[i, j] = false
    else
        is_in[i, j] = true
    end

    return nothing
end

function kernel_phi_init!(ϕ, ϕ0, is_in)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    @inbounds begin
        if is_in[i, j]
            ϕ[i, j] = ϕ0
        else
            ϕ[i, j] = 0.0
        end
    end

    return nothing
end

#   Main
#   ====

# dird = string(dir,"Data/")
dird = string(file, "Data/")
isdir(dird) ? rm(dird, recursive=true) : nothing
mkpath(dir)
mkpath(dird)

is_in = CUDA.ones(Bool, Nx, Nr)

ϕ = CUDA.zeros(Tf, Nx, Nr)
ϕ_temp = CUDA.zeros(Tf, Nx, Nr)
∇ϕ = CUDA.zeros(Tf, Nx, Nr, 2)
Δϕ = CUDA.zeros(Tf, Nx, Nr)

μ = CUDA.zeros(Tf, Nx, Nr)

interface = CUDA.zeros(Ti, xdel)

global xm = xm0
global ψ = 0.0
global cutpoint = xdel

#CPU arrays for saving to disk.
ϕw = zeros(Tf, Nx, Nr)

#Kernels declaration
gpukernel_comp_derivative = @cuda launch = false kernel_comp_derivative!(Δϕ, ∇ϕ, ϕ, ϕ0, a, a0, atip, lzone, tail, lzone0, dx, xl, xr, yb, yt, xm, ym)
gpukernel_comp_μ = @cuda launch = false kernel_comp_μ!(μ, ϕ, Δϕ, ϕa, ϕb, β, k)
gpukernel_diffusion = @cuda launch = false kernel_diffusion!(ϕ_temp, ϕ, ϕ0, ϕa, ϕb, μ, β, k, M, dt, dx, xl, xr, yb, yt, xm, ym, is_in)
gpukernel_create_boundary = @cuda launch = false kernel_create_boundary!(is_in, xm, ym)
gpukernel_phi_init = @cuda launch = false kernel_phi_init!(ϕ, ϕ0, is_in)
gpukernel_displacement = @cuda launch = false kernel_displacement!(ϕ_temp, ϕ, ϕ0, is_in, xr, cutpoint)
gpukernel_interface_tracking = @cuda launch = false kernel_interface_tracking!(ϕ, interface, Nr, xdel)

# Initialization
gpukernel_create_boundary(is_in, xm, ym; threads=a2D_block, blocks=a2D_grid)
gpukernel_phi_init(ϕ, ϕ0, is_in; threads=a2D_block, blocks=a2D_grid)
#pert = CUDA.rand(Tf, Nx, Nr)
#@. ϕ *= (1 + (pert - 0.5) * 1e-2)

gpukernel_comp_derivative(Δϕ, ∇ϕ, ϕ, ϕ0, a, a0, atip, lzone, tail, lzone0, dx, xl, xr, yb, yt, xm, ym; threads=a2D_block, blocks=a2D_grid)
gpukernel_comp_μ(μ, ϕ, Δϕ, ϕa, ϕb, β, k; threads=a2D_block, blocks=a2D_grid)

#save(string(dird, "data_", @sprintf("%08i", 0), ".jld"), "PF", Array(ϕ), "tip", xm)
GC.gc()

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
            gpukernel_comp_derivative(Δϕ, ∇ϕ, ϕ, ϕ0, a, a0, atip, lzone, tail, lzone0, dx, xl, xr, yb, yt, xm, ym; threads=a2D_block, blocks=a2D_grid)
            gpukernel_comp_μ(μ, ϕ, Δϕ, ϕa, ϕb, β, k; threads=a2D_block, blocks=a2D_grid)
            gpukernel_diffusion(ϕ_temp, ϕ, ϕ0, ϕa, ϕb, μ, β, k, M, dtd, dx, xl, xr, yb, yt, xm, ym, is_in; threads=a2D_block, blocks=a2D_grid)
            copyto!(ϕ, ϕ_temp)
            if any(isnan, ϕ)
                println("t=", t, " and idt = ", i, " and ϕ is NaN after diffusion")
                println("Job is over, error")
                return 1
            end
        end

        #Discrete MT growth update.
        ψ += dt * k0
        if ψ >= 1.0
            global xm += 1
            global ψ -= 1.0
            gpukernel_create_boundary(is_in, xm, ym; threads=a2D_block, blocks=a2D_grid)
            @. @views ϕ[xm, 1:ym] = 0.0
            @. @views ϕ_temp[xm, 1:ym] = 0.0
            if xm == Nx - xlim
                println("t= ", t, ", xm= ", xm, " and Displacement !!!!!!!!!!!!!")
                #Determine the portion to delete.
                #->track interface in the back zone.
                gpukernel_interface_tracking(ϕ, interface, Nr, xdel; threads=256, blocks=Block_Inter)
                #-> find first point at 0 from the zone border away from the wall.
                global cutpoint = findlast(x -> x > 4, Array(interface))
                if isnothing(cutpoint)
                    #-> if no point found -> film -> delete up to that point.
                    global cutpoint = xdel
                end
                println("cutpoint= ", cutpoint, " !!!!!!!!!!!!!!!!!!!!!!!")
                #displacement.
                global xm = xm - cutpoint
                println("xm after displacement: ", xm, " !!!!!!!!!!!!!")
                gpukernel_create_boundary(is_in, xm, ym; threads=a2D_block, blocks=a2D_grid)
                gpukernel_displacement(ϕ_temp, ϕ, ϕ0, is_in, xr, cutpoint; threads=a2D_block, blocks=a2D_grid)
                copyto!(ϕ, ϕ_temp)
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
            save(string(dird, "data_", @sprintf("%08i", t), ".jld"), "PF", ϕw, "tip", xm)
            GC.gc()
            CUDA.memory_status()
        end

    end
end
