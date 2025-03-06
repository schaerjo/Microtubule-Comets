include("InputParametersClusterArray.jl")

println(Dates.now())

####Simulation######################################################################################################

#   Collide and stream
#   ====================
function kernel_collide!(fn, f, F, F_axi, ρ_, v, is_in, ω, w, ξ)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    @inbounds begin
        # Density and velocity
        ρ = ρ_[i, j]
        vx = v[i, j, 1]
        vr = v[i, j, 2]
        v2 = (vx * vx + vr * vr)
        # Source term factor
        ω_ = (1 - 0.5 * ω)
        Fx = ω_ * F[i, j, 1]
        Fr = ω_ * F[i, j, 2]
        #cylindrical terms
        θ = -ρ_[i, j] * vr / (9 * (j - 0.5))
        Fx_axi = F_axi[i, j, 1]
        Fr_axi = F_axi[i, j, 2]
        if is_in[i, j]
            for α = 1:9
                fn[i, j, α] = (1.0 - ω) * f[i, j, α] + ω * (ρ * w[α] * (1 + 3 * (ξ[α, 1] * vx + ξ[α, 2] * vr) - 1.5 * v2 + 4.5 * (ξ[α, 1] * vx + ξ[α, 2] * vr)^2)) + w[α] * (Fx * (3 * (ξ[α, 1] - vx) + 9 * ((ξ[α, 1] * vx + ξ[α, 2] * vr)) * ξ[α, 1]) + Fr * (3 * (ξ[α, 2] - vr) + 9 * ((ξ[α, 1] * vx + ξ[α, 2] * vr)) * ξ[α, 2])) + θ + (1 / 6) * (ξ[α, 1] * Fx_axi + ξ[α, 2] * Fr_axi)
            end
        else
            for α = 1:9
                fn[i, j, α] = 0.0 #ρ0*w[α]
            end
        end
    end
    return nothing
end


# Stream 2D:
function kernel_stream!(f, fpc, ξ, Nx, Nr)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    for α = 1:9
        ii = mod(i + ξ[α, 1], 1:Nx)
        jj = mod(j + ξ[α, 2], 1:Nr)
        f[ii, jj, α] = fpc[i, j, α]
    end

    return nothing
end

function kernel_apply_BB!(f, fpc, x1, x2, y1, y2, xm, ym)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i_ = i-1; ip = i+1
    if j==ym+1 && i==x1 #Bottom left corner
        f[i,j,3] = fpc[i,j,5]
        f[i,j,6] = fpc[i,j,8]
        f[i,j,2] = fpc[i,j,4]
        f[i,j,7] = fpc[i,j,9]
        f[i,j,9] = fpc[i,j,7]
    elseif j==y1 && i==x2 #Bottom right corner
        f[i,j,3] = fpc[i,j,5]
        f[i,j,7] = fpc[i,j,9]
        f[i,j,4] = fpc[i,j,2]
        f[i,j,6] = fpc[i_,j,9]
        f[i,j,8] = fpc[i,j,6]
    elseif j==y2 && i==x2 #Top right corner
        f[i,j,5] = fpc[i,j,3]
        f[i,j,8] = fpc[i,j,6]
        f[i,j,4] = fpc[i,j,2]
        f[i,j,7] = fpc[i,j,9]
        f[i,j,9] = fpc[i,j,7]
    elseif j==y2 && i==x1 #Top Left corner
        f[i,j,5] = fpc[i,j,3]
        f[i,j,9] = fpc[i,j,7]
        f[i,j,2] = fpc[i,j,4]
        f[i,j,6] = fpc[i,j,8]
        f[i,j,8] = fpc[i,j,6]
    elseif j==y1 && i==xm+1 #Bottom MT corner
        f[i,j,3] = fpc[i,j,5]
        f[i,j,6] = fpc[i,j,8]
        f[i,j,2] = fpc[i,j,4]
        f[i,j,7] = fpc[ip,j,8]
        f[i,j,9] = fpc[i,j,7]
    elseif j==ym+1 && i==xm+1 #Top MT corner
        f[i,j,6] = fpc[i,j,8]
    elseif j==y1 && i>xm+1 #Bottom boundary
        f[i,j,3] = fpc[i,j,5]
        f[i,j,6] = fpc[i_,j,9]
        f[i,j,7] = fpc[ip,j,8]
    elseif j==ym+1 && i<xm+1 #MT boundary
        f[i,j,3] = fpc[i,j,5]
        f[i,j,6] = fpc[i,j,8]
        f[i,j,7] = fpc[i,j,9]
    elseif j==y2 #Top boundary
        f[i,j,5] = fpc[i,j,3]
        f[i,j,8] = fpc[i,j,6]
        f[i,j,9] = fpc[i,j,7]
    elseif i==x1 #Left boundary
        f[i,j,2] = fpc[i,j,4]
        f[i,j,6] = fpc[i,j,8]
        f[i,j,9] = fpc[i,j,7]
    elseif i==xm+1 && j<ym+1 #Face of MT.
        f[i,j,2] = fpc[i,j,4]
        f[i,j,6] = fpc[i,j,8]
        f[i,j,9] = fpc[i,j,7]
    elseif i==x2 #Right boundary
        f[i,j,4] = fpc[i,j,2]
        f[i,j,8] = fpc[i,j,6]
        f[i,j,7] = fpc[i,j,9]
    end
    return nothing
end

#= function kernel_apply_BB!(f, fpc, x1, x2, y1, y2, xm, ym)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i_ = i - 1
    ip = i + 1
    jm = j - 1
    jp = j + 1
    if j == ym + 1 && i == x1 #Bottom left corner
        f[i, j, 3] = fpc[i, j, 5]
        f[i, j, 6] = fpc[i, j, 8]
        f[i, j, 2] = fpc[i, j, 2]#
        f[i, j, 7] = fpc[i, j, 9]
        f[i, j, 9] = fpc[i, jp, 9]#
    elseif j == y1 && i == x2 #Bottom right corner
        f[i, j, 3] = fpc[i, j, 5]
        f[i, j, 7] = fpc[i, j, 9]
        f[i, j, 4] = fpc[i, j, 4]#
        f[i, j, 6] = fpc[i_, j, 9]
        f[i, j, 8] = fpc[i, jp, 8]#
    elseif j == y2 && i == x2 #Top right corner
        f[i, j, 5] = fpc[i, j, 3]
        f[i, j, 8] = fpc[i, j, 6]
        f[i, j, 4] = fpc[i, j, 4]#
        f[i, j, 7] = fpc[i, jm, 7]#
        f[i, j, 9] = fpc[i, j, 7]
    elseif j == y2 && i == x1 #Top Left corner
        f[i, j, 5] = fpc[i, j, 3]
        f[i, j, 9] = fpc[i, j, 7]
        f[i, j, 2] = fpc[i, j, 2]#
        f[i, j, 6] = fpc[i, jm, 6]#
        f[i, j, 8] = fpc[i, j, 6]
    elseif j == y1 && i == xm + 1 #Bottom MT corner
        f[i, j, 3] = fpc[i, j, 5]
        f[i, j, 6] = fpc[i, j, 8]
        f[i, j, 2] = fpc[i, j, 4]
        f[i, j, 7] = fpc[ip, j, 8]
        f[i, j, 9] = fpc[i, j, 7]
    elseif j == ym + 1 && i == xm + 1 #Top MT corner
        f[i, j, 6] = fpc[i, j, 8]
    elseif j == y1 && i > xm + 1 #Bottom boundary
        f[i, j, 3] = fpc[i, j, 5]
        f[i, j, 6] = fpc[i_, j, 9]
        f[i, j, 7] = fpc[ip, j, 8]
    elseif j == ym + 1 && i < xm + 1 #MT boundary
        f[i, j, 3] = fpc[i, j, 5]
        f[i, j, 6] = fpc[i, j, 8]
        f[i, j, 7] = fpc[i, j, 9]
    elseif j == y2 #Top boundary
        f[i, j, 5] = fpc[i, j, 3]
        f[i, j, 8] = fpc[i, j, 6]
        f[i, j, 9] = fpc[i, j, 7]
    elseif i == x1 #Left boundary
        f[i, j, 2] = fpc[i, j, 2]#
        f[i, j, 6] = fpc[i, jm, 6]#
        f[i, j, 9] = fpc[i, jp, 7]#
    elseif i == xm + 1 && j < ym + 1 #Face of MT.
        f[i, j, 2] = fpc[i, j, 4]
        f[i, j, 6] = fpc[i, j, 8]
        f[i, j, 9] = fpc[i, j, 7]
    elseif i == x2 #Right boundary
        f[i, j, 4] = fpc[i, j, 4]#
        f[i, j, 8] = fpc[i, jp, 8]#
        f[i, j, 7] = fpc[i, jm, 7]#
    end
    return nothing
end =#

#   Meso to macro
#   ===============
function kernel_comp_ρ_v!(ρ, v, f, ξ, F, is_in)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if !is_in[i, j]
        @inbounds ρ[i, j] = 0
        @inbounds v[i, j, 1] = 0.0
        v[i, j, 2] = 0.0
        return nothing
    end

    @inbounds ρ[i, j] = 0
    @inbounds v[i, j, 1] = 0.5 * F[i, j, 1]
    v[i, j, 2] = 0.5 * F[i, j, 2]

    @inbounds for α = 1:9
        ρ[i, j] += f[i, j, α]
        v[i, j, 1] += f[i, j, α] * ξ[α, 1]
        v[i, j, 2] += f[i, j, α] * ξ[α, 2]
    end

    @inbounds v[i, j, 1] *= 1 / ρ[i, j]
    @inbounds v[i, j, 2] *= 1 / ρ[i, j]

    return nothing
end

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

#   Force computation
#   =================
function kernel_comp_F!(F, ϕ, ϕa, ϕb, ϕ0, μ, β, k, a0, grav, x1, x2, y1, y2, xm, ym)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    #i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    #jm = mod(j-1,1:Nr); jp = mod(j+1,1:Nr)
    i_ = i - 1
    ip = i + 1
    jm = j - 1
    jp = j + 1

    r = (j - 0.5)

    Fx = 0.0
    Fr = 0.0
    μ0 = 0.0

    @inbounds begin
        if j == ym + 1 && i == x1 #Bottom left corner.
            Fx += -ϕ[i, j] * (μ[ip, j] - μ[i, j]) * 0.5
            Fr += -ϕ[i, j] * (μ[i, jp] - μ[i, j]) * 0.5
        elseif j == y1 && i == x2 #Bottom right corner.
            μ0 = 4 * β * (ϕ0 - ϕa) * (ϕ0 - ϕb) * (ϕ0 - (ϕa + ϕb) * 0.5) - k * (ϕ[i, j] - ϕ0) #Chemical potential just outside domain.
            Fx += -ϕ[i, j] * (μ0 - μ[i_, j]) * 0.5
            Fr += -ϕ[i, j] * (μ[i, jp] - μ[i, j]) * 0.5
        elseif j == y2 && i == x2 #Top right corner.
            μ0 = 4 * β * (ϕ0 - ϕa) * (ϕ0 - ϕb) * (ϕ0 - (ϕa + ϕb) * 0.5) - k * (ϕ[i, j] - ϕ0) #Chemical potential just outside domain.
            Fx += -ϕ[i, j] * (μ0 - μ[i_, j]) * 0.5
            Fr += -ϕ[i, j] * (μ[i, j] - μ[i, jm]) * 0.5
        elseif j == y2 && i == x1 #Top left corner.
            Fx += -ϕ[i, j] * (μ[ip, j] - μ[i, j]) * 0.5
            Fr += -ϕ[i, j] * (μ[i, j] - μ[i, jm]) * 0.5
        elseif j == y1 && i == xm + 1 #Bottom MT corner.
            Fx += -ϕ[i, j] * (μ[ip, j] - μ[i, j]) * 0.5
            Fr += -ϕ[i, j] * (μ[i, jp] - μ[i, j]) * 0.5
        elseif j == y1 && i > xm + 1 #Bottom.
            Fx += -ϕ[i, j] * (μ[ip, j] - μ[i_, j]) * 0.5
            Fr += -ϕ[i, j] * (μ[i, jp] - μ[i, j]) * 0.5
        elseif j == ym + 1 && i < xm + 1 #Bottom MT.
            Fx += -ϕ[i, j] * (μ[ip, j] - μ[i_, j]) * 0.5
            Fr += -ϕ[i, j] * (μ[i, jp] - μ[i, j]) * 0.5
        elseif i == x1 #Left.
            Fx += -ϕ[i, j] * (μ[ip, j] - μ[i, j]) * 0.5
            Fr += -ϕ[i, j] * (μ[i, jp] - μ[i, jm]) * 0.5
        elseif i == xm + 1 && j < ym + 1 #Face of MT.
            Fx += -ϕ[i, j] * (μ[ip, j] - μ[i, j]) * 0.5
            Fr += -ϕ[i, j] * (μ[i, jp] - μ[i, jm]) * 0.5
        elseif i == x2 #Right.
            μ0 = 4 * β * (ϕ0 - ϕa) * (ϕ0 - ϕb) * (ϕ0 - (ϕa + ϕb) * 0.5) - k * (ϕ[i, j] - ϕ0) #Chemical potential just outside domain.
            Fx += -ϕ[i, j] * (μ0 - μ[i_, j]) * 0.5
            Fr += -ϕ[i, j] * (μ[i, jp] - μ[i, jm]) * 0.5
        elseif j == y2 #Top.
            Fx += -ϕ[i, j] * (μ[ip, j] - μ[i_, j]) * 0.5
            Fr += -ϕ[i, j] * (μ[i, j] - μ[i, jm]) * 0.5
        else
            Fx += -ϕ[i, j] * (μ[ip, j] - μ[i_, j]) * 0.5
            Fr += -ϕ[i, j] * (μ[i, jp] - μ[i, jm]) * 0.5
        end
        F[i, j, 1] += Fx + grav#*ϕ[i,j]
        F[i, j, 2] += Fr
    end
    return nothing
end

function kernel_comp_F_axi!(F_axi, v, ρ_, f, ω, nu, x1, x2, xm, ym)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    #i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    i_ = i - 1
    ip = i + 1

    r = (j - 0.5)

    Fx = 0.0
    Fr = 0.0

    @inbounds begin
        #Density and velocity
        ρ = ρ_[i, j]
        vx = v[i, j, 1]
        vr = v[i, j, 2]
        vrvr = vr * vr
        vxvr = vx * vr

        #Computation with paper trick.

        if i == x1 #Left boundary.
            Fx = -ρ * vxvr / r - ((nu) / r) * (3 * ω * (f[i, j, 6] - f[i, j, 7] + f[i, j, 8] - f[i, j, 9] - ρ * vxvr) + (v[ip, j, 2] - v[i, j, 2]) / 2)
            Fr = -ρ * vrvr / r - ρ * nu * vr / (r * r) - ((nu) / r) * (1.5 * ω * (f[i, j, 3] - f[i, j, 5] + f[i, j, 6] + f[i, j, 7] - f[i, j, 8] - f[i, j, 9] - ρ * vr) + 0.5 * vr / r)
        elseif i == x2 #Right boundary.
            Fx = -ρ * vxvr / r - ((nu) / r) * (3 * ω * (f[i, j, 6] - f[i, j, 7] + f[i, j, 8] - f[i, j, 9] - ρ * vxvr) - (v[i_, j, 2]) / 2)
            Fr = -ρ * vrvr / r - ρ * nu * vr / (r * r) - ((nu) / r) * (1.5 * ω * (f[i, j, 3] - f[i, j, 5] + f[i, j, 6] + f[i, j, 7] - f[i, j, 8] - f[i, j, 9] - ρ * vr) + 0.5 * vr / r)
        elseif i == xm + 1 && j < ym + 1 #Face of MT.
            Fx = -ρ * vxvr / r - ((nu) / r) * (3 * ω * (f[i, j, 6] - f[i, j, 7] + f[i, j, 8] - f[i, j, 9] - ρ * vxvr) + (v[ip, j, 2]) / 2)
            Fr = -ρ * vrvr / r - ρ * nu * vr / (r * r) - ((nu) / r) * (1.5 * ω * (f[i, j, 3] - f[i, j, 5] + f[i, j, 6] + f[i, j, 7] - f[i, j, 8] - f[i, j, 9] - ρ * vr) + 0.5 * vr / r)
        else
            Fx = -ρ * vxvr / r - ((nu) / r) * (3 * ω * (f[i, j, 6] - f[i, j, 7] + f[i, j, 8] - f[i, j, 9] - ρ * vxvr) + (v[ip, j, 2] - v[i_, j, 2]) / 2)
            Fr = -ρ * vrvr / r - ρ * nu * vr / (r * r) - ((nu) / r) * (1.5 * ω * (f[i, j, 3] - f[i, j, 5] + f[i, j, 6] + f[i, j, 7] - f[i, j, 8] - f[i, j, 9] - ρ * vr) + 0.5 * vr / r)
        end

        F_axi[i, j, 1] = Fx
        F_axi[i, j, 2] = Fr
    end
    return nothing
end

#   Cahn_Hilliard
#   =============
function kernel_advection!(ϕn, ϕ, v, dt, dx, x1, x2, y1, y2, xm, ym, is_in)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    #i_ = mod(i-1,1:Nx); ip = mod(i+1,1:Nx)
    #jm = mod(j-1,1:Nr); jp = mod(j+1,1:Nr)
    i_ = i - 1
    ip = i + 1
    jm = j - 1
    jp = j + 1

    r = (j - 0.5) * dx
    ∂xϕvx = 0.0
    ∂rϕvr = 0.0
    @inbounds begin
        if j == ym + 1 && i == x1 #Bottom left corner.
            ∂xϕvx -= v[i, j, 1] > 0 ? v[i, j, 1] * ϕ[i, j] : 0.0
            ∂xϕvx -= v[ip, j, 1] < 0 ? ϕ[ip, j] * v[ip, j, 1] : 0.0

            ∂rϕvr += v[i, j, 2] > 0 ? -v[i, j, 2] * ϕ[i, j] - ϕ[i, j] * v[i, j, 2] / r : 0.0
            ∂rϕvr -= v[i, jp, 2] < 0 ? ϕ[i, jp] * v[i, jp, 2] : 0.0
        elseif j == y1 && i == x2 #Bottom right corner.
            ∂xϕvx += -abs(v[i, j, 1]) * ϕ[i, j]
            ∂xϕvx += v[i_, j, 1] > 0 ? ϕ[i_, j] * v[i_, j, 1] : 0.0

            ∂rϕvr += v[i, j, 2] > 0 ? -v[i, j, 2] * ϕ[i, j] - ϕ[i, j] * v[i, j, 2] / r : 0.0
            ∂rϕvr -= v[i, jp, 2] < 0 ? ϕ[i, jp] * v[i, jp, 2] : 0.0
        elseif j == y2 && i == x2 #Top Right corner.
            ∂xϕvx += -abs(v[i, j, 1]) * ϕ[i, j]
            ∂xϕvx += v[i_, j, 1] > 0 ? ϕ[i_, j] * v[i_, j, 1] : 0.0

            ∂rϕvr += v[i, j, 2] < 0 ? v[i, j, 2] * ϕ[i, j] + ϕ[i, j] * v[i, j, 2] / r : 0.0
            ∂rϕvr += v[i, jm, 2] > 0 ? ϕ[i, jm] * v[i, jm, 2] : 0.0
        elseif j == y2 && i == x1 #Top left corner.
            ∂xϕvx -= v[i, j, 1] > 0 ? v[i, j, 1] * ϕ[i, j] : 0.0
            ∂xϕvx -= v[ip, j, 1] < 0 ? ϕ[ip, j] * v[ip, j, 1] : 0.0

            ∂rϕvr += v[i, j, 2] < 0 ? v[i, j, 2] * ϕ[i, j] + ϕ[i, j] * v[i, j, 2] / r : 0.0
            ∂rϕvr += v[i, jm, 2] > 0 ? ϕ[i, jm] * v[i, jm, 2] : 0.0
        elseif j == y1 && i == xm + 1 #Bottom MT corner.
            ∂xϕvx -= v[i, j, 1] > 0 ? v[i, j, 1] * ϕ[i, j] : 0.0
            ∂xϕvx -= v[ip, j, 1] < 0 ? ϕ[ip, j] * v[ip, j, 1] : 0.0

            ∂rϕvr += v[i, j, 2] > 0 ? -v[i, j, 2] * ϕ[i, j] - ϕ[i, j] * v[i, j, 2] / r : 0.0
            ∂rϕvr -= v[i, jp, 2] < 0 ? ϕ[i, jp] * v[i, jp, 2] : 0.0
        elseif j == y2 #Top
            ∂xϕvx += -abs(v[i, j, 1]) * ϕ[i, j]
            ∂xϕvx += v[i_, j, 1] > 0 ? ϕ[i_, j] * v[i_, j, 1] : 0.0
            ∂xϕvx -= v[ip, j, 1] < 0 ? ϕ[ip, j] * v[ip, j, 1] : 0.0

            ∂rϕvr += v[i, j, 2] < 0 ? v[i, j, 2] * ϕ[i, j] + ϕ[i, j] * v[i, j, 2] / r : 0.0
            ∂rϕvr += v[i, jm, 2] > 0 ? ϕ[i, jm] * v[i, jm, 2] : 0.0
        elseif i == x1 #Left
            ∂xϕvx -= v[i, j, 1] > 0 ? v[i, j, 1] * ϕ[i, j] : 0.0
            ∂xϕvx -= v[ip, j, 1] < 0 ? ϕ[ip, j] * v[ip, j, 1] : 0.0

            ∂rϕvr += -abs(v[i, j, 2]) * ϕ[i, j] - ϕ[i, j] * v[i, j, 2] / r
            ∂rϕvr += v[i, jm, 2] > 0 ? ϕ[i, jm] * v[i, jm, 2] : 0.0
            ∂rϕvr -= v[i, jp, 2] < 0 ? ϕ[i, jp] * v[i, jp, 2] : 0.0
        elseif i == xm + 1 && j < ym + 1 #Face of MT
            ∂xϕvx -= v[i, j, 1] > 0 ? v[i, j, 1] * ϕ[i, j] : 0.0
            ∂xϕvx -= v[ip, j, 1] < 0 ? ϕ[ip, j] * v[ip, j, 1] : 0.0

            ∂rϕvr += -abs(v[i, j, 2]) * ϕ[i, j] - ϕ[i, j] * v[i, j, 2] / r
            ∂rϕvr += v[i, jm, 2] > 0 ? ϕ[i, jm] * v[i, jm, 2] : 0.0
            ∂rϕvr -= v[i, jp, 2] < 0 ? ϕ[i, jp] * v[i, jp, 2] : 0.0
        elseif i == x2 #Right.
            ∂xϕvx += -abs(v[i, j, 1]) * ϕ[i, j]
            ∂xϕvx += v[i_, j, 1] > 0 ? ϕ[i_, j] * v[i_, j, 1] : 0.0

            ∂rϕvr += -abs(v[i, j, 2]) * ϕ[i, j] - ϕ[i, j] * v[i, j, 2] / r
            ∂rϕvr += v[i, jm, 2] > 0 ? ϕ[i, jm] * v[i, jm, 2] : 0.0
            ∂rϕvr -= v[i, jp, 2] < 0 ? ϕ[i, jp] * v[i, jp, 2] : 0.0
        elseif j == y1 && i > xm + 1 #Bottom.
            ∂xϕvx += -abs(v[i, j, 1]) * ϕ[i, j]
            ∂xϕvx += v[i_, j, 1] > 0 ? ϕ[i_, j] * v[i_, j, 1] : 0.0
            ∂xϕvx -= v[ip, j, 1] < 0 ? ϕ[ip, j] * v[ip, j, 1] : 0.0

            ∂rϕvr += v[i, j, 2] > 0 ? -v[i, j, 2] * ϕ[i, j] - ϕ[i, j] * v[i, j, 2] / r : 0.0
            ∂rϕvr -= v[i, jp, 2] < 0 ? ϕ[i, jp] * v[i, jp, 2] : 0.0
        elseif j == ym + 1 && i < xm + 1 #Bottom MT
            ∂xϕvx += -abs(v[i, j, 1]) * ϕ[i, j]
            ∂xϕvx += v[i_, j, 1] > 0 ? ϕ[i_, j] * v[i_, j, 1] : 0.0
            ∂xϕvx -= v[ip, j, 1] < 0 ? ϕ[ip, j] * v[ip, j, 1] : 0.0

            ∂rϕvr += v[i, j, 2] > 0 ? -v[i, j, 2] * ϕ[i, j] - ϕ[i, j] * v[i, j, 2] / r : 0.0
            ∂rϕvr -= v[i, jp, 2] < 0 ? ϕ[i, jp] * v[i, jp, 2] : 0.0
        elseif is_in[i, j]
            ∂xϕvx += -abs(v[i, j, 1]) * ϕ[i, j]
            ∂xϕvx += v[i_, j, 1] > 0 ? ϕ[i_, j] * v[i_, j, 1] : 0.0
            ∂xϕvx -= v[ip, j, 1] < 0 ? ϕ[ip, j] * v[ip, j, 1] : 0.0

            ∂rϕvr += -abs(v[i, j, 2]) * ϕ[i, j] - ϕ[i, j] * v[i, j, 2] / r
            ∂rϕvr += v[i, jm, 2] > 0 ? ϕ[i, jm] * v[i, jm, 2] : 0.0
            ∂rϕvr -= v[i, jp, 2] < 0 ? ϕ[i, jp] * v[i, jp, 2] : 0.0
        end
        if is_in[i, j]
            ϕn[i, j] = ϕ[i, j] + dt * (∂xϕvx + ∂rϕvr)
        end
    end
    return nothing
end

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

function kernel_displacement!(ϕn, ϕ, ϕ0, ρn, ρ, ρ0, vn, v, fn, f, w, is_in, x2, cutpoint)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    ip = i + cutpoint

    @inbounds begin
        if is_in[i, j]
            if x2 - i < cutpoint #right
                ϕn[i, j] = ϕ0
                ρn[i, j] = ρ0
                vn[i, j, 1] = 0.0
                vn[i, j, 2] = 0.0
                for α = 1:9
                    fn[i, j, α] = ρ0 * w[α]
                end
            else
                ϕn[i, j] = ϕ[ip, j]
                ρn[i, j] = ρ[ip, j]
                vn[i, j, 1] = v[ip, j, 1]
                vn[i, j, 2] = v[ip, j, 2]
                for α = 1:9
                    fn[i, j, α] = f[ip, j, α] #ρ0*w[α]
                end
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
function kernel_initialize_pop!(f, w, ρ0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    for α in eachindex(w)
        f[i, j, α] = w[α] * ρ0
    end

    return nothing
end

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

f = CUDA.ones(Tf, Nx, Nr, Qlbm)
f_t = CUDA.zeros(Tf, Nx, Nr, Qlbm)
ρ = CUDA.ones(Tf, Nx, Nr) * ρ0
ρ_temp = CUDA.ones(Tf, Nx, Nr) * ρ0
is_in = CUDA.ones(Bool, Nx, Nr)
v = CUDA.zeros(Tf, Nx, Nr, 2)
v_temp = CUDA.zeros(Tf, Nx, Nr, 2)

ϕ = CUDA.zeros(Tf, Nx, Nr)
ϕ_temp = CUDA.zeros(Tf, Nx, Nr)
∇ϕ = CUDA.zeros(Tf, Nx, Nr, 2)
Δϕ = CUDA.zeros(Tf, Nx, Nr)

F = CUDA.zeros(Tf, Nx, Nr, 2)
μ = CUDA.zeros(Tf, Nx, Nr)

F_axi = CUDA.zeros(Tf, Nx, Nr, 2)

interface = CUDA.zeros(Ti, xdel)

global xm = xm0
global ψ = 0.0
global cutpoint = xdel

#CPU arrays for saving to disk.
ϕw = zeros(Tf, Nx, Nr)
ρw = ones(Tf, Nx, Nr) * ρ0
vw = zeros(Tf, Nx, Nr, 2)

#Kernels declaration
gpukernel_collide = @cuda launch = false kernel_collide!(f_t, f, F, F_axi, ρ, v, is_in, ω, w, ξ)
gpukernel_stream = @cuda launch = false kernel_stream!(f, f_t, ξ, Nx, Nr)
gpukernel_apply_BB = @cuda launch = false kernel_apply_BB!(f, f_t, xl, xr, yb, yt, xm, ym)
gpucomp_ρ_v = @cuda launch = false kernel_comp_ρ_v!(ρ, v, f, ξ, F, is_in)
gpukernel_comp_derivative = @cuda launch = false kernel_comp_derivative!(Δϕ, ∇ϕ, ϕ, ϕ0, a, a0, atip, lzone, tail, lzone0, dx, xl, xr, yb, yt, xm, ym)
gpukernel_comp_μ = @cuda launch = false kernel_comp_μ!(μ, ϕ, Δϕ, ϕa, ϕb, β, k)
gpukernel_comp_F = @cuda launch = false kernel_comp_F!(F, ϕ, ϕa, ϕb, ϕ0, μ, β, k, a0, grav, xl, xr, yb, yt, xm, ym)
gpukernel_comp_F_axi = @cuda launch = false kernel_comp_F_axi!(F_axi, v, ρ, f, ω, nu, xl, xr, xm, ym)
gpukernel_advection = @cuda launch = false kernel_advection!(ϕ_temp, ϕ, v, dt, dx, xl, xr, yb, yt, xm, ym, is_in)
gpukernel_diffusion = @cuda launch = false kernel_diffusion!(ϕ_temp, ϕ, ϕ0, ϕa, ϕb, μ, β, k, M, dt, dx, xl, xr, yb, yt, xm, ym, is_in)
gpukernel_initialize_pop = @cuda launch = false kernel_initialize_pop!(f, w, ρ0)
gpukernel_create_boundary = @cuda launch = false kernel_create_boundary!(is_in, xm, ym)
gpukernel_phi_init = @cuda launch = false kernel_phi_init!(ϕ, ϕ0, is_in)
gpukernel_displacement = @cuda launch = false kernel_displacement!(ϕ_temp, ϕ, ϕ0, ρ_temp, ρ, ρ0, v_temp, v, f_t, f, w, is_in, xr, cutpoint)
gpukernel_interface_tracking = @cuda launch = false kernel_interface_tracking!(ϕ, interface, Nr, xdel)

# Initialization
gpukernel_initialize_pop(f, w, ρ0; threads=a2D_block, blocks=a2D_grid)
gpukernel_create_boundary(is_in, xm, ym; threads=a2D_block, blocks=a2D_grid)
gpukernel_phi_init(ϕ, ϕ0, is_in; threads=a2D_block, blocks=a2D_grid)
#pert = CUDA.rand(Tf, Nx, Nr)
#@. ϕ *= (1 + (pert - 0.5) * 1e-2)

gpukernel_comp_derivative(Δϕ, ∇ϕ, ϕ, ϕ0, a, a0, atip, lzone, tail, lzone0, dx, xl, xr, yb, yt, xm, ym; threads=a2D_block, blocks=a2D_grid)
gpukernel_comp_μ(μ, ϕ, Δϕ, ϕa, ϕb, β, k; threads=a2D_block, blocks=a2D_grid)
gpukernel_comp_F(F, ϕ, ϕa, ϕb, ϕ0, μ, β, k, a0, grav, xl, xr, yb, yt, xm, ym; threads=a2D_block, blocks=a2D_grid)
gpukernel_comp_F_axi(F_axi, v, ρ, f, ω, nu, xl, xr, xm, ym; threads=a2D_block, blocks=a2D_grid)

#save(string(dird, "data_", @sprintf("%08i", 0), ".jld"), "density", Array(ρ), "velocity", Array(v), "PF", Array(ϕ), "tip", xm)
GC.gc()

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

    for t = 1:NΔt
        #Lattice Boltzmann for ρ and v.
        gpukernel_collide(f_t, f, F, F_axi, ρ, v, is_in, ω, w, ξ; threads=a2D_block, blocks=a2D_grid)
        gpukernel_stream(f, f_t, ξ, Nx, Nr; threads=a2D_block, blocks=a2D_grid)
        gpukernel_apply_BB(f, f_t, xl, xr, yb, yt, xm, ym; threads=a2D_block, blocks=a2D_grid)

        #Advection
        gpukernel_advection(ϕ_temp, ϕ, v, dt, dx, xl, xr, yb, yt, xm, ym, is_in; threads=a2D_block, blocks=a2D_grid)
        copyto!(ϕ, ϕ_temp)
        if any(isnan, ϕ)
            println("t=", t, " and ϕ is NaN after advection")
            println("Job is over, error")
            return 1
        end
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

        gpucomp_ρ_v(ρ, v, f, ξ, F, is_in; threads=a2D_block, blocks=a2D_grid)

        #Discrete MT growth update.
        #= ψ += dt*k0
        if ψ >= 1.0
            global ψ -= 1.0
            global ψ2 += 1
            gpukernel_displacement(ϕ_temp, ϕ, ϕ0, ρ_temp, ρ, ρ0, v_temp, v, f_t, f, w, is_in, xr; threads = a2D_block, blocks = a2D_grid)
            copyto!(ϕ, ϕ_temp)
            copyto!(ρ, ρ_temp)
            copyto!(v, v_temp)
            copyto!(f, f_t)
            if ψ2 == ψlim
                global ψ2 = 0
            end
        end =#
        ψ += dt * k0
        if ψ >= 1.0
            global xm += 1
            global ψ -= 1.0
            gpukernel_create_boundary(is_in, xm, ym; threads=a2D_block, blocks=a2D_grid)
            @. @views ϕ[xm, 1:ym] = 0.0
            @. @views ϕ_temp[xm, 1:ym] = 0.0
            @. @views ρ[xm, 1:ym] = 0.0
            @. @views v[xm, 1:ym, :] = 0.0
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
                gpukernel_displacement(ϕ_temp, ϕ, ϕ0, ρ_temp, ρ, ρ0, v_temp, v, f_t, f, w, is_in, xr, cutpoint; threads=a2D_block, blocks=a2D_grid)
                copyto!(ϕ, ϕ_temp)
                copyto!(ρ, ρ_temp)
                copyto!(v, v_temp)
                copyto!(f, f_t)
            end
        end

        @. F_axi = 0.0
        gpukernel_comp_F_axi(F_axi, v, ρ, f, ω, nu, xl, xr, xm, ym; threads=a2D_block, blocks=a2D_grid)
        @. F = 0.0
        gpukernel_comp_derivative(Δϕ, ∇ϕ, ϕ, ϕ0, a, a0, atip, lzone, tail, lzone0, dx, xl, xr, yb, yt, xm, ym; threads=a2D_block, blocks=a2D_grid)
        gpukernel_comp_μ(μ, ϕ, Δϕ, ϕa, ϕb, β, k; threads=a2D_block, blocks=a2D_grid)
        gpukernel_comp_F(F, ϕ, ϕa, ϕb, ϕ0, μ, β, k, a0, grav, xl, xr, yb, yt, xm, ym; threads=a2D_block, blocks=a2D_grid)

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
            copyto!(ρw, ρ)
            copyto!(vw, v)
            save(string(dird, "data_", @sprintf("%08i", t), ".jld"), "density", ρw, "velocity", vw, "PF", ϕw, "tip", xm)
            GC.gc()
            CUDA.memory_status()
        end

    end
end
