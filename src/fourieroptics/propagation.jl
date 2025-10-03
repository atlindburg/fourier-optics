using FFTW

function angular_spectrum_1D(E0::Vector{ComplexF64}, dx::Float64, wavelength::Float64, z::Float64)
    N = length(E0)
    FX = [(i <= N÷2 ? i-1 : i-1-N) / (N*dx) for i in 1:N]   # proper FFT frequencies
    k = 2*pi / wavelength
    kz = sqrt.(k^2 .- (2*pi*FX).^2 .+ 0im)                  # avoid NaNs for evanescent
    H = exp.(1im * kz * z)
    U0 = fft(E0)
    return ifft(U0 .* H)
end

function z_steps_julia(E0::Vector{ComplexF64}, dx::Float64, wavelength::Float64, z_steps::Vector{Float64})
    Nx = length(E0)
    Nz = length(z_steps)
    intensity_map = zeros(Float64, Nx, Nz)
    I0 = maximum(abs.(E0).^2)   # normalize to initial field max

    for j in 1:Nz
        E_z = angular_spectrum_1D(E0, dx, wavelength, z_steps[j])
        intensity_map[:, j] = abs.(E_z).^2 ./ I0
    end

    return intensity_map
end
