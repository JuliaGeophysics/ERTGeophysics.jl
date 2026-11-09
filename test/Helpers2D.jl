# Small surveys, meshes and models shared by the tests

const ORIGIN = [62.25, 25.75]

# n equally spaced electrodes centred on y = 0
line(n, a) = collect(range(-(n - 1) * a / 2, (n - 1) * a / 2, length = n))

survey(n = 16, a = 5.0; quadrupoles = ert2d_dipole_dipole(n; a_max = 2, n_max = 4), azimuth = 90.0) =
    BuildSurvey2D(origin = ORIGIN, azimuth = azimuth, positions = line(n, a), quadrupoles = quadrupoles)

# padded mesh for a survey: cps cells per spacing, layers a/div growing to depth, bottom padding
function test_mesh(data; cps = 2, div = 4.0, depth = 30.0, n_pad = 10, pad_factor = 1.5, surface = nothing)
    a = ert2d_spacing(data)
    layers = ert2d_geometric_layers(a, depth; first_layer_div = div)
    BuildMesh2D(electrodes = data.electrodes, y_core_cell = a / cps, n_pad = n_pad, pad_factor = pad_factor,
                layers = vcat(layers, ert2d_padding(layers[end], 10, pad_factor)), surface = surface)
end

# surface potential of a unit current on a two-layer earth (image series)
function two_layer_rhoa(d, ρ1, ρ2, h)
    κ = (ρ2 - ρ1) / (ρ2 + ρ1)
    V(r) = ρ1 / (2π) * (1 / r + 2 * sum(κ^n / sqrt(r^2 + (2n * h)^2) for n in 1:400))
    pot(e, s) = (e == 0 || s == 0) ? 0.0 : V(abs(d.electrodes[e] - d.electrodes[s]))
    R = [pot(d.m[i], d.a[i]) - pot(d.m[i], d.b[i]) - pot(d.n[i], d.a[i]) + pot(d.n[i], d.b[i]) for i in eachindex(d.a)]
    R .* ert2d_geometric_factor(d)
end

# synthetic observed data of a model on a mesh: R = ρa / K with relative noise (forward finer than the default)
function synthetic(mesh, ρ, data; noise = 0.03, seed = 1, refine = 12)
    K = ert2d_geometric_factor(data)
    r = ForwardSolve2D(mesh, ρ, data; refine)
    R = r.rhoa ./ K .* (1 .+ noise .* randn(MersenneTwister(seed), length(K)))
    DataFile2D(title = "test", electrode_names = data.electrode_names, latitudes = data.latitudes,
               longitudes = data.longitudes, x_positions = data.x_positions, electrodes = data.electrodes,
               z_positions = data.z_positions, a = data.a, b = data.b, m = data.m, n = data.n, resistance = R,
               resistance_error = max(noise, 0.01) .* abs.(R), geometric_factor = K, origin = data.origin,
               azimuth = data.azimuth)
end

block_model(mesh; host = 100.0, blocks = [(y_range = (-15.0, -5.0), z_range = (2.0, 8.0), resistivity = 10.0),
                                          (y_range = (5.0, 18.0), z_range = (2.0, 7.0), resistivity = 1000.0)]) =
    ert2d_block_model(mesh; background_resistivity = host, blocks)
