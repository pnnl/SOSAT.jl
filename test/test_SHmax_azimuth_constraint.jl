using SOSAT
using Test
using Random
using Distributions
using PythonPlot
const plt = PythonPlot

Random.seed!(1234)
plt.matplotlib.use("Agg")

const GENERATE_PLOTS = haskey(ENV, "JULIA_TEST_PLOTS") || 
                       get(ENV, "JULIA_TEST_PLOTS", "false") == "true"

# test with normal faulting and vertical/horizontal alignment
alpha = 0.0
beta = deg2rad(90.0)
gamma = deg2rad(-70.0)
Sv = 24.5
SHmax = 19.6
SHmin = 13.8
Pp = 9.47
so = SOSAT.StressState(Sv, SHmax, SHmin, Pp, alpha, beta, gamma)

shmax_mean = deg2rad(20.0)
shmax_kappa = 1.0/deg2rad(10.0)^2
SHmax_azimuth_dist = Distributions.VonMises(shmax_mean, shmax_kappa)
shmax_azimuth_indicator = SOSAT.SHmaxAzimuthConstraint(SHmax_azimuth_dist)

# for testing we limit Aphi to 2.0 so all stress states are either
# normal faulting or strike slip. This will disallow the minimum horizontal
# stress from exceeding the vertical stress

mudist = Distributions.truncated(Distributions.LogNormal(-0.53,0.1),0.1,0.9)
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8)

indicators = Vector{SOSAT.StressIndicator}(undef, 2)
indicators[1] = fc
indicators[2] = shmax_azimuth_indicator

# perturb SHmax and SHmin but not Sv (which is S1)
# don't perturb pore pressure or Euler angles
stress_std = [0.0,0.5,0.5,0.0]

angle_std = [deg2rad(0.0),deg2rad(0.0),deg2rad(1.0)]
Nsample = 1000000
#Nsample = 10
Nburn = 10000
Nthin = 2

sampler = SOSAT.MCMCSampler(so,
                        indicators,
                        stress_std,
                        angle_std)
initial_likelihood = SOSAT.EvaluateInitialLikelihood(sampler, verbose=true)
println("Initial likelihood = ",initial_likelihood)

println("Starting MCMC sampling...")
samples, acceptance_ratio = SOSAT.Sample(sampler,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true)

# analyze the results
SHmin_samples = SOSAT.GetSHminSamples(samples)
SHmin_max = maximum(SHmin_samples)
println("Max SHmin sample: ",SHmin_max)
SHmin_min = minimum(SHmin_samples)
println("Min SHmin sample: ",SHmin_min)

SHmin_median, SHmin_range, SHmin_IQR = SOSAT.GetStressComponentStatistics(samples, :SHmin)
println("SHmin median: ",SHmin_median," range: ",SHmin_range," IQR: ",SHmin_IQR)

__, __, __, __, SHmax_dir, __ = GeographicallyOrderStresses(samples)

if GENERATE_PLOTS
    # plot the stress samples in the stress polygon space
    stress_poly_ax, stress_poly_fig = SOSAT.StressPolygonHist(samples, nbins=200, shmax_component=2, shmin_component=3)
    SOSAT.plot_stress_polygon(stress_poly_ax, Sv, Pp, 0.8; color="k")
    # plot the 1:1 line
    stress_poly_ax.plot([0.0,4.0],[0.0,4.0], color="gray", linestyle="--")
    stress_poly_ax.set_xlim(0.0, 4.0)
    stress_poly_ax.set_ylim(0.0, 4.0)
    stress_poly_ax.set_xlabel("SHmin")
    stress_poly_ax.set_ylabel("SHmax")
    stress_poly_fig.savefig("SHmax_azimuth_constraint_test_polygon.png")

    stereonet_fig = SOSAT.PrincipalStressStereonet(samples)
    savefig(joinpath("SHmax_azimuth_constraint_stereonet.png"), dpi=300)
    close(stereonet_fig)

    SHmax_azimuth_deg = rad2deg.(atan.(SHmax_dir[:, 2], SHmax_dir[:, 1]))
    println("First five SHmax azimuths (degrees): ", SHmax_azimuth_deg[1:5])
    
    SHmax_az_hist_fig = plt.figure(figsize=(6,4))
    shmax_az_ax = SHmax_az_hist_fig.add_subplot(1,1,1)
    shmax_az_ax.hist(SHmax_azimuth_deg, bins=30, density=true, color="blue", alpha=0.7)
    shmax_az_ax.set_xlabel("SHmax Azimuth (degrees)")
    SHmax_az_hist_fig.savefig(joinpath("SHmax_azimuth_constraint_shmax_dir_hist.png"), dpi=300)
    close(SHmax_az_hist_fig)

end
