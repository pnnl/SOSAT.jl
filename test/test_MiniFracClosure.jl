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
beta = 0.0
gamma = deg2rad(70.0)
Sv = 1.1
SHmax = 0.9
SHmin = 0.8
Pp = 0.41
so = SOSAT.StressState(Sv, SHmax, SHmin, Pp, alpha, beta, gamma)

mudist = Distributions.truncated(Distributions.LogNormal(-0.53,0.1),0.1,0.9)
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8)

mfc_dist = Distributions.truncated(Distributions.Normal(1.0,0.05),0.3,2.0)
mfc_constraint = SOSAT.MiniFracClosure(mfc_dist)
indicators = Vector{SOSAT.StressIndicator}(undef, 2)
indicators[1] = fc
indicators[2] = mfc_constraint

# perturb SHmax and SHmin but not Sv (which is S1)
# don't perturb pore pressure or Euler angles
stress_std = [0.0,0.5,0.5,0.0]
#angle_std = [deg2rad(2.0),deg2rad(2.0),deg2rad(0.0)]
angle_std = [deg2rad(0.0),deg2rad(0.0),deg2rad(0.0)]
Nsample = 1000000
#Nsample = 10
Nburn = 10000
Nthin = 2

sampler = SOSAT.MCMCSampler(so,
                        indicators,
                        stress_std,
                        angle_std)

println("Starting MCMC sampling...")
samples, acceptance_ratio = SOSAT.Sample(sampler,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true)

Smin_samples = SOSAT.GetSminSamples(samples)
@test all(Smin_samples .>= Pp)

Smin_median, Smin_range, Smin_IQR = SOSAT.GetStressComponentStatistics(samples, :Smin)
println("Smin median: ",Smin_median," range: ",Smin_range," IQR: ",Smin_IQR)

# the fault constraint biases Smin away from the mean imposed by the mini-frac closure
@test isapprox(Smin_median, 1.1; atol=0.1)
@test isapprox(Smin_range, 0.3; atol=0.1)

if GENERATE_PLOTS
    stress_poly_ax, stress_poly_fig = SOSAT.StressPolygonHist(samples, nbins=200, shmax_component=2, shmin_component=3)
    SOSAT.plot_stress_polygon(stress_poly_ax, Sv, Pp, 0.8; color="k")
    # plot the 1:1 line
    stress_poly_ax.plot([0.0,4.0],[0.0,4.0], color="gray", linestyle="--")
    stress_poly_ax.set_xlim(0.0, 4.0)
    stress_poly_ax.set_ylim(0.0, 4.0)
    stress_poly_ax.set_xlabel("SHmin")
    stress_poly_ax.set_ylabel("SHmax")
    stress_poly_fig.savefig("MiniFrac_constraint_test_polygon.png")
end