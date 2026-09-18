using SOSAT
using Test
using Random
using Distributions
using PythonPlot

const plt = PythonPlot
Random.seed!(1234)
plt.matplotlib.use("Agg")


"""
This test uses the MCMC sampler with the frictional faulting constraint
and a prior constraint on the Euler angle to "encourage" a vertical/horizontal orientation
"""
const GENERATE_PLOTS = haskey(ENV, "JULIA_TEST_PLOTS") || 
                       get(ENV, "JULIA_TEST_PLOTS", "false") == "true"
TVD_m = 3000.0

avg_overburden_density = 2300.0 # kg/m3
g = 9.81 # m/s2
sigv = avg_overburden_density * g * TVD_m / 1.0e6 # MPa
println("Vertical Stress is ",sigv)

pp_MPa = TVD_m * 10.0 / 1000.0 # assume 10 MPa/km
println("Pore pressure is ",pp_MPa)

#----------------------------
# Fault Friction distribution
#----------------------------
mudist = Distributions.truncated(Distributions.LogNormal(-0.53,0.1),0.1,0.9)

# create the fault constraint
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8)

# create the Euler angle prior constraint to encourage vertical/horizontal stress orientation
vert_S3_constraint = SOSAT.VerticalPrincipalStressConstraint(:S3, deg2rad(5.0))

indicators = Vector{SOSAT.StressIndicator}(undef,2)

indicators[1] = fc
indicators[2] = vert_S3_constraint

SHmax = sigv * 0.9
SHmin = sigv * 0.7
alpha = deg2rad(0.0)
beta = deg2rad(0.0)
gamma = deg2rad(0.0)
so = SOSAT.StressState(SHmax, SHmin, sigv, pp_MPa,
                        alpha,
                        beta,
                        gamma)

# perturb SHmax and SHmin but not Sv (which is S3) or the pore pressure
stress_std = [5.0,5.0,0.0,0.0]

# allow all three angles to vary
angle_std = [deg2rad(1.0),deg2rad(1.0),deg2rad(1.0)]
Nsample = 1000000
Nburn = 1000
Nthin = 2

sampler = SOSAT.MCMCSampler(so,
                        indicators,
                        stress_std,
                        angle_std)
initial_likelihood = SOSAT.EvaluateInitialLikelihood(sampler)

println("Initial likelihood = ",initial_likelihood)

samples, acceptance_ratio = SOSAT.Sample(sampler,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true)

if GENERATE_PLOTS
    alpha_hist_fig = SOSAT.StressComponentHist(samples, :alpha, bins=100)
    alpha_hist_fig.savefig("OrientationPrior_test_alpha_hist.png")
    plt.close(alpha_hist_fig)
    beta_hist_fig = SOSAT.StressComponentHist(samples, :beta, bins=100)
    beta_hist_fig.savefig("OrientationPrior_test_beta_hist.png")
    plt.close(beta_hist_fig)
    gamma_hist_fig = SOSAT.StressComponentHist(samples, :gamma, bins=100)
    gamma_hist_fig.savefig("OrientationPrior_test_gamma_hist.png")
    
    principal_stress_stereonet_fig = SOSAT.PrincipalStressStereonet(samples, alpha=1.0e-2)
    principal_stress_stereonet_fig.savefig("OrientationPrior_test_stereonet.png")

    stress_poly_ax, stress_poly_fig = SOSAT.StressPolygonHist(samples,
                                                                    nbins=200,
                                                                    shmax_component=1,
                                                                    shmin_component=2)

    
    SOSAT.plot_stress_polygon(stress_poly_ax, sigv, pp_MPa, 0.6; color="b")
    SOSAT.plot_stress_polygon(stress_poly_ax, sigv, pp_MPa, 0.8; color="k")
    stress_poly_ax.set_xlim(25.0,200.0)
    stress_poly_ax.set_ylim(25.0,200.0)
    stress_poly_fig.savefig("OrientationPrior_test_polygon.png")
    plt.close(stress_poly_fig)
    plt.close(principal_stress_stereonet_fig)
end