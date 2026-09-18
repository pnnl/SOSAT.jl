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
to produce a probabilistic version of the Figure 2 from Zoback et al 2003
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

if GENERATE_PLOTS
    mu_array = LinRange(0.0,1.0,100)
    pmu = pdf.(mudist, mu_array)
    mu_fig = plt.figure()
    mu_ax = mu_fig.add_subplot(111)
    jnk = mu_ax.plot(mu_array, pmu, color="k")
    mu_ax.set_xlabel("Fault Friction Coefficient")
    mu_ax.set_ylabel("Probability Density")
    mu_fig.savefig("Fault_Friction_PDF.png")
end
# create the fault constraint
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8)

SHmax = sigv * 0.9
SHmin = sigv * 0.7
alpha = deg2rad(0.0)
beta = deg2rad(0.0)
gamma = 0.0
so = SOSAT.StressState(SHmax, SHmin, sigv, pp_MPa,
                        alpha,
                        beta,
                        gamma)
#b = SOSAT.BoreholeStress.WellState(mud_pressure_max_MPa, 0.0, 0.0)

indicators = Vector{SOSAT.StressIndicator}(undef,1)

indicators[1] = fc


# perturb SHmax and SHmin but not Sv (which is S3)
# don't perturb pore pressure or Euler angles
stress_std = [5.0,5.0,0.0,0.0]
#angle_std = [deg2rad(2.0),deg2rad(2.0),deg2rad(0.0)]
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

# get the summary statistics for the stress components
S1_median, S1_range, S1_IQR = SOSAT.GetStressComponentStatistics(samples, :S1)
S2_median, S2_range, S2_IQR = SOSAT.GetStressComponentStatistics(samples, :S2)
S3_median, S3_range, S3_IQR = SOSAT.GetStressComponentStatistics(samples, :S3)
alpa_median, alpha_range, alpha_IQR = SOSAT.GetStressComponentStatistics(samples, :alpha)
beta_median, beta_range, beta_IQR = SOSAT.GetStressComponentStatistics(samples, :beta)
gamma_median, gamma_range, gamma_IQR = SOSAT.GetStressComponentStatistics(samples, :gamma)
mu = 0.8                            

if GENERATE_PLOTS
    stress_poly_ax, stress_poly_fig = SOSAT.StressPolygonHist(samples,
                                                                nbins=200,
                                                                shmax_component=1,
                                                                shmin_component=2)

    
    SOSAT.plot_stress_polygon(stress_poly_ax, sigv, pp_MPa, 0.6; color="b")
    SOSAT.plot_stress_polygon(stress_poly_ax, sigv, pp_MPa, 0.8; color="k")
    stress_poly_ax.set_xlim(25.0,200.0)
    stress_poly_ax.set_ylim(25.0,200.0)
    stress_poly_fig.savefig("Fault_Constraint_test_polygon.png")
    plt.close(stress_poly_fig)

    S1fig = SOSAT.StressComponentHist(samples, :S1, label="S1 (MPa)")
    S1fig.savefig("Fault_Constraint_test_S1_marginal_posterior.png")
    plt.close(S1fig)
    S2fig = SOSAT.StressComponentHist(samples, :S2, label="S2 (MPa)")
    S2fig.savefig("Fault_Constraint_test_S2_marginal_posterior.png")
    plt.close(S2fig)
    alpha_fig = SOSAT.StressComponentHist(samples, :alpha, label="Alpha (deg)", bins=50)
    alpha_fig.savefig("Fault_Constraint_test_alpha_azimuth_marginal_posterior.png")
    plt.close(alpha_fig)
    beta_fig = SOSAT.StressComponentHist(samples, :beta, label="Beta (deg)", bins=50)
    beta_fig.savefig("Fault_Constraint_test_Beta_deviation_marginal_posterior.png")
    plt.close(beta_fig)
    gamma_fig = SOSAT.StressComponentHist(samples, :gamma, label="Gamma (deg)", bins=50)
    gamma_fig.savefig("Fault_Constraint_test_Gamma_euler_marginal_posterior.png")
    plt.close(gamma_fig)
end