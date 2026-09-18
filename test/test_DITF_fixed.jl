using SOSAT
using Test
using Random
using Distributions
using PythonPlot
const plt = PythonPlot

Random.seed!(1234)
plt.matplotlib.use("Agg")
"""
This test is meant to reproduce Figure 7 from Moos and Zoback 1990
by using the DITF constraint along with a fault friction constraint

For this test we do not allow the stress tensor to rotate. We fix it so that
SHmax is in the direction of the DITF.
"""

const GENERATE_PLOTS = haskey(ENV, "JULIA_TEST_PLOTS") || 
                       get(ENV, "JULIA_TEST_PLOTS", "false") == "true"

save_samples = false
water_depth = 4000.0
water_density = 1025.0 # kg/m3
basalt_depth = 1000.0
basalt_density = 2800.0 # kg/m3

g = 9.81 # m/s2
sigv = (water_density * g * water_depth +
        basalt_density * g * basalt_depth) / 1.0e6 # MPa
println("Vertical Stress is ",sigv)

pp_MPa = (water_depth + basalt_depth) * 10.0 / 1000.0 # assume 10 MPa/km
println("Pore pressure is ",pp_MPa)

#----------------------------
# Fault Friction distribution
#----------------------------
mudist = Distributions.truncated(Distributions.LogNormal(-0.53,0.1),0.1,0.9)
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8)

mud_pressure_max_MPa = pp_MPa + 6.0
println("max mud pressure is ",mud_pressure_max_MPa)

# note Young's modulus should be in MPa so 60.0e3 is 60 GPa
YM = 60.0e3
PR=0.25
CTE=5.0e-6
tensile_strength_dist = Distributions.Uniform(0.0,20.0)

ditf_param = SOSAT.DITFParameters(true,          # DITF observed
                                    DITF_azimuth=deg2rad(0.0), # DITF azimuth
                                    DITF_azimuth_err=deg2rad(5.0), # DITF azimuth error
                                    DITF_omega=deg2rad(0.0),  # DITF omega
                                    DITF_omega_err=deg2rad(5.0),  # DITF omega error
                                    tensile_strength_dist=tensile_strength_dist,
                                    mud_pressure_dist=Distributions.Uniform(pp_MPa,mud_pressure_max_MPa),
                                    mud_temperature_dist=Distributions.Uniform(50.0,51.0),
                                    well_azimuth=0.0,
                                    well_inclination=0.0,
                                    formation_temperature=50.5,
                                    YM=YM,
                                    PR=PR,
                                    CTE=CTE)
ditf_param.state.tensile_strength=0.0
ditf_param.state.mud_pressure=pp_MPa + 3.0
ditf_param.state.mud_temperature=50.5
TS_array = LinRange(0.0,30.0,100)
pTS = Distributions.pdf.(tensile_strength_dist, TS_array)

if GENERATE_PLOTS
    TS_fig = plt.figure()
    TS_ax = TS_fig.add_subplot(111)
    jnk = TS_ax.plot(TS_array, pTS, color="k")
    TS_ax.set_xlabel("Tensile Strength (MPa)")
    TS_ax.set_ylabel("Probability Density")
    TS_fig.savefig("tensile_strength_PDF.png")
end
# for initial testing we do not perturb the UCS, or mud temperature or pressure
ditf_ind = SOSAT.DITF(ditf_param,
                        0.0,  # tensile strength proposal distribution
                        0.0, # mud pressure proposal distribution
                        0.0) # mud_temperature proposal distribution
println("ditf_ind= ",ditf_ind)
# for this test we set all Euler rotations to zero so that the first
# principal stress is N-S, the second is E-W, and the third is vertical

# the subhorizontal stress that may be either S1 or S2 but is

# oriented North prior to Euler rotations
#SHmax = 80.0
S1 = 120.0
S2 = sigv
S3 = sigv
# oriented East prior to Euler rotations
#SHmin = 0.9 * sigv
#SHmin = sigv
# the vertical stress
alpha = deg2rad(0.0)
beta = deg2rad(0.0)
gamma = deg2rad(0.0)
so = SOSAT.StressState(S1, S2, S3, pp_MPa,
                        alpha,
                        beta,
                        gamma)

S_geo = SOSAT.GeographicStress(so);                        
println("Geographic Stress Tensor (MPa):")
println("|  $(round(S_geo[1,1], digits=2))  $(round(S_geo[1,2], digits=2))  $(round(S_geo[1,3], digits=2))|")
println("|  $(round(S_geo[2,1], digits=2))  $(round(S_geo[2,2], digits=2))  $(round(S_geo[2,3], digits=2))|")
println("|  $(round(S_geo[3,1], digits=2))  $(round(S_geo[3,2], digits=2))  $(round(S_geo[3,3], digits=2))|")

# test that S_geo[1,1] is close to S2 since S2 is vertical
@test isapprox(S_geo[3,3], S3; atol=1e-5)
# test that S_geo[1,2] is close to zero since there are no rotations
@test isapprox(S_geo[1,2], 0.0; atol=1e-5)
# test that S_geo[1,1] is close to S1 since S1 is North
@test isapprox(S_geo[1,1], S1; atol=1e-5)
# test that S_geo[2,2] is close to S2 since S2 is East
@test isapprox(S_geo[2,2], S2; atol=1e-5)

indicators = Vector{SOSAT.StressIndicator}(undef,2)

indicators[1] = fc
indicators[2] = ditf_ind

# perturb SHmax and SHmin but not Sv (which is S3)
# don't perturb pore pressure or Euler angles
stress_std = [5.0,5.0,0.0,0.0]
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
initial_likelihood = SOSAT.EvaluateInitialLikelihood(sampler, verbose=true)
println("Initial likelihood = ",initial_likelihood)

# test that initial likelihood is approximately 0.00782
#@test isapprox(initial_likelihood, 0.00782; atol=1e-4)

if initial_likelihood == 0.0
    println("Initial stress state has zero likelihood - adjust initial stress state")
    new_state, found = SOSAT.FindLikelyInitialState(sampler, num_samples=1000, verbose=false)
    if found == false
        println("Could not find a valid initial stress state with non-zero likelihood")
        return
    else
        println("Found a more likely initial stress state from reduced sampling")
        sampler.stress_state = new_state
        println(sampler.stress_state)
        initial_likelihood = SOSAT.EvaluateInitialLikelihood(sampler, verbose=true)
        println("New initial likelihood = ",initial_likelihood)
    end
end 

println("Starting MCMC sampling...")
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

println("S1 median: ",S1_median," range: ",S1_range," IQR: ",S1_IQR)

# test that the S1 median is approximately 96.5
#@test isapprox(S1_median, 96.5; atol=0.5)

# test that the range of S1 is approximately 75.0
#@test isapprox(S1_range, 75.0; atol=1.0)

# test that the IQR of S1 is approximately 24.2
#@test isapprox(S1_IQR, 24.2; atol=1.0)

println("S2 median: ",S2_median," range: ",S2_range," IQR: ",S2_IQR)
# test that the S2 median is approximately 64.0
#@test isapprox(S2_median, 64.0; atol=0.5)

# test that the range of S2 is approximately 20.3
#@test isapprox(S2_range, 20.3; atol=1.0)

# test that the IQR of S2 is approximately 7.1
#@test isapprox(S2_IQR, 7.1; atol=1.0)

println("S3 median: ",S3_median," range: ",S3_range," IQR: ",S3_IQR)
# test that the S3 median is equal to sigv
@test isapprox(S3_median, sigv; atol=1e-5)

# test that the range of S3 is approximately zero
@test isapprox(S3_range, 0.0; atol=1e-5)

# compute the correlation between S1 and S2
S_corr = SOSAT.GetStressComponentCorrelation(samples, :S1, :S2)
println("Correlation between S1 and S2: ",S_corr)

# test that the correlation between S1 and S2 is approximately 0.95
#@test isapprox(S_corr, 0.95; atol=0.02)

if GENERATE_PLOTS
    # get the state from the last sample to plot the DITF constraint
    deltaT =  ditf_param.formation_temperature - ditf_param.state.mud_temperature
    deltaP = ditf_param.state.mud_pressure - pp_MPa

    shmin_array,shmax_array = SOSAT.getDITFStressArrays(pp_MPa, sigv, 0.8, deltaP, deltaT,
                                                YM, PR, CTE,
                                                ditf_param.state.tensile_strength, 100)
    # plot the stress samples in the stress polygon space
    stress_poly_ax, stress_poly_fig = SOSAT.StressPolygonHist(samples, nbins=200, shmax_component=1, shmin_component=2)
    SOSAT.plot_stress_polygon(stress_poly_ax, sigv, pp_MPa, 0.8; color="k")
    # plot the 1:1 line
    stress_poly_ax.plot([0.0,200.0],[0.0,200.0], color="gray", linestyle="--")
    stress_poly_ax.plot(shmin_array, shmax_array, color="r")
    stress_poly_ax.set_xlim(45.0, 150.0)
    stress_poly_ax.set_ylim(45.0, 150.0)
    stress_poly_ax.set_xlabel("SHmin (MPa)")
    stress_poly_ax.set_ylabel("SHmax (MPa)")
    stress_poly_fig.savefig("DITF_test_polygon_ditf_fixed.png")

    S1fig = SOSAT.StressComponentHist(samples, :S1, label="S1 (MPa)")
    S1fig.savefig("DITF_testS1_marginal_posterior_ditf_fixed.png")

    S2fig = SOSAT.StressComponentHist(samples, :S2, label="S2 (MPa)")
    S2fig.savefig("DITF_testS2_marginal_posterior_ditf_fixed.png")

    S3fig = SOSAT.StressComponentHist(samples, :S3, label="S3 (MPa)")
    S3fig.savefig("DITF_testS3_marginal_posterior_ditf_fixed.png")

    ppfig = SOSAT.StressComponentHist(samples, :Pp, label="Pore Pressure (MPa)")
    ppfig.savefig("DITF_test_Pp_marginal_posterior_ditf_fixed.png")
    plt.close(ppfig)

    alpha_fig = SOSAT.StressComponentHist(samples, :alpha, label="Alpha (deg)", bins=50)
    alpha_fig.savefig("DITF_test_alpha_azimuth_marginal_posterior_ditf_fixed.png")
    plt.close(alpha_fig)

    beta_fig = SOSAT.StressComponentHist(samples, :beta, label="Beta (deg)", bins=50)
    beta_fig.savefig("DITF_test_Beta_deviation_marginal_posterior_ditf_fixed.png")
    plt.close(beta_fig)

    gamma_fig = SOSAT.StressComponentHist(samples, :gamma, label="Gamma Euler Angle (deg)", bins=50)
    gamma_fig.savefig("DITF_test_Gamma_euler_marginal_posterior_ditf_fixed.png")
    plt.close(gamma_fig)

    principal_stress_stereonet_fig = SOSAT.PrincipalStressStereonet(samples)
    principal_stress_stereonet_fig.savefig("DITF_test_fixed_principal_stress_stereonet.png")



    stress_mat = SOSAT.GetStressMatrix(samples;
                                        thin=10)
    labels = ["S1 (MPa)","S2 (MPa)","S3 (MPa)","Pp (MPa)",
            "Alpha (deg)","Beta (deg)","Gamma (deg)"]
    fig = plt.figure(figsize=(10,10))

    println("Generating trace plots...")

    # make a grid spec with one column of plots, one for each of the 
    # six variables
    gs = fig.add_gridspec(7,1)
    for i in 1:7
        ax = fig.add_subplot(gs[i-1,0])
        ax.plot(stress_mat[:,i], color="blue", alpha=0.5)
        ax.set_ylabel(labels[i])
        ax.set_xlabel("Sample Index")
    end
    fig.tight_layout()
    fig.savefig("DITF_test_stress_trace_plots_ditf_fixed.png")
    # use the function in StressUtils.jl to get the DITF limit line to plot in the
end
plt.close("all")

if save_samples == true
    SOSAT.WriteTraceToH5(samples, "DITF_test_stress_samples_ditf_fixed.h5")
end