using SOSAT
using Test
using Random, Distributions
using PythonPlot
const plt = PythonPlot

Random.seed!(1234)
plt.matplotlib.use("Agg")
"""
This test is meant to reproduce Figure 7 from Moos and Zoback 1990
by using the DITF constraint along with a fault friction constraint

Except for this case we do not have a DITF observed in the wellbore.
So that the stress states sampled should all lie below the DITF limit line.

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

ditf_param = SOSAT.DITFParameters(false,          # DITF observed
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
S2 = 80.0
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


indicators = Vector{SOSAT.StressIndicator}(undef,2)

indicators[1] = fc
indicators[2] = ditf_ind

# perturb SHmax and SHmin but not Sv (which is S3)
# don't perturb pore pressure or Euler angles
stress_std = [5.0,5.0,0.0,0.0]
#angle_std = [deg2rad(2.0),deg2rad(2.0),deg2rad(0.0)]
angle_std = [deg2rad(0.0),deg2rad(0.0),deg2rad(0.0)]
Nsample = 500000
#Nsample = 10
Nburn = 10000
Nthin = 2

sampler = SOSAT.MCMCSampler(so,
                        indicators,
                        stress_std,
                        angle_std)
initial_likelihood = SOSAT.EvaluateInitialLikelihood(sampler, verbose=true)
println("Initial likelihood = ",initial_likelihood)


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

# test that the S1 median is approximately 81.2
@test isapprox(S1_median, 81.2; atol=1.0)

# test that the range of S1 is approximately 79.5
@test isapprox(S1_range, 79.5; atol=1.0)

# test that the IQR of S1 is approximately 21.6
@test isapprox(S1_IQR, 21.6; atol=1.0)

println("S2 median: ",S2_median," range: ",S2_range," IQR: ",S2_IQR)
# test that the S2 median is approximately 81.4
@test isapprox(S2_median, 81.4; atol=1.0)

# test that the range of S2 is approximately 73.9
@test isapprox(S2_range, 73.9; atol=1.0)

# test that the IQR of S2 is approximately 21.7
@test isapprox(S2_IQR, 21.7; atol=1.0)

println("S3 median: ",S3_median," range: ",S3_range," IQR: ",S3_IQR)
# test that the S3 median is equal to sigv
@test isapprox(S3_median, sigv; atol=1e-5)

# test that the range of S3 is approximately zero
@test isapprox(S3_range, 0.0; atol=1e-5)

# compute the correlation between S1 and S2
S_corr = SOSAT.GetStressComponentCorrelation(samples, :S1, :S2)
println("Correlation between S1 and S2: ",S_corr)

# test that the correlation between S1 and S2 is approximately 0.49
#@test isapprox(S_corr, 0.49; atol=0.02)

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
    stress_poly_fig.savefig("No_DITF_test_polygon_ditf.png")

    S1fig = SOSAT.StressComponentHist(samples, :S1, label="S1 (MPa)")
    S1fig.savefig("No_DITF_testS1_marginal_posterior_ditf.png")

    S2fig = SOSAT.StressComponentHist(samples, :S2, label="S2 (MPa)")
    S2fig.savefig("No_DITF_testS2_marginal_posterior_ditf.png")

    S3fig = SOSAT.StressComponentHist(samples, :S3, label="S3 (MPa)")
    S3fig.savefig("No_DITF_testS3_marginal_posterior_ditf.png")

    ppfig = SOSAT.StressComponentHist(samples, :Pp, label="Pore Pressure (MPa)")
    ppfig.savefig("No_DITF_test_Pp_marginal_posterior_ditf.png")
    plt.close(ppfig)

    alpha_fig = SOSAT.StressComponentHist(samples, :alpha, label="Alpha (deg)", bins=50)
    alpha_fig.savefig("No_DITF_test_alpha_azimuth_marginal_posterior_ditf.png")
    plt.close(alpha_fig)

    beta_fig = SOSAT.StressComponentHist(samples, :beta, label="Beta (deg)", bins=50)
    beta_fig.savefig("No_DITF_test_Beta_deviation_marginal_posterior_ditf.png")
    plt.close(beta_fig)

    gamma_fig = SOSAT.StressComponentHist(samples, :gamma, label="Gamma Euler Angle (deg)", bins=50)
    gamma_fig.savefig("No_DITF_test_Gamma_euler_marginal_posterior_ditf.png")
    plt.close(gamma_fig)

    principal_stress_stereonet_fig = SOSAT.PrincipalStressStereonet(samples)
    principal_stress_stereonet_fig.savefig("No_DITF_test_principal_stress_stereonet.png")

    plt.close("all")
end

if save_samples == true
    SOSAT.WriteTraceToH5(samples, "No_DITF_test_stress_samples_ditf.h5")
end