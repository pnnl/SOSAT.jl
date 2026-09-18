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

"""

const GENERATE_PLOTS = haskey(ENV, "JULIA_TEST_PLOTS") || 
                       get(ENV, "JULIA_TEST_PLOTS", "false") == "true"

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
                                    DITF_azimuth=deg2rad(20.0), # DITF azimuth
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
S1 = 115.0
S2 = sigv
S3 = sigv
# oriented East prior to Euler rotations
#SHmin = 0.9 * sigv
#SHmin = sigv
# the vertical stress
alpha = deg2rad(0.0)
beta = deg2rad(90.0)
gamma = deg2rad(70.0)
so = SOSAT.StressState(S1, S2, S3, pp_MPa,
                        alpha,
                        beta,
                        gamma)
#b = SOSAT.BoreholeStress.WellState(mud_pressure_max_MPa, 0.0, 0.0)

indicators = Vector{SOSAT.StressIndicator}(undef,2)

indicators[1] = fc
#indicators[2] = ppc
indicators[2] = ditf_ind

# perturb SHmax and SHmin but not Sv (which is S3)
# don't perturb pore pressure or Euler angles
stress_std = [5.0,5.0,0.0,0.0]
#angle_std = [deg2rad(2.0),deg2rad(2.0),deg2rad(0.0)]
angle_std = [deg2rad(1.0),deg2rad(1.0),deg2rad(1.0)]
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

if GENERATE_PLOTS           
    # get the state from the last sample to plot the DITF constraint
    deltaT =  ditf_param.formation_temperature - ditf_param.state.mud_temperature
    deltaP = ditf_param.state.mud_pressure - pp_MPa
    # use the function in SOSAT.jl to get the DITF limit line to plot in the
    # stress polygon space
    shmin_array,shmax_array = SOSAT.getDITFStressArrays(pp_MPa, sigv, 0.8, deltaP, deltaT,
                                                YM, PR, CTE,
                                                ditf_param.state.tensile_strength, 100)
    # plot the stress samples in the stress polygon space
    stress_poly_ax, stress_poly_fig = SOSAT.StressPolygonHist(samples, nbins=200)
    SOSAT.plot_stress_polygon(stress_poly_ax, sigv, pp_MPa, 0.8; color="k")
    # plot the 1:1 line
    stress_poly_ax.plot([0.0,200.0],[0.0,200.0], color="gray", linestyle="--")
    stress_poly_ax.plot(shmin_array, shmax_array, color="r")
    stress_poly_ax.set_xlim(45.0, 150.0)
    stress_poly_ax.set_ylim(45.0, 150.0)
    stress_poly_ax.set_xlabel("SHmin (MPa)")
    stress_poly_ax.set_ylabel("SHmax (MPa)")
    stress_poly_fig.savefig("DITF_test_polygon_ditf_true_general.png")
    


    S1fig = SOSAT.StressComponentHist(samples, :S1, label="Subvertical Principal Stress (MPa)")
    S1fig.savefig("DITF_testS1_marginal_posterior_ditf_true_general.png")

    S2fig = SOSAT.StressComponentHist(samples, :S2, label="Subhorizontal Maximum Principal Stress (MPa)")
    S2fig.savefig("DITF_testS2_marginal_posterior_ditf_true_general.png")

    S3fig = SOSAT.StressComponentHist(samples, :S3, label="Subhorizontal Minimum Principal Stress (MPa)")
    S3fig.savefig("DITF_testS3_marginal_posterior_ditf_true_general.png")

    ppfig = SOSAT.StressComponentHist(samples, :Pp, label="Pore Pressure (MPa)")
    ppfig.savefig("DITF_test_Pp_marginal_posterior_ditf_true_general.png")
    plt.close(ppfig)

    alpha_fig = SOSAT.StressComponentHist(samples, :alpha, label="Alpha (deg)", bins=50)
    alpha_fig.savefig("DITF_test_alpha_azimuth_marginal_posterior_ditf_true_general.png")
    plt.close(alpha_fig)

    beta_fig = SOSAT.StressComponentHist(samples, :beta, label="Beta (deg)", bins=50)
    beta_fig.savefig("DITF_test_Beta_deviation_marginal_posterior_ditf_true_general.png")
    plt.close(beta_fig)

    gamma_fig = SOSAT.StressComponentHist(samples, :gamma, label="Gamma Euler Angle (deg)", bins=50)
    gamma_fig.savefig("DITF_test_Gamma_euler_marginal_posterior_ditf_true_general.png")
    plt.close(gamma_fig)
    
    SOSAT.WriteTraceToH5(samples, "DITF_test_stress_samples_ditf_true_general.h5")
end