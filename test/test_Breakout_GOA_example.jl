using SOSAT
using Test
using Random
using Distributions
using PythonPlot
const plt = PythonPlot

Random.seed!(1234)
plt.matplotlib.use("Agg")
"""
This test is meant to reproduce the results from Pathfinder well in the Eugene Island 330
field in the Gulf of America (GOA) from Peska and Zoback (1995) and
Zoback et al (2003) where the stress state is constrained by
both fault friction and Breakouts in a deviated well.

"""

const GENERATE_PLOTS = haskey(ENV, "JULIA_TEST_PLOTS") || 
                       get(ENV, "JULIA_TEST_PLOTS", "false") == "true"

save_samples = false
        

Sv = 43.0; # MPa
println("Vertical Stress is ",Sv)

pp_MPa = 29.0 # MPa
println("Pore pressure is ",pp_MPa)

SHmin = 37.1 # MPa
println("Minimum Horizontal Stress is ",SHmin)


#
# For this example we order the principal stresses as S1 >= S2 >= S3
# Thus Sv = S1, SHmax = S2, SHmin = S3 (normal faulting stress regime)

# Before the Euler rotations Sv is oriented North
# SHmax is oriented East
# SHmin is oriented downward
#
# We need to select the Euler rotations so that Sv is vertical, Shmin is oriented 35 degrees
# clockwise from North and SHmax is oriented 105 degrees clockwise from North.

alpha = deg2rad(0.0)
beta = deg2rad(90.0)
gamma = deg2rad(-35.0)
so = SOSAT.StressState(Sv, 40.0, SHmin, pp_MPa,
                        alpha,
                        beta,
                        gamma)

S_geo = SOSAT.GeographicStress(so);                        
println("Geographic Stress Tensor (MPa):")
println("|  $(round(S_geo[1,1], digits=2))  $(round(S_geo[1,2], digits=2))  $(round(S_geo[1,3], digits=2))|")
println("|  $(round(S_geo[2,1], digits=2))  $(round(S_geo[2,2], digits=2))  $(round(S_geo[2,3], digits=2))|")
println("|  $(round(S_geo[3,1], digits=2))  $(round(S_geo[3,2], digits=2))  $(round(S_geo[3,3], digits=2))|")

# perturb SHmax and SHmin but not Sv (which is S1) or the pore pressure
# we perturm SHmin a little bit to allow the minifrac closure constraint to
# control the magnitude of SHmin. We make the perturbation small so that we don't
# have a high rejection rate because of the narrow range of SHmin allowed by the minifrac closure constraint.
stress_std = [0.0,5.0,0.1,0.0]

# don't perturb alpha or beta but allow perturbation of gamma
# this will keep the principal stresses vertical/horizontal but allow the azimuth of the principal stresses to change.
angle_std = [deg2rad(0.0),deg2rad(0.0),deg2rad(2.0)]

#----------------------------
# Fault Friction distribution
#----------------------------
mudist = Distributions.truncated(Distributions.LogNormal(-0.53,0.1),0.1,0.9)
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8)

mfc_dist = Distributions.Uniform(SHmin-0.1, SHmin+0.1)
mfc_constraint = SOSAT.MiniFracClosure(mfc_dist)

borehole_deviation_deg = 32.0
println("Borehole deviation is ",borehole_deviation_deg," degrees from vertical")

borehole_azimuth_deg = 35.0
println("Borehole azimuth is ",borehole_azimuth_deg," degrees clockwise from North")

breakout_angle_deg = 17.0
println("Breakout angle is ",breakout_angle_deg," degrees from the bottom of the borehole")

#
# The strength is not well described in the papers. In Peska and Zoback they refer to C_o and have a plots for a range
# of C_o/Sv ranging from 0.5 (21.5 MPa) to 4.5 (193.5 MPa). But in Zoabck et al 2003 they refer to a
# "Critical Strength" range of around 20 to 32 (exact range is hard to determine from the plot).
# assume that this "Critical Strength" is the C = C_o * (1.0 + 0.6 * mu_internal) that is referenced in
# Moos and Zoback (1990) for breakout prediction, but it may be the full Mohr-Coulomb criteria like is used
# in SOSAT.jl. If we assume that they are using the full Mohr-Coulomb criteria, it isn't straitforward to
# back out the "critical strength" to UCS because it depends on the magnitude of the radial and tangential stresses,
# which varies with the azimuth around the borehole and the magnitude of SHmax, which is not a fixed number.
# 
# We do our best by trying to use the formula from Moos and Zoback (1990) to back out UCS from the "Critical Strength" range.
#
# From the caption of Plate 3 it seems that they are using an internal friction coefficient of 1.0
mu_internal = 1.0

UCS_min = 20.0 / (1.0 + 0.6 * mu_internal)
UCS_max = 32.0 / (1.0 + 0.6 * mu_internal)
println("Using UCS range of ",UCS_min," to ",UCS_max," MPa for breakout calculation")

# We set a low initial UCS so that it is easier to find a valid initial state
UCS_init = 19.0
#
#
# We set the probablity distribtion for the max mud pressure 0.5 MPa above
# pore pressure
mud_pressure_max_MPa = pp_MPa+0.5

# However, for this exmaple we will make the mud pressure fixed so that it is
# only essential that it is between the pore pressure and max mud pressure.

# a mud weight that gives a mud pressure slightly above pore pressure
mud_pressure_MPa = pp_MPa+0.1
println("mud pressure is ",mud_pressure_MPa)

# Young's and Poisson ratio are not given in the paper so we assume
# typical values for sedimentary rocks
YM = 6.0e3
PR=0.25
CTE=5.0e-6
UCS_dist = Distributions.Uniform(UCS_min,UCS_max)

bo_param = SOSAT.BreakoutParameters(true, #::Any
                                breakout_azimuth=deg2rad(breakout_angle_deg), #breakout_azimuth,
                                breakout_azimuth_err=deg2rad(5.0), # breakout_azimuth_err,
                                UCS_dist=UCS_dist, # UCS_dist,
                                mud_pressure_dist=Distributions.Uniform(pp_MPa,mud_pressure_max_MPa), #  mud_pressure_dist,
                                mud_temperature_dist=Distributions.Uniform(50.0,51.0), #  mud_temperature_dist,
                                rock_friction_dist=Distributions.Uniform(0.9,1.1),#  rock_friction_dist,
                                well_azimuth=deg2rad(borehole_azimuth_deg), # well_azimuth, 
                                well_inclination=deg2rad(borehole_deviation_deg), #well_inclination, 
                                formation_temperature=50.0,# formation_temperature, 
                                YM=YM, # YM, 
                                PR=PR, # PR,
                                CTE=CTE) #  CTE)

bo_param.state.UCS = UCS_init
bo_param.state.mud_pressure = mud_pressure_MPa
bo_param.state.mud_temperature = 50.5
bo_param.state.rock_friction = mu_internal

# We allow perturbation of UCS and leave all other parameters fixed
breakout_indicator = SOSAT.Breakout(bo_param,
                                    0.1, # UCS proposal
                                    0.0, # mud pressure proposal
                                    0.0, # mud temperature proposal
                                    0.0); # rock friction proposal


indicators = Vector{SOSAT.StressIndicator}(undef,3)


indicators[1] = fc
indicators[2] = mfc_constraint
indicators[3] = breakout_indicator


Nsample = 100000
#Nsample = 10
Nburn = 1000
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




UCS = breakout_indicator.breakout_parameters.state.UCS
println("UCS used in breakout calculation: ",UCS)
mu = breakout_indicator.breakout_parameters.state.rock_friction
println("Rock friction used in breakout calculation: ",mu)
deltaP = breakout_indicator.breakout_parameters.state.mud_pressure - pp_MPa
println("Delta P used in breakout calculation: ",deltaP)
deltaT = breakout_indicator.breakout_parameters.state.mud_temperature - breakout_indicator.breakout_parameters.formation_temperature
println("Delta T used in breakout calculation: ",deltaT)
println("Pore pressure: ",pp_MPa)
println("Vertical Stress: ",Sv)

if GENERATE_PLOTS 
shmin_array, shmax_array = SOSAT.GetBreakoutStressArrays(pp_MPa, Sv, UCS, mu, deltaP, deltaT,
                                YM, PR, CTE, 100)
# plot the stress samples in the stress polygon space
stress_poly_ax, stress_poly_fig = SOSAT.StressPolygonHist(samples, nbins=200, shmax_component=1, shmin_component=2)
SOSAT.plot_stress_polygon(stress_poly_ax, Sv, pp_MPa, 0.8; color="k")
# plot the 1:1 line
stress_poly_ax.plot([0.0,100.0],[0.0,100.0], color="gray", linestyle="--")
stress_poly_ax.plot(shmin_array, shmax_array, color="r")
stress_poly_ax.set_xlim(15.0, 100.0)
stress_poly_ax.set_ylim(15.0, 100.0)
stress_poly_ax.set_xlabel("SHmin (MPa)")
stress_poly_ax.set_ylabel("SHmax (MPa)")
stress_poly_fig.savefig("GOA_test_polygon.png")

S1_fig = SOSAT.StressComponentHist(samples, :S1, label="S1 (MPa)", bins=50)
S1_fig.savefig("GOA_test_S1_marginal_posterior.png")
plt.close(S1_fig)
S2_fig = SOSAT.StressComponentHist(samples, :S2, label="S2 (MPa)", bins=50)
S2_fig.savefig("GOA_test_S2_marginal_posterior.png")
plt.close(S2_fig)
SHmax_median, SHMax_range, SHMax_IRQ = SOSAT.GetStressComponentStatistics(samples, :SHmax)
println("SHmax median (MPa): ",SHmax_median)
println("SHmax range (MPa): ",SHMax_range)
println("SHmax IQR (MPa): ",SHMax_IRQ)
@test isapprox(SHmax_median, 47.4, atol=1.0)
@test isapprox(SHMax_range, 26.5, atol=1.0)
@test isapprox(SHMax_IRQ, 6.7, atol=1.0)
S3_fig = SOSAT.StressComponentHist(samples, :S3, label="S3 (MPa)", bins=50)
S3_fig.savefig("GOA_test_S3_marginal_posterior.png")

alpha_fig = SOSAT.StressComponentHist(samples, :alpha, label="Alpha (deg)", bins=50)
alpha_fig.savefig("GOA_test_alpha_azimuth_marginal_posterior.png")
plt.close(alpha_fig)

beta_fig = SOSAT.StressComponentHist(samples, :beta, label="Beta (deg)", bins=50)
beta_fig.savefig("GOA_test_Beta_deviation_marginal_posterior.png")
plt.close(beta_fig)

gamma_fig = SOSAT.StressComponentHist(samples, :gamma, label="Gamma Euler Angle (deg)", bins=50)
gamma_fig.savefig("GOA_test_Gamma_euler_marginal_posterior.png")
plt.close(gamma_fig)
gamma_median, gamma_range, gamma_IQR = SOSAT.GetStressComponentStatistics(samples, :gamma)
println("Gamma median (deg): ",rad2deg(gamma_median))
println("Gamma range (deg): ",rad2deg.(gamma_range))
println("Gamma IQR (deg): ",rad2deg.(gamma_IQR))
@test isapprox(rad2deg(gamma_median), -51.0, atol=1.0)
@test isapprox(rad2deg(gamma_range), 40.0, atol=1.0)
@test isapprox(rad2deg(gamma_IQR), 7.63, atol=1.0)
principal_stress_stereonet_fig = SOSAT.PrincipalStressStereonet(samples, alpha=0.01)
principal_stress_stereonet_fig.savefig("GOA_test_stress_stereonet.png")

gamma_samples = SOSAT.GetGammaSamples(samples)
shmax_az = -1.0 .* gamma_samples .+ pi/2.0
shmax_samples = SOSAT.GetSHmaxSamples(samples)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.scatter(shmax_samples, rad2deg.(shmax_az), color="blue", alpha=0.5, s=1)
#ax.set_xlim(37.1,43.0)
ax.set_xlabel("SHmax (MPa)")
ax.set_ylabel("SHmax Azimuth (deg)")
ax.set_xlim(SHmin,Sv)
ax.set_ylim(90.0, 180.0)
fig.savefig("GOA_test_SHmax_azimuth_vs_SHmax.png")

#
# This plot compares well with the plot in Figure 19 of Zoback et al (2003)

plt.close("all")
end