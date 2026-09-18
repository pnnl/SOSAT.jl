# # Breakout Constraint Examples
#
# ## Background
# 
# Bell and Gough were the first to suggest that elongation of boreholes along a particulate
# azimuth could be used to infer the orientation of the maximum horizontal stress
# (Bell and Gough, 1979). They suggested that the Kirsch equations for stress around
# a circular hole could be used to relate the observed breakout direction to the stress directions.
#
# The tangential stress at the borehole wall is given by (Jaeger and Cook, 2007):
#
# `` S_{\theta\theta} = (S_{Hmax} + S_{Hmin}) - 2(S_{Hmax} - S_{Hmin})cos(2\theta) - \Delta P ``
#
# where ``\Delta P`` is the difference between the mud pressure and pore pressure, ``P_p``. The radial stress is simply
#
# `` S_{rr} = \Delta P``
#
# In terms of the *effective* stress, the hoop stress becomes
#
# `` S'_{\theta\theta} = S_{Hmax} + S_{Hmin} - 2(S_{Hmax} - S_{Hmin})cos(2\theta) - 2 P_p - \Delta P ``
#
# Moos and Zoback (1990) suggested that breakouts occur when the hoop stress at the borehole wall
#
# `` S_{\theta \theta} = S_{Hmax} + S_{Hmin} - 2(S_{Hmax} - S_{Hmin})cos(2\theta) - 2 P_o \geq C ``
#
# where ``C`` is the compressive strength of the rock. They argue that if the mud pressure is close to
# the pore pressure, then the stress state at the borehole wall is biaxial (i.e. vertical stress and hoop stress).
# The cite the work of Wiebols and Cook (1968) to suggest that the compressive strength in a biaxial stress state is approximately
#
# `` C_b = C_o (1.0 + 0.6 \mu) ``
#
# where ``C_o`` is the uniaxial compressive strength and ``\mu`` is the coefficient of internal friction.
#
# Using this, and assuming that ``\mu \approx 0.6`` they use a value of ``C_b \approx 1.36 C_o`` to estimate the compressive strength.
#
# However, in general the mud weight is not equal to the pore pressure. For this more general case the Mohr-Coulomb failure criterion
# is often used to describe the failure criterion can be used. There are many ways of expressing the Mohr-Coulomb criterion, but one common form is:
#
# `` f(S_1, S_3) = S_1  - [(1+\mu^2)^{1/2} + \mu]^2 S_3 - 2 S_o [(1+\mu^2)^{1/2} + \mu] ``
#
# where ``S_o`` is the cohesion, ``\mu`` is the coefficient of internal friction, and ``f`` is a failure function defined
# such that failure occurs when ``f=0``, ``f<0`` indicates states below failure. Note here that the unconfined compressive
# strength is given by:
#
# `` UCS = C_o = 2 S_o [(1+\mu^2)^{1/2} + \mu] ``
#
# Applying this to the borehole wall and evaluating failure between ``S_{\theta\theta}`` and ``S_{rr}`` gives:
#
# `` f(S_{\theta\theta}, S_{rr}) = (S_{Hmax} + S_{Hmin}) - 2(S_{Hmax} - S_{Hmin})cos(2\theta) - 2 P_p - \Delta P - [(1+\mu^2)^{1/2} + \mu]^2 \Delta P - UCS ``
#
# The minimum value will occur at ``\theta = 90^\circ`` (i.e. aligned with ``S_{Hmin}``), so the breakout constraint becomes:
#
# `` f(S_{Hmax}, S_{Hmin}) = 3 S_{Hmax} - S_{Hmin} - 2 P_p - \Delta P - [(1+\mu^2)^{1/2} + \mu]^2 \Delta P - UCS ``
#
# Solving this for ``S_{Hmax}`` gives:
#
# `` S_{Hmax} = \frac{1}{3} [ S_{Hmin} + 2 P_p + \Delta P + [(1+\mu^2)^{1/2} + \mu]^2 \Delta P + UCS ] ``
#
# ## Example with Breakouts
#
# Here we use the continental crust example from Moos and Zoback
#

using SOSAT
using Distributions
using PyPlot
const plt = PyPlot

plt.matplotlib.use("Agg")

depth = 5000.0 # depth in meters
avg_overburden_density = 2600.0 # kg/m3
g = 9.81 # m/s2
sigv = avg_overburden_density * g * depth / 1.0e6 # MPa
println("Vertical Stress is ",sigv)
pp_MPa = (depth) * 10.0 / 1000.0 # assume 10 MPa/km
println("Pore pressure is ",pp_MPa)

mudist = Distributions.truncated(Distributions.LogNormal(-0.53,0.1),0.1,0.9);
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8);

mud_pressure_max_MPa = pp_MPa+0.1
println("max mud pressure is ",mud_pressure_max_MPa)

# note Young's modulus should be in MPa so 60.0e3 is 60 GPa
YM = 60.0e3
PR=0.25
CTE=5.0e-6
UCS = 200.00
UCS_dist = Distributions.Uniform(UCS-5.0,UCS+5.0)    

bo_param = SOSAT.BreakoutParameters(true, #::Any
                                breakout_azimuth=deg2rad(90.0), #breakout_azimuth,
                                breakout_azimuth_err=deg2rad(5.0), # breakout_azimuth_err,
                                UCS_dist=UCS_dist, # UCS_dist,
                                mud_pressure_dist=Distributions.Uniform(pp_MPa,mud_pressure_max_MPa), #  mud_pressure_dist,
                                mud_temperature_dist=Distributions.Uniform(50.0,51.0), #  mud_temperature_dist,
                                rock_friction_dist=Distributions.Uniform(0.5,0.7),#  rock_friction_dist,
                                well_azimuth=0.0, # well_azimuth, 
                                well_inclination=0.0, #well_inclination, 
                                formation_temperature=50.0,# formation_temperature, 
                                YM=YM, # YM, 
                                PR=PR, # PR,
                                CTE=CTE) #  CTE)

bo_param.state.UCS = UCS
bo_param.state.mud_pressure = pp_MPa
bo_param.state.mud_temperature = 50.5
bo_param.state.rock_friction = 0.6

breakout_indicator = SOSAT.Breakout(bo_param,
                                    0.0, # UCS proposal
                                    0.0, # mud pressure proposal
                                    0.0, # mud temperature proposal
                                    0.0); # rock friction proposal

indicators = Vector{SOSAT.StressIndicator}(undef,2)

indicators[1] = fc
indicators[2] = breakout_indicator

# oriented North prior to Euler rotations

S1 = 200.0
S2 = sigv
S3 = sigv
# oriented East prior to Euler rotations


alpha = deg2rad(0.0)
beta = deg2rad(0.0)
gamma = deg2rad(0.0)
so = SOSAT.StressState(S1, S2, S3, pp_MPa,
                        alpha,
                        beta,
                        gamma)
# perturb SHmax and SHmin but not Sv (which is S3)
# don't perturb pore pressure or Euler angles
stress_std = [5.0,5.0,0.0,0.0]

angle_std = [deg2rad(0.0),deg2rad(0.0),deg2rad(0.0)]

sampler = SOSAT.MCMCSampler(so,
                        indicators,
                        stress_std,
                        angle_std)
initial_likelihood = SOSAT.EvaluateInitialLikelihood(sampler, verbose=true)
println("Initial likelihood = ",initial_likelihood)
if initial_likelihood == 0.0
    println("Initial stress state has zero likelihood - adjust initial stress state")
    new_state, found = SOSAT.FindLikelyInitialState(sampler, num_samples=10000, verbose=false)
    if found == false
        println("Could not find a valid initial stress state with non-zero likelihood")
        stop()
    else
        println("Found a more likely initial stress state from reduced sampling")
        sampler.stress_state = new_state
        println(sampler.stress_state)
        initial_likelihood = SOSAT.EvaluateInitialLikelihood(sampler, verbose=true)
        println("New initial likelihood = ",initial_likelihood)
    end
end 

# MCMC parameters
# we seek a large number of samples to fully reproduce the figure from the paper
Nsample = 500000;

# burn-in period of 10k samples so it "forgets" the initial state
Nburn = 1000;

# thinning interval of 2 to reduce autocorrelation in the samples
Nthin = 2;

# The initial likelihood is non-zero so we can proceed with sampling.
println("Starting MCMC sampling...")
samples = SOSAT.Sample(sampler,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true);

                        # plot the stress samples in the stress polygon space
stress_poly_ax, stress_poly_fig = SOSAT.StressPolygonHist(samples, nbins=200, shmax_component=1, shmin_component=2);

SOSAT.plot_stress_polygon(stress_poly_ax, sigv, pp_MPa, 0.8)

shmin_array, shmax_array = SOSAT.GetBreakoutStressArrays(pp_MPa,
                                                            sigv,
                                                            UCS, # UCS
                                                            0.6,   # rock friction
                                                            0.0,   # deltaP
                                                            0.0,   # deltaT
                                                            YM,# YM
                                                            PR,  # PR
                                                            CTE, # CTE
                                                            100)

stress_poly_ax.plot(shmin_array, shmax_array, "r-", label="Breakout Constraint")
# plot the 1:1 line
stress_poly_ax.plot([0.0,400.0],[0.0,400.0], color="gray", linestyle="--")
stress_poly_ax.set_xlim(0.0, 400.0)
stress_poly_ax.set_ylim(0.0, 400.0)
stress_poly_fig.savefig("Breakout_constraint_example_continental_crust.png")
#md # ![Breakout constraint example continental crust](Breakout_constraint_example_continental_crust.png)
#
# This compares well with the figure from Moos and Zoback (1990):
#
# ## Example without Breakouts
# Now we use the same parameters but run the analysis indicating that there are no breakouts.
# This should constrain the stress state to be below the breakout constraint.
breakout_indicator.breakout_parameters.breakout_observed = false

S1 = 100.0
so = SOSAT.StressState(S1, S2, S3, pp_MPa,
                        alpha,
                        beta,
                        gamma)
sampler2 = SOSAT.MCMCSampler(so,
                        indicators,
                        stress_std,
                        angle_std)
initial_likelihood = SOSAT.EvaluateInitialLikelihood(sampler2, verbose=true)
println("Initial likelihood for case without breakouts = ",initial_likelihood)

# The initial likelihood is non-zero so we can proceed with sampling.
println("Starting MCMC sampling...")
samples2 = SOSAT.Sample(sampler2,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true);

                        # plot the stress samples in the stress polygon space
stress_poly_ax2, stress_poly_fig2 = SOSAT.StressPolygonHist(samples2, nbins=200, shmax_component=1, shmin_component=2);

SOSAT.plot_stress_polygon(stress_poly_ax2, sigv, pp_MPa, 0.8)


stress_poly_ax2.plot(shmin_array, shmax_array, "r-", label="Breakout Constraint")
# plot the 1:1 line
stress_poly_ax2.plot([0.0,400.0],[0.0,400.0], color="gray", linestyle="--")
stress_poly_ax2.set_xlim(0.0, 400.0)
stress_poly_ax2.set_ylim(0.0, 400.0)
stress_poly_fig2.savefig("Breakout_constraint_example_continental_crust_no_breakouts.png")
#md # ![Breakout constraint example continental crust](Breakout_constraint_example_continental_crust_no_breakouts.png)


# Now we run the deviated borehole example from Peska and Zoback (1995) and Zoback et al (2003).
Sv = 43.0; # MPa
println("Vertical Stress is ",Sv)

pp_MPa = 29.0; # MPa
println("Pore pressure is ",pp_MPa)

SHmin = 37.1; # MPa
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

alpha = deg2rad(0.0);
beta = deg2rad(90.0);
gamma = deg2rad(-35.0);
so = SOSAT.StressState(Sv, 40.0, SHmin, pp_MPa,
                        alpha,
                        beta,
                        gamma);

S_geo = SOSAT.GeographicStress(so);                        
println("Geographic Stress Tensor (MPa):");
println("|  $(round(S_geo[1,1], digits=2))  $(round(S_geo[1,2], digits=2))  $(round(S_geo[1,3], digits=2))|");
println("|  $(round(S_geo[2,1], digits=2))  $(round(S_geo[2,2], digits=2))  $(round(S_geo[2,3], digits=2))|");
println("|  $(round(S_geo[3,1], digits=2))  $(round(S_geo[3,2], digits=2))  $(round(S_geo[3,3], digits=2))|");

# perturb SHmax and SHmin but not Sv (which is S1) or the pore pressure
# we perturm SHmin a little bit to allow the minifrac closure constraint to
# control the magnitude of SHmin. We make the perturbation small so that we don't
# have a high rejection rate because of the narrow range of SHmin allowed by the minifrac closure constraint.
stress_std = [0.0,5.0,0.1,0.0];

# don't perturb alpha or beta but allow perturbation of gamma
# this will keep the principal stresses vertical/horizontal but allow the azimuth of the principal stresses to change.
angle_std = [deg2rad(0.0),deg2rad(0.0),deg2rad(2.0)];

#----------------------------
# Fault Friction distribution
#----------------------------
mudist = Distributions.truncated(Distributions.LogNormal(-0.53,0.1),0.1,0.9);
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8);

mfc_dist = Distributions.Uniform(SHmin-0.1, SHmin+0.1);
mfc_constraint = SOSAT.MiniFracClosure(mfc_dist);
borehole_deviation_deg = 32.0;
println("Borehole deviation is ",borehole_deviation_deg," degrees from vertical");

borehole_azimuth_deg = 35.0;
println("Borehole azimuth is ",borehole_azimuth_deg," degrees clockwise from North");

breakout_angle_deg = 17.0;
println("Breakout angle is ",breakout_angle_deg," degrees from the bottom of the borehole");

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
mu_internal = 1.0;

UCS_min = 20.0 / (1.0 + 0.6 * mu_internal);
UCS_max = 32.0 / (1.0 + 0.6 * mu_internal);
println("Using UCS range of ",UCS_min," to ",UCS_max," MPa for breakout calculation");

# We set a low initial UCS so that it is easier to find a valid initial state
UCS_init = 19.0;
#
#
# We set the probablity distribtion for the max mud pressure 0.5 MPa above
# pore pressure
mud_pressure_max_MPa = pp_MPa+0.5;

# However, for this exmaple we will make the mud pressure fixed so that it is
# only essential that it is between the pore pressure and max mud pressure.

# a mud weight that gives a mud pressure slightly above pore pressure
mud_pressure_MPa = pp_MPa+0.1;
println("mud pressure is ",mud_pressure_MPa);

# Young's and Poisson ratio are not given in the paper so we assume
# typical values for sedimentary rocks
YM = 6.0e3;
PR=0.25;
CTE=5.0e-6;
UCS_dist = Distributions.Uniform(UCS_min,UCS_max);

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
                                CTE=CTE); #  CTE)

bo_param.state.UCS = UCS_init;
bo_param.state.mud_pressure = mud_pressure_MPa;
bo_param.state.mud_temperature = 50.5;
bo_param.state.rock_friction = mu_internal;

# We allow perturbation of UCS and leave all other parameters fixed
breakout_indicator = SOSAT.Breakout(bo_param,
                                    0.1, # UCS proposal
                                    0.0, #  mud pressure proposal
                                    0.0, # mud temperature proposal
                                    0.0); # rock friction proposal


indicators = Vector{SOSAT.StressIndicator}(undef,3);


indicators[1] = fc;
indicators[2] = mfc_constraint;
indicators[3] = breakout_indicator;


Nsample = 100000;
Nburn = 1000;
Nthin = 2;

sampler = SOSAT.MCMCSampler(so,
                        indicators,
                        stress_std,
                        angle_std);
initial_likelihood = SOSAT.EvaluateInitialLikelihood(sampler, verbose=true);
println("Initial likelihood = ",initial_likelihood);


println("Starting MCMC sampling...");
samples = SOSAT.Sample(sampler,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true);




UCS = breakout_indicator.breakout_parameters.state.UCS;
println("UCS used in breakout calculation: ",UCS);
mu = breakout_indicator.breakout_parameters.state.rock_friction;
println("Rock friction used in breakout calculation: ",mu);
deltaP = breakout_indicator.breakout_parameters.state.mud_pressure - pp_MPa;
println("Delta P used in breakout calculation: ",deltaP);
deltaT = breakout_indicator.breakout_parameters.state.mud_temperature - breakout_indicator.breakout_parameters.formation_temperature;
println("Delta T used in breakout calculation: ",deltaT);
println("Pore pressure: ",pp_MPa);
println("Vertical Stress: ",Sv);

gamma_fig = SOSAT.StressComponentHist(samples, :gamma, label="Gamma Euler Angle (deg)", bins=50)
gamma_fig.savefig("GOA_test_Gamma_euler_marginal_posterior.png")
plt.close(gamma_fig)
#md # ![Gamma Euler Angle Marginal Posterior](GOA_test_Gamma_euler_marginal_posterior.png)

principal_stress_stereonet_fig = SOSAT.PrincipalStressStereonet(samples, alpha=0.01)
principal_stress_stereonet_fig.savefig("GOA_test_stress_stereonet.png")
plt.close(principal_stress_stereonet_fig)
#md # ![Stress Stereonet](GOA_test_stress_stereonet.png)
gamma_samples = SOSAT.GetGammaSamples(samples)
shmax_az = -1.0 .* gamma_samples .+ pi/2.0
shmax_samples = SOSAT.GetSHmaxSamples(samples)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.scatter(shmax_samples, rad2deg.(shmax_az), color="blue", alpha=0.5, s=1)
#ax.set_xlim(37.1,43.0)
ax.set_xlabel("SHmax (MPa)")
ax.set_ylabel("SHmax Azimuth (deg)")
ax.set_xlim(39.0,43.0)
ax.set_ylim(90.0, 180.0)
fig.savefig("GOA_test_SHmax_azimuth_vs_SHmax.png")
#md # ![SHmax vs SHmax Azimuth](GOA_test_SHmax_azimuth_vs_SHmax.png)
#
# This plot compares well with the plot in Figure 19 of Zoback et al (2003)

plt.close("all")


