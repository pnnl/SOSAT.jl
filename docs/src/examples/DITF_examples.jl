# # Drilling-Induced Tensile Fracture (DITF) Examples
#
# This example demonstrates how to use SOSAT>jl to reproduce Figure 7 from Moos and Zoback 1990
# by using the DITF constraint along with a fault friction constraint. The example begins
# with a minimal example that closely matches the figure (only closely because it does include
# some stochastic variation in the DITF parameters). A second example is provided that
# includes stress tensor rotation and shows how the DITF constraint can be used in a more
# realistic setting to account for uncertainty in stress orientations.

# ## Example with Fixed Principal Stress Directions

# For the first example we do not allow the stress tensor to rotate. We fix it so that
## SHmax is in the direction of the DITF.

using SOSAT
using Test
using Distributions
using PyPlot
const plt = PyPlot

plt.matplotlib.use("Agg")

# We take the parameters from the paper. The water depth is 4000 m, with a density of 1025 kg/m3
# The basalt layer is 1000 m thick with a density of 2800 kg/m3
water_depth = 4000.0; # m
water_density = 1025.0; # kg/m3
basalt_depth = 1000.0; # m
basalt_density = 2800.0; # kg/m3

g = 9.81; # m/s2

# Calculate vertical stress from overburden weight. SOSAT.jl requires consistent units for pressure and stress
# it can be any consistent unit system, but here we use MPa
sigv = (water_density * g * water_depth +
        basalt_density * g * basalt_depth) / 1.0e6;
println("Lithostatic stress is ",sigv," MPa")

# Calculate pore pressure assuming 10 MPa/km
pp_MPa = (water_depth + basalt_depth) * 10.0 / 1000.0;
println("Pore pressure is ",pp_MPa," MPa")

# # Stress Constraints

# Now we start creating the constraints. The workflow is to first create the stress indicators,
# then combine them into a Vector{StressIndicator} that will be passed to the MCMC sampler.

# ## Fault Friction constraint
#
# We model the fault friction using a truncated lognormal distribution with log-mean -0.53 and log-standard deviation 0.1
# This creates a distribution with a mean of approximately 0.6 and a standard deviation of approximately 0.15, but it is trimmed
# to only allow values between 0.1 and 0.9.
mudist = Distributions.truncated(Distributions.LogNormal(-0.53,0.1),0.1,0.9);
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8);

# The same distribution is used in the Fault Friction Example, so we won't plot it again here.

# ## DITF Constraint
#
# For a drilling-induced tensile fracture (DITF) constraint, we need to create a DITFParameters struct
# that holds all of the parameters for the DITF model. We create distributions for the uncertain parameters such as tensile strength and mud pressure.

# The first parameter we evaluate is the mud pressure. Interpreting the presence or absence of DITF in an image log for
# stress information requires knowledge of the mud pressure that the section of borehole has experienced between when it was
# drilled and when the image log was collected. For a DITF we care about what the maximum mud pressure it experienced during that
# time. Of course, we rarely actually know this. We probably know approximately what mud weight the well was drilled with, but we
# don't know how that static pressure changed as the drilling mud was circulated, as the drill string was tripped in and out
# of the hole, and when logging tools were tripped in and out. Therefore, there is significant uncertainty in the maximum
# mud pressure that is difficult to exactly estimate. In this example we assume the maximum mud pressure experienced by the section of
# borehole being analyzed was somewhere between the pore pressure of the formation and 6 MPa above the pore pressure.

mud_pressure_max_MPa = pp_MPa + 6.0;
println("max mud pressure is ",mud_pressure_max_MPa," MPa")

# To account for the thermo-mechanical effects on the stress state around the borehole, we need to provide
# Young's modulus, Poisson's ratio, and the coefficient of thermal expansion for the formation
# The Young's modulus needs to be in the same units as pressure and stress (MPa in this example) 
# so the a Young's modulus of 60 GPa is 60.0e3 MPa
YM = 60.0e3; # MPa
PR=0.25;

# Coefficient of thermal expansion in 1/C
CTE=5.0e-6; # 1/C

# Finally, we create a distribution for the tensile strength of the formation. In this example we
# use a uniform distribution between 0 and 20 MPa.
tensile_strength_dist = Distributions.Uniform(0.0,20.0)

# Now we can create the DITFParameters struct. The first parameter in the struct is a boolean
# that indicates whether a DITF was observed (true) or not observed (false). In this example we are modeling
# a DITF that was observed in the image log, so we set this to true.

# The next two parameters are the azimuth and azimuth uncertainty of the DITF. In this example we
# indicate that the DITF is oriented North-South (zero azimuth), and that there is a 5 degree uncertainty, so the azimuth is
# set to deg2rad(0.0) and the azimuth error is set to deg2rad(5.0). The next two parameters are the omega angle and its uncertainty.
# Omega is the angle between the DITF plane and the borehole axis. In this example we assume the borehole is vertical
# and the DITF is also vertical (DITF aligned with borehole axis), so omega is also zero with a 5 degree uncertainty. These orientations
# become much more important in the second example where we allow the stress tensor to rotate. However, if the stress directions
# picked for the fix example are not well aligned with the DITF orientations, then the MCMC sampler will have a very difficult time finding stress states
# with high likelihood and the values it finds can be very unexpected and unrealistic.

# A few other things to mention. We assume that the mud temperature is very close to 50 C and has very little uncertainty. The formation temperature is
# also assumed to be in the middle of this range (50.5 C).  This is to make it easier to reproduce the figure in the paper. In a real application there
# is likely to be more uncertainty in mud temperature.
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
                                    CTE=CTE);

# The DITFParameters struct contains a field called "state" that holds the current values of the stochastic parameters to make
# this repeatable. If these are not set they are just set to random realizations from the specified distributions.
# Here we set the initial values for these parameters. We set the tensile strength to 0                                    

ditf_param.state.tensile_strength=0.0;
ditf_param.state.mud_pressure=pp_MPa + 3.0;
ditf_param.state.mud_temperature=50.5;

# Now we create the DITF constraint object itself. This takes the DITFParameters struct along with proposal distribution standard deviations
# for the stochastic parameters. These standard deviations control how much the parameters are perturbed during MCMC sampling.
# To reproduce the example from the paper we set the proposal distributions to zero to make these deterministic at the value set above.
ditf_ind = SOSAT.DITF(ditf_param,
                        0.0,  # tensile strength proposal distribution
                        0.0, # mud pressure proposal distribution
                        0.0); # mud_temperature proposal distribution

# We now need to create a StressState object to set a starting point for the MCMC sampler.
# For the sampler to function it needs to start at a stress state with a non-zero likelihood
# If the chosen initial stress state has a zero likelihood the sampler has a method to try
# to find one. However, it will only find one by varying the stress magnitudes and directions
# that have nonzero proposal distributions.

# SOSAT.jl uses a geographic coordinate system where X1 is North, X2 is East, and X3 is vertical (downward positive).
# the positive downward part is important to bear in mind. Downward is chosen because this creates a right-handed coordinate system
# There are other common coordinate systems, such as East-North-Up, but SOSAT.jl uses North-East-Down.

# Conventionally, principal stresses are ordered such that S1 >= S2 >= S3. However, SOSAT doesn't require this and these can
# be viewed simply as labels. There are many combinations of principal stress orientations and magnitudes that can produce the same stress state.
# It is just important to understand what values you are setting here and how they rotate with the Euler angles.
# For this example we will honor the conventional ordering and set S1 >= S2 >= S3.

# In this example we set S1 to be 120 MPa, and S2 and S3 to be equal to the vertical stress. Remember that in the unrotated
# frame S1 is aligned North, S2 is East, and S3 is vertical.
S1 = 120.0;
S2 = sigv;
S3 = sigv;

# For this example we want to setup a strike-slip stress state where SHmax is aligned with the DITF azimuth (North-South).
# As it turns out, this is the case when the principal stresses are aligned with the geographic coordinates so
# we can set all of the Euler rotation angles to zero.

alpha = deg2rad(0.0);
beta = deg2rad(0.0);
gamma = deg2rad(0.0);

# We create the stress state object using principal stresses and Euler rotations
so = SOSAT.StressState(S1, S2, S3, pp_MPa,
                        alpha,
                        beta,
                        gamma);

# To double check we print out the geographic stress tensor
S_geo = SOSAT.GeographicStress(so);                        
println("Geographic Stress Tensor (MPa):")
println("|  $(round(S_geo[1,1], digits=2))  $(round(S_geo[1,2], digits=2))  $(round(S_geo[1,3], digits=2))|")
println("|  $(round(S_geo[2,1], digits=2))  $(round(S_geo[2,2], digits=2))  $(round(S_geo[2,3], digits=2))|")
println("|  $(round(S_geo[3,1], digits=2))  $(round(S_geo[3,2], digits=2))  $(round(S_geo[3,3], digits=2))|")

# It is a good practice to plot the resulting stress tensor in a stereonet to verify that the stress directions
# are what you expect. Here we create a lower-hemisphere projection stereonet plot of the principal stress directions.
# First we need to put the stress state object into a vector since the function for making a stereonet
# plot is meant to operate on a vector of stress state objects produced by MCMC sampling
stress_states = Vector{SOSAT.StressState}(undef,1);
stress_states[1] = so;
stereonet_fig = SOSAT.PrincipalStressStereonet(stress_states);
stereonet_fig.savefig("DITF_test_fixed_principal_stress_stereonet.png");
# Here is the resulting plot
#md # ![Stereonet Plot](DITF_test_fixed_principal_stress_stereonet.png)

indicators = Vector{SOSAT.StressIndicator}(undef,2);

indicators[1] = fc;
indicators[2] = ditf_ind;

# perturb SHmax (S1) and SHmin (S2) but not Sv (S3)
# don't perturb pore pressure (fourth argument)
stress_std = [5.0,5.0,0.0,0.0];

# for this first example we do not perturb the Euler angles
angle_std = [deg2rad(0.0),deg2rad(0.0),deg2rad(0.0)];

# MCMC parameters
# we seek a large number of samples to fully reproduce the figure from the paper
Nsample = 100000;

# burn-in period of 10k samples so it "forgets" the initial state
Nburn = 1000;

# thinning interval of 2 to reduce autocorrelation in the samples
Nthin = 2;

# We now create the MCMC sampler object with the initial stress state, the indicators, and the perturbation standard deviations

sampler = SOSAT.MCMCSampler(so,
                        indicators,
                        stress_std,
                        angle_std);

# We evaluate the initial likelihood to ensure our starting point is reasonable.                        
initial_likelihood = SOSAT.EvaluateInitialLikelihood(sampler, verbose=true);
println("Initial likelihood = ",initial_likelihood)

# The initial likelihood is non-zero so we can proceed with sampling.
println("Starting MCMC sampling...")
samples = SOSAT.Sample(sampler,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true);

# plot the marginal posterior distributions for S1 and S2
# and compute get the summary statistics for the stress components
S1fig = SOSAT.StressComponentHist(samples, :S1, label="S1 (MPa)")
S1fig.savefig("DITF_testS1_marginal_posterior_ditf_fixed.png")
#md # ![S1 Marginal Posterior](DITF_testS1_marginal_posterior_ditf_fixed.png)


S1_median, S1_range, S1_IQR = SOSAT.GetStressComponentStatistics(samples, :S1)
println("S1 median: ",round(S1_median)," range: ",round(S1_range)," IQR: ",round(S1_IQR))
S2fig = SOSAT.StressComponentHist(samples, :S2, label="S2 (MPa)")
S2fig.savefig("DITF_testS2_marginal_posterior_ditf_fixed.png")
#md # ![S2 Marginal Posterior](DITF_testS2_marginal_posterior_ditf_fixed.png)

S2_median, S2_range, S2_IQR = SOSAT.GetStressComponentStatistics(samples, :S2)
println("S2 median: ",round(S2_median)," range: ",round(S2_range)," IQR: ",round(S2_IQR))


# Now we look at the joint distribution between SHmax and SHmin in the stress polygon space
# get the state from the last sample to plot the DITF constraint in the figure
deltaT =  ditf_param.formation_temperature - ditf_param.state.mud_temperature;
deltaP = ditf_param.state.mud_pressure - pp_MPa;
# use the function in StressUtils.jl to get the DITF limit line to plot in the
# stress polygon space
shmin_array,shmax_array = SOSAT.getDITFStressArrays(pp_MPa, sigv, 0.8, deltaP, deltaT,
                                                YM, PR, CTE,
                                                ditf_param.state.tensile_strength, 100);
# plot the stress samples in the stress polygon space
stress_poly_ax, stress_poly_fig = SOSAT.StressPolygonHist(samples, nbins=200, shmax_component=1, shmin_component=2);
SOSAT.plot_stress_polygon(stress_poly_ax, sigv, pp_MPa, 0.8; color="k");
# plot the 1:1 line
stress_poly_ax.plot([0.0,200.0],[0.0,200.0], color="gray", linestyle="--");
stress_poly_ax.plot(shmin_array, shmax_array, color="r")
stress_poly_ax.set_xlim(45.0, 150.0);
stress_poly_ax.set_ylim(45.0, 150.0);
stress_poly_ax.set_xlabel("SHmin (MPa)");
stress_poly_ax.set_ylabel("SHmax (MPa)");
stress_poly_fig.savefig("DITF_test_polygon_ditf_fixed.png");
# Here is the resulting plot
#md # ![Stress Polygon](DITF_test_polygon_ditf_fixed.png)

# And we compute the correlation between S1 and S2
S_corr = SOSAT.GetStressComponentCorrelation(samples, :S1, :S2)
println("Correlation between S1 and S2: ",round(S_corr, digits=3))

# As is evident from the stress polygon plot, the DITF constraint produces a highly correlated
# relationship between SHmax and SHmin.


# ## Example with Variable Principal Stress Directions

# In this second example we allow the stress tensor to rotate by setting non-zero proposal
# distribution standard deviations for the Euler angles. This allows the MCMC sampler to explore
# different stress orientations to find those that are most consistent with the DITF observation.

# Here we will only allow the azimuth of SHmax to vary by setting a non-zero proposal standard deviation for gamma
# which is the rotation about the vertical axis. We leave alpha and beta at zero so that the vertical stress
# remains vertical.

# We use the same parameters as the previous example, but now we set the angle_std to non-zero values.
angle_std = [deg2rad(2.0),deg2rad(0.0),deg2rad(0.0)];

# We create a new MCMC sampler with the same initial stress state and indicators, but the new angle perturbations
sampler = SOSAT.MCMCSampler(so,
                        indicators,
                        stress_std,
                        angle_std);
# We evaluate the initial likelihood to ensure our starting point is reasonable.                        
initial_likelihood2 = SOSAT.EvaluateInitialLikelihood(sampler, verbose=true);
println("Initial likelihood = ",initial_likelihood2)
println("Starting MCMC sampling with variable principal stress directions...")
samples2 = SOSAT.Sample(sampler,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true);

# We plot the marginal distribution for alpha
alpha_fig = SOSAT.StressComponentHist(samples2, :alpha, label="Alpha (deg)")
alpha_fig.savefig("DITF_test_alpha_marginal_posterior_ditf_variable.png")
#md # ![Alpha Marginal Posterior](DITF_test_alpha_marginal_posterior_ditf_variable.png)

# As we would expect, the distribution for alpha remains tightly centered around zero degrees and mirrors the
# uncertainty set for the orientation of the DITF.

# We plot the marginal distribution for beta
beta_fig = SOSAT.StressComponentHist(samples2, :beta, label="Beta (deg)")
beta_fig.savefig("DITF_test_beta_marginal_posterior_ditf_variable.png")
#md # ![Beta Marginal Posterior](DITF_test_beta_marginal_posterior_ditf_variable.png)

# Since for this exaple we only allowed gamma to vary, the distribution for beta was not perturbed and is exactly zero for all samples

# We plot the marginal distribution for gamma
gamma_fig = SOSAT.StressComponentHist(samples2, :gamma, label="Gamma (deg)")
gamma_fig.savefig("DITF_test_gamma_marginal_posterior_ditf_variable.png")
#md # ![Gamma Marginal Posterior](DITF_test_gamma_marginal_posterior_ditf_variable.png)
#

# Since for this exaple we only allowed gamma to vary, the distribution for gamma was also not perturbed and is exactly zero for all samples

# plot the marginal posterior distributions for S1 and S2
S1fig2 = SOSAT.StressComponentHist(samples2, :S1, label="S1 (MPa)")
S1fig2.savefig("DITF_testS1_marginal_posterior_ditf_variable.png")
#md # ![S1 Marginal Posterior](DITF_testS1_marginal_posterior_ditf_variable.png)
S1_median2, S1_range2, S1_IQR2 = SOSAT.GetStressComponentStatistics(samples2, :S1)
println("S1 median: ",round(S1_median2)," range: ",round(S1_range2)," IQR: ",round(S1_IQR2))
S2fig2 = SOSAT.StressComponentHist(samples2, :S2, label="S2 (MPa)")
S2fig2.savefig("DITF_testS2_marginal_posterior_ditf_variable.png")
#md # ![S2 Marginal Posterior](DITF_testS2_marginal_posterior_ditf_variable.png)
S2_median2, S2_range2, S2_IQR2 = SOSAT.GetStressComponentStatistics(samples2, :S2)
println("S2 median: ",round(S2_median2)," range: ",round(S2_range2)," IQR: ",round(S2_IQR2))

# Lets look at the statistics for alpha
alpha_median, alpha_range, alpha_IQR = SOSAT.GetStressComponentStatistics(samples2, :alpha)
println("Alpha median: ",round(rad2deg(alpha_median), digits=2)," range: ",round(rad2deg(alpha_range), digits=2)," IQR: ",round(rad2deg(alpha_IQR), digits=2))


# plot the stress samples in the stress polygon space
stress_poly_ax2, stress_poly_fig2 = SOSAT.StressPolygonHist(samples2, nbins=200, shmax_component=1, shmin_component=2);
SOSAT.plot_stress_polygon(stress_poly_ax2, sigv, pp_MPa, 0.8; color="k");
# plot the 1:1 line
stress_poly_ax2.plot([0.0,200.0],[0.0,200.0], color="gray", linestyle="--");
stress_poly_ax2.plot(shmin_array, shmax_array, color="r")
stress_poly_ax2.set_xlim(45.0, 150.0);
stress_poly_ax2.set_ylim(45.0, 150.0);
stress_poly_ax2.set_xlabel("SHmin (MPa)");
stress_poly_ax2.set_ylabel("SHmax (MPa)");
stress_poly_fig2.savefig("DITF_test_polygon_ditf_variable.png");
# Here is the resulting plot
#md # ![Stress Polygon](DITF_test_polygon_ditf_variable.png)

# Now we plot the stereonet of the principal stress directions

stereonet_fig2 = SOSAT.PrincipalStressStereonet(samples2, alpha=0.01);
stereonet_fig2.savefig("DITF_test_variable_principal_stress_stereonet.png");
# Here is the resulting plot
#md # ![Stereonet Plot](DITF_test_variable_principal_stress_stereonet.png)

# ## Example with Absence of DITF

# In this third example we consider the case where there is no DITF observed in the image log.
# This provide no information on the stress orientation, but does provide an upper limit on the magnitude of SHmax.
# To reflect this lack of information about the orientation of the stress tensor, we allow variation of in alpha
# 

# We use the same parameters as the previous example, but now we set the angle_std to non-zero values.
angle_std = [deg2rad(2.0),deg2rad(0.0),deg2rad(0.0)];

# We need to modify the DITF indicator to reflect the absence of a DITF

ditf_ind.ditf_parameters.DITF_observed = false;

# We leave all other settings the same. When there is no DITF observed the angles for the DITF orientation
# are simply ingored, though they remain required inputs to the constructor.


# We evaluate the initial likelihood to ensure our starting point is reasonable.                        
initial_likelihood3 = SOSAT.EvaluateInitialLikelihood(sampler, verbose=true);
println("Initial likelihood = ",initial_likelihood3)
println("Starting MCMC sampling with variable principal stress directions...")
samples3 = SOSAT.Sample(sampler,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true);

# We plot the marginal distribution for alpha
alpha_fig = SOSAT.StressComponentHist(samples3, :alpha, label="Alpha (deg)")
alpha_fig.savefig("DITF_test_alpha_marginal_posterior_no_ditf_variable.png")
#md # ![Alpha Marginal Posterior](DITF_test_alpha_marginal_posterior_no_ditf_variable.png)

# As we would expect, alpha varies uniformly across its entire valid range

        
# plot the marginal posterior distributions for S1 and S2
S1fig2 = SOSAT.StressComponentHist(samples3, :S1, label="S1 (MPa)")
S1fig2.savefig("DITF_testS1_marginal_posterior_no_ditf_variable.png")
#md # ![S1 Marginal Posterior](DITF_testS1_marginal_posterior_no_ditf_variable.png)
S1_median2, S1_range2, S1_IQR2 = SOSAT.GetStressComponentStatistics(samples3, :S1)
println("S1 median: ",round(S1_median2)," range: ",round(S1_range2)," IQR: ",round(S1_IQR2))
S2fig2 = SOSAT.StressComponentHist(samples3, :S2, label="S2 (MPa)")
S2fig2.savefig("DITF_testS2_marginal_posterior_no_ditf_variable.png")
#md # ![S2 Marginal Posterior](DITF_testS2_marginal_posterior_no_ditf_variable.png)
S2_median2, S2_range2, S2_IQR2 = SOSAT.GetStressComponentStatistics(samples3, :S2)
println("S2 median: ",round(S2_median2)," range: ",round(S2_range2)," IQR: ",round(S2_IQR2))

# Lets look at the statistics for alpha
alpha_median, alpha_range, alpha_IQR = SOSAT.GetStressComponentStatistics(samples3, :alpha)
println("Alpha median: ",round(rad2deg(alpha_median), digits=2)," range: ",round(rad2deg(alpha_range), digits=2)," IQR: ",round(rad2deg(alpha_IQR), digits=2))


# plot the stress samples in the stress polygon space
stress_poly_ax2, stress_poly_fig2 = SOSAT.StressPolygonHist(samples3, nbins=200, shmax_component=1, shmin_component=2);
SOSAT.plot_stress_polygon(stress_poly_ax2, sigv, pp_MPa, 0.8; color="k");
# plot the 1:1 line
stress_poly_ax2.plot([0.0,200.0],[0.0,200.0], color="gray", linestyle="--");
stress_poly_ax2.plot(shmin_array, shmax_array, color="r")
stress_poly_ax2.set_xlim(45.0, 150.0);
stress_poly_ax2.set_ylim(45.0, 150.0);
stress_poly_ax2.set_xlabel("SHmin (MPa)");
stress_poly_ax2.set_ylabel("SHmax (MPa)");
stress_poly_fig2.savefig("DITF_test_polygon_no_ditf_variable.png");
# Here is the resulting plot
#md # ![Stress Polygon](DITF_test_polygon_no_ditf_variable.png)
#
# As expected the absence of a DITF observation produces an upper limit on SHmax, but
# with the admissible stress states lying below the DITF limit line. There is one notable
# exception to the curves shown in Moos and Zoback 1990, which is in the upper portion of the 
# thrust faulting regime there is a trimmed region. This is the result of the formation of
# tensile horizontal fractures. The vertical stress concentration around a vertical borehole
# is
#
# `` \sigma_{zz} = S_v - 2\nu (S_hmax - S_hmin) \cos(2\theta) - P_p - \alpha \Delta T E/(1-\nu) ``

# Now we plot the stereonet of the principal stress directions

stereonet_fig2 = SOSAT.PrincipalStressStereonet(samples3, alpha=0.01);
stereonet_fig2.savefig("DITF_test_variable_principal_stress_stereonet_no_ditf.png");
# Here is the resulting plot
#md # ![Stereonet Plot](DITF_test_variable_principal_stress_stereonet_no_ditf.png)