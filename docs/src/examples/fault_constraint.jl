# # Fault Constraint Analysis with MCMC
#
# This example demonstrates how to use MCMC sampling to analyze stress states
# under fault friction constraints. We'll set up a stress state at depth,
# define fault friction parameters, and use Bayesian sampling to explore
# the uncertainty in stress magnitudes.
#
# This test uses the MCMC sampler with the frictional faulting constraint
# to produce a probabilistic version of the Figure 2 from Zoback et al 2003
#
# ## Background
#
# In subsurface stress analysis, fault friction provides important constraints
# on the allowable stress states. The Coulomb friction criterion states that
# shear stress on a fault cannot exceed μ times the effective normal stress,
# where μ is the coefficient of friction. This creates bounds on the relative
# magnitudes of principal stresses.

using SOSAT
using Distributions
import PyPlot as plt

# ## Setting Up the Stress Environment
#
# We start by calculating the vertical stress (overburden) and pore pressure
# at a given depth. These form the foundation of our stress analysis.

TVD_m = 3000.0

avg_overburden_density = 2300.0 # kg/m3
g = 9.81 # m/s2

# Calculate vertical stress from overburden weight
sigv = avg_overburden_density * g * TVD_m / 1.0e6 # MPa
println("Vertical Stress: $(round(sigv, digits=2)) MPa")

# Estimate pore pressure using hydrostatic gradient (10 MPa/km)
pp_MPa = TVD_m * 10.0 / 1000.0
println("Pore Pressure: $(round(pp_MPa, digits=2)) MPa")

# ## Fault Friction Distribution
#
# We define the friction coefficient using a truncated log-normal distribution.
# This captures the natural variability in fault friction while keeping values
# within realistic bounds (0.1 to 0.9).

mudist = Distributions.truncated(Distributions.LogNormal(-0.53,0.1),0.1,0.9)

# Create the fault constraint object with both a distribution and starting point
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8)

# We make a plot to visualize the fault friction distribution
mu_array = LinRange(0.0,1.0,100)
pmu = pdf.(mudist, mu_array)
mu_fig = plt.figure()
mu_ax = mu_fig.add_subplot(111)
jnk = mu_ax.plot(mu_array, pmu, color="k")
mu_ax.set_xlabel("Fault Friction Coefficient")
mu_ax.set_ylabel("Probability Density")
mu_fig.savefig("Fault_Friction_PDF.png")

#md # ![Fault Friction PDF](Fault_Friction_PDF.png)

# ## Initial Stress State
#
# We define an initial guess for the stress state. In this case, we assume
# a normal faulting regime where the vertical stress is the maximum principal
# stress, and the horizontal stresses are fractions of the vertical stress.
SHmax = sigv * 0.9
SHmin = sigv * 0.7

# Euler angles defining stress tensor orientation (all zero = principal axes aligned)
alpha = deg2rad(0.0)
beta = deg2rad(0.0)
gamma = 0.0

# ## MCMC Sampling Setup
#
# We set up the MCMC sampler to explore uncertainty in stress magnitudes and
# orientations. The sampler will perturb the stress state and accept/reject
# samples based on how well they satisfy the fault friction constraints.

# Define stress indicators (constraints) - in this case, just fault friction
so = SOSAT.StressState(SHmax, SHmin, sigv, pp_MPa,
                        alpha,
                        beta,
                        gamma)

indicators = Vector{SOSAT.StressIndicator}(undef,1)
indicators[1] = fc

# Standard deviations for stress perturbations (SHmax, SHmin, Sv, pore pressure)
stress_std = [5.0,5.0,0.0,0.0]

# Standard deviations for angle perturbations (alpha, beta, gamma)
angle_std = [deg2rad(1.0),deg2rad(1.0),deg2rad(1.0)]

# MCMC parameters
Nsample = 1000000
Nburn = 1000
Nthin = 2

# ## Running the MCMC Sampler
#
# Create the sampler and evaluate the initial likelihood to ensure our
# starting point is reasonable.
sampler = SOSAT.MCMCSampler(so,
                        indicators,
                        stress_std,
                        angle_std)
initial_likelihood = SOSAT.EvaluateInitialLikelihood(sampler)
println("Initial likelihood: $(round(initial_likelihood, digits=4))")

# Run the MCMC sampling
println("Starting MCMC sampling...")
samples = SOSAT.Sample(sampler,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true)

println("Sampling complete. Generated $(length(samples)) samples.")

# ## Visualization: Stress Polygon
#
# The stress polygon is a powerful visualization tool that shows the
# relationship between maximum and minimum horizontal stresses. We overlay
# theoretical fault stability envelopes for different friction coefficients.

# Create stress polygon histogram from samples

stress_poly_ax, stress_poly_fig = SOSAT.StressPolygonHist(samples,
                                                              nbins=200,
                                                              shmax_component=1,
                                                              shmin_component=2)

    
# Overlay theoretical stability envelopes
mu = 0.8                            
SOSAT.plot_stress_polygon(stress_poly_ax, sigv, pp_MPa, 0.6; color="b")
SOSAT.plot_stress_polygon(stress_poly_ax, sigv, pp_MPa, 0.8; color="k")
stress_poly_ax.set_xlim(25.0,200.0)
stress_poly_ax.set_ylim(25.0,200.0)
stress_poly_fig.savefig("Fault_Constraint_test_polygon.png", dpi=150, bbox_inches="tight")

# Here is the resulting plot
# (This markdown will be processed by Literate.jl)
#md # ![Stress Polygon](Fault_Constraint_test_polygon.png)

plt.close(stress_poly_fig)

# ## Results Interpretation
#
# The stress polygon shows the probability distribution of horizontal stress
# magnitudes that are compatible with fault friction constraints. The colored
# lines represent theoretical stability boundaries:
#
# - **Blue line (μ = 0.6)**: Lower friction boundary
# - **Black line (μ = 0.8)**: Higher friction boundary  
#
# Stress states that plot above these lines would cause fault reactivation
# for the corresponding friction coefficient. The MCMC sampling naturally
# concentrates samples in regions that satisfy the friction constraints,
# providing a Bayesian estimate of stress uncertainty.
#
# The concentration of samples near the stability boundaries indicates that
# the fault friction constraint is actively limiting the allowable stress
# states, which is typical in tectonically active regions.
#
#
# The fault constraint does not provide any direct information about the principal stresses
# themselves, so we can also look at the marginal posterior distributions
# for the Euler angles
alpha_fig = SOSAT.StressComponentHist(samples, :alpha, label="Alpha (deg)", bins=50)
alpha_fig.savefig("Fault_Constraint_test_alpha_azimuth_marginal_posterior.png")
plt.close(alpha_fig)
beta_fig = SOSAT.StressComponentHist(samples, :beta, label="Beta (deg)", bins=50)
beta_fig.savefig("Fault_Constraint_test_Beta_deviation_marginal_posterior.png")
plt.close(beta_fig)
gamma_fig = SOSAT.StressComponentHist(samples, :gamma, label="Gamma (deg)", bins=50)
gamma_fig.savefig("Fault_Constraint_test_Gamma_euler_marginal_posterior.png")
plt.close(gamma_fig)
#md # ![Alpha Angle Posterior](Fault_Constraint_test_alpha_azimuth_marginal_posterior.png)
#md # ![Beta Angle Posterior](Fault_Constraint_test_Beta_deviation_marginal_posterior.png)
#md # ![Gamma Angle Posterior](Fault_Constraint_test_Gamma_euler_marginal_posterior.png)
#
# These plots show a near uniform distribution for the Euler angles in the ranges
# -pi <= alpha < pi
# -pi/2 <= beta <= pi/2
# -pi <= gamma < pi
#
# The non-uniformity in these distributions is due to the limited number of samples used in this
# example. With a larger number of samples the distributions would approach uniformity.