# SOSAT.jl

Documentation for SOSAT.jl - A Julia package for Markov Chain Monte Carlo analyses of subsurface stress states.

## Overview

This package provides tools for:
- Stress tensor analysis
- Coordinate system transformations
- MCMC sampling of stress states
- Uncertainty quantification

## Quick Start

```julia
using SOSAT
using Distributions

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

samples = SOSAT.Sample(sampler,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true)

mu = 0.8                            
stress_poly_ax, stress_poly_fig = SOSAT.StressPolygonHist(samples, nbins=200)


StressUtils.plot_stress_polygon(stress_poly_ax, sigv, pp_MPa, 0.6; color="b")
StressUtils.plot_stress_polygon(stress_poly_ax, sigv, pp_MPa, 0.8; color="k")
stress_poly_ax.set_xlim(25.0,200.0)
stress_poly_ax.set_ylim(25.0,200.0)
stress_poly_fig.savefig("Fault_Constraint_test_polygon.png")
plt.close(stress_poly_fig)
