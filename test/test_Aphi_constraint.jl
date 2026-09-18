using SOSAT
using Test
using Random
using Distributions
using PythonPlot
const plt = PythonPlot

Random.seed!(1234)
plt.matplotlib.use("Agg")

const GENERATE_PLOTS = haskey(ENV, "JULIA_TEST_PLOTS") || 
                       get(ENV, "JULIA_TEST_PLOTS", "false") == "true"

# test with normal faulting and vertical/horizontal alignment
alpha = 0.0
beta = 0.0
gamma = deg2rad(70.0)
Sv = 1.1
SHmax = 0.9
SHmin = 0.8
Pp = 0.41
so = SOSAT.StressState(Sv, SHmax, SHmin, Pp, alpha, beta, gamma)

aphi_dist = Distributions.Uniform(0.0, 2.0)
aphi_indicator = SOSAT.AphiConstraint(aphi_dist)
prob = SOSAT.StressStateLikelihood(so, aphi_indicator)
println("Testing AphiConstraint with NF and vert/horiz")
@test prob == 0.5

# test with strike-slip faulting and vertical/horizontal alignment
alpha = 0.0
beta = deg2rad(90.0)
gamma = deg2rad(70.0)
Sv = 1.1
SHmax = 1.2
SHmin = 0.8
Pp = 0.41
so = SOSAT.StressState(Sv, SHmax, SHmin, Pp, alpha, beta, gamma)

aphi_dist = Distributions.Uniform(0.0, 2.0)
aphi_indicator = SOSAT.AphiConstraint(aphi_dist)
prob = SOSAT.StressStateLikelihood(so, aphi_indicator)
println("Testing AphiConstraint with SS and vert/horiz")
@test prob == 0.5

# test with thrust faulting and vertical/horizontal alignment
alpha = 0.0
beta = deg2rad(90.0)
gamma = deg2rad(70.0)
Sv = 1.1
SHmax = 1.3
SHmin = 1.2
Pp = 0.41
so = SOSAT.StressState(Sv, SHmax, SHmin, Pp, alpha, beta, gamma)

aphi_dist = Distributions.Uniform(0.0, 2.0)
aphi_indicator = SOSAT.AphiConstraint(aphi_dist)
prob = SOSAT.StressStateLikelihood(so, aphi_indicator)
println("Testing AphiConstraint with TF and vert/horiz")
@test prob == 0.0

# test with normal faulting and rotated alignment
alpha = 0.0
beta = deg2rad(80.0)
gamma = deg2rad(70.0)
Sv = 1.1
SHmax = 0.9
SHmin = 0.8
Pp = 0.41
so = SOSAT.StressState(Sv, SHmax, SHmin, Pp, alpha, beta, gamma)

aphi_dist = Distributions.Uniform(0.0, 2.0)
aphi_indicator = SOSAT.AphiConstraint(aphi_dist)
prob = SOSAT.StressStateLikelihood(so, aphi_indicator)
println("Testing AphiConstraint with NF and beta=80")
println("prob= ",prob)
@test prob == 0.5


# Now we reset to the original stress state and run MCMC sampling with the Aphi constraint
alpha = 0.0
beta = deg2rad(90.0)
gamma = deg2rad(70.0)
Sv = 1.1
SHmax = 1.4
SHmin = 0.8
Pp = 0.41
so = SOSAT.StressState(Sv, SHmax, SHmin, Pp, alpha, beta, gamma)

# for testing we limit Aphi to 2.0 so all stress states are either
# normal faulting or strike slip. This will disallow the minimum horizontal
# stress from exceeding the vertical stress

aphi_dist = Distributions.Uniform(0.0, 2.0)
aphi_indicator = SOSAT.AphiConstraint(aphi_dist)

mudist = Distributions.truncated(Distributions.LogNormal(-0.53,0.1),0.1,0.9)
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8)

indicators = Vector{SOSAT.StressIndicator}(undef, 2)
indicators[1] = fc
indicators[2] = aphi_indicator

# perturb SHmax and SHmin but not Sv (which is S1)
# don't perturb pore pressure or Euler angles
stress_std = [0.0,0.5,0.5,0.0]
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

println("Starting MCMC sampling...")
samples, acceptance_ratio = SOSAT.Sample(sampler,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true)

# analyze the results
SHmin_samples = SOSAT.GetSHminSamples(samples)
SHmin_max = maximum(SHmin_samples)
println("Max SHmin sample: ",SHmin_max)
SHmin_min = minimum(SHmin_samples)
println("Min SHmin sample: ",SHmin_min)
@test all(SHmin_samples .<= Sv)
SHmin_median, SHmin_range, SHmin_IQR = SOSAT.GetStressComponentStatistics(samples, :SHmin)
println("SHmin median: ",SHmin_median," range: ",SHmin_range," IQR: ",SHmin_IQR)
if GENERATE_PLOTS
    # plot the stress samples in the stress polygon space
    stress_poly_ax, stress_poly_fig = SOSAT.StressPolygonHist(samples, nbins=200, shmax_component=2, shmin_component=3)
    SOSAT.plot_stress_polygon(stress_poly_ax, Sv, Pp, 0.8; color="k")
    # plot the 1:1 line
    stress_poly_ax.plot([0.0,4.0],[0.0,4.0], color="gray", linestyle="--")
    stress_poly_ax.set_xlim(0.0, 4.0)
    stress_poly_ax.set_ylim(0.0, 4.0)
    stress_poly_ax.set_xlabel("SHmin")
    stress_poly_ax.set_ylabel("SHmax")
    stress_poly_fig.savefig("Aphi_constraint_test_polygon.png")
    
end
# Now test that the strike-slip and reverse faulting regimes work
aphi_dist = Distributions.Uniform(1.0, 3.0)
aphi_indicator.Aphi_dist = aphi_dist

indicators[2] = aphi_indicator
# test with thrust faulting and vertical/horizontal alignment
alpha = 0.0
beta = deg2rad(90.0)
gamma = deg2rad(70.0)
Sv = 1.1
SHmax = 1.3
SHmin = 1.2
Pp = 0.41
so = SOSAT.StressState(Sv, SHmax, SHmin, Pp, alpha, beta, gamma)

# perturb SHmax and SHmin but not Sv (which is S1)
# don't perturb pore pressure or Euler angles
stress_std = [0.0,0.5,0.5,0.0]
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

println("Starting MCMC sampling...")

samples2, acceptance_ratio2 = SOSAT.Sample(sampler,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false,
                        verbose=true)

# analyze the results
println("for strike-slip and reverse faulting case")
println("Sv= ",Sv)
SHmax_samples = SOSAT.GetSHmaxSamples(samples2)
SHmax_max = maximum(SHmax_samples)
println("Max SHmax sample: ",SHmax_max)
SHmax_min = minimum(SHmax_samples)
println("Min SHmax sample: ",SHmax_min)
@test all(SHmax_samples .>= Sv)
SHmax_median, SHmax_range, SHmax_IQR = SOSAT.GetStressComponentStatistics(samples2, :SHmax)
println("SHmax median: ",SHmax_median," range: ",SHmax_range," IQR: ",SHmax_IQR)


if GENERATE_PLOTS
    # plot the stress samples in the stress polygon space
    stress_poly_ax2, stress_poly_fig2 = SOSAT.StressPolygonHist(samples2, nbins=200, shmax_component=2, shmin_component=3)
    SOSAT.plot_stress_polygon(stress_poly_ax2, Sv, Pp, 0.8; color="k")
    # plot the 1:1 line
    stress_poly_ax2.plot([0.0,4.0],[0.0,4.0], color="gray", linestyle="--")
    stress_poly_ax2.set_xlim(0.0, 4.0)
    stress_poly_ax2.set_ylim(0.0, 4.0)
    stress_poly_ax2.set_xlabel("SHmin")
    stress_poly_ax2.set_ylabel("SHmax")
    stress_poly_fig2.savefig("Aphi_constraint_test_polygon2.png")
    plt.close("all")
end