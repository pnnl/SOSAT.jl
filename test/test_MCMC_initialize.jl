using SOSAT
using Test
using Random
using Distributions

Random.seed!(1234)

# test with normal faulting and vertical/horizontal alignment
alpha = 0.0
beta = deg2rad(90.0)
gamma = deg2rad(70.0)
Sv = 1.1
SHmax = 1.3
SHmin = 1.2
Pp = 0.41
so = SOSAT.StressState(Sv, SHmax, SHmin, Pp, alpha, beta, gamma)

mudist = Distributions.truncated(Distributions.LogNormal(-0.53,0.1),0.1,0.9)
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8)

aphi_dist = Distributions.Uniform(0.0, 2.0)
aphi_indicator = SOSAT.AphiConstraint(aphi_dist)

indicators = [fc, aphi_indicator]
stress_std = [0.0, 0.05*Sv, 0.05*Sv, 0.0]  # last one a guess by brent
angle_std = [0.0, 0.0, deg2rad(1.0)]

sampler = SOSAT.MCMCSampler(so, indicators, stress_std, angle_std)

L_init_orig = SOSAT.EvaluteInitialLogLikelihood(sampler)
println("initial stress state:")
println("S1 = ", sampler.stress_state.S1)
println("S2 = ", sampler.stress_state.S2)
println("S3 = ", sampler.stress_state.S3)
println("Pp = ", sampler.stress_state.Pp)
println("alpha = ", sampler.stress_state.alpha)
println("beta = ", sampler.stress_state.beta)
println("gamma = ", sampler.stress_state.gamma)
println("Aphi= ", SOSAT.Aphi(sampler.stress_state))
println("L_init= ",L_init_orig)

fixed_stress_components = [1,4]
fixed_angle_components = [1,2]
new_stress_state = SOSAT.FindLikelyInitialState(sampler)
sampler.stress_state = new_stress_state


L_init = SOSAT.EvaluteInitialLogLikelihood(sampler)
println("new stress state:")
println("S1 = ", sampler.stress_state.S1)
println("S2 = ", sampler.stress_state.S2)
println("S3 = ", sampler.stress_state.S3)
println("Pp = ", sampler.stress_state.Pp)
println("alpha = ", sampler.stress_state.alpha)
println("beta = ", sampler.stress_state.beta)
println("gamma = ", sampler.stress_state.gamma)
println("Aphi= ", SOSAT.Aphi(sampler.stress_state))
println("new L_init= ",L_init)
@test L_init > L_init_orig