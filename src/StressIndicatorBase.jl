
using .BoreholeStress: StressState
BS = BoreholeStress

"""
Import the stress state struct from the BoreholeStress module, which defines
the in situ state of stress using three principal stresses, a pore pressure,
and three rotation angles.

"""

"""
StressIndicator is an abstract type from which all concrete stress indicator
types inherit. Each stress indicator needs to provide a function with the
following name and signature:

    function StressStateLikelihood(s::StressState,i::<StressIndicatorStruct>)

This function should return a single floating point value that is the likelihood
of the given stress state given the indicator. This function should include the
likelihood of any stochastic parameters that belong to the indicator (strength,
friction, etc.).

For indicators with uncertain parameters (strength, friction, etc.), they should
also provide a function with the following name and signature:

    function PerturbIndicatorState(i::<StressIndicatorStruct>)

This function should return a new indicator with perturbed stochastic variables.
If the proposed MCMC sample is accepted, this new perturbed indicator will
replace the old one.

"""
abstract type StressIndicator end

"""
     PerturbIndicatorState(ind::StressIndicator)
This is a default function for indicators that do not have
state variables. It simply returns the same indicator as was
passed in without modification.
"""
function PerturbIndicatorState(ind::StressIndicator)

    return ind
end


mutable struct UniformDistribution
    LL
    UL
end

function pdf(x,dist::UniformDistribution)
    density = 1.0/(dist.UL-dist.LL)
    if ((x<=dist.UL) && (x>=dist.LL))
        return density
    else
        return 0.0
    end
end

function logpdf(x,dist::UniformDistribution)
    return log(pdf(x,dist))
end

function StressTotalLikelihood(strial::StressState,
                               indicators::AbstractVector{StressIndicator};
                               verbose=false)
    L = 1.0
    n=1
    for i in indicators
        Li= StressStateLikelihood(strial,i)
        L *= Li
        if (verbose)
            println("Likelihood for indicator ",n," = ",Li)
        end
        n+=1
    end
    return L
end

function StressTotalLogLikelihood(strial::StressState,
                                  indicators::AbstractVector{StressIndicator};
                                  verbose=false)
    L = 0.0
    n=1

    alpha_trial = strial.alpha

    for i in indicators
        Li= StressStateLogLikelihood(strial,i)

        # debug
        # print("$Li, ")

        L += Li
        if (verbose)
            println("Log Likelihood for indicator ",n," = ",Li)
        end
        n+=1
    end

    # debug
    # println()
    
    return L
end

function PerturbStressStateFixedOrder(s::StressState, stress_std, angle_std)

    # perturb S1, ensuring that it remains more compressive than S2
    S1_dist = Distributions.truncated(Distributions.Normal(0.0,stress_std[1]),-(s.S1-s.S2),Inf)
    dS1 =Distributions.rand(S1_dist,1)[1]
    S1_new = s.S1+dS1

    # purturb S2 ensuring that it remains greater than s3 and does not exceed S1
    S2_dist = Distributions.truncated(Distributions.Normal(0.0,stress_std[2]),-(s.S2-s.S3),(S1_new-s.S2))
    dS2 =Distributions.rand(S2_dist,1)[1]
    S2_new = s.S2+dS2

    # perturb S3 ensuring that it remains greater than zero and less than S2
    S3_dist = Distributions.truncated(Distributions.Normal(0.0,stress_std[3]),-s.S3,(S2_new-s.S3))
    dS3 =Distributions.rand(S3_dist,1)[1]
    S3_new = s.S3+dS3

    # perturb pore pressure, ensuring that it remains compressive
    Pp_dist = Distributions.truncated(Distributions.Normal(0.0,stress_std[4]),-(s.Pp),Inf)
    dPp = Distributions.rand(Pp_dist,1)[1]
    Pp_new = s.Pp+dPp

    alpha_dist = Distributions.Normal(0.0,angle_std[1])
    beta_dist = Distributions.Normal(0.0,angle_std[2])
    gamma_dist = Distributions.Normal(0.0,angle_std[3])


    dalpha = Distributions.rand(alpha_dist,1)[1]
    dbeta = Distributions.rand(beta_dist,1)[1]
    dgamma = Distributions.rand(gamma_dist,1)[1]
    
    snew = StressState(S1_new,
                       S2_new,
                       S3_new,
                       Pp_new,
                       s.alpha+dalpha,
                       s.beta+dbeta,
                       s.gamma+dgamma)

    # cannonicalize the angles using Rotations.jl
    BS.natural_canonicalize!(snew)
    
    return snew

end


function PerturbStressState(s::StressState, stress_std, angle_std)

    # perturb S1, ensuring that it remains compressive
    S1_dist = Distributions.truncated(Distributions.Normal(0.0,stress_std[1]),-(s.S1),Inf)
    dS1 =Distributions.rand(S1_dist,1)[1]
    S1_new = s.S1+dS1

    # purturb S2 ensuring that it remains compressive
    S2_dist = Distributions.truncated(Distributions.Normal(0.0,stress_std[2]),-(s.S2),Inf)
    dS2 =Distributions.rand(S2_dist,1)[1]
    S2_new = s.S2+dS2

    # perturb S3 ensuring that it remains compressive
    S3_dist = Distributions.truncated(Distributions.Normal(0.0,stress_std[3]),-(s.S3),Inf)
    dS3 =Distributions.rand(S3_dist,1)[1]
    S3_new = s.S3+dS3

    # perturb pore pressure, ensuring that it remains compressive
    Pp_dist = Distributions.truncated(Distributions.Normal(0.0,stress_std[4]),-(s.Pp),Inf)
    dPp = Distributions.rand(Pp_dist,1)[1]
    Pp_new = s.Pp+dPp

    alpha_dist = Distributions.Normal(0.0,angle_std[1])
    beta_dist = Distributions.Normal(0.0,angle_std[2])
    gamma_dist = Distributions.Normal(0.0,angle_std[3])

    dalpha = Distributions.rand(alpha_dist,1)[1]
    dbeta = Distributions.rand(beta_dist,1)[1]
    dgamma = Distributions.rand(gamma_dist,1)[1]

    snew = StressState(S1_new,
                       S2_new,
                       S3_new,
                       Pp_new,
                       s.alpha+dalpha,
                       s.beta+dbeta,
                       s.gamma+dgamma)
    #println("alpha = ",s.alpha, " dalpha = ",dalpha," new alpha = ", snew.alpha)
    # cannonicalize the angles using Rotations.jl
    BS.natural_canonicalize!(snew)
    
    
    return snew

end

function PerturbIndicators(indicators::AbstractVector{StressIndicator})
    itrial = Vector{StressIndicator}(undef,length(indicators))
    n = 1
    for i in indicators
        itrial[n] = PerturbIndicatorState(i)
        n += 1
    end

    return itrial
end
