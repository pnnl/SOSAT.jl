
"""
FaultConstraint(mu_dist)

Constrains the magnitude of stress differences using a
probability distribution for the Coulomb friction coefficient. It is constructed
with any univariate probability distribution for the friction coefficient, an
initial value for the friction coefficient, and a standard deviation for
a normal proposal distribution. Unlike the first version fo SOSAT, the friction
coefficient is a stochastic parameter that is perturbed as part of the MCMC
sampling.

end
"""
mutable struct FaultConstraint <: StressIndicator
    mu_dist
    mu
    std_mu_proposal
    function FaultConstraint(;mu_dist=Distributions.LogNormal(-0.53,0.1),
                              mu=0.6,
                              std_mu_proposal=0.05)
        new(mu_dist,mu,std_mu_proposal)
    end
end


function PerturbIndicatorState(fc_ind::FaultConstraint)

    d_mu = Distributions.rand(
             Distributions.Normal(0.0,fc_ind.std_mu_proposal),1)[1]


    fc_ind_new = deepcopy(fc_ind)

    fc_ind_new.mu += d_mu

    return fc_ind_new

end


"""
CriticalFriction(s::StressState)

This function returns the critical friction coefficient that would result
in shear failure between the maximum and minimum principal stresses at the
provided stress state. Returns Inf if pore pressure exceeds any principal stress
(hydraulic fracturing regime).
"""
function CriticalFriction(s::StressState)
    smin,sint,smax = sort([s.S1,s.S2,s.S3])
    
    # Check for hydraulic fracturing condition - pore pressure exceeding any principal stress
    if (s.Pp >= smin)
        return Inf  # Infinite friction required, indicating hydraulic fracturing regime
    end
    
    # Denominator is guaranteed positive when no tensile effective stresses exist
    arg = (smax-smin)/(smax+smin-2.0*s.Pp)
    if (arg > 1.0)
        arg = 1.0
    elseif (arg < 0.0)
        arg = 0.0
    end
    phic = asin.(arg)
    muc = tan(phic)
    return muc
end

"""
StressStateLikelihood(s::StressState,i::FaultConstraint)

This function computes the likelihood of the given stress state and fault
constraint indicator. If the current stress state is stable under the current
realization of the friction coefficient, then the likelihood is simply the
likelihood of the current realization of mu with respect to the prior distribution
for mu. If the current stress state is unstable with respect to the current realization
of mu, then the likelihood is zero.

"""
function StressStateLikelihood(s::StressState,i::FaultConstraint)
    muc = CriticalFriction(s)
    #return 1.0 - cdf(i.mu_dist,muc)
    mu_likelihood = Distributions.pdf(i.mu_dist,i.mu)
    if (muc<=i.mu)
        return mu_likelihood
    else
        return 0.0
    end

end

function StressStateLogLikelihood(s::StressState,i::FaultConstraint)
    
    muc = CriticalFriction(s)
    mu_loglikelihood = Distributions.logpdf(i.mu_dist,i.mu)
    
    if (muc <= i.mu)
        return mu_loglikelihood
    else
        return -Inf
    end
end
