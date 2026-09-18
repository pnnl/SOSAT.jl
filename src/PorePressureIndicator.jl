


"""
PorePressureConstraint(Pp_dist)

Constrains the magnitude of pore pressure using a specified distribution

end
"""
mutable struct PorePressureConstraint <: StressIndicator
    Pp_dist
end

"""
StressStateLikelihood(s::StressState,i::FaultConstraint)

This function computes the likelihood of the given stress state and fault
constraint indicator. The likelihood is computed as that portion of the
friction coefficient cumulative distribution function that exceeds the
critical friction coefficient for this stress state.

"""
function StressStateLikelihood(s::StressState,i::PorePressureConstraint)

    return Distributions.pdf(i.Pp_dist,s.Pp)

end

function StressStateLogLikelihood(s::StressState,i::PorePressureConstraint)

    return Distributions.logpdf(i.Pp_dist,s.Pp)

end
