
"""
StressComponentConstraint(dist,component)

A generic stress indicator that can specify a probability distribution
for any specific component of the stress. This is usually used for the
lithostatic component of stress using a density log integral.
"""
mutable struct StressComponentConstraint <: StressIndicator
    dist
    component
end



function StressStateLikelihood(s::StressState,i::StressComponentConstraint)

    sig_i = 0.0
    if (i.component == 1)
        sig_i = s.S1
    elseif (i.component == 2)
        sig_i = s.S2
    elseif (i.component == 3)
        sig_i = s.S3
    else
        println("ERROR: invalid stress component ",i.component)
    end
    return Distributions.pdf(i.dist,sig_i)
end

function StressStateLogLikelihood(s::StressState,i::StressComponentConstraint)

    sig_i = 0.0
    if (i.component == 1)
        sig_i = s.S1
    elseif (i.component == 2)
        sig_i = s.S2
    elseif (i.component == 3)
        sig_i = s.S3
    else
        println("ERROR: invalid stress component ",i.component)
    end
    return Distributions.logpdf(i.dist,sig_i)
end


"""
StressComponentRatioConstraint(dist,component)

A stress constraint that can specify a probability distribution
for a ratio of two principal stresses.

    dist is the probability density function for the ratio of the two components
    component1 (numerator) and component2 (denominator) are the two components
"""
mutable struct StressComponentRatioConstraint <: StressIndicator
    comp_ratio_dist
    component1
    component2
end

function ComputeComponentRatio(s::StressState,i::StressComponentRatioConstraint)
    comp1 = 0.0
    if (i.component1 == 1)
        comp1 = s.S1
    elseif (i.component1 == 2)
        comp1 = s.S2
    elseif (i.component1 == 3)
        comp1 = s.S3
    else
        println("ERROR: invalid stress component ",i.component)
    end

    comp2 = 0.0
    if (i.component2 == 1)
        comp2 = s.S1
    elseif (i.component2 == 2)
        comp2 = s.S2
    elseif (i.component2 == 3)
        comp2 = s.S3
    else
        println("ERROR: invalid stress component ",i.component)
    end
    return comp1/comp2
end


function StressStateLikelihood(s::StressState,i::StressComponentRatioConstraint)
    comp_ratio = ComputeComponentRatio(s, i)
    return Distributions.pdf(i.comp_ratio_dist,comp_ratio)
end

function StressStateLogLikelihood(s::StressState,i::StressComponentRatioConstraint)
    comp_ratio = ComputeComponentRatio(s, i)
    return Distributions.logpdf(i.comp_ratio_dist,comp_ratio)
end


