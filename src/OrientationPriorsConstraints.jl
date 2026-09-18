using Distributions

mutable struct SHmaxAzimuthConstraint <: StressIndicator
    azimuth_dist
end
function StressStateLogLikelihood(s::StressState,
                                  i::SHmaxAzimuthConstraint)
    Sv, SHmax, SHmin, SV_dir, SHmax_dir, SHmin_dir = GeographicallyOrderStresses(s)
    azimuth = atan(SHmax_dir[2], SHmax_dir[1])
    return Distributions.logpdf(i.azimuth_dist, azimuth)
end
mutable struct EulerAnglePriorConstraint <: StressIndicator
    constraint_dist
    angle_symbol::Symbol
end

function StressStateLogLikelihood(s::StressState,
                                  i::EulerAnglePriorConstraint)
    angle_value = getfield(s, i.angle_symbol)
    return Distributions.logpdf(i.constraint_dist, angle_value)
end

mutable struct VerticalPrincipalStressConstraint <: StressIndicator
    component_symbol::Symbol
    angle_dist
    function VerticalPrincipalStressConstraint(component_symbol::Symbol, angle_std_dev::Float64)
        angle_dist = Distributions.VonMises(0.0, 1.0/angle_std_dev^2)
        return new(component_symbol, angle_dist)
    end
end


function StressStateLogLikelihood(s::StressState,
                                  i::VerticalPrincipalStressConstraint)
    S1_dir, S2_dir, S3_dir = PrincipalStressDirections(s)
    if i.component_symbol == :S1
        angle = acos(abs(S1_dir[3]))
    elseif i.component_symbol == :S2
        angle = acos(abs(S2_dir[3]))
    elseif i.component_symbol == :S3
        angle = acos(abs(S3_dir[3]))
    else
        error("Invalid component_symbol: $(i.component_symbol). Must be :S1, :S2, or :S3.")
    end
    return Distributions.logpdf(i.angle_dist, angle)
end

mutable struct BetaPriorConstraint <: StressIndicator
    beta_dist
end

function StressStateLogLikelihood(s::StressState,
                                  i::BetaPriorConstraint)
    return Distributions.logpdf(i.beta_dist,s.beta)
end

function StressStateLikelihood(s::StressState,
                               i::BetaPriorConstraint)
    return Distributions.pdf(i.beta_dist,s.beta)
end
