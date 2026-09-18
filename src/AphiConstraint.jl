using LinearAlgebra

"""
AphiConstraint(Aphi_dist)

Constrains the faulting regime by setting a probability distribution for the
A_phi parameter.

The scalar measure of the faulting regime is defined
        in Simpson (1997):

            phi = (sigma2-sigma3)/ (sigma1-sigma3)
            n = 0, 1, and 2 respectively for normal, strike-slip,
            and reverse faulting regime respectively.
            A_phi = (n+0.5) + (-1)^n*(phi-0.5)

        Using this definition, A_phi value ranges from 0 to 3, where
        [0,1] corresponds to normal faulting regime,
        [1,2] corresponds to strike-slip faulting regime,  and
        [2,3] corresponds to reverse faulting regime.

"""

mutable struct AphiConstraint <: StressIndicator
    Aphi_dist
end


"""
FindMostVerticalComponent(s::StressState)

This function determines which of the three principal stresses in the
StressState is aligned closest to the vertical direction. This is then
used to calculate the A_phi parameter. This function returns an integer
with 1 indicating S1 is closest to vertical, 2 indicating S2 and 3 indicating
S3.
"""
function FindMostVerticalComponent(s::StressState)
    S1_dir, S2_dir, S3_dir = BS.PrincipalStressDirections(s)
    # now find which has the largest magnitude in the 3 dir
    third_components = [abs(S1_dir[3]), abs(S2_dir[3]), abs(S3_dir[3])]
    index_of_largest = argmax((third_components))
    return index_of_largest
end

"""
Aphi(Sv, SHmax, SHmin)

Computes the scalar stress regime measure A_phi
"""
function Aphi(Sv, SHmax, SHmin)
    if (Sv > SHmax)
        # normal faulting
        n = 0
        S1 = Sv
        S2 = SHmax
        S3 = SHmin
    elseif (SHmax >= Sv && SHmin <= Sv)
        # strike-slip faulting
        n = 1
        S1 = SHmax
        S2 = Sv
        S3 = SHmin
    else
        # reverse faulting
        n = 2
        S1 = SHmax
        S2 = SHmin
        S3 = Sv
    end

    phi = (S2 - S3) / (S1 - S3)
    A_phi = (n + 0.5) + (-1)^n * (phi - 0.5)
    return A_phi
end

function Aphi(s::StressState)
    Sv, SHmax, SHmin = GeographicallyOrderStresses(s)
    return Aphi(Sv, SHmax, SHmin)
end

function StressStateLikelihood(s::StressState,i::AphiConstraint)
    return Distributions.pdf(i.Aphi_dist,Aphi(s))
end

function StressStateLogLikelihood(s::StressState, i::AphiConstraint)
    return Distributions.logpdf(i.Aphi_dist, Aphi(s))
end