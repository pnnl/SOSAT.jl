using LinearAlgebra
using Distributions
"""
MiniFracClosure(Pc_dist)

Constrains the value of the minimum principal stress using a univariate
probability distribution for the value of the closure stress
"""
mutable struct MiniFracClosure <: StressIndicator
    Pc_dist
end



"""
FractureOpeningPressure(Popen_dist,
                       well_azimuth,
                       well_inclination)

A stress indicator using fracture opening pressure. The evaluation accounts
for rotated stress and inclined wells.
"""
mutable struct FractureOpeningPressure <: StressIndicator
    Popen_dist
    well_azimuth
    well_inclination
end


"""
FractureOpeningPressure(Popen_dist,
                       well_azimuth,
                       well_inclination)

A stress indicator using fracture opening pressure of a fracture with 
a specified orientation. The evaluation accounts for rotated stress and
inclined wells.
"""
mutable struct FractureOpeningPressure_wOrientation <: StressIndicator
    Popen_dist
    well_azimuth
    well_inclination
    dip_trend_dist
    dip_angle_dist
    # the dip trend and angle will be state variables that
    # are perturbed by the MCMC
    dip_trend_state
    dip_angle_state
    std_dip_trend_proposal
    std_dip_angle_proposal
end
"""
    Create a constructor function that will initialize the fracture orientation
    with a random draw from the dip trend and dip angle distributions and set
    a default value for the proposal distributions standard deviation
"""

function FractureOpeningPressure_wOrientation(Popen_dist,
                                              well_azimuth,
                                              well_inclination,
                                              dip_trend_dist,
                                              dip_angle_dist)
    ret_obj = FractureOpeningPressure_wOrientation(Popen_dist,
                                         well_azimuth,
                                         well_inclination,
                                         dip_trend_dist,
                                         dip_angle_dist,
                                         Distributions.rand(dip_trend_dist,1)[1],
                                         Distributions.rand(dip_angle_dist,1)[1],
                                         deg2rad(2.0),
                                         deg2rad(2.0))
    return ret_obj
end

function PerturbIndicatorState( fo_w_orient::FractureOpeningPressure_wOrientation)
    d_dip_trend = Distributions.rand(
          Distributions.Normal(0.0,
                               fo_w_orient.std_dip_trend_proposal),
                                    1)[1]
    d_dip_angle = Distributions.rand(
            Distributions.Normal(0.0,
                                 fo_w_orient.std_dip_angle_proposal),
                                     1)[1]
    fo_w_orient_new = deepcopy(fo_w_orient)
    fo_w_orient_new.dip_trend_state += d_dip_trend
    fo_w_orient_new.dip_angle_state += d_dip_angle
    return fo_w_orient_new
end


function StressStateLogLikelihood(s::StressState,
    i::FractureOpeningPressure_wOrientation)

    b = BS.WellState(s.Pp,i.well_inclination,i.well_azimuth)

    # compute theta and omega from dip trend and dip

    # first compute the dip vector from the current dip angle
    # and dip trend state. This will be in geographic coordinates
    dip_vec_g = BS.DipVectorFromDipAndTrend(i.dip_angle_state,
                                            i.dip_trend_state)
    # println("dip_vec_g= ",dip_vec_g)
    Rg = BS.GeographicToBoreholeRotation(b.delta, b.phi)
    dip_vec_b = Rg*dip_vec_g
    """
    println("i.dip_angle_state= ",rad2deg(i.dip_angle_state))
    println("i.dip_trend_state= ",rad2deg(i.dip_trend_state))
    println("i.dip_angle_dist= ",i.dip_angle_dist)
    println("dip_vec_b= ",dip_vec_b)
    """
    # compute the dip trend in borehole coordinates. This will be the
    # theta vector of the dip trend.
    dip_angle_b, dip_trend_b = BS.DipTrendFromDipVector(dip_vec_b)
    # println("dip_angle_b= ",rad2deg(dip_angle_b))
    # println("dip_trend_b= ",rad2deg(dip_trend_b))
    # The theta vector describing the fracture strike, where maximum
    # opening occurs, is 90 degrees away from this
    theta = dip_trend_b + 0.5*Base.MathConstants.pi
    # println("theta= ",rad2deg(theta))
    # the omega angle 90 degrees from the dip angle in 
    # borehole coordinates because we want the stress perpendicular
    # to the fracture 
    omega = dip_angle_b - deg2rad(90.0)
    # println("omega= ",rad2deg(omega))

    # then call a new function similar to FracturePressure but looking
    # for the opening pressure on a fracture with a specified orientation
    popen = BS.FractureOpeningPressure_wOrientation(s, b, theta, omega;
                    To=0.0, PR=0.25)
    # println("popen= ",popen)

    """
    We need to compare popen to the minimum far-field principal stress
    and set popen equal to the MAXIMUM of the two since fracture
    reopening (taking any significant fluid) cannot occur at anything
    less than the maximum of these two. We should also do this for the
    other minifrac opening constraint

    ^^^ no, we shouldn't and don't do this because this is specifically for a
    sleeve reopening test.
    """
    # smin = min(s.S1,s.S2,s.S3)
    # popen_final = max(popen,smin)
    
    popen_final = popen
    # println("popen_final= ",popen_final)
    
    # println("logpdf(popen_final)= ",Distributions.logpdf(i.Popen_dist,
    #                                                      popen_final))
    # println("logpdf(dip_trend)= ",Distributions.logpdf(i.dip_trend_dist,
    #                              i.dip_trend_state))
    # println("logpdf(dip_angle)= ",Distributions.logpdf(i.dip_angle_dist, 
    #                              i.dip_angle_state))
    log_likelihood = 0.0
    log_likelihood += Distributions.logpdf(i.dip_trend_dist, i.dip_trend_state)
    log_likelihood += Distributions.logpdf(i.dip_angle_dist, i.dip_angle_state)
    log_likelihood += Distributions.logpdf(i.Popen_dist,
                                           popen_final)
    return log_likelihood
end

function StressStateLikelihood(s::StressState,
    i::FractureOpeningPressure_wOrientation)

    loglikelihood = StressStateLogLikelihood(s,i)
    return exp(loglikelihood)
end
"""
InducedFractureOrientation(dip_trend_dist, dip_angle_dist)

A stress indicator that uses observed induced fracture orientations to
constrain the stress state. The struct is constructed using a probability
distribution for the dip trend and dip angle. Dip trend and dip angle
should be specified in geographic coordinates.
"""
mutable struct InducedFractureOrientation <: StressIndicator
    well_azimuth
    well_inclination
    dip_trend_dist
    dip_angle_dist
end


function StressStateLikelihood(s::StressState,i::MiniFracClosure)
    return Distributions.pdf(i.Pc_dist, minimum([s.S3,s.S2,s.S1]))
end

function StressStateLogLikelihood(s::StressState,i::MiniFracClosure)
    return Distributions.logpdf(i.Pc_dist, minimum([s.S3,s.S2,s.S1]))
end

function StressStateLikelihood(s::StressState,i::FractureOpeningPressure)

    b = BS.WellState(s.Pp,i.well_inclination,i.well_azimuth)
    Pfrac,tmin,omega = BS.FracturePressure(s,b;To=0.0,PR=0.25)
    # need to use the maximum of the fracturing pressure at the borehole
    # wall and the minimum principal stress
    #smin = min(s.S1,s.S2,s.S3)
    #popen = max(Pfrac,smin)
    popen = Pfrac
    return Distributions.pdf(i.Popen_dist, popen)

end

function StressStateLogLikelihood(s::StressState,i::FractureOpeningPressure)

    b = BS.WellState(s.Pp,i.well_inclination,i.well_azimuth)
    Pfrac,tmin,omega = BS.FracturePressure(s,b;To=0.0,PR=0.25)
    # need to use the maximum of the fracturing pressure at the borehole
    # wall and the minimum principal stress
    #smin = min(s.S1,s.S2,s.S3)
    #popen = max(Pfrac,smin)
    popen = Pfrac
    # println("popen= ",popen)
    return Distributions.logpdf(i.Popen_dist, popen)

end

# we will need a log likelihood for this before we can use it
# looks like it wasn't quite finished
function StressStateLikelihood(s::StressState, 
                               i::InducedFractureOrientation)

    b = BS.WellState(s.Pp,i.well_inclination,i.well_azimuth)
    Pfrac,theta_min,omega = BS.FracturePressure(s,b;To=0.0,PR=0.25)
    
    
    # this is a dip vector in the borehole coordinate system
    dip_vec_pred_BH = BS.BoreholeDipVector(theta_min,omega)
    
    Rgeo2BH = BS.GeographicToBoreholeRotation(b.delta,b.phi)

    dip_vec_pred_geo = Transpose(Rgeo2BH)*dip_vec_pred_BH
    
    dip_angle_pred, dip_trend_pred = BS.DipTrendFromDipVector(dip_vec_pred_geo)

    # the Julia VonMises distribution does not wrap as I expected
    # so the dip trend should be converted to the range 0-2pi and
    # the dip trend distribution should also have a mean value in
    # this range. It will not, for example, recognize 90 degrees
    # and -270 degrees as the same angle
    while dip_trend_pred < 0.0
        dip_trend_pred += 2.0*pi
    end

    while dip_trend_pred > 2.0*pi
        dip_trend_pred -= 2.0*pi
    end
    
    Pdip = Distributions.pdf(i.dip_angle_dist, dip_angle_pred)
    Ptrend = Distributions.pdf(i.dip_trend_dist, dip_trend_pred)

    return Pdip*Ptrend
end


"""
    StressStateLogLikelihood(s::StressState, 
                          i::InducedFractureOrientation)

Computes the likelihood of the given stress state given the
observed induced fracture orientation Struct
"""
function StressStateLogLikelihood(s::StressState, 
                                  i::InducedFractureOrientation)

    b = BS.WellState(s.Pp,i.well_inclination,i.well_azimuth)
    Pfrac,tmin,omega = BS.FracturePressure(s,b;To=0.0,PR=0.25)

    # this is a dip vector in the borehole coordinate system
    dip_vec_pred_BH = BS.BoreholeDipVector(tmin,omega)

    Rgeo2BH = BS.GeographicToBoreholeRotation(b.delta,b.phi)

    dip_vec_pred_geo = Transpose(Rgeo2BH)*dip_vec_pred_BH

    dip_angle_pred, dip_trend_pred = BS.DipTrendFromDipVector(dip_vec_pred_geo)
    while dip_trend_pred < 0.0
        dip_trend_pred += 2.0*pi
    end

    while dip_trend_pred > 2.0*pi
        dip_trend_pred -= 2.0*pi
    end
    

    logPdip = Distributions.logpdf(i.dip_angle_dist, dip_angle_pred)
    logPtrend = Distributions.logpdf(i.dip_trend_dist, dip_trend_pred)

    return logPdip + logPtrend
end