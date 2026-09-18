
"""
BreakoutState(UCS,mud_pressure,rock_friction)

The BreakoutState struct holds the (scalar) values for the uncertain parameters that control
    the breakout analysis (beyond the stress state). Since these parameters
    are uncertain they are included in the inference, and so their state needs
    to be perturbed and accepted or rejected with the MCMC algorithm along
    with the stress state.
"""
mutable struct BreakoutState
    UCS
    mud_pressure
    mud_temperature
    rock_friction
end

"""
BreakoutParameters(breakout_observed,
                   breakout_azimuth,
                   breakout_azimuth_err,
                   UCS_dist,
                   mud_pressure_dist,
                   mud_temperature_dist,
                   rock_friction_dist,
                   well_azimuth,
                   well_inclination,
                   formation_temperature,
                   YM,
                   PR,
                   CTE,
                   bo_state)

The BreakoutParameters struct holds all of the parameters (deterministic and
    stochastic) that are needed to constrain the stress state using the
    presence or absence of breakouts. The struct holds these parameters as
    well as keeping a BreakoutState struct that holds the current value of the
    state variables. These are perturbed and then accepted or rejected as
    part of the MCMC sampling.


"""
mutable struct BreakoutParameters
    breakout_observed
    breakout_azimuth
    breakout_azimuth_err
    UCS_dist
    mud_pressure_dist
    mud_temperature_dist
    rock_friction_dist
    well_azimuth
    well_inclination
    formation_temperature
    YM
    PR
    CTE
    state::BreakoutState
    """

    function BreakoutParameters(breakout_observed,
                                breakout_azimuth,
                                breakout_azimuth_err,
                                UCS_dist,
                                mud_pressure_dist,
                                mud_temperature_dist,
                                rock_friction_dist,
                                well_azimuth,
                                well_inclination,
                                formation_temperature,
                                YM,
                                PR,
                                CTE)

    Creates a new BreakoutParameter struct and initializes the BreakoutState
        variable using a random realization of the relevant distributions
    """
    function BreakoutParameters(breakout_observed;
                                breakout_azimuth,
                                breakout_azimuth_err,
                                UCS_dist,
                                mud_pressure_dist,
                                mud_temperature_dist,                                
                                rock_friction_dist,
                                well_azimuth,
                                well_inclination,
                                formation_temperature,
                                YM,
                                PR,
                                CTE)

        bo_state = BreakoutState(Distributions.rand(UCS_dist,1)[1],
                                 Distributions.rand(mud_pressure_dist,1)[1],
                                 Distributions.rand(mud_temperature_dist,1)[1],
                                 Distributions.rand(rock_friction_dist,1)[1]

        )
        new(breakout_observed,
            breakout_azimuth,
            breakout_azimuth_err,
            UCS_dist,
            mud_pressure_dist,
            mud_temperature_dist,
            rock_friction_dist,
            well_azimuth,
            well_inclination,
            formation_temperature,
            YM,
            PR,
            CTE,
            bo_state)
    end
end


"""
    Breakout(breakout_parameters,
             std_UCS_proposal,
             std_mud_pressure_proposal,
             std_mud_temperature_proposal,
             std_rock_friction_proposal)

A stress indicator using the presence  or absence of breakouts to constrain
the magnitude of the principal stress differences. This indicator takes a
boolean 'breakout_observed', which indicates if a breakout is observed or not. It also
takes a BreakoutParameters struct, and standard deviations for the proposal
distributions for the UCS, mud pressure, and rock friction, and mud temperature
in its constructor.

The BreakoutParameters struct includes state variables that holds specific
scalar values of the UCS, mud pressure, and rock friction coefficient. These
values are perturbed using normal distributions with the specified proposal
distributions. The state is either accepted or rejected as part of the MCMC
sampling.
"""
mutable struct Breakout <: StressIndicator
    breakout_parameters::BreakoutParameters
    std_UCS_proposal
    std_mud_pressure_proposal
    std_mud_temperature_proposal
    std_rock_friction_proposal
end

"""
    PerturbBreakoutState(bo_state::BreakoutState,
                         std_UCS,
                         std_mud_pressure,
                         std_rock_friction,
                         std_mud_temperature)


This function perturbs the breakout state using a normal proposal distribution
for the UCS, mud pressure, mud temperature, and rock friction. It returns a new BreakoutState
struct that holds the perturbed state.
"""
function PerturbIndicatorState(bo_ind::Breakout)

    dUCS = Distributions.rand(
           Distributions.Normal(0.0,bo_ind.std_UCS_proposal),1)[1]
    d_mudp = Distributions.rand(
           Distributions.Normal(0.0,bo_ind.std_mud_pressure_proposal),1)[1]
    d_mud_temperature = Distributions.rand(
          Distributions.Normal(0.0,bo_ind.std_mud_temperature_proposal),1)[1]
    d_rock_friction = Distributions.rand(
           Distributions.Normal(0.0,bo_ind.std_rock_friction_proposal),1)[1]
    
    bo_ind_new = deepcopy(bo_ind)

    bo_ind_new.breakout_parameters.state.UCS += dUCS
    bo_ind_new.breakout_parameters.state.mud_pressure += d_mudp
    bo_ind_new.breakout_parameters.state.mud_temperature += d_mud_temperature
    bo_ind_new.breakout_parameters.state.rock_friction += d_rock_friction

    return bo_ind_new
end

"""
     BreakoutParameterLogLikelihood(bo::BreakoutParameters)

This function computes the log likelihood of the breakout state parameters (UCS
mud pressure, and rock friction) using the probability distributions for these
parameters contained in the BreakoutParameters struct. This is used in the
inference since combinations of stress and breakout parameters that have
low likelihood will be more likely to be rejected.
"""

function BreakoutParameterLogLikelihood(bo::BreakoutParameters)
    out = 0.0

    UCS_loglikelihood = Distributions.logpdf(bo.UCS_dist,bo.state.UCS)
    MP_loglikelihood = Distributions.logpdf(bo.mud_pressure_dist,bo.state.mud_pressure)
    MT_loglikelihood = Distributions.logpdf(bo.mud_temperature_dist,bo.state.mud_temperature)
    RF_loglikelihood = Distributions.logpdf(bo.rock_friction_dist,bo.state.rock_friction)


    out += UCS_loglikelihood
    out += MP_loglikelihood
    out += MT_loglikelihood
    out += RF_loglikelihood
    
    # debug
    return (out, UCS_loglikelihood, MP_loglikelihood, MT_loglikelihood, RF_loglikelihood)
    # return out
end

function BreakoutParameterLikelihood(bo::BreakoutParameters)
    return exp(BreakoutParameterLogLikelihood(bo))
end

"""
     BreakoutProbability(s::StressState,p::BreakoutParameters)

This function computes the probability that a breakout will occur given the
stress state and BreakoutParameters. This ends up being equal to the likelihood
of the state parameters if the Mohr-Coulomb criteria is exceeded, and zero
otherwise. This function uses the well inclination and azimuth together with
the stress directions and magnitudes to compute the three principal stresses
at the borehole wall. It then uses a Mohr-Coulomb criteria to determine if a breakout
will occur at the specified breakout azimuth.
"""
function BreakoutAtAzimuthProbability(s::StressState,p::BreakoutParameters)

    return exp(BreakoutAtAzimuthLogProbability(s, p))

end

"""
        BreakoutLogProbability(s::StressState,p::BreakoutParameters)

This function computes the log probability that a breakout will occur at a specified
azimuthal location on the borehole wall given the stress state and BreakoutParameters.
This ends up being equal to the log likelihood of the state parameters if the Mohr-Coulomb
criteria is exceeded, and -Inf otherwise. 
"""

function BreakoutAtAzimuthLogProbability(s::StressState,p::BreakoutParameters)
        # The Distributions.VonMises distribution is defined on [-pi,pi]
    # but azimuths are defined on [0,2pi], so we need to compute
    # the alternative angle that is equivalent in this space
    # Map angle to [-π, π] and choose the equivalent angle with higher probability
    function choose_best_angle(angle, distribution)
        angle1 = mod(angle + π, 2π) - π  # Map to [-π, π]
        angle2 = mod(angle, 2π) - π      # Map equivalent angle to [-π, π]
        
        return Distributions.pdf(distribution, angle1) >= Distributions.pdf(distribution, angle2) ? angle1 : angle2
    end
    theta_distribution = Distributions.VonMises(p.breakout_azimuth,1.0/(p.breakout_azimuth_err^2))

    # get the state of the stochastic variables
    UCS = p.state.UCS
    mud_pressure = p.state.mud_pressure
    mud_temperature = p.state.mud_temperature
    mu = p.state.rock_friction
    
    (state_loglikelihood, ucs_debug, mp_debug, mt_debug, rf_debug) = BreakoutParameterLogLikelihood(p)

    b = BS.WellState(mud_pressure,p.well_inclination,p.well_azimuth)

    sigtmax_norm, theta_max = BS.MaximumPrincipalStress(s, b, PR=p.PR)

    best_tmin = choose_best_angle(theta_max, theta_distribution)
    azimuth_loglikelihood = Distributions.logpdf(theta_distribution, best_tmin)

    deltaT = p.formation_temperature - mud_temperature

    # if the mud is much cooler than the rock then the thermal effect will
    # reduce the likelihood of breakouts. However, the thermal stress isn't instantaneous
    # so a breakout could form before the thermal stress has time to develop. To account for this we set the thermal effect to zero if the temperature difference is negative, which allows breakouts to form even if the mud is cooler than the rock.
    if deltaT > 0
        deltaT = 0.0
    end
    
    TEF = p.CTE*p.YM/(1.0-p.PR)
    
    sigma_rr = mud_pressure - s.Pp
    smax = sigtmax_norm  - TEF*deltaT

    tau = (smax-sigma_rr)/2.0
    sig_m = (smax+sigma_rr)/2.0

    phi = atan.(mu)

    So = 0.5 * UCS / (sqrt(1.0 + mu^2) + mu)

    f = tau - sig_m*sin.(phi) - So*cos.(phi)
    #=
    println("in BreakoutAtAzimuthLogProbability")
    println("smax = ", smax)
    println("f = ", f)
    println("state_loglikelihood = ", state_loglikelihood)
    println("azimuth_loglikelihood = ", azimuth_loglikelihood)
    =#
    if (f >= 0) && (azimuth_loglikelihood > -Inf)
        # A breakout is predicted to occur at the specified azimuth and a breakout is observed so return the likelihood of the state and azimuth parameters
      return state_loglikelihood + azimuth_loglikelihood
    else
        #println("Breakout observed but not predicted")
        #println("relevant parameters:")
        #println("sigtmax_norm = ", sigtmax_norm)
        #println("theta_max = ", theta_max)
        #println("best_tmin = ", best_tmin)
        #println("azimuth_loglikelihood = ", azimuth_loglikelihood)
        #println("theta_distribution = ", theta_distribution)
        #println("deltaT = ", deltaT)
        #println("smax = ", smax)
        #println("UCS = ", UCS)
        #println("f=", f)
      return -Inf
    end

end

"""
    NoBreakoutAnywhereLogProbability(s::StressState,p::BreakoutParameters)

This function computes the log likelihood of the stress state and breakout parameters
    given the observation that no breakout is observed anywhere on the borehole wall.
    If a breakout is predicted to occur anywhere around the borehole wall then the probability is zero (log likelhood -Inf).
    If no breakout is predicted anywhere then the log likelihood is simply the log likelihood of the breakout parameters.
"""
function NoBreakoutAnywhereLogProbability(s::StressState,p::BreakoutParameters)

    # get the state of the stochastic variables
    UCS = p.state.UCS
    mud_pressure = p.state.mud_pressure
    mud_temperature = p.state.mud_temperature
    mu = p.state.rock_friction

    (state_loglikelihood, ucs_debug, mp_debug, mt_debug, rf_debug) = BreakoutParameterLogLikelihood(p)

    b = BS.WellState(mud_pressure,p.well_inclination,p.well_azimuth)

    sigtmax_norm, theta_max = BS.MaximumPrincipalStress(s, b, PR=p.PR)

    sigma_rr = mud_pressure - s.Pp
    deltaT = p.formation_temperature - mud_temperature
    # if the mud is much cooler than the rock then the thermal effect will
    # reduce the likelihood of breakouts. However, the thermal stress isn't instantaneous
    # so a breakout could form before the thermal stress has time to develop. To account for this we set the thermal effect to zero if the temperature difference is negative, which allows breakouts to form even if the mud is cooler than the rock.
    if deltaT > 0
        deltaT = 0.0
    end
    
    TEF = p.CTE*p.YM/(1.0-p.PR)

    smax = sigtmax_norm - TEF*deltaT

    tau = (smax-sigma_rr)/2.0
    sig_m = (smax+sigma_rr)/2.0

    phi = atan.(mu)

    So = 0.5 * UCS / (sqrt(1.0 + mu^2) + mu)

    f = tau - sig_m*sin.(phi) - So*cos.(phi)
    #=
    println("in NoBreakoutAnywhereLogProbability")
    println("smax = ", smax)
    println("f= ", f)
    =#
    if (f >= 0)
        # A breakout is predicted to occur somewhere but is not observed so the probability is zero
        
        #println("Breakout predicted but not observed")
        #println("Here are the details of the predicted breakout that is inconsistent with the observation:")
        #println("tau = ", tau)
        #println("sig_m = ", sig_m)
        #println("phi = ", phi)
        #println("UCS = ", UCS)
        #println("mud pressure = ", mud_pressure)
        #println("mud temperature = ", mud_temperature)
        #println("rock friction = ", mu)
        #println("smax = ", smax)
        #println("sigma_rr = ", sigma_rr)
        #println("TEF*deltaT = ", TEF*deltaT)
        #println("f = ", f)
        return -Inf
    else
        # a breakout is not predicted to occur anywhere so, so return the state likelihood
      return state_loglikelihood
    end

end
"""
StressStateLikelihood(s::StressState,i::BreakoutPresent)

This function computes the likelihood of the provided stress state and
breakout state parameters given the observation that a breakout is observed
or not.
"""
function StressStateLikelihood(s::StressState,i::Breakout)

    return exp(StressStateLogLikelihood(s, i))
end

"""
StressStateLikelihood(s::StressState,i::Breakout)

This function computes the likelihood of the provided stress state and
breakout state parameters given the observation that a breakout is observed
or not.
"""
function StressStateLogLikelihood(s::StressState,i::Breakout)
    if (i.breakout_parameters.breakout_observed)
        return BreakoutAtAzimuthLogProbability(s, i.breakout_parameters)
    else
        return NoBreakoutAnywhereLogProbability(s, i.breakout_parameters)
    end
end
