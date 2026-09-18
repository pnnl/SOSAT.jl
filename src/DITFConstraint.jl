
"""
DITFState(tensile_strength,mud_pressure,mud_temperature)

The DITFState struct holds the (scalar) values for the uncertain parameters that control
    the DITF analysis (beyond the stress state). Since these parameters
    are uncertain they are included in the inference, and so their state needs
    to be perturbed and accepted or rejected with the MCMC algorithm along
    with the stress state.
"""
mutable struct DITFState
    tensile_strength
    mud_pressure
    mud_temperature
end

"""
DITFParameters(DITF_observed,
               DITF_azimuth,
               DITF_azimuth_err,
               DITF_omega,
               DITF_omega_err,
               tensile_strength_dist,
               mud_pressure_dist,
               mud_temperature_dist,
               well_azimuth,
               well_inclination,
               formation_temperature,
               YM,
               PR,
               CTE,
               DITF_state)

The DITFParameters struct holds all of the parameters (deterministic and
    stochastic) that are needed to constrain the stress state using the
    presence or absence of DITF. The struct holds these parameters as
    well as keeping a DITFState struct that holds the current value of the
    state variables. These are perturbed and then accepted or rejected as
    part of the MCMC sampling.


"""
mutable struct DITFParameters
    DITF_observed
    DITF_azimuth
    DITF_azimuth_err
    DITF_omega
    DITF_omega_err
    tensile_strength_dist
    mud_pressure_dist
    mud_temperature_dist
    well_azimuth
    well_inclination
    formation_temperature
    YM
    PR
    CTE
    state::DITFState
    """

    function DITFParameters(DITF_observed,
                            DITF_azimuth,
                            DITF_azimuth_err,
                            DITF_omega,
                            DITF_omega_err,
                            tensile_strength_dist,
                            mud_pressure_dist,
                            mud_temperature_dist,
                            well_azimuth,
                            well_inclination,
                            formation_temperature,
                            YM,
                            PR,
                            CTE)

    Creates a new DITFParameter struct and initializes the DITFState
        variable using a random realization of the relevant distributions
    """
    function DITFParameters(DITF_observed;
                            DITF_azimuth,
                            DITF_azimuth_err,
                            DITF_omega,
                            DITF_omega_err,
                            tensile_strength_dist,
                            mud_pressure_dist,
                            mud_temperature_dist,
                            well_azimuth,
                            well_inclination,
                            formation_temperature,
                            YM,
                            PR,
                            CTE)

        ditf_state = DITFState(Distributions.rand(tensile_strength_dist,1)[1],
                               Distributions.rand(mud_pressure_dist,1)[1],
                               Distributions.rand(mud_temperature_dist,1)[1])

        new(DITF_observed,
            DITF_azimuth,
            DITF_azimuth_err,
            DITF_omega,
            DITF_omega_err,        
            tensile_strength_dist,
            mud_pressure_dist,
            mud_temperature_dist,
            well_azimuth,
            well_inclination,
            formation_temperature,
            YM,
            PR,
            CTE,
            ditf_state)
    end
end


"""
    DITF(ditf_parameters,
         std_tensile_strength_proposal,
         std_mud_pressure_proposal,
         std_mud_temperature_proposal)

A stress indicator using the presence  or absence of DITF to constrain
the magnitude of the principal stress differences. This indicator takes a
boolean 'DITF_observed', which indicates if a DITF is observed or not. It also
takes a DITFParameter struct, and standard deviations for the proposal
distributions for the tensile_strength, mud pressure, and mud temperature in its constructor.
The DITFParameter struct includes state variables that holds specific
scalar values of the tensile strength, mud pressure, and mud temperature. These
values are perturbed using normal distributions with the specified proposal
distributions. The state is either accepted or rejected as part of the MCMC
sampling.
"""
mutable struct DITF <: StressIndicator
    ditf_parameters::DITFParameters
    std_tensile_strength_proposal
    std_mud_pressure_proposal
    std_mud_temperature_proposal
end

"""
    PerturbDITFState(bo_state::DITFState,
                     std_tensile_strength,
                     std_mud_pressure,
                     std_mud_temperature)

This function perturbs the DITF state using a normal proposal distribution
for the tensile strength, mud pressure, and mud temperature. It returns a new DITFState
struct that holds the perturbed state.
"""
function PerturbIndicatorState(ditf_ind::DITF)
    dTS = Distributions.rand(
           Distributions.Normal(0.0,ditf_ind.std_tensile_strength_proposal),1)[1]
    d_mudp = Distributions.rand(
           Distributions.Normal(0.0,ditf_ind.std_mud_pressure_proposal),1)[1]
    d_mudt = Distributions.rand(
           Distributions.Normal(0.0,ditf_ind.std_mud_temperature_proposal),1)[1]

    ditf_ind_new = deepcopy(ditf_ind)

    ditf_ind_new.ditf_parameters.state.tensile_strength += dTS
    ditf_ind_new.ditf_parameters.state.mud_pressure += d_mudp
    ditf_ind_new.ditf_parameters.state.mud_temperature += d_mudt
    return ditf_ind_new
end

"""
     DITFParameterLogLikelihood(bo::DITFParameters)

This function computes the log likelihood of the DITF state parameters (tensile strength,
mud pressure, and mud temperature) using the probability distributions for these
parameters contained in the DITFParameters struct. This is used in the
inference since combinations of stress and DITF parameters that have
low likelihood will be more likely to be rejected.
"""

function DITFParameterLogLikelihood(ditf::DITFParameters)
    out = 0.0

    tensile_strength_loglikelihood = Distributions.logpdf(ditf.tensile_strength_dist,ditf.state.tensile_strength)
    MP_loglikelihood = Distributions.logpdf(ditf.mud_pressure_dist,ditf.state.mud_pressure)
    MT_loglikelihood = Distributions.logpdf(ditf.mud_temperature_dist,ditf.state.mud_temperature)
    
    out += tensile_strength_loglikelihood
    out += MP_loglikelihood
    out += MT_loglikelihood
    if out == -Inf

        #println("Invalid DITF parameters. Here are the relevant values for this calculation:")
        #println("tensile_strength = ", ditf.state.tensile_strength)
        #println("mud_pressure = ", ditf.state.mud_pressure)
        #println("mud_temperature = ", ditf.state.mud_temperature)
        #println("tensile_strength_loglikelihood = ", tensile_strength_loglikelihood)
        #println("MP_loglikelihood = ", MP_loglikelihood)
        #println("MT_loglikelihood = ", MT_loglikelihood)
    end
    return out
end

function DITFParameterLikelihood(ditf::DITFParameters)
    return exp(DITFParameterLogLikelihood(ditf))
end

"""
     DITFProbability(s::StressState,p::DITFParameters)

This function computes the probability that a DITF will occur given the
stress state and DITFParameters. This ends up being equal to the likelihood
of the state parameters if the tensile strength is exceeded, and zero
otherwise. This function uses the well inclination and azimuth together with
the stress directions and magnitudes to compute the maximum principal stress
at the borehole wall. It then compares the least compressive in-plane principal
stress to the tensile strength. If the tensile stress is greater than the
tensile strength, the likelihood of the stress state is returned, otherwise
a zero probablity is returned.
"""
function DITFAtAzimuthProbability(s::StressState,p::DITFParameters)

    return exp(DITFAtAzimuthLogProbability(s, p))

end


function NoDITFAtAzimuthProbability(s::StressState,p::DITFParameters)

    return exp(NoDITFAtAzimuthLogProbability(s, p))

end

"""
        DITFAtAzimuthLogProbability(s::StressState,p::DITFParameters)
This function computes the log likelihood of the stress state and DITF parameters
given the observation that a DITF is observed at a specific azimuth and with a 
specific angle omega between the plane of the DITF and the plane perpendicular
to the borehole axis. If a DITF is predicted to occur, then we compare the azimuth
and omega of the minimum principal stress to the observed DITF azimuth and omega,
and compute the log likelihood of these observations given the uncertainty in
the DITF azimuth and omega. If no DITF is predicted to occur, then the probability
is zero (log likelihood -Inf).
"""

function DITFAtAzimuthLogProbability(s::StressState,p::DITFParameters)

    fixed_direction = false
    returned_likelihood = -Inf
    state_loglikelihood = DITFParameterLogLikelihood(p)
    mud_pressure = p.state.mud_pressure
    b = BS.WellState(mud_pressure,p.well_inclination,p.well_azimuth)
    
    if fixed_direction
        # This method checks to see if the tensile stress on the observed DITF plane exceeds the tensile strength. It doesn't
        # check other directions.
            sig_norm, sig_shear = BS.NormalAndShear(s, b, p.DITF_azimuth, p.DITF_omega, PR=p.PR)
        deltaT = p.formation_temperature - p.state.mud_temperature
        TEF = p.CTE*p.YM/(1.0-p.PR)
        smin = sig_norm - TEF*deltaT

        # setup so that if the tensile strength (negative is tensile)
        # exceeds the tensile strength, f will be positive
        f = - (smin + p.state.tensile_strength)
        if (f >= 0)
           returned_likelihood = state_loglikelihood
        end
    else
        # This method will find the location and direction of the minimum stress before accounting for thermal stresses
        # It will then compare the azimuth and angle of this minimum stress to the observed DITF azimuth and angle, and
        # compute a likelihood based on the uncertainty in these observations. Finally, it will check if the minimum stress
        # is tensile enough to cause a DITF given the tensile strength and thermal stresses. If not, then the likelihood is
        # set to zero (log likelihood -Inf).

        # The Distributions.VonMises distribution is defined on [-pi,pi]
        # but azimuths are defined on [0,2pi], so we need to compute
        # the alternative angle that is equivalent in this space
        # Map angle to [-π, π] and choose the equivalent angle with higher probability
        function choose_best_angle(angle, distribution)
            angle1 = mod(angle + π, 2π) - π  # Map to [-π, π]
            angle2 = mod(angle, 2π) - π      # Map equivalent angle to [-π, π]
            
            return Distributions.pdf(distribution, angle1) >= Distributions.pdf(distribution, angle2) ? angle1 : angle2
        end
        theta_distribution = Distributions.VonMises(p.DITF_azimuth,1.0/(p.DITF_azimuth_err^2))
        omega_distribution = Distributions.VonMises(p.DITF_omega,1.0/(p.DITF_omega_err^2))

        sigt_min, tmin, omega = BS.MinimumPrincipalStress(s, b, PR=p.PR)
        best_tmin = choose_best_angle(tmin, theta_distribution)
        best_omega = choose_best_angle(omega, omega_distribution)

        
        # compute likelihood of DITF azimuth observation
        azimuth_loglikelihood = Distributions.logpdf(theta_distribution, best_tmin)
        omega_loglikelihood = Distributions.logpdf(omega_distribution, best_omega)
        
        deltaT = p.formation_temperature - p.state.mud_temperature
        TEF = p.CTE*p.YM/(1.0-p.PR)

        smin = sigt_min - TEF*deltaT

        # setup so that if the tensile strength (negative is tensile)
        # exceeds the tensile strength, f will be positive
        f = - (smin + p.state.tensile_strength)
        #=
        println("in DITFAtAzimuthLogProbability")
        println("s= ",s)
        println("s.gamma (deg)= ", rad2deg(s.gamma))
        println("90 - s.gamma (deg)= ", 90.0 - rad2deg(s.gamma))
        println("b= ",b)
        println("tmin (deg)= ", rad2deg(tmin))
        println("best_tmin (deg)= ", rad2deg(best_tmin))
        println("omega (deg)= ", rad2deg(omega))
        println("best_omega (deg)= ", rad2deg(best_omega))
        println("theta_distribution parameters: ", p.DITF_azimuth," ", 1.0/(p.DITF_azimuth_err^2))
        println("omega_distribution parameters: ", p.DITF_omega, " ", 1.0/(p.DITF_omega_err^2))
        println("state_loglikelihood = ", state_loglikelihood)
        println("azimuth_loglikelihood = ", azimuth_loglikelihood)
        println("omega_loglikelihood = ", omega_loglikelihood)
        println("smin= ", smin)
        println("tensile_strength = ", p.state.tensile_strength)
        println("f = ", f)
        =#
        if (f >= 0)
           returned_likelihood =  state_loglikelihood + azimuth_loglikelihood + omega_loglikelihood
        end
        if (returned_likelihood == -Inf)

            #println("s= ",s)
            #println("state_loglikelihood = ", state_loglikelihood)
            #println("DITF observed but not predicted. Here are the relevant values for this calculation:")
            #println("azimuth= ", p.DITF_azimuth)
            #println("omega= ", p.DITF_omega)
            #println("tmin= ", tmin)
            #println("omega= ", omega)
            #println("best_tmin= ", best_tmin)
            #println("best_omega= ", best_omega)
            #println("azimuth_loglikelihood = ", azimuth_loglikelihood)
            #println("omega_loglikelihood = ", omega_loglikelihood)
            #println("deltaT = ", deltaT)
            #println("TEF*deltaT = ", TEF*deltaT)
            #println("smin = ", smin)
            #println("tensile strength = ", p.state.tensile_strength)
            #println("mud_pressure = ", mud_pressure)
            #println("f = ", f)
        end
    end


    return returned_likelihood
    
end

"""
        NoDITFAtAzimuthLogProbability(s::StressState,p::DITFParameters)
This function computes the log likelihood of the stress state and DITF parameters
given the observation that no DITF forms at a specific azimuth and with a 
specific angle omega between the plane of the DITF and the plane perpendicular
to the borehole axis. If a DITF is predicted to occur, then we compare the azimuth
and omega of the minimum principal stress to the observed DITF azimuth and omega,
and compute the log likelihood of these observations given the uncertainty in
the DITF azimuth and omega. If no DITF is predicted to occur, then the probability
is zero (log likelihood -Inf).
"""

function NoDITFAtAzimuthLogProbability(s::StressState,p::DITFParameters)


    # get the state of the stochastic variables
    mud_pressure = p.state.mud_pressure

    state_loglikelihood = DITFParameterLogLikelihood(p)

    b = BS.WellState(mud_pressure,p.well_inclination,p.well_azimuth)
    #sigt_min, tmin, omega = BS.MinimumPrincipalStress(s, b, PR=p.PR)
    sig_norm, sig_shear = BS.NormalAndShear(s, b, p.DITF_azimuth, p.DITF_omega, PR=p.PR)

    
    deltaT = p.formation_temperature - p.state.mud_temperature
    TEF = p.CTE*p.YM/(1.0-p.PR)

    smin = sig_norm - TEF*deltaT

    # setup so that if the tensile strength (negative is tensile)
    # exceeds the tensile strength, f will be positive
    f = - (smin + p.state.tensile_strength)

    if (f <= 0)
        return state_loglikelihood
    else
        #=
        println("s= ",s)
        println("b= ",b)
        println("p.DITF_azimuth= ",p.DITF_azimuth)
        println("DITF predicted but not observed. Here are the relevant values for this calculation:")
        println("smin = ", smin)
        println("sig_norm = ", sig_norm)
        println("TEF*deltaT = ", TEF*deltaT)
        println("tensile strength = ", p.state.tensile_strength)
        println("mud_pressure = ", mud_pressure)
        println("f = ", f)
        =#
        return -Inf
        
    end

end

function NoDITFAnywhereProbability(s::StressState,p::DITFParameters)

    return exp(NoDITFAnywhereLogProbability(s, p))

end

"""
     NoDITFAnywhereLogProbability(s::StressState,p::DITFParameters)
This function computes the log likelihood of the stress state and DITF parameters
given the observation that no DITF is observed. If a DITF is predicted to occur
anywhere around the borehole, then the probability is zero (log likelihood -Inf). If no DITF is predicted to occur, then the log likelihood of the state parameters is returned.
"""

function NoDITFAnywhereLogProbability(s::StressState,p::DITFParameters)
    # get the state of the stochastic variables
    mud_pressure = p.state.mud_pressure

    state_loglikelihood = DITFParameterLogLikelihood(p)

    b = BS.WellState(mud_pressure,p.well_inclination,p.well_azimuth)
    sigt_min, tmin, omega = BS.MinimumPrincipalStress(s, b, PR=p.PR)
    
    deltaT = p.formation_temperature - p.state.mud_temperature
    TEF = p.CTE*p.YM/(1.0-p.PR)

    smin = sigt_min - TEF*deltaT

    # setup so that if the tensile strength (negative is tensile)
    # exceeds the tensile strength, f will be positive
    f = - (smin + p.state.tensile_strength)

    if (f >= 0)
        #=
        println("DITF predicted but not observed")
        println("smin = ", smin)
        println("tensile strength = ", p.state.tensile_strength)
        println("sigt_min = ", sigt_min)
        println("mud_pressure = ", mud_pressure)
        println("f = ", f)
        println("tmin= ",tmin)
        println("omega= ", omega)
        println("YM = ", p.YM)
        println("deltaT = ", deltaT)
        println("TEF*deltaT = ", TEF*deltaT)
        # a DITF is predicted but not observed, so the probability is zero
        # i.e. log likelihood is -Inf
        =#
        return -Inf
    else
        #=
        println("DITF not predicted and not observed")
        println("s= ",s)
        println("smin = ", smin)
        println("tensile strength = ", p.state.tensile_strength)
        println("sigt_min = ", sigt_min)
        println("mud_pressure = ", mud_pressure)
        println("f = ", f)
        println("tmin= ",tmin)
        println("omega= ", omega)
        println("YM = ", p.YM)
        println("deltaT = ", deltaT)
        println("TEF*deltaT = ", TEF*deltaT)
        # a DITF is predicted to not occur, so return the state likelihood
        =#
        return state_loglikelihood
    end
end


"""
StressStateLikelihood(s::StressState,i::DITF)

This function computes the likelihood of the provided stress state and
breakout state parameters given the observation that a DITF is observed
or not.
"""
function StressStateLikelihood(s::StressState,i::DITF)

    return exp(StressStateLogLikelihood(s, i))
end

function StressStateLogLikelihood(s::StressState,i::DITF)

    if i.ditf_parameters.DITF_observed
        # return the log probability of DITF at azimuth
        # and orientation
        return DITFAtAzimuthLogProbability(s,i.ditf_parameters)
    else
        # no DITF observed, so return the log probability
        # that no DITF occurs anywhere
        #return NoDITFAtAzimuthLogProbability(s,i.ditf_parameters)
        return NoDITFAnywhereLogProbability(s,i.ditf_parameters)
        
    end

end
