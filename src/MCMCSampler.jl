using HDF5
using Rotations
using PythonCall

mutable struct MCMCSampler
    stress_state::StressState
    indicators::Vector{StressIndicator}
    stress_std::Vector{Float64}
    angle_std::Vector{Float64}
    function MCMCSampler(s_o::StressState,
                         indicators::Vector{StressIndicator},
                         stress_std::Vector{Float64},
                         angle_std::Vector{Float64})
        # the stress_std should have a proposal standard deviation for
        # each of the three principal stresses and the pore pressure
        # so it should have exaxtly four elements
        if length(stress_std) != 4
            error("The stress proposal standard deviation vector should have exactly four elements")
        end 
        if length(angle_std) != 3
            error("The Euler angle proposal standard deviation vector should have exactly three elements")
        end
        new(s_o, indicators, stress_std, angle_std)
    end
end

function EvaluateInitialLikelihood(sampler::MCMCSampler;verbose=false)
    L1 = StressTotalLogLikelihood(sampler.stress_state,
                                  sampler.indicators,
                                  verbose=verbose)
    return exp(L1)
end

function EvaluteInitialLogLikelihood(sampler::MCMCSampler;verbose=false)
    L1 = StressTotalLogLikelihood(sampler.stress_state,
                                  sampler.indicators,
                                  verbose=verbose)
    return L1
end

function vec_from_stress_state(ss::StressState)
    return [ss.S1, ss.S2, ss.S3, ss.Pp, ss.alpha, ss.beta, ss.gamma]
end

function update_stress_state_with_vec(stress_vec, ss::StressState)
    ss.S1 = stress_vec[1]
    ss.S2 = stress_vec[2]
    ss.S3 = stress_vec[3]
    ss.Pp = stress_vec[4]
    ss.alpha = stress_vec[5]
    ss.beta = stress_vec[6]
    ss.gamma = stress_vec[7]
end

"""
    FindLikelyInitialState(sampler::MCMCSampler; num_samples::Int = Int(1e4))

return: StressState, boolean indicating if a valid stress state was found
"""
function FindLikelyInitialState(sampler::MCMCSampler; num_samples::Int = Int(1e6), verbose::Bool = false)

    L_init = SOSAT.EvaluteInitialLogLikelihood(sampler, verbose=verbose)
    println("L_init= ",L_init)
    println("initial stress state: ", sampler.stress_state)
    function remove_constraint(indicator)
        if indicator isa Breakout
            return true
        elseif indicator isa DITF
            return true
        elseif indicator isa AphiConstraint
            return true
        else
            return false
        end
    end
    # loop through the indicators and remove any breakout or DITF indicators
    reduced_sampler = deepcopy(sampler)
    #filter!(x -> !(x isa Breakout || x isa DITF), reduced_sampler.indicators)
    reduced_indicators = [x for x in reduced_sampler.indicators if !remove_constraint(x)]
    reduced_sampler.indicators = reduced_indicators
    #println("initial indicators= ",sampler.indicators)
    #println("reduced_indicators= ",reduced_sampler.indicators)
    L_init_reduced = SOSAT.EvaluteInitialLogLikelihood(reduced_sampler;verbose=verbose)
    if verbose
        println("Number of indicators reduced from ",length(sampler.indicators),
                " to ",length(reduced_sampler.indicators))
        println("Initial log-likelihood reduced from ",L_init,
                " to ",L_init_reduced,
                " by removing breakout and DITF indicators")
        println("starting sampling with reduced sampler")
    end

    reduced_samples, aceptance_ratio = Sample(reduced_sampler, num_samples, Nburn=1,Nthin=1,verbose=verbose)
    println("Finished sampling with reduced sampler. Acceptance ratio = ", aceptance_ratio)
    highest_likelihood = -Inf
    highest_index = -1
    for i in eachindex(reduced_samples)
        sample_likelihood = StressTotalLogLikelihood(reduced_samples[i], sampler.indicators,verbose=verbose)
        if sample_likelihood > highest_likelihood
            highest_likelihood = sample_likelihood
            highest_index = i
        end
    end

    if (highest_index == -1)
        return reduced_samples[1],false
    else
        best_state = reduced_samples[highest_index]
        return best_state,true
    end
end


function Sample(sampler::MCMCSampler,
                Nsample::Int;
                Nburn=1000,
                Nthin=10,
                FixedOrder=false,
                verbose::Bool = false)

    Naccept = 0
    Nreject = 0

    if FixedOrder==false
        stress_perturbation_func = PerturbStressState
    else
        stress_perturbation_func = PerturbStressStateFixedOrder
    end
    samples = Vector{StressState}(undef,Nsample)
    accept_dist = Distributions.Uniform(0.0,1.0)
    s1 = deepcopy(sampler.stress_state)
    L1 = StressTotalLogLikelihood(s1,sampler.indicators,verbose=verbose)
    if L1 == -Inf
        println("Initial stress state has zero likelihood - attemping to find valid initial stress state")
        new_state, found = SOSAT.FindLikelyInitialState(sampler, num_samples=10000, verbose=false)
        if found == false
            # try with more samples
            new_state, found = SOSAT.FindLikelyInitialState(sampler, num_samples=100000, verbose=false)
        end
        if found == false
            for indicator in sampler.indicators
                println("Indicator: ",indicator)
                L_ind = StressStateLogLikelihood(sampler.stress_state, indicator)
                println("  Log Likelihood = ",L_ind)
            end
            println("Could not find a valid initial stress state with non-zero likelihood")
            println("initial stress state: ", sampler.stress_state)
            error("Initial stress state has zero likelihood. Please choose a different initial stress state.")
            return
        else
            println("Found a more likely initial stress state from reduced sampling")
            println("Initial stress state: ", sampler.stress_state)
            sampler.stress_state = new_state
            println("New stress state: ", sampler.stress_state)
            initial_likelihood = SOSAT.EvaluateInitialLikelihood(sampler, verbose=true)
            println("New initial likelihood = ",initial_likelihood)
            L1 = initial_likelihood
        end
    end
    iter = 1
    verbose && println("L1= ",L1)

    
    while iter<(Nburn)
        # Block Gibbs sampling: alternate between updating stress state and indicator parameters
        if iter % 2 == 1
            # Update stress state only, keep indicators fixed
            strial = stress_perturbation_func(s1,
                                              sampler.stress_std,
                                              sampler.angle_std)
            itrial = sampler.indicators  # Keep indicators fixed
        else
            # Update indicators only, keep stress state fixed
            strial = deepcopy(s1)  # Keep stress state fixed
            itrial = PerturbIndicators(sampler.indicators)
        end
        
        Ltrial = StressTotalLogLikelihood(strial,itrial)

        a = exp(Ltrial - L1)
        u = Distributions.rand(accept_dist,1)

        if (a>u[1])
            s1 = deepcopy(strial)
            if ((s1.S1<s1.Pp) || (s1.S2<s1.Pp))
                print("s.S1= ",s1.S1," s.S2= ",s1.S2,"\n")
                println("a= ",a," u= ",u[1])
            end
            sampler.indicators = deepcopy(itrial)  # Update the sampler's indicators
            L1 = Ltrial
        end
        iter += 1
        
    end

    n = 1  # sample counter
    m = 1  # thinning counter
    iter = 1
    nchar_bar = 50

    verbose && print("|")
    for i in range(1,stop=50)
      verbose && print("-")
    end
    verbose && print("| \n")
    flush(stdout)
    nsample_per_char = convert(typeof(nchar_bar),round(Nsample/nchar_bar))
    bar_counter = 0

    verbose && print("|")
    flush(stdout)
    while n<(Nsample+1)
        # Block Gibbs sampling: alternate between updating stress state and indicator parameters
        if iter % 2 == 1
            # Update stress state only, keep indicators fixed
            strial = PerturbStressState(s1,sampler.stress_std,sampler.angle_std)
            itrial = sampler.indicators  # Keep indicators fixed
        else
            # Update indicators only, keep stress state fixed
            strial = deepcopy(s1)  # Keep stress state fixed
            itrial = PerturbIndicators(sampler.indicators)
        end
        
        Ltrial = StressTotalLogLikelihood(strial,itrial, verbose=false)

        a = exp(Ltrial - L1)
        u = Distributions.rand(accept_dist,1)
        Lsamp = 0.0
        if (a>u[1])
            s1 = deepcopy(strial)
            if ((s1.S1<s1.Pp) || (s1.S2<s1.Pp))
                print("s.S1= ",s1.S1," s.S2= ",s1.S2,"\n")
                println("a= ",a," u= ",u[1])
            end
            sampler.indicators = deepcopy(itrial)  # Update the sampler's indicators
            # println("   sample accepted")
            L1 = Ltrial
            Naccept += 1
        else
            # println("   sample rejected")
            Nreject += 1
        end

        if (m==Nthin)
            # keep samples
            #println("recording sample ",n," with likelihood ",Lsamp)
            bar_counter += 1
            samples[n] = deepcopy(s1)
            if s1.S1 < s1.Pp || s1.S2 < s1.Pp || s1.S3 < s1.Pp
                println("Warning: Stress state has principal stress less than pore pressure")
                println("Log Likelihood = ",L1)
            end
            m = 1
            n+= 1
            if (bar_counter==nsample_per_char)
              verbose && print("-")
              flush(stdout)
                bar_counter = 0
            end
        else
            m += 1
        end
        iter += 1
    end
    verbose && print("| \n")
    Ntotal = Naccept + Nreject
    acceptance_ratio = Naccept / Ntotal
    verbose && println("Acceptance ratio =  ", acceptance_ratio)
    return samples, acceptance_ratio
end

