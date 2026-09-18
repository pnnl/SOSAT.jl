# StressUtils.jl


function DITF_SHmax(shmin, pp, deltaP, deltaT, YM, PR, CTE, tensile_strength)
    TEF = CTE * YM / (1.0-PR)
    return 3.0 * shmin .- 2.0 * pp .-
           deltaP .- TEF * deltaT .+ tensile_strength
end

function DITF_SHmin(shmax, pp, deltaP, deltaT, YM, PR, CTE, tensile_strength)
    TEF = CTE * YM / (1.0-PR)
    return (shmax .+ 2.0 * pp .+ deltaP .+ TEF * deltaT .- tensile_strength) / 3.0
end

function faultS3Limit(pp, mu, S1_total)
    return (S1_total - pp) / ((mu^2+1.0)^0.5 + mu)^2 + pp
end

function faultS1Limit(pp, mu, S3_total)
    return (S3_total - pp) * ((mu^2+1.0)^0.5 + mu)^2 + pp
end

function getDITFStressArrays(pp, Sv, mu, deltaP, deltaT,
                             YM, PR, CTE, tensile_strength, number_of_points)
    shmin_lower_limit = faultS3Limit(pp, mu, Sv)
    shmax_upper_limit = TF_shmax_limit(pp, Sv, mu)
    shmin_upper_limit = DITF_SHmin(shmax_upper_limit, pp, deltaP, deltaT,
                                     YM, PR, CTE, tensile_strength)
    shmin_array = LinRange(shmin_lower_limit, shmin_upper_limit, number_of_points)
    shmax_array = DITF_SHmax(shmin_array, pp, deltaP, deltaT,
                             YM, PR, CTE, tensile_strength)
    return shmin_array, shmax_array
end

function GetBreakoutStressArrays(pp, Sv, UCS, mu, deltaP, deltaT,
                                    YM, PR, CTE, number_of_points)
    shmin_lower_limit = faultS3Limit(pp, mu, Sv)
    shmin_upper_limit = TF_shmax_limit(pp, Sv, mu)
    shmin_array = LinRange(shmin_lower_limit, shmin_upper_limit, number_of_points)
    TEF = CTE * YM / (1.0-PR)
    shmax_array = (shmin_array .+ 2.0 * pp .+ deltaP  .+ TEF*deltaT .+ (sqrt(1+mu^2) + mu)^2 * deltaP .+ UCS ) / 3.0
    # trim any values where shmax is less than shmin
    valid_indices = findall(shmax_array .>= shmin_array)
    shmin_array = shmin_array[valid_indices]
    shmax_array = shmax_array[valid_indices]

    return shmin_array, shmax_array
end

function NF_shmin_limit(pp, Sv, mu)
    return (Sv .- pp) ./ ((mu^2 .+ 1.0)^0.5 .+ mu)^2 .+ pp
end

function SS_shmax_limit(shmin, pp, Sv, mu)
    return (shmin .- pp) .* ((mu^2 .+ 1.0)^0.5 .+ mu)^2 .+ pp
end

function TF_shmax_limit(pp, Sv, mu)
    return (Sv .- pp) * ((mu^2 .+ 1.0)^0.5 .+ mu)^2 .+ pp
end

function plot_stress_polygon(ax, sv, pp, mu; color="b")
    nf_shmin_limit_value = NF_shmin_limit(pp, sv, mu)
    nf_shmax_array = LinRange(nf_shmin_limit_value, sv, 100)
    ax.plot(nf_shmin_limit_value*ones(100), nf_shmax_array, color=color)

    ss_shmin_array = LinRange(nf_shmin_limit_value, sv, 100)
    ss_shmax_array = SS_shmax_limit(ss_shmin_array, pp, sv, mu)
    ax.plot(ss_shmin_array, ss_shmax_array, color=color)

    tf_shmax_value = TF_shmax_limit(pp, sv, mu)
    tf_shmin_array = LinRange(sv, tf_shmax_value, 100)
    ax.plot(tf_shmin_array, tf_shmax_value*ones(100), color=color)
end
    
function SminHist(stress_samples::Vector{StressState})

    smin_samples = GetSminSamples(stress_samples)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    jnk = ax.hist(smin_samples,density=true,color="k",)
    ax.set_xlabel("Minimum Principal Stress")
    ax.set_ylabel("Probability Density")
    return fig
end

function StressComponentHist(stress_samples::Vector{StressState}, stress_component::Symbol;
                             label=string(stress_component),
                             bins=50)
    comp_samples = _get_component_samples(stress_samples, stress_component)

    fig = plt.figure()
    if stress_component == :alpha || stress_component == :beta || stress_component == :gamma
        # convert to degrees
        comp_samples = rad2deg.(comp_samples)
    end
    ax = fig.add_subplot(111)
    jnk = ax.hist(comp_samples,density=true,color="k",bins=bins)
    ax.set_xlabel(label)
    ax.set_ylabel("Probability Density")
    return fig

end

# Schmidt projection function
# assume trend and plunge are in radians
function schmidt_projection(trend, plunge)
    r = sqrt(2) * sin((pi/2.0 - plunge)/2)
    x = r * sin(trend)
    y = r * cos(trend)
    return x, y
end

# Function to create stereonet grid
function add_stereonet_grid(ax)
    # Outer circle (primitive circle)
    patches = plt.matplotlib.patches
    circle = patches.Circle((0, 0), 1.0, fill=false, color="black", linewidth=2)
    ax.add_patch(circle)
    
    # Small circles (constant plunge lines)
    plunge_lines = [20, 30, 40, 50, 60, 70, 80]
    for plunge in plunge_lines
        r = sqrt(2) * sin((pi/2.0 - deg2rad(plunge))/2)
        circle = patches.Circle((0, 0), r, fill=false, color="gray", linewidth=0.5, linestyle="--")
        ax.add_patch(circle)
        # Label the plunge circles
        ax.text(r + 0.05, 0, "$(plunge)°", fontsize=8, ha="left", va="center")
    end

    # Great circles (constant trend lines - these are more complex)
    trends = deg2rad.([0, 90, 180, 270])  # Every 90 degrees
    for trend in trends
        # Create points along the great circle
        plunge_range = deg2rad(0):deg2rad(2):deg2rad(90)
        x_line = []
        y_line = []
        for p in plunge_range
            x, y = schmidt_projection(trend, p)
            push!(x_line, x)
            push!(y_line, y)
        end
        ax.plot(x_line, y_line, color="gray", linewidth=0.5)
        
        # Add azimuth labels at the edge
        x_label, y_label = schmidt_projection(trend, deg2rad(2))
        ax.text(x_label * 1.2, y_label * 1.2, "$(Int(rad2deg(trend)))°", 
                fontsize=10, ha="center", va="center", weight="bold")
    end

end

"""
    PrincipalStressStereonet(stress_samples::Vector{StressState}; use_mplstereonet=true)

Create a lower-hemisphere stereonet projection showing the density distribution
of principal stress orientations from MCMC samples.

"""
function PrincipalStressStereonet(stress_samples::Vector{StressState}; alpha=0.7)

    Sv, SHmax, SHmin, Sv_dir, SHmax_dir, SHmin_dir = GeographicallyOrderStresses(stress_samples)
    
    # Extract orientations for each principal stress
    Sv_azimuths = atan.(Sv_dir[:, 2], Sv_dir[:, 1])

    # z is positive downward so a positive number is a downward plunge, as it should be
    Sv_plunges = asin.(Sv_dir[:, 3])

    SHmax_azimuths = atan.(SHmax_dir[:, 2], SHmax_dir[:, 1])
    SHmax_plunges = asin.(SHmax_dir[:, 3])
    
    SHmin_azimuths = atan.(SHmin_dir[:, 2], SHmin_dir[:, 1])
    SHmin_plunges = asin.(SHmin_dir[:, 3])
    
    # if the plunge is negative, adjust azimuth by 180 degrees and make plunge positive
    for i in eachindex(Sv_plunges)
        if Sv_plunges[i] < 0.0
            Sv_plunges[i] = -Sv_plunges[i]
            Sv_azimuths[i] = mod(Sv_azimuths[i] + pi, 2pi)
        end
        if SHmax_plunges[i] < 0.0
            SHmax_plunges[i] = -SHmax_plunges[i]
            SHmax_azimuths[i] = mod(SHmax_azimuths[i] + pi, 2pi)
        end
        if SHmin_plunges[i] < 0.0
            SHmin_plunges[i] = -SHmin_plunges[i]
            SHmin_azimuths[i] = mod(SHmin_azimuths[i] + pi, 2pi)
        end
    end

    titles = ["Sv", "SHmax (Maximum Horizontal Stress)", "SHmin (Minimum Horizontal Stress)"]
    data_sets = [(Sv_azimuths, Sv_plunges), (SHmax_azimuths, SHmax_plunges), (SHmin_azimuths, SHmin_plunges)]
    colors = ["red", "green", "blue"]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, aspect="equal")
    for i in 1:3
        azimuths, plunges = data_sets[i]
        # Project to stereonet coordinates - fix the broadcasting
        projections = schmidt_projection.(azimuths, plunges)
        x = [p[1] for p in projections]
        y = [p[2] for p in projections]
        # Scatter plot
        sc = ax.scatter(x, y, c=colors[i], s=20, alpha=alpha, label=titles[i])
    end

    ax.legend(loc="upper right")
    # Set limits with some padding
    margin = sqrt(2) + 0.3
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)

    ax.set_xticks([])
    ax.set_yticks([])
    add_stereonet_grid(ax)    
    plt.tight_layout()
    return fig
end


function SintHist(stress_samples::Vector{StressState})

    sint_samples = GetSintSamples(stress_samples)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    jnk = ax.hist(sint_samples,density=true,color="k",)
    ax.set_xlabel("Intermediate Principal Stress")
    ax.set_ylabel("Probability Density")
    return fig
end

function SmaxHist(stress_samples::Vector{StressState})

    smax_samples = GetSmaxSamples(stress_samples)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    jnk = ax.hist(smax_samples,density=true,color="k",)
    ax.set_xlabel("Maximum Principal Stress")
    ax.set_ylabel("Probability Density")
    return fig
end

function SintSminHist(stress_samples::Vector{StressState};nbins=50)
    sint_samples = GetSintSamples(stress_samples)
    smin_samples = GetSminSamples(stress_samples)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = ax.hist2d(smin_samples,sint_samples,
                  bins=(nbins,nbins),
                  cmap=plt.cm.Greys,
                  density=true)
    plt.colorbar(h[3])
    ax.set_xlabel("Minimum Principal Stress")
    ax.set_ylabel("Intermediate Principal Stress")
    return ax,fig

end

function StressPolygonHist(stress_samples::Vector{StressState};nbins=50,shmax_component=2,shmin_component=3)

    if shmax_component == 1
        shmax = GetS1Samples(stress_samples)
    elseif shmax_component == 2
        shmax = GetS2Samples(stress_samples)
    elseif shmax_component == 3
        shmax = GetS3Samples(stress_samples)
    else
        error("Error: shmax_component must be 1, 2, or 3")
    end
    if shmin_component == 1
        shmin = GetS1Samples(stress_samples)
    elseif shmin_component == 2
        shmin = GetS2Samples(stress_samples)
    elseif shmin_component == 3
        shmin = GetS3Samples(stress_samples)
    else
        error("Error: shmin_component must be 1, 2, or 3")
    end
    if shmax_component == shmin_component
        error("Error: shmax_component and shmin_component must be different")
    end

    # swap them if needed so that S1 is more compressive than S2
    for i in range(1,stop=length(shmin))
        # if S1 (SHmax) is less than S2 ()
        if (shmax[i]<shmin[i])
            t = shmin[i]
            shmin[i] = shmax[i]
            shmax[i] = t
        end
    end

    fig = plt.figure()
    ax = fig.add_subplot(111)
    h = ax.hist2d(shmin,shmax,
                  bins=(nbins,nbins),
                  cmap=plt.cm.Greys,
                  density=true)
    plt.colorbar(h[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Minimum Principal Stress")
    ax.set_ylabel("Intermediate/Maximum Principal Stress")
    return ax,fig

end

function GetPorePressureSamples(stress_samples::Vector{StressState})
    ptype = typeof(stress_samples[1].Pp)
    pp_samples = Vector{ptype}(undef,length(stress_samples))
    i=1
    for s in stress_samples
        pp_samples[i] = s.Pp
        i += 1
    end
    return pp_samples
end

function GetAlphaSamples(stress_samples::Vector{StressState})
    atype = typeof(stress_samples[1].alpha)
    alpha_samples = Vector{atype}(undef,length(stress_samples))
    i=1
    for s in stress_samples
        alpha_samples[i] = s.alpha
        i += 1
    end
    return alpha_samples
end

function GetBetaSamples(stress_samples::Vector{StressState})
    btype = typeof(stress_samples[1].beta)
    beta_samples = Vector{btype}(undef,length(stress_samples))
    i=1
    for s in stress_samples
        beta_samples[i] = s.beta
        i += 1
    end
    return beta_samples
end


function GetGammaSamples(stress_samples::Vector{StressState})
    gtype = typeof(stress_samples[1].gamma)
    gamma_samples = Vector{gtype}(undef,length(stress_samples))
    i=1
    for s in stress_samples
        gamma_samples[i] = s.gamma
        i += 1
    end
    return gamma_samples
end

function GetS1Samples(stress_samples::Vector{StressState})

    stype = typeof(stress_samples[1].S1)
    S1_samples = Vector{stype}(undef,length(stress_samples))
    i=1
    for s in stress_samples
        S1_samples[i] = s.S1
        i += 1
    end
    return S1_samples
end

function GetS2Samples(stress_samples::Vector{StressState})

    stype = typeof(stress_samples[1].S2)
    S2_samples = Vector{stype}(undef,length(stress_samples))
    i=1
    for s in stress_samples
        S2_samples[i] = s.S2
        i += 1
    end
    return S2_samples
end

function GetS3Samples(stress_samples::Vector{StressState})

    stype = typeof(stress_samples[1].S3)
    S3_samples = Vector{stype}(undef,length(stress_samples))
    i=1
    for s in stress_samples
        S3_samples[i] = s.S3
        i += 1
    end
    return S3_samples
end

"""
    _get_component_samples(stress_samples::Vector{StressState}, component::Symbol)

Helper function to extract samples for a specific stress component using symbols.
Valid components are: :S1, :S2, :S3, :Pp, :alpha, :beta, :gamma
"""
function _get_component_samples(stress_samples::Vector{StressState}, component::Symbol)
    if component == :S1
        return GetS1Samples(stress_samples)
    elseif component == :S2
        return GetS2Samples(stress_samples)
    elseif component == :S3
        return GetS3Samples(stress_samples)
    elseif component == :Pp
        return GetPorePressureSamples(stress_samples)
    elseif component == :alpha
        return GetAlphaSamples(stress_samples)
    elseif component == :beta
        return GetBetaSamples(stress_samples)
    elseif component == :gamma
        return GetGammaSamples(stress_samples)
    elseif component == :Smin
        return GetSminSamples(stress_samples)
    elseif component == :SHmax
        return GetSHmaxSamples(stress_samples)
    elseif component == :SHmin
        return GetSHminSamples(stress_samples)
    elseif component == :Sv
        return GetSvSamples(stress_samples)
    else
        error("Invalid stress component: $component. Valid options are: :S1, :S2, :S3, :Pp, :alpha, :beta, :gamma, :Smin")
    end
end

function GetStressComponentStatistics(stress_samples::Vector{StressState}, component::Symbol)

    comp_samples = _get_component_samples(stress_samples, component)

    median_val = median(comp_samples)
    range_val = maximum(comp_samples) - minimum(comp_samples)
    IQR_val = quantile(comp_samples, 0.75) - quantile(comp_samples, 0.25)

    return median_val, range_val, IQR_val
end

"""
    GetStressComponentCorrelation(stress_samples::Vector{StressState}, component1::Symbol, component2::Symbol)
Compute the Pearson correlation coefficient between two stress components
across the provided stress samples. `component1` and `component2` should be valid component symbols
(:S1, :S2, :S3, :Pp, :alpha, :beta, :gamma)
"""
function GetStressComponentCorrelation(stress_samples::Vector{StressState}, component1::Symbol, component2::Symbol)

    comp1_samples = _get_component_samples(stress_samples, component1)
    comp2_samples = _get_component_samples(stress_samples, component2)

    corr_val = cor(comp1_samples, comp2_samples)

    return corr_val
end
function GetSminSamples(stress_samples::Vector{StressState})

    stype = typeof(stress_samples[1].S3)
    Smin_samples = Vector{stype}(undef,length(stress_samples))
    i=1
    for s in stress_samples
        Smin_samples[i] = minimum([s.S3,s.S2,s.S1])
        i += 1
    end
    return Smin_samples
end

function GetSintSamples(stress_samples::Vector{StressState})

    stype = typeof(stress_samples[1].S2)
    Sint_samples = Vector{stype}(undef,length(stress_samples))
    i=1
    for s in stress_samples
        smin,sint,smax = sort([s.S1,s.S2,s.S3])
        Sint_samples[i] = sint
        i += 1
    end
    return Sint_samples
end


function GetSmaxSamples(stress_samples::Vector{StressState})

    stype = typeof(stress_samples[1].S1)
    Smax_samples = Vector{stype}(undef,length(stress_samples))
    i=1

    for s in stress_samples
        smin,sint,smax = sort([s.S1,s.S2,s.S3])
        Smax_samples[i] = smax
        i += 1
    end
    return Smax_samples
end

function GetSHminSamples(stress_samples::Vector{StressState})
    _, _, SHmin, _, _, _ = GeographicallyOrderStresses(stress_samples)

    return SHmin
end

function GetSHmaxSamples(stress_samples::Vector{StressState})
    _, SHmax, _, _, _, _ = GeographicallyOrderStresses(stress_samples)

    return SHmax
end

function GetSvSamples(stress_samples::Vector{StressState})
    Sv, _, _, _, _, _ = GeographicallyOrderStresses(stress_samples)

    return Sv
end

function GetSHmaxAzimuth(stress_samples::Vector{StressState})
    _, _, _, _, SHmax_dir, _ = GeographicallyOrderStresses(stress_samples)
    SHmax_azimuth = atan.(SHmax_dir[:, 2], SHmax_dir[:, 1])
    return SHmax_azimuth
end
"""
    GetStressMatrix(samples::Vector{StressState}; thin::Int=1)
"""
function GetStressMatrix(samples::Vector{StressState};
                                          thin::Int=1)

    s1_samples = GetS1Samples(samples)
    s2_samples = GetS2Samples(samples)
    s3_samples = GetS3Samples(samples)
    pp_samples = GetPorePressureSamples(samples)
    alpha_samples = GetAlphaSamples(samples)
    beta_samples = GetBetaSamples(samples)
    gamma_samples = GetGammaSamples(samples)
    data = hcat(s1_samples, s2_samples, s3_samples, pp_samples, alpha_samples, beta_samples, gamma_samples)
    if thin > 1
        data = data[1:thin:end, :]
    end
    return data
end


function WriteTraceToH5(samples, filename)

    # create an array for the geographic stresses
    s_xx = Vector{Float64}(undef, length(samples))
    s_yy = Vector{Float64}(undef, length(samples))
    s_zz = Vector{Float64}(undef, length(samples))
    s_yz = Vector{Float64}(undef, length(samples))
    s_xz = Vector{Float64}(undef, length(samples))
    s_xy = Vector{Float64}(undef, length(samples))
    S1 = Vector{Float64}(undef, length(samples))
    S2 = Vector{Float64}(undef, length(samples))
    S3 = Vector{Float64}(undef, length(samples))
    alpha = Vector{Float64}(undef, length(samples))
    beta = Vector{Float64}(undef, length(samples))
    gamma = Vector{Float64}(undef, length(samples))
    Pp = Vector{Float64}(undef, length(samples))

    i = 1
    for s in samples
        s_geo_i = BS.PrincipalToGeographic(s.S1, s.S2, s.S3,
                                           s.alpha, s.beta, s.gamma)
        s_xx[i] = s_geo_i[1, 1]
        s_yy[i] = s_geo_i[2, 2]
        s_zz[i] = s_geo_i[3, 3]
        s_yz[i] = s_geo_i[2, 3]
        s_xz[i] = s_geo_i[1, 3]
        s_xy[i] = s_geo_i[1, 2]
        S1[i] = s.S1
        S2[i] = s.S2
        S3[i] = s.S3
        alpha[i] = s.alpha
        beta[i] = s.beta
        gamma[i] = s.gamma
        Pp[i] = s.Pp
        i += 1
    end
    fid = h5open(filename, "w")
    fid["S_XX"] = s_xx
    fid["S_YY"] = s_yy
    fid["S_ZZ"] = s_zz
    fid["S_YZ"] = s_yz
    fid["S_XZ"] = s_xz
    fid["S_XY"] = s_xy
    fid["S1"] = S1
    fid["S2"] = S2
    fid["S3"] = S3
    fid["alpha"] = alpha
    fid["beta"] = beta
    fid["gamma"] = gamma
    fid["Pp"] = Pp
    close(fid)
end

"""
ReadGeographicStressRealizations(filename)

Reads geographic stress realizations from an h5 file and loads 
them into a 6xk matrix where k is the number of realizations.

Returns:
a 6xk matrix with the geographic stresses
"""
function ReadGeographicStressRealizations(filename)

    fid = h5open(filename, "r")

    S_XX = fid["S_XX"]
    S_YY = fid["S_YY"]
    S_ZZ = fid["S_ZZ"]
    S_YZ = fid["S_YZ"]
    S_XZ = fid["S_XZ"]
    S_XY = fid["S_XY"]
    k = length(S_XX)

    Smat = Matrix{Float64}(undef,6,k)
    Smat[1,:] = S_XX[:]
    Smat[2,:] = S_YY[:]
    Smat[3,:] = S_ZZ[:]
    Smat[4,:] = S_YZ[:]
    Smat[5,:] = S_XZ[:]
    Smat[6,:] = S_XY[:]

    return Smat
end

function LoadStressStateFromFile(filename)

    fid = h5open(filename, "r")

    S_1 = fid["S1"]
    S_2 = fid["S2"]
    S_3 = fid["S3"]
    Pp = fid["Pp"]
    alpha = fid["alpha"]
    beta = fid["beta"]
    gamma = fid["gamma"]
    k = length(S_1)

    ss = Vector{BS.StressState}(undef,k)

    for i in range(1,stop=k)
        ss[i] = BS.StressState(S_1[i],
                           S_2[i],
                           S_3[i],
                           Pp[i],
                           alpha[i],
                           beta[i],
                           gamma[i])
    end
    return ss
end

function GeographicallyOrderStresses(stress_states::Vector{StressState})
    Sv_array = Vector{Float64}(undef,length(stress_states))
    SHmax_array = Vector{Float64}(undef,length(stress_states))
    SHmin_array = Vector{Float64}(undef,length(stress_states))
    Sv_dir_array = Array{Float64}(undef,length(stress_states),3)
    SHmax_dir_array = Array{Float64}(undef,length(stress_states),3)
    SHmin_dir_array = Array{Float64}(undef,length(stress_states),3)
    i=1
    for s in stress_states
        Sv, SHmax, SHmin, Sv_dir, SHmax_dir, SHmin_dir = GeographicallyOrderStresses(s)
        Sv_array[i] = Sv
        SHmax_array[i] = SHmax
        SHmin_array[i] = SHmin
        Sv_dir_array[i,:] = Sv_dir
        SHmax_dir_array[i,:] = SHmax_dir
        SHmin_dir_array[i,:] = SHmin_dir
        i += 1
    end
    return Sv_array, SHmax_array, SHmin_array, Sv_dir_array, SHmax_dir_array, SHmin_dir_array
end

function GeographicallyOrderStresses(s::StressState)
   
    # first get the principal stress directions
    direction_vecs = BS.PrincipalStressDirections(s)

    # now find which has the largest magnitude in the 3 dir
    third_components = [abs(direction_vecs[1][3]), abs(direction_vecs[2][3]), abs(direction_vecs[3][3])]
    index_of_largest = argmax((third_components))
    SV_index = Int[]
    SHmax_index = Int[]
    SHmin_index = Int[]
    if index_of_largest == 1
        # S1 is most vertical
        Sv = s.S1
        SV_index = 1
        if s.S2 > s.S3
            SHmax = s.S2
            SHmax_index = 2
            SHmin = s.S3
            SHmin_index = 3
        else
            SHmax = s.S3
            SHmax_index = 3
            SHmin = s.S2
            SHmin_index = 2
        end
    elseif index_of_largest == 2
        # S2 is most vertical
        Sv = s.S2
        SV_index = 2
        if s.S1 > s.S3
            SHmax = s.S1
            SHmax_index = 1
            SHmin = s.S3
            SHmin_index = 3
        else
            SHmax = s.S3
            SHmax_index = 3
            SHmin = s.S1
            SHmin_index = 1
        end
    else
        # S3 is most vertical
        Sv = s.S3
        SV_index = 3
        if s.S1 > s.S2
            SHmax = s.S1
            SHmax_index = 1
            SHmin = s.S2
            SHmin_index = 2
        else
            SHmax = s.S2
            SHmax_index = 2
            SHmin = s.S1
            SHmin_index = 1
        end
    end

    return Sv, SHmax, SHmin, direction_vecs[SV_index], direction_vecs[SHmax_index], direction_vecs[SHmin_index]
end
