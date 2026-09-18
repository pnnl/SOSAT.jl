module BoreholeStress

BS = BoreholeStress

using LinearAlgebra
using Roots
using Optim
using Rotations: RotZYX

psi2MPa = 1.0/145.0377
MPa2psi = 1.0/psi2MPa
m2ft = 3.2808
ft2m = 1.0/m2ft

"""
    StressState(S1, S2, S3, Pp, alpha, beta, gamma)

A struct to specify an arbitrary 3D stress state

# Arguments
S1 
    the stress magnitude in the x1 (north) direction in the unrotated frame
S2
    the stress magnitude in the x2 (east) direction in the unrotated frame
S3
    the stress magnitude in the x3 (vertical) direction in the unrotated frame
Pp
    the pore pressure
alpha
    rotation about the x3 axis
beta
    rotation about th x2' axis
gamma
    rotation about the x3'' axis

Notes
-----
A general 3D stress state can be constructed with the three magnitudes and
three Euler angles. The rotations are applied by first rotating by `alpha`
about the x3 (vertical) axis. The axes after this first rotation are denoted
x1', x2', and x3'. Next, a rotation of `beta` is applied about the x2' axis,
giving new axes x1'', x2'', and x3''. Finally, a rotation of `gamma` is applied
about the x3'' axis.

This is known as the ZYZ convention for Euler angles. There are two pitfalls
of this approach. First, if beta = 0 or pi, then alpha and gamma are not
uniquely defined (this is known as gimbal lock). Second, there are multiple
combinations of angles that can produce the same final orientation. For example,
(alpha, beta, gamma) and (alpha+pi, pi-beta, gamma+pi) will produce the same final
orientation.

To avoid the second problem, we will use the Rotation.jl package to find the cannonical
representation of the Euler angles from the perturbed rotation matrix.

"""
mutable struct StressState
    S1::AbstractFloat
    S2::AbstractFloat
    S3::AbstractFloat
    Pp::AbstractFloat
    alpha::AbstractFloat
    beta::AbstractFloat
    gamma::AbstractFloat
end

function natural_canonicalize!(s::StressState)
    # Use Rotations.jl's natural canonicalization
    R = Matrix(RotZYX(s.alpha, s.beta, s.gamma))
    euler = RotZYX(R)
    
    s.alpha = euler.theta1  # Will be in [-π, π]
    s.beta = euler.theta2   # Will be in [0, π]  
    s.gamma = euler.theta3  # Will be in [-π, π]
end
"""
The well state is defined with the following parameters:

Pw
    Wellbore pressure
phi
    Deviation angle, which is the angle formed between a downward vertical
    vector and the axis of the well
delta
    Azimuth of the horizontal projection of the well axis. Undefined for
    a vertical well
"""
mutable struct WellState
    Pw
    phi
    delta
end

"""
    S3FrictionalLimit(S1, Pp, mu)

Calculate the minimum value of the minimum principal stress, S3, allowed with a
fault friction coefficient of mu and a pore pressure of Pp
"""
function S3FrictionalLimit(S1, Pp, mu)
    sig1 = S1-Pp

    fac = (sqrt(mu^2+1)+mu)^2

    sig3 = sig1/fac

    return sig3+Pp
end

"""
    EulerRotation(alpha, beta, gamma)

This function returns the rotation matrix that transforms the geographic
stress tensor to the principal stress tensor using ZYX Euler angles. This is
the same as the INVERSE rotation matrix for transforming FROM principal stress 
coordinates TO geographic coordinates.

The columns of this rotation matrix are the principal stress directions expressed
in geographic coordinates, with the first column being the direction of S1,
the second column being the direction of S2, and the third column being the
direction of S3.

The rotation matrix returned from this can be used to compute the geographic
stress tensor from the principal stress tensor as follows:
    S_geographic = R^T × S_principal × R

The rotations are applied by first rotating by `alpha`
about the x3 (vertical) axis. The axes after this first rotation are denoted
x1', x2', and x3'. Next, a rotation of `beta` is applied about the x2' axis,
giving new axes x1'', x2'', and x3''. Finally, a rotation of `gamma` is applied
about the x3'' axis.

This follows ZYX intrinsic Euler angle convention and returns the inverse matrix for 
use in stress tensor transformations. The angle alpha has a range defined modulo 2π.
In general this could be any range of this length (e.g. [-π, π] or [0, 2π]), but since
I use the Rotations.jl package we will use the range assumed by that package, which is
[-π, π]. The angle
beta is defined in the range [0, π], and gamma is defined modulo
"""
function EulerRotation(alpha, beta, gamma)
    # Use Rotations.jl for mathematical correctness
    # Return transpose for our stress transformation convention
    R_forward = Matrix(RotZYX(alpha, beta, gamma))
    return transpose(R_forward)
end

"""
    PrincipalToGeographic(S1, S2, S3, alpha, beta, gamma)

Compute the stress tensor as a 3x3 matrix using three stress magnitudes and
three Euler angles.
"""
function PrincipalToGeographic(S1, S2, S3, alpha, beta, gamma)
    Ss = [S1 0 0; 0 S2 0; 0 0 S3]
    R = EulerRotation(alpha,beta,gamma)
    return Transpose(R)*Ss*R
end

function GeographicStress(SS::StressState)
    return PrincipalToGeographic(SS.S1,
                                SS.S2,
                                SS.S3,
                                SS.alpha,
                                SS.beta,
                                SS.gamma)
end

function PrincipalStressDirections(SS::StressState)
    R = EulerRotation(SS.alpha, SS.beta, SS.gamma)
    # Return the rows of R as the principal stress directions
    S1_dir = R[1, :]  # [north, east, down] components
    S2_dir = R[2, :]
    S3_dir = R[3, :]
    return [S1_dir, S2_dir, S3_dir]
end

"""
    PrincipalStressOrientations(SS::StressState)

Compute the geographical orientation parameters (azimuth and plunge) for each 
principal stress direction from the Euler angles in the StressState.

Returns a vector of named tuples with azimuth (degrees from north, clockwise positive)
and plunge (degrees below horizontal) for S1, S2, and S3 respectively.
"""
function PrincipalStressOrientations(SS::StressState)
    # Get principal stress directions from rotation matrix
    directions = PrincipalStressDirections(SS)
    
    orientations = []
    
    for dir in directions
        north, east, down = dir
        
        # Azimuth (from north, clockwise positive)
        azimuth = atan(east, north)
        if azimuth < 0
            azimuth += 2π
        end
        
        # Plunge (angle below horizontal, positive downward)
        # Note: In your coordinate system, positive down is negative z
        horizontal_mag = sqrt(north^2 + east^2)
        plunge = atan(down, horizontal_mag)  # Negate down to get correct sign
        
        push!(orientations, (azimuth=rad2deg(azimuth), plunge=rad2deg(plunge)))
    end
    
    return orientations
end


"""
    GeographicToBoreholeRotation(delta, phi)

Computes the rotation matrix that transforms from a geographic coordinate system
to a local borehole coordinate system.

The geographic coordinate system has x1 in the north direction, x2 in the east
direction, and x3 downward. The borehole coordinate system is described in
terms of unit vectors x1b, x2b, and x3b. x3b is directed along the borehole
axis in the direction of increasing measured depth; x1b is perpendicular to
x3b and toward the bottom of the well; x2b is horizontal and orthogonal to both
x1b and x3b; the system is described in Figure A1 in Peska and Zoback, 1995;

Note that for zero inclination (i.e. a vertical well), the azimuth is technicall undefined but 
this rotation matrix will produce a coordinate system with x2b pointing south and x1b pointing west. To
get geographic (North, East, Down) coordinates from borehole coordinates in this case the azimuth
should be set to 180 degrees.
"""
function GeographicToBoreholeRotation(delta, phi)
    # note that this makes for a non-righ-hand system
    R = zeros(Float64,3,3)

    R[1,1] = -cos(delta)*cos(phi)
    R[2,1] = sin(delta)
    R[3,1] = cos(delta)*sin(phi)

    R[1,2] = -sin(delta)*cos(phi)
    R[2,2] = -cos(delta)
    R[3,2] = sin(delta)*sin(phi)

    R[1,3] = sin(phi)
    R[2,3] = 0
    R[3,3] = cos(phi)

    return R
end

"""
    GeographicToBorehole(Sg, delta, phi)

Transforms a stress tensor from a geographic coordinate system to the local
wellbore system.

The geographic stress tensor, Sg, should be specified as a 3x3 matrix. The well
is described by its azimuth, delta, and deviation, phi.
"""
function GeographicToBorehole(Sg, delta, phi)
    R = GeographicToBoreholeRotation(delta,phi)
    return R*Sg*Transpose(R)

end

"""
    BoreholeRadialVector(theta)

Compute the Cartesian radial vector in the local wellbore coordinate system for
the radial angle theta.

Theta is measured from the x1b unit vector with positive (right-hand) rotation
about the x3b vector; x3b is directed downward along the well axis and x1b is
directed toward the bottom of the well
"""
function BoreholeRadialVector(theta)

    rvec = [cos.(theta),sin.(theta),0.0]
    return rvec
end

"""
    BoreholeThetaVector(theta)

Return a unit vector tangent to the radial vector at angle theta from the
borehole x1b axis. The angle theta, which describes the orientation of
the radial vector, and the vector theta, which is the borehole tangent
vector should not be confused (despite the confusing naming convention)
"""
function BoreholeThetaVector(theta)
    rvec = BoreholeRadialVector(theta)
    zvec = [0.0,0.0,1.0]
    tvec = cross(zvec,rvec)
    return tvec
end

"""
    BoreholeDipVector(theta, omega)

Compute the dip vector of the wellbore as a 3D vector in geographic coordinates. This is not computing a general dip vector from
geographic coordinates to borehole coordinates, but is instead
expressing the borehole trajectory itself as a dip vector
"""
function BoreholeDipVector(theta, omega)

    dipvec = [-sin.(theta)*sin.(omega),
              cos.(theta)*sin.(omega),
              cos.(omega)]

    if dipvec[3]<0.0
        dipvec = -dipvec
    end

    return dipvec
end

"""
    DipVectorFromDipAndTrend(dip_angle, dip_trend)

Compute the dip vector in 3D geographic coordinates from a specified dip angle
and dip trend azimuth.

As a reminder, dip angle is measured from horizontal in the direction of
steepest descent of a plane; dip_trend is the orientation, measured from north
of the horizontal projection of the dip vector; the dip vector is a Cartesian
vector directed downward along the direction of steepest descent of a plane
"""
function DipVectorFromDipAndTrend(dip_angle, dip_trend)

    dipvec= [cos.(dip_angle)*cos.(dip_trend),
             cos.(dip_angle)*sin.(dip_trend),
             sin.(dip_angle)]

    if dipvec[3]<0.0
        dipvec = -dipvec
    end
    return dipvec
end

"""
    DipTrendFromDipVector(dip_vec)

Compute the dip and dip trend from a 3D Cartesian dip vector

Returns:
dip, dip_trend
"""
function DipTrendFromDipVector(dip_vec)

    Nvec = [1.0, 0.0, 0.0]
    Evec = [0.0, 1.0, 0.0]
    Dvec = [0.0, 0.0, 1.0]
    Ncomp = dot(Nvec, dip_vec)
    Ecomp = dot(Evec, dip_vec)
    
    mag = norm(dip_vec)
    Hmag = sqrt(Ncomp^2 + Ecomp^2)
    dip_trend = atan(Ecomp,Ncomp)

    dip_angle = acos(Hmag/mag)

    return dip_angle, dip_trend
end


"""
    PoleVectorFromDipAndTrend(dip_angle, dip_trend)

Returns a pole vector for a plane given a dip angle and dip trend
"""
function PoleVectorFromDipAndTrend(dip_angle, dip_trend)

    polevec= [-sin.(dip_angle).*cos.(dip_trend),
              -sin.(dip_angle).*sin.(dip_trend),
               cos.(dip_angle)]
    if polevec[3]<0.0
        polevec = -polevec
    end

    return polevec
end

"""
    BoreholePoleVector(theta, omega)

Returns the pole vector of a fracture expressed in well coordinates.

theta
    azimuth of the fracture obserations
omega
    orientation of the fracture relative to a plan perpendicular to the
    wellbore axis

Returns:

A 3D pole vector in local well coordinates
"""
function BoreholePoleVector(theta, omega)

    pvec= [sin.(theta)*cos.(omega),
           -cos.(theta)*cos.(omega),
           sin.(omega)]

    if pvec[3]<0.0
        pvec = - pvec
    end

    return pvec
end

"""
    BoreholeStresses(Sb, PR, Pp, P;theta=range(0, stop=Base.MathConstants.pi, length=101))

Computes the stress state at the borehole wall for an arbitrary far-field
stress state.

Arguments:
Sb
    The far-field stresses expressed in the borehole coordinate system
PR
    The Poisson's ratio
Pp
    The pore pressure in the formation
P
    The pressure of the fluid inside the borehole

Returns:
theta
    the azimuth considered, either a scalar or an array of values
sig_rr
    the magnitude of the radial stress at the borehole wall, equal to the difference
    between the borehole pressure and the pore pressure
sigt_max
    the magnitude of the most compressive tangential principal stress at the borehole wall
sigt_min
    the magnitude of the least compressive tangential principal stress at the borehole wall
omega
    the angle formed between the direction of maximum tangential principal stress and the
    tangent to the borehole axis. See Figure A2
    in Peska and Zoback, 1995
"""
function BoreholeStresses(Sb, PR, Pp, P;
                          theta=range(0,
                          stop=Base.MathConstants.pi,
                          length=101))
    sig = Sb - Matrix(1.0I,3,3)*Pp
    dP = P-Pp

    #sigzz = sig[3,3].-2.0*PR*(sig[1,1]-sig[2,2]).*cos.(2.0*theta).+4*PR*sig[1,2]*sin.(2.0*theta)
    # comparing to other solutions, it looks like the sign on the sig[1,2] term is wrong in the Peska & Zoback paper
    sigzz = sig[3,3].-2.0*PR*(sig[1,1]-sig[2,2]).*cos.(2.0*theta).-4*PR*sig[1,2]*sin.(2.0*theta)
    #sigtt = sig[1,1].+sig[2,2].-2.0*(sig[1,1]-sig[2,2]).*cos.(2.0*theta).+4*sig[1,2].*sin.(2.0*theta).-dP
    # comparing to other solutions, it looks like the sign on the sig[1,2] term is wrong in the Peska & Zoback paper
    sigtt = sig[1,1].+sig[2,2].-2.0*(sig[1,1]-sig[2,2]).*cos.(2.0*theta).-4*sig[1,2].*sin.(2.0*theta).-dP
    sigtz = 2.0.*(sig[2,3].*cos.(theta).-sig[1,3].*sin.(theta))
    sigrr= ones(Float64,length(theta))*dP
    sigt_max = zeros(Float64,length(theta))
    sigt_min = zeros(Float64,length(theta))
    omega = zeros(Float64,length(theta))

    for i in 1:1:length(theta)

        s = zeros(Float64,2,2)
        #s[1,1] = sigrr[i]
        s[1,1] = sigtt[i]
        s[2,2] = sigzz[i]
        s[1,2] = sigtz[i]
        s[2,1] = s[1,2]
        #sig3[i],sig2[i],sig1[i] = sort(eigvals(s))
        sigt_max[i] = 0.5*(sigzz[i]+sigtt[i]+sqrt((sigzz[i]-sigtt[i])^2+4.0*sigtz[i]^2))
        sigt_min[i] = 0.5*(sigzz[i]+sigtt[i]-sqrt((sigzz[i]-sigtt[i])^2+4.0*sigtz[i]^2))
        #omega[i] = 0.5*atan(2.0*sigtz[i]/(sigzz[i]-sigtt[i]))+Base.MathConstants.pi*0.5
        #E = eigvecs(s,[sig3[i],sig2[i],sig1[i]])
        D,V = eigen(s)

        # we need to find the largest (most positive) eigenvalue, which will
        # be the maximum compressive stress. The associated eigenvector will
        # be the direction of the maximum principal stress and tensile fractures
        # will propagate in this direction. The direction will be defined relative
        # to the {r,theta,z} coordinate system, where in this case Theta
        # is actually a cartesian vector acting tangent to the borehole wall
        # in the direction of its circumference. If

        sigmax,indmax = findmax(D)
        emax = V[:,indmax]
        # emax[1] will be the tangential component of the maximum principal stress
        # emax[2] will be the axial component of the maximum principal stress
        omega[i] = atan(emax[1],emax[2])
        if (omega[i]<0.0)
            omega[i]+=Base.MathConstants.pi
        end
    end


    return theta,sigrr,sigt_max,sigt_min,omega
end

#=
function SleeveOpeningPressure(Sb,PR,Pp)

    BoreholeStresses(Sb,PR,Pp,P

end
=#


"""
    NormalAndShear(s::StressState, b::WellState, theta, omega;PR=0.25)

Computes the normal and shear stress at the wellbore wall on a specified plane.

An arbitrary stress state (including pore pressure) can be specified using a
StressState struct.

An arbitrary well orientation and wellbore pressure can be specified using a
WellState struct.

The plane of interest is specified using the two angles theta and omega. Theta
is specified in the local wellbore coordinate system measured as a positive
rotation about the downward wellbore axis measured from the bottom of the well.
Omega is the angle of the plane measured from a plane perpendicular to the
well axis.

Returns:

normal_stress
    A scalar or array containing the normal stress magnitude (same shape as theta)
shear_stress
    A scalar or array containing the shear stress magnitude (same shape as theta)
"""
function NormalAndShear(s::StressState, b::WellState, theta, omega;PR=0.25)
    Sg = PrincipalToGeographic(s.S1,
                               s.S2,
                               s.S3,
                               s.alpha,
                               s.beta,
                               s.gamma)

    S = GeographicToBorehole(Sg,b.delta,b.phi)

    sig = S - Matrix(1.0I,3,3)*s.Pp

    dP = b.Pw-s.Pp


    sigzz = sig[3,3].-2.0*PR*(sig[1,1]-sig[2,2]).*cos.(2.0*theta).-4*PR*sig[1,2]*sin.(2.0*theta)
    sigtt = sig[1,1].+sig[2,2].-2.0*(sig[1,1]-sig[2,2]).*cos.(2.0*theta).-4*sig[1,2].*sin.(2.0*theta).-dP
    sigtz = 2.0.*(sig[2,3].*cos.(theta).-sig[1,3].*sin.(theta))

    #println("s= ",s)
    #println("b= ",b)
    #println("sigzz= ",sigzz)
    #println("sigtt= ",sigtt)
    #println("sigtz= ",sigtz)
    #println("sigrr = ",dP)
    #println("theta= ",theta)
    #println("omega= ",omega)

    slocal = zeros(Float64,3,3)

    slocal[1,1] = sigtt
    slocal[2,2] = sigzz
    slocal[1,2] = sigtz
    slocal[2,1] = slocal[1,2]
    slocal[3,3] = dP

    # rotate by angle omega about the r (3) direction
    Romega = [cos.(omega) -sin.(omega) 0; sin.(omega) cos.(omega) 0; 0 0 1]
    #println("Romega= ",Romega)
    s_omega = transpose(Romega)*slocal*Romega
    #println("s_omega= ",s_omega)
    return s_omega[1,1],s_omega[1,2]

end

"""
    BoreholeStresses(s::StressState, b::WellState, theta; PR=0.25)

Compute the stress tensor at the wellbore wall for an arbitrary stress state
and wellbore orientation

Returns:
theta
the azimuth considered, either a scalar or an array of values
sig_rr
the magnitude of the radial stress at the borehole wall, equal to the difference
between the borehole pressure and the pore pressure
sigt_max
the magnitude of the most compressive tangential principal stress at the borehole wall
sigt_min
the magnitude of the least compressive tangential principal stress at the borehole wall
omega
the angle formed between the direction of maximum tangential principal stress and the
tangent to the borehole axis.
"""
function BoreholeStresses(s::StressState, b::WellState, theta; PR=0.25)

    Sg = PrincipalToGeographic(s.S1,
                               s.S2,
                               s.S3,
                               s.alpha,
                               s.beta,
                               s.gamma)
    S = GeographicToBorehole(Sg,b.delta,b.phi)
    tr,sigrr,sigt_max,sigt_min,omega = BoreholeStresses(S,PR,s.Pp,b.Pw,theta=theta)
    return tr,sigrr,sigt_max,sigt_min,omega
end

"""
    MinimumPrincipalStress(s::StressState, b::WellState; PR=0.25)

Compute the minimum (least compressive) tangential principal stress at the borehole wall
for an arbitrary stress state and well orientation.

All locations around the circumference of the well are searched to find the
minimum principal stress and its azimuthal location.

Returns:
smin
    The magnitude of the minimum tangential principal stress
theta
    The azimuth (in local borehole system) where the minimum value occurs
omega
    The angle between the direction of minimum principal stress and the plane
    perpendicular to the wellbore axis
"""
function MinimumPrincipalStress(s::StressState, b::WellState; PR=0.25)

    function smin(t)
        tr,srr,sigt_max,sigt_min,o=BoreholeStresses(s,b,t[1])
        return sigt_min[1]
    end

    result = optimize(smin,0.0,Base.MathConstants.pi)
    tmin = Optim.minimizer(result)[1]
    tr,srr,sigt_max,sigt_min,omega=BoreholeStresses(s,b,tmin)
    return sigt_min[1],tmin[1],omega[1]

end

"""
    MaximumPrincipalStress(s::StressState, b::WellState; PR=0.25)

Compute the maximum (most compressive) principal tangential stress at the borehole wall
for an arbitrary stress state and well orientation.

All locations around the circumference of the well are searched to find the
maximum  tangential principal stress and its azimuthal location.

Returns:
smin
    The magnitude of the maximum principal stress
theta
    The azimuth (in local borehole system) where the maximum value occurs
"""
function MaximumPrincipalStress(s::StressState,b::WellState;PR=0.25)

    function smax(t)
        tr,srr,sigt_max,sigt_min,o=BoreholeStresses(s,b,t[1])
        # add negative sign to find maximum
        return -sigt_max[1]
    end

    result = optimize(smax,0.0,Base.MathConstants.pi)
    tmax = Optim.minimizer(result)[1]
    # compute the actual maximum value by multiplying by -1
    smax_val = -smax([tmax])
    return smax_val,tmax

end


"""
    FracturePressure(s::StressState, b::WellState;To=0.0, PR=0.25)

Computes the fracture pressure and expected fracture orientation for an
arbitrary stress state and wellbore orientation. Tensile strength, To, defaults
to zero and the Poisson's ratio defaults to 0.25

Returns:
Pfrac
    Expected fracture pressure
theta_min
    Azimuth of the expected fracture in the local wellbore coordinate system
omega
    Angle between the expected fracture plane and the plane perpendicular to
    the wellbore axis
"""
function FracturePressure(s::StressState, b::WellState;To=0.0, PR=0.25)

    function frac(P)
        bchange = deepcopy(b)
        bchange.Pw = P
        smin,tmin = MinimumPrincipalStress(s,bchange,PR=PR)
        #println("smin= ",smin)
        return smin+To
    end

    Pfrac = 0.0
    #result = nlsolve(frac,[0.0],method= :newton)
    # check if the well is already fractured at zero dP
    smin_dP0 = frac(s.Pp)
    if (smin_dP0<= 0.0)
        Pfrac = s.Pp
    else
        # search for a value of P that will provide an upper
        # bound for the bisection method. This should produce
        # a very robust solve
        Ptest = s.Pp*2.0+0.1  # this will help if Pp=0
        smin_P2 = frac(Ptest)
        while smin_P2>0.0
            Ptest *= 2.0
            smin_P2 = frac(Ptest)
        end

        Pfrac = find_zero(frac,(s.Pp,Ptest),Bisection())
    end

    bchange2 = deepcopy(b)
    bchange2.Pw = Pfrac
    smin,tmin = MinimumPrincipalStress(s,bchange2,PR=PR)
    ret = BoreholeStresses(s,bchange2,tmin,PR=PR)
    omega = ret[5][1]
    return Pfrac,tmin,omega
end

"""
    FracturePressure(s::StressState, b::WellState;To=0.0, PR=0.25)

Computes the fracture pressure and expected fracture orientation for an
arbitrary stress state and wellbore orientation. Tensile strength, To, defaults
to zero and the Poisson's ratio defaults to 0.25

Returns:
Pfrac
    Expected fracture pressure
theta_min
    Azimuth of the expected fracture in the local wellbore coordinate system
omega
    Angle between the expected fracture plane and the plane perpendicular to
    the wellbore axis
"""
function FractureOpeningPressure_wOrientation(s::StressState,
                                              b::WellState,
                                              theta,
                                              omega;
                                              To=0.0, PR=0.25)

    function Pnorm(P)
        bchange = deepcopy(b)
        bchange.Pw = P
        normal_stress, shear_stress = NormalAndShear(s,
                                                     bchange,
                                                     theta,
                                                     omega)
        return normal_stress
    end

    Popen = 0.0
    #result = nlsolve(frac,[0.0],method= :newton)
    # check if the fracture is already fractured at zero dP
    popen_dP0 = Pnorm(s.Pp)
    if (popen_dP0<= 0.0)
        Popen = s.Pp
    else
        # search for a value of P that will provide an upper
        # bound for the bisection method. This should produce
        # a very robust solve
        Ptest = s.Pp*2.0+0.1  # this will help if Pp=0
        popen_P2 = Pnorm(Ptest)

        while popen_P2>0.0
            Ptest *= 2.0
            popen_P2 = Pnorm(Ptest)
        end
        Popen = find_zero(Pnorm,(s.Pp,Ptest),Bisection())
    end

    return Popen
end

"""
    FindStress(b::WellState, S1, S3, Pp, dip_angle, dip_trend;
               PR=0.25,
               alpha_guess=deg2rad(20.0),
               beta_guess=deg2rad(70.0))

Finds the intermediate stress that predicts fracture opening pressing equal to
that specified in the WellState, b, with the specified maximum (S1) and minimum
(S3) principal stress, and pore pressure, Pp, with the specified fracture
dip angle and dip trend. Dip angle and dip trend should be specified in geographic
coordinates

Returns:
alpha
    First Euler angle
beta
    Second Euler angle
gamma
    Third Euler angle
S2
    intermediate principal stress
"""
function FindStress(b::WellState, S1, S3, Pp, dip_angle, dip_trend;
                   PR=0.25,
                   alpha_guess=deg2rad(20.0),
                   beta_guess=deg2rad(70.0))
    # dip angle and trend should be in geographic coordinates

    # create a dip vector in geographic coordinates
    #d_g = [cos(dip_angle)*cos(dip_trend),cos(dip_angle)*sin(dip_trend),sin(dip_angle)]
    p_g = PoleVectorFromDipAndTrend(dip_angle,dip_trend)
    function StressResidual(arg)
        alpha = arg[1]
        beta = arg[2]
        S2 = arg[3]
        bdummy = deepcopy(b)
        bdummy.Pw = Pp
        s = StressState(S1,S2,S3,Pp,alpha,beta,0.0)
        Pfrac_pred,tmin,omega = FracturePressure(s,bdummy,To=0.0,PR=PR)
        # create a predicted dip trend in borehole coordinates

        pole_pred_b = BoreholePoleVector(tmin,omega)

        Rg = GeographicToBoreholeRotation(b.delta,b.phi)

        pole_pred_g = transpose(Rg)*pole_pred_b

        err = 0.0
        # normalized fracture pressure error
        err += ((b.Pw-Pfrac_pred)/b.Pw)^2
        err += (pole_pred_g[1]-p_g[1])^2
        err += (pole_pred_g[2]-p_g[2])^2
        err += (pole_pred_g[3]-p_g[3])^2
        println("err= ",sqrt(err))
        return sqrt(err)
    end

    guess = [alpha_guess,beta_guess,deepcopy(b.Pw)]
    fo= StressResidual(guess)
    println("fo= ",fo)
    result = optimize(StressResidual,
                      guess,
                      Optim.Options(iterations=200,
                                    f_tol = 1.0e-4,
                                    show_every=50))   # need to double check the calling syntax here

    # this will return alpha, beta, gamma, and the intermediate principal stress
    return Optim.minimizer(result)
end


end
