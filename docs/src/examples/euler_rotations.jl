
# # Euler Rotations Tutorial
#
# This tutorial demonstrates how to transform stress tensors between
# coordinate systems using Euler rotations.
#
# ## Background
#
# In underground stress analysis, we often need to transform between:
# - Principal stress coordinates (aligned with stress eigenvectors)
# - Geographic coordinates (North-East-Down)
# - Borehole coordinates (aligned with measurement tools)
#
# Euler rotations provide a systematic way to perform these transformations. There are many different
# conventions for Euler angles; here we use the Z-Y'-X'' convention to go from the principal stress system
#  to the geographic system. In this convention:
#
# 1. Rotate by angle alpha about the Z axis (vertical)
# 2. Rotate by angle beta about the new Y axis (called Y' after the first rotation)
# 3. Rotate by angle gamma about the new X axis (called X'' after the second rotation)
# 
# In the formulation uses in SOSAT.jl the angles are defined over the ranges
# -pi <= alpha < pi
# -pi/2 <= beta <= pi/2
# -pi <= gamma < pi
#
# This is driven by the underling use of the Rotations.jl package for handling rotations.
#
# Different conventions may be used by other software packages or authors. For example Peska and Zoback (1995)
# use 0<= alpha < 2pi, -pi/2 <= beta <= pi/2, and 0 <= gamma < 2pi. The end result is the same as there
# are equivalent angles in both conventions that produce the same rotation matrix.
#
# ## Example

using SOSAT
using PyPlot
const plt = PyPlot
plt.matplotlib.use("Agg")

# This is just creating principal stress magnitudes, which aren't critical for the orientations
water_depth = 4000.0
water_density = 1025.0
basalt_depth = 1000.0
basalt_density = 2800.0

g = 9.81

# Calculate vertical stress from overburden weight
sigv = (water_density * g * water_depth +
        basalt_density * g * basalt_depth) / 1.0e6 # MPa
println("Vertical Stress is ",sigv)

pp_MPa = (water_depth + basalt_depth) * 10.0 / 1000.0 # assume 10 MPa/km
println("Pore pressure is ",pp_MPa)


# The initial orientation is with S1 oriented North, S2 East, S3 vertical

S1 = 120.0
S2 = sigv
S3 = sigv

# Euler angles defining stress tensor orientation
# alpha is a rotation about the Z axis (vertical)
# beta is a rotation about the new Y axis
# gamma is a rotation about the new X axis

alpha = deg2rad(90.0)

# after the alpha rotation S1 is East, S2 is South, S3 remains vertical

beta = deg2rad(90.0)

# afer rotating by beta about the new Y axis, S1 is now vertical, S2 is East, S3 is South

gamma = deg2rad(0.0)

# for this example we leave gamma at zero

# We create the stress state object using principal stresses and Euler rotations
so = SOSAT.StressState(S1, S2, S3, pp_MPa,
                        alpha,
                        beta,
                        gamma)
S_geo = SOSAT.GeographicStress(so)

println("Geographic Stress Tensor (MPa):")
println("|  $(round(S_geo[1,1], digits=2))  $(round(S_geo[1,2], digits=2))  $(round(S_geo[1,3], digits=2))|")
println("|  $(round(S_geo[2,1], digits=2))  $(round(S_geo[2,2], digits=2))  $(round(S_geo[2,3], digits=2))|")
println("|  $(round(S_geo[3,1], digits=2))  $(round(S_geo[3,2], digits=2))  $(round(S_geo[3,3], digits=2))|")


# Now we make a lower-hemisphere projection stereonet plot of the principal stress directions
# First we need to put the stress state object into a vector since the function for making a stereonet
# plot is meant to operate on a vector of stress state objects produced by MCMC sampling
stress_states = Vector{SOSAT.StressState}(undef,1)
stress_states[1] = so

stereonet_fig = SOSAT.PrincipalStressStereonet(stress_states)

stereonet_fig.savefig("Euler_example1_principal_stress_stereonet.png")

# Here is the resulting plot

#md # ![Stereonet Plot](Euler_example1_principal_stress_stereonet.png)

# As expected, S1 is vertical (appears at the bottom of the lower hemisphere plot),
# S3 is pointing East (90 degrees azimuth), and S2 is pointing South (180 degrees azimuth).

# Note that because we are visualizing principal stress directions, stress acts the same in two orthogonal
# directions. Thus, S3 pointing East is equivalent to S3 pointing West (270 degrees azimuth). Because of small
# numerical differences in the calculations when the principal stresses are very close to horizontal they may
# plot in either of the two equivalent directions. This is evident in this plot because S3 shows up at 270 degrees
# instead of 90 degrees. This is normal and expected.

# Now we look at a variation on this example where we set gamma to -20 degrees so that SHmax is
# oriented 20 degrees East of North

gamma = deg2rad(-20.0)
so2 = SOSAT.StressState(S1, S2, S3, pp_MPa,
                        alpha,
                        beta,
                        gamma)
S_geo2 = SOSAT.GeographicStress(so2)
println("Geographic Stress Tensor with gamma=-20 degrees (MPa):")
println("|  $(round(S_geo2[1,1], digits=2))  $(round(S_geo2[1,2], digits=2))  $(round(S_geo2[1,3], digits=2))|")
println("|  $(round(S_geo2[2,1], digits=2))  $(round(S_geo2[2,2], digits=2))  $(round(S_geo2[2,3], digits=2))|")
println("|  $(round(S_geo2[3,1], digits=2))  $(round(S_geo2[3,2], digits=2))  $(round(S_geo2[3,3], digits=2))|")

stress_states2 = Vector{SOSAT.StressState}(undef,1)
stress_states2[1] = so2
stereonet_fig2 = SOSAT.PrincipalStressStereonet(stress_states2)
stereonet_fig2.savefig("Euler_example2_principal_stress_stereonet.png")
# Here is the resulting plot

#md # ![Stereonet Plot](Euler_example2_principal_stress_stereonet.png)

# As expected, S1 is still vertical (bottom of the plot), S2 is now oriented at 20 degrees/200 degrees azimuth,
# and S3 is oriented at 110 degrees/290 degrees azimuth. As explained above, only one of these two orthogonal directions
# is shown for each principal stress.


# As a final example, set alpha 110 degrees and set beta to 75 degrees so that S3 plunges 15 degrees from horizontal
# and S1 deviates from vertical by 15 degrees. We set gamma to zero.

alpha = deg2rad(110.0)

# after rotation about the z axis by 110 degrees S1 is oriented 110 degrees, S2 is oriented 200 degrees, S3 remains vertical

beta = deg2rad(75.0)

# after rotation about the new Y axis by 75 degrees, S3 plunges 15 degrees from horizontal toward 110 degrees azimuth
# S2 remains in the horizontal plane at 200 degrees azimuth, and S1 deviates 15 degrees from vertical toward 110 degrees azimuth
gamma = deg2rad(0.0)

# The rotation matrix for this is found with

R = SOSAT.EulerRotation(alpha, beta, gamma)

# Which is
println("Rotation Matrix for alpha=110, beta=75, gamma=0 degrees:")
println("|  $(round(R[1,1], digits=4))  $(round(R[1,2], digits=4))  $(round(R[1,3], digits=4))|")
println("|  $(round(R[2,1], digits=4))  $(round(R[2,2], digits=4))  $(round(R[2,3], digits=4))|")
println("|  $(round(R[3,1], digits=4))  $(round(R[3,2], digits=4))  $(round(R[3,3], digits=4))|")

# The rows of this matrix correspond to the geographic coordinates of the principal stress directions.

so3 = SOSAT.StressState(S1, S2, S3, pp_MPa,
                        alpha,
                        beta,
                        gamma)
S_geo3 = SOSAT.GeographicStress(so3)
println("Geographic Stress Tensor with beta=75 degrees (MPa):")
println("|  $(round(S_geo3[1,1], digits=2))  $(round(S_geo3[1,2], digits=2))  $(round(S_geo3[1,3], digits=2))|")
println("|  $(round(S_geo3[2,1], digits=2))  $(round(S_geo3[2,2], digits=2))  $(round(S_geo3[2,3], digits=2))|")
println("|  $(round(S_geo3[3,1], digits=2))  $(round(S_geo3[3,2], digits=2))  $(round(S_geo3[3,3], digits=2))|")
stress_states3 = Vector{SOSAT.StressState}(undef,1)
stress_states3[1] = so3
stereonet_fig3 = SOSAT.PrincipalStressStereonet(stress_states3)
stereonet_fig3.savefig("Euler_example3_principal_stress_stereonet.png")
# Here is the resulting plot
#md # ![Stereonet Plot](Euler_example3_principal_stress_stereonet.png)

# As expected, S1 deviates 15 degrees from vertical upwards toward 110 degrees azimuth. However, the projection of this to the lower hemisphere
# is plotted as 15 degrees from vertical downwards toward 290 degrees azimuth. S3 plunges 15 degrees from horizontal toward 110 degrees azimuth.
# S2 remains horizontal at 200 degrees azimuth.


plt.close("all")