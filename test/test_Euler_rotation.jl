using SOSAT
using Test
using Random
using Rotations

Random.seed!(1234)
# do a simple stress tensor rotation using the ZYX Tait-Bryan angles
# we will simply set S1=3.0, S2=2.0, S3=10 and apply:

# alpha=90, which should result in 
# the X' (S1 direction) becoming Y direction/East
# and Y' (S2 direction) becoming the negative X direction/South
#
# then apply beta=90, which should result in:
# x1 = X'' (S1 direction) becoming Z direction/Up
# x2 = Y'' (S2 direction) becoming -Y direction/South
# x3 = Z'' (S3 direction) becoming X direction/East

# gamma is zero, but it would rotation about the verticl axis (Z'') such that it would
# rotate the X'' and Y'' in the counter-clockwise direction when looking down the Z axis

# so the stress tensor should start as
# | 3 0 0 |
# | 0 2 0 |
# | 0 0 1 |
#
# and end as
# | 2 0 0 |
# | 0 1 0 |
# | 0 0 3 |
S1 = 3.0
S2 = 2.0
S3 = 1.0
pp = 0.0
alpha = deg2rad(90.0)
beta = deg2rad(90.0)
gamma = deg2rad(0.0)

S_geographic = SOSAT.BoreholeStress.PrincipalToGeographic(S1, S2, S3,
                                                            alpha,
                                                            beta,
                                                            gamma)
println("S_geographic = ", S_geographic)
@test isapprox(S_geographic[1,1], 2.0; atol=1e-6)
@test isapprox(S_geographic[2,2], 1.0; atol=1e-6)
@test isapprox(S_geographic[3,3], 3.0; atol=1e-6)