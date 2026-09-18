using SOSAT
using Test
using Random
using Distributions
using PythonPlot
const plt = PythonPlot

Random.seed!(1234)
plt.matplotlib.use("Agg")

const GENERATE_PLOTS = haskey(ENV, "JULIA_TEST_PLOTS") || 
                       get(ENV, "JULIA_TEST_PLOTS", "false") == "true"

"""
This test is meant to reproduce Figure A2 from Peska and Zoback (1995)

The figure is of the stress distribution around a borehole with

Deviation angle, phi = 0.0 degrees
Azimuth angle, delta = 0.0 degrees

The principal stresses are described as S1=S2=3.12*S3 and the 
Euler angles are alpha=0 degrees, beta=90 degrees and gamma = 4.5 degrees.

The figure plots the minimum and maximum tangential stresses, and the radial stress around the borehole wall as a function
of the angle around the borehole with zero degrees being the bottom of the borehole. The figure also
plots the angle, omega, between the direction of maximum tangential stress and the borehole axis.
"""


S1 = 58.06
S2 = S1 * 1.0
S3 = 0.7 * S1
Pp = 22.4
alpha = deg2rad(0.0)
beta = deg2rad(90.0)
gamma = deg2rad(90.0) - deg2rad(4.5)
phi = deg2rad(0.0)
delta = 3.77
println("gamma (deg)= ", rad2deg(gamma))
println("gamma (rad)= ", gamma)

ss = SOSAT.StressState(S1,S2,S3,Pp,alpha,beta,gamma)

ws = SOSAT.WellState(Pp,
                        phi,
                        delta)
theta = range(0, stop=pi, length=180)

theta_ret, sig_rr, sigt_max, sigt_min, omega = SOSAT.BoreholeStresses(ss, ws, theta; PR=0.25)


minimum_tangential_stress = minimum(sigt_min)


location_of_minimum = theta_ret[findmin(sigt_min)[2]]
print("Location of Minimum Tangential Stress (degrees) = ", rad2deg(location_of_minimum), "\n")


maximum_tangential_stress = maximum(sigt_max)
print("Maximum Tangential Stress / S1 = ", maximum_tangential_stress/S1, "\n")

location_of_maximum = theta_ret[findmax(sigt_max)[2]]
print("Location of Maximum Tangential Stress (degrees) = ", rad2deg(location_of_maximum), "\n")
function choose_best_angle(angle, distribution)
    angle1 = mod(angle + π, 2π) - π  # Map to [-π, π]
    angle2 = mod(angle, 2π) - π      # Map equivalent angle to [-π, π]
    
    return Distributions.pdf(distribution, angle1) >= Distributions.pdf(distribution, angle2) ? angle1 : angle2
end
theta_dist_mean = deg2rad(4.8)
theta_dist_err = deg2rad(5.0)
println("line 73 theta_dist_mean = ", theta_dist_mean)
println("line 74 theta_dist_err = ", theta_dist_err)
theta_distribution = Distributions.VonMises(theta_dist_mean,1.0/(theta_dist_err^2))
omega_distribution = Distributions.VonMises(deg2rad(0.0),1.0/(deg2rad(5.0)^2))

sigt_min, tmin, omega = SOSAT.MinimumPrincipalStress(ss, ws, PR=0.25)
best_tmin = choose_best_angle(tmin, theta_distribution)
best_omega = choose_best_angle(omega, omega_distribution)
println("ss= ", ss)
println("ws= ", ws)

println("line 82 tmin= ", tmin)
println("line 83 tmin (deg)= ", rad2deg(tmin))
println("line 84 best_tmin (deg) = ", rad2deg(best_tmin))
println("line 85 best_omega (deg) = ", rad2deg(best_omega))

omega_max = maximum(omega)
print("Maximum Omega (degrees) = ", rad2deg(omega_max), "\n")


omega_min = minimum(omega)
print("Minimum Omega (degrees) = ", rad2deg(omega_min), "\n")


if GENERATE_PLOTS
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    ax.plot(rad2deg.(theta_ret), sigt_max./S1, color="black", label="Max Tangential Stress")
    ax.plot(rad2deg.(theta_ret), sigt_min./S1, color="red",label="Min Tangential Stress")
    ax.plot(rad2deg.(theta_ret), sig_rr./S1, color="orange", label="Radial Stress")
    ax.set_xlabel("Angle around Borehole (degrees)")
    ax.set_ylabel("Effective Stress/S1")
    ax.legend(loc="upper right")

    # Set x-axis limits and ticks to match paper figure
    ax.set_xlim(-20, 200)
    x_ticks = -20:10:200  # Ticks every 10 degrees
    x_labels = [0, 30, 60, 90, 120, 150, 180]  # Labels only at these values
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([x in x_labels ? string(x) : "" for x in x_ticks])
    
    # Set y-axis limits and ticks to match paper figure
    ax.set_ylim(-1.5, 4.0)
    ax.set_yticks([0.0, 1.0, 2.0])

    ax2.plot(rad2deg.(theta_ret), rad2deg.(omega), color="green",label="Angle of Max Tangential Stress")
    ax2.set_xlabel("Angle around Borehole (degrees)")
    ax2.set_ylabel("Omega (degrees)")
    ax2.legend(loc="upper left")
    
    # Set y-axis limits and ticks for ax2 to match paper figure
    ax2.set_ylim(-100, 200)
    y2_ticks = [0, 50, 100, 150]
    y2_labels = [0, 100]  # Labels only at 0 and 100
    ax2.set_yticks(y2_ticks)
    ax2.set_yticklabels([y in y2_labels ? string(y) : "" for y in y2_ticks])

    plt.tight_layout()
    plt.savefig("BoreholeStress_408_test.png", dpi=300)
end