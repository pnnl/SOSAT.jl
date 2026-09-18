# # Borehole Stress Example
#
# This example shows how to use the Borehole Stress solutions in SOSAT.jl to calculate the stress
# state around an arbitrarily oriented borehole with arbitrarily oriented far-field stresses.
#
# The example here reproduces Figure A2 from Peska and Zoback (1995)
#
# The figure is of the stress distribution around a borehole with
#
# Deviation angle, phi = 30.0 degrees
# Azimuth angle, delta = 45.0 degrees

# The principal stresses are described as S1=S2=3.12*S3 and the 
# Euler angles are alpha=0 degrees, beta=65 degrees and gamma = 0 degrees.

# The figure plots the minimum and maximum tangential stresses, and the radial stress around the borehole wall as a function
# of the angle around the borehole with zero degrees being the bottom of the borehole. The figure also
# plots the angle, omega, between the direction of maximum tangential stress and the borehole axis.

using SOSAT
using Test
using Distributions
using PyPlot
const plt = PyPlot

plt.matplotlib.use("Agg");

S3 = 1.0;
S1 = 3.12*S3;
S2 = 3.12*S3;
Pp = 0.0;
alpha = deg2rad(0.0);
beta = deg2rad(65.0);
gamma = deg2rad(0.0);
phi = deg2rad(30.0);
delta = deg2rad(45.0);
ss = SOSAT.StressState(S1,S2,S3,Pp,alpha,beta,gamma);

ws = SOSAT.WellState(Pp,
                        phi,
                        delta);
theta = range(0, stop=pi, length=180);

theta_ret, sig_rr, sigt_max, sigt_min, omega = SOSAT.BoreholeStresses(ss, ws, theta; PR=0.25);

fig = plt.figure(figsize=(8,4));
ax = fig.add_subplot(111);
ax2 = ax.twinx();

ax.plot(rad2deg.(theta_ret), sigt_max./S1, color="black", label="Max Tangential Stress");
ax.plot(rad2deg.(theta_ret), sigt_min./S1, color="red",label="Min Tangential Stress");
ax.plot(rad2deg.(theta_ret), sig_rr./S1, color="orange", label="Radial Stress");
ax.set_xlabel("Angle around Borehole (degrees)");
ax.set_ylabel("Effective Stress/S1");
ax.legend(loc="upper right");

# Set x-axis limits and ticks to match paper figure
ax.set_xlim(-20, 200);
x_ticks = -20:10:200;  # Ticks every 10 degrees
x_labels = [0, 30, 60, 90, 120, 150, 180];  # Labels only at these values
ax.set_xticks(x_ticks);
ax.set_xticklabels([x in x_labels ? string(x) : "" for x in x_ticks]);

# Set y-axis limits and ticks to match paper figure
ax.set_ylim(-1.5, 4.0);
ax.set_yticks([0.0, 1.0, 2.0]);

ax2.plot(rad2deg.(theta_ret), rad2deg.(omega), color="green",label="Angle of Max Tangential Stress");
ax2.set_xlabel("Angle around Borehole (degrees)");
ax2.set_ylabel("Omega (degrees)");
ax2.legend(loc="upper left");
# Set y-axis limits and ticks for ax2 to match paper figure
ax2.set_ylim(-100, 200);
y2_ticks = [0, 50, 100, 150];
y2_labels = [0, 100];  # Labels only at 0 and 100
ax2.set_yticks(y2_ticks);
ax2.set_yticklabels([y in y2_labels ? string(y) : "" for y in y2_ticks]);

plt.tight_layout();
plt.savefig("BoreholeStress_PeskaZoback1995_FigA2.png", dpi=300);
#md  # ![Borehole Stress Distribution](BoreholeStress_PeskaZoback1995_FigA2.png)

# In this plot zero degrees is at the bottom of the borehole and 180 degrees is at the top of the borehole.

# We can use functions to find where the minimum and maximum values occur and what the orientations are:
minimum_tangential_stress = minimum(sigt_min);
println("Minimum Tangential Stress / S1 = ", minimum_tangential_stress/S1, "\n");

location_of_minimum = theta_ret[findmin(sigt_min)[2]];
println("Location of Minimum Tangential Stress (degrees) = ", rad2deg(location_of_minimum), "\n")

maximum_tangential_stress = maximum(sigt_max);
println("Maximum Tangential Stress / S1 = ", maximum_tangential_stress/S1, "\n");

location_of_maximum = theta_ret[findmax(sigt_max)[2]];
println("Location of Maximum Tangential Stress (degrees) = ", rad2deg(location_of_maximum), "\n")

omega_max = maximum(omega);
println("Maximum Omega (degrees) = ", rad2deg(omega_max), "\n");

omega_min = minimum(omega);
println("Minimum Omega (degrees) = ", rad2deg(omega_min), "\n");

