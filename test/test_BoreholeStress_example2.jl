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

water_depth = 4000.0
water_density = 1025.0 # kg/m3
basalt_depth = 1000.0
basalt_density = 2800.0 # kg/m3

g = 9.81 # m/s2
sigv = (water_density * g * water_depth +
        basalt_density * g * basalt_depth) / 1.0e6 # MPa
println("Vertical Stress is ",sigv)

pp_MPa = (water_depth + basalt_depth) * 10.0 / 1000.0 # assume 10 MPa/km
println("Pore pressure is ",pp_MPa)
S1 = 120.0
S2 = sigv
S3 = sigv

alpha = deg2rad(0.0)
beta = deg2rad(0.0)
gamma = deg2rad(0.0)
phi = deg2rad(0.0)
delta = deg2rad(0.0)

ss = SOSAT.StressState(S1,S2,S3,pp_MPa,alpha,beta,gamma)

S_geo = SOSAT.GeographicStress(ss)
println("Geographic Stress Tensor (MPa):")
println("|  $(round(S_geo[1,1], digits=2))  $(round(S_geo[1,2], digits=2))  $(round(S_geo[1,3], digits=2))|")
println("|  $(round(S_geo[2,1], digits=2))  $(round(S_geo[2,2], digits=2))  $(round(S_geo[2,3], digits=2))|")
println("|  $(round(S_geo[3,1], digits=2))  $(round(S_geo[3,2], digits=2))  $(round(S_geo[3,3], digits=2))|")

ws = SOSAT.WellState(pp_MPa+5.0,
                        phi,
                        delta)
theta = range(0, stop=pi, length=180)

theta_ret, sig_rr, sigt_max, sigt_min, omega = SOSAT.BoreholeStresses(ss, ws, theta; PR=0.25)

# use the MinimumPrincipalStress function to find the location of the least compressive
# tangential stress
smin, theta_min, omega_min = SOSAT.MinimumPrincipalStress(ss, ws; PR=0.25)
println("Minimum Tangential Stress (MPa) = ", smin)
println("Location of Minimum Tangential Stress (degrees) = ", rad2deg(theta_min))
println("Omega (degrees) = ", rad2deg(omega_min))
if GENERATE_PLOTS
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    ax.plot(rad2deg.(theta_ret), sigt_max, color="black", label="Max Tangential Stress")
    ax.plot(rad2deg.(theta_ret), sigt_min, color="red",label="Min Tangential Stress")
    ax.plot(rad2deg.(theta_ret), sig_rr, color="orange", label="Radial Stress")
    ax.set_xlabel("Angle around Borehole (degrees)")
    ax.set_ylabel("Effective Stress (MPa)")
    ax.legend(loc="upper right")

    
    ax2.plot(rad2deg.(theta_ret), rad2deg.(omega), color="green",label="Angle of Max Tangential Stress")
    ax2.set_xlabel("Angle around Borehole (degrees)")
    ax2.set_ylabel("Omega (degrees)")
    ax2.set_title("Angle of Maximum Tangential Stress")
    ax2.legend(loc="upper left")
    
    
    plt.tight_layout()
    plt.savefig("BoreholeStress_Example2.png", dpi=300)
end