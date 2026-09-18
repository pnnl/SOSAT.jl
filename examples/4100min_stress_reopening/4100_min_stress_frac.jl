using SOSAT
using Distributions
import PyPlot
const plt = PyPlot

#----------------------------
# Fault Friction distribution
#----------------------------
mudist = Distributions.truncated(Distributions.LogNormal(-0.53,0.1),0.1,0.9)
fc = SOSAT.FaultConstraint(mu_dist=mudist,mu=0.8)

mu_array = LinRange(0.0,1.0,100)
pmu = Distributions.pdf.(mudist, mu_array)
mu_fig = plt.figure()
mu_ax = mu_fig.add_subplot(111)
jnk = mu_ax.plot(mu_array, pmu, color="k")
mu_ax.set_xlabel("Fault Friction Coefficient")
mu_ax.set_ylabel("Probability Density")
mu_fig.savefig("Fault_Friction_PDF.png")

#---------------------------------
# Pore pressure prior distribution
#---------------------------------
Ppdist = Distributions.Uniform(0.0,13.0)
ppc = SOSAT.PorePressureConstraint(Ppdist)

Pp_array = LinRange(-0.01,15.0,100)
pPp = Distributions.pdf.(Ppdist, Pp_array)
Pp_fig = plt.figure()
Pp_ax = Pp_fig.add_subplot(111)
jnk = Pp_ax.plot(Pp_array, pPp, color="k")
Pp_ax.set_xlabel("Pore Pressure (MPa)")
Pp_ax.set_ylabel("Probability Density")
Pp_fig.savefig("Pore_Pressure_Prior_PDF.png")

#---------------------------------------
# minifrac closure pressure distribution
#---------------------------------------
mfc_dist = Distributions.Uniform(19.0,22)
mfc = SOSAT.MiniFracClosure(mfc_dist)

mfc_array = LinRange(15.0,25.0,100)
pmfc = Distributions.pdf.(mfc_dist, mfc_array)
mfc_fig = plt.figure()
mfc_ax = mfc_fig.add_subplot(111)
jnk = mfc_ax.plot(mfc_array, pmfc, color="k")
mfc_ax.set_xlabel("Fracture Closure Pressure (MPa)")
mfc_ax.set_ylabel("Probability Density")
mfc_fig.savefig("Fracture_Closure_Pressure_PDF.png")

#---------------------------------------
# fracture opening pressure distribution
#---------------------------------------
frac_dip_dist = Distributions.VonMises(deg2rad(65.0),85.0)
frac_dip_trend_dist = Distributions.VonMises(deg2rad(200.0),45.0)

frac_dip_array = LinRange(0.0,100.0,100)
pfrac_dip = Distributions.pdf.(frac_dip_dist, deg2rad.(frac_dip_array))
frac_dip_fig = plt.figure()
frac_dip_ax = frac_dip_fig.add_subplot(111)
jnk = frac_dip_ax.plot(frac_dip_array, pfrac_dip, color="k")
frac_dip_ax.set_xlabel("Induced Fracture Dip (deg)")
frac_dip_ax.set_ylabel("Probability Density")
frac_dip_fig.savefig("Fracture_Dip_PDF.png")

frac_dip_trend_array = LinRange(150.0,250.0,100)
pfrac_dip_trend = Distributions.pdf.(frac_dip_trend_dist, 
                                    deg2rad.(frac_dip_trend_array))
frac_dip_trend_fig = plt.figure()
frac_dip_trend_ax = frac_dip_trend_fig.add_subplot(111)
jnk = frac_dip_trend_ax.plot(frac_dip_trend_array,
                                pfrac_dip_trend, color="k")
frac_dip_trend_ax.set_xlabel("Induced Fracture Dip Trend (deg)")
frac_dip_trend_ax.set_ylabel("Probability Density")
frac_dip_trend_fig.savefig("Fracture_Dip_Trend_PDF.png")

op_dist = Distributions.Uniform(18,23.8)
well_azimuth = deg2rad(0.0)
well_inclination = deg2rad(0.0)

"""
This constraint just looks for opening of a fracture of a specified
orientation and does not look at stresses at other orientations
"""
op_w_orient = SOSAT.FractureOpeningPressure_wOrientation(op_dist,
                                                         well_azimuth,
                                                         well_inclination,
                                                         frac_dip_trend_dist,
                                                         frac_dip_dist)

"""
This one looks for a tensile stress to be developed at any location
"""
op  = SOSAT.FractureOpeningPressure(op_dist,
                                    well_azimuth,
                                    well_inclination)

"""
This one looks for the orientation of the plane where
a tensile stress develops at the minimum pressure
"""
ifc = SOSAT.InducedFractureOrientation(0.0, 0.0,
                                       frac_dip_trend_dist,
                                       frac_dip_dist)

op_array = LinRange(15.0,25.0,100)
pop = Distributions.pdf.(op_dist, op_array)
op_fig = plt.figure()
op_ax = op_fig.add_subplot(111)
jnk = op_ax.plot(op_array, pop, color="k")
op_ax.set_xlabel("Fracture Opening Pressure (MPa)")
op_ax.set_ylabel("Probability Density")
op_fig.savefig("Fracture_Opening_Pressure_PDF.png")

op_array = LinRange(15.0,25.0,100)
pop = Distributions.pdf.(op_dist, op_array)
op_fig = plt.figure()
op_ax = op_fig.add_subplot(111)
jnk = op_ax.plot(op_array, pop, color="k")
op_ax.set_xlabel("Fracture Opening Pressure (MPa)")
op_ax.set_ylabel("Probability Density")
op_fig.savefig("Fracture_Opening_Pressure_PDF.png")


UCS_dist = Distributions.Uniform(115.0,180.0)
bo_param = SOSAT.BreakoutParameters(UCS_dist=UCS_dist,
                                mud_pressure_dist=Distributions.Uniform(0.0,0.1),
                                rock_friction_dist=Distributions.Uniform(0.2,0.21),
                                well_azimuth=0.0,
                                well_inclination=0.0,
                                formation_temperature=30.0,
                                mud_temperature=30.0,
                                breakout_azimuth=deg2rad(0.0),
                                YM=1.0e9,
                                PR=0.25,
                                CTE=5.0e-6)

UCS_array = LinRange(100.0,200.0,100)
pUCS = Distributions.pdf.(UCS_dist, UCS_array)
UCS_fig = plt.figure()
UCS_ax = UCS_fig.add_subplot(111)
jnk = UCS_ax.plot(UCS_array, pUCS, color="k")
UCS_ax.set_xlabel("Unconfined Compressive Strength (MPa)")
UCS_ax.set_ylabel("Probability Density")
UCS_fig.savefig("UCS_PDF.png")

bo_ind = SOSAT.Breakout(false,
                    bo_param,
                    5.0,  # UCS proposal distribution
                    0.0, # mud pressure proposal distribution
                    0.0) # rock friction proposal distribution

# use a prior distribution that perfers a vertical direction
# for the vertical stress
beta_prior_dist = Distributions.VonMises(deg2rad(90.0),45.0)
beta_prior = SOSAT.BetaPriorConstraint(beta_prior_dist)

litho_orient_array = LinRange(deg2rad(-90.0),deg2rad(90.0),100)
beta_array = deg2rad(90.0) .- litho_orient_array
plitho_orient = Distributions.pdf(beta_prior_dist, beta_array)
litho_orient_prior_fig = plt.figure()
litho_orient_prior_ax = litho_orient_prior_fig.add_subplot(111)
jnk = litho_orient_prior_ax.plot(rad2deg.(litho_orient_array),
                                 plitho_orient, color="k")
litho_orient_prior_ax.set_xlabel("Lithostatic Stress Orientation (deg)")
litho_orient_prior_ax.set_ylabel("Probability Density")
litho_orient_prior_fig.savefig("Lithostatic_Orientation_Prior_PDF.png")

# theta and omega are in borehole coordinate system

# the subhorizontal stress that may be either S1 or S2 but is
# oriented North prior to Euler rotations
SHmax = 35.0
#SEast = 45.0
# the subvertical stress that may be either S1 or S2
# but is oriented East prior to Euler rotations
Sv = 35.0

# the subhorizontal stress that is S3 but is initially vertical
# prior to the Euler rotations
SHmin = 21.0
Pp = 7.0
alpha = deg2rad(20.0)
beta = deg2rad(90.0)
gamma = 0.0
so = SOSAT.StressState(Sv, SHmax, SHmin, Pp,
                        alpha,
                        beta,
                        gamma)
b = SOSAT.BoreholeStress.WellState(0.0, 0.0, 0.0)

indicators = Vector{SOSAT.StressIndicator}(undef,7)

indicators[1] = fc
indicators[2] = ppc
indicators[3] = mfc
#indicators[4] = op_w_orient
indicators[4] = op
indicators[5] = ifc
indicators[6] = bo_ind
indicators[7] = beta_prior
# perturb SHmax and SHmin but not Sv (which is S2)
stress_std = [0.0,1.0,1.0,1.0]
angle_std = [deg2rad(2.0),deg2rad(2.0),deg2rad(0.0)]
#Nsample = 50000
#Nburn = 1000
Nsample = 100
Nburn = 1
Nthin = 2

samples = SOSAT.MCMCSample(so,
                        indicators,
                        stress_std,
                        angle_std,
                        Nsample;
                        Nburn=Nburn,
                        Nthin=Nthin,
                        FixedOrder=false)

S1fig = SOSAT.StressComponentHist(1, samples, label="Subvertical Principal Stress (MPa)")
S1fig.savefig("4100SV_marginal_posterior.png")

S2fig = SOSAT.StressComponentHist(2, samples, label="Subhorizontal Maximum Principal Stress (MPa)")
S2fig.savefig("4100Shmax_marginal_posterior.png")

S3fig = SOSAT.StressComponentHist(3, samples, label="Subhorizontal Minimum Principal Stress (MPa)")
S3fig.savefig("4100Shmin_marginal_posterior.png")

ppfig = SOSAT.PorePressureHist(samples, label="Pore Pressure (MPa)")
ppfig.savefig("4100_Pp_marginal_posterior.png")

alpha_fig = SOSAT.AlphaHist(samples, label="Azimuth of Subhorizontal Minimum Principal Stress (deg)")
alpha_fig.savefig("4100_SHmin_azimuth_marginal_posterior.png")

beta_samples = SOSAT.GetBetaSamples(samples)
beta_fig = plt.figure()
ax_beta = beta_fig.add_subplot(111)
jnk = ax_beta.hist(90.0 .- rad2deg.(beta_samples), density=true, color="k")
ax_beta.set_xlabel("Deviation of Subvertical Stress (deg)")
ax_beta.set_ylabel("Probability Density")
#beta_fig = SOSAT.BetaHist(samples, la)
beta_fig.savefig("4100_Sv_deviation_marginal_posterior.png")


gamma_fig = SOSAT.GammaHist(samples)
gamma_fig.savefig("4100_gamma_hist.png")

SOSAT.WriteTraceToH5(samples, "4100_Stress_Samples.h5")
