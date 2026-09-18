using SOSAT
using Plots
using Plots.Measures  # For margin units like mm
using Printf

"""
Generate a figure that should reproduce Figure 2 from Holmes and Swift 1970, showing the temperature profile of the mud in the drill stem and annulus
"""

H = 15000.0  # depth
r = 0.5 * (8.0 + 3.0/8.0) / 12.0  # wellbore radius in ft
rp = 0.5 * (6.0 + 5.0/8.0) / 12.0  # drill pipe radius in ft
circulation_rate_bbl_hr = 300.0  # circulation rate in bbl/hr
cp = 0.4  # mud specific heat in BTU/lbm-°F
rho_mud = 10.0  # mud density in lbm/gallon
m = circulation_rate_bbl_hr * 42.0 * rho_mud  # mud mass flow rate in lbm/hr
hp = 30.0  # heat transfer coefficient across drill pipe wall in BTU/(ft²-°F-hr)
U = 1.0  # heat transfer coefficient between annulus and formation in BTU/(ft²-°F-hr)
G = 0.0127  # geothermal gradient in °F/ft
Ts = 59.5  # surface temperature in °F
Tpi = 75.0  # mud inlet temperature in °F
x = collect(range(0, stop=H, length=100))
Tp, Ta = pipe_and_annulus_temperature(x, H, m, cp, r, rp, hp, U, G, Ts, Tpi)


p = plot(x./1000, Tp, label="Drill Stem", color=:blue, linewidth=2,
         xlabel="Depth in Thousands of Feet", 
         ylabel="Temperature in degrees °F",
         grid=false)
plot!(p, x./1000, Ta, label="Annulus", color=:red, linewidth=2)
plot!(p, x./1000, G*x .+ Ts, label="Formation", color=:black, linewidth=2, linestyle=:dash)
savefig(p, "Holmes_and_Swift_thermal_profile.png")

# Convert to SI units for testing
println("\\nSI Unit Conversions:")
println("===================")

# Length conversions (ft to m)
H_si = H * 0.3048  # m
r_si = r * 0.3048  # m  
rp_si = rp * 0.3048  # m
println("H_si = ", H_si, " m (", H, " ft)")
println("r_si = ", r_si, " m (", r, " ft)")
println("rp_si = ", rp_si, " m (", rp, " ft)")

# Flow rate conversions (bbl/hr to m³/hr)
circulation_rate_si = circulation_rate_bbl_hr * 0.158987  # m³/hr
println("circulation_rate_si = ", circulation_rate_si, " m³/hr (", circulation_rate_bbl_hr, " bbl/hr)")

# Specific heat conversion (BTU/lbm-°F to J/kg-K) 
cp_si = cp * 4186.8  # J/kg-K
println("cp_si = ", cp_si, " J/kg-K (", cp, " BTU/lbm-°F)")

# Density conversion (lbm/gal to kg/m³)
rho_mud_si = rho_mud * 119.826  # kg/m³ 
println("rho_mud_si = ", rho_mud_si, " kg/m³ (", rho_mud, " lbm/gal)")

# Mass flow rate (lbm/hr to kg/s) - NOTE: Converting to kg/s for dimensional consistency
m_si = circulation_rate_si * rho_mud_si / 3600  # kg/s (divide by 3600 to convert hr to s)
println("m_si = ", m_si, " kg/s (", m, " lbm/hr)")

# Heat transfer coefficients (BTU/(ft²-°F-hr) to W/(m²-K))
hp_si = hp * 5.67826  # W/(m²-K)
U_si = U * 5.67826  # W/(m²-K)
println("hp_si = ", hp_si, " W/(m²-K) (", hp, " BTU/(ft²-°F-hr))")
println("U_si = ", U_si, " W/(m²-K) (", U, " BTU/(ft²-°F-hr))")

# Temperature conversions (°F to °C)
Ts_si = (Ts - 32) * 5/9  # °C
Tpi_si = (Tpi - 32) * 5/9  # °C
println("Ts_si = ", Ts_si, " °C (", Ts, " °F)")
println("Tpi_si = ", Tpi_si, " °C (", Tpi, " °F)")

# Geothermal gradient (°F/ft to °C/m)
G_si = G * (5/9) / 0.3048  # °C/m
println("G_si = ", G_si, " °C/m (", G, " °F/ft)")

# Calculate temperatures using SI units
x_si = collect(range(0, stop=H_si, length=100))  # meters
Tp_si, Ta_si = pipe_and_annulus_temperature(x_si, H_si, m_si, cp_si, r_si, rp_si, hp_si, U_si, G_si, Ts_si, Tpi_si)

# Temperatures are already in °C - no conversion needed for plotting
Tp_C = Tp_si  # already in °C
Ta_C = Ta_si  # already in °C
Formation_C = G_si.*x_si .+ Ts_si  # already in °C

# Create SI unit plot with meaningful Tufte-style tick marks and annotations
# Find meaningful temperatures and depths
bottom_hole_temp_formation = G_si * H_si + Ts_si  # Formation temp at bottom hole
bottom_hole_temp_pipe = Tp_C[end]  # Pipe temp at bottom hole  
bottom_hole_temp_annulus = Ta_C[end]  # Annulus temp at bottom hole

# Find peak temperatures and their locations
max_pipe_temp = maximum(Tp_C)
max_annulus_temp = maximum(Ta_C)
max_pipe_idx = argmax(Tp_C)
max_annulus_idx = argmax(Ta_C)
max_pipe_depth = x_si[max_pipe_idx] / 1000  # Convert to km
max_annulus_depth = x_si[max_annulus_idx] / 1000  # Convert to km

# Simplified meaningful tick marks with cleaner labels (meter-level precision)
meaningful_depths = [0, round(max_annulus_depth, digits=3), round(H_si/1000, digits=3)]  # Surface, peak, and bottom hole
meaningful_temps = [round(Ts_si), round(Tpi_si)]  # Rounded surface and inlet temps

p_si = plot(x_si./1000, Tp_C, label="Drill Stem", color=:blue, linewidth=2,
           xlabel="Depth in Kilometers", 
           ylabel="Temperature in degrees °C",
           grid=false,
           xticks=meaningful_depths,
           yticks=meaningful_temps,
           xrotation=30,  # Rotate x-axis labels by 30 degrees
           top_margin=15mm)  # Add top margin to prevent label cutoff
plot!(p_si, x_si./1000, Ta_C, label="Annulus", color=:red, linewidth=2)
plot!(p_si, x_si./1000, Formation_C, label="Formation", color=:black, linewidth=2, linestyle=:dash)

# Simple, clean annotations - just numbers with better positioning
annotate!(p_si, [(H_si/1000, bottom_hole_temp_pipe + 3, text("$(round(Int,bottom_hole_temp_pipe))", 10, :black))])  # Bottom hole
annotate!(p_si, [(H_si/1000, bottom_hole_temp_formation + 6, text("$(round(Int,bottom_hole_temp_formation))", 10, :black))])  # Formation at bottom
annotate!(p_si, [(max_annulus_depth, max_annulus_temp + 3, text("$(round(Int,max_annulus_temp))", 10, :red))])  # Peak
savefig(p_si, "Holmes_and_Swift_thermal_profile_SI.png")

println("\\nSI unit analysis complete! Check Holmes_and_Swift_thermal_profile_SI.png")