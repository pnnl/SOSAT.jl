"""
Borehole thermal model from Holmes and Swift (1970)

This model calculates mud temperatures in drill pipe and annulus during drilling operations.
The function is unit-agnostic and works with either imperial/oilfield units or SI units,
provided ALL inputs use a consistent unit system.

# Input Parameters

## Imperial/Oilfield Units (as used in Holmes & Swift 1970):
- `H`: depth of borehole (ft)
- `r`: radius of borehole (ft) 
- `k`: formation thermal conductivity (BTU/hr-ft-°F) [not used in this function]
- `c`: formation specific heat (BTU/lbm-°F) [not used in this function]
- `cp`: mud specific heat (BTU/lbm-°F)
- `ρ`: formation density (lbm/ft³) [not used in this function]
- `Ts`: surface temperature (°F)
- `G`: geothermal gradient (°F/ft)
- `rp`: drill pipe radius (ft)
- `hp`: heat transfer coefficient across drill pipe wall (BTU/(ft²-°F-hr))
- `m`: mud mass flow rate (lbm/hr)
- `U`: heat transfer coefficient between annulus and formation (BTU/(ft²-°F-hr))
- `Tpi`: mud inlet temperature (°F)

## SI Units (for international use):
- `H`: depth of borehole (m)
- `r`: radius of borehole (m)
- `cp`: mud specific heat (J/(kg⋅K))
- `Ts`: surface temperature (°C)
- `G`: geothermal gradient (°C/m) 
- `rp`: drill pipe radius (m)
- `hp`: heat transfer coefficient across drill pipe wall (W/(m²⋅K))
- `m`: mud mass flow rate (kg/s) ⚠️ NOTE: Must be per SECOND, not per hour!
- `U`: heat transfer coefficient between annulus and formation (W/(m²⋅K))
- `Tpi`: mud inlet temperature (°C)

# Returns
- `Tp`: temperature of mud in the drill stem (same units as input temperatures)
- `Ta`: temperature of mud in the annulus (same units as input temperatures)

# Important Notes
1. **Dimensional Consistency**: All parameters must use a consistent unit system
2. **Time Units**: For SI units, mass flow rate MUST be in kg/s (not kg/hr) to maintain 
   dimensional consistency with W/(m²⋅K) heat transfer coefficients
3. **Temperature Units**: Use °C for SI units, °F for imperial units. The model works with
   temperature differences, so °C and K are equivalent (no need for absolute temperatures)
4. **Reference**: Holmes, C.S. and Swift, S.C. (1970). "Calculation of circulating mud 
   temperatures." Journal of Petroleum Technology, 22(6), 670-674.

# Example Usage
```julia
# Imperial units (Holmes & Swift example)
x = collect(range(0, 15000, 100))  # depth points (ft)
Tp, Ta = pipe_and_annulus_temperature(x, 15000.0, 126000.0, 0.4, 0.349, 0.276, 30.0, 1.0, 0.0127, 59.5, 75.0)

# SI units (°C)
x_si = collect(range(0, 4572, 100))  # depth points (m)
Tp_si, Ta_si = pipe_and_annulus_temperature(x_si, 4572.0, 15.88, 1675.0, 0.106, 0.084, 170.3, 5.68, 0.0231, 15.3, 23.9)
```
"""

function pipe_and_annulus_temperature(x, H, m, cp, r, rp, hp, U, G, Ts, Tpi)
    A = (m.*cp)./(2.0 .* π .* rp .* hp)
    B = (r.*U)./(rp.*hp)
    C1 = (B./(2.0 .* A)).*(1.0 .+ sqrt.(1.0 .+ (4.0 ./B)))
    C2 = (B./(2.0 .* A)).*(1.0 .- sqrt.(1.0 .+ (4.0 ./B)))
    C3 = 1.0 .+ (B./2.0).*(1.0 .+ sqrt.(1.0 .+ (4.0 ./B)))
    C4 = 1.0 .+ (B./2.0).*(1.0 .- sqrt.(1.0 .+ (4.0 ./B)))

    # integration constants on pg 672
    K2 = (G*A .- (Tpi .- Ts .+ G*A).*exp.(C1.*H).*(1.0 .- C3))./(((exp.(C2.*H)).*(1.0 .- C4))-((exp.(C1.*H)).*(1.0 .- C3)))
    K1 = Tpi .- K2 .- Ts .+ G*A 

    # temperature of mud in the drill stem
    Tp = K1.*(exp.(C1.*x)) .+ K2.*(exp.(C2.*x)) .+ G.*x .+ Ts .- G*A
    # temperature of mud in the annulus
    Ta = K1.*C3.*(exp.(C1.*x)) .+ K2.*C4.*(exp.(C2.*x)) .+ G.*x .+ Ts

    return Tp, Ta
end
    
function Tp_fn(x, K1, K2, C1, C2, G, A, Ts)
    # temperature of mud in the drill stem
    Tp = K1.*(exp.(C1.*x)) .+ K2.*(exp.(C2.*x)) .+ G.*x .+ Ts .- G*A
    return Tp
end

function Ta_fn(x, K1, K2, C1, C2, C3, C4, G, Ts)
    # temperature of mud in the annulus
    Ta = K1.*C3.*(exp.(C1.*x)) .+ K2.*C4.*(exp.(C2.*x)) .+ G.*x .+ Ts 
    return Ta
end

