using SOSAT
using Test
using Random
using Distributions
using PythonPlot
const plt = PythonPlot

Random.seed!(1234)

#include("test_fracture_opening_w_orientation.jl")

# we need to update this test to use the mud temperature
# distribution for the breakout constaint

@testset "SOSAT.jl Tests" begin
    include("Aqua.jl")
    include("test_Aphi_constraint.jl")
    include("test_DITF_fixed.jl")
    include("test_DITF_general.jl")
    include("test_DITF_vertical_horizontal.jl")
    include("test_no_DITF.jl")
    include("test_Breakout_GOA_example.jl")
    include("test_Breakout_fixed.jl")
    include("test_fault_constraint.jl")
    include("test_Euler_rotation.jl")
    include("test_EulerPrior.jl")
    include("test_MiniFracClosure.jl")
end