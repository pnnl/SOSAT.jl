module SOSAT

# Write your package code here.

using Distributions
import PythonPlot
const plt = PythonPlot


# source code

include("./BoreholeStress.jl")
include("./StressIndicatorBase.jl")
include("./PorePressureIndicator.jl")
include("./FaultConstraint.jl")
include("./MiniFracConstraint.jl")
include("./BreakoutConstraint.jl")
include("./DITFConstraint.jl")
include("./StressComponentConstraint.jl")
include("OrientationPriorsConstraints.jl")
include("AphiConstraint.jl")
include("./MCMCSampler.jl")
include("./StressUtils.jl")
include("./BoreholeThermalModel.jl")

# import what we want to export
using .BoreholeStress: StressState, WellState, GeographicStress, EulerRotation, PrincipalStressOrientations, PrincipalStressDirections, BoreholeStresses, MinimumPrincipalStress, MaximumPrincipalStress
using .BoreholeStress: NormalAndShear


# exports
# Core types and functions
export StressState, WellState, GeographicStress, StressIndicator, EulerRotation, PrincipalStressOrientations, PrincipalStressDirections
export PerturbIndicatorState, StressTotalLikelihood, StressTotalLogLikelihood
export BoreholeStresses, MinimumPrincipalStress, MaximumPrincipalStress
# Constraint types  
export PorePressureConstraint, FaultConstraint
export Breakout, DITF, StressComponentConstraint, StressComponentRatioConstraint
export MiniFracClosure, FractureOpeningPressure_wOrientation, InducedFractureOrientation
export EulerAnglePriorConstraint, VerticalPrincipalStressConstraint
export SHmaxAzimuthConstraint
# MCMC functionality
export MCMCSampler, LoadStressStateFromFile

# Analysis and plotting
export StressComponentHist
export StressPolygonHist, PrincipalStressStereonet
export GetStressComponentStatistics, GetStressComponentCorrelation

# Sample extraction
export GetS1Samples, GetS2Samples, GetS3Samples, GetPorePressureSamples
export GetAlphaSamples, GetBetaSamples, GetGammaSamples, GetStressMatrix
export GeographicallyOrderStresses
export GetSHmaxAzimuth
export plot_stress_polygon, DITF_SHmax, DITF_SHmin, faultS3Limit, faultS1Limit, getDITFStressArrays
export NF_shmin_limit, SS_shmax_limit, TF_shmax_limit

export pipe_and_annulus_temperature
end
