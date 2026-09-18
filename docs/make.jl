using Documenter
using Literate
using SOSAT

# Generate examples from Literate.jl sources
example_dir = joinpath(@__DIR__, "src", "examples")
output_dir = joinpath(@__DIR__, "src", "generated")

# Create output directory if it doesn't exist
mkpath(output_dir)

# Convert Literate.jl files to markdown
for file in readdir(example_dir)
    if endswith(file, ".jl")
        Literate.markdown(
            joinpath(example_dir, file),
            output_dir;
            documenter = true,  # This enables Documenter.jl integration
            execute = true,       # Execute code blocks during build
            codefence = "```julia" => "```"
        )
    end
end

# Build documentation
makedocs(
    sitename = "SOSAT.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        size_threshold = nothing,
        size_threshold_warn = nothing,
        repolink = "https://stash.pnnl.gov/scm/~burg137/sosat.jl.git"
    ),
    modules = [SOSAT],
    pages = [
        "Home" => "index.md",
        "Examples" => [
            "Euler Rotations" => "generated/euler_rotations.md",
            "Borehole Stress Example" => "generated/BoreholeStress_example.md",
            "Fault Constraints" => "generated/fault_constraint.md",
            "DITF Constraint Examples" => "generated/DITF_examples.md",
            "Breakout Constraint Example" => "generated/Breakout_examples.md"
        ],
        "API Reference" => "api.md"
    ],
    clean = true,
    checkdocs = :none,
    repo = "https://stash.pnnl.gov/scm/~burg137/sosat.jl.git"
)
