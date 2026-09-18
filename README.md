# Installing SOSAT

SOSAT.jl is a Julia package designed to perform Markov Chain Monte Carlo uncertainty analyses of subsurface stress data. It applies the Metropolis-Hastings algorithm. This package required a valid
Python environment with matplotlib installed. You must set an environmental variable pointing to that
python interpreter:

JULIA_PYTHONCALL_EXE=<path to Python interpreter>

To use this package run Julia in the terminal, then:

    julia --project=. -e 'using Pkg; Pkg.instantiate()'

This will download and compile all dependencies. Then, you can run tests with

    julia --project=. -e 'using Pkg; Pkg.test()'

## Building Documentation

From the repository root:

    julia --project=docs -e 'using Pkg; Pkg.instantiate()'
    julia --project=docs docs/make.jl

The first command only needs to be run once (or after `docs/Project.toml`
changes); it installs Documenter and other doc-building dependencies.
The second command builds the documentation into `docs/build/`. The documentation is a self-contained set of HTML pages that you can load in your web browser.

## Getting Started

The documentation contains complete functioning examples for a variety of constraints on the state of stress. You should be able to use them with modification of the input values for your use case. If you have feedback or questions please reach out at sosat@pnnl.gov