#!/usr/bin/env julia

# Build custom Julia system image with precompiled Test module
# This dramatically speeds up Julia execution (10-20x faster startup)
#
# Usage:
#   julia scripts/build_julia_sysimage.jl
#
# This creates: ~/.julia/sysimages/julia_with_test.so
# Use with: julia --sysimage ~/.julia/sysimages/julia_with_test.so

using Pkg

# Install PackageCompiler if not already installed
if !haskey(Pkg.project().dependencies, "PackageCompiler")
    println("Installing PackageCompiler...")
    Pkg.add("PackageCompiler")
end

using PackageCompiler

# Create directory for custom sysimage
sysimage_dir = joinpath(homedir(), ".julia", "sysimages")
mkpath(sysimage_dir)

sysimage_path = joinpath(sysimage_dir, "julia_with_test.so")

println("=" ^ 80)
println("Building custom Julia sysimage with precompiled Test module")
println("This will take 2-5 minutes but makes future runs 10-20x faster!")
println("=" ^ 80)

# Create precompile script that uses Test module
precompile_script = """
using Test

# Precompile common test patterns
@test 1 + 1 == 2
@test_throws DivideError 1 ÷ 0

# Precompile common functions
function example_add(a, b)
    return a + b
end

@test example_add(2, 3) == 5

println("Precompile script completed")
"""

precompile_file = joinpath(sysimage_dir, "precompile_test.jl")
write(precompile_file, precompile_script)

# Build custom sysimage with Test module precompiled
try
    create_sysimage(
        [:Test],  # Packages to precompile
        sysimage_path=sysimage_path,
        precompile_execution_file=precompile_file,
        cpu_target="generic"  # Works on all CPUs
    )

    println("=" ^ 80)
    println("✅ Custom sysimage built successfully!")
    println("Location: $sysimage_path")
    println()
    println("To use this sysimage:")
    println("  julia --sysimage $sysimage_path your_script.jl")
    println()
    println("Expected speedup: 10-20x faster startup for code using Test module")
    println("=" ^ 80)

catch e
    println("=" ^ 80)
    println("❌ Error building sysimage:")
    println(e)
    println("=" ^ 80)
    exit(1)
end

# Clean up precompile file
rm(precompile_file, force=true)
