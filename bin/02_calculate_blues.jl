#!/usr/bin/env julia
"""
Calculate BLUE (Best Linear Unbiased Estimator) values for each environment.
Fits mixed models using environment-specific formulas and extracts BLUE values.
"""

using MixedModels, DataFrames, CSV, JSON, CategoricalArrays, StatsModels

println("Current directory: ", pwd())
println(ARGS[2])

# Get command line arguments
input_pheno_path = ARGS[1]
input_formulas_path = ARGS[2]
mapping_json_path = ARGS[3]
output_blues_path = ARGS[4]

# Read data
println("\nReading data...")
phenotype_data = CSV.read(input_pheno_path, DataFrame)
formulas_df = CSV.read(input_formulas_path, DataFrame)

# Check if data uses standard column names (from 00_filter_phenotype_data.py)
# Standard names: trait_value, env, geno, trait_name
standard_cols = ["trait_value", "env", "geno", "trait_name"]
has_standard = all(col in names(phenotype_data) for col in standard_cols)

if has_standard
    println("Using standard column names (data was standardized by 00_filter_phenotype_data.py)")
    env_col_name = "env"
    trait_col_name = "trait_value"
    trait_value_col_name = "trait_value"
    trait_name_col_name = "trait_name"
    geno_col_name = "geno"
    family_col_name = "family"
else
    # Fall back to reading mapping file for non-standard column names   
    if !isfile(mapping_json_path)
        error("Column mapping file not found: $mapping_json_path\n" *
              "Data does not have standard column names and mapping file is required.")
    end
    
    mapping = JSON.parsefile(mapping_json_path)
    env_col_name = mapping["env"]
    # Check for trait_value or trait in mapping
    trait_value_col_name = haskey(mapping, "trait_value") ? mapping["trait_value"] : mapping["trait"]
    trait_col_name = trait_value_col_name
    geno_col_name = mapping["geno"]
    family_col_name = get(mapping, "family", "family")
    trait_name_col_name = get(mapping, "trait_name", "trait_name")
    
    println("Using column mappings from file:")
    println("  Environment: $env_col_name")
    println("  Trait Value: $trait_value_col_name")
    println("  Genotype: $geno_col_name")
    println("  Family: $family_col_name")
    println("  Trait Name: $trait_name_col_name")
end

# Verify required columns exist
required_cols = [env_col_name, trait_value_col_name, geno_col_name, trait_name_col_name]
missing_cols = setdiff(required_cols, names(phenotype_data))
if !isempty(missing_cols)
    error("Missing required columns in data: $(missing_cols)")
end

# Check optional columns (assume they have one level if missing - they won't be used in model)
has_family = family_col_name in names(phenotype_data)
if !has_family
    println("Note: Family column not found - will be skipped (assumed to have one level)")
end

# Get unique environments from the dataframe using column name
env_col = Symbol(env_col_name)
environments = unique(phenotype_data[:, env_col])
println("\nFound $(length(environments)) environments: $(environments)")

results_list = DataFrame[]

# Process each environment
for environment in environments
    println("\nProcessing environment: $environment")
    
    # Filter data for this environment using dynamic column name
    env_filter = phenotype_data[:, env_col] .== environment
    environment_data = phenotype_data[env_filter, :]
    
    # Get formula for this environment
    formula_row = formulas_df[formulas_df.Environment .== environment, :]
    
    if nrow(formula_row) == 0
        println("Warning: No formula found for environment $environment, skipping...")
        continue
    end
    
    formula_str = formula_row.Formula[1]
    formula = Meta.parse("@formula($formula_str)")
    parsed_formula = eval(formula)
    
    println("Formula: $parsed_formula")
    
    # Fit mixed model
    model = fit(MixedModel, parsed_formula, environment_data)
    println("Model fitted for $environment")
    
    # Extract fixed effects and calculate BLUE values
    fixed_coefficients = fixef(model)
    design_matrix = model.X
    blue_values = design_matrix * fixed_coefficients
    
    # Create results dataframe using dynamic column names
    trait_col = Symbol(trait_col_name)
    trait_name_col = Symbol(trait_name_col_name)
    geno_col = Symbol(geno_col_name)

    results_df = DataFrame(
        env = environment_data[:, env_col],
        geno = environment_data[:, geno_col],
        trait_name = environment_data[:, trait_name_col],
        BLUE_values = blue_values,
        observed_value = environment_data[:, trait_col]
    )
    
    # Add optional family column if it exists
    if has_family
        family_col_sym = Symbol(family_col_name)
        results_df[!, :family] = environment_data[:, family_col_sym]
    end
    
    # Keep unique genotypes (one BLUE per genotype per environment)
    results_filtered = unique(results_df, :geno)
    push!(results_list, results_filtered)
    
    println("Completed $environment: $(nrow(results_filtered)) genotypes")
end

# Combine all results

all_results = vcat(results_list...)
CSV.write(output_blues_path, all_results)
println("\nAll results saved to: $output_blues_path")
println("Total genotypes: $(nrow(all_results))")
