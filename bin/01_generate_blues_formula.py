#!/usr/bin/env python3
"""
Generate BLUE (Best Linear Unbiased Estimator) formulas for mixed model analysis.

Processes filtered phenotype data and generates environment-specific formulas for BLUE calculations.
Creates mixed model formulas with random effects based on the column mapping configuration.

Arguments:
    -o, --output_dir (str, required): Output directory for results (should contain filtered data
        from 00_filter_phenotype_data.py)
    -p, --input_pheno_path (str, optional): Path to filtered phenotype data CSV
        (default: {output_dir}/data/00_filtered_phenotype_data.csv)
    -f, --output_formulas_path (str, optional): Path to output formulas CSV file
        (default: {output_dir}/data/01_BLUEs_formulas.csv)
    -d, --output_pheno_path (str, optional): Path to output phenotype data CSV file
        (default: {output_dir}/data/01_BLUEs_formula_phenotype_data.csv)
    -m, --mapping_json_path (str, optional): Path to mapping JSON file
        (default: {output_dir}/data/pheno_column_mapping.json)
    -b, --filter_by_name (str, optional): Path to file containing geno_env combinations to filter by
        (one per line, format: geno_env). Only environments with these combinations will be processed.

Outputs:
    - BLUE formulas CSV file with environment-specific formulas
    - Phenotype data CSV file prepared for BLUE calculation
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import json
import warnings

# Add project root to Python path to enable imports from lib/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from lib/
from lib.preprocess_phenotypes_fns import generate_valid_random_terms

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate BLUE formulas for mixed model analysis")

parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for results (should contain filtered data from 00_filter_phenotype_data.py)")
parser.add_argument("-p", "--input_pheno_path", type=str, default=None, help="Path to filtered phenotype data CSV (default: {output_dir}/data/00_filtered_phenotype_data.csv)")
parser.add_argument("-f", "--output_formulas_path", type=str, default=None, help="Path to output formulas CSV file (default: {output_dir}/data/01_BLUEs_formulas.csv)")
parser.add_argument("-d", "--output_pheno_path", type=str, default=None, help="Path to output phenotype data CSV file (default: {output_dir}/data/01_BLUEs_formula_phenotype_data.csv)")
parser.add_argument("-m", "--mapping_json_path", type=str, default=None, help="Path to mapping JSON file (default: {output_dir}/data/pheno_column_mapping.json)")
parser.add_argument("-b", "--filter_by_name", type=str, default=None, help="Path to file containing geno_env combinations to filter by (one per line, format: geno_env). Only environments with these combinations will be processed.")

args = parser.parse_args()

# Setup
output_dir = Path(args.output_dir)
if not output_dir.exists():
    raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

# Determine input files
if args.input_pheno_path:
    input_pheno_path = Path(args.input_pheno_path)
else:
    input_pheno_path = output_dir / 'data' / '00_filtered_phenotype_data.csv'

# Determine output file
if args.output_formulas_path:
    output_formulas_path = Path(args.output_formulas_path)
else:
    output_formulas_path = output_dir / 'data' / '01_BLUEs_formulas.csv'

# Determine output phenotype file
if args.output_pheno_path:
    output_pheno_path = Path(args.output_pheno_path)
else:
    output_pheno_path = output_dir / 'data' / '01_BLUEs_formula_phenotype_data.csv'

# Determine mapping file
if args.mapping_json_path:
    mapping_json_path = Path(args.mapping_json_path)
else:
    mapping_json_path = output_dir / 'data' / 'pheno_column_mapping.json'

# Check if files exist
if not input_pheno_path.exists():
    raise FileNotFoundError(f"Filtered data file not found: {input_pheno_path}\nPlease run 00_filter_phenotype_data.py first to generate filtered data.")

if not mapping_json_path.exists():
    raise FileNotFoundError(f"Mapping JSON file not found: {mapping_json_path}\nPlease create a mapping file. See data/pheno_column_mapping.json.example for an example.")

# Load filtered phenotype data
phenotype_data = pd.read_csv(input_pheno_path)

# Load mapping file
with open(mapping_json_path, 'r') as f:
    mapping = json.load(f)

# Get column mappings - handle both old and new mapping formats
geno_col = mapping.get('geno')
env_col = mapping.get('env', 'env')
trait_value_col = mapping.get('trait_value') or mapping.get('trait', 'trait_value')
trait_name_col = mapping.get('trait_name', trait_value_col)

# Apply filter_by_name if provided
if args.filter_by_name:
    filter_file = Path(args.filter_by_name)
    if not filter_file.exists():
        raise FileNotFoundError(f"Filter file not found: {filter_file}")
    
    with open(filter_file, 'r') as f:
        keep_combinations = set(line.strip() for line in f if line.strip())
    
    # Create geno_env column for filtering
    phenotype_data['geno_env'] = phenotype_data[geno_col].astype(str) + '_' + phenotype_data[env_col].astype(str)
    
    phenotype_data = phenotype_data[phenotype_data['geno_env'].isin(keep_combinations)].copy()
    
    # Drop the temporary geno_env column
    phenotype_data = phenotype_data.drop(columns=['geno_env'])

# Access mapped column names (use standard names directly since data was standardized)
# Standard columns: trait_value, trait_name, env, geno
model_data = pd.DataFrame()
model_data['trait_value'] = phenotype_data[trait_value_col]
if trait_name_col in phenotype_data.columns:
    model_data['trait_name'] = phenotype_data[trait_name_col]
else:
    model_data['trait_name'] = trait_value_col  # Use trait_value_col as fallback
model_data['env'] = phenotype_data[env_col]
model_data['geno'] = phenotype_data[geno_col].astype(str)

# Handle optional columns (fill with '1' if missing - will be filtered out by generate_valid_random_terms)
optional_cols = mapping.keys()
reverse_mapping = {}

for col in optional_cols:
    if col != "random_effects_variables":
        if mapping[col]:
            # Check if column exists in mapping and in data
            if mapping[col] in phenotype_data.columns:
                model_data[col] = phenotype_data[mapping[col]].astype(str)
                reverse_mapping[mapping[col]] = col
            else:
                warnings.warn(f"Column {col}: {mapping[col]} not found in data and will be skipped. Check {args.mapping_json_path}. Available phenotype data columns: {list(phenotype_data.columns)}")
        else:
            warnings.warn(f"Column {col} set to None and will be skipped. Check {args.mapping_json_path} if not intended.")

# Create nested random effect terms
random_effects_variables = mapping.get('random_effects_variables', [])
new_random_effects_variables = []
for old_effect_name in random_effects_variables:
    new_effect_names = []
    split_old_effect = old_effect_name.split('_') #[value1, value2, ..., valueN]
    if len(split_old_effect) == 1:
        new_effect_names_str = split_old_effect[0]
    else:
        for old_name in split_old_effect:
            new_name = reverse_mapping.get(old_name, None)
            if new_name is None:
                if old_name in model_data.columns:
                    new_name = old_name
                else: 
                    raise ValueError(f"{old_name} is missing from both the data and column mapping file.")
            new_effect_names.append(new_name)
        new_effect_names_str = '_'.join(new_effect_names)
        model_data[new_effect_names_str] = model_data[new_effect_names].astype(str).agg('_'.join, axis=1)
    
    new_random_effects_variables.append(new_effect_names_str)
        
formulas = {}

# Generate formulas for each environment
for environment in model_data["env"].unique():
    environment_data = model_data[model_data["env"] == environment]
    
    if not random_effects_variables:
        warnings.warn("No random effects provided in mapping file. BLUEs formula will be {formula}")
        formula = "trait_value ~ geno"
    else:
        random_effects = generate_valid_random_terms(environment_data, new_random_effects_variables)
        
        # Build formula (use trait_value as the response variable)
        fixed_effects = "trait_value ~ geno + "
        formula = fixed_effects + " + ".join(random_effects)
    
    formulas[environment] = formula

# Output formulas
formulas_df = pd.DataFrame.from_dict(
    formulas,
    orient='index',
    columns=['Formula']
)
formulas_df.reset_index(inplace=True)
formulas_df.rename(columns={'index': 'Environment'}, inplace=True)

output_pheno_path.parent.mkdir(parents=True, exist_ok=True)
model_data.to_csv(output_pheno_path, index=False)

# Ensure output directory exists
output_formulas_path.parent.mkdir(parents=True, exist_ok=True)
formulas_df.to_csv(output_formulas_path, index=False, quoting=1)
print(f"Generated {len(formulas)} BLUE formulas for {len(model_data['env'].unique())} environments. Saved to: {output_formulas_path}")
