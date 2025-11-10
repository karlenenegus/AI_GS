#!/usr/bin/env python3
"""
Filter phenotype data for genomic prediction analysis.

Removes outliers, missing values, and genotypes with insufficient observations.
Filters data using IQR-based outlier detection and ensures genotypes have
observations across a minimum number of environments.

Arguments:
    -m, --mapping_json_path (str, required): Path to column mapping JSON file for phenotype data
    -f, --input_pheno_path (str, required): Path to phenotype file (CSV format)
    -o, --output_pheno_path (str, required): Path to output filtered phenotype file (CSV format)
    -g, --geno_col (str, required): Name of genotype column in input phenotype file
    -p, --pheno_col (str, required): Name of phenotype column in input phenotype file
    -e, --env_col (str, required): Name of environment column in input phenotype file
    -i, --iqr_multiplier (float, default=1.5): IQR multiplier for outlier filtering
    -n, --min_environments (int, default=2): Minimum number of environments a genotype must have
    -b, --filter_by_name (str, optional): Path to file containing geno_env combinations to keep
        (one per line, format: geno_env)

Outputs:
    - Filtered phenotype CSV file with outliers and insufficient genotypes removed
    - Column mapping JSON file (created/updated if needed)
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import warnings

# Add project root to Python path to enable imports from lib/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from lib/
from lib.preprocess_phenotypes_fns import (
    filter_iqr_outliers,
    load_json,
    save_json
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Filter phenotype data for genomic prediction")

parser.add_argument("-m", "--mapping_json_path", type=str, required=True, help="Path to mapping JSON file") 
parser.add_argument("-f", "--input_pheno_path", type=str, required=True, help="Path to phenotype file (CSV or Excel format)")
parser.add_argument("-o", "--output_pheno_path", type=str, required=True, help="Path to output filtered phenotype file (CSV or Excel format)")
parser.add_argument("-g", "--geno_col", type=str, required=True, help="Name of genotype column in input phenotype file")
parser.add_argument("-p", "--pheno_col", type=str, required=True, help="Name of phenotype column in input phenotype file")
parser.add_argument("-e", "--env_col", type=str, required=True, help="Name of environment column in input phenotype file")
parser.add_argument("-i", "--iqr_multiplier", type=float, default=1.5, help="IQR multiplier for outlier filtering (default: 1.5)")
parser.add_argument("-n", "--min_environments", type=int, default=2, help="Minimum number of environments a genotype must have (default: 2)")
parser.add_argument("-b", "--filter_by_name", type=str, default=None, help="Path to file containing geno_env combinations to keep (one per line, format: geno_env)")
args = parser.parse_args()

# Load phenotype data
if args.input_pheno_path.endswith('.csv'):
    raw_phenotype_data = pd.read_csv(args.input_pheno_path)
elif args.pheno_file.endswith(('.xls', '.xlsx')):
    raw_phenotype_data = pd.read_excel(args.input_pheno_path)
else:
    raise ValueError(f"Unsupported file format: {args.input_pheno_path}")

# Load JSON mapping file for column names
mapping_json_path = args.mapping_json_path
write_mapping = False

if not Path(mapping_json_path).exists():
    write_mapping = True
    column_mapping = {
        "trait": args.pheno_col,
        "trait_value": None,
        "trait_name": None,
        "env": args.env_col,
        "family": None,
        "geno": args.geno_col,
        "field": None,
        "rep": None,
        "set": None,
        "block": None,
        "row": None,
        "column": None
    }
    warnings.warn(f"Column mapping file not found: {mapping_json_path}\n Creating basic column mapping file that may need to be updated for subsequent processes. See pheno_column_mapping.json in README for an example.")
else:
    column_mapping = load_json(mapping_json_path)


# Filter missing phenotypes
missing_mask = raw_phenotype_data[args.pheno_col].isna()
n_missing = missing_mask.sum()
phenotype_data = raw_phenotype_data.loc[~missing_mask]

# Filter outlier phenotypes
filtered_phenotype_data = filter_iqr_outliers(
    data=phenotype_data,
    geno_col=args.geno_col,
    env_col=args.env_col,
    pheno_col=args.pheno_col,
    iqr_multiplier=args.iqr_multiplier
)

# Filter genotypes with observations in less than min_environments environments
genotype_env_counts = filtered_phenotype_data[[args.geno_col, args.env_col]].drop_duplicates()
env_counts = genotype_env_counts[args.geno_col].value_counts()
valid_genotypes = env_counts[env_counts >= args.min_environments].index
n_invalid = len(env_counts) - len(valid_genotypes)
valid_genotype_mask = filtered_phenotype_data[args.geno_col].isin(valid_genotypes)
filtered_phenotype_data = filtered_phenotype_data.loc[valid_genotype_mask]

# Apply filter_by_name if provided
if args.filter_by_name:
    filter_file = Path(args.filter_by_name)
    if not filter_file.exists():
        raise FileNotFoundError(f"Filter file not found: {filter_file}")
    
    with open(filter_file, 'r') as f:
        keep_combinations = set(line.strip() for line in f if line.strip())
    
    # Create geno_env column for filtering
    filtered_phenotype_data['geno_env'] = filtered_phenotype_data[args.geno_col].astype(str) + '_' + filtered_phenotype_data[args.env_col].astype(str)
    
    filtered_phenotype_data = filtered_phenotype_data[filtered_phenotype_data['geno_env'].isin(keep_combinations)].copy()
    
    # Drop the temporary geno_env column
    filtered_phenotype_data = filtered_phenotype_data.drop(columns=['geno_env'])
    
if column_mapping['geno'] != args.geno_col:
    warnings.warn("Genotype column specified in column mapping file and arguments differ. Overwriting column mapping file.")
    column_mapping['geno'] = args.geno_col
    write_mapping = True
if column_mapping['env'] != args.env_col:
    warnings.warn("Environment column specified in column mapping file and arguments differ. Overwriting column mapping file.")
    column_mapping['env'] = args.env_col
    write_mapping = True
if column_mapping['trait'] != args.pheno_col:
    warnings.warn("Environment column specified in column mapping file and arguments differ. Overwriting column mapping file.")
    column_mapping['trait'] = args.pheno_col
    write_mapping = True

if write_mapping:
    save_json(column_mapping, mapping_json_path)
    print(f"Column mapping saved to: {args.mapping_json_path}")

# Save filtered phenotype data with standard column names
Path(args.output_pheno_path).parent.mkdir(parents=True, exist_ok=True)
filtered_phenotype_data.to_csv(args.output_pheno_path, index=False)
print(f"Filtered {len(raw_phenotype_data)} -> {len(filtered_phenotype_data)} rows. Saved to: {args.output_pheno_path}")
