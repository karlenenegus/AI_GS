#!/usr/bin/env python3
"""
Preprocess phenotype data using BLUE values.

Splits data into train/validation/test sets and scales phenotype values by environment.
Uses stratified group k-fold cross-validation to ensure genotypes are not split across sets.

Arguments:
    -r, --randomstate (int, required): Random state for reproducible random processes
    -d, --output_dir (str, required): Output directory for results
    -f, --input_pheno_file (str, required): Path to phenotype file (CSV or Excel format)
    -o, --output_pheno_file_prefix (str, required): File prefix for output phenotype files.
        Will append 'Training', 'Testing', 'Validation' to generate .csv files
    -m, --mapping_json_path (str, required): Path to JSON mapping file
    -g, --geno_col (str, default='geno'): Name of genotype column in phenotype file
    -p, --pheno_col (str, default='BLUE_values'): Name of phenotype column in phenotype file
    -e, --env_col (str, default='env'): Name of environment column in phenotype file
    -t, --train_split (float, default=0.80): Percentage of whole dataset used for training
    -s, --test_split (float, default=0.10): Percentage of whole dataset used for testing
    -v, --validation_split (float, default=0.10): Percentage of whole dataset used for validation
    --envs_hold_out (int, default=0): Hold out n environments for testing. If > 0, creates
        environment dictionary file for unseen environment handling

Note: train_split + test_split + validation_split must equal 1.0

Outputs:
    - {prefix}_Training.csv: Training set with scaled phenotypes
    - {prefix}_Validation.csv: Validation set with scaled phenotypes
    - {prefix}_Testing.csv: Testing set with scaled phenotypes
    - keep_geno_prefixes_{split}.txt: Genotype lists for each split
    - conversion_keys/env_dict_file.csv: Environment dictionary (if envs_hold_out > 0)
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedGroupKFold

# Add project root to Python path to enable imports from lib/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from lib/
from lib.preprocess_phenotypes_fns import (
    convert2output,
    scale_phenotype
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Preprocess phenotype data using BLUE values")

parser.add_argument("-r", "--randomstate", type=int, required=True, help="Random state for reproducible random processes")
parser.add_argument("-d", "--output_dir", type=str, required=True, help="Output directory for additional")
parser.add_argument("-f", "--input_pheno_file", type=str, required=True, help="Path to phenotype file (CSV or Excel format)")
parser.add_argument("-o", "--output_pheno_file_prefix", type=str, required=True, help="File prefix for output phenotype file. Will append 'Training', 'Testing', 'Validation' prefix to generate .csv")
parser.add_argument("-m", "--mapping_json_path", type=str, required=True, help="Path to json ")


parser.add_argument("-g", "--geno_col", type=str, default="geno", help="Name of genotype column in phenotype file (default: 'geno' - standard name)")
parser.add_argument("-p", "--pheno_col", type=str, default="BLUE_values", help="Name of phenotype column in phenotype file (default: 'BLUE_values' from BLUE calculation)")
parser.add_argument("-e", "--env_col", type=str, default="env", help="Name of environment column in phenotype file (default: 'env' - standard name)")

parser.add_argument("-t", "--train_split", type=float, default=0.80, help="Percentage of whole dataset used to generate the training fraction")
parser.add_argument("-s", "--test_split", type=float, default=0.10, help="Percentage of whole dataset used to generate the test fraction")
parser.add_argument("-v", "--validation_split", type=float, default=0.10, help="Percentage of whole dataset used to generate the validation fraction")

parser.add_argument("--envs_hold_out", type = int, default=0, help="Hold out n environments for testing (default: False)")

args = parser.parse_args()

if args.validation_split + args.test_split + args.train_split != 1:
        raise ValueError("The sum of the validation, test, and train splits must be equal to 1 (or use 1.0, 0, 0 for all data in one split)")

# Setup
output_dir = Path(args.output_dir)
if args.envs_hold_out > 0:
    env_hold_out = True
else:
    env_hold_out = False

# Load BLUE results
input_pheno_file = Path(args.input_pheno_file)
if not input_pheno_file.exists():
    raise FileNotFoundError(f"BLUE results file not found: {input_pheno_file}")

pheno_data = pd.read_csv(input_pheno_file)

# Load column mapping to get trait name
mapping_file = Path(args.mapping_json_path)
if not mapping_file.exists():
    raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

with open(mapping_file, 'r') as f:
    column_mapping = json.load(f)

# Save genotype list for filtering
geno_list_file = output_dir / "data" / "keep_geno_prefixes.txt"
pheno_data[args.geno_col].astype(str).to_csv(
    geno_list_file,
    index=False,
    header=False
)

# Handle special case: (1.0, 0, 0) - all data goes to one split
if args.train_split == 1.0 and args.test_split == 0 and args.validation_split == 0:
    pheno_data['fold'] = 0  # All in fold 0
    k_train = 1
    k_val = 0
    k_test = 0
    k_fold = 1
elif args.test_split == 1.0 and args.train_split == 0 and args.validation_split == 0:
    pheno_data['fold'] = 0  # All in fold 0
    k_train = 0
    k_val = 0
    k_test = 1
    k_fold = 1
elif args.validation_split == 1.0 and args.train_split == 0 and args.test_split == 0:
    pheno_data['fold'] = 0  # All in fold 0
    k_train = 0
    k_val = 1
    k_test = 0
    k_fold = 1
else:
    # Create cross-validation folds
    min_split = min([args.validation_split, args.test_split, args.train_split])
    max_k = 100

    scale = min(round(1 / min_split), max_k)

    k_train = round(args.train_split * scale, 2)
    k_val   = round(args.validation_split * scale, 2)
    k_test  = round(args.test_split * scale, 2)
    k_fold = round(k_train + k_val + k_test, 2)
    k_fold = int(k_fold)

    split_kfold = StratifiedGroupKFold(n_splits=int(k_fold), shuffle=True, random_state=args.randomstate)
    pheno_data['fold'] = -1

    for fold_number, (_, test_idx) in enumerate(split_kfold.split(pheno_data, pheno_data[args.env_col], pheno_data[args.geno_col])):
        test_genotypes = pheno_data[args.geno_col].values[test_idx]
        pheno_data.loc[pheno_data[args.geno_col].isin(test_genotypes), "fold"] = fold_number

# Handle environment holdout
if env_hold_out:
    unique_envs = pheno_data[args.env_col].unique()
    np.random.seed(args.randomstate)
    env_to_hold_out = np.random.choice(unique_envs, size=args.env_hold_out)
    
    # Create environment dictionary
    env_df = pd.DataFrame({
        'env': unique_envs,
        'env_index': range(1, len(unique_envs) + 1)
    })
    env_df.to_csv(output_dir / 'data' / 'encoding_keys' / 'env_dict_file.csv', index=False)

# Create train/test/val splits based on folds
if k_fold == 1:
    # Special case: all data in one split
    if args.train_split == 1.0:
        train_idx = np.ones(len(pheno_data), dtype=bool)
        test_idx = np.zeros(len(pheno_data), dtype=bool)
        val_idx = np.zeros(len(pheno_data), dtype=bool)
    elif args.test_split == 1.0:
        train_idx = np.zeros(len(pheno_data), dtype=bool)
        test_idx = np.ones(len(pheno_data), dtype=bool)
        val_idx = np.zeros(len(pheno_data), dtype=bool)
    else:  # validation_split == 1.0
        train_idx = np.zeros(len(pheno_data), dtype=bool)
        test_idx = np.zeros(len(pheno_data), dtype=bool)
        val_idx = np.ones(len(pheno_data), dtype=bool)
else:
    fold_range = range(k_fold)
    train_idx = np.isin(pheno_data['fold'], fold_range[:int(k_train)])
    test_idx = np.isin(pheno_data['fold'], fold_range[int(k_train):int(k_train+k_val)] if k_val > 0 else [])
    val_idx = np.isin(pheno_data['fold'], fold_range[int(k_train+k_val):] if k_val > 0 else fold_range[int(k_train):])

# Adjust splits if environment holdout is enabled
if env_hold_out:
    env_holdout_idx = np.isin(pheno_data[args.env_col], [env_to_hold_out])
    
    # Exclude the held-out environment from train and val
    train_idx = train_idx & ~env_holdout_idx
    val_idx = val_idx & ~env_holdout_idx
    
    # Add the full held-out environment to test set
    test_idx = test_idx | env_holdout_idx

# Use the trait name from phenotypes
if 'trait_name' not in pheno_data.columns:
    pheno_data['trait_name'] = args.pheno_col
elif args.pheno_col != 'trait_value':
    pheno_data['trait_name'] = args.pheno_col + '_' + pheno_data['trait_name'].astype(str)

trait_name = 'trait_name'

# Create split datasets
data_train = pheno_data.iloc[train_idx].copy()
data_test = pheno_data.iloc[test_idx].copy()
data_val = pheno_data.iloc[val_idx].copy()

# Convert splits to output format
data_train = convert2output(
    data=data_train,
    genotype_col=args.geno_col,
    environ_col=args.env_col,
    phenotype_col=args.pheno_col,
    phenotype_name=trait_name
)
data_val = convert2output(
    data=data_val,
    genotype_col=args.geno_col,
    environ_col=args.env_col,
    phenotype_col=args.pheno_col,
    phenotype_name=trait_name
)
data_test = convert2output(
    data=data_test,
    genotype_col=args.geno_col,
    environ_col=args.env_col,
    phenotype_col=args.pheno_col,
    phenotype_name=trait_name
)

# Scale phenotype values by environment
# After convert2output, columns are renamed to standard names: geno, env, trait_value, trait_name
env_col_name = 'env'  # Standard name after convert2output
unique_envs = data_train[env_col_name].unique()

data_train_scaled = [
    scale_phenotype(
        data=data_train[data_train[env_col_name] == env],
        env=env,
        mode="train",
        location=str(output_dir),
        pheno_col='trait_value'  # Standard name after convert2output
    )
    for env in unique_envs
]

data_val_scaled = [
    scale_phenotype(
        data=data_val[data_val[env_col_name] == env],
        env=env,
        mode="inference",
        location=str(output_dir),
        pheno_col='trait_value'  # Standard name after convert2output
    )
    for env in unique_envs
]

data_test_scaled = [
    scale_phenotype(
        data=data_test[data_test[env_col_name] == env],
        env=env,
        mode="inference",
        location=str(output_dir),
        pheno_col='trait_value'  # Standard name after convert2output
    )
    for env in unique_envs
]

# Handle unseen environment scaling if holdout is enabled
if env_hold_out:
    data_test_unseen_scaled = scale_phenotype(
        data=data_test[data_test[env_col_name] == env_to_hold_out],
        env=env_to_hold_out,
        mode="train",
        location=str(output_dir),
        pheno_col='trait_value'  # Standard name after convert2output
    )
    data_test_scaled = data_test_scaled + [data_test_unseen_scaled]

# Combine scaled data
data_train_scaled = pd.concat(data_train_scaled, ignore_index=True)
data_val_scaled = pd.concat(data_val_scaled, ignore_index=True)
data_test_scaled = pd.concat(data_test_scaled, ignore_index=True)

# Write output files
data_train_scaled.to_csv(f'{args.output_pheno_file_prefix}_Training.csv', index=False)
data_val_scaled.to_csv(f'{args.output_pheno_file_prefix}_Validation.csv', index=False)
data_test_scaled.to_csv(f'{args.output_pheno_file_prefix}_Testing.csv', index=False)

# Save genotype list for filtering
prefixes_training_file = output_dir / "data" / f"keep_geno_prefixes_training.txt"
data_train_scaled['geno'].astype(str).to_csv(
    prefixes_training_file,
    index=False,
    header=False
)

prefixes_validation_file = output_dir / "data" / f"keep_geno_prefixes_validation.txt"
data_val_scaled['geno'].astype(str).to_csv(
    prefixes_validation_file,
    index=False,
    header=False
)

prefixes_testing_file = output_dir / "data" / f"keep_geno_prefixes_testing.txt"
data_test_scaled['geno'].astype(str).to_csv(
    prefixes_testing_file,
    index=False,
    header=False
)

print(f"Split complete: Train={len(data_train_scaled)}, Val={len(data_val_scaled)}, Test={len(data_test_scaled)}. Saved to: {output_dir}")
