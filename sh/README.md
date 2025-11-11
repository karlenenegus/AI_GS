# AI Genomic Selection Pipeline - Shell Script Version

This directory contains code for processing phenotype and genotype data for use with the AI-GS pipeline.

## Overview

Steps:
0. **Filter phenotype data** - Removes outliers and filters data based on quality criteria
1. **Generate BLUE formulas** - Creates formulas for Best Linear Unbiased Estimates
2. **Calculate BLUEs** - Computes BLUE values using mixed models (Julia)
3. **Split phenotypes** - Divides data into training, validation, and testing sets
4. **Encode genotypes** - Converts genotype data into encoded format for machine learning

## Prerequisites

### Software Requirements
- **Python** (with conda environment)
- **Julia** (version 1.10.4 or compatible)

### Environment Setup
1. Activate the conda environment:
   ```bash
   source <path_to_conda_environment>/bin/activate
   ```
2. Set Julia environment variables (adjust paths as needed):
   ```bash
   export JULIA_DEPOT_PATH=<path_to_julia_depot>
   export JULIA_PROJECT=<path_to_julia_project>
   ```

### Julia Packages
The script automatically installs required Julia packages on first run:
- MixedModels
- DataFrames
- CSV
- JSON
- CategoricalArrays
- StatsModels

## Usage

### Running the Pipeline

To run locally:

```bash
bash run_pipeline.sh
```


### Configuration

Before running, you may need to modify the following in `run_pipeline.sh`:

- **OUTPUT_DIR**: Directory where results will be saved (default: `./results/`)
- **INPUT_FILE_PHENO**: Path to phenotype file (.csv)
- **INPUT_FILE_GENO**: Path to genotype file (.hmp)
- **PHENOTYPE_COLS**: Path to phenotype column names to standard column names mapping (.json)


### Input Files

The pipeline expects the following input files:

1. **Phenotype data**: Phenotype data should be formatted like example at ./data/pheno/example_phenotypes.csv
2. **Phenotype mapping**: Phenotype column mapping file should be formatted like example at `./data/pheno/pheno_column_mapping.json`
3. **Genotype data**: Genotype data should be in HapMap format like exammple data. `./data/geno/example_geno.hmp`

### Output Files

Results are saved in the `OUTPUT_DIR` directory structure:

```
results/
├── data/
│   ├── 00_filtered_phenotype_data.csv          # Step 0: Filtered phenotype data
│   ├── 01_BLUEs_formulas.csv                   # Step 1: BLUE formulas
│   ├── 01_BLUEs_formula_phenotype_data.csv     # Step 1: Phenotype data with formulas
│   ├── 02_BLUEs_results.csv                    # Step 2: Calculated BLUE values
│   ├── 03_Phenotype_Data_Training.csv          # Step 3: Training set
│   ├── 03_Phenotype_Data_Validation.csv        # Step 3: Validation set
│   └── 03_Phenotype_Data_Testing.csv           # Step 3: Testing set
├── encoding_config.yaml                        # Step 4: Encoding configuration (created during Training)
├── Encodings_Training_parquet/                 # Step 4: Training encodings (parquet files)
│   ├── encodings.pkl                           # Encodings pickle file
│   └── [parquet data files]
├── Encodings_Validation_parquet/               # Step 4: Validation encodings
│   ├── encodings.pkl
│   └── [parquet data files]
└── Encodings_Testing_parquet/                   # Step 4: Testing encodings
    ├── encodings.pkl
    └── [parquet data files]
```

**Note:** The encoding directories (`Encodings_*_parquet/`) contain parquet files with combined phenotype and genotype encodings. The `encoding_config.yaml` file is only created during the Training step and is required for Validation and Testing steps.

## Pipeline Steps

### Step 0: Filter Phenotype Data
Filters phenotype data using IQR-based outlier detection and minimum environment requirements.

**Script:** `bin/00_filter_phenotype_data.py`

**Parameters:**
- `--iqr_multiplier`: IQR multiplier for outlier detection (default: 1.5)
- `--min_environments`: Minimum number of environments required (default: 2)

**Output:** `data/00_filtered_phenotype_data.csv`

### Step 1: Generate BLUE Formulas
Generates formulas for calculating Best Linear Unbiased Estimates.

**Script:** `bin/01_generate_blues_formula.py`

**Output:** 
- `data/01_BLUEs_formulas.csv`
- `data/01_BLUEs_formula_phenotype_data.csv`

### Step 2: Calculate BLUEs
Calculates BLUE values using mixed models in Julia. This step requires Julia and the specified packages.

**Script:** `bin/02_calculate_blues.jl`

**Output:** `data/02_BLUEs_results.csv`

### Step 3: Split Phenotypes
Splits the processed phenotype data into training, validation, and testing sets.

**Script:** `bin/03_split_phenotypes.py`

**Parameters:**
- `--randomstate`: Random seed for reproducibility (default: 30)
- `--envs_hold_out`: Number of environments to hold out (default: 0)

**Output:**
- `data/03_Phenotype_Data_Training.csv`
- `data/03_Phenotype_Data_Validation.csv`
- `data/03_Phenotype_Data_Testing.csv`

### Step 4: Encode Genotypes
Encodes genotype data for AI genomic selection models. Creates parquet files with combined phenotype and genotype encodings.

**Script:** `bin/04_encode_genotypes.py`

**Parameters:**
- `--encoding_window_size`: Window size for encoding (default: 10)
- `--shift`: Shift parameter for encoding (default: 0)
- `--encoding_mode`: Encoding mode (default: dosage)

**Output:**
- `encoding_config.yaml` (created during Training step)
- `Encodings_Training_parquet/` with parquet files
- `Encodings_Validation_parquet/` with parquet files
- `Encodings_Testing_parquet/` with parquet files

## Notes
