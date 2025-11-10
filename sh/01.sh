#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=0-8:00:00  # max job runtime
#SBATCH --cpus-per-task=1  # number of processor cores
#SBATCH --nodes=1  # number of nodes
#SBATCH --mem=128G  # max memory
#SBATCH --ntasks-per-node=8   # 36 processor core(s) per node
#SBATCH --mail-user=knegus@iastate.edu  # email address
#SBATCH --mail-type=BEGIN,END

#SBATCH --output=./jobs/job.%J.out # tell it to store the output console text to a file called job.<assigned job number>.out
#SBATCH --error=./jobs/job.%J.err # tell it to store the error messages from the program (if it doesn't write them to normal console output) to a file called job.<assigned job number>.err

cd /work/LAS/jmyu-lab/knegus/AI_GS_NextFlow
source /work/LAS/jmyu-lab/knegus/AIGS_20250120/bin/activate

OUTPUT_DIR='./results/train1'
mkdir -p ${OUTPUT_DIR}

# Step 1: Filter phenotype data
echo "Step 1: Filtering phenotype data..."
python bin/00_filter_phenotype_data.py \
    --mapping_json_path ./data/pheno/pheno_column_mapping.json \
    --input_pheno_path ./data/pheno/photoperiod_phenotypes.csv \
    --output_pheno_path ${OUTPUT_DIR}/data/filtered_phenotype_data.csv \
    --geno_col geno_code \
    --pheno_col gdd_dta \
    --env_col env \
    --iqr_multiplier 1.5 \
    --min_environments 2

# Step 2: Generate BLUE formulas
echo "Step 2: Generating BLUE formulas..."
python bin/01_generate_blues_formula.py \
    --output_dir ${OUTPUT_DIR} \
    --input_pheno_path ${OUTPUT_DIR}/data/filtered_phenotype_data.csv \
    --output_formulas_path ${OUTPUT_DIR}/data/BLUEs_formulas.csv \
    --output_pheno_path ${OUTPUT_DIR}/data/BLUEs_formula_phenotype_data.csv \
    --mapping_json_path ./data/pheno/pheno_column_mapping.json

# Step 3: Calculate BLUEs (Julia script)
module load julia/1.10.4-py311-jctw3xe

export JULIA_DEPOT_PATH=/home/knegus/local/julia_lib
export JULIA_PROJECT=/home/knegus/local/julia_lib/BLUEs

julia -e 'using Pkg; Pkg.add(["MixedModels", "DataFrames", "CSV", "JSON", "CategoricalArrays", "StatsModels"]); Pkg.precompile()'


echo "Step 3: Calculating BLUEs..." ##TODO: Fix argument passing so julia will run
julia bin/02_calculate_blues.jl \
    ${OUTPUT_DIR}/data/BLUEs_formula_phenotype_data.csv \
    ${OUTPUT_DIR}/data/BLUEs_formulas.csv \
    ./data/pheno/pheno_column_mapping.json \
    ${OUTPUT_DIR}/data/BLUEs_results.csv

#Step 4: Split phenotypes into train/test/validation
echo "Step 4: Splitting phenotypes..."
python bin/03_split_phenotypes.py \
    --randomstate 30 \
    --output_dir ${OUTPUT_DIR} \
    --mapping_json_path ./data/pheno/pheno_column_mapping.json \
    --input_pheno_file ${OUTPUT_DIR}/data/BLUEs_results.csv \
    --output_pheno_file_prefix ${OUTPUT_DIR}/data/01_Phenotype_Data \
    --geno_col geno \
    --pheno_col BLUE_values \
    --env_col env \
    --envs_hold_out 0


# Step 5: Encode genotypes
echo "Step 5: Encoding genotypes..."
python bin/04_encode_genotypes.py \
    --Training \
    --output_folder ${OUTPUT_DIR} \
    --input_hmp_file ./data/geno/SNPs_Final_2k.hmp \
    --pheno_file ${OUTPUT_DIR}/data/01_Phenotype_Data_Training.csv \
    --geno_col geno \
    --pheno_col trait_value \
    --env_col env \
    --encoding_window_size 10 \
    --shift 0 \
    --encoding_mode dosage

# python bin/04_encode_genotypes.py \
#     --Validation \
#     --output_folder ${OUTPUT_DIR} \
#     --input_hmp_file ./data/geno/SNPs_Final_2k.hmp \
#     --pheno_file ${OUTPUT_DIR}/data/01_Phenotype_Data_Validation.csv \
#     --geno_col geno \
#     --pheno_col BLUE_values \
#     --env_col env \
#     --use_config

# python bin/04_encode_genotypes.py \
#     --Testing \
#     --output_folder ${OUTPUT_DIR} \
#     --input_hmp_file ./data/geno/SNPs_Final_2k.hmp \
#     --pheno_file ${OUTPUT_DIR}/data/01_Phenotype_Data_Testing.csv \
#     --geno_col geno \
#     --pheno_col BLUE_values \
#     --env_col env \
#     --use_config

echo "Phenotype preprocessing complete!" 
