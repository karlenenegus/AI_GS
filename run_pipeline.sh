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

module load julia/1.10.4-py311-jctw3xe

export JULIA_DEPOT_PATH=/home/knegus/local/julia_lib
export JULIA_PROJECT=/home/knegus/local/julia_lib/BLUEs

julia -e 'using Pkg; Pkg.add(["MixedModels", "DataFrames", "CSV", "JSON", "CategoricalArrays", "StatsModels"]); Pkg.precompile()'


OUTPUT_DIR='./results'
mkdir -p ${OUTPUT_DIR}

INPUT_FILE_PHENO='./data/pheno/photoperiod_phenotypes.csv'
INPUT_FILE_GENO='./data/geno/SNPs_Final_2k.hmp'
PHENOTYPE_COLS='./data/pheno/pheno_column_mapping.json'

# Step 1: Filter phenotype data
echo "Step 1: Filtering phenotype data..."
python bin/00_filter_phenotype_data.py \
    --mapping_json_path ${PHENOTYPE_COLS} \
    --input_pheno_path ${INPUT_FILE_PHENO} \
    --output_pheno_path ${OUTPUT_DIR}/data/00_filtered_phenotype_data.csv \
    --geno_col geno_code \
    --pheno_col gdd_dta \
    --env_col env

# Step 2: Generate BLUE formulas
echo "Step 2: Generating BLUE formulas..."
python bin/01_generate_blues_formula.py \
    --output_dir ${OUTPUT_DIR} \
    --mapping_json_path ${PHENOTYPE_COLS}

# Step 3: Calculate BLUEs (Julia script)
echo "Step 3: Calculating BLUEs..."
julia bin/02_calculate_blues.jl \
    ${OUTPUT_DIR}/data/01_BLUEs_formula_phenotype_data.csv \
    ${OUTPUT_DIR}/data/01_BLUEs_formulas.csv \
    ${PHENOTYPE_COLS} \
    ${OUTPUT_DIR}/data/02_BLUEs_results.csv

#Step 4: Split phenotypes into train/test/validation
echo "Step 4: Splitting phenotypes..."
python bin/03_split_phenotypes.py \
    --randomstate 30 \
    --output_dir ${OUTPUT_DIR} \
    --mapping_json_path ${PHENOTYPE_COLS} \
    --input_pheno_file ${OUTPUT_DIR}/data/02_BLUEs_results.csv \
    --output_pheno_file_prefix ${OUTPUT_DIR}/data/03_Phenotype_Data

# Step 5: Encode genotypes
echo "Step 5: Encoding genotypes..."
python bin/04_encode_genotypes.py \
    --Training \
    --output_folder ${OUTPUT_DIR} \
    --input_hmp_file ${INPUT_FILE_GENO} \
    --pheno_file ${OUTPUT_DIR}/data/03_Phenotype_Data_Training.csv \
    --encoding_mode dosage

python bin/04_encode_genotypes.py \
    --Validation \
    --output_folder ${OUTPUT_DIR} \
    --input_hmp_file ${INPUT_FILE_GENO} \
    --pheno_file ${OUTPUT_DIR}/data/03_Phenotype_Data_Validation.csv \
    --use_config

python bin/04_encode_genotypes.py \
    --Testing \
    --output_folder ${OUTPUT_DIR} \
    --input_hmp_file ${INPUT_FILE_GENO} \
    --pheno_file ${OUTPUT_DIR}/data/03_Phenotype_Data_Testing.csv \
    --use_config

echo "Phenotype preprocessing complete!" 
