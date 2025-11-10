#!/bin/bash
# data/download_data.sh
#
# Downloads public genomic data used in this pipeline
# Run this script before running the pipeline
# 
# This script downloads the following data:
# - ZeaGBSv27_publicSamples_imputedV5_AGPv4-181023.vcf.gz as ZeaGBSv27.vcf.gz
# - Hung_etal_2012_PNAS.zip

set -e

echo "================================================"
echo "Downloading Public Genotype & Phenotype Data"
echo "================================================"
echo ""


### Data has been removed from Cyverse. Currently unable to pull data from new location in Box. Download manually from curl locations below:
# curl -L -J "https://cornell.app.box.com/s/o7wtp1ewuqlw3dalr1920lungxnomnrg" -O ./data/geno/ZeaGBSv27.vcf.gz
# curl -L -J "https://cornell.app.box.com/s/yc95tqcgkyh4vtjwtbi3m3cfha47i748" -O ./data/pheno/Hung_etal_2012_PNAS.zip

# # Genotype Data
# mkdir -p data/geno

# echo "Step 1: Downloading genotype data (~2.60GB)..."

# #wget -nc https://de.cyverse.org/api/download?path=%2Fiplant%2Fhome%2Fshared%2Fpanzea%2Fgenotypes%2FGBS%2Fv27%2FZeaGBSv27_publicSamples_imputedV5_AGPv4-181023.vcf.gz -O ./data/geno/ZeaGBSv27.vcf.gz

# if [ ! -f ./data/geno/ZeaGBSv27.vcf.gz ]; then
#     echo "Error: Genotype file download failed." >&2
#     exit 1
# fi

# # Phenotype Data
# mkdir -p data/pheno

# echo ""
# echo "Step 2: Downloading phenotype data (~4.7MB)..."

# #wget -nc https://de.cyverse.org/api/download?path=%2Fiplant%2Fhome%2Fshared%2Fpanzea%2Fphenotypes%2FHung_etal_2012_PNAS_data-120523.zip -O ./data/pheno/Hung_etal_2012_PNAS.zip

# if [ ! -f ./data/pheno/Hung_etal_2012_PNAS.zip ]; then
#     echo "Error: Phenotype file download failed." >&2
#     exit 1
# fi

# unzip -q ./data/pheno/Hung_etal_2012_PNAS.zip -d ./data/pheno/Hung_etal_2012_PNAS

# rm ./data/pheno/Hung_etal_2012_PNAS.zip

echo "Step 3: Converting XLS to CSV..."
python << 'PYTHON_SCRIPT'

import pandas as pd

xls = pd.read_excel('./data/pheno/Hung_etal_2012_PNAS/Dataset S1.Photoperiod_paper_data_supplement_raw_data.xls')
xls.to_csv('./data/pheno/photoperiod_phenotypes.csv', index=False)
PYTHON_SCRIPT

if [ ! -f ./data/pheno/photoperiod_phenotypes.csv ]; then
    echo "Error: CSV conversion failed." >&2
    exit 1
fi

iconv -f ISO-8859-1 -t UTF-8 ./data/pheno/photoperiod_phenotypes.csv > ./data/pheno/photoperiod_phenotypes.csv
iconv -c -f UTF-8 -t ASCII//TRANSLIT -c ./data/pheno/photoperiod_phenotypes.csv > ./data/pheno/photoperiod_phenotypes.csv

echo ""
echo "================================================"
echo "Data download complete."
echo "You can now run: nextflow run main.nf"
echo "================================================"