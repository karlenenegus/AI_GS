import os
import warnings
import argparse
from lib.encoding_data_fns import Make_Embeddings, PhenotypeData, EncodingConfig

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--Training", dest="train_type", action="store_const", const="Training", help="Set train_type to 'Train'")
parser.add_argument("--Validation", dest="train_type", action="store_const", const="Validation", help="Set train_type to 'Validate'")
parser.add_argument("--Testing", dest="train_type", action="store_const", const="Testing", help="Set train_type to 'Test'")
parser.set_defaults(train_type="Training")
parser.add_argument("-o", "--output_folder", help="Full path to the output folder, format like: './Output_all/train1'")
parser.add_argument("-h", "--input_hmp_file", type=str, required=True, help="Full path to the genotype file in HapMap format")

parser.add_argument("-f", "--pheno_file", default=None, help="Optional. Full path to the phenotype file in CSV format, format like: './Data/PHENO/phenotype_data.csv'")
parser.add_argument("-g", "--geno_col", default="geno", help="The name of the genotype column in the phenotype file (default: 'geno' - standard name)")
parser.add_argument("-p", "--pheno_col", default="trait_value", help="The name of the phenotype column in the phenotype file (default: 'trait_value' - standard name)")
parser.add_argument("-e", "--env_col", default="env", help="The name of the environment column in the phenotype file (default: 'env' - standard name)")


parser.add_argument("--encoding_window_size", type=int, default=10, help="window size for local window kernel methods and batch conversion of dosage snps")
parser.add_argument("--shift", type=int, default=0, help="How many SNPs each window overlaps. Default of 0 results in non-overlapping windows")
parser.add_argument("--nSubsample", type=int, default=100, help="Used for local window kernel methods. Number of individuals to include when random sampling individuals to construct the Nystroem kernel estimation.")
parser.add_argument("--kernel_type", type=str, default='rbf', help="Kernel type used for nystroem-KPCA encoding methods. See sklearn.kernel_approximation.Nystroem for more information. Options: 'rbf', 'cosine', 'sigmoid', 'polynomial")
parser.add_argument("--gamma", type=float, default=None, help="Gamma parameter for select kernels within the nystroem-KPCA encoding method. See sklearn.kernel_approximation.Nystroem for more information. Default option interpreted differnetly based on method.")
parser.add_argument("--encoding_mode", type=str, default=0, help="Encoding approach. Options: 'kmeans-cosine', 'nystroem-KPCA', 'dosage'")


args = parser.parse_args()

hmp_metadata_names = ["rs#", "alleles", "chrom", "pos", "strand", "assembly#", "center", "protLSID", "assayLSID", "panelLSID", "QCcode"]

pheno_data_file = f'{args.output_folder}/01_Phenotype_Data_{args.train_type}.csv'
if not os.path.exists(pheno_data_file):
    warnings.warn(f"Phenotype data file location does not match expected: {pheno_data_file}. \n Please run 02_split_phenotypes.py first.")

get_phenotypes = PhenotypeData(geno_column_name=args.geno_col, pheno_column_names=[args.pheno_col], env_column_name=args.env_col, phenotype_file=pheno_data_file)

pheno_data = get_phenotypes.generate_phenotype_data(output_folder=args.output_folder, train_type=args.train_type, unseen_environments=True)

if args.train_type == "Training":
    encoding_config = EncodingConfig(
        encoding_window_size=9,
        shift=0,
        nSubsample=0,
        kernel_type="cosine",
        gamma=None,
        encoding_mode="dosage",
        encoding_length=None # will be filled in later with the actual encoding size
    )
    encoding_config.save(f"{args.output_folder}/encoding_config.yaml")
else:
    encoding_config = EncodingConfig.load(f"{args.output_folder}/encoding_config.yaml")

print('start emb ->')

mk_emb = Make_Embeddings(input_file=args.hmp_file, hmp_metadata_column_names=hmp_metadata_names, encoding_config=encoding_config, output_dir=args.output_folder)

if args.train_type == "Training":
    mk_emb.training_mode(file_prefix=args.train_type, ind_names_to_keep=pheno_data[args.geno_col])
else:
    mk_emb.inference_mode(file_prefix=args.train_type, ind_names_to_keep=pheno_data[args.geno_col])

print('finish save encodings')


mk_emb.stream_encodings_to_parquet(
    phenotype_data=pheno_data,
    batch_size=100,
    file_prefix=args.train_type
)
print('finish pheno -> pheno + embeddings')
