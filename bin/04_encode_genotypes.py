import os
import sys
import warnings
from pathlib import Path
import argparse

# Add project root to Python path to enable imports from lib/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from lib.encoding_data_fns import Make_Embeddings, PhenotypeData, EncodingConfig

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--Training", dest="train_type", action="store_const", const="Training", help="Set train_type to 'Train'")
parser.add_argument("--Validation", dest="train_type", action="store_const", const="Validation", help="Set train_type to 'Validate'")
parser.add_argument("--Testing", dest="train_type", action="store_const", const="Testing", help="Set train_type to 'Test'")
parser.set_defaults(train_type="Training")
parser.add_argument("-o", "--output_folder", help="Full path to the output folder, format like: './Output_all/train1'")
parser.add_argument("--input_hmp_file", type=str, required=True, help="Full path to the genotype file in HapMap format")

parser.add_argument("-f", "--pheno_file", default=None, help="Full path to the phenotype file in CSV format, format like: '.results/data/phenotype_data.csv'")
parser.add_argument("-g", "--geno_col", default="geno", help="The name of the genotype column in the phenotype file (default: 'geno' - standard name)")
parser.add_argument("-p", "--pheno_col", default="trait_value", help="The name of the phenotype column in the phenotype file (default: 'trait_value' - standard name)")
parser.add_argument("-e", "--env_col", default="env", help="The name of the environment column in the phenotype file (default: 'env' - standard name)")

parser.add_argument("--envs_hold_out", action="store_true", default=False, help="If you held out an environment for validation or testing during data splitted. If True, an env_dict_file.csv needs to be generated which contains all environments. (default: False)")
parser.add_argument("-d", "--env_dict", default=None, help="Full path to the environment dictionary file. Required if using --env-hold-out flag. Column 1 contains unique environment col names and column 2 contains 1..n assignments")

parser.add_argument("--use_config", action="store_true", default=False, help="Use existing encoding configuration file. Encoding config file is automatically saved from training.")
parser.add_argument("--encoding_window_size", type=int, default=10, help="window size for local window kernel methods and batch conversion of dosage snps")
parser.add_argument("--shift", type=int, default=0, help="How many SNPs each window overlaps. Default of 0 results in non-overlapping windows")
parser.add_argument("--encoding_mode", type=str, default=None, help="Encoding approach. Options: 'kmeans-cosine', 'nystroem-KPCA', 'dosage'")

# nystroem-KPCA args
parser.add_argument("--nSubsample", type=int, default=None, help="Used for local window kernel methods. Number of individuals to include when random sampling individuals to construct the Nystroem kernel estimation.")
parser.add_argument("--kernel_type", type=str, default=None, help="Kernel type used for nystroem-KPCA encoding methods. See sklearn.kernel_approximation.Nystroem for more information. Options: 'rbf', 'cosine', 'sigmoid', 'polynomial")
parser.add_argument("--gamma", type=float, default=None, help="Gamma parameter for select kernels within the nystroem-KPCA encoding method. See sklearn.kernel_approximation.Nystroem for more information. Default option interpreted differnetly based on method.")



args = parser.parse_args()

hmp_metadata_names = ["rs#", "alleles", "chrom", "pos", "strand", "assembly#", "center", "protLSID", "assayLSID", "panelLSID", "QCcode"]

pheno_data_file = args.pheno_file
if not Path(pheno_data_file).exists():
    pheno_data_file = f'{args.output_folder}/01_Phenotype_Data_{args.train_type}.csv'
    if not os.path.exists(pheno_data_file):
        # Check parent directory for split files
        parent_dir = Path(args.pheno_file).parent
        split_files = list(parent_dir.glob(f'*_{args.train_type}.csv'))
        if split_files:
            pheno_data_file = str(split_files[0])
        else:
            warnings.warn(f"Phenotype data file location does not match expected: {args.pheno_file}. \n Please run 02_split_phenotypes.py first.")

get_phenotypes = PhenotypeData(geno_column_name=args.geno_col, pheno_column_names=[args.pheno_col], env_column_name=args.env_col, phenotype_file=pheno_data_file)

pheno_data = get_phenotypes.generate_phenotype_data(output_folder=args.output_folder, train_type=args.train_type, unseen_environments=args.envs_hold_out)

if args.use_config:
    encoding_config = EncodingConfig.load(f"{args.output_folder}/encoding_config.yaml")
else:
    if args.train_type == "Training":
        encoding_config = EncodingConfig(
            encoding_window_size=args.encoding_window_size,
            shift=args.shift,
            nSubsample=args.nSubsample,
            kernel_type=args.kernel_type,
            gamma=args.gamma,
            encoding_mode=args.encoding_mode,
            encoding_length=None # will be filled in later with the actual encoding size
        )
        encoding_config.save(f"{args.output_folder}/encoding_config.yaml")
    else:
        value_error_message = f"Encoding configuration file not found: {args.output_folder}/encoding_config.yaml. \n This file is required for validation and testing sets. Please run training first to create it."
        raise ValueError(value_error_message)

print('start emb ->')

mk_emb = Make_Embeddings(input_file=args.input_hmp_file, hmp_metadata_column_names=hmp_metadata_names, encoding_config=encoding_config, output_dir=args.output_folder)

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
