"""
Functions for encoding genotype and phenotype data for genomic prediction.
"""

import os
import json
import re
from itertools import chain
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from warnings import warn
import pyarrow.parquet as pq
import pyarrow as pa
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import normalize
from sklearn.cluster import kmeans_plusplus

from helper_fns import save_json, load_json
# Note: LDWindowAnalyzer import should be added when available
# from Dependencies.ld_analyzer import LDWindowAnalyzer

def make_dict(data, column):
    obj = np.unique(data.loc[:, column])
    n_obj = obj.shape[0] + 1
    return dict(zip(obj, range(1, n_obj)))


@dataclass
class EncodingConfig:
    encoding_window_size: int = 9
    shift: int = 0
    nSubsample: int = 0
    kernel_type: str = "cosine"
    gamma: Optional[float] = None
    encoding_mode: str = "dosage"
    encoding_length: Optional[int] = None  # Filled later

    def update_encoding_length(self, length: int):
        """Update encoding_length after building encodings."""
        self.encoding_length = int(length)

    def _to_dict(self) -> Dict[str, Any]:
        """Convert the config into a plain dictionary."""
        return asdict(self)

    def save(self, path: str):
        """
        Save the configuration as a YAML file.
        Creates directories if necessary.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self._to_dict(), f, sort_keys=False)
        print(f"EncodingConfig saved to {path}")

    @classmethod
    def load(cls, path: str) -> "EncodingConfig":
        """
        Load configuration from a YAML file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


class PhenotypeData:
    def __init__(self, geno_column_name: str, pheno_column_names: List[str], env_column_name: str, phenotype_file: str, wide_or_long_format: str = "wide"):
        """
        Initializes the PhenotypeData class with the provided configuration.

        Args:
            geno_column_name (str): The name of the genotype column in the phenotype file.
            pheno_column_names (List[str]): The names of the phenotype columns in the phenotype file.
            env_column_name (str): The name of the environment column in the phenotype file.
            phenotype_file (str): The path to the phenotype file.
            wide_or_long_format (str): The format of the phenotype file, "wide" or "long".
        """
        self.geno_column_name = geno_column_name
        self.pheno_column_names = pheno_column_names
        self.env_column_name = env_column_name
        self.phenotypes_long = None

        self.wide_phenotype_data = True if wide_or_long_format == "wide" else False

        self.pheno_data = pd.read_csv(phenotype_file)
        self.pheno_data = self.pheno_data.replace('nan', np.nan)

        if self.wide_phenotype_data:
            self._subset_pheno_columns()
            self._wide_to_long()
        else:
            self.phenotypes_long = self.pheno_data

    def _subset_pheno_columns(self):
        """
        Subsets the phenotype file to the relevant columns. Used for wide format phenotype files.
        """

        if type(self.pheno_column_names) is str:
            self.pheno_column_names = [self.pheno_column_names]
        
        columns_needed = [self.env_column_name, 
                          self.geno_column_name] + \
                          self.pheno_column_names
        missing_cols = [col for col in columns_needed if col not in self.pheno_data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in phenotype file: {', '.join(missing_cols)}. Please check your phenotype file.")
        
        self.pheno_data = self.pheno_data[columns_needed]

    def generate_phenotype_data(self, output_folder, train_type, unseen_environments):
        """
        Convert environment and trait columns to numeric using dictionaries.
        
        Creates dictionaries during training, loads them during validation/testing.
        """
        os.makedirs(f'{output_folder}/encoding_keys/', exist_ok=True)
        env_dict_path = f'{output_folder}/encoding_keys/env_dict_file.csv'
        trait_dict_path = f'{output_folder}/encoding_keys/trait_dict_file.csv'
        
        # During training, create dictionaries
        if train_type == "Training":
            # Always create trait dictionary during training
            trait_dict = make_dict(self.phenotypes_long, 'variable')
            pd.DataFrame.from_dict(trait_dict, orient='index').to_csv(
                trait_dict_path, header=False
            )
            
            if not unseen_environments:
                # Create environment dictionary if not expecting environments not in training set
                env_dict = make_dict(self.phenotypes_long, self.env_column_name)
                pd.DataFrame.from_dict(env_dict, orient='index').to_csv(
                    env_dict_path, header=False
                )
            else:
                # If expecting environments not in training set, check if env_dict already exists
                if os.path.exists(env_dict_path):
                    print(
                        "Using existing env_dict_file.csv. If this is unexpected set "
                        "'unseen_environments = False'. Expecting new environments in testing set."
                    )
                else:
                    raise ValueError(
                        f'{env_dict_path} does not exist. Pre-generated file is required '
                        'for processing new environments in inference sets.'
                    )
        
        # During validation/testing, dictionaries must already exist
        else:
            if not os.path.exists(trait_dict_path):
                raise FileNotFoundError(
                    f'{trait_dict_path} does not exist. Must run training first to create dictionaries.'
                )
            if not os.path.exists(env_dict_path):
                raise FileNotFoundError(
                    f'{env_dict_path} does not exist. Must run training first to create dictionaries.'
                )
        
        # Load dictionaries (either just created or from existing files)
        env_df = pd.read_csv(env_dict_path, header=None, index_col=0)
        env_dict = env_df.iloc[:, 0].to_dict()
        
        trait_df = pd.read_csv(trait_dict_path, header=None, index_col=0)
        trait_dict = trait_df.iloc[:, 0].to_dict()
        
        # Convert columns to numeric
        phenotypes_numeric = self.phenotypes_long.copy()
        with pd.option_context('future.no_silent_downcasting', True):
            phenotypes_numeric[self.env_column_name] = phenotypes_numeric[self.env_column_name].replace(env_dict)
            
            if phenotypes_numeric[self.env_column_name].isna().any():
                unmatched_env = phenotypes_numeric[phenotypes_numeric[self.env_column_name].isna()]
                warn(
                    f"Warning: Some entries in '{self.env_column_name}' did not match any value in env_dict. "
                    f"Unmatched entries:\n{unmatched_env}"
                )
            
            phenotypes_numeric['variable'] = phenotypes_numeric['variable'].replace(trait_dict)
            
            if phenotypes_numeric['variable'].isna().any():
                unmatched_trait = phenotypes_numeric[phenotypes_numeric['variable'].isna()]
                warn(
                    f"Warning: Some entries in 'variable' did not match any value in trait_dict. "
                    f"Unmatched entries:\n{unmatched_trait}"
                )
        
        return phenotypes_numeric
    
    def _wide_to_long(self):
        """
        Processes a wide format phenotype file and returns the long-form DataFrame with the ID as the index.
        """

        self.pheno_data["ID"] = self.pheno_data.index

        long_pheno = pd.melt(
            self.pheno_data,
            id_vars=["ID", self.geno_column_name, self.env_column_name],
            value_vars=self.pheno_column_names
        )
        new_pheno = long_pheno[long_pheno.loc[:, 'value'].notna()]
        new_pheno.index = new_pheno[self.geno_column_name]
        
        self.phenotypes_long = new_pheno


class Make_Encodings():
    def __init__(self, input_file, hmp_metadata_column_names: Optional[List[str]] = None, encoding_config=None, shift=0, kernel_type=None, gamma=None, encoding_window_size_in = 10, encoding_window_size_out=None, nSubsample=0, encoding_mode='dosage', output_dir="./Output_all/train_tmp", std_dict_file="standardization_dict.json"):        
        
        self.encoding_config = encoding_config
        self.input_file = input_file

        if encoding_config is None:
            self.kernel_type = kernel_type
            self.gamma = gamma
            self.nSubsample = nSubsample
            self.encoding_mode = encoding_mode
            self.shift = shift
            self.encoding_window_size_in = encoding_window_size_in
            self.encoding_window_size_out = encoding_window_size_out
        else:
            # encoding_config is a dataclass, access as attributes not dictionary
            self.kernel_type = encoding_config.kernel_type
            self.gamma = encoding_config.gamma
            self.nSubsample = encoding_config.nSubsample
            self.encoding_mode = encoding_config.encoding_mode
            self.shift = encoding_config.shift
            self.encoding_window_size_in = encoding_config.encoding_window_size_in
            self.encoding_window_size_out = encoding_config.encoding_window_size_out
            
        # if self.encoding_mode == "dosage":
        #     #todo id required parameters send warning if none
        #     ...
        # elif self.encoding_mode == "landmark-cosine":
        #     #todo id required parameters send warning if none
        #     ...
        # elif self.encoding_mode == "nystroem-KPCA":
        #     #todo id required parameters send warning if none
        #     ...
        # else:
        #     raise ValueError("Encoding method {self.encoding_mode} is not recognized. Use 'dosage', 'nystroem-KPCA', or 'landmark-cosine' instead.")

        self.output_dir = output_dir
        self.std_dict_file = std_dict_file
        self.snp_index = None
        self.chromosome_breaks = None
        self.encoding_data = None
        
        if self.kernel_type == 'cosine':
            self.gamma = None

        self.snp_data = self._HapMap_to_DataFrame(data_file=self.input_file, metadata_columns=hmp_metadata_column_names)
        
        # Update nSNPs from the shape of snp_data (number of SNP columns)
        self.nSNPs = self.snp_data.shape[1]
    
    def _HapMap_to_DataFrame(
        self, 
        data_file: str = './Data/Geno.hmp', 
        metadata_columns: Optional[List[str]] = None
    ) -> None:
        """
        Processes the HapMap input data file and creates a DataFrame with updated column names.
        
        The DataFrame is stored in self.snp_data with columns named as "chrom_pos".

        Args:
            data_file: Path to the HapMap input data file.
            metadata_columns: List of metadata column names. Defaults to standard HapMap metadata.
        """
        if metadata_columns is None:
            metadata_columns = [
                'rs#', 'alleles', 'chrom', 'pos', 'strand', 'assembly#', 
                'center', 'protLSID', 'assayLSID', 'panelLSID', 'QCcode'
            ]
        if not "chrom" in metadata_columns:
            raise ValueError("Invalid HapMap format: chrom column is required in metadata_columns")
        if not "pos" in metadata_columns:
            raise ValueError("Invalid HapMap format: pos column is required in metadata_columns")

        # Read metadata
        metadata = pd.read_csv(data_file, sep = '\t', dtype=str, usecols=metadata_columns)
        
        # Read the rest of the hmp data (excluding metadata columns)
        data = pd.read_csv(data_file, sep = '\t', dtype=str, usecols=lambda x: x not in metadata_columns).T
        
        if all([re.match("^0_", i) for i in data.index]):
            fix_index = [re.sub(":([^\t]+)", "", re.sub("^0_", "", i)) for i in data.index]
            data.index = fix_index
        else:
            fix_index = [re.sub(":([^\t]+)", "", i) for i in data.index]
            data.index = fix_index
        
        # Extract chromosome and position columns
        chromosomes = metadata["chrom"]
        positions = metadata["pos"]
        chromosome_shifts = metadata.index[metadata["chrom"] != metadata["chrom"].shift()].to_list()
        self.chromosome_breaks = chromosome_shifts + [len(metadata.index)]

        # Determine maximum lengths for zero-padding
        max_len_chr = chromosomes.astype(str).map(len).max()
        max_len_pos = positions.astype(str).map(len).max()
        
        # Zero-padding chromosomes and positions for names
        chr_long = chromosomes.astype(str).str.zfill(max_len_chr).tolist()
        pos_long = positions.astype(str).str.zfill(max_len_pos).tolist()

        # Combine chromosome and position to create new column names
        names = np.array(chr_long, dtype = object) + "_" + np.array(pos_long, dtype = object)
        data.columns = names
        return data
        
    def _convert_SNP2Numeric(self, input_column: pd.Series, encoding_mode: str):
        """Convert nucleotide SNP data to allele dosage format and standardize."""
        
        geno_counts = input_column.value_counts()
        geno_counts = geno_counts.drop('NN', errors='ignore')
        
        if geno_counts.empty:
            standardized_col = pd.Series(0, index=input_column.index)
            return standardized_col, {input_column.name: {'NN': 0}}
        
        indices = [g for g in geno_counts.index if len(g) % 2 == 0]
        hom_indices = [g for g in indices if g[:len(g)//2] == g[len(g)//2:]]
        het_indices = [g for g in indices if g[:len(g)//2] != g[len(g)//2:]]
        
        if len(het_indices) > 1:
            het_first = het_indices[0]
            het_reverse = het_first[len(het_first)//2:] + het_first[:len(het_first)//2]
            warn(
                f"Warning: More than 1 heterozygous genotypes {het_indices} found for SNP {input_column.name}. "
                f"Only the first ({het_first}) and reverse of the first ({het_reverse}) will be used."
            )
            
        if len(hom_indices) > 2:
            warn(f"Warning: More than 2 homozygous genotypes found for SNdP {input_column.name}. Only the first two will be used.")
        
        irregular_indicies = [g for g in geno_counts.index if len(g) % 2 != 0]
        
        if not het_indices and (len(irregular_indicies)==1):
            het_indices = irregular_indicies
            print(f"Using genotype {irregular_indicies} as heterozygous for SNP {input_column.name}.")
        
        #dosage_dict = {'NN': np.nan}
        dosage_dict = {'NN': -1}
        
        if len(hom_indices) == 1:
            major = hom_indices[0]
            dosage_dict[major] = 0
            het = het_indices[0] if len(het_indices) >= 1 else None
            if het: 
                het2 = het[len(het)//2:] + het[:len(het)//2]
                dosage_dict[het] = 1
                dosage_dict[het2] = 1
        elif len(hom_indices) >= 2:
            major, minor = hom_indices[:2]
            dosage_dict[major] = 0
            dosage_dict[minor] = 2
            het = het_indices[0] if len(het_indices) >= 1 else None
            if het:
                dosage_dict[het] = 1
                
        numeric_column = input_column.map(dosage_dict).astype(float)

        # return numeric_column, {input_column.name: dosage_dict} 

        # Option 1
        if encoding_mode == "dosage":
            std_column = pd.to_numeric(numeric_column, errors='coerce').fillna(-1)
            std_dict = dosage_dict

        else: # Option 2
            # Standardize the dosage values    
            input_mean = numeric_column.mean()
            input_std = numeric_column.std() if numeric_column.std() != 0 else 1
            dosage_dict["NN"] = input_mean

            std_dict = {
                k: ((int(v) - input_mean) / input_std if not np.isnan(v) else input_mean)
                for k, v in dosage_dict.items()
            }
    
        with pd.option_context('future.no_silent_downcasting', True):
            std_column = input_column.replace(std_dict)
            
            # Check for any remaining string values (shouldn't happen, but useful for debugging)
            string_values = std_column[std_column.map(type) == str]
            if not string_values.empty: 
                warn(
                    f"Warning: Some values in SNP {input_column.name} could not be converted. "
                    f"String values: {string_values.values}"
                )

        std_column = pd.to_numeric(std_column, errors='coerce').fillna(0)

        return std_column, {input_column.name: std_dict} 
    
    def _generate_window_groups(self, i: int, end: int, chromosome_breaks: List[int]) -> Tuple[int, int]:
        """
        Generate groups based on chromosome boundaries.
        
        Args:
            i: Index in chromosome_breaks
            end: End position
            chromosome_breaks: List of chromosome break positions
            
        Returns:
            Tuple of (start, end) positions
        """
        return (chromosome_breaks[i - 1], end)
    
    def _generate_window_indices(
        self, 
        window_size: int, 
        shift: int, 
        chromosome_start: int, 
        chromosome_end: int
    ) -> List[Tuple[int, int]]:
        """
        Generate chunk indices for sliding window analysis.
        
        Args:
            window_size: Number of SNPs per window
            shift: Step size between windows
            chromosome_start: Starting index of the chromosome
            chromosome_end: Ending index of the chromosome
            
        Returns:
            List of (start, end) tuples for each window
        """
        step = window_size - shift
        chromosome_size = chromosome_end - chromosome_start
        n_windows = int(np.ceil(chromosome_size / step))
        
        ranges_list = []
        
        for i in range(n_windows):
            start = step * i + chromosome_start

            if start >= chromosome_end:
                break

            ini = max(chromosome_start, start)
            fin = min(chromosome_end, ini + window_size)
            
            start2 = step * (i + 1) + chromosome_start
            if start2 + window_size > chromosome_end:
                ranges_list.append((ini, chromosome_end))
                break
            else: 
                ranges_list.append((ini, fin))
                
        return ranges_list

    def _encoding_within_window(
        self, 
        window: pd.DataFrame, 
        nSubsample: int,
        inference: bool, 
        output_folder: str, 
        landmark_idxs: Optional[np.array] = None,
        size_per_window: Optional[int] = None, 
        kernel_type: Optional[str] = None, 
        gamma: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply encoding methods within a window of SNPs.
        
        Args:
            window: Subset of SNP data representing the window. Shape = [individuals, snp_window]
            nSubsample: Number of individuals for Nystroem approximation
            size_per_window: Number of encodings per window. If None for encoding_type = "nystroem-kpca" PCs = 95% variance explained will be fit.
            inference: If True, load saved transforms; if False, fit new transforms
            output_folder: Directory to save/load transforms
            kernel_type: Type of kernel ('cosine', 'rbf', etc.)
            gamma: Gamma parameter for radial basis function kernel (None for cosine)
        
        Returns:
            Array of encodings with shape [individuals, encoding_size_per_window]
        """
        if not isinstance(window, pd.DataFrame):
            window = pd.DataFrame(window)
            
        st = window.columns[0]
        ed = window.columns[-1]      
        
        if self.encoding_mode == 'dosage':
            x = window
            if not inference: 
                selected_features = x.columns.tolist()
                X_selected = x[selected_features]
                
                if not os.path.exists(f'{output_folder}/encoding_keys/selected_features/'):
                    os.makedirs(f'{output_folder}/encoding_keys/selected_features/')
                    
                joblib.dump(selected_features, f'{output_folder}/encoding_keys/selected_features/selected_features_window{st}-{ed}.pkl')
            else:
                selected_features_path = f'{output_folder}/encoding_keys/selected_features/selected_features_window{st}-{ed}.pkl'
                if not os.path.exists(selected_features_path):
                    raise FileNotFoundError(
                        f"'{selected_features_path}' does not exist! "
                        "Your inference data does not match the training data."
                    )
                
                selected_features = joblib.load(selected_features_path)
                X_selected = x[selected_features]
            return X_selected

        elif self.encoding_mode == 'nystroem-KPCA':
            x = normalize(window, norm='l2', axis=1)

            if not inference:          
                if gamma is None:
                    nystroem = Nystroem(kernel=kernel_type, n_components=nSubsample, random_state=42)
                else:
                    nystroem = Nystroem(kernel=kernel_type, n_components=nSubsample, random_state=42, gamma=gamma)
                    
                # Don't use low variance features as landmark features
                filter_variance = VarianceThreshold(threshold=1e-6)
                
                x_unique = np.unique(x, axis=0)
                x_uq_filtered = filter_variance.fit_transform(x_unique)
                x_uq_transformed = nystroem.fit_transform(x_uq_filtered)
                
                if size_per_window:
                    n_components_pca = size_per_window
                else:
                    n_components_pca = None
                
                # Fit final PCA with specified number of components
                pca2 = PCA(n_components=n_components_pca)
                pca2.fit(x_uq_transformed)
                
                x_filtered = filter_variance.transform(x)
                x_transformed = nystroem.transform(x_filtered)
                x_pca = pca2.transform(x_transformed)
            
                # Create output directories
                for dir_name in ['nystroem', 'pca', 'filter_variance']:
                    dir_path = f'{output_folder}/encoding_keys/{dir_name}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                
                # Calculate variance explained statistics
                var_exp_1 = pca2.explained_variance_ratio_[:4]
                var_exp_2 = pca2.explained_variance_ratio_[-1]
                cumulative_variance = np.cumsum(pca2.explained_variance_ratio_)
                total_variance = cumulative_variance[-1]
    
                print(
                    f"Variance explained by encoding {st}-{ed}: "
                    f"PC's 1-4 {var_exp_1} | PC {n_components_pca} {var_exp_2} | "
                    f"N components for 0.95: {n_components_pca}",
                    flush=True
                )
                
                # Save transforms
                joblib.dump(nystroem, f'{output_folder}/encoding_keys/nystroem/nystroem_window{st}-{ed}.pkl')
                joblib.dump(pca2, f'{output_folder}/encoding_keys/pca/pca_window{st}-{ed}.pkl')
                joblib.dump(filter_variance, f'{output_folder}/encoding_keys/filter_variance/filter_variance_window{st}-{ed}.pkl')

            else:
                nystroem = joblib.load(f'{output_folder}/encoding_keys/nystroem/nystroem_window{st}-{ed}.pkl')
                pca2 = joblib.load(f'{output_folder}/encoding_keys/pca/pca_window{st}-{ed}.pkl')
                filter_variance = joblib.load(f'{output_folder}/encoding_keys/filter_variance/filter_variance_window{st}-{ed}.pkl')
                
                x_filtered = filter_variance.transform(x)
                x_transformed = nystroem.transform(x_filtered)
                x_pca = pca2.transform(x_transformed)
            return x_pca
                
        elif self.encoding_mode == 'kmeans-cosine':
            x = normalize(window, norm='l2', axis=1)

            if not inference:
                x_unique = np.unique(x, axis=0)
                landmarks = x_unique[landmark_idxs]
                x_transformed = x @ landmarks.T

                if not os.path.exists(f'{output_folder}/encoding_keys/landmarks/'):
                    os.makedirs(f'{output_folder}/encoding_keys/landmarks/')
                
                joblib.dump(landmarks, f'{output_folder}/encoding_keys/landmarks/landmarks_window{st}-{ed}.pkl')

            else:
                landmarks_path = f'{output_folder}/encoding_keys/landmarks/landmarks_window{st}-{ed}.pkl'
                if not os.path.exists(landmarks_path):
                    raise FileNotFoundError(
                        f"'{landmarks_path}' does not exist! "
                        "Your inference data does not match the training data."
                    )
                
                landmarks = joblib.load(landmarks_path)
                x_transformed = x @ landmarks.T
                
            return x_transformed

    def _apply_standardization(self, data: pd.Series, std_dict: dict):
        """Standardize nucleotide data using precomputed scaled dosage values."""
        snp_dict = std_dict.get(data.name)
        return data.map(lambda x: snp_dict.get(x, 0))
    
    def _filter_SNP_data(self, snp_data: pd.DataFrame, pheno_data_names: pd.Series):
        """Filters the full SNP dataset based on individual names provided from the phenotype dataset. 
        
        Args:
            snp_data (pd.DataFrame): DataFrame generated by InputData Class call. 
            pheno_data_names (pd.Series): Genotype name column subsetted from the phenotype dataset.

        Returns:
            pd.DataFrame: DataFrame [subset_ind, SNPs]."""
            
        snp_data_mask = snp_data.index.isin(pheno_data_names)
        
        if not snp_data_mask.any():
            raise NameError(f"Names in genotype file and phenotype file do not match. Check name formatting in files assigned to 'geno_data_file' and 'pheno_data_file'.") 
            
        return snp_data[snp_data_mask]
    
    def _get_index(self):
        return self.snp_index
    
    def _map_indices_to_windows(self, ranges):
        """
        Given a list of (start, end) tuples, return a dict mapping row index to window ID.
        """
        index_to_window = {
            idx: f'window_{i}' 
            for i, (start, end) in enumerate(ranges)
            for idx in range(start, end)
        }
        return index_to_window


    def training_mode(self, file_prefix: str, ind_names_to_keep: pd.Series):
        """
        Generate encodings for the training data.

        Args:
            file_prefix (str): Prefix for the output file name
            ind_names_to_keep (pd.Series): Names of the individuals to keep

        Returns:
            None
        """
        snp_data = self._filter_SNP_data(self.snp_data, ind_names_to_keep)
        self.snp_index = snp_data.index
        
        std_data, std_dicts = zip(*[self._convert_SNP2Numeric(snp_data.iloc[:, i], self.encoding_mode) for i in range(snp_data.shape[1]) if len(snp_data.iloc[:, i].value_counts().drop("NN", errors="ignore"))>1]) 
        
        full_std_dict = {k: v for d in std_dicts for k, v in d.items()}
        save_json(full_std_dict, f'{self.output_dir}/encoding_keys/{self.std_dict_file}')
        
        # TODO: Uncomment when LDWindowAnalyzer is available
        # analyzer = LDWindowAnalyzer(
        #     hmp_file=self.input_file,
        #     std_dict_path=f'{self.output_dir}/{self.std_dict_file}',
        #     ind_names=pheno_data_names,
        #     window_size=100000,
        #     step_size=50000
        # )
        # _ = analyzer.compute_ld_windows()
        # analyzer.save_assignments(f'{self.output_dir}/ld_window_assignments.npy')
        
        std_data = pd.DataFrame(std_data)
        
        step = self.encoding_window_size_in - self.shift
        p = std_data.shape[0] # number of SNPs
        n_windows=int(np.ceil(p/(step)))
        
        corrected_n_windows = sum(1 for i in range(n_windows) if step * i + self.encoding_window_size_in <= p)

        if corrected_n_windows < 1:
            raise ValueError("Zero windows generated. Parameters: 'nSNPs', 'shift', and/or 'nMin' need to be adjusted smaller for this dataset.")
        elif 1 < corrected_n_windows < 100:
            warn(f"Only {corrected_n_windows} windows were generated with the specified parameters. Check that 'nSNPs' and/or 'shift', are appropriate.")

        n_individuals = std_data.shape[1]

        groups = [self._generate_window_groups(i, end, self.chromosome_breaks) for i, end in enumerate(self.chromosome_breaks) if i != 0]
        
        window_range = [self._generate_window_indices(window_size=self.encoding_window_size_in, shift=self.shift, chromosome_start=start, chromosome_end=end) for start, end in groups]
        
        window_range_list = list(chain.from_iterable(window_range))
        
        window_names_dict  = {
            f'window_{w}': [list(full_std_dict.keys())[i] for i in range(start, end)]
            for w, (start, end) in enumerate(window_range_list)
        }
        save_json(window_names_dict, f"{self.output_dir}/encoding_keys/SNPsPerWindow.json")
        
        landmark_idxs = None
        
        if self.encoding_mode == "dosage":
            encodings = np.full((n_individuals, len(window_names_dict), self.encoding_window_size_in*2), np.nan)
        if self.encoding_mode == "nystroem-KPCA":
            encodings = np.full((n_individuals, len(window_names_dict), self.encoding_window_size_in*2), np.nan)
        if self.encoding_mode == "landmark-cosine":
            landmark_idxs = np.random.choice(n_individuals, self.nSubsample, replace=False)
            encodings = np.full((n_individuals, len(window_names_dict), self.nSubsample), np.nan)
        else:
            raise ValueError("Encoding method {self.encoding_mode} is not recognized. Use 'dosage', 'nystroem-KPCA', or 'landmark-cosine' instead.")

        window_sizes = []
        for i, (key, value) in enumerate(window_names_dict.items()):
            window = std_data.loc[value]
            window_output = self._encoding_within_window(
                window.T, 
                nSubsample=self.nSubsample, 
                size_per_window=self.encoding_window_size_out, 
                kernel_type=self.kernel_type, 
                gamma=self.gamma, 
                inference=False, 
                output_folder=self.output_dir,
                landmark_idxs=landmark_idxs
            )
            size = window_output.shape[1]
            window_sizes[i] = size
            encodings[:, i, :size] = window_output
            
        if self.encoding_mode == "dosage":
            n_individuals, n_windows, enc_size = encodings.shape
            encodings = encodings.reshape(n_individuals, n_windows * enc_size)
            total_size = sum(window_sizes)
            encodings = encodings[:, :total_size]
            
        elif self.encoding_mode == "nystroem-KPCA":
            encodings = encodings[:, :, :min(window_sizes)]
            
        elif self.encoding_mode == "landmark-cosine":
            n_individuals, n_windows, window_size = encodings.shape
            flat_encoding = encodings.reshape(n_individuals * n_windows, window_size)
            variance = np.var(flat_encoding, axis=0)

            # Step 3: Select top-k landmarks by variance
            top_k = self.encoding_window_size_out
            landmark_idx = np.argsort(variance)[-top_k:]

            # Step 4: Keep only top-k landmarks in original shape
            encodings = encodings[:, :, landmark_idx]
            
        
        out_file_name = f"{self.output_dir}/Encodings_{file_prefix}_parquet/encodings.pkl"
        
        if not os.path.exists(f"{self.output_dir}/Encodings_{file_prefix}_parquet"):  
            os.makedirs(f"{self.output_dir}/Encodings_{file_prefix}_parquet")
        
        joblib.dump(encodings, out_file_name)
        self.encoding_data = encodings

        config_path = os.path.join(self.output_dir, "encoding_config.yaml")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file) or {}

        config["encoding_length"] = int(self.encoding_data.shape[1])

        with open(config_path, "w") as file:
            yaml.safe_dump(config, file, sort_keys=False)

        print(f'Encoding data saved to {out_file_name}', flush=True)
    
    def inference_mode(self, file_prefix: str, ind_names_to_keep: pd.Series): 
        """
        Generate encodings for the inference data.

        Args:
            file_prefix (str): Prefix for the output file name
            ind_names_to_keep (pd.Series): Names of the individuals to keep

        Returns:
            None
        """
        snp_data = self._filter_SNP_data(self.snp_data, ind_names_to_keep)
        self.snp_index = snp_data.index
        
        std_dict = load_json(f'{self.output_dir}/encoding_keys/{self.std_dict_file}')
        
        std_data = [self._apply_standardization(snp_data.iloc[:, i], std_dict) for i in range(snp_data.shape[1]) if snp_data.iloc[:, i].name in std_dict.keys()]
        
        std_data = pd.DataFrame(std_data)

        step = self.nSNPs - self.shift
        p = std_data.shape[0]  # number of SNPs
        n_windows = int(np.ceil(p / step))
        
        corrected_n_windows = sum(1 for i in range(n_windows) if step * i + self.nSNPs <= p)

        if corrected_n_windows < 1:
            raise ValueError("Zero L-RBF Chunks generated. Parameters: 'nSNPs', 'shift', and/or 'nMin' need to be adjusted smaller for this dataset.")
        elif 1 < corrected_n_windows < 100:
            warn(f"Only {corrected_n_windows} chunks were generated with the specified parameters. Check that 'nSNPs' and/or 'shift', are appropriate.")

        n_individuals = std_data.shape[1]
        
        #Parallelize this in the future
        window_names_dict = load_json(f"{self.output_dir}/encoding_keys/SNPsPerWindow.json")

        if self.encoding_mode == "dosage":
            total_size = len(std_dict.keys())
        else:
            total_size = len(window_names_dict) * self.encoding_window_size

        encodings = np.zeros((n_individuals, total_size))

        current_idx = 0
        for i, (key, value) in enumerate(window_names_dict.items()):
            window = std_data.loc[value]
            window_output = self._encoding_within_window(
                window.T, 
                nSubsample=self.nSubsample, 
                size_per_window=self.encoding_window_size, 
                kernel_type=self.kernel_type, 
                gamma=self.gamma, 
                inference=True, 
                output_folder=self.output_dir
            )
            
            if self.encoding_mode == "dosage":
                size = window_output.shape[1]
                encodings[:, current_idx:current_idx + size] = window_output
                current_idx += size
            else:
                encodings[:, i*self.encoding_window_size:(i+1)*self.encoding_window_size] = window_output

        if not os.path.exists(f"{self.output_dir}/Encodings_{file_prefix}_parquet"):
            os.makedirs(f"{self.output_dir}/Encodings_{file_prefix}_parquet")
        
        out_file_name = f"{self.output_dir}/Encodings_{file_prefix}_parquet/encodings.pkl"
        joblib.dump(encodings, out_file_name)
        self.encoding_data = encodings
        print(f'Encoding data saved to {out_file_name}', flush=True)

    def _data_chunker(self, data: pd.DataFrame, batch_size: int):
        shuffled_pos = np.random.permutation(len(data))
        return (data.iloc[shuffled_pos[pos:pos + batch_size], :] for pos in range(0, len(shuffled_pos), batch_size))

    def stream_encodings_to_parquet(self, phenotype_data: pd.DataFrame, batch_size: int = 10, file_prefix: str = "Training"):
        """
        Stream phenotype data with encodings to parquet files in batches.
        
        Args:
            phenotype_data: DataFrame with phenotype data. Should have columns:
                - genotype column (name varies)
                - environment column (name varies) 
                - value column (trait values)
                - variable column (trait names, if in long format)
            batch_size: Number of rows per batch
            file_prefix: Prefix for output directory (e.g., "Training", "Validation", "Testing")
        """
        for batch in self._data_chunker(phenotype_data, batch_size):
            # Get row indices in encodings that match phenotype chunk
            snp_index = self._get_index()
            indices_in_snp_index = snp_index.get_indexer(batch.index)
            mask = indices_in_snp_index != -1

            if not mask.any():
                continue

            batch_masked = batch[mask].copy()
            indices_masked = indices_in_snp_index[mask]
            encodings_masked = self.encoding_data[indices_masked]

            # Add encodings as a new column (as list)
            batch_masked["encodings"] = list(encodings_masked)

            # Use standard column names: geno, env, trait_value, trait_name
            # After convert2output: geno, env, trait_value, trait_name
            # From PhenotypeData.generate_phenotype_data: original column names + 'variable', 'value'
            # Check for standard names first, then fall back to alternatives
            if 'env' in batch_masked.columns:
                env_col = 'env'
            elif 'environment' in batch_masked.columns:
                env_col = 'environment'
            else:
                # Try to find environment column by checking common names
                env_candidates = [col for col in batch_masked.columns if 'env' in col.lower() or 'environment' in col.lower()]
                if env_candidates:
                    env_col = env_candidates[0]
                else:
                    raise ValueError("Could not find environment column in phenotype_data. Check if the environment column name is correct in the pheno_column_mapping.json file.")
            
            # Reset index to save individual ID as a column
            # Map to final output names: name, labels, trait_input_ids, env_input_ids
            rename_dict = {}
            
            # Genotype column
            if 'geno' in batch_masked.columns:
                rename_dict['geno'] = 'name'
            elif 'genotype' in batch_masked.columns:
                rename_dict['genotype'] = 'name'
            
            # Trait value column
            if 'trait_value' in batch_masked.columns:
                rename_dict['trait_value'] = 'labels'
            elif 'value' in batch_masked.columns:
                rename_dict['value'] = 'labels'
            elif 'phenotype_value' in batch_masked.columns:
                rename_dict['phenotype_value'] = 'labels'
                
            # Trait name column
            if 'trait_name' in batch_masked.columns:
                rename_dict['trait_name'] = 'trait_input_ids'
            elif 'variable' in batch_masked.columns:
                rename_dict['variable'] = 'trait_input_ids'
            elif 'phenotype_name' in batch_masked.columns:
                rename_dict['phenotype_name'] = 'trait_input_ids'
            
            # Environment column (always rename to env_input_ids)
            if env_col != 'env':
                rename_dict[env_col] = 'env_input_ids'
            else:
                rename_dict['env'] = 'env_input_ids'
            
            batch_masked = batch_masked.reset_index(drop=True).rename(columns=rename_dict)
            columns = ['encodings', 'name', 'labels', 'trait_input_ids', 'env_input_ids']
            # Only keep columns that exist
            columns = [col for col in columns if col in batch_masked.columns]
            batch_masked = batch_masked[columns]
            
            # Convert to PyArrow Table
            table = pa.Table.from_pandas(batch_masked)

            output_path = f"{self.output_dir}/Encodings_{file_prefix}_parquet/" 
            os.makedirs(output_path, exist_ok=True)
            pq.write_to_dataset(table, output_path)

