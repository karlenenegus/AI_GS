import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os
import json
import warnings
from .helper_fns import save_json, load_json



def filter_iqr_outliers(data, geno_col, env_col, pheno_col, iqr_multiplier=1.5):
    """
    Remove outliers based on IQR filtering:
    - Removes trait values outside the IQR bounds across genotypes and environments.
    
    Parameters:
        data (pd.DataFrame): Input data with genotype, environment, and phenotype columns.
        geno_col (str): Column name for genotype (e.g., RIL).
        env_col (str): Column name for environment.
        pheno_col (str): Column name for phenotype/trait.
        iqr_multiplier (float): Multiplier for IQR threshold (default is 1.5).
    
    Returns:
        pd.DataFrame: Filtered DataFrame with outliers removed.
    """
    df = data.copy()

    # IQR per genotype (RIL)
    ril_q1 = df.groupby(geno_col)[pheno_col].quantile(0.25)
    ril_q3 = df.groupby(geno_col)[pheno_col].quantile(0.75)
    ril_iqr = ril_q3 - ril_q1

    df = df.merge(ril_q1.rename('ril_q1'), left_on=geno_col, right_index=True)
    df = df.merge(ril_q3.rename('ril_q3'), left_on=geno_col, right_index=True)
    df = df.merge(ril_iqr.rename('ril_iqr'), left_on=geno_col, right_index=True)

    # IQR per environment
    env_q1 = df.groupby(env_col)[pheno_col].quantile(0.25)
    env_q3 = df.groupby(env_col)[pheno_col].quantile(0.75)
    env_iqr = env_q3 - env_q1

    df = df.merge(env_q1.rename('env_q1'), left_on=env_col, right_index=True)
    df = df.merge(env_q3.rename('env_q3'), left_on=env_col, right_index=True)
    df = df.merge(env_iqr.rename('env_iqr'), left_on=env_col, right_index=True)

    # Apply IQR-based filtering
    ril_mask = (df[pheno_col] >= df['ril_q1'] - iqr_multiplier * df['ril_iqr']) & \
               (df[pheno_col] <= df['ril_q3'] + iqr_multiplier * df['ril_iqr'])

    env_mask = (df[pheno_col] >= df['env_q1'] - iqr_multiplier * df['env_iqr']) & \
               (df[pheno_col] <= df['env_q3'] + iqr_multiplier * df['env_iqr'])

    # Keep only values within both bounds
    filtered_df = df[ril_mask & env_mask].copy()

    # Drop intermediate columns
    return filtered_df[data.columns]
    
# Generate valid random terms
def generate_valid_random_terms(df, term_list):
    """
    Generate list of random effect terms that have more than 1 unique level.

    
    Parameters:
        df (pd.DataFrame): Dataframe containing the data.
        term_list (list): List of lists of variable names representing hierarchical nesting.
                          For nested terms, joined columns are expected. Use a single string (e.g., ["rep_field"]) as the column name. Example: ["field", "rep_field", "block_set_rep_field"]
    
    Returns:
        list: List of random effect terms in formula format, e.g., ["(1|field)", "(1|rep_field)"]
    """
    random_terms = []

    for var in term_list:
        # Count unique levels for each variable in the prefix
        if var not in df.columns:
            print(f"Trying: {var} | {var} missing in data columns. Skipping Term.")
        else:
            unique_counts = df[var].nunique()
            print(f"Trying: {var} | unique levels: {unique_counts}")

            # All variables must have more than 1 unique level to be valid
            if unique_counts > 1:
                random_term = f"(1|{var})"
                random_terms.append(random_term)

            else:
                # No valid prefix found for this term
                print(f"  -> {var} not valid. Less than 2 levels.")

    return random_terms

def convert2output(data, genotype_col, environ_col, phenotype_col, phenotype_name):
    """
    Convert input data to output format by renaming columns to standard names.
    
    Renames columns to standard format:
    - genotype_col -> "geno"
    - environ_col -> "env"
    - phenotype_col -> "trait_value"
    - phenotype_name -> "trait_name"
    
    Also preserves 'fold' column if present.
    Drops rows with missing values.
    
    Parameters:
        data (pd.DataFrame): Input dataframe with genotype, environment, and phenotype columns.
        genotype_col (str): Name of the genotype column in input data.
        environ_col (str): Name of the environment column in input data.
        phenotype_col (str): Name of the phenotype column in input data.
        phenotype_name (str): Name of the phenotype in input data.
    
    Returns:
        pd.DataFrame: DataFrame with standardized column names and no missing values.
    """
    keep_cols = [genotype_col, environ_col, phenotype_col, phenotype_name]
    if 'fold' in data.columns:
        keep_cols.append('fold')
    else:
        warnings.warn("No 'fold' column available. Record of folds not preserved")
    
    data = data.loc[:, keep_cols].copy()
    
    data = data.rename(columns={genotype_col: 'geno', environ_col: 'env', phenotype_col: 'trait_value', phenotype_name: 'trait_name'})
    data = data.dropna()
    return data


def scale_phenotype(data, env, mode, output_dir, pheno_col):
    """
    Scale the phenotype column for a single environment using StandardScaler.

    Parameters:
        data (pd.DataFrame): Subset of data for one environment.
        env (str or int): The environment identifier.
        mode (str): Scaling mode - 'train', 'inference', or 'inverse'.
                   - 'train': Fit scaler and transform data, save scaler to disk.
                   - 'inference': Load scaler and transform data.
                   - 'inverse': Load scaler and inverse transform data to original scale.
        output_dir (str): Directory to save/load the scaler.
        pheno_col (str): Column name for phenotype values.

    Returns:
        pd.DataFrame: DataFrame with phenotype column scaled according to mode.
    """
    os.makedirs(f'{output_dir}/data/pheno_scalers/', exist_ok=True)
    scaler_path = f"{output_dir}/data/pheno_scalers/pheno_scaler_env_{env}.pkl"
    scaler_stats_path = f"{output_dir}/data/pheno_scalers/pheno_scaler_env_{env}_stats.csv"

    if mode == "train":
        scaler = StandardScaler()
        data.loc[:, pheno_col] = scaler.fit_transform(data[[pheno_col]])
        joblib.dump(scaler, scaler_path)
        
        mean_values = scaler.mean_
        scale_values = scaler.scale_
        pd.DataFrame({'mean': mean_values, 'scale': scale_values}).to_csv(scaler_stats_path, index=False)
        
    elif mode == "inverse":
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found for environment {env} at {scaler_path}")
        scaler = joblib.load(scaler_path)
        data.loc[:, f"{pheno_col}_unscaled"] = scaler.inverse_transform(data[[pheno_col]])
    else:
        raise ValueError("Mode must be 'train' or 'inverse'.")

    return data