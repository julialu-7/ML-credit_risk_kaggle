#!/usr/bin/env python
# coding: utf-8

# In[1]:


import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import dask

dataPath = "parquet/train/"



# In[2]:


def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64).alias(col))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date).alias(col))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String).alias(col))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date).alias(col))
        return df

def convert_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:  
        if df[col].dtype.name in ['object', 'string']:
            df[col] = df[col].astype("string").astype('category')
            current_categories = df[col].cat.categories
            new_categories = current_categories.to_list() + ["Unknown"]
            new_dtype = pd.CategoricalDtype(categories=new_categories, ordered=True)
            df[col] = df[col].astype(new_dtype)
    return df

def helper(df: pl.DataFrame, agg_func=pl.mean) -> pl.DataFrame:
    agg_exprs = [agg_func(col).alias(col) for col in df.select(pl.col(pl.NUMERIC_DTYPES)).columns if col != 'case_id']
    agg_exprs += [pl.first(col).alias(col) for col in df.select(pl.all().exclude(pl.NUMERIC_DTYPES)).columns]

    result = df.group_by('case_id') \
           .agg(*agg_exprs) \
           .sort(by='case_id') \
           .select(df.columns) \
           .drop('num_group1' if 'num_group2' not in df.columns else ['num_group1', 'num_group2'])
    return result
    



# In[4]:


class Aggregator:
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_median = [pl.median(col).alias(f"mean_{col}") for col in cols]

        return expr_max + expr_first + expr_last + expr_median
    
    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_median = [pl.median(col).alias(f"mean_{col}") for col in cols]
        return  expr_max + expr_first + expr_last + expr_median
    
    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_count = [pl.count(col).alias(f"count_{col}") for col in cols]
        return  expr_first + expr_last + expr_count
    
    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        return  expr_max + expr_first + expr_last
    
    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        return  expr_last
    
    def get_exprs(df, depth):
        if depth == 2:
            df = df.sort(by = ['num_group1', 'num_group2'])
        else:
            df = df.sort(by = ['num_group1'])
            
        exprs = Aggregator.num_expr(df) + \
                Aggregator.date_expr(df) + \
                Aggregator.str_expr(df) + \
                Aggregator.other_expr(df) + \
                Aggregator.count_expr(df)

        return exprs


# In[14]:

def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(set_table_dtypes)
    if depth in [1,2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df, depth))
    return df

TRAIN = "parquet/train/"
data_store = {
    "df_base": read_file(TRAIN + "train_base.parquet"),
    "depth_0": [
        read_file(TRAIN + "train_static_cb_0.parquet"),
        read_file(TRAIN + "train_static_0_0.parquet"),
    ],
    "depth_1": [
        read_file(TRAIN + "train_applprev_1_0.parquet", 1),
        read_file(TRAIN + "train_tax_registry_a_1.parquet", 1),
        read_file(TRAIN + "train_tax_registry_b_1.parquet", 1),
        read_file(TRAIN + "train_tax_registry_c_1.parquet", 1),
        read_file(TRAIN + "train_credit_bureau_a_1_0.parquet", 1),
        read_file(TRAIN + "train_credit_bureau_b_1.parquet", 1),
        read_file(TRAIN + "train_other_1.parquet", 1),
        read_file(TRAIN + "train_person_1.parquet", 1),
        read_file(TRAIN + "train_deposit_1.parquet", 1),
        read_file(TRAIN + "train_debitcard_1.parquet", 1),
    ],
    "depth_2": [
        read_file(TRAIN + "train_credit_bureau_b_2.parquet", 2),
        read_file(TRAIN + "train_credit_bureau_a_2_0.parquet", 2),
        read_file(TRAIN + "train_applprev_2.parquet", 2),
        read_file(TRAIN + "train_person_2.parquet", 2)
    ]
}


def select_numerical_columns(df, case_id_column='case_id'):
    numerical_columns = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
    return df.select(numerical_columns)


def feature_eng(df_base, depth_0, depth_1, depth_2):
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_numerical = select_numerical_columns(df)
        df_base = df_base.join(df_numerical, how="left", on="case_id", suffix=f"_{i}")
    return df_base

df_train = feature_eng(**data_store)

# In[4]:
df_train_pd = df_train.to_pandas()

for col in df_train_pd.columns:
    if df_train_pd[col].isna().sum() >= 0.5 * df_train_pd.shape[0]:
        df_train_pd.drop(col, axis = 1, inplace = True)
        
float_imputer = SimpleImputer(strategy='mean')
for col in df_train_pd.columns:
    df_train_pd.loc[:, col] = df_train_pd.fit_transform(df_train_pd[[col]])
    
df_train_pd.dropna()
# In[4]:

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df_train is a pandas DataFrame with numerical features only
# Step 1: Standardize the data
scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train_pd)

# Step 2: Apply PCA
pca = PCA()
X_pca = pca.fit_transform(df_train_scaled)

# Step 3: Visualize the PCA results
plt.figure(figsize=(10, 8))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Step 4: Analyze the Loadings
loadings = pd.DataFrame(pca.components_.T, columns=['PC' + str(i) for i in range(1, len(pca.components_) + 1)], index=df_train_pd.columns)
# You might want to visualize the loadings as well
sns.heatmap(loadings, cmap='viridis', center=0)
plt.show()

# Detailed analysis of loadings
# You can print the loadings or export them to CSV for detailed examination
# loadings.to_csv('pca_loadings.csv')

# In[5]: