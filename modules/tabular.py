# Tabular

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import r2_score



plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

########################## EDA #############################

def dataset_info(df: pd.DataFrame):
    '''
    Print Pandas.DataFrame info like: len, feature, shape, columns name, dtype
    '''
    print("\nDataset Info:")
    print(f"Total samples: {len(df):,}")
    print(f"Features: {df.shape[1]}")
    print(f"Data Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print("Dtypes:\n", df.dtypes)

def missing_report(df: pd.DataFrame):
    missing_df = df.isna().sum().sort_values(ascending=False)
    print("Missing Value:")
    print(pd.DataFrame({"Missing" : missing_df, "Percent" : (missing_df / len(df) * 100).round(2)}))

def column_report(df: pd.DataFrame):
    print("Column Report:")
    print(pd.DataFrame({"Dtypes" : df.dtypes, "Nunique" : df.nunique()}))

########################## End of EDA #############################


################################# Descriptive Statistics #######################################

################# Numeric ########################

def numeric_stat_report(df: pd.DataFrame):
    numeric_feature = list(df.select_dtypes(include=["number"]).columns)
    return numeric_feature, df.describe(include=["number"])

def numeric_plot(df: pd.DataFrame, config: dict | None = ..., plot="histogram"):
    if config is None:
        config = {
            "normal": list(df.select_dtypes(include=["number"]).columns)
        }

    numeric_feature = [(transform, col) for transform, columns in config.items() for col in columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (transform, col) in enumerate(numeric_feature):
        row_idx, col_idx = i // 3, i % 3

        if transform == "normal":
            clean_data = df[col].dropna()
            
        elif transform == "log":
            clean_data = np.log10(df[col].dropna() + 1) # + 1 for zero value (not neg)
            axes[row_idx, col_idx].xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"$10^{{{int(x)}}}$")
            )
            
        else:
            nah_not_implement()

        axes[row_idx, col_idx].set_xlabel(col)
        axes[row_idx, col_idx].set_title(f"Distribution of {col}")

        if plot == "histogram":
            axes[row_idx, col_idx].hist(clean_data, bins=30)
        elif plot == "boxplot":
            sns.boxplot(x=clean_data, ax=axes[row_idx, col_idx])
        else:
            nah_not_implement()
    
    plt.tight_layout()
    plt.show()

    return


################ End of Numeric ####################

################ Categorical #######################

def categorical_stat_report(df: pd.DataFrame):
    categorical_feature = list(df.select_dtypes(include=["object", "category"]).columns)
    for col in categorical_feature:
        print(f"================== Top 10 {col} ==================")
        print("Unique value:", df[col].nunique())
        print("Missing value:", df[col].isna().sum())
        display(df[col].value_counts().head(10))
    
    return categorical_feature

############### End of Categorical #################

################################# End of Descriptive Statistics #######################################



################################# Preprocessing #######################################

def make_column_pipeline(config: dict) -> Pipeline:
    """
    config: dict {"name": (impute, [list_of_columns_index])}. Default = None = dropna
    Example:
    config = {
        "impute": (SimpleImputer(strategy="mean"), [0, 1])
    }
    """
    transformer = [(name, impute, cols) for name, (impute, cols) in config.items()]
    return Pipeline(steps=[
        ("missing_value", ColumnTransformer(transformers=transformer, remainder="passthrough"))
    ])

def make_preprocess_pipeline(step_list: list):
    return Pipeline(steps=[(f"step_{i}",step) for i, step in enumerate(step_list)])

def get_preprocesser(step: str, type: str):

    num_imputer_dict = {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "constant": SimpleImputer(strategy="constant")
    }

    cate_imputer_dict = {
        "most": SimpleImputer(strategy="most_frequent"),
        "constant": SimpleImputer(strategy="constant")
    }

    scaler_dict = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "log1p_robust": Log1pRobustScaler()
    }

    pca_dict = {
        "pca_0.95_auto": PCA(n_components=0.95, svd_solver="auto"),
        "pca_0.95_full": PCA(n_components=0.99, svd_solver="full"),
        "pca_0.99_auto": PCA(n_components=0.99, svd_solver="auto"),
        "pca_0.99_full": PCA(n_components=0.99, svd_solver="full")
    }

    encoder_dict = {
        "onehot": OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        "ordinal": OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    }

    step_dict = {
        "num_impute": num_imputer_dict,
        "cate_impute": cate_imputer_dict,
        "scale": scaler_dict,
        "pca": pca_dict,
        "encode": encoder_dict
    }

    if step in step_dict:
        if type == "all":
            return step_dict[step]
        elif type in step_dict[step]:
            return step_dict[step][type]
        else:
            print(f"Unknown {type} of step {step}")
            raise ValueError
    else:
        print("Unknown step")
        raise ValueError


################################# End of Preprocessing #######################################

def preprocess(preprocess_config: list, data: pd.DataFrame, target: str):
    results = pd.DataFrame(columns=[name for name in preprocess_config[0]] + ["data"])

    data = data.dropna(subset=[target])

    numeric_cols = list(data.select_dtypes(include=["number"]).columns)
    cate_cols = list(data.select_dtypes(include=["object", "category"]).columns)

    if target in numeric_cols:
        numeric_cols.remove(target)
    else:
        cate_cols.remove(target)

    for config in preprocess_config:

        if config["num_impute"] == "none":
            data = data.dropna(subset=numeric_cols)
        if config["cate_impute"] == "none":
            data = data.dropna(subset=cate_cols)

        X = data.drop(columns=target)
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numeric_steps = [(step, get_preprocesser(step, config[step])) for step in ("num_impute", "scale", "pca") if config[step] != "none"]
        cate_steps = [(step, get_preprocesser(step, config[step])) for step in ("cate_impute", "encode") if config[step] != "none"]
        if len(numeric_steps) == 0 and len(cate_steps) == 0:
            continue

        transformers = []
        if len(numeric_steps) != 0:
            transformers.append(("numeric", Pipeline(steps=numeric_steps), numeric_cols))
        if len(cate_steps) != 0:
            transformers.append(("category", Pipeline(steps=cate_steps), cate_cols))
        
        pipe = ColumnTransformer(transformers=transformers, remainder="passthrough")
        # numeric_pipe = Pipeline(steps=numeric_steps)
        # cate_pipe = Pipeline(steps=cate_steps)
        # pipe = ColumnTransformer(transformers=[
        #     ("numeric", numeric_pipe, numeric_cols),
        #     ("category", cate_pipe, cate_cols)
        # ])

        X_train = pipe.fit_transform(X_train)
        X_test = pipe.transform(X_test)

        results.loc[len(results)] = [config[step] for step in config] + [(X_train, X_test, y_train, y_test)]
    
    return results

################################# Trainning #######################################

def train_one_model(params, split_datasets, y_transform=None):
    results = split_datasets.drop(columns=["data"])
    results["r2"] = 0
    for i, (X_train, X_test, y_train, y_test) in enumerate(split_datasets["data"]):
        model = Pipeline(steps=[("model", params["model"])])
        model.set_params(**params)
        model.set_output(transform="pandas")

        if y_transform is not None:
            y_train = y_transform.fit_transform(y_train.values.reshape(-1, 1))
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if y_transform is not None:
            y_pred = y_transform.inverse_transform(y_pred.reshape(-1, 1))
        
        results["r2"][i] = r2_score(y_test, y_pred)
    
    return results

################################# End of Trainning #######################################







from sklearn.base import BaseEstimator, TransformerMixin

class Log1pRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = RobustScaler()
        self.columns_ = None
        self.index_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
            self.index_ = X.index
        X_log = np.log1p(X)
        self.scaler.fit(X_log)
        return self

    def transform(self, X):
        X_log = np.log1p(X)
        X_scaled = self.scaler.transform(X_log)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_scaled, columns=self.columns_, index=X.index)
        return X_scaled

    def inverse_transform(self, X):
        X_inv = self.scaler.inverse_transform(X)
        X_orig = np.expm1(X_inv)
        if self.columns_ is not None:
            return pd.DataFrame(X_orig, columns=self.columns_)
        return X_orig

    def set_output(self, *, transform = None): # Dummy function
        return self

def nah_not_implement():
    print("Nah, update later :)")
    raise NotImplementedError