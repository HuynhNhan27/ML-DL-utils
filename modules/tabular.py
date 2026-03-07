# Tabular

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import r2_score, f1_score
from tqdm import tqdm


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

########################## EDA #############################

def dataset_info(df: pd.DataFrame):
    '''
    In thông tin của DataFrame: len, feature, shape, columns name, dtype
    '''
    print("\nDataset Info:")
    print(f"Total samples: {len(df):,}")
    print(f"Features: {df.shape[1]}")
    print(f"Data Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print("Dtypes:\n", df.dtypes)

def missing_report(df: pd.DataFrame):
    """
    In báo cáo missing values của DataFrame, sắp xếp theo số lượng giảm dần,
    kèm theo phần trăm thiếu trên tổng số mẫu.
    """
    missing_df = df.isna().sum().sort_values(ascending=False)
    print("Missing Value:")
    print(pd.DataFrame({"Missing" : missing_df, "Percent" : (missing_df / len(df) * 100).round(2)}))

def column_report(df: pd.DataFrame):
    """
    In báo cáo tổng quan từng cột: kiểu dữ liệu (dtype) và số lượng giá trị unique.
    """
    print("Column Report:")
    print(pd.DataFrame({"Dtypes" : df.dtypes, "Nunique" : df.nunique()}))

########################## End of EDA #############################


################################# Descriptive Statistics #######################################

################# Numeric ########################

def numeric_stat_report(df: pd.DataFrame):
    """
    Trả về danh sách tên cột số và bảng thống kê mô tả (count, mean, std, min, quartiles, max).
    """
    numeric_feature = list(df.select_dtypes(include=["number"]).columns)
    return numeric_feature, df.describe(include=["number"])

def numeric_plot(df: pd.DataFrame, config: dict, plot="histogram"):
    """
    Vẽ biểu đồ phân phối cho các cột số.

    Args:
        df     : DataFrame đầu vào.
        config : dict ánh xạ kiểu biến đổi trục x tới danh sách tên cột.
                 Ví dụ: {"normal": ["col_a"], "log": ["col_b"]}.
                 Nếu None, tất cả cột số được vẽ theo trục "normal".
        plot   : loại biểu đồ, "histogram" hoặc "boxplot".
    """
    if config is None:
        config = {
            "normal": list(df.select_dtypes(include=["number"]).columns)
        }

    numeric_feature = [(transform, col) for transform, columns in config.items() for col in columns]

    cols = 3
    rows = len(numeric_feature) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.reshape(rows, cols)
    for i, (transform, col) in enumerate(numeric_feature):
        row_idx, col_idx = i // cols, i % cols

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
    """
    In top 10 giá trị phổ biến nhất, số unique và số missing cho từng cột categorical.

    Returns:
        Danh sách tên các cột categorical.
    """
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
    """
    Tạo một sklearn Pipeline từ danh sách các transformer, tự động đặt tên bước là step_0, step_1, ...

    Args:
        step_list: danh sách các sklearn transformer/estimator.

    Returns:
        sklearn Pipeline.
    """
    return Pipeline(steps=[(f"step_{i}",step) for i, step in enumerate(step_list)])

def get_preprocesser(step: str, type: str):
    """
    Trả về một sklearn transformer tương ứng với bước và loại được chỉ định.

    Args:
        step : tên bước xử lý. Các giá trị hợp lệ:
                 "num_impute"  – imputer cho cột số  ("mean", "median", "constant")
                 "cate_impute" – imputer cho cột cate ("most", "constant")
                 "scale"       – scaler               ("standard", "minmax", "robust", "log1p_robust")
                 "pca"         – giảm chiều PCA        ("pca_0.95_auto", "pca_0.95_full", "pca_0.99_auto", "pca_0.99_full")
                 "encode"      – encoder cate          ("onehot", "ordinal")
        type : tên cụ thể của transformer trong bước đó, hoặc "all" để lấy toàn bộ dict.

    Returns:
        Sklearn transformer tương ứng, hoặc dict nếu type="all".
    """
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
        "pca_0.95_full": PCA(n_components=0.95, svd_solver="full"),
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
    """
    Chạy nhiều cấu hình tiền xử lý trên cùng một DataFrame và trả về tập train/test tương ứng.

    Mỗi config trong preprocess_config là một dict có các key:
        num_impute  : "mean" | "median" | "constant" | "none" (dropna)
        cate_impute : "most" | "constant" | "none" (dropna)
        scale       : "standard" | "minmax" | "robust" | "log1p_robust" | "none"
        pca         : "pca_0.95_auto" | "pca_0.95_full" | "pca_0.99_auto" | "pca_0.99_full" | "none"
        encode      : "onehot" | "ordinal" | "none"

    Args:
        preprocess_config : danh sách các config dict.
        data              : DataFrame gốc (có cả feature và target).
        target            : tên cột nhãn.

    Returns:
        DataFrame với các cột config và cột "data" chứa tuple (X_train, X_test, y_train, y_test).
    """
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

def train_one_model(params, split_datasets, score, y_transform=None):
    """
    Train một model trên nhiều bộ dữ liệu đã preprocess và đánh giá theo metric chỉ định.

    Args:
        params         : dict chứa key "model" (sklearn estimator) và các hyperparameter dạng
                         pipeline (ví dụ: {"model": Ridge(), "model__alpha": 1.0}).
        split_datasets : DataFrame từ hàm preprocess(), cột "data" chứa
                         tuple (X_train, X_test, y_train, y_test).
        score          : metric đánh giá: "r2" (regression) hoặc "f1" (classification).
        y_transform    : sklearn transformer áp dụng lên nhãn trước khi train (ví dụ: Log1pRobustScaler).
                         Nếu None, nhãn giữ nguyên.

    Returns:
        DataFrame với các cột config (không có "data") và cột score tương ứng.
    """
    results = split_datasets.drop(columns=["data"])
    results[score] = 0
    for i, (X_train, X_test, y_train, y_test) in tqdm(
        enumerate(split_datasets["data"]), total=len(split_datasets), desc="Training"
    ):
        model = Pipeline(steps=[("model", params["model"])])
        model.set_params(**params)
        model.set_output(transform="pandas")

        if y_transform is not None:
            y_train = y_transform.fit_transform(y_train.values.reshape(-1, 1))
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        
        if score == "r2":
            if y_transform is not None:
                y_pred = y_transform.inverse_transform(y_pred.reshape(-1, 1))

            results[score][i] = r2_score(y_test, y_pred)

        elif score == "f1":
            if y_transform is not None:
                y_test = y_transform.transform(y_test.values.reshape(-1, 1))
            
            results[score][i] = f1_score(y_test, y_pred)
        else:
            raise ValueError
        
    return results

def cross_validate_model(model, X, y, cv=5, score="f1"):
    """
    Wrapper cho sklearn cross_val_score. Đánh giá model qua k-fold CV và in mean ± std.

    Args:
        model : sklearn estimator.
        X     : features (array-like hoặc DataFrame).
        y     : nhãn (array-like hoặc Series).
        cv    : số fold, mặc định 5.
        score : metric đánh giá: "f1", "accuracy", hoặc "r2".

    Returns:
        dict với các key:
            "scores" : np.ndarray các fold scores,
            "mean"   : float mean score,
            "std"    : float std score.
    """
    score_map = {
        "f1":       "f1",
        "accuracy": "accuracy",
        "r2":       "r2",
    }

    if score not in score_map:
        raise ValueError(f"score phải là một trong {list(score_map.keys())}, nhận được: '{score}'")

    scores = cross_val_score(model, X, y, cv=cv, scoring=score_map[score])
    mean, std = scores.mean(), scores.std()

    fold_lines = "  ".join([f"fold {i+1}: {s:.4f}" for i, s in enumerate(scores)])
    print(f"[Cross-Validation] score={score}, cv={cv}")
    print(f"  {fold_lines}")
    print(f"  => mean={mean:.4f} ± std={std:.4f}")

    return {"scores": scores, "mean": mean, "std": std}

def hyperparameter_search(model, param_grid, X, y, method="grid", cv=5, score="f1", n_iter=10):
    """
    Wrapper cho GridSearchCV / RandomizedSearchCV.

    Args:
        model      : sklearn estimator.
        param_grid : dict hoặc list of dicts tham số cần tìm kiếm.
        X          : features (array-like hoặc DataFrame).
        y          : nhãn (array-like hoặc Series).
        method     : "grid"   → GridSearchCV,
                     "random" → RandomizedSearchCV.
        cv         : số fold cross-validation, mặc định 5.
        score      : metric đánh giá: "f1", "accuracy", hoặc "r2".
        n_iter     : số tổ hợp thử khi method="random", mặc định 10.

    Returns:
        Fitted search object (GridSearchCV hoặc RandomizedSearchCV).
        Truy cập kết quả qua .best_params_, .best_score_, .cv_results_.
    """
    if score not in ("f1", "accuracy", "r2"):
        raise ValueError(f"score phải là 'f1', 'accuracy' hoặc 'r2', nhận được: '{score}'")

    if method == "grid":
        search = GridSearchCV(model, param_grid, cv=cv, scoring=score, refit=True)
    elif method == "random":
        search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv, scoring=score, refit=True, random_state=42)
    else:
        raise ValueError(f"method phải là 'grid' hoặc 'random', nhận được: '{method}'")

    search.fit(X, y)

    print(f"[Hyperparameter Search] method={method}, cv={cv}, score={score}")
    print(f"  best_params_ : {search.best_params_}")
    print(f"  best_score_  : {search.best_score_:.4f}")

    return search

################################# End of Trainning #######################################







from sklearn.base import BaseEstimator, TransformerMixin

class Log1pRobustScaler(BaseEstimator, TransformerMixin):
    """
    Scaler kết hợp log1p và RobustScaler: áp dụng log(1+x) trước, sau đó scale bằng RobustScaler.
    Phù hợp cho dữ liệu số có phân phối lệch (skewed) và nhiều outlier.
    Hỗ trợ cả numpy array và pandas DataFrame, và inverse_transform để khôi phục giá trị gốc.
    """
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