import numpy as np
import pandas as pd
import os

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm

from typing import List, Optional

############################## EDA ##############################
def special_text_count(df, col):
    """
    Đếm số lượng ký tự đặc biệt, mentions (@), hashtags (#) và links trong cột col.
    Thêm 4 cột mới vào DataFrame: special_count, mention_count, hashtag_count, link_count.

    Args:
        df : DataFrame có cột col chứa văn bản.
        col: Tên cột có văn bản muốn đếm.

    Returns:
        DataFrame gốc được bổ sung 4 cột đếm.
    """
    def count_special(text):
        special_key = re.findall(r"[^\w\s,]", text) # Not a-z, 0-9, _, space, ","
        mentions = re.findall(r"@\w+", text)
        hashtags = re.findall(r"#\w+", text)
        links = re.findall(r"http\S+", text)
        return len(special_key), len(mentions), len(hashtags), len(links)

    df[["special_count", "mention_count", "hashtag_count", "link_count"]] = df[col].apply(
        lambda x: pd.Series(count_special(x))
    )

    print("Special key count:", df["special_count"].sum())
    print("Mention (@) count:", df["mention_count"].sum())
    print("Hashtag (#) count:", df["hashtag_count"].sum())
    print("Link count:", df["link_count"].sum())

    return df

def advanced_bow_eda(df, col, stop_words):
    """
    Tokenize văn bản trong cột col sau khi loại bỏ links, hashtags, mentions và ký tự
    không phải chữ cái, đồng thời lọc stopwords. Trả về DataFrame kèm cột "tokens"
    và bảng tần suất toàn bộ từ vựng.

    Args:
        df         : DataFrame.
        col        : Cột col chứa văn bản.
        stop_words : tập hợp (set) các stopwords cần lọc.

    Returns:
        df        : DataFrame gốc được bổ sung cột col + "_tokens" (list of str).
        word_freq : Counter ánh xạ token -> tần suất xuất hiện.
    """
    def advanced_bow(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # xóa link
        text = re.sub(r"@\w+|#\w+", "", text)                 # xóa hashtag, mention
        text = re.sub(r"[^a-z\s]", "", text)                  
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words]
        return tokens

    df[col + "_tokens"] = df[col].apply(advanced_bow)

    all_tokens = [t for doc in df[col + "_tokens"] for t in doc]
    word_freq = Counter(all_tokens)

    return df, word_freq


############################### Preprocess ###############################

def text_preprocess(
    df,
    lowercase=True,
    tokenizer="split",
    remove_stopwords=True,
    lemmatize=False,
    join_space=True,
    stop_words=None
):
    """
    Làm sạch và token hóa danh sách văn bản: lowercase, xóa links/hashtags/mentions/ký tự đặc biệt,
    tokenize, lọc stopwords, lemmatize (tuỳ chọn).

    Args:
        df               : iterable các chuỗi văn bản đầu vào.
        lowercase        : chuyển về chữ thường trước khi xử lý.
        tokenizer        : "split" (str.split) hoặc "nltk" (word_tokenize).
        remove_stopwords : lọc stopwords khỏi danh sách token.
        lemmatize        : áp dụng WordNetLemmatizer lên từng token.
        join_space       : nếu True, trả về chuỗi đã join bằng khoảng trắng;
                           nếu False, trả về list of tokens.
        stop_words       : tập stopwords tuỳ chỉnh; mặc định dùng NLTK English.

    Returns:
        List[str] hoặc List[List[str]] tuỳ theo join_space.
    """
    def preprocess_single(text):
        if lowercase:
            text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\w+|#\w+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        if tokenizer == "split":
            tokens = text.split()
        elif tokenizer == "nltk":
            tokens = word_tokenize(text)
        else:
            raise ValueError("Unknown tokenizer, use 'split' or 'nltk'")
        
        if remove_stopwords:
            tokens = [t for t in tokens if t not in stop_words]
        if lemmatize:
            tokens = [lemmatizer_tool(t) for t in tokens]
        
        if join_space:
            return " ".join(tokens)
        else:
            return tokens
    
    if stop_words is None:
        stop_words = set(stopwords.words("english"))
    lemmatizer_tool = WordNetLemmatizer()

    return [preprocess_single(text) for text in df]


############################# Features Extract #################################
class TextEmbedding(BaseEstimator, TransformerMixin):
    """
    Method:
        - 'bow'   : Bag of Words
        - 'tfidf' : TF-IDF
        - 'none'  : giữ nguyên token (cho deep learning)
    """
    def __init__(self, method: str = "tfidf", max_features: Optional[int] = 5000):
        self.method = method
        self.max_features = max_features

        if self.method == "bow":
            self.vectorizer = CountVectorizer(max_features=self.max_features)
        elif self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        elif self.method != "none": # if "none" do nothing
            raise ValueError("method must be 'bow', 'tfidf', or 'none'")

    def fit(self, X: List[List[str]], y=None):
        # texts = [" ".join(tokens) for tokens in X] # rejoin for sklearn :v
        if self.method != "none":
            self.vectorizer.fit(X)
        return self

    def transform(self, X: List[List[str]]):
        # texts = [" ".join(tokens) for tokens in X]
        if self.method != "none":
            return self.vectorizer.transform(X).toarray()
        else:
            return X


class SequencePadder(BaseEstimator, TransformerMixin):
    """
    Padding độ dài chuỗi cho mô hình deep learning.
    """
    def __init__(self, max_len: int = 50, pad_token: str = "<PAD>"):
        self.max_len = max_len
        self.pad_token = pad_token

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[List[str]]):
        padded = []
        for seq in X:
            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            else:
                seq = seq + [self.pad_token] * (self.max_len - len(seq))
            padded.append(seq)
        return np.array(padded)


def build_text_pipeline(
    embedding="tfidf",
    max_feature=5000,
    use_padding=False,
    padding_max_len=50
):
    """
    Tạo sklearn Pipeline cho feature extraction văn bản.

    Args:
        embedding      : phương pháp vector hóa: "bow", "tfidf", hoặc "none".
        max_feature    : số lượng feature tối đa cho CountVectorizer/TfidfVectorizer.
        use_padding    : thêm bước SequencePadder sau embedding (dùng cho deep learning).
        padding_max_len: độ dài tối đa của chuỗi khi dùng padding.

    Returns:
        sklearn Pipeline với bước "embed" (và "pad" nếu use_padding=True).
    """
    steps = [
        ("embed", TextEmbedding(method=embedding, max_features=max_feature))
    ]
    if use_padding:
        steps.append(("pad", SequencePadder(max_len=padding_max_len)))

    return Pipeline(steps)

############################# Training #################################

def multi_preprocess_fe(preprocess_fe_config: list, data: pd.DataFrame, text_col: str, target: str, data_path: str):
    """
    Xử lý văn bản và trích xuất features theo nhiều cấu hình, lưu kết quả ra disk.
    Bỏ qua các config đã được lưu từ lần chạy trước (cache theo đường dẫn).

    Mỗi config trong preprocess_fe_config là một dict có các key (đều tuỳ chọn):
        tokenizer        : "split" | "nltk"         (mặc định: "split")
        remove_stopwords : bool                      (mặc định: True)
        lemmatize        : bool                      (mặc định: False)
        embedding        : "bow" | "tfidf" | "none"  (mặc định: "tfidf")
        max_features     : int                       (mặc định: 5000)

    Args:
        preprocess_fe_config : danh sách config dict.
        data                 : DataFrame có cột văn bản và cột nhãn.
        text_col             : tên cột văn bản.
        target               : tên cột nhãn.
        data_path            : thư mục gốc để lưu các thư mục kết quả.

    Returns:
        DataFrame với các cột config và cột "save_path" chứa đường dẫn thư mục dữ liệu.
    """
    results = pd.DataFrame(columns=[name for name in preprocess_fe_config[0]] + ["save_path"])

    X = data[text_col]
    y = data[target]

    for i, config in enumerate(preprocess_fe_config):

        tokenizer = config["tokenizer"] if "tokenizer" in config else "split"
        remove_stopwords = config["remove_stopwords"] if "remove_stopwords" in config else True
        lemmatize = config["lemmatize"] if "lemmatize" in config else False
        embedding = config["embedding"] if "embedding" in config else "tfidf"
        max_features = config["max_features"] if "max_features" in config else 5000

        path = os.path.join(data_path, f"tok={tokenizer},stpw={remove_stopwords},lem={lemmatize},emb={embedding},maxf={max_features}/")
        results.loc[len(results)] = [config[step] for step in config] + [path]
        if os.path.exists(path):
            continue
        os.makedirs(path)

        X_clean = text_preprocess(df=X, tokenizer=tokenizer, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
        target_encoder = OrdinalEncoder()
        y_clean = target_encoder.fit_transform(y.values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean)

        pipe = build_text_pipeline(embedding=embedding, max_feature=max_features)

        X_train = pipe.fit_transform(X_train)
        X_test = pipe.transform(X_test)

        np.save(os.path.join(path, "X_train"), X_train)
        np.save(os.path.join(path, "X_test"), X_test)
        np.save(os.path.join(path, "y_train"), y_train)
        np.save(os.path.join(path, "y_test"), y_test)

    return results

def train_one_model(params, split_datasets, score):
    """
    Train một model trên nhiều bộ dữ liệu đã preprocess (đọc từ disk) và đánh giá.

    Args:
        params        : dict chứa key "model" (sklearn estimator) và các hyperparameter
                        dạng pipeline (ví dụ: {"model": SVC(), "model__C": 1.0}).
        split_datasets: DataFrame từ multi_preprocess_fe() có cột "save_path".
        score         : metric đánh giá, hiện hỗ trợ "f1" (weighted).

    Returns:
        DataFrame với các cột config (không có "save_path") và cột score.
    """
    results = split_datasets.drop(columns=["save_path"])
    results[score] = 0
    for i, data_path in tqdm(
        enumerate(split_datasets["save_path"]), total=len(split_datasets), desc="Training"
    ):
        X_train = np.load(os.path.join(data_path, "X_train.npy"))
        X_test = np.load(os.path.join(data_path, "X_test.npy"))
        y_train = np.load(os.path.join(data_path, "y_train.npy"))
        y_test = np.load(os.path.join(data_path, "y_test.npy"))

        model = Pipeline(steps=[("model", params["model"])])
        model.set_params(**params)
        model.set_output(transform="pandas")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if score == "f1":
            # if y_transform is not None:
            #     y_test = y_transform.transform(y_test.values.reshape(-1, 1))
            
            results[score][i] = f1_score(y_test, y_pred, average="weighted")
        else:
            raise ValueError
        
    return results

def train_multi_model(grid, params_df, data_path, score):
    """
    Train nhiều model với các hyperparameter khác nhau trên một bộ dữ liệu cố định.

    Args:
        grid      : list các param dict, mỗi dict có key "model" và các hyperparameter pipeline.
        params_df : DataFrame mô tả grid (mỗi hàng tương ứng một config trong grid).
        data_path : đường dẫn thư mục chứa X_train.npy, X_test.npy, y_train.npy, y_test.npy.
        score     : metric đánh giá, hiện hỗ trợ "f1" (weighted).

    Returns:
        DataFrame (giống params_df) được bổ sung cột score.
    """
    results_df = params_df
    results_df[score] = 0
    
    X_train = np.load(os.path.join(data_path, "X_train.npy"))
    X_test = np.load(os.path.join(data_path, "X_test.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))
    
    for i, params in tqdm(
        enumerate(grid), total=len(grid), desc="Training"
    ):
        model = Pipeline(steps=[("model", params["model"])])
        model.set_params(**params)
        model.set_output(transform="pandas")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if score == "f1":
            results_df[score][i] = f1_score (y_test, y_pred, average="weighted")
        else:
            raise ValueError
    
    return results_df