# Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from collections import Counter
from typing import List, Optional, Tuple, Dict

from PIL import Image, ImageEnhance

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report

from tqdm import tqdm


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


########################## EDA ##########################

def dataset_info(image_paths: List, labels: List):
    """
    In thông tin cơ bản của dataset ảnh.

    Args:
        image_paths: list đường dẫn ảnh (str).
        labels: list nhãn cho từng ảnh.
    """
    class_counts = Counter(labels)
    sizes = []
    for p in image_paths:
        with Image.open(p) as img:
            sizes.append(img.size)  # (W, H)

    widths, heights = zip(*sizes)

    print("\nDataset Info:")
    print(f"Total samples  : {len(image_paths):,}")
    print(f"Num classes    : {len(class_counts)}")
    print(f"Classes        : {sorted(class_counts.keys())}")
    print(f"Image width    : min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
    print(f"Image height   : min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count:,}")


def class_distribution(labels: List):
    """
    Vẽ plot cho phân phối nhãn.

    Args:
        labels: list nhãn.
    """
    class_counts = Counter(labels)
    classes = sorted(class_counts.keys(), key=str)
    counts = [class_counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.8), 5))
    bars = ax.bar([str(c) for c in classes], counts)
    ax.bar_label(bars)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def show_samples(
    image_paths: List,
    labels: List,
    n_per_class: int = 3,
    figsize_per_cell: Tuple[int, int] = (2, 2)
):
    """
    In ra một vài ảnh của từng class.

    Args:
        image_paths: list các đường dẫn ảnh.
        labels: list các nhãn.
        n_per_class: số ảnh in ra cho từng nhãn.
        figsize_per_cell: kích thước cho mỗi cell plot.
    """
    label_to_paths: Dict = {}
    for path, label in zip(image_paths, labels):
        label_to_paths.setdefault(label, []).append(path)

    classes = sorted(label_to_paths.keys(), key=str)
    n_cols = n_per_class
    n_rows = len(classes)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows)
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    for row, cls in enumerate(classes):
        paths = label_to_paths[cls][:n_per_class]
        for col, path in enumerate(paths):
            img = Image.open(path)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_title(str(cls), fontsize=10, loc="left")

    plt.suptitle("Sample Images per Class", fontsize=12)
    plt.tight_layout()
    plt.show()


def pixel_stats(images: np.ndarray):
    """
    EDA thống kê giá trị các pixel của từng kênh.

    Args:
        images: numpy array of shape (N, H, W, C) or (N, H, W)
    """
    images = images.astype(np.float32)
    if images.ndim == 4:
        n_channels = images.shape[-1]
        channel_names = ["R", "G", "B", "A"][:n_channels] if n_channels <= 4 else [str(i) for i in range(n_channels)]
        print("Pixel Stats (per channel):")
        for c in range(n_channels):
            ch = images[..., c]
            print(f"  [{channel_names[c]}] mean={ch.mean():.3f}, std={ch.std():.3f}, min={ch.min():.1f}, max={ch.max():.1f}")
    else:
        print("Pixel Stats (grayscale):")
        print(f"  mean={images.mean():.3f}, std={images.std():.3f}, min={images.min():.1f}, max={images.max():.1f}")


########################## End of EDA ##########################


########################## Preprocessing ##########################

class ImageResizer(BaseEstimator, TransformerMixin):
    """
    Resize một list các PIL Images hoặc numpy arrays về kích thước target_size.

    Args:
        target_size: (width, height)
        mode: PIL image mode ('RGB', 'L', etc.)
    """
    def __init__(self, target_size: Tuple[int, int] = (64, 64), mode: str = "RGB"):
        self.target_size = target_size
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        resized = []
        for img in X:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert(self.mode).resize(self.target_size)
            resized.append(np.array(img))
        return np.array(resized)


class ImageNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize.

    Args:
        method: 'minmax', 'standard', 'none'
    """
    def __init__(self, method: str = "minmax"):
        self.method = method
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        X = X.astype(np.float32)
        if self.method == "standard":
            axes = tuple(range(X.ndim - 1)) if X.ndim == 4 else None
            self.mean_ = X.mean(axis=axes) if axes else X.mean()
            self.std_ = X.std(axis=axes) if axes else X.std()
        return self

    def transform(self, X):
        X = X.astype(np.float32)
        if self.method == "minmax":
            return X / 255.0
        elif self.method == "standard":
            return (X - self.mean_) / (self.std_ + 1e-8)
        elif self.method == "none":
            return X
        else:
            raise ValueError(f"Unknown method '{self.method}', use 'minmax', 'standard', or 'none'")


class GrayscaleConverter(BaseEstimator, TransformerMixin):
    """
    Chuyển ảnh (N, H, W, 3) RGB thành ảnh xám (N, H, W)
    công thức: 0.299R + 0.587G + 0.114B.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.ndim == 3:
            return X
        return (0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2]).astype(np.float32)


class ImageFlattener(BaseEstimator, TransformerMixin):
    """
    Flatten mỗi ảnh thành vector 1 chiều: (H, W) hoặc (H, W, C) -> vector.
    Input shape: (N, H, W) or (N, H, W, C) -> output: (N, H*W*C).
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(len(X), -1)

class ImageAugmentor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer cho data augmentation dùng PIL.
    Chỉ augment khi train_mode=True, trả về ảnh gốc khi train_mode=False (inference).

    Augmentations:
        - Random horizontal flip (flip_h) (dùng random.rand())
        - Random vertical flip   (flip_v) (dùng random.rand())
        - Random rotation trong [-rotate_deg, +rotate_deg]
        - Random brightness jitter trong [1-brightness, 1+brightness]
        - Random contrast jitter  trong [1-contrast,   1+contrast  ]

    Args:
        flip_h     : bật random horizontal flip, mặc định True. (ngang)
        flip_v     : bật random vertical flip, mặc định False. (dọc)
        rotate_deg : góc xoay tối đa (độ), mặc định 30.
        brightness : cường độ jitter độ sáng, mặc định 0.2.
        contrast   : cường độ jitter độ tương phản, mặc định 0.2.
        train_mode : nếu False, transform() bỏ qua mọi augmentation.
    """
    def __init__(
        self,
        flip_h: bool = True,
        flip_v: bool = False,
        rotate_deg: float = 30,
        brightness: float = 0.2,
        contrast: float = 0.2,
        train_mode: bool = True
    ):
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.rotate_deg = rotate_deg
        self.brightness = brightness
        self.contrast = contrast
        self.train_mode = train_mode
    
    def fit(self, X, y=None):
        return self
    
    def _augment_one(self, img : Image.Image) -> Image.Image:
        if self.flip_h and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_v and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if self.rotate_deg:
            angle = np.random.uniform(-self.rotate_deg, self.rotate_deg)
            img = img.rotate(angle, resample=Image.BILINEAR)
        if self.brightness:
            factor = np.random.uniform(1 - self.brightness, 1 + self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if self.contrast:
            factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)
        return img

    def transform(self, X):
        def _to_pil(img):
            return Image.fromarray(img) if isinstance(img, np.ndarray) else img
        
        if self.train_mode:
            return np.array([np.array(self._augment_one(_to_pil(img))) for img in X])
        
        return np.array([np.array(_to_pil(img)) for img in X])

########################## End of Preprocessing ##########################


########################## Build Pipeline ##########################

def build_image_pipeline(
    target_size: Tuple[int, int] = (64, 64),
    mode: str = "RGB",
    grayscale: bool = False,
    normalize: str = "minmax",
    flatten: bool = True,
    use_pca: bool = False,
    pca_components: float = 0.95
) -> Pipeline:
    """
    Build a sklearn-compatible image preprocessing pipeline.

    Args:
        target_size : (width, height) to resize images
        mode        : PIL mode for loading ('RGB', 'L')
        grayscale   : convert RGB to grayscale after resize
        normalize   : 'minmax', 'standard', or 'none'
        flatten     : flatten images to 1-D (required for classical ML)
        use_pca     : apply PCA dimensionality reduction (requires flatten=True)
        pca_components: number of components or variance ratio for PCA

    Returns:
        sklearn Pipeline
    """
    steps = [("resize", ImageResizer(target_size=target_size, mode=mode))]

    if grayscale:
        steps.append(("grayscale", GrayscaleConverter()))

    steps.append(("normalize", ImageNormalizer(method=normalize)))

    if flatten:
        steps.append(("flatten", ImageFlattener()))
        if use_pca:
            steps.append(("pca", PCA(n_components=pca_components)))

    return Pipeline(steps)


########################## End of Build Pipeline ##########################


########################## Load Data ##########################

def load_images_from_dir(
    root_dir: str,
    mode: str = "RGB"
) -> Tuple[List[Image.Image], List[str]]:
    """
    Load images from a directory with the structure:
        root_dir/
            class_a/img1.jpg, img2.png, ...
            class_b/img3.jpg, ...

    Args:
        root_dir : path to root directory
        mode     : PIL image mode ('RGB', 'L', 'RGBA')

    Returns:
        images: list of PIL Images (not yet resized/processed)
        labels: list of class name strings
    """
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images: List[Image.Image] = []
    labels: List[str] = []
    root_path = Path(root_dir)

    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    for class_dir in tqdm(class_dirs, desc="Loading classes"):
        class_name = class_dir.name
        img_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
        for img_path in img_paths:
            img = Image.open(img_path).convert(mode)
            images.append(img)
            labels.append(class_name)

    return images, labels


def get_image_paths_and_labels(root_dir: str) -> Tuple[List[str], List[str]]:
    """
    Return (image_paths, labels) without loading images into memory.
    Useful for EDA on large datasets.

    Args:
        root_dir: path to root directory (same structure as load_images_from_dir)

    Returns:
        image_paths: list of absolute path strings
        labels: list of class name strings
    """
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths: List[str] = []
    labels: List[str] = []
    root_path = Path(root_dir)

    for class_dir in sorted(root_path.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in VALID_EXTS:
                image_paths.append(str(img_path))
                labels.append(class_dir.name)

    return image_paths, labels


########################## End of Load Data ##########################


########################## Multi Preprocess ##########################

def multi_preprocess(
    preprocess_config: List[Dict],
    images: List,
    labels: List,
    save_dir: str
) -> pd.DataFrame:
    """
    Process images with multiple configs and save splits to disk.
    Re-uses previously saved results (skips if path exists).

    Each config dict may include:
        - target_size   : tuple (W, H), default (64, 64)
        - mode          : str, PIL mode, default 'RGB'
        - grayscale     : bool, default False
        - normalize     : 'minmax' | 'standard' | 'none', default 'minmax'
        - use_pca       : bool, default False
        - pca_components: float or int, default 0.95

    Args:
        preprocess_config : list of config dicts
        images            : list of PIL Images
        labels            : list of label strings
        save_dir          : root directory to save processed data

    Returns:
        DataFrame with config columns + 'save_path'
    """
    results = pd.DataFrame(columns=list(preprocess_config[0].keys()) + ["save_path"])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    for config in tqdm(preprocess_config, desc="Preprocessing configs"):
        target_size   = config.get("target_size",    (64, 64))
        mode          = config.get("mode",           "RGB")
        grayscale     = config.get("grayscale",      False)
        normalize     = config.get("normalize",      "minmax")
        use_pca       = config.get("use_pca",        False)
        pca_components = config.get("pca_components", 0.95)

        size_str = f"{target_size[0]}x{target_size[1]}"
        path = os.path.join(
            save_dir,
            f"size={size_str},mode={mode},gray={grayscale},norm={normalize},pca={use_pca}/"
        )
        results.loc[len(results)] = [config.get(k) for k in preprocess_config[0].keys()] + [path]

        if os.path.exists(path):
            continue
        os.makedirs(path)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            images, y, test_size=0.2, random_state=42, stratify=y
        )

        pipe = build_image_pipeline(
            target_size=target_size,
            mode=mode,
            grayscale=grayscale,
            normalize=normalize,
            flatten=True,
            use_pca=use_pca,
            pca_components=pca_components
        )

        X_train = pipe.fit_transform(X_train_raw)
        X_test  = pipe.transform(X_test_raw)

        np.save(os.path.join(path, "X_train"), X_train)
        np.save(os.path.join(path, "X_test"),  X_test)
        np.save(os.path.join(path, "y_train"), y_train)
        np.save(os.path.join(path, "y_test"),  y_test)

    return results


########################## End of Multi Preprocess ##########################


########################## Training ##########################

def train_one_model(
    params: Dict,
    split_datasets: pd.DataFrame,
    score: str = "f1"
) -> pd.DataFrame:
    """
    Train one model configuration across multiple preprocessed datasets.

    Args:
        params        : dict with 'model' key (sklearn estimator) and optional params
        split_datasets: DataFrame from multi_preprocess with a 'save_path' column
        score         : 'f1' | 'accuracy'

    Returns:
        DataFrame (same columns as split_datasets minus 'save_path') + score column
    """
    results = split_datasets.drop(columns=["save_path"]).copy()
    results[score] = 0.0

    for i, data_path in tqdm(
        enumerate(split_datasets["save_path"]), total=len(split_datasets), desc="Training"
    ):
        X_train = np.load(os.path.join(data_path, "X_train.npy"))
        X_test  = np.load(os.path.join(data_path, "X_test.npy"))
        y_train = np.load(os.path.join(data_path, "y_train.npy")).ravel()
        y_test  = np.load(os.path.join(data_path, "y_test.npy")).ravel()

        model = Pipeline(steps=[("model", params["model"])])
        model.set_params(**{k: v for k, v in params.items() if k != "model"})

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if score == "f1":
            results.at[i, score] = f1_score(y_test, y_pred, average="weighted")
        elif score == "accuracy":
            results.at[i, score] = accuracy_score(y_test, y_pred)
        else:
            raise ValueError(f"Unknown score '{score}', use 'f1' or 'accuracy'")

    return results


def train_multi_model(
    grid: List[Dict],
    params_df: pd.DataFrame,
    data_path: str,
    score: str = "f1"
) -> pd.DataFrame:
    """
    Train multiple model configurations on a single preprocessed dataset.

    Args:
        grid      : list of param dicts, each must contain a 'model' key
        params_df : DataFrame describing the grid parameters (one row per config)
        data_path : path to preprocessed data directory (containing .npy files)
        score     : 'f1' | 'accuracy'

    Returns:
        DataFrame (same as params_df) + score column
    """
    results_df = params_df.copy()
    results_df[score] = 0.0

    X_train = np.load(os.path.join(data_path, "X_train.npy"))
    X_test  = np.load(os.path.join(data_path, "X_test.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy")).ravel()
    y_test  = np.load(os.path.join(data_path, "y_test.npy")).ravel()

    for i, params in tqdm(
        enumerate(grid), total=len(grid), desc="Training"
    ):
        model = Pipeline(steps=[("model", params["model"])])
        model.set_params(**{k: v for k, v in params.items() if k != "model"})

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if score == "f1":
            results_df.at[i, score] = f1_score(y_test, y_pred, average="weighted")
        elif score == "accuracy":
            results_df.at[i, score] = accuracy_score(y_test, y_pred)
        else:
            raise ValueError(f"Unknown score '{score}', use 'f1' or 'accuracy'")

    return results_df

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    score: str = "f1"
) -> np.ndarray:
    """
    K-fold cross-validation cho image classifier.
    In fold-by-fold scores + mean ± std.

    Args:
        model : sklearn estimator (đã được khởi tạo).
        X     : feature array (N, d).
        y     : label array (N,).
        cv    : số folds, mặc định 5.
        score : 'f1' | 'accuracy', mặc định 'f1'.

    Returns:
        scores: numpy array kết quả từng fold.
    """
    scoring = "f1_weighted" if score == "f1" else score
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    print(f"\nCross-Validation ({cv}-fold) | metric: {score}")
    print("-" * 40)
    for i, s in enumerate(scores, start=1):
        print(f"  Fold {i}: {s:.4f}")
    print("-" * 40)
    print(f"  Mean : {scores.mean():.4f}")
    print(f"  Std  : {scores.std():.4f}")

    return scores


def hyperparameter_search(
    model,
    param_grid: Dict,
    X: np.ndarray,
    y: np.ndarray,
    method: str = "grid",
    cv: int = 5,
    score: str = "f1"
):
    """
    GridSearchCV / RandomizedSearchCV wrapper.
    In best_params_ và best_score_.

    Args:
        model      : sklearn estimator (đã được khởi tạo).
        param_grid : dict hoặc list of dicts các hyperparameter cần tìm kiếm.
        X          : feature array (N, d).
        y          : label array (N,).
        method     : 'grid' dùng GridSearchCV, 'random' dùng RandomizedSearchCV.
        cv         : số folds cross-validation, mặc định 5.
        score      : 'f1' | 'accuracy', mặc định 'f1'.

    Returns:
        searcher: fitted GridSearchCV hoặc RandomizedSearchCV object.
    """
    scoring = "f1_weighted" if score == "f1" else score

    if method == "grid":
        searcher = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    elif method == "random":
        searcher = RandomizedSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    else:
        raise ValueError(f"Unknown method '{method}', use 'grid' or 'random'")

    searcher.fit(X, y)

    print(f"\nHyperparameter Search ({method}SearchCV) | metric: {score}")
    print("-" * 40)
    print(f"  Best params : {searcher.best_params_}")
    print(f"  Best score  : {searcher.best_score_:.4f}")

    return searcher


########################## End of Training ##########################
