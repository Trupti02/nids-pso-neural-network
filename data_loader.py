import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import COLUMNS, CATEGORICAL

def load_dataset(path):
    df = pd.read_csv(path, header=None)
    if str(df.iloc[0, 0]).strip().lower() == "duration":
        df = df.iloc[1:].reset_index(drop=True)
    df = df.iloc[:, :len(COLUMNS)]
    df.columns = COLUMNS[:df.shape[1]]
    for c in df.columns:
        if c not in CATEGORICAL + ["label"]:
            df[c] = pd.to_numeric(df[c])
    df["y"] = (df["label"].astype(str).str.strip().str.lower() != "normal").astype(int)
    return df.drop(columns=[c for c in ("label", "difficulty") if c in df.columns])

def get_feature_lists(df):
    feat = [c for c in df.columns if c != "y"]
    num = [c for c in feat if c not in CATEGORICAL]
    return feat, num

def build_preprocessor(num_cols):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), num_cols)])