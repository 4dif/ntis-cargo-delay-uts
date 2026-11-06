"""
ntis_prep.py — Kelompok 08 (PT Sun Suka Abadi) — Proposal UTS Pembelajaran Mesin 2026
Modul rekayasa fitur & preprocessing. File ini WAJIB berada di folder yang sama dengan
model .joblib karena fungsi build_features dipanggil di dalam pipeline deployment.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, TargetEncoder

TARGET = "Delivery Status"
RAW_INPUT = ["Contrac date", "Waktu Penutupan Gudang", "STYLE", "QTYO", "Dest", "Smode", "Season", "FWD"]
FWD_KEEP = ["APL LOGISTICS", "UPS", "EXPEDITOR INDONESIA", "APEX", "EFL GLOBAL"]
SEASON_KEEP = ["2026 FAL", "2026 HOL"]

# Set fitur final UTS (7 fitur, tanpa Order_Month -> bebas risiko kebocoran temporal)
NUM_COLS = ["QTYO", "Closing_Month"]
CAT_COLS = ["Smode", "Dest", "FWD", "Season_Group"]
TE_COLS = ["STYLE"]
FEATURES = NUM_COLS + CAT_COLS + TE_COLS
# Set fitur Tugas 04-06 (8 fitur) -> hanya untuk ablasi
NUM_COLS_T06 = ["QTYO", "Order_Month", "Closing_Month"]


def load_raw(path):
    """Membaca CSV ekspor ERP NTIS (2 baris keterangan di atas header)."""
    df = pd.read_csv(path, skiprows=2)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df[df["PO"].notna()].reset_index(drop=True)


def labeled_subset(df):
    """Label final hanya untuk kargo yang sudah memiliki TANGGAL KAPAL BERANGKAT."""
    return df[df["TANGGAL KAPAL BERANGKAT"].notna()].reset_index(drop=True)


def build_features(df):
    """Mengubah kolom mentah ERP menjadi fitur pra-keberangkatan (tanpa kolom pasca-kejadian)."""
    out = pd.DataFrame(index=df.index)
    out["QTYO"] = pd.to_numeric(df["QTYO"], errors="coerce")
    out["Order_Month"] = pd.to_datetime(df["Contrac date"], errors="coerce").dt.month   # ablasi saja
    out["Closing_Month"] = pd.to_datetime(df["Waktu Penutupan Gudang"], errors="coerce").dt.month
    out["Smode"] = df["Smode"].astype("string")
    out["Dest"] = df["Dest"].astype("string")
    fwd = df["FWD"].astype("string")
    out["FWD"] = fwd.where(fwd.isin(FWD_KEEP), "lainnya")            # forwarder jarang -> 'lainnya'
    season = df["Season"].astype("string")
    out["Season_Group"] = season.where(season.isin(SEASON_KEEP), "lainnya")
    out["STYLE"] = df["STYLE"].astype("string")
    return out


def make_preprocessor(num_cols=NUM_COLS, seed=42):
    """ColumnTransformer: dipelajari HANYA dari data latih di setiap fold (dibungkus Pipeline)."""
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", RobustScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    te = TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return ColumnTransformer([("num", num, list(num_cols)), ("cat", cat, CAT_COLS), ("te", te, TE_COLS)],
                             remainder="drop", verbose_feature_names_out=False)


def make_imputer(num_cols=NUM_COLS):
    """Imputer awal (untuk skenario SMOTENC yang tidak menerima nilai hilang); output DataFrame."""
    ct = ColumnTransformer([("num", SimpleImputer(strategy="median"), list(num_cols)),
                            ("cat", SimpleImputer(strategy="most_frequent"), CAT_COLS + TE_COLS)],
                           remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")
