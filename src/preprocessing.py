# -*- coding: utf-8 -*-
"""
Production Preprocessing & Feature Engineering Pipeline.

Provides reusable, scikit-learn-compatible transformers and pipelines
for data ingestion, localized imputation, domain feature engineering,
and column transformation without data leakage.
"""

from typing import Dict, List, Optional, Tuple, Any, cast
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.utils import DATA_PROCESSED_PATH

# -----------------------------------------------------------------------------
# Feature Column Definitions
# -----------------------------------------------------------------------------

TARGET_COLUMNS = [
    "Good_Investment",
    "Future_Price_5Y",
    "Investment_Score",
    "Annual_Appreciation_Rate",
    "City_Tier_Base_Rate",
]

DROP_COLUMNS = [
    "ID",
    "Good_Investment",
    "Future_Price_5Y",
    "Investment_Score",
    "Annual_Appreciation_Rate",
    "City_Tier_Base_Rate",
    "Locality_Median_PPS",
    "Is_Below_Locality_Median_PPS",
]

NUMERIC_FEATURES = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt_Capped",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "School_Hospital_Density",
    "Floor_Ratio",
    "Infrastructure_Score",
    "Amenities_Count",
    "Public_Transport_Score",
    "Furnished_Score",
]

BINARY_FEATURES = [
    "Has_Pool",
    "Has_Gym",
    "Has_Clubhouse",
    "Has_Garden",
    "Has_Playground",
    "Has_Security",
    "Has_Parking",
    "Is_Below_Locality_Median_PPS",
]

CATEGORICAL_FEATURES = [
    "Property_Type",
    "Facing",
    "Owner_Type",
    "Availability_Status",
    "City",
    "State",
]


# -----------------------------------------------------------------------------
# Custom Scikit-Learn Transformers
# -----------------------------------------------------------------------------

class PropertyFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer that derives all analytical and
    domain features from raw or cleaned property listings.
    
    Prevents data leakage by fitting localized pricing benchmarks and
    capping thresholds exclusively on training splits.
    """

    def __init__(self, ref_year: int = 2025) -> None:
        self.ref_year = ref_year
        self.locality_medians_: Dict[str, float] = {}
        self.city_medians_: Dict[str, float] = {}
        self.global_median_pps_: float = 13000.0
        self.pps_cap_upper_: float = 80000.0
        self.num_medians_: Dict[str, float] = {}
        self.cat_modes_: Dict[str, str] = {}

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "PropertyFeatureEngineer":
        """
        Learn localized medians, global fallbacks, and outlier capping bounds
        from the training split.
        """
        X_df = X.copy()

        # Compute or extract raw Price_per_SqFt for benchmark estimation
        if "Price_per_SqFt" not in X_df.columns:
            pps = (X_df["Price_in_Lakhs"] * 100000.0) / np.maximum(X_df["Size_in_SqFt"], 1.0)
        else:
            pps = X_df["Price_per_SqFt"]
        X_df["Price_per_SqFt"] = pps

        # Learn locality and city median price-per-sqft benchmarks
        if "Locality" in X_df.columns:
            loc_dict: Dict[Any, Any] = X_df.groupby("Locality")["Price_per_SqFt"].median().to_dict()
            self.locality_medians_ = {str(k): float(val) for k, val in loc_dict.items()}
        if "City" in X_df.columns:
            city_dict: Dict[Any, Any] = X_df.groupby("City")["Price_per_SqFt"].median().to_dict()
            self.city_medians_ = {str(k): float(val) for k, val in city_dict.items()}

        pps_vals = np.asarray(X_df["Price_per_SqFt"].dropna(), dtype=float)
        self.global_median_pps_ = float(np.nanmedian(pps_vals)) if len(pps_vals) > 0 else 13000.0
        self.pps_cap_upper_ = float(np.nanpercentile(pps_vals, 99.5)) if len(pps_vals) > 0 else 80000.0

        # Learn fallback medians and modes for imputation
        for col in ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Floor_No", "Total_Floors", 
                    "Year_Built", "Nearby_Schools", "Nearby_Hospitals"]:
            if col in X_df.columns:
                c_vals = np.asarray(X_df[col].dropna(), dtype=float)
                self.num_medians_[col] = float(np.nanmedian(c_vals)) if len(c_vals) > 0 else 0.0

        for col in ["State", "City", "Locality", "Property_Type", "Furnished_Status", 
                    "Facing", "Owner_Type", "Availability_Status", 
                    "Public_Transport_Accessibility", "Parking_Space", "Security"]:
            if col in X_df.columns:
                mode_vals = X_df[col].dropna().mode()
                self.cat_modes_[col] = str(mode_vals.iloc[0]) if not mode_vals.empty else "Unknown"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw listing inputs into fully enriched feature matrix.
        """
        df = X.copy()

        # 1. Text Normalization & Whitespace Harmonization
        for col, default_val in self.cat_modes_.items():
            if col in df.columns:
                col_s = cast(pd.Series, df[col])
                df[col] = col_s.fillna(default_val).astype(str).str.strip()
            else:
                df[col] = default_val

        if "Furnished_Status" in df.columns:
            col_furn = cast(pd.Series, df["Furnished_Status"])
            df["Furnished_Status"] = col_furn.replace({
                "Semi-furnished": "Semi-Furnished",
                "semi-furnished": "Semi-Furnished",
                "furnished": "Furnished",
                "unfurnished": "Unfurnished"
            })

        if "Availability_Status" in df.columns:
            col_avail = cast(pd.Series, df["Availability_Status"])
            df["Availability_Status"] = col_avail.replace({
                "Ready_to_Move": "Ready to Move",
                "Under_Construction": "Under Construction"
            })

        # 2. Numeric Imputation & Casting
        for col, default_val in self.num_medians_.items():
            if col in df.columns:
                num_s = cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))
                df[col] = num_s.fillna(default_val)
            else:
                df[col] = default_val

        # 3. Floor Sanity Correction (Floor_No <= Total_Floors)
        df["Total_Floors"] = np.maximum(df["Floor_No"], df["Total_Floors"])

        # 4. Dynamic Property Age
        has_age = "Age_of_Property" in df.columns and not bool(df["Age_of_Property"].isnull().all())
        if not has_age:
            df["Age_of_Property"] = np.maximum(0, self.ref_year - df["Year_Built"])

        # 5. Exact Price per SqFt Calculation & Outlier Capping
        df["Price_per_SqFt"] = ((df["Price_in_Lakhs"] * 100000.0) / np.maximum(df["Size_in_SqFt"], 1.0)).round(2)
        df["Price_per_SqFt_Capped"] = df["Price_per_SqFt"].clip(upper=self.pps_cap_upper_)

        # 6. Unpack Amenities
        amenities_str = df["Amenities"].astype(str) if "Amenities" in df.columns else pd.Series([""] * len(df))
        all_amenities = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]
        for amenity in all_amenities:
            col_name = f"Has_{amenity}"
            if col_name not in df.columns:
                df[col_name] = amenities_str.str.contains(amenity, na=False).astype(int)

        df["Amenities_Count"] = df[[f"Has_{a}" for a in all_amenities]].sum(axis=1)

        # 7. Security and Parking Binary Indicators
        if "Has_Security" not in df.columns:
            sec_series = df["Security"].astype(str).str.strip().str.lower()
            df["Has_Security"] = sec_series.isin(["yes", "gated", "guard", "cctv", "1", "true"]).astype(int)

        if "Has_Parking" not in df.columns:
            park_series = df["Parking_Space"].astype(str).str.strip().str.lower()
            df["Has_Parking"] = park_series.isin(["yes", "1", "true"]).astype(int)

        # 8. Ordinal Scales
        transport_map = {"Low": 1, "Medium": 2, "High": 3}
        furnished_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}

        df["Public_Transport_Score"] = df["Public_Transport_Accessibility"].apply(
            lambda val: transport_map.get(str(val).strip(), 2)
        ).astype(int)
        df["Furnished_Score"] = df["Furnished_Status"].apply(
            lambda val: furnished_map.get(str(val).strip(), 1)
        ).astype(int)

        # 9. Composite Domain Features
        df["School_Hospital_Density"] = df["Nearby_Schools"] + df["Nearby_Hospitals"]
        df["Floor_Ratio"] = (df["Floor_No"] / np.maximum(df["Total_Floors"], 1)).round(4)

        t_norm = (df["Public_Transport_Score"] - 1) / 2.0
        sh_norm = (df["School_Hospital_Density"] - 2) / 18.0
        am_norm = df["Amenities_Count"] / 5.0

        df["Infrastructure_Score"] = (
            0.25 * t_norm +
            0.25 * sh_norm +
            0.20 * df["Has_Security"] +
            0.15 * df["Has_Parking"] +
            0.15 * am_norm
        ) * 100.0
        df["Infrastructure_Score"] = df["Infrastructure_Score"].round(2)

        # 10. Locality Relative Price Index
        loc_med = df["Locality"].apply(
            lambda loc: self.locality_medians_.get(str(loc), np.nan)
        )
        city_med = df["City"].apply(
            lambda c: self.city_medians_.get(str(c), np.nan)
        )
        combined_med = loc_med.fillna(city_med).fillna(self.global_median_pps_)
        df["Locality_Median_PPS"] = combined_med
        df["Is_Below_Locality_Median_PPS"] = (df["Price_per_SqFt"] <= df["Locality_Median_PPS"]).astype(int)

        return df


# -----------------------------------------------------------------------------
# Pipeline Builders
# -----------------------------------------------------------------------------

def get_preprocessor(
    numeric_features: Optional[List[str]] = None,
    binary_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build a Scikit-Learn ColumnTransformer for scaling continuous features,
    passing through binary indicators, and one-hot encoding categoricals.
    """
    num_cols = numeric_features or NUMERIC_FEATURES
    bin_cols = binary_features or BINARY_FEATURES
    cat_cols = categorical_features or CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("bin", "passthrough", bin_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def build_pipeline(
    model: BaseEstimator,
    include_feature_engineering: bool = True,
    numeric_features: Optional[List[str]] = None,
    binary_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Pipeline:
    """
    Construct an end-to-end Scikit-Learn Pipeline combining feature engineering,
    column transformation, and model estimation.
    """
    preprocessor = get_preprocessor(numeric_features, binary_features, categorical_features)
    
    steps: List[Tuple[str, Any]] = []
    if include_feature_engineering:
        steps.append(("feature_engineer", PropertyFeatureEngineer()))
    steps.append(("preprocessor", preprocessor))
    steps.append(("model", model))

    return Pipeline(steps=steps)


# -----------------------------------------------------------------------------
# Dataset Ingestion & Splitting Helpers
# -----------------------------------------------------------------------------

def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the cleaned housing dataset.
    """
    path = filepath or str(DATA_PROCESSED_PATH)
    df = pd.read_csv(path)
    return df


def get_train_test_split(
    df: Optional[pd.DataFrame] = None,
    target: str = "Good_Investment",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare feature matrix X and target y with data leakage safeguards.
    Drops target columns, intermediate target scores, and IDs from X.
    Applies stratified splitting for binary classification.
    """
    data = load_data() if df is None else df.copy()

    if target not in data.columns:
        raise ValueError(f"Target '{target}' not found in dataset columns: {list(data.columns)}")

    y = data[target]

    # Exclude all target-related columns and leakage features from X
    cols_to_drop = [c for c in DROP_COLUMNS if c in data.columns]
    X = data.drop(columns=cols_to_drop)

    stratify = y if target == "Good_Investment" else None
    res = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    X_train = cast(pd.DataFrame, res[0])
    X_test = cast(pd.DataFrame, res[1])
    y_train = cast(pd.Series, res[2])
    y_test = cast(pd.Series, res[3])

    return X_train, X_test, y_train, y_test


def get_feature_names(fitted_pipeline: Pipeline) -> List[str]:
    """
    Extract human-readable feature names after preprocessor transformations.
    """
    preprocessor: ColumnTransformer = fitted_pipeline.named_steps["preprocessor"]
    feature_names: List[str] = []

    for name, trans, cols in preprocessor.transformers_:
        if name == "num" and isinstance(cols, list):
            feature_names.extend(cols)
        elif name == "bin" and isinstance(cols, list):
            feature_names.extend(cols)
        elif name == "cat" and hasattr(trans, "get_feature_names_out"):
            cat_out = trans.get_feature_names_out(cols)
            feature_names.extend(list(cat_out))

    return feature_names
