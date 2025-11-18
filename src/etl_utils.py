from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("data/raw/dataset.csv")
PROCESSED_DATA_PATH = Path("data/processed/dataset_clean.csv")


@dataclass
class ValidationReport:
    """Container for validation outputs."""

    row_count: int
    unique_entities: int
    date_range: str
    null_counts: Dict[str, int]
    negative_values_columns: List[str]
    duplicate_rows: int


def extract_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Extract the dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    return pd.read_csv(path)


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, enrich, and validate the dataset."""
    transformed = df.copy()

    # Standardise column names
    transformed = transformed.rename(
        columns={
            "lifeExp": "life_expectancy",
            "gdpPercap": "gdp_per_capita",
            "pop": "population",
        }
    )

    # Remove duplicates and drop obvious invalid rows
    transformed = transformed.drop_duplicates()
    transformed = transformed.dropna(subset=["country", "continent", "year"])

    # Ensure numeric types
    numeric_cols = ["year", "population", "life_expectancy", "gdp_per_capita"]
    transformed[numeric_cols] = transformed[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Handle missing numeric values by forward/backward fill within country groups
    transformed[numeric_cols] = transformed.groupby("country")[numeric_cols].transform(
        lambda group: group.ffill().bfill()
    )

    # Remaining nulls -> global median
    transformed[numeric_cols] = transformed[numeric_cols].fillna(transformed[numeric_cols].median())

    # Feature engineering
    transformed["gdp_total"] = transformed["gdp_per_capita"] * transformed["population"]
    transformed["life_stage"] = pd.cut(
        transformed["life_expectancy"],
        bins=[0, 50, 65, 80, np.inf],
        labels=["developing", "emerging", "established", "longevity"],
        right=False,
    )

    # GDP growth rate per country
    transformed = transformed.sort_values(["country", "year"])
    transformed["gdp_per_capita_growth_pct"] = (
        transformed.groupby("country")["gdp_per_capita"].pct_change() * 100
    )
    transformed["life_expectancy_change"] = transformed.groupby("country")["life_expectancy"].diff()

    # Replace inf values from pct_change with NaN then fill with 0
    transformed = transformed.replace([np.inf, -np.inf], np.nan)
    transformed["gdp_per_capita_growth_pct"] = transformed["gdp_per_capita_growth_pct"].fillna(0)
    transformed["life_expectancy_change"] = transformed["life_expectancy_change"].fillna(0)

    return transformed


def validate_data(df: pd.DataFrame, entity_col: str = "country", date_col: str = "year") -> ValidationReport:
    """Generate a validation report for the transformed dataset."""
    null_counts = df.isna().sum().to_dict()
    
    # Check for negative values in numeric columns (excluding date/ID columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    negative_cols = [col for col in numeric_cols if (df[col] < 0).any()]

    # Get entity count and date range if columns exist
    unique_entities = df[entity_col].nunique() if entity_col in df.columns else 0
    if date_col in df.columns:
        date_range = f"{df[date_col].min()} - {df[date_col].max()}"
    else:
        date_range = "N/A"

    return ValidationReport(
        row_count=len(df),
        unique_entities=unique_entities,
        date_range=date_range,
        null_counts=null_counts,
        negative_values_columns=negative_cols,
        duplicate_rows=df.duplicated().sum(),
    )


def load_data(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    """Persist the transformed dataset to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_pipeline() -> ValidationReport:
    """Run the full ETL pipeline and return the validation report."""
    raw_df = extract_data()
    transformed_df = transform_data(raw_df)
    load_data(transformed_df)
    return validate_data(transformed_df)
