"""
⚡ Preprocessing Script for Housing Regression MLE

- Reads train/eval/holdout CSVs from data/raw/.
- Cleans and normalizes city names.
- Maps cities to metros and merges lat/lng.
- Drops duplicates and extreme outliers.
- Saves cleaned splits to data/processed/.

"""

"""
Preprocessing: city normalization + (optional) lat/lng merge, duplicate drop, outlier removal.

- Production defaults read from data/raw/ and write to data/processed/
- Tests can override `raw_dir`, `processed_dir`, and pass `metros_path=None`
  to skip merge safely without touching disk assets.
"""

import re
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Manual fixes for known mismatches (notebook-aligned labels)
CITY_MAPPING = {
    "Las Vegas-Henderson-Paradise": "Las Vegas-Henderson-North Las Vegas",
    "Denver-Aurora-Lakewood": "Denver-Aurora-Centennial",
    "Houston-The Woodlands-Sugar Land": "Houston-Pasadena-The Woodlands",
    "Austin-Round Rock-Georgetown": "Austin-Round Rock-San Marcos",
    "Miami-Fort Lauderdale-Pompano Beach": "Miami-Fort Lauderdale-West Palm Beach",
    "San Francisco-Oakland-Berkeley": "San Francisco-Oakland-Fremont",
    "DC_Metro": "Washington-Arlington-Alexandria",
    "Atlanta-Sandy Springs-Alpharetta": "Atlanta-Sandy Springs-Roswell",
}


def normalize_city(s: str) -> str:
    """Lowercase, strip, unify dashes. Safe for NA."""
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = re.sub(r"[–—-]", "-", s)          # unify dashes
    s = re.sub(r"\s+", " ", s)            # collapse spaces
    return s


def canonical_metro_name(s: str) -> str:
    """
    Normalize metro labels to a merge key.
    Example: "New York-Newark-Jersey City, NY-NJ" -> "new york-newark-jersey city"
    """
    if pd.isna(s):
        return s
    s = normalize_city(s)
    return re.sub(r",.*$", "", s).strip()


def clean_and_merge(df: pd.DataFrame, metros_path: str | None = "data/raw/usmetros.csv") -> pd.DataFrame:
    """
    Normalize city names, optionally merge lat/lng from metros dataset.
    If `city_full` column or `metros_path` is missing, skip gracefully.
    """

    if "city_full" not in df.columns:
        print("⚠️ Skipping city merge: no 'city_full' column present.")
        return df

    df = df.copy()
    # Keep notebook city labels, and merge via canonical keys.
    df["city_full"] = df["city_full"].replace(CITY_MAPPING)
    df["city_key"] = df["city_full"].apply(canonical_metro_name)

    # If no metros file provided / exists, skip merge
    if not metros_path or not Path(metros_path).exists():
        print("⚠️ Skipping lat/lng merge: metros file not provided or not found.")
        return df

    # Merge lat/lng
    metros = pd.read_csv(metros_path)
    if "metro_full" not in metros.columns or not {"lat", "lng"}.issubset(metros.columns):
        print("⚠️ Skipping lat/lng merge: metros file missing required columns.")
        return df

    metros["metro_key"] = metros["metro_full"].apply(canonical_metro_name)
    # Some metro names repeat across states (e.g., Albany). Keep the largest by population.
    if "population" in metros.columns:
        metros = metros.sort_values("population", ascending=False)
    metros = metros.drop_duplicates(subset=["metro_key"], keep="first")

    df = df.merge(
        metros[["metro_key", "lat", "lng"]],
        how="left",
        left_on="city_key",
        right_on="metro_key",
    )
    df.drop(columns=["city_key", "metro_key"], inplace=True, errors="ignore")

    missing = df[df["lat"].isnull()]["city_full"].unique()
    if len(missing) > 0:
        print("⚠️ Still missing lat/lng for:", missing)
    else:
        print("✅ All cities matched with metros dataset.")
    return df



def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates while keeping different dates/years."""
    before = df.shape[0]
    df = df.drop_duplicates(subset=df.columns.difference(["date", "year"]), keep=False)
    after = df.shape[0]
    print(f"✅ Dropped {before - after} duplicate rows (excluding date/year).")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers in median_list_price (> 19M)."""
    if "median_list_price" not in df.columns:
        return df
    before = df.shape[0]
    df = df[df["median_list_price"] <= 19_000_000].copy()
    after = df.shape[0]
    print(f"✅ Removed {before - after} rows with median_list_price > 19M.")
    return df


def preprocess_split(
    split: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = "data/raw/usmetros.csv",
) -> pd.DataFrame:
    """Run preprocessing for a split and save to processed_dir."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}.csv"
    df = pd.read_csv(path)

    df = clean_and_merge(df, metros_path=metros_path)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    out_path = processed_dir / f"cleaning_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Preprocessed {split} saved to {out_path} ({df.shape})")
    return df


def run_preprocess(
    splits: tuple[str, ...] = ("train", "eval", "holdout"),
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = "data/raw/usmetros.csv",
):
    for s in splits:
        preprocess_split(s, raw_dir=raw_dir, processed_dir=processed_dir, metros_path=metros_path)


if __name__ == "__main__":
    run_preprocess()
