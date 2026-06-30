"""
build_config.py
───────────────
Generates soybean_train.json and soybean_test.json for:
  Train years : 2018, 2019, 2020, 2021
  Test  years : 2022
  States      : Illinois (IL), Iowa (IA), Minnesota (MN),
                North Carolina (NC), North Dakota (ND)

Long-term HRRR: 3 prior years (year-3, year-2, year-1)
Short-term HRRR: April–September (6 months)
Sentinel       : 2 seasons per year (Apr-Jun, Jul-Sep)

USAGE (run locally or in Colab before training):
  python build_config.py

PATHS  (edit these two lines to match your setup):
  CSV_PATH   : county_info CSV with columns: FIPS, County, State
  OUTPUT_DIR : where to write the JSON files
"""

import json
import os
import pandas as pd

# ── EDIT THESE PATHS ─────────────────────────────────────────────────────────
CSV_PATH   = "county_info.csv"          # your county info CSV
OUTPUT_DIR = "data"                     # folder for output JSON files
# ─────────────────────────────────────────────────────────────────────────────

TARGET_STATES = {
    "ILLINOIS":      "IL",
    "IOWA":          "IA",
    "MINNESOTA":     "MN",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA":  "ND",
}

TRAINING_YEARS = [2018, 2019, 2020, 2021]
TESTING_YEARS  = [2022]
NUM_LONG_YEARS = 3   # years of historical HRRR data per sample


# ─────────────────────────────────────────────────────────────────────────────

def load_counties(csv_path: str) -> list[dict]:
    """Load and filter the county CSV for our target states."""
    df = pd.read_csv(csv_path)

    # Normalise column names
    df.columns = df.columns.str.strip()

    # Keep only target states (handle both upper and title case)
    df["State"] = df["State"].str.upper().str.strip()
    df = df[df["State"].isin(TARGET_STATES.keys())].copy()
    df["State"] = df["State"].map(TARGET_STATES)

    # Ensure FIPS is zero-padded 5-digit string
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)

    return df.to_dict("records")


def build_short_term(fips: str, state: str, year: int) -> list[str]:
    """6 monthly HRRR CSVs covering the growing season (Apr–Sep)."""
    sc = fips[:2]
    return [
        f"WRF-HRRR Computed Dataset/data/{year}/{state}/HRRR_{sc}_{state}_{year}-{str(m).zfill(2)}.csv"
        for m in range(4, 10)
    ]


def build_long_term(fips: str, state: str, year: int) -> list[list[str]]:
    """
    3 prior years × 12 months of monthly HRRR CSVs.
    E.g. for year=2018 → years 2015, 2016, 2017.
    """
    sc = fips[:2]
    return [
        [
            f"WRF-HRRR Computed Dataset/data/{y}/{state}/HRRR_{sc}_{state}_{y}-{str(m).zfill(2)}.csv"
            for m in range(1, 13)
        ]
        for y in range(year - NUM_LONG_YEARS, year)
    ]


def build_sentinel(fips: str, state: str, year: int) -> list[str]:
    """2 seasonal Sentinel-2 HDF5 files: spring (Apr–Jun) and summer (Jul–Sep)."""
    sc = fips[:2]
    return [
        f"Sentinel-2 Imagery/data/AG/{year}/{state}/Agriculture_{sc}_{state}_{year}-04-01_{year}-06-30.h5",
        f"Sentinel-2 Imagery/data/AG/{year}/{state}/Agriculture_{sc}_{state}_{year}-07-01_{year}-09-30.h5",
    ]


def build_one_entry(year: int, county: dict) -> dict:
    fips   = str(county["FIPS"]).zfill(5)
    state  = county["State"]
    name   = county.get("County", "")
    return {
        "FIPS":        fips,
        "year":        year,
        "county":      name,
        "state":       state,
        "county_ansi": fips[2:],
        "state_ansi":  fips[:2],
        "data": {
            "HRRR": {
                "short_term": build_short_term(fips, state, year),
                "long_term":  build_long_term(fips, state, year),
            },
            # Note: filename uses "Soybean" (no s), folder uses "Soybeans"
            "USDA":     f"USDA Crop Dataset/data/Soybeans/{year}/USDA_Soybean_County_{year}.csv",
            "sentinel": build_sentinel(fips, state, year),
        },
    }


def build_and_save(years: list[int], output_path: str) -> int:
    counties = load_counties(CSV_PATH)
    data     = [build_one_entry(yr, c) for yr in years for c in counties]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data):5d} entries → {output_path}")
    return len(data)


if __name__ == "__main__":
    print("Generating JSON config files...")
    n_train = build_and_save(TRAINING_YEARS, os.path.join(OUTPUT_DIR, "soybean_train.json"))
    n_test  = build_and_save(TESTING_YEARS,  os.path.join(OUTPUT_DIR, "soybean_test.json"))
    print(f"\nDone!  Train: {n_train}  |  Test: {n_test}")
