import pandas as pd
import os
import json

DATA_DIR = "data"
OUTPUT_FILE = "metadata.json"

SKIP_COLS = {"Measure Name", "Measure Name ", "Unit"}


def is_english(text):
    try:
        str(text).encode("ascii")
        return True
    except (UnicodeEncodeError, AttributeError):
        return False


def extract_metadata(filepath):
    xl = pd.ExcelFile(filepath)

    # --- description from Cover Page ---
    cover = pd.read_excel(filepath, sheet_name="Cover Page", header=None)
    name_row = cover[cover[0] == "Template Name"]
    name = str(name_row.iloc[0, 1]).strip() if not name_row.empty else "Unknown"

    # --- Relational DB ---
    df = pd.read_excel(filepath, sheet_name="Relational DB", skiprows=2)

    # English dimension columns (skip standard ones, skip Unnamed)
    dimensions = [
        str(c).strip()
        for c in df.columns
        if is_english(str(c))
        and str(c).strip() not in SKIP_COLS
        and not str(c).startswith("Unnamed")
    ]

    # Year column = second-to-last column
    year_col = df.columns[-2]
    years = (
        sorted(df[year_col].dropna().astype(int).unique().tolist())
        if pd.api.types.is_numeric_dtype(df[year_col])
        else []
    )

    # Chapter from folder name
    parts = filepath.replace("\\", "/").split("/")
    chapter = parts[1] if len(parts) > 2 else "unknown"

    return {
        "name": name,
        "dimensions": dimensions,
        "years": years,
        "chapter": chapter,
        "path": filepath.replace("\\", "/"),
    }


def build_metadata():
    metadata = {}
    errors = []

    for root, dirs, files in os.walk(DATA_DIR):
        for fname in sorted(files):
            if not fname.endswith(".xlsx"):
                continue
            if "Copy" in fname or fname.startswith("~"):
                continue

            filepath = os.path.join(root, fname)
            try:
                meta = extract_metadata(filepath)
                key = filepath.replace("\\", "/")
                metadata[key] = meta
                print(f"OK  {key}")
                print(f"    {meta['name']}")
                print(f"    dims: {meta['dimensions']}")
                print(f"    years: {meta['years'][:3]}...{meta['years'][-1] if meta['years'] else 'N/A'}")
                print()
            except Exception as e:
                errors.append((filepath, str(e)))
                print(f"ERR {filepath} -> {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {len(metadata)} files saved to {OUTPUT_FILE}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for path, err in errors:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    build_metadata()
