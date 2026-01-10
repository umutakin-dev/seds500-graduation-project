"""
Analyze tabular data files (Excel, CSV) to understand structure and statistics.

Usage:
    python src/analyze_data.py path/to/data.xlsx
    python src/analyze_data.py path/to/data.csv --sheet "Sheet2"
"""

import argparse
import pandas as pd
import numpy as np
import sys
import io
from pathlib import Path

# Fix Unicode encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def analyze_dataframe(df: pd.DataFrame, name: str = "Dataset"):
    """Analyze a DataFrame and print summary statistics."""

    print("=" * 70)
    print(f"DATASET OVERVIEW: {name}")
    print("=" * 70)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Column info
    print("\n" + "=" * 70)
    print("COLUMNS")
    print("=" * 70)
    print(f"{'#':<3} {'Column Name':<40} {'Type':<12} {'Non-Null':<15} {'Unique':<10}")
    print("-" * 70)

    for i, col in enumerate(df.columns):
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        n_unique = df[col].nunique()
        col_display = col[:38] + ".." if len(str(col)) > 40 else col
        print(f"{i+1:<3} {col_display:<40} {dtype:<12} {non_null:>6}/{len(df):<6} {n_unique:<10}")

    # Data types summary
    print("\n" + "=" * 70)
    print("DATA TYPES SUMMARY")
    print("=" * 70)
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")

    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    print(f"\nNumeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Datetime columns: {len(datetime_cols)}")

    # Numeric statistics
    if numeric_cols:
        print("\n" + "=" * 70)
        print("NUMERIC COLUMNS STATISTICS")
        print("=" * 70)
        numeric_stats = df[numeric_cols].describe().T
        numeric_stats['missing'] = len(df) - df[numeric_cols].count()
        print(numeric_stats[['count', 'mean', 'std', 'min', 'max', 'missing']].to_string())

    # Categorical statistics
    if categorical_cols:
        print("\n" + "=" * 70)
        print("CATEGORICAL COLUMNS (top 5 values each)")
        print("=" * 70)
        for col in categorical_cols[:10]:  # Limit to first 10 categorical columns
            print(f"\n{col}:")
            value_counts = df[col].value_counts().head(5)
            for val, count in value_counts.items():
                val_display = str(val)[:50] + "..." if len(str(val)) > 50 else val
                print(f"  {val_display}: {count} ({count/len(df)*100:.1f}%)")

    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].sort_values(ascending=False)
    if len(missing_cols) > 0:
        print("\n" + "=" * 70)
        print("MISSING VALUES")
        print("=" * 70)
        for col, count in missing_cols.head(15).items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        if len(missing_cols) > 15:
            print(f"  ... and {len(missing_cols) - 15} more columns with missing values")

    # Sample rows
    print("\n" + "=" * 70)
    print("SAMPLE ROWS (first 3)")
    print("=" * 70)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 200)
    print(df.head(3).to_string())

    return {
        'shape': df.shape,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'datetime_cols': datetime_cols,
        'missing_cols': missing_cols.to_dict() if len(missing_cols) > 0 else {}
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze tabular data files")
    parser.add_argument("file", type=str, help="Path to data file (Excel or CSV)")
    parser.add_argument("--sheet", type=str, default=None, help="Sheet name for Excel files")
    parser.add_argument("--list-sheets", action="store_true", help="List all sheets in Excel file")
    args = parser.parse_args()

    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    # Determine file type and read
    if file_path.suffix.lower() in ['.xlsx', '.xls']:
        if args.list_sheets:
            xl = pd.ExcelFile(file_path)
            print(f"Sheets in {file_path.name}:")
            for i, sheet in enumerate(xl.sheet_names):
                print(f"  {i+1}. {sheet}")
            return

        sheet_name = args.sheet if args.sheet else 0
        print(f"Reading Excel file: {file_path.name} (sheet: {sheet_name})")
        df = pd.read_excel(file_path, sheet_name=sheet_name)

    elif file_path.suffix.lower() == '.csv':
        print(f"Reading CSV file: {file_path.name}")
        df = pd.read_csv(file_path)
    else:
        print(f"Error: Unsupported file type: {file_path.suffix}")
        return

    analyze_dataframe(df, file_path.name)


if __name__ == "__main__":
    main()
