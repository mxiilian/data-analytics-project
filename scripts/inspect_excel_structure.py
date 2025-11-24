import pandas as pd
import os

def inspect_file(filepath):
    print(f"--- Inspecting {filepath} ---")
    try:
        xl = pd.ExcelFile(filepath)
        print(f"Sheet names: {xl.sheet_names}")
        for sheet in xl.sheet_names:
            print(f"\nSheet: {sheet}")
            df = pd.read_excel(filepath, sheet_name=sheet, nrows=5)
            print("Columns:", df.columns.tolist())
            print(df.head())
            print("-" * 30)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

def main():
    files = [f for f in os.listdir('.') if f.endswith('.xlsx') and not f.startswith('~$')]
    if not files:
        print("No Excel files found.")
        return

    print(f"Found {len(files)} Excel files.")
    for f in files:
        inspect_file(f)

if __name__ == "__main__":
    main()

