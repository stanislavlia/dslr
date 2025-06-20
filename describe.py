#!/usr/bin/env python3


import sys
import pandas as pd
from custom import describe


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: describe.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    try:
        df = pd.read_csv(csv_file).set_index('Index')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    describe(df)
