#!/usr/bin/env python3

import sys
import pickle
import pandas as pd
from Standardizer_dslr import StandardizeDf
from dslr_preprocess import dslr_preprocess


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


    df_processed = dslr_preprocess(df)

    standardizer = StandardizeDf('Hogwarts House')
    
    standardizer.fit(df_processed)
    
    with open('standardizer.pkl', 'wb') as file:
        pickle.dump(standardizer, file)
    
    df_standardized = standardizer.transform(df)
    
    with open('train_std.csv', 'w') as file:
        df_standardized.to_csv(file)
