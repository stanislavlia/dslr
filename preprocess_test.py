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
    
    with open('standardizer.pkl', 'rb') as file:
        standardizer = pickle.load(file)

    df_standardized = standardizer.transform(df)
    
    with open('test_std.csv', 'w') as file:
        df_standardized.to_csv(file)
    
    # with open('vif_elim_cols.csv', 'r') as file:
    #     elim_cols = pd.read_csv(file)
    
    # elim_cols = elim_cols['dropped_feature'].tolist()
    
    # df_standardized.drop(columns=elim_cols, inplace=True)
        
    # with open('test_vif_elim.csv', 'w') as file:
    #     df_standardized.to_csv(file)

