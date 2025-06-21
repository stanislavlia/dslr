#!/usr/bin/env python3


import sys
import pickle
import pandas as pd
from dslr_preprocess import target_decode


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: describe.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    try:
        with open('final_clf.pkl', 'rb') as file:
            clf = pickle.load(file)
        with open(csv_file, 'r') as file:
            df = pd.read_csv(file).set_index('Index')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    X_te = df.drop(columns='Hogwarts House', errors='ignore').to_numpy()

    y_pred = clf.predict(X_te)

    df_pred = pd.DataFrame(y_pred).set_index(df.index).rename(columns={0: 'Hogwarts House'})

    df_pred['Hogwarts House'] = target_decode(df_pred['Hogwarts House'])

    with open('houses.csv', 'w') as file:
        df_pred.to_csv(file)

    print("Predictions are saved to 'houses.csv'")
