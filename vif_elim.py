#!/usr/bin/env python3


import sys
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif_elimination(X, k=1, thresh=None, verbose=True):
    """
    Iteratively drop the variable with highest VIF.
    """
    X = X.dropna().copy()
    history = []

    for i in range(k):
        # Compute VIFs
        vif = pd.Series(
            [variance_inflation_factor(X.values, idx)
             for idx in range(X.shape[1])],
            index=X.columns
        )
        
        # Optionally stop if below threshold
        if thresh is not None and vif.max() <= thresh:
            if verbose:
                print(f"All VIFs ≤ {thresh:.2f}; stopping early at iteration {i}.")
            break
        
        # Identify and drop the worst offender
        worst_feature = vif.idxmax()
        worst_vif     = vif.max()
        history.append((worst_feature, worst_vif))
        
        if verbose:
            print(f"Iteration {i+1}: dropping '{worst_feature}' (VIF = {worst_vif:.2f})")
        
        X = X.drop(columns=[worst_feature])
    
    history_df = pd.DataFrame(history, columns=['dropped_feature', 'VIF'])
    return X, history_df

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: describe.py <csv_file>')
        sys.exit(1)

    csv_file = sys.argv[1]
    try:
        df = pd.read_csv(csv_file).set_index('Index')
    except Exception as e:
        print(f'Error reading CSV file: {e}')
        sys.exit(1)

    df_elim, elim_cols = vif_elimination(df, k=10, thresh=10)
    
    print("VIF elimination completed. Results:")
    print(elim_cols)
    
    with open('train_vif_elim.csv', 'w') as file:
        df_elim.to_csv(file)
        
    with open('vif_elim_cols.csv', 'w') as file:
        elim_cols.to_csv(file)
