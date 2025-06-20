#!/usr/bin/env python3


import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dslr_preprocess import target_decode
from house_colors import house_colors


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

    df['Hogwarts House'] = target_decode(df['Hogwarts House'])

    sns.pairplot(df,
                hue='Hogwarts House',
                palette=house_colors,
                height=0.7,
                aspect=1)
    plt.show()
