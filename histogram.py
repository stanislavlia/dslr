#!/usr/bin/env python3


import sys
import pandas as pd
import matplotlib.pyplot as plt
from house_colors import house_colors, labels


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


    for feature in df.drop(columns=['Hogwarts House']).columns:
        if df[feature].dtype != 'object':
            plt.figure(figsize=(6, 4))
            for house_idx in range(4):
                subset = df[df['Hogwarts House'] == labels[house_idx]]
                plt.hist(subset[feature],
                        alpha=0.5,
                        edgecolor="black",
                        color=house_colors[house_idx],
                        label=labels[house_idx])
            plt.legend()
            plt.title(f'Histogram of {feature} by House')
            plt.show()
