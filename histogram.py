#!/usr/bin/env python3


import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
            plt.figure(figsize=(16, 9))
            for house_idx in range(4):
                subset = df[df['Hogwarts House'] == labels[house_idx]]
                # Histogram
                plt.hist(subset[feature],
                        alpha=0.5,
                        edgecolor="black",
                        bins=32,
                        density=True,
                        color=house_colors[house_idx],
                        label=labels[house_idx])
                #KDE (Kernel Density Estimate) curve
                sns.kdeplot(subset[feature],
                            color=house_colors[house_idx],
                            linestyle='-',
                            linewidth=4)
            plt.legend()
            plt.title(f'Histogram and KDE of {feature} by House')
            plt.ylabel('Density')
            plt.xlabel(feature)
            plt.show()
