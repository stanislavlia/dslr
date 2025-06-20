#!/usr/bin/env python3


import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from custom import corr_max
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


    corr_max_pair = corr_max(df.drop(columns=['Hogwarts House']))


    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=df,
                    x=corr_max_pair[0],
                    y=corr_max_pair[1],
                    hue='Hogwarts House',
                    palette=house_colors)
    plt.title('Visualization which displays a scatter plot answering the next question:\nWhat are the two features that are similar?')
    handles, _ = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels)
    plt.show()
