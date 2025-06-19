#!/usr/bin/env python3


import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from label_encode import label_encode


house_colors = ['#222f5b', '#2a623d', '#740001', '#ecb939']
labels = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']


def plot_feature(df, feature):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=feature, kde=True)
    plt.title(f'Histogram of {feature}')

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x=feature)
    plt.title(f'Boxplot of {feature}')

    plt.tight_layout()
    plt.show()


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

    df['Hogwarts House'] = label_encode(df['Hogwarts House'], 'house_mapping.json')

    with open('house_mapping.json', 'r') as json_file:
        houses = json.load(json_file)

    for feature in df.drop(columns=['Hogwarts House']).columns:
        plt.figure(figsize=(6, 4))
        for house in houses.keys():
            subset = df[df['Hogwarts House'] == int(house)]
            plt.hist(subset[feature],
                    alpha=0.5,
                    edgecolor="black",
                    color=house_colors[int(house)],
                    label=houses[house])
        plt.legend()
        plt.title(f'Histogram of {feature} by House')
        plt.show()
