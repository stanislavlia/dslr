#!/usr/bin/env python3


import sys
import pandas as pd


def count_mean(series):
    count = 0
    total = 0
    for value in series:
        if value is not None and value == value:
            count += 1
            total += value
    return count, total / count if count != 0 else float('nan')


def std(series):
    _, m = count_mean(series)
    c = 0
    ssd = 0  # sum of squared differences
    for value in series:
        if value is not None and value == value:
            ssd += (value - m) ** 2
            c += 1
    return (ssd / (c - 1)) ** 0.5 if c > 1 else float('nan')


def min_max(series):
    min_val = None
    max_val =  None
    for value in series:
        if value is not None and value == value:
            if min_val is None or value < min_val:
                min_val = value
            if max_val is None or value > max_val:
                max_val = value
    return min_val, max_val


def percentile(series, p):
    arr = [v for v in series if v is not None and not v != v]
    n = len(arr)
    if n == 0:
        return float('nan')
    arr.sort()
    k = (n - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= n:
        return arr[f]
    d0 = arr[f] * (c - k)
    d1 = arr[c] * (k - f)
    return d0 + d1


def describe(df):
    # Find numeric columns
    numeric_cols = []
    for col in df.columns:
        for val in df[col]:
            # Detect if this column is numerical by finding the first non-null value
            if val is not None and not val != val:
                if isinstance(val, (int, float)) and col.lower() != 'index':
                    numeric_cols.append(col)
                break

    # Header row
    row_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    stats = {name: [] for name in row_names}
    for col in numeric_cols:
        count, mean = count_mean(df[col])
        stats['count'].append(count)
        stats['mean'].append(mean)
        stats['std'].append(std(df[col]))
        min_val, max_val = min_max(df[col])
        stats['min'].append(min_val)
        stats['25%'].append(percentile(df[col], 25))
        stats['50%'].append(percentile(df[col], 50))
        stats['75%'].append(percentile(df[col], 75))
        stats['max'].append(max_val)

    # Print like pandas describe, cutting column names to max 16 characters
    print('{:>10}'.format(''), end='')
    for col in numeric_cols:
        print(' {:>16}'.format(str(col)[:16]), end='')
    print()
    for row in row_names:
        print('{:>10}'.format(row), end='')
        for val in stats[row]:
            print(' {:16.6f}'.format(val), end='')
        print()


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
