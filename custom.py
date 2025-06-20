import pandas as pd


def count(series):
    count = 0
    for value in series:
        if value is not None and not value != value:
            count += 1
    return count


def mean(series):
    count = 0
    total = 0
    for value in series:
        if value is not None and not value != value:
            count += 1
            total += value
    return total / count if count != 0 else float('nan')


def std(series):
    m = mean(series)
    c = 0
    ssd = 0  # sum of squared differences
    for value in series:
        if value is not None and not value != value:
            ssd += (value - m) ** 2
            c += 1
    return (ssd / (c - 1)) ** 0.5 if c > 1 else float('nan')


def min(series):
    min_val = None
    for value in series:
        if value is not None and not value != value:
            if min_val is None or value < min_val:
                min_val = value
    return min_val


def max(series):
    max_val =  None
    for value in series:
        if value is not None and not value != value:
            if max_val is None or value > max_val:
                max_val = value
    return max_val


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


def corr_custom(x, y):
    n = len(x)
    mx = mean(x)
    my = mean(y)
    # cov is the sample covariance between x and y:
    # cov = sum((xi - mean(x)) * (yi - mean(y))) / (n - 1)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    std_x = std(x)
    std_y = std(y)
    if std_x == 0 or std_y == 0:
        return float('nan')
    return cov / (std_x * std_y)


def corr_max(df):
    corr_max = 0
    corr_max_pair = []
    for x in df.columns:
        for y in df.columns:
            if x != y:
                corr = corr_custom(df[x], df[y])
                if abs(corr) > abs(corr_max):
                    corr_max = corr
                    corr_max_pair = [x, y]
    return corr_max_pair


def corr_matrix(df):
    cols = df.columns
    N = len(cols)
    matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i in cols:
        for j in cols:
            if i == j:
                matrix.loc[i, j] = 1.0
            else:
                matrix.loc[i, j] = corr_custom(df[i], df[j])
    return matrix
