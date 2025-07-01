import pandas as pd


def is_nan(value):
    return value != value


def count(series):
    count = 0
    for value in series:
        if value is not None and not is_nan(value):
            count += 1
    return count


def mean(series):
    count = 0
    total = 0
    for value in series:
        if value is not None and not is_nan(value):
            count += 1
            total += value
    return total / count if count != 0 else float('nan')


def std(series):
    mean_value = mean(series)
    count = 0
    ssd = 0  # sum of squared differences
    for value in series:
        if value is not None and not is_nan(value):
            ssd += (value - mean_value) ** 2
            count += 1
    return (ssd / (count - 1)) ** 0.5 if count > 1 else float('nan')


def min(series):
    min_val = None
    for value in series:
        if value is not None and not is_nan(value):
            if min_val is None or value < min_val:
                min_val = value
    return min_val


def max(series):
    max_val =  None
    for value in series:
        if value is not None and not is_nan(value):
            if max_val is None or value > max_val:
                max_val = value
    return max_val


def percentile(series, percentile):
    # Filter out None and NaN values
    valid_values = [v for v in series if v is not None and not is_nan(v)]

    values_amount = len(valid_values)
    if values_amount == 0:
        return float('nan')
    
    # Sort values
    valid_values.sort()

    # Find index for specified precentile
    float_index = (values_amount - 1) * percentile / 100
    floor_index = int(float_index)
    ceil_index = floor_index + 1

    if ceil_index >= values_amount:
        return valid_values[floor_index]
    
    # Interpolate between values at floor and ceil indices
    weight_floor = valid_values[floor_index] * (ceil_index - float_index)
    weight_ceil = valid_values[ceil_index] * (float_index - floor_index)
    return weight_floor + weight_ceil


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
        stats['count'].append(count(df[col]))
        stats['mean'].append(mean(df[col]))
        stats['std'].append(std(df[col]))
        stats['min'].append(min(df[col]))
        stats['25%'].append(percentile(df[col], 25))
        stats['50%'].append(percentile(df[col], 50))
        stats['75%'].append(percentile(df[col], 75))
        stats['max'].append(max(df[col]))

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


def covariance(series_x, series_y):
    mean_x = mean(series_x)
    mean_y = mean(series_y)
    count = 0
    cov_sum = 0
    for x, y in zip(series_x, series_y):
        if (x is not None and not is_nan(x)) and (y is not None and not is_nan(y)):
            cov_sum += (x - mean_x) * (y - mean_y)
            count += 1
    cov_value = cov_sum / (count - 1) if count > 1 else float('nan')
    return cov_value


def correlation(series_x, series_y):
    std_x = std(series_x)
    std_y = std(series_y)
    if std_x == 0 or std_y == 0 or is_nan(std_x) or is_nan(std_y):
        return float('nan')
    corr_value = covariance(series_x, series_y) / (std_x * std_y)
    return corr_value


def most_correlated_pair(df):
    max_corr = float()
    best_pair = (None, None)
    for i in df.columns:
        for j in df.columns:
            if i == j:
                continue
            x = df[i]
            y = df[j]
            corr = correlation(x, y)
            if not is_nan(corr) and abs(corr) > abs(max_corr):
                max_corr = corr
                best_pair = (i, j)
    return best_pair
