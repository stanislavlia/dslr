import pandas as pd
from custom import mean, percentile


dict_encode = {'Ravenclaw': 0, 'Slytherin': 1, 'Gryffindor': 2, 'Hufflepuff': 3}
dict_decode = {0: 'Ravenclaw', 1: 'Slytherin', 2: 'Gryffindor', 3: 'Hufflepuff'}


def target_encode(series):
    return series.map(dict_encode)


def target_decode(series):
    return series.map(dict_decode)


def dslr_preprocess(df):
    df.drop(columns=['First Name',
                     'Last Name',
                     'Birthday',
                     'Best Hand',
                     'Arithmancy',
                     'Care of Magical Creatures'
                     ], inplace=True, errors='ignore')

    cols_with_nan = df.columns[df.isna().any()].tolist()
    cols_fill_median = cols_with_nan[:-2]
    cols_fill_mean = cols_with_nan[-2:]
    
    for col in cols_fill_median:
        df[col] = df[col].fillna(percentile(df[col], 50))
    for col in cols_fill_mean:
        df[col] = df[col].fillna(mean(df[col]))

    df['Hogwarts House'] = target_encode(df['Hogwarts House'])
    
    return df
