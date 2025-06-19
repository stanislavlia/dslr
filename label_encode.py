import json
import pandas as pd

def label_encode(series, json_name='mapping.json'):
    mapping = {}
    result = []
    current_label = 0
    for value in series:
        key = value
        if key not in mapping:
            mapping[key] = current_label
            current_label += 1
        result.append(mapping[key])

    with open(json_name, 'w') as file:
        json.dump({v: k for k, v in mapping.items()}, file)

    return pd.Series(result)