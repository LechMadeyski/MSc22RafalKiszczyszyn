import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import numpy as np

class TcpRandomOverSampler:

    def __init__(self) -> None:
        self._sampler = RandomOverSampler(random_state=44, sampling_strategy=1.0)

    def iter_resampled(self, df: pd.DataFrame, y_column, drop):
        if y_column not in drop:
            drop.append(y_column)
        
        X = df.drop(labels=drop, axis=1) 
        y = df[y_column]

        self._sampler.fit_resample(X, y)
        for index in self._sampler.sample_indices_:
            yield df.iloc[index]


# Example DataFrame with IDs and features
data = {
    'ID': [1, 2, 3, 4, 5],
    'Feature1': [10, 20, 30, 10, 50],
    'Feature2': [15, 25, 35, 15, 55],
    'Target': [0, 0, 0, 1, 1]
}
df = pd.DataFrame(data)

# Initialize RandomOverSampler
ros = TcpRandomOverSampler()

# Fit and apply oversampler
for sample in  ros.iter_resampled(df, 'Target', ['ID']):
    print(sample)
