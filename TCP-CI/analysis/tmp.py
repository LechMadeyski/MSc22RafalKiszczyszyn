import pandas as pd

# Creating the dataframe from the provided data
data = {
    'SID': ['S2', 'S8', 'S9', 'S12', 'S13', 'S14', 'S16', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25'],
    'COV': [61.6, 47.5, 29.4, 22.4, 59.0, 70.5, 12.9, 23.9, 10.1, 10.7, 10.3, 7.3, 6.7],
    'TES': [38.3, 16.2, 9.8, 8.1, 44.1, 20.2, 6.7, 12.7, 5.9, 4.6, 5.1, 2.3, 3.8],
    'REC': [0.5, 0.6, 0.3, 1.1, 1.1, 0.6, 0.6, 1.0, 1.3, 0.6, 0.7, 1.3, 0.4],
    'TC': [1.1, 0.9, 0.5, 0.5, 1.2, 1.4, 0.2, 0.7, 0.2, 0.3, 0.2, 0.2, 0.1],
    'TT': [6.0, 22.4, 20.7, 9.5, 7.5, 7.7, 6.3, 17.7, 12.3, 48.8, 68.1, 13.7, 15.5],
    'C/T': [18, 4, 3, 5, 16, 18, 4, 4, 2, 1, '<1', 1, 1]
}

df = pd.DataFrame(data)

# Calculating the percent for each row
df['COV'] = (round((df['COV'] / (df['COV'] + df['TES'] + df['REC'])) * 100, 1))
df['TES'] = (round((df['TES'] / (df['COV'] + df['TES'] + df['REC'])) * 100, 1))
df['REC'] = (round((df['REC'] / (df['COV'] + df['TES'] + df['REC'])) * 100, 1))

from io import StringIO
stream = StringIO()
df.to_csv(stream, index=False)

print(stream.getvalue())