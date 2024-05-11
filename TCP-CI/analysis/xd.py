import pandas as pd

if __name__ == '__main__':    
    df = pd.read_csv("C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\datasets.csv")
    x = df.sort_values(by=['Best APFDc', '# Observations', 'Avg. TCFR / Build (%)'], ascending=[False, False, False])
    x.to_csv('subjects.csv')
    print(x)
