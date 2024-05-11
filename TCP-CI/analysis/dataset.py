import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pfa import PFA
import matplotlib.pyplot as plt

WORKDIR = "C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\datasets"

def avg_failure_rate_per_build(executions: pd.DataFrame):
    # Group by 'build' and then calculate the counts of each verdict
    verdict_counts = executions.groupby('build')['verdict'].value_counts().unstack(fill_value=0)
    verdict_counts['failure_rate'] = (
        verdict_counts.get(1, 0) + 
        verdict_counts.get(2, 0) + 
        verdict_counts.get(3, 0)
    ) / (
        verdict_counts.get(0, 0) + 
        verdict_counts.get(1, 0) + 
        verdict_counts.get(2, 0) + 
        verdict_counts.get(3, 0)
    )

    rates = verdict_counts[verdict_counts['failure_rate'] != 0]
    average_failure_rate = rates['failure_rate'].mean()
    return average_failure_rate * 100

def avg_introduced_tc_per_build(executions: pd.DataFrame, outliers):    
    grouped_tests = executions[~executions['test'].isin(outliers)].groupby('build').agg({
        'test': list,           # Collects all tests for each build into a list
        'verdict': 'sum'
    }).reset_index() 
    
    known = set()
    known_ = set()
    failed = 0
    for index, row in grouped_tests.sort_values(by='build', ascending=True).iterrows():
        tests = set(row['test'])
        updated = known.union(tests)
        if index > 0:
            new = len(tests.difference(known))
            if row['verdict'] > 0:       
                failed += 1
                new_ = len(updated.difference(known_))
                print(row['build'], new_, new)
                known_ = {x for x in updated}
        known = updated
    print(failed)    

def get_all_roots():
    roots = []

    for item in os.listdir(WORKDIR):
        item_path = os.path.join(WORKDIR, item)
        if os.path.isdir(item_path):
            roots.append(item_path)

    return roots


def transform(value):
    x = {
        "M": lambda n: n * 1_000_000,
        "k": lambda n: n * 1_000
    }

    return x[value[-1]](float(value[:-1]))

def get_ds_path(subject):
    ds_name = subject.replace("/", "@")
    return WORKDIR + f"\\{ds_name}"

def add_avg_failure_rate_per_build(df):
    failure_rates = []
    for subject in df['Subject'].to_list():
        ds_path = get_ds_path(subject)
        ds_df = pd.read_csv(ds_path + "\\exe.csv")
        failure_rate = avg_failure_rate_per_build(ds_df)
        failure_rates.append(failure_rate)
        
    df['Avg.#TCFR / Build (%)'] = failure_rates

def add_observations_count(df):
    x = []
    for index, row in df.iterrows():
        print(row['SID'], row['# Failed Builds'] * row['Avg.#TC / Build'])
        x.append(row['# Failed Builds'] * row['Avg.#TC / Build'])

    df['Learning Size'] = x

def add_score(df):
    x = []
    for index, row in df.iterrows():
        print(row['SID'], row['Avg.#TCFR / Build (%)'] * row['# Observations'])
        x.append(row['Avg.#TCFR / Build (%)'] * row['# Observations'])

    df['Score'] = x

if __name__ == '__main__':    
    df = pd.read_csv("C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\datasets.csv")
    
    for sid in range(25):
        ds_path = get_ds_path(df['Subject'].to_list()[sid])
        ds_df = pd.read_csv(ds_path + "\\exe.csv")

        failed_builds = set(pd.read_csv(ds_path + "\\dataset.csv")['Build'].to_list())
        ds_df = ds_df[ds_df['build'].isin(failed_builds)]

        tests = ds_df.groupby('test').agg({
            'build': list,           # Collects all tests for each build into a list
            'verdict': 'sum'
        }).reset_index() 

        tests = tests[tests['verdict'] == 0]['test'].reset_index(drop=True)
        print("SID", sid + 1, "Negatives:", len(tests))
        tests.to_csv(ds_path + '\\negatives.csv')

    # outliers = pd.read_csv(ds_path + "\\tsp_accuracy_results\\full-outliers\\outliers.csv")
    
    
    
    exit(0)

    failed_builds = set(pd.read_csv(ds_path + "\\dataset.csv")['Build'].to_list())
    # avg_introduced_tc_per_build(ds_df, failed_builds, [68362, 30139, 80136, 80135, 75495])

    ds = ds_df[~ds_df['test'].isin(set(outliers['test']))]
    x = ds.pivot_table(index='test', columns='build', values='verdict', fill_value=-1)
    x[sorted(failed_builds)].to_csv(f'dist-sid{sid+1}.csv')

    # df.to_csv("C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\datasets.csv")
    
    # print(df.sort_values(by='Score', ascending=False)['SID'].to_list())
    exit(0)
    
    df_ = df.drop(['SID', 'Subject', 'Timeperiod (months)', '# Commits'], axis=1, inplace=False)

    # pfa.fit(df.values)
    # print(len(pfa.indices_))
    # print(list(df.columns[sorted(pfa.indices_)]))

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_[['# Failed Builds', 'Failure Rate (%)', 'Avg.#TC / Build', 'Avg.TestTime (min)', 'Avg.#TCFR / Build (%)']])

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=0).fit(df_scaled)
    df['Cluster'] = kmeans.labels_

    plt.bar(df['SID'], df['Cluster'])
    plt.xlabel('SID')
    plt.show()
    print(df[['SID', 'Cluster']])

    # for root in get_all_roots():
    #     executions = pd.read_csv(os.path.join(root, "exe.csv"))
    #     failure_rate = avg_failure_rate_per_build(executions)
    #     print(root, failure_rate)
