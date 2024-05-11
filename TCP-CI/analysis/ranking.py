import os
import pandas as pd

WORKDIR = "C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\datasets"

PRED_COLS = [
    "qid",
    "Q",
    "target",
    "verdict",
    "duration",
    "test",
    "build",
    "no.",
    "score",
    "indri",
]

def get_ds_path(subject):
    ds_name = subject.replace("/", "@")
    return WORKDIR + f"\\{ds_name}"

def get_all_builds(workdir):
    roots = []

    for item in os.listdir(workdir):
        item_path = os.path.join(workdir, item)
        if os.path.isdir(item_path):
            roots.append(item_path)

    return roots

df = pd.read_csv("C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\datasets.csv")
ds_path = get_ds_path(df['Subject'].to_list()[5 - 1])
print(ds_path)

failing = set()

for build_path in get_all_builds(os.path.join(ds_path, "tsp_accuracy_results", "full-outliers")):
    ranking = pd.read_csv(os.path.join(build_path, 'pred.txt'), names=PRED_COLS, delimiter=' ')
    for index, row in ranking.iterrows():
        if row['verdict'] > 0:
            failing.add(row['test'])

ts = {test: {'exe': [], 'rank': []} for test in failing}
builds = {}

prev = set()
for build_path in get_all_builds(os.path.join(ds_path, "tsp_accuracy_results", "full-outliers")):
    ranking = pd.read_csv(os.path.join(build_path, 'pred.txt'), names=PRED_COLS, delimiter=' ')
    build_no = ranking['build'][0]
    builds[build_no] = {'x': 0, 'apfd': 0, 'apfdc': 0}

    curr = []
    for index, row in ranking.iterrows():
        test = row['test']
        if row['verdict'] == 0:
            continue
        
        curr.append(test)

        # f = 1 if row['verdict'] == 0 else -1

        # builds[build_no]['rank'].append((row['test'], f * row['no.'], row['duration']))
        # ts[test]['exe'].append(row['duration'])
        # ts[test]['rank'].append(f * row['no.'])

    x = len(prev.symmetric_difference(curr))
    prev.update(curr)
    builds[build_no]['x'] = x / len(prev)
    prev = set(curr)

results = pd.read_csv(os.path.join(ds_path, "tsp_accuracy_results", "full-outliers", "results.csv"))
for index, row in results.iterrows():
    builds[row['build']]['apfd'] = row['apfd']
    builds[row['build']]['apfdc'] = row['apfdc']

# for test in ts:
#     tc = ts[test]
#     print(test, 'duration', sum(tc['exe']) / len(tc['exe']), tc['rank'])

# print()

x = []
for build in builds:
    x.append(builds[build]['x'])
    print(build, builds[build])

print(sum(x) / len(x))
