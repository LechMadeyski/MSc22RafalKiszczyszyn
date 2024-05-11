import os
import pandas as pd
import random
import re

def read_last_line(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        return None

    # Read the file and return the last line
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines:
            return lines[-1].strip()  # Using strip() to remove any trailing newline character
        else:
            return None

def get_all_builds(workdir):
    builds = []

    for item in os.listdir(workdir):
        item_path = os.path.join(workdir, item)
        if os.path.isdir(item_path):
            builds.append(item_path)

    return builds

def pick_evenly_distributed_items(lst, num_items):
    if num_items > len(lst):
        raise ValueError("Number of items to pick cannot exceed the length of the list")
    
    # Calculate the size of each segment
    segment_size = len(lst) // num_items
    
    # Pick one item from each segment
    selected_items = []
    for i in range(num_items):
        # Determine the start of the current segment
        start = i * segment_size
        # If it's the last segment, it can go up to the end of the list
        end = start + segment_size if i < num_items - 1 else len(lst)
        # Pick a random item from the segment
        random_index = random.randint(start, end - 1)
        selected_items.append(lst[random_index])
    
    return selected_items

def get_feature_candidates_to_remove(builds, k=5):
    candidates = {}
    for build in builds:
        df = pd.read_csv(os.path.join(build, 'feature_stats.csv'))
        sorted_df = df.sort_values(by='frequency', ascending=True)
        N = len(sorted_df)

        for index, feature_id in enumerate(sorted_df['feature_id'].tolist()):
            if feature_id in candidates:
                candidates[feature_id] += (N - index) * (N - index)
            else:
                candidates[feature_id] = (N - index) * (N - index)
    
    x = pd.DataFrame(list(candidates.items()), columns=['feature_id', 'rank'])
    return x.sort_values(by='rank', ascending=False).head(k)['feature_id'].tolist()


def get_results(result_file):
    data = pd.read_csv(result_file)
    return pd.Series(data.apfdc.values, index=data.build).to_dict()

def all(base_dir):
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    result_files = [f for f in files if "result" in f]
    
    dict = {}
    for f in result_files:
        featuresCount = re.search(r'_(\d+)\.csv', f).group(1)
        dict[int(featuresCount)] = get_results(base_dir + "\\" + f)

    return dict

def compare():
    WORKDIR = 'C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\datasets\\apache@curator\\tsp_accuracy_results'
    baseline = get_results(WORKDIR + '\\full-outliers\\results.csv')
    xxx = all(WORKDIR + '\\feature-selection')
    for featuresCount in sorted(xxx, reverse=True):
        results: dict = xxx[featuresCount]
        # print(results, [baseline[build]  for build in results])
        avg = sum(results.values()) / len(results.values())
        avg_baseline = sum([baseline[build]  for build in results]) / len(results.values())
        print(featuresCount, (avg / avg_baseline) * 100 - 100)

def get_feature_map(file):
    data = pd.read_csv(file)
    return pd.Series(data.key.values, index=data.value).to_dict()

if __name__ == '__main__':
    # compare()

    WORKDIR = 'C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\datasets\\apache@airavata'
    feature_map = get_feature_map(WORKDIR + "\\feature_id_map.csv")
    builds = get_all_builds(WORKDIR + "\\tsp_accuracy_results\\full-outliers")
    dropped = set(read_last_line(WORKDIR + "\\tsp_accuracy_results\\feature-selection\\dropped.txt").split(";")[:-10])

    allf = set([feature_map[id] for id in range(1, 150)])
    print("bestrfe", allf.difference(dropped))

    candidates = get_feature_candidates_to_remove(builds, k=len(dropped))
    print("best", [feature_map[id] for id in set(range(1, 150)).difference(candidates)])

    selected = set([feature_map[id] for id in candidates])
    result = dropped.intersection(selected)
    print(len(dropped), len(result))
