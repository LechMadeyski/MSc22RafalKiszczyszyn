import os
import pandas as pd
import numpy as np
import re
from enumerators import SubjectEnumerator
from collectors import RandomForestResultsCollector
from metrics import wilcoxon, cles
from settings import SELECTED, SUBJECTS

FGROUPS = {
    "REC": {
        "total": 19,
        "selector": lambda name: name.startswith("REC_")
    },
    "TES": {
        "total": 44,
        "selector": lambda name: name.startswith("TES_")
    },
    "COV_A": {
        "total": 87,
        "selector": lambda name: "COV_" in name
    },
    "COV": {
        "total": 6,
        "selector": lambda name: name.startswith("COV_") or name.startswith("DET_COV_")
    },
    "COD_COV_C": {
        "total": 44,
        "selector": lambda name: re.match("COD_COV_(PRO|CHN|COM)_C_", name)
    },
    "COD_COV_IMP": {
        "total": 37,
        "selector": lambda name: re.match("COD_COV_(PRO|CHN|COM)_IMP_", name)
    }
}

class FeatureSelector:

    def __init__(self, subjects: SubjectEnumerator, experiment_name: str):
        self._subjects = subjects
        self._experiment_name = experiment_name

    def select(self, filter=None, k=None):
        self._all_features = set()
        self._selected = {}
        self._subjects.enumerate(filter=filter, process=self._process)
        
        if k is None:
            return self._selected

        features = {}
        for sid in self._selected:
            selected = self._selected[sid][k]
            for f in selected:
                if f in features:
                    features[f] += 1
                else:
                    features[f] = 1

        return features

    def _process(self, sid: str, ds_path: str):
        self._selected[sid] = {}
        if len(self._all_features) == 0:
            dataset = pd.read_csv(os.path.join(ds_path, 'dataset.csv'))
            self._all_features = set(dataset.columns.tolist())
        
        results = {}
        dropped = os.path.join(ds_path, 'tsp_accuracy_results', self._experiment_name, 'dropped.txt')        
        with open(dropped, 'r') as file:
            for line in file.readlines():
                features = line.split(";")
                selected = self._all_features.difference(features)
                results[len(selected)] = selected

        self._selected[sid] = results


class ResultsCollector:

    def __init__(self, subjects: SubjectEnumerator, experiment_name: str):
        self._subjects = subjects
        self._experiment_name = experiment_name

    def collect(self, filter=None, k=None):
        self._collected = {}
        self._subjects.enumerate(
            filter=filter, 
            process=lambda sid, ds_path: self._process(sid, ds_path, k)
        )
        
        return self._collected

    def _process(self, sid: str, ds_path: str, k: int):       
        path = os.path.join(ds_path, 'tsp_accuracy_results', self._experiment_name, f'results_{k}.csv')        
        df = pd.read_csv(path)
        self._collected[sid] = pd.Series(df["r_apfdc"].values, index=df["build"]).to_dict()
        

def feature_selection(subjects, selected, exp_name, k):
    selector = FeatureSelector(subjects, exp_name)
    frequencies = selector.select(selected, k)

    selected = sorted(frequencies.keys(), key=lambda f: frequencies[f], reverse=True)[:k+4]
    return [f for f in selected if f not in ['Test', 'Build', 'Verdict', 'Duration']]


def format(num, k=1):
    return float(f"{{:.{k}f}}".format(num))


def distribution():
    frange = range(150, 0, -5)
    results = {fkey: [] for fkey in FGROUPS}
    results["k"] = list(frange)
    for k in frange:
        selected = feature_selection(SUBJECTS, SELECTED, 'feature-selection', k)
        selected_ = feature_selection(SUBJECTS, SELECTED, 'feature-selection-os', k)
        
        A = set(selected)
        B = set(selected_)

        print(k, A - B, B - A)

        for fkey in FGROUPS:
            fgroup = FGROUPS[fkey]
            count = len([fname for fname in selected if fgroup["selector"](fname)])
            x = count / fgroup["total"]
            results[fkey].append(f"{count} ({format(x * 100)}%)")
    
    df = pd.DataFrame(results)
    print(df)


def comparison():
    baseline = RandomForestResultsCollector(SUBJECTS, 'full-outliers').collect(SELECTED)
    frange = range(150, 0, -5)

    K = {k: ([], []) for k in frange}
    changes = {sid: {} for sid in SELECTED}
    
    for k in frange:
        results = ResultsCollector(SUBJECTS, 'feature-selection').collect(SELECTED, k=k)
        for sid in SELECTED:
            n = []
            for build in results[sid]:
                apfdc = results[sid][build]
                b_apfdc = baseline[sid][str(build)]["apfdc"]
                diff = apfdc - b_apfdc
                # if diff >= 0:
                #     K[k] += 1
                n.append(diff)
                
                x, y = K[k]
                x.append(apfdc)
                y.append(b_apfdc)

            chn = np.average(n)
            changes[sid][k] = chn

    for k in K:
        x, y = K[k]
        _, p_value = wilcoxon(x, y)
        cl = cles(x, y)
        print(k, format(np.average(x), 2), format(np.average(y), 2), format(p_value, 3), format(cl, 2))


if __name__ == '__main__':
    # distribution()
    # comparison()
    print(15, feature_selection(SUBJECTS, SELECTED, 'feature-selection-os', 15))
    print(30, feature_selection(SUBJECTS, SELECTED, 'feature-selection-os', 30))
    # print(50, feature_selection(SUBJECTS, SELECTED, 'feature-selection', 50))
    # print(80, feature_selection(SUBJECTS, SELECTED, 'feature-selection', 80))

    # K = {k: np.average(K[k]) for k in K}
    # print(K)
    # print(sorted(K, key=lambda k: K[k], reverse=True))

    # data = {"SID": [], "Worst": [], "Best": [], "Negatives": []}
    # for sid in changes:
    #     worst, k_w = 100, -1
    #     best, k_b = -100, -1
    #     negatives = []
    #     for k in changes[sid]:
    #         chn = changes[sid][k]
    #         if chn < 0:
    #             negatives.append(k)

    #         if chn <= worst:
    #             worst = chn
    #             k_w = k
            
    #         if chn >= best:
    #             best = chn
    #             k_b = k

    #     data["SID"].append(sid) 
    #     data["Worst"].append((k_w, format(worst, 3)))
    #     data["Best"].append((k_b, format(best, 3)))
    #     data["Negatives"].append(negatives)

    # print(pd.DataFrame(data))
