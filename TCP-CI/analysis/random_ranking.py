import numpy as np
import pandas as pd
import dataclasses

def calc_apfd(ranking: pd.DataFrame):
    n = len(ranking)
    if n <= 1:
        return 1.0
    m = len(ranking[ranking["verdict"] > 0])
    fault_pos_sum = np.sum(ranking[ranking["verdict"] > 0].index + 1)
    apfd = 1 - fault_pos_sum / (n * m) + 1 / (2 * n)
    return float("{:.3f}".format(apfd))

def calc_apfdc(ranking: pd.DataFrame):
    n = len(ranking)
    if n <= 1:
        return 1.0
    m = len(ranking[ranking["verdict"] > 0])
    costs = ranking["duration"].values.tolist()
    failed_costs = 0.0
    for tfi in ranking[ranking["verdict"] > 0].index:
        failed_costs += sum(costs[tfi:]) - (costs[tfi] / 2)
    apfdc = failed_costs / (sum(costs) * m)
    return float("{:.3f}".format(apfdc))

def norm(value, min, max):
    value_norm = (value - min) / (max - min)
    return  float("{:.3f}".format(value_norm))

@dataclasses.dataclass
class EvaluationResult:
    min: float
    max: float
    value: float
    value_norm: float


class EvaluationService:
    
    COLUMNS = [
        "rank",
        "verdict",
        "duration",
    ]
    
    @staticmethod
    def evaluate(ranking: pd.DataFrame):
        best = ranking.sort_values(by=['verdict', 'duration'], ascending=[False, True], inplace=False, ignore_index=True)
        worst = ranking.sort_values(by=['verdict', 'duration'], ascending=[True, False], inplace=False, ignore_index=True)

        random = ranking.sample(frac=1).reset_index(drop=True)
        apfd_predicted = calc_apfd(random)

        apfd_max = calc_apfd(best)
        apfd_min = calc_apfd(worst)
        apfd = EvaluationResult(
            min=apfd_min, 
            max=apfd_max, 
            value=apfd_predicted, 
            value_norm=norm(apfd_predicted, apfd_min, apfd_max)
        )

        random = ranking.sample(frac=1).reset_index(drop=True)
        apfdc_predicted = calc_apfdc(random)

        apfdc_max = calc_apfdc(best)
        apfdc_min = calc_apfdc(worst)
        apfdc = EvaluationResult(
            min=apfdc_min, 
            max=apfdc_max, 
            value=apfdc_predicted, 
            value_norm=norm(apfdc_predicted, apfdc_min, apfdc_max)
        )
        
        return apfd, apfdc

import os

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

results = {
    "SID": [],
    "APFD": [],
    "APFD std": [],
    "APFDc": [],
    "APFDc std": []
}

df = pd.read_csv("C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\datasets.csv")

ds_path = get_ds_path(df['Subject'].to_list()[23])
print(ds_path)

APFD = []; APFD_std = []
APFDc = []; APFDc_std = []
for iteration in range(30):
    apfd = []
    apfdc = []
    for build_path in get_all_builds(os.path.join(ds_path, "tsp_accuracy_results", "full-outliers")):    
        ranking = pd.read_csv(os.path.join(build_path, 'pred.txt'), names=PRED_COLS, delimiter=' ')
        apfd_, apfdc_ = EvaluationService.evaluate(ranking)
        apfd.append(apfd_.value_norm)
        apfdc.append(apfdc_.value_norm)

    print(iteration, np.average(apfd), np.std(apfd), np.average(apfdc), np.std(apfdc))

    APFD.append(np.average(apfd))
    APFD_std.append(np.std(apfd)) 
    APFDc.append(np.average(apfdc))
    APFDc_std.append(np.std(apfdc))

results['SID'].append('SID25')
results['APFD'].append(np.average(APFD))
results['APFD std'].append(np.average(APFD_std))
results['APFDc'].append(np.average(APFDc))
results['APFDc std'].append(np.average(APFDc_std))

print(np.average(APFD), np.average(APFD_std), np.average(APFDc), np.average(APFDc_std))