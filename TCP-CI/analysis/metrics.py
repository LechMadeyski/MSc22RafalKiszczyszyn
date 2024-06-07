import numpy as np
import pandas as pd
import dataclasses
from scipy.stats import wilcoxon


def cles(data1, data2):
    data1, data2 = np.array(data1), np.array(data2)
    return np.mean(data1[:, None] > data2)


def calc_apfd(ranking: pd.DataFrame):
    n = len(ranking)
    if n <= 1:
        return 1.0
    m = len(ranking[ranking["verdict"] > 0])
    fault_pos_sum = np.sum(ranking[ranking["verdict"] > 0].index + 1)
    apfd = 1 - fault_pos_sum / (n * m) + 1 / (2 * n)
    return float("{:.2f}".format(apfd))


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
    return float("{:.2f}".format(apfdc))


def norm(value, min, max):
    value_norm = (value - min) / (max - min)
    return float("{:.2f}".format(value_norm))


def evaluate(score):
    if isinstance(score, list):
        return np.average(score)
    else:
        return score


def compare(pairs, measure, filter=None):
    results = {'SID': [], 'p-value': [], "CL": []}
    for (sid, m1, m2) in pairs:
        x1 = [evaluate(m1[build][measure]) for build in m1 if (True if filter is None else build in filter)]
        x2 = [evaluate(m2[build][measure]) for build in m2 if (True if filter is None else build in filter)]

        if len(x1) != len(x2):
            raise Exception(f"Number of observations differ for {sid}, hence comparison is invalid")
        
        _, p_value = wilcoxon(x1, x2)
        cles_ = cles(x1, x2)

        results["SID"].append(sid)
        results["p-value"].append(float("{:.2f}".format(p_value)))
        results["CL"].append(float("{:.2f}".format(cles_)))
        ((sid, float("{:.2f}".format(p_value)), float("{:.2f}".format(cles_))))

    return pd.DataFrame(results)



def summarize(results, measure="apfd", filter=None):
    values = []
    for build in results:
        if filter is not None and build not in filter:
            continue 
        
        value = results[build][measure]
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)

    x = map(
        lambda value: float("{:.2f}".format(value)), 
        [np.average(values), np.std(values)]
    )

    return list(x)


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
        random = False
        
        if len(set(ranking["verdict"].values.tolist())) == 1:
            return EvaluationResult(min=0.5, max=0.5, value=0.5, value_norm=0.5), EvaluationResult(min=0.5, max=0.5, value=0.5, value_norm=0.5) 
        
        if len(set(ranking["rank"].values.tolist())) == 1 and len(ranking) > 1:
            return EvaluationResult(min=0.5, max=0.5, value=0.5, value_norm=0.5), EvaluationResult(min=0.5, max=0.5, value=0.5, value_norm=0.5) 

        # Ensure ranking is sorted
        predicted = ranking.sort_values("rank", ascending=True, inplace=False, ignore_index=True)
        best = ranking.sort_values(by=['verdict', 'duration'], ascending=[False, True], inplace=False, ignore_index=True)
        worst = ranking.sort_values(by=['verdict', 'duration'], ascending=[True, False], inplace=False, ignore_index=True)

        apfd_max = calc_apfd(best)
        apfd_min = calc_apfd(worst)
        apfd_predicted = calc_apfd(predicted)
        apfd = EvaluationResult(
            min=apfd_min, 
            max=apfd_max, 
            value=apfd_predicted if not random else -1, 
            value_norm=norm(apfd_predicted, apfd_min, apfd_max) if not random else -1
        )

        apfdc_max = calc_apfdc(best)
        apfdc_min = calc_apfdc(worst)
        apfdc_predicted = calc_apfdc(predicted)
        apfdc = EvaluationResult(
            min=apfdc_min, 
            max=apfdc_max, 
            value=apfdc_predicted if not random else -1, 
            value_norm=norm(apfdc_predicted, apfdc_min, apfdc_max) if not random else -1
        )
        
        return apfd, apfdc