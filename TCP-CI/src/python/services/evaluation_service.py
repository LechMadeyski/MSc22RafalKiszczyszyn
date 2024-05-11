import dataclasses
import pandas as pd
import numpy as np

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
        random = False

        if len(set(ranking["verdict"].values.tolist())) == 1:
            return EvaluationResult(min=0.5, max=0.5, value=0.5, value_norm=0.5), EvaluationResult(min=0.5, max=0.5, value=0.5, value_norm=0.5) 

        if len(set(ranking["rank"].values.tolist())) == 1 and len(ranking) > 1:
            random = True # indicates that ranking is random because of the same rank values.

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
