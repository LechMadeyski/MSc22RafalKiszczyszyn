import numpy as np
import pandas as pd
import os

WORKDIR = "C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\datasets"

class SubjectEnumerator:

    def __init__(self, path, workdir):
        self._subjects: pd.DataFrame = pd.read_csv(path)
        self._workdir = workdir

    def enumerate(self, filter=None, process=None):
        for index, row in self._subjects.iterrows():
            if filter is not None and row['SID'] not in filter:
                continue
            
            if process is not None:
                ds_path = self.get_ds_path(row['Subject'])
                process(row['SID'], ds_path)
        pass  

    def get_ds_path(self, subject_name):
        ds_name = subject_name.replace("/", "@")
        return self._workdir + f"\\{ds_name}"


class FailingRateCollector:

    def __init__(self, subjects: SubjectEnumerator, experiment_name: str):
        self._subjects = subjects
        self._experiment_name = experiment_name

    def collect(self, filter=None):
        self._collected = {}
        self._subjects.enumerate(filter=filter, process=self._process)
        return self._collected

    def _process(self, sid: str, ds_path: str):
        
        
        excluded = set(pd.read_csv(
            os.path.join(ds_path, 'tsp_accuracy_results', 'full-outliers', 'outliers.csv')
        )['test'].to_list())

        dataset = pd.read_csv(
            os.path.join(ds_path, 'dataset.csv')
        )

        dataset = dataset[~dataset["Test"].isin(excluded)]
        self._calc_failing_rate(sid, dataset)
        
    def _calc_failing_rate(self, sid, dataset: pd.DataFrame):
        failing = (dataset['Test'] > 0).sum()
        failing_rate = failing / len(dataset)
        self._collected[sid]["failing_rate"] = failing_rate

    def _count_never_failing_tests(self, sid, dataset: pd.DataFrame):
        never_failing = dataset.groupby('Test')['Verdict'].apply(lambda x: (x == 0).all()).sum()
        all = len(set(dataset['Test'].to_list()))
        self._collected[sid]["never_failing"] = never_failing / all

    def _count_x(self, sid, dataset: pd.DataFrame):
        fails = (dataset['Verdict'] > 0).sum()
        tests = dataset[dataset['Verdict'] > 0].groupby('Test').size().sort_values(ascending=False)

        results = {}
        for ratio in [0.9, 0.8, 0.7, 0.6, 0.5]:
            cumulative_sum = 0
            num_tests = 0
            for count in tests:
                cumulative_sum += count
                num_tests += 1
                if cumulative_sum >= ratio * fails:
                    results[ratio] = num_tests / len(tests)
                    break
        
        self._collected[sid]["fails_coverage"] = results


SUBJECTS = SubjectEnumerator(
    'C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\datasets.csv', 
    'C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\datasets'
)

SELECTED = [f'S{i+1}' for i in range(25)]

