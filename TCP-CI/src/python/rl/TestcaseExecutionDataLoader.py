import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

from .ci_cycle import CICycleLog

class TcpRandomOverSampler:

    def __init__(self) -> None:
        self._sampler = RandomOverSampler(random_state=44, sampling_strategy=1.0)

    def iter_resampled(self, df: pd.DataFrame, y_column, drop):
        if y_column not in drop:
            drop.append(y_column)
        
        X = df.drop(labels=drop, axis=1) 
        y = df[y_column]

        self._sampler.fit_resample(X, y)
        for index in self._sampler.sample_indices_:
            yield df.iloc[index]
    
    def resample(self, df: pd.DataFrame, y_column, drop):
        if y_column not in drop:
            drop.append(y_column)
        
        X = df.drop(labels=drop, axis=1) 
        y = df[y_column]

        self._sampler.fit_resample(X, y)
        df.iloc[[self._sampler.sample_indices_]]


class TestCaseExecutionDataLoader:    
    def __init__(self, data_path):
        self.data_path = data_path
        self.test_data = None
        self.build_time_d = None

    def _normalize_dataset(self, dataset, scaler = None):
        non_feature_cols = [
            'Build',
            'Test',
            'Verdict',
            'Duration',
        ]
        feature_dataset = dataset.drop(non_feature_cols, axis=1)
        if scaler == None:
            scaler = MinMaxScaler()
            scaler.fit(feature_dataset)
        normalized_dataset = pd.DataFrame(
            scaler.transform(feature_dataset),
            columns=feature_dataset.columns,
        )
        for col in non_feature_cols:
            normalized_dataset[col] = dataset[col]

        return normalized_dataset

    def load_data(self):
        builds_df = pd.read_csv(
            self.data_path / "builds.csv", parse_dates=["started_at"]
        )
        
        self.build_time_d = dict(
            zip(
                builds_df["id"].values.tolist(), builds_df["started_at"].values.tolist()
            )
        )

        dataset = pd.read_csv(self.data_path / "dataset.csv")
        dataset['Verdict'] = dataset['Verdict'].apply(lambda v: int(v > 0))
        self.test_data = dataset # self._normalize_dataset(dataset)

        # builds: list = dataset["Build"].unique().tolist()
        # builds.sort(key=lambda b: self.build_time_d[b])

    def pre_process(self, oversampling=False, features=None):
        sampler = TcpRandomOverSampler()

        def iter_cycle(cycle):
            for _, test_case in cycle.iterrows():
                yield test_case

        def iter_oversampled_cycle(cycle):
            for test_case in sampler.iter_resampled(cycle, "Verdict", ["Build", "Test", "Verdict", "Duration"]):
                yield test_case

        builds: list = self.test_data["Build"].unique().tolist()
        builds.sort(key=lambda b: self.build_time_d[b])

        logs = self._get_ci_logs(builds, features, iter_cycle=iter_cycle)
        if oversampling:
            logs_train = self._get_ci_logs(builds, features, iter_cycle=iter_oversampled_cycle)
        else:
            logs_train = logs
        
        return logs_train, logs

    def _get_ci_logs(self, builds, features, iter_cycle):
        logs = []; tests = {}
        for build in builds:
            cycle = self.test_data[self.test_data["Build"] == build]
            log = CICycleLog(build)
            visited = set()
            
            for test_case in iter_cycle(cycle):
                test_id = test_case["Test"]
                if test_id not in tests:
                    tests[test_id] = [0, 0, 0, 0]
                
                if features is not None and len(features) > 0:
                    others = test_case[features].tolist()
                else:
                    others = test_case.drop(labels=["Build", "Test", "Verdict", "Duration", "REC_TotalAvgExeTime"]).tolist()

                log.add_test_case_enriched(
                    test_id=test_case["Test"],
                    last_exec_time=test_case["Duration"],
                    verdict=test_case["Verdict"],
                    avg_exec_time=test_case["REC_TotalAvgExeTime"],
                    failure_history=list(tests[test_id]),
                    rest_hist=[],
                    complexity_metrics=others,
                    cycle_id=build)
            
                if test_id not in visited:
                    last_results = tests[test_id] 
                    last_results.pop(0)
                    last_results.append(int(test_case["Verdict"]))
                    tests[test_id] = last_results
                    visited.add(test_id)

            logs.append(log)
        return logs