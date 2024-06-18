import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from enumerators import SubjectEnumerator
from metrics import EvaluationService, summarize

import pickle
def dump_cache(cache):
    path = "C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\cache.bin"
    with open(path, 'wb') as f:
        pickle.dump(cache, f)

def load_cache():
    path = "C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\cache.bin"
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_duration_reduction(ranking: pd.DataFrame):
    ranking = ranking.convert_dtypes()
    failures = ranking[ranking['verdict'] == 1]
    
    f_index = failures.index[0]
    ttf = ranking.loc[:f_index, 'duration'].sum()
    
    l_index = failures.index[-1]
    ttl = ranking.loc[:l_index, 'duration'].sum()
    
    return ttf, ttl


class AcerResultsCollector:

    def __init__(self, subjects: SubjectEnumerator, experiment_name: str):
        self._subjects = subjects
        self._experiment_name = experiment_name

    def collect(self, filter=None):
        self._cache = load_cache()
        self._collected = {}
        self._subjects.enumerate(filter=filter, process=self._process)
        dump_cache(self._cache)
        return self._collected

    def _process(self, sid: str, ds_path: str):
        key = f"ACER_{self._experiment_name}_{ds_path}_{sid}" 
        if key in self._cache:
            self._collected[sid] = self._cache[key]
            return
        
        results = {}
        
        exp_path = os.path.join(ds_path, 'tsp_accuracy_results', self._experiment_name)        
        files = sorted([f for f in os.listdir(exp_path) if f.endswith('.csv') and f[:-4].isdigit()])
        for file in files:
            build = file[:-4]
            path = os.path.join(exp_path, file)

            ranking = pd.read_csv(path)
            ranking = ranking.rename(columns={'last_exec_time': 'duration'})
            ranking['rank'] = ranking.index + 1

            apfd, apfdc = EvaluationService.evaluate(ranking)   
            ttf, ttl = get_duration_reduction(ranking)         
            results[build] = {
                'apfd': apfd.value_norm, 
                '_apfd_': apfd.value,
                'apfdc': apfdc.value_norm, 
                '_apfdc_': apfdc.value,
                'ttf': ttf,
                'ttl': ttl
            }

        self._cache[key] = results
        self._collected[sid] = results


class RandomForestResultsCollector:

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

    def __init__(self, subjects: SubjectEnumerator, experiment_name: str):
        self._subjects = subjects
        self._experiment_name = experiment_name

    def collect(self, filter=None):
        self._cache = load_cache()
        self._collected = {}
        self._subjects.enumerate(filter=filter, process=self._process)
        dump_cache(self._cache)
        return self._collected

    def _process(self, sid: str, ds_path: str):
        key = f"ACER_{self._experiment_name}_{ds_path}_{sid}"
        if key in self._cache:
            self._collected[sid] = self._cache[key]
            return
        
        results = {}
        
        exp_path = os.path.join(ds_path, 'tsp_accuracy_results', self._experiment_name)        
        builds = sorted(self._get_all_builds(exp_path))
        for build in builds:
            path = os.path.join(exp_path, build, 'pred.txt')

            ranking = pd.read_csv(path, names=self.PRED_COLS, delimiter=' ')
            ranking['rank'] = ranking["score"].rank(method="min", ascending=False)

            apfd, apfdc = EvaluationService.evaluate(ranking)            
            ttf, ttl = get_duration_reduction(ranking)
            results[build] = {
                'apfd': apfd.value_norm, 
                '_apfd_': apfd.value,
                'apfdc': apfdc.value_norm, 
                '_apfdc_': apfdc.value,
                'ttf': ttf,
                'ttl': ttl
            } 

        self._cache[key] = results
        self._collected[sid] = results

    def _get_all_builds(self, exp_path):
        builds = []

        for item in os.listdir(exp_path):
            item_path = os.path.join(exp_path, item)
            if os.path.isdir(item_path):
                builds.append(item)

        return builds


class RandomApproachResultsCollector:

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

    def __init__(self, subjects: SubjectEnumerator, experiment_name: str):
        self._subjects = subjects
        self._experiment_name = experiment_name

    def collect(self, filter=None):
        self._cache = load_cache()
        self._collected = {}
        self._subjects.enumerate(filter=filter, process=self._process)
        dump_cache(self._cache)
        return self._collected

    def _process(self, sid: str, ds_path: str):
        key = f"RND_{ds_path}_{sid}" 
        if key in self._cache:
            self._collected[sid] = self._cache[key]
            return
        
        results = {}
        
        exp_path = os.path.join(ds_path, 'tsp_accuracy_results', self._experiment_name)        
        builds = sorted(self._get_all_builds(exp_path))
        for build in builds:
            path = os.path.join(exp_path, build, 'pred.txt')

            ranking = pd.read_csv(path, names=self.PRED_COLS, delimiter=' ')
            
            apfd = []
            apfdc = []
            ttf = []
            ttl = []
            for _ in range(30):
                ranking = ranking.sample(frac=1).reset_index(drop=True)
                ranking['rank'] = ranking.index + 1
                
                x, y = EvaluationService.evaluate(ranking)   
                apfd.append(x)
                apfdc.append(y)

                ttf_, ttl_ = get_duration_reduction(ranking)
                ttf.append(ttf_)
                ttl.append(ttl_)        
                
            results[build] = {
                'apfd': list(map(lambda x: x.value_norm, apfd)), 
                '_apfd_': list(map(lambda x: x.value, apfd)),
                'apfdc': list(map(lambda x: x.value_norm, apfdc)), 
                '_apfdc_': list(map(lambda x: x.value, apfdc)),
                'ttf': ttf,
                'ttl': ttl
            }

        self._cache[key] = results
        self._collected[sid] = results

    def _get_all_builds(self, exp_path):
        builds = []

        for item in os.listdir(exp_path):
            item_path = os.path.join(exp_path, item)
            if os.path.isdir(item_path):
                builds.append(item)

        return builds


class SimpleHeuristicResultsCollector:

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

    def __init__(self, subjects: SubjectEnumerator, experiment_name: str, feature_name: str, ascending: bool):
        self._subjects = subjects
        self._experiment_name = experiment_name
        self._feature_name = feature_name
        self._ascending = ascending

    def collect(self, filter=None):
        self._cache = load_cache()
        self._collected = {}
        self._subjects.enumerate(filter=filter, process=self._process)
        dump_cache(self._cache)
        return self._collected

    def _normalize(self, dataset):
        non_feature_cols = ["Build", "Test", "Verdict", "Duration"]
        feature_dataset = dataset.drop(non_feature_cols, axis=1)
        
        scaler = MinMaxScaler()
        scaler.fit(feature_dataset)

        normalized_dataset = pd.DataFrame(
            scaler.transform(feature_dataset),
            columns=feature_dataset.columns,
        )
        for col in non_feature_cols:
            normalized_dataset[col] = dataset[col]

        return normalized_dataset

    def _process(self, sid: str, ds_path: str):
        key = f"H_{ds_path}_{sid}" 
        if key in self._cache:
            self._collected[sid] = self._cache[key]
            return
        
        dataset = pd.read_csv(os.path.join(ds_path, 'dataset.csv'))
        dataset = self._normalize(dataset)

        results = {}
        exp_path = os.path.join(ds_path, 'tsp_accuracy_results', self._experiment_name)        
        builds = sorted(self._get_all_builds(exp_path))
        for build in builds:
            path = os.path.join(exp_path, build, 'pred.txt')

            included = set(pd.read_csv(path, names=self.PRED_COLS, delimiter=' ')['test'].to_list())
            ranking_ = {"test": [], "verdict": [], "duration": [], "feature": [], 'recent_duration': []}
            for _, row in dataset[dataset["Build"] == int(build)].iterrows():
                test = row['Test']
                if test not in included:
                    continue
                
                ranking_['test'].append(test)
                ranking_['feature'].append(row[self._feature_name])
                ranking_['recent_duration'].append(row["REC_Age"])
                ranking_['verdict'].append(int(row["Verdict"] > 0))
                ranking_['duration'].append(row["Duration"])
            
            ranking = pd.DataFrame(ranking_)
            
            apfd = []
            apfdc = []
            ttf = []
            ttl = []
            for _ in range(30):
                ranking = ranking.sample(frac=1).reset_index(drop=True)
                # ranking = ranking.sort_values(by="feature", ascending=self._ascending, ignore_index=True)
                ranking = ranking.sort_values(by=["feature", "recent_duration"], ascending=[self._ascending, True], ignore_index=True)
                ranking['rank'] = ranking.index + 1

                x, y = EvaluationService.evaluate(ranking)   
                apfd.append(x)
                apfdc.append(y) 

                ttf_, ttl_ = get_duration_reduction(ranking)
                ttf.append(ttf_)
                ttl.append(ttl_)

                
            results[build] = {
                'apfd': list(map(lambda x: x.value_norm, apfd)), 
                '_apfd_': list(map(lambda x: x.value, apfd)),
                'apfdc': list(map(lambda x: x.value_norm, apfdc)), 
                '_apfdc_': list(map(lambda x: x.value, apfdc)),
                'ttf': ttf,
                'ttl': ttl
            }

        self._cache[key] = results
        self._collected[sid] = results

    def _get_all_builds(self, exp_path):
        builds = []

        for item in os.listdir(exp_path):
            item_path = os.path.join(exp_path, item)
            if os.path.isdir(item_path):
                builds.append(item)

        return builds


class BuildsCollector:

    def __init__(self, subjects: SubjectEnumerator, experiment_name: str):
        self._subjects = subjects
        self._experiment_name = experiment_name

    def collect(self, filter=None):
        self._collected = {}
        self._subjects.enumerate(filter=filter, process=self._process)
        return self._collected

    def _process(self, sid: str, ds_path: str):        
        exp_path = os.path.join(ds_path, 'tsp_accuracy_results', self._experiment_name)        
        builds = sorted(self._get_all_builds(exp_path))
        self._collected[sid] = builds

    def _get_all_builds(self, exp_path):
        builds = []

        for item in os.listdir(exp_path):
            item_path = os.path.join(exp_path, item)
            if os.path.isdir(item_path):
                builds.append(item)

        return builds


class TestingDurationCollector:

    def __init__(self, subjects: SubjectEnumerator, experiment_name: str):
        self._subjects = subjects
        self._experiment_name = experiment_name

    def collect(self, filter=None):
        self._collected = {}
        self._subjects.enumerate(filter=filter, process=self._process)
        return self._collected

    def _process(self, sid: str, ds_path: str):              
        all_builds = pd.read_csv(os.path.join(ds_path, 'builds.csv'), parse_dates=["started_at"])
        all_builds['id'] = all_builds['id'].astype(str)
        order = dict(
            zip(
                all_builds["id"].values.tolist(), all_builds["started_at"].values.tolist()
            )
        )

        exp_path = os.path.join(ds_path, 'tsp_accuracy_results', self._experiment_name)  
        builds = sorted(self._get_all_builds(exp_path), key=lambda b: order[b])
        first = builds[0]
        
        executions = pd.read_csv(os.path.join(ds_path, 'exe.csv'))
        executions['build'] = executions['build'].astype(str)

        def started(build):
            return order[build]
        
        executions = executions.sort_values(by='build', key=lambda b: b.map(started))

        ts_size = executions.groupby('build')['test'].nunique().to_dict()
        d_total = executions['duration'].sum()
        grouped_df = executions.groupby('build')['duration'].sum().reset_index()
        duration_dict = dict(zip(grouped_df['build'], grouped_df['duration'])) 
        fd_total = executions[executions['build'].isin(builds)]['duration'].sum()

        self._collected[sid] = {
            "d_total": d_total, 
            "builds": list(duration_dict),
            "fd_total": fd_total,
            "f_builds": list(builds),
            "durations": duration_dict,
            "ts_size": ts_size,
            "avg_ts_size": np.average(list(map(lambda x: ts_size[x], ts_size)))
        }

    def _get_all_builds(self, exp_path):
        builds = []

        for item in os.listdir(exp_path):
            item_path = os.path.join(exp_path, item)
            if os.path.isdir(item_path):
                builds.append(item)

        return builds


class GenericResultsCollector:

    def __init__(self, collector, builds=None) -> None:
        self._collector = collector
        self._builds = builds if builds is not None else {} 
        self.collected_ = []

    def collect_into_df(self, selected):
        apfd = []; apfd_std = []
        _apfd_ = []; _apfd_std_ = []
        apfdc = []; apfdc_std = []
        _apfdc_ = []; _apfdc_std_ = []
        self.collected_ = {sid: None for sid in selected}
        for sid in selected:
            self.collected_[sid] = self._collector.collect(filter=[sid])[sid]
            
            builds=None
            if sid in self._builds:
                builds = self._builds[sid]
            
            [avg, std] = summarize(self.collected_[sid], measure="apfd", filter=builds)
            apfd.append(avg)
            apfd_std.append(std)

            [avg, std] = summarize(self.collected_[sid], measure="_apfd_", filter=builds)
            _apfd_.append(avg)
            _apfd_std_.append(std)
            
            [avg, std] = summarize(self.collected_[sid], measure="apfdc", filter=builds)
            apfdc.append(avg)
            apfdc_std.append(std)

            [avg, std] = summarize(self.collected_[sid], measure="_apfdc_", filter=builds)
            _apfdc_.append(avg)
            _apfdc_std_.append(std)

        return pd.DataFrame({
            'SID': selected,
            'APFD (avg)': apfd,
            'APFD (std)': apfd_std,
            '_APFD_ (avg)': _apfd_,
            '_APFD_ (std)': _apfd_std_,
            'APFDc (avg)': apfdc,
            'APFDc (std)': apfdc_std,
            '_APFDc_ (avg)': _apfdc_,
            '_APFDc_ (std)': _apfdc_std_
        })        

    def collected(self, sid):
        if sid not in self._builds:
            return self.collected_[sid]
        
        return {build: self.collected_[sid][build] for build in self.collected_[sid] if build in self._builds[sid]}
