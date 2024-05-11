import numpy as np
import pandas as pd
import os
import dataclasses

WORKDIR = "C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\datasets"

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


def summarize(results, filter=None):
    apfd = []
    apfdc = []
    _apfdc_ = []
    for build in results:
        if filter is not None and build not in filter:
            continue 
        
        apfd.append(results[build]['apfd'])
        apfdc.append(results[build]['apfdc'])
        _apfdc_.append(results[build]['_apfdc_'])

    x = map(
        lambda value: float("{:.3f}".format(value)), 
        [np.average(apfd), np.std(apfd), np.average(apfdc), np.std(apfdc), np.average(_apfdc_), np.std(_apfdc_)]
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


class AcerResultsCollector:

    def __init__(self, subjects: SubjectEnumerator, experiment_name: str):
        self._subjects = subjects
        self._experiment_name = experiment_name

    def collect(self, filter=None):
        self._collected = {}
        self._subjects.enumerate(filter=filter, process=self._process)
        return self._collected

    def _process(self, sid: str, ds_path: str):
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
            results[build] = {'apfd': apfd.value_norm, 'apfdc': apfdc.value_norm, '_apfdc_': apfdc.value}

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
        self._collected = {}
        self._subjects.enumerate(filter=filter, process=self._process)
        return self._collected

    def _process(self, sid: str, ds_path: str):
        results = {}
        
        exp_path = os.path.join(ds_path, 'tsp_accuracy_results', self._experiment_name)        
        builds = sorted(self._get_all_builds(exp_path))
        for build in builds:
            path = os.path.join(exp_path, build, 'pred.txt')

            ranking = pd.read_csv(path, names=self.PRED_COLS, delimiter=' ')
            ranking['rank'] = ranking.index + 1

            apfd, apfdc = EvaluationService.evaluate(ranking)            
            results[build] = {'apfd': apfd.value_norm, 'apfdc': apfdc.value_norm, '_apfdc_': apfdc.value}

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
        self._collected = {}
        self._subjects.enumerate(filter=filter, process=self._process)
        return self._collected

    def _process(self, sid: str, ds_path: str):
        results = {}
        
        exp_path = os.path.join(ds_path, 'tsp_accuracy_results', self._experiment_name)        
        builds = sorted(self._get_all_builds(exp_path))
        for build in builds:
            path = os.path.join(exp_path, build, 'pred.txt')

            ranking = pd.read_csv(path, names=self.PRED_COLS, delimiter=' ').sample(frac=1).reset_index(drop=True)
            ranking['rank'] = ranking.index + 1

            apfd, apfdc = EvaluationService.evaluate(ranking)            
            results[build] = {'apfd': apfd.value_norm, 'apfdc': apfdc.value_norm, '_apfdc_': apfdc.value}

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
        self._collected = {}
        self._subjects.enumerate(filter=filter, process=self._process)
        return self._collected

    def _process(self, sid: str, ds_path: str):
        dataset = pd.read_csv(os.path.join(ds_path, 'dataset.csv'))
        features = {}
        for _, row in dataset.iterrows():
            build = str(int(row['Build']))
            if build not in features:
                features[build] = {}

            test = row['Test']
            features[build][test] = row

        results = {}
        exp_path = os.path.join(ds_path, 'tsp_accuracy_results', self._experiment_name)        
        builds = sorted(self._get_all_builds(exp_path))
        for build in builds:
            path = os.path.join(exp_path, build, 'pred.txt')

            ranking = pd.read_csv(path, names=self.PRED_COLS, delimiter=' ')
            
            feature_values = []
            for test in ranking['test']:
                feature_values.append(features[build][test][self._feature_name])
            
            ranking['feature'] = feature_values
            ranking = ranking.sort_values(by="feature", ascending=self._ascending)
            ranking['rank'] = ranking.index + 1

            apfd, apfdc = EvaluationService.evaluate(ranking)            
            results[build] = {'apfd': apfd.value_norm, 'apfdc': apfdc.value_norm, '_apfdc_': apfdc.value}

        self._collected[sid] = results

    def _get_all_builds(self, exp_path):
        builds = []

        for item in os.listdir(exp_path):
            item_path = os.path.join(exp_path, item)
            if os.path.isdir(item_path):
                builds.append(item)

        return builds


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


class GenericResultsCollector:

    def __init__(self, collector, builds=None) -> None:
        self._collector = collector
        self._builds = builds if builds is not None else {} 

    def collect_into_df(self, selected):
        apfd = []; apfd_std = []
        apfdc = []; apfdc_std = []
        _apfdc_ = []; _apfdc_std_ = []
        for sid in selected:
            collected = self._collector.collect(filter=[sid])
            
            builds=None
            if sid in self._builds:
                builds = self._builds[sid]
            
            [a, b, c, d, e, f] = summarize(collected[sid], filter=builds)
            apfd.append(a)
            apfd_std.append(b)
            apfdc.append(c)
            apfdc_std.append(d)
            _apfdc_.append(e)
            _apfdc_std_.append(f)

        return pd.DataFrame({
            'SID': selected,
            'APFD (avg)': apfd,
            'APFD (std)': apfd_std,
            'APFDc (avg)': apfdc,
            'APFDc (std)': apfdc_std,
            '_APFDc_ (avg)': _apfdc_,
            '_APFDc_ (std)': _apfdc_std_  
        })        


def join(df: pd.DataFrame, origin: pd.DataFrame, ds_name, measure):
    df[f'{ds_name} {measure}'] = origin[f'{measure} (avg)'].astype(str) + ' ± ' + origin[f'{measure} (std)'].astype(str)


def feature_selection(subjects, selected, exp_name, k):
    selector = FeatureSelector(subjects, exp_name)
    frequencies = selector.select(selected, k)

    return sorted(frequencies.keys(), key=lambda f: frequencies[f], reverse=True)[:k+4]


def experiment_0(subjects, selected):
    x = pd.DataFrame({'SID': selected})
    
    full = GenericResultsCollector(
        RandomForestResultsCollector(subjects, 'full-outliers')
    ).collect_into_df(selected)
    
    join(x, full, 'FULL', 'APFD')
    join(x, full, 'FULL', 'APFDc')
    join(x, full, 'FULL', '_APFDc_')

    # H = GenericResultsCollector(
    #     SimpleHeuristicResultsCollector(subjects, 'full-outliers', 'REC_TotalFailRate', ascending=False)
    # ).collect_into_df(selected)
    
    # join(x, H, 'H', 'APFD')
    # join(x, H, 'H', 'APFDc')

    # rnd = GenericResultsCollector(
    #     RandomApproachResultsCollector(subjects, 'full-outliers')
    # ).collect_into_df(selected)

    # join(x, rnd, 'RND', 'APFD')
    # join(x, rnd, 'RND', 'APFDc')

    # imp = GenericResultsCollector(
    #     RandomForestResultsCollector(subjects, 'wo-impacted-outliers')
    # ).collect_into_df(selected)

    # tes = GenericResultsCollector(
    #     RandomForestResultsCollector(subjects, 'W-Code-outliers')
    # ).collect_into_df(selected)

    # rec = GenericResultsCollector(
    #     RandomForestResultsCollector(subjects, 'W-Execution-outliers')
    # ).collect_into_df(selected)

    # cov = GenericResultsCollector(
    #     RandomForestResultsCollector(subjects, 'W-Coverage-outliers')
    # ).collect_into_df(selected)

    print(x)

# compare ACER-PA with RF, RA and H on original feature set and datasets
def rl_experiment(subjects, selected, baseline_exp_name, exp_name):
    x = pd.DataFrame({'SID': selected})
    
    builds = BuildsCollector(subjects, baseline_exp_name).collect(selected)

    rf = GenericResultsCollector(
        RandomForestResultsCollector(subjects, baseline_exp_name),
        builds=builds
    ).collect_into_df(selected)
    
    join(x, rf, 'RF', 'APFD')
    join(x, rf, 'RF', 'APFDc')

    acer = GenericResultsCollector(
        AcerResultsCollector(subjects, exp_name),
        builds=builds
    ).collect_into_df(selected)

    join(x, acer, 'ACER-PA', 'APFD')
    join(x, acer, 'ACER-PA', 'APFDc')

    # H = GenericResultsCollector(
    #     SimpleHeuristicResultsCollector(subjects, 'full-outliers', 'REC_TotalFailRate', ascending=False)
    # ).collect_into_df(selected)
    
    # join(x, H, 'H', 'APFD')
    # join(x, H, 'H', 'APFDc')

    rnd = GenericResultsCollector(
        RandomApproachResultsCollector(subjects, 'full-outliers')
    ).collect_into_df(selected)

    join(x, rnd, 'RND', 'APFD')
    join(x, rnd, 'RND', 'APFDc')

    print(x)


SUBJECTS = SubjectEnumerator(
    'C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\datasets.csv', 
    'C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\datasets'
)

# SELECTED = sorted(['S1', 'S2', 'S3', 'S4', 'S5', 'S12', 'S17', 'S9', 'S15', 'S25'])
SELECTED = sorted(['S1', 'S2', 'S4', 'S5', 'S12', 'S17', 'S9', 'S15', 'S25'])

rl_experiment(SUBJECTS, ['S20'], 'full-outliers', 'rl')
rl_experiment(SUBJECTS, ['S20'], 'full-outliers', 'rl-f20')
rl_experiment(SUBJECTS, ['S20'], 'full-outliers', 'rl-os-f20')
rl_experiment(SUBJECTS, ['S20'], 'full-outliers', 'rl-os-f20-er4')
rl_experiment(SUBJECTS, ['S20'], 'full-outliers', 'rl-os-f20-rel_reward')

# print(feature_selection(SUBJECTS, None, 'feature-selection', k=20))
