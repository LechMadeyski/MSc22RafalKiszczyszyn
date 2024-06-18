from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import subprocess
from .feature_extractor.feature import Feature
from pathlib import Path
from tqdm import tqdm
from .services.evaluation_service import EvaluationService
from .services.feature_selection_service import FeatureSelectionService
import sys
import re
import os
import logging
from imblearn.over_sampling import RandomOverSampler
from datetime import datetime


class TcpRandomOverSampler:

    def __init__(self) -> None:
        self._sampler = RandomOverSampler(random_state=44, sampling_strategy=1.0)
    
    def resample(self, df: pd.DataFrame, y_column):
        X = df.drop([Feature.VERDICT, Feature.DURATION, Feature.BUILD, Feature.TEST], axis=1)
        # X = df.filter(regex='^f\d+$')
        y = df[y_column]

        self._sampler.fit_resample(X, y)
        np.random.shuffle(self._sampler.sample_indices_)
        return df.iloc[self._sampler.sample_indices_]


class RankLibLearner:

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

    def __init__(self, config):
        self._sampler = TcpRandomOverSampler()
        self.config = config
        self.feature_id_map_path = config.output_path / "feature_id_map.csv"
        
        if self.feature_id_map_path.exists():
            feature_id_map_df = pd.read_csv(self.feature_id_map_path)
            keys = feature_id_map_df["key"].values.tolist()
            values = feature_id_map_df["value"].values.tolist()
            self.feature_id_map = dict(zip(keys, values))
            self.next_fid = max(values) + 1
        else:
            self.feature_id_map = {}
            self.next_fid = 1
        self.dropped_feature_ids = set()

        builds_df = pd.read_csv(
            config.output_path / "builds.csv", parse_dates=["started_at"]
        )
        
        self.build_time_d = dict(
            zip(
                builds_df["id"].values.tolist(), builds_df["started_at"].values.tolist()
            )
        )
        
        self.ranklib_path = Path("assets") / "RankLib.jar"
        self.math3_path = Path("assets") / "commons-math3.jar"

    def get_feature_id(self, feature_name):
        if feature_name not in self.feature_id_map:
            self.feature_id_map[feature_name] = self.next_fid
            self.next_fid += 1
        return self.feature_id_map[feature_name]

    def save_feature_id_map(self):
        keys = list(self.feature_id_map.keys())
        values = list(self.feature_id_map.values())
        feature_id_map_df = pd.DataFrame({"key": keys, "value": values})
        feature_id_map_df.to_csv(self.feature_id_map_path, index=False)

    def update_dropped_features(self, feature_names):
        self.dropped_feature_ids = set([self.get_feature_id(name) for name in feature_names])

    def normalize_dataset(self, dataset, scaler):
        non_feature_cols = [
            Feature.BUILD,
            Feature.TEST,
            Feature.VERDICT,
            Feature.DURATION,
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

        return normalized_dataset, feature_dataset, scaler

    def convert_to_ranklib_dataset(self, dataset, scaler=None, oversampling=False):
        if dataset.empty:
            return None
        dataset = dataset.copy()
        dataset[Feature.VERDICT] = dataset[Feature.VERDICT].apply(lambda v: int(v > 0))
        normalized_dataset, feature_dataset, _ = self.normalize_dataset(dataset, scaler)
        builds = normalized_dataset[Feature.BUILD].unique()
        ranklib_ds_rows = []
        for i, build in list(enumerate(builds)):
            build_ds = normalized_dataset[
                normalized_dataset[Feature.BUILD] == build
            ].copy()

            if oversampling:
                build_ds = self._sampler.resample(build_ds, Feature.VERDICT)

            build_ds["B_Verdict"] = (build_ds[Feature.VERDICT] > 0).astype(int)
            build_ds.sort_values(
                ["B_Verdict", Feature.DURATION],
                ascending=[False, True],
                inplace=True,
                ignore_index=True,
            )
            build_ds.drop("B_Verdict", axis=1, inplace=True)
            build_ds["Target"] = -build_ds.index + len(build_ds)
            for _, record in build_ds.iterrows():
                row_items = [int(record["Target"]), f"qid:{i+1}"]
                row_feature_items = []
                for _, f in enumerate(feature_dataset.columns):
                    fid = self.get_feature_id(f)
                    row_feature_items.append(f"{fid}:{record[f]}")
                row_feature_items.sort(key=lambda v: int(v.split(":")[0]))
                row_items.extend(row_feature_items)
                row_items.extend(
                    [
                        "#",
                        int(record["Target"]),
                        int(record[Feature.VERDICT]),
                        int(record[Feature.DURATION]),
                        int(record[Feature.TEST]),
                        int(record[Feature.BUILD]),
                    ]
                )
                ranklib_ds_rows.append(row_items)
        headers = (
            ["target", "qid"]
            + [f"f{i+1}" for i in range(len(feature_dataset.columns))]
            + ["hashtag", "i_target", "i_verdict", "i_duration", "i_test", "i_build"]
        )
        self.save_feature_id_map()
        return pd.DataFrame(ranklib_ds_rows, columns=headers)

    def create_ranklib_training_sets(
        self, ranklib_ds, output_path, custom_test_builds=None, transform=None, force=False, oversampling=False
    ):
        builds = ranklib_ds["i_build"].unique().tolist()
        builds.sort(key=lambda b: self.build_time_d[b])
        if custom_test_builds is None:
            test_builds = set(builds[-self.config.test_count :])
        else:
            test_builds = [b for b in custom_test_builds if b in builds]
        
        logging.info("Creating training sets")

        X = {b: None for b in builds}
        for i, build in tqdm(list(enumerate(builds)), desc="Creating training sets"):
            if oversampling:
                ds = ranklib_ds[ranklib_ds["i_build"] == build]
                X[build] = self._sampler.resample(ds, 'i_verdict')

            if build not in test_builds:
                continue
            
            train_ds = ranklib_ds[ranklib_ds["i_build"].isin(builds[:i])]
            if len(train_ds) == 0:
                continue

            if oversampling:
                # First Implementation: All training build oversampled at once
                # train_ds = self._sampler.resample(train_ds, 'i_verdict')
                sets = [X[b] for b in builds[:i]]
                train_ds = pd.concat(sets, ignore_index=True)

            test_ds = ranklib_ds[ranklib_ds["i_build"] == build]
            build_out_path = output_path / str(build)
            
            if force and build_out_path.exists() and build_out_path.is_dir():
                for item in build_out_path.iterdir():
                    if item.is_dir():
                        item.rmdir()
                    else:
                        item.unlink()
                build_out_path.rmdir()
            build_out_path.mkdir(parents=True, exist_ok=True)
            
            if (
                force or (
                    not (output_path / str(build) / "train.txt").exists()
                    and not (output_path / str(build) / "model.txt").exists())
            ):
                train_ds.to_csv(
                    output_path / str(build) / "train.txt",
                    sep=" ",
                    header=False,
                    index=False,
                )
            if force or (not (output_path / str(build) / "test.txt").exists()):
                test_ds.to_csv(
                    output_path / str(build) / "test.txt",
                    sep=" ",
                    header=False,
                    index=False,
                )

    def extract_and_save_feature_stats(self, feature_stats_output, output_path):
        feature_freq_map = {i: 0 for i in self.feature_id_map.values() if i not in self.dropped_feature_ids}
        matches = re.finditer(
            r"Feature\[(\d+)\]\s+:\s+(\d+)", feature_stats_output, re.MULTILINE
        )
        for match in matches:
            feature_id = int(match.group(1))
            feature_freq = int(match.group(2))
            feature_freq_map[feature_id] = feature_freq
        feature_stats_df = pd.DataFrame(
            {
                "feature_id": list(feature_freq_map.keys()),
                "frequency": list(feature_freq_map.values()),
            }
        )
        feature_stats_df.sort_values("feature_id", ignore_index=True, inplace=True)
        feature_stats_df.to_csv(output_path / "feature_stats.csv", index=False)

    def train_and_test(self, build_ds_path, ranker, params, dataset_df, suffix=""):
        train_path = build_ds_path / "train.txt"
        test_path = build_ds_path / "test.txt"
        model_path = build_ds_path / f"model{suffix}.txt"
        pred_path = build_ds_path / f"pred{suffix}.txt"

        if not model_path.exists():
            builds_count = dataset_df[Feature.BUILD].nunique()
            if ranker == 8 and builds_count < 52:
                params["srate"] = 1.0
            params_cmd = " ".join(
                [f"-{name} {value}" for name, value in params.items()]
            )
            train_command = f"java -jar {self.ranklib_path} -train {train_path} -ranker {ranker} {params_cmd} -save {model_path} -silent"
            train_out = subprocess.run(train_command, shell=True, capture_output=True)
            if train_out.returncode != 0:
                logging.error(f"Error in training:\n{train_out.stderr}")
                sys.exit()

        if not pred_path.exists():
            pred_command = f"java -jar {self.ranklib_path} -load {model_path} -rank {test_path} -indri {pred_path}"
            if "metric2T" in params:
                pred_command += f" -metric2T {params['metric2T']}"
            pred_out = subprocess.run(pred_command, shell=True, capture_output=True)
            if pred_out.returncode != 0:
                logging.error(f"Error in predicting:\n{pred_out.stderr}")
                sys.exit()
            if suffix != "":
                os.remove(str(model_path))
        
        pred_df = (
            pd.read_csv(
                pred_path,
                sep=" ",
                names=RankLibLearner.PRED_COLS,
            )
            # Shuffle predictions when predicted scores are equal to randomize the order.
            .sample(frac=1).reset_index(drop=True)
        )

        pred_df["rank"] = pred_df["score"].rank(method="min", ascending=False)

        return EvaluationService.evaluate(pred_df[EvaluationService.COLUMNS])
        

    def train_and_test_all(self, output_path, ranker, dataset_df, custom_ds_paths=None):
        results = {
            "build": [], 
            "apfd_min": [], "apfd_max": [], "apfd": [], "r_apfd": [],
            "apfdc_min": [], "apfdc_max": [], "apfdc": [], "r_apfdc": [],
            "time": []
        }
        
        ds_paths = list(p for p in output_path.glob("*") if p.is_dir()) if custom_ds_paths is None else custom_ds_paths 
        logging.info("Starting training phase")
        
        for build_ds_path in tqdm(ds_paths, desc="Training"):
            start = datetime.now()
            apfd, apfdc = self.train_and_test(
                build_ds_path, ranker[0], ranker[1], dataset_df
            )
            stop = datetime.now()

            results["build"].append(int(build_ds_path.name))
            
            results["apfd_min"].append(apfd.min)
            results["apfd_max"].append(apfd.max)
            results["apfd"].append(apfd.value)
            results["r_apfd"].append(apfd.value_norm)

            results["apfdc_min"].append(apfdc.min)
            results["apfdc_max"].append(apfdc.max)
            results["apfdc"].append(apfdc.value)
            results["r_apfdc"].append(apfdc.value_norm)

            results["time"].append(stop - start)

            if not (build_ds_path / "feature_stats.csv").exists():
                feature_stats_command = f"java -cp {self.ranklib_path};{self.math3_path} ciir.umass.edu.features.FeatureManager -feature_stats {build_ds_path / 'model.txt'}"
                feature_stats_out = subprocess.run(
                    feature_stats_command, shell=True, capture_output=True
                )
                if feature_stats_out.returncode != 0:
                    logging.error(f"Error in training:\n{feature_stats_out.stderr}")
                    sys.exit()
                self.extract_and_save_feature_stats(
                    feature_stats_out.stdout.decode("utf-8"), build_ds_path
                )

            # df = pd.DataFrame(results)
            # df.to_csv(output_path / "results.csv", index=False)

        results_df = pd.DataFrame(results)
        results_df["build_time"] = results_df["build"].apply(
            lambda b: self.build_time_d[b]
        )
        results_df.sort_values("build_time", ignore_index=True, inplace=True)
        results_df.drop("build_time", axis=1, inplace=True)
        
        print("Avg. APDFc", np.average(results_df['apfdc']), "std", np.std(results_df['apfdc']))

        return results_df

    def evaluate_heuristic(self, hname, suite_ds):
        asc_suite = suite_ds.sort_values(hname, ascending=True, ignore_index=True)
        asc_pred = pd.DataFrame(
            {
                "verdict": asc_suite[Feature.VERDICT].values,
                "duration": asc_suite[Feature.DURATION].values,
                "rank": asc_suite[hname].values,
            }
        )

        apfd_asc, apfdc_asc = EvaluationService.evaluate(asc_pred)

        dsc_suite = suite_ds.sort_values(hname, ascending=False, ignore_index=True)
        dsc_pred = pd.DataFrame(
            {
                "verdict": dsc_suite[Feature.VERDICT].values,
                "duration": dsc_suite[Feature.DURATION].values,
                "rank": dsc_suite[hname].values,
            }
        )

        apfd_dsc, apfdc_dsc = EvaluationService.evaluate(dsc_pred)

        return apfd_asc.value_norm, apfd_dsc.value_norm, apfdc_asc.value_norm, apfdc_dsc.value_norm

    def test_heuristics(self, dataset_df, results_path):
        apfd_results = {"build": []}
        apfdc_results = {"build": []}
        all_builds = dataset_df[Feature.BUILD].unique().tolist()
        all_builds.sort(key=lambda b: self.build_time_d[b])
        
        logging.info("Starting to test heuristics")
        for build in tqdm(all_builds, desc="Testing heuristics"):
            suite_ds = dataset_df[dataset_df[Feature.BUILD] == build]
            apfd_results["build"].append(build)
            apfdc_results["build"].append(build)
            for fname, fid in self.feature_id_map.items():
                apfd_asc, apfd_dsc, apfdc_asc, apfdc_dsc = self.evaluate_heuristic(
                    fname, suite_ds
                )

                apfd_results.setdefault(f"{fid}-asc", []).append(apfd_asc)
                apfd_results.setdefault(f"{fid}-dsc", []).append(apfd_dsc)
                apfdc_results.setdefault(f"{fid}-asc", []).append(apfdc_asc)
                apfdc_results.setdefault(f"{fid}-dsc", []).append(apfdc_dsc)
        
        pd.DataFrame(apfd_results).to_csv(
            results_path / "heuristic_apfd_results.csv", index=False
        )
        
        pd.DataFrame(apfdc_results).to_csv(
            results_path / "heuristic_apfdc_results.csv", index=False
        )

    def run_data_balancing_experiments(self, dataset_df, name, results_path, negatives: pd.DataFrame, ranker=None):
        if ranker == None:
            ranker = (self.config.best_ranker, self.config.best_ranker_params)
        
        logging.info("Converting data to RankLib format.")
        ranklib_ds = self.convert_to_ranklib_dataset(dataset_df)
        logging.info("Finished converting data to RankLib format.")
        
        def transform(ds: pd.DataFrame):
            positives = ds[ds['i_verdict'] > 0]
            excluded = negatives.sample(frac=1).reset_index(drop=True)['test'].to_list()
            before = len(ds)
            ds = ds[~ds["i_test"].isin(excluded[:len(excluded) // 2])]
            return pd.concat([ds, positives.sample(n=before - len(ds), replace=True)], ignore_index=True)
           
        traning_sets_path = results_path / name
        self.create_ranklib_training_sets(ranklib_ds, traning_sets_path, transform=transform)
        results = self.train_and_test_all(traning_sets_path, ranker, dataset_df)
        results.to_csv(traning_sets_path / f"results.csv", index=False)
        

    def run_feature_selection_experiments(self, dataset_df, name, results_path, ranker=None, oversampling=False):
        numberOfFeatures = dataset_df.shape[1] - 4
        logging.info(f"Using {numberOfFeatures} feature(s)")
        
        if ranker == None:
            ranker = (self.config.best_ranker, self.config.best_ranker_params)
        
        logging.info("Converting data to RankLib format.")
        ranklib_ds = self.convert_to_ranklib_dataset(dataset_df)
        logging.info("Finished converting data to RankLib format.")
        
        traning_sets_path = results_path / name
        self.create_ranklib_training_sets(ranklib_ds, traning_sets_path, force=True, oversampling=oversampling)

        all_ds_paths = list(p for p in traning_sets_path.glob("*") if p.is_dir())
        custom_ds_paths = FeatureSelectionService.pick_evenly_distributed_items(all_ds_paths, 10)
        results = self.train_and_test_all(traning_sets_path, ranker, dataset_df, custom_ds_paths)
        results.to_csv(traning_sets_path / f"results_{numberOfFeatures}.csv", index=False)
        
        candidates = FeatureSelectionService.get_feature_candidates_to_remove(custom_ds_paths)
        id_to_name = {v: k for k, v in self.feature_id_map.items()}
        dropped_feature_names = [id_to_name.get(feature_id) for feature_id in candidates if feature_id in id_to_name]

        return dropped_feature_names

    def run_accuracy_experiments(self, dataset_df, name, results_path, ranker=None, oversampling=False):
        if ranker == None:
            ranker = (self.config.best_ranker, self.config.best_ranker_params)
        logging.info("Converting data to RankLib format.")
        ranklib_ds = self.convert_to_ranklib_dataset(dataset_df, oversampling=False)
        logging.info("Finished converting data to RankLib format.")
        traning_sets_path = results_path / name
        self.create_ranklib_training_sets(ranklib_ds, traning_sets_path, oversampling=oversampling)
        results = self.train_and_test_all(traning_sets_path, ranker, dataset_df)
        results.to_csv(traning_sets_path / "results.csv", index=False)

    def convert_decay_datasets(self, datasets_path):
        original_ds_df = pd.read_csv(self.config.output_path / "dataset.csv")
        _, _, scaler = self.normalize_dataset(original_ds_df, None)
        decay_dataset_paths = list(p for p in datasets_path.glob("*") if p.is_dir())
        for decay_dataset_path in tqdm(
            decay_dataset_paths, desc="Converting decay datasets"
        ):
            test_file = decay_dataset_path / "test.txt"
            if test_file.exists():
                continue

            decay_ds_df = pd.read_csv(decay_dataset_path / "dataset.csv")
            # Reoder columns for MinMaxScaler
            decay_ds_df = decay_ds_df[original_ds_df.columns.tolist()]

            decay_ranklib_ds = self.convert_to_ranklib_dataset(decay_ds_df, scaler)
            decay_ranklib_ds.to_csv(
                test_file,
                sep=" ",
                header=False,
                index=False,
            )

    def test_decay_datasets(self, eval_model_paths, datasets_path):
        ranklib_path = Path("assets") / "RankLib.jar"
        for model_path in tqdm(eval_model_paths, desc=f"Testing models"):
            model_file = model_path / "model.txt"
            test_file = datasets_path / model_path.name / "test.txt"
            pred_file = datasets_path / model_path.name / "pred.txt"
            pred_command = f"java -jar {ranklib_path} -load {model_file} -rank {test_file} -indri {pred_file}"
            pred_out = subprocess.run(pred_command, shell=True, capture_output=True)
            if pred_out.returncode != 0:
                logging.error(f"Error in predicting:\n{pred_out.stderr}")
                sys.exit()
            preds_df = (
                pd.read_csv(
                    pred_file,
                    sep=" ",
                    names=RankLibLearner.PRED_COLS,
                )
                # Shuffle predictions when predicted scores are equal to randomize the order.
                .sample(frac=1).reset_index(drop=True)
            )
            pred_builds = preds_df["build"].unique().tolist()
            results = {"build": [], "apfd": [], "apfdc": []}
            for pred_build in pred_builds:
                pred_df = (
                    preds_df[preds_df["build"] == pred_build]
                    .copy()
                    .reset_index(drop=True)
                    .sort_values("score", ascending=False, ignore_index=True)
                )
                apfd = self.compute_apfd(pred_df)
                apfdc = self.compute_apfdc(pred_df)
                results["build"].append(pred_build)
                results["apfd"].append(apfd)
                results["apfdc"].append(apfdc)
            results_df = pd.DataFrame(results)
            results_df["build_time"] = results_df["build"].apply(
                lambda b: self.build_time_d[b]
            )
            results_df.sort_values("build_time", ignore_index=True, inplace=True)
            results_df.drop("build_time", axis=1, inplace=True)
            results_df.to_csv(
                datasets_path / model_path.name / "results.csv",
                index=False,
            )

    def run_decay_test_experiments(self, datasets_path, models_path):
        self.convert_decay_datasets(datasets_path)
        model_paths = list(p for p in models_path.glob("*") if p.is_dir())
        self.test_decay_datasets(model_paths, datasets_path)
