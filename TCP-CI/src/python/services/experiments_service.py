import pandas as pd
# from ..decay_dataset_factory import DecayDatasetFactory
# from ..dataset_factory import DatasetFactory
from ..feature_extractor.feature import Feature
# from ..module_factory import ModuleFactory
import sys
from ..ranklib_learner import RankLibLearner
# from ..code_analyzer.code_analyzer import AnalysisLevel
# from ..results.results_analyzer import ResultAnalyzer
from ..hyp_param_opt import HypParamOpt
from .data_service import DataService
from ..rl.TestcaseExecutionDataLoader import TestCaseExecutionDataLoader
from pathlib import Path
from enum import Enum
import logging
import os

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


class Experiment(Enum):
    FULL = "FULL"
    WO_IMP = "WO_IMP"
    WO_TES_COM = "WO_TES_COM"
    WO_TES_PRO = "WO_TES_PRO"
    WO_TES_CHN = "WO_TES_CHN"
    WO_REC = "WO_REC"
    WO_COV = "WO_COV"
    WO_COD_COV_COM = "WO_COD_COV_COM"
    WO_COD_COV_PRO = "WO_COD_COV_PRO"
    WO_COD_COV_CHN = "WO_COD_COV_CHN"
    WO_DET_COV = "WO_DET_COV"
    W_Code = "W_Code"
    W_Execution = "W_Execution"
    W_Coverage = "W_Coverage"


class ExperimentsService:
    @staticmethod
    def run_rl_based_experiments(args):
        features = args.features
        exp_name = "rl"

        if args.oversampling:
            exp_name += "-os"

        if features is not None and len(features) > 0:
            exp_name += f"-f{len(features)}"

        if args.replay_ratio > 0:
            exp_name += f"-er{args.replay_ratio}"

        if args.policy == "LSTM":
            exp_name += f"-lstm"

        if args.exp_name is not None:
            exp_name = args.exp_name

        loader = TestCaseExecutionDataLoader(args.output_path)
        loader.load_data()
        loader.test_data = DataService.remove_outlier_tests(
            args.output_path, loader.test_data
        )

        ci_logs_train, ci_logs_test = loader.pre_process(features=features, oversampling=args.oversampling)

        from ..rl.TPDRL import experiment
        from ..rl.Config import Config

        conf = Config()
        conf.win_size = 4
        conf.cycle_count = 999999
        conf.first_cycle = 0

        conf.output_path = str(args.output_path / "tsp_accuracy_results" / exp_name)
        conf.log_file = str(conf.output_path + "\\log.txt")
        
        experiment(
            mode=args.rlenv, 
            algo=("ACER", {"replay_ratio": args.replay_ratio, "policy": args.policy}), 
            train_ds=ci_logs_train,
            test_ds=ci_logs_test, 
            start_cycle=0, 
            end_cycle=99999999, 
            episodes=200, 
            model_path=conf.output_path,
            dataset_name="",
            conf=conf)

        return
        
    
    @staticmethod
    def run_feature_selection_experiments(args): 
        dataset_path = args.output_path / "dataset.csv"
        if not dataset_path.exists():
            logging.error("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        
        logging.info("Reading the dataset.")
        learner = RankLibLearner(args)
        dataset_df = pd.read_csv(dataset_path)
        builds_count = dataset_df[Feature.BUILD].nunique()
        if builds_count <= args.test_count:
            logging.error(
                f"Not enough builds for training: require at least {args.test_count + 1}, found {builds_count}"
            )
            sys.exit()
        
        results_path = args.output_path / "tsp_accuracy_results"
        outliers_dataset_df = DataService.remove_outlier_tests(
            args.output_path, dataset_df
        )
        logging.info("Finished reading the dataset.")

        logging.info(
            f"***** Running RFE experiment for {dataset_path.parent.name} *****"
        )

        exp_name = 'feature-selection'
        if args.oversampling:
            exp_name += '-os'
        
        features_file_path = os.path.join(results_path, exp_name, "dropped.txt")
        line = read_last_line(features_file_path)
        dropped_features = line.split(";") if line else []
        while len(dropped_features) <= 145:            
            logging.info(f"Number of dropped features: {len(dropped_features)}")

            learner.update_dropped_features(dropped_features)
            new_dropped_features = learner.run_feature_selection_experiments(
                outliers_dataset_df.drop(dropped_features, axis=1),
                exp_name,
                results_path,
                oversampling=args.oversampling
            )

            dropped_features.extend(new_dropped_features)
            with open(features_file_path, 'a') as file:
                file.write(";".join(dropped_features) + "\n")
        
        logging.info("Done feature selection experiments")
    
    @staticmethod
    def run_best_ranker_experiments(args):
        dataset_path = args.output_path / "dataset.csv"
        if not dataset_path.exists():
            logging.error("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        logging.info("Reading the dataset.")
        learner = RankLibLearner(args)
        dataset_df = pd.read_csv(dataset_path)
        builds_count = dataset_df[Feature.BUILD].nunique()
        if builds_count <= args.test_count:
            logging.error(
                f"Not enough builds for training: require at least {args.test_count + 1}, found {builds_count}"
            )
            sys.exit()
        results_path = args.output_path / "tsp_accuracy_results"
        outliers_dataset_df = DataService.remove_outlier_tests(
            args.output_path, dataset_df
        )
        logging.info("Finished reading the dataset.")

        logging.info(
            f"***** Running {args.experiment.value} experiment for {dataset_path.parent.name} *****"
        )
        
        if args.experiment == Experiment.FULL:
            exp_name = "full-outliers"
            if args.oversampling:
                exp_name += "-os3"
            
            if args.features:
                exp_name += f"-f{len(args.features)}"
                x = [Feature.BUILD, Feature.VERDICT, Feature.TEST, Feature.DURATION]
                x.extend(args.features)
                outliers_dataset_df = outliers_dataset_df[x]

            learner.run_accuracy_experiments(
                outliers_dataset_df, exp_name, results_path, oversampling=args.oversampling
            )
        
        
        elif args.experiment == Experiment.WO_IMP:
            learner.run_accuracy_experiments(
                outliers_dataset_df.drop(Feature.IMPACTED_FEATURES, axis=1),
                "wo-impacted-outliers",
                results_path,
            )
        elif (
            args.experiment.value.startswith("WO_")
            and args.experiment != Experiment.WO_IMP
        ):
            feature_groups_names = {
                "TES_COM": Feature.TES_COM,
                "TES_PRO": Feature.TES_PRO,
                "TES_CHN": Feature.TES_CHN,
                "REC": Feature.REC,
                "COV": Feature.COV,
                "COD_COV_COM": Feature.COD_COV_COM,
                "COD_COV_PRO": Feature.COD_COV_PRO,
                "COD_COV_CHN": Feature.COD_COV_CHN,
                "DET_COV": Feature.DET_COV,
            }
            feature_group = args.experiment.value[3:]
            names = feature_groups_names[feature_group]
            learner.run_accuracy_experiments(
                outliers_dataset_df.drop(names, axis=1),
                f"wo-{feature_group}-outliers",
                results_path,
            )
        elif args.experiment.value.startswith("W_"):
            test_code_features = Feature.TES_COM + Feature.TES_PRO + Feature.TES_CHN
            test_execution_features = Feature.REC
            test_coverage_features = (
                Feature.COV
                + Feature.COD_COV_COM
                + Feature.COD_COV_PRO
                + Feature.COD_COV_CHN
                + Feature.DET_COV
            )
            high_level_feature_groups = {
                "Code": test_code_features,
                "Execution": test_execution_features,
                "Coverage": test_coverage_features,
            }
            non_feature_cols = [
                Feature.BUILD,
                Feature.TEST,
                Feature.VERDICT,
                Feature.DURATION,
            ]
            feature_group = args.experiment.value[2:]
            names = high_level_feature_groups[feature_group]
            learner.run_accuracy_experiments(
                outliers_dataset_df[non_feature_cols + names],
                f"W-{feature_group}-outliers",
                results_path,
            )
        logging.info("Done run_best_ranker_experiments")

    @staticmethod
    def run_all_tcp_rankers(args):
        dataset_path = args.output_path / "dataset.csv"
        if not dataset_path.exists():
            logging.error("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        logging.info(f"##### Running experiments for {dataset_path.parent.name} #####")
        learner = RankLibLearner(args)
        dataset_df = pd.read_csv(dataset_path)
        builds_count = dataset_df[Feature.BUILD].nunique()
        if builds_count <= args.test_count:
            logging.error(
                f"Not enough builds for training: require at least {args.test_count + 1}, found {builds_count}"
            )
            sys.exit()
        outliers_dataset_df = DataService.remove_outlier_tests(
            args.output_path, dataset_df
        )
        rankers = {
            0: ("MART", {"tree": 30}),
            6: (
                "LambdaMART",
                {"tree": 30, "metric2T": "NDCG@10", "metric2t": "NDCG@10"},
            ),
            2: ("RankBoost", {}),
            4: ("CoordinateAscent", {}),
            7: ("ListNet", {}),
            8: ("RandomForest", {}),
        }
        results_path = args.output_path / "tcp_rankers"
        for id, info in rankers.items():
            name, params = info
            logging.info(
                f"***** Running {name} full feature set without Outliers experiments *****"
            )
            learner.run_accuracy_experiments(
                outliers_dataset_df, name, results_path, ranker=(id, params)
            )

    @staticmethod
    def run_decay_test_experiments(args):
        pass
        # logging.info(f"Running decay tests for {args.output_path.name}")
        # repo_miner_class = ModuleFactory.get_repository_miner(AnalysisLevel.FILE)
        # repo_miner = repo_miner_class(args)
        # change_history_df = repo_miner.load_entity_change_history()
        # dataset_factory = DatasetFactory(
        #     args,
        #     change_history_df,
        #     repo_miner,
        # )
        # dataset_df = pd.read_csv(args.output_path / "dataset.csv")
        # decay_ds_factory = DecayDatasetFactory(dataset_factory, args)
        # models_path = args.output_path / "tsp_accuracy_results" / "full-outliers"
        # decay_ds_factory.create_decay_datasets(dataset_df, models_path)

        # learner = RankLibLearner(args)
        # datasets_path = args.output_path / "decay_datasets"
        # learner.run_decay_test_experiments(datasets_path, models_path)
        # logging.info(f"All finished and results are saved at {datasets_path}")
        # print()

    @staticmethod
    def analyze_results(args):
        pass
        # result_analyzer = ResultAnalyzer(args)
        # result_analyzer.analyze_results()

    @staticmethod
    def hyp_param_opt(args):
        pass
        # optimizer = HypParamOpt(args)
        # logging.info(f"***** Running {args.output_path.name} hypopt *****")
        # build_ds_path = Path(args.output_path / "hyp_param_opt" / str(args.build))
        # optimizer.run_optimization(build_ds_path, args.comb_index)
