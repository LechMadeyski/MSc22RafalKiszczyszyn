import os

# os.add_dll_directory(os.environ["SCITOOLS_HOME"])
os.environ["OUTDATED_IGNORE"] = "1"

# from src.python.services.data_collection_service import DataCollectionService
from src.python.services.experiments_service import ExperimentsService, Experiment
import argparse
# from src.python.code_analyzer.code_analyzer import AnalysisLevel
# from src.python.entities.entity import Language
from pathlib import Path
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def dataset(args):
    # DataCollectionService.create_dataset(args)
    pass

def tr_torrent(args):
    # DataCollectionService.process_tr_torrent(args)
    pass

def learn(args):
    if args.ranking_models == "best":
        ExperimentsService.run_best_ranker_experiments(args)
    elif args.ranking_models == "all":
        ExperimentsService.run_all_tcp_rankers(args)

def learn_rl(args):
    ExperimentsService.run_rl_based_experiments(args)

def select_features(args):
    ExperimentsService.run_feature_selection_experiments(args)


def hypopt(args):
    ExperimentsService.hyp_param_opt(args)


def decay_test(args):
    ExperimentsService.run_decay_test_experiments(args)


def results(args):
    ExperimentsService.analyze_results(args)


def add_dataset_parser_arguments(parser):
    parser.add_argument(
        "-p",
        "--project-path",
        help="Project's source code git repository path.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--project-slug",
        help="The project's GitHub slug, e.g., apache/commons.",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--ci-data-path",
        help="Path to CI datasource root directory, including RTP-Torrent and Travis Torrent.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--test-path",
        help="Specifies the relative root directory of the test source code.",
        type=Path,
        default=None,
    )
    # parser.add_argument(
    #     "-l",
    #     "--level",
    #     help="Specifies the granularity of feature extraction.",
    #     type=AnalysisLevel,
    #     choices=[AnalysisLevel.FILE],
    #     default=AnalysisLevel.FILE,
    # )
    parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    # parser.add_argument(
    #     "--language",
    #     help="Project's main language",
    #     type=Language,
    #     choices=[Language.JAVA],
    #     default=Language.JAVA,
    # )
    parser.add_argument(
        "-n",
        "--build-window",
        help="Specifies the number of recent builds to consider for computing features.",
        type=int,
        required=False,
        default=6,
    )


ARGS = None

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    tr_torrent_parser = subparsers.add_parser(
        "tr_torrent",
        help="Process travis torrent logs and build info.",
    )
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Create training dataset including all test case features for each CI cycle.",
    )
    learn_parser = subparsers.add_parser(
        "learn",
        help="Perform learning experiments on collected features using RankLib.",
    )
    learn_rl_parser = subparsers.add_parser(
        "rl",
        help="Perform learning experiments using pairwise ACER.",
    )
    feature_selection_parser = subparsers.add_parser(
        "rfe",
        help="Select best features using RFE algorithm and best ML ranking model in RankLib."
    )
    hypopt_parser = subparsers.add_parser(
        "hypopt",
        help="Perform hyperparameter optimization for the best ML ranking model in RankLib.",
    )
    decay_test_parser = subparsers.add_parser(
        "decay_test",
        help="Perform ML ranking models decay test experiments on trained models.",
    )
    results_parser = subparsers.add_parser(
        "results",
        help="Analyze the results from experiments and generate tables.",
    )

    add_dataset_parser_arguments(dataset_parser)
    dataset_parser.set_defaults(func=dataset)

    tr_torrent_parser.set_defaults(func=tr_torrent)
    tr_torrent_parser.add_argument(
        "-r",
        "--repo",
        help="The login and name of the repo seperated by @ (e.g., presto@prestodb)",
        type=str,
        required=True,
    )
    tr_torrent_parser.add_argument(
        "-i",
        "--input-path",
        help="Specifies the directory to of travis torrent raw data.",
        type=Path,
        required=True,
    )
    tr_torrent_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save resulting data.",
        type=Path,
        default=".",
    )

    learn_parser.set_defaults(func=learn)
    learn_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    learn_parser.add_argument(
        "-t",
        "--test-count",
        help="Specifies the number of recent builds to test the trained models on.",
        type=int,
        default=50,
    )
    learn_parser.add_argument(
        "-r",
        "--ranking-models",
        help="Specifies the ranking model(s) to use for learning.",
        type=str,
        default="best",
        choices=["best", "all"],
    )
    learn_parser.add_argument(
        "-e",
        "--experiment",
        help="Specifies the experiment to run. Only works when the best ranking model is selected.",
        type=Experiment,
        default=Experiment.FULL,
        choices=Experiment,
    )

    learn_rl_parser.set_defaults(func=learn_rl)
    learn_rl_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    learn_rl_parser.add_argument(
        "-t",
        "--test-count",
        help="Specifies the number of recent builds to test the trained models on.",
        type=int,
        default=50,
    )

    feature_selection_parser.set_defaults(func=select_features)
    feature_selection_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    feature_selection_parser.add_argument(
        "-t",
        "--test-count",
        help="Specifies the number of recent builds to test the trained models on.",
        type=int,
        default=50,
    )

    hypopt_parser.set_defaults(func=hypopt)
    hypopt_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory of all datasets.",
        type=Path,
        default=".",
    )
    hypopt_parser.add_argument(
        "-b",
        "--build",
        help="Specifies the build id for running the optimization.",
        type=int,
        required=True,
    )
    hypopt_parser.add_argument(
        "-i",
        "--comb-index",
        help="Specifies the index of the hyperparameter combination.",
        type=int,
        required=True,
    )

    decay_test_parser.set_defaults(func=decay_test)
    decay_test_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    decay_test_parser.add_argument(
        "-p",
        "--project-path",
        help="Project's source code git repository path.",
        type=Path,
        default=None,
    )

    results_parser.set_defaults(func=results)
    results_parser.add_argument(
        "-d",
        "--data-path",
        help="Path to the root folder of all datasets.",
        type=Path,
        default=None,
    )
    results_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save resulting tables.",
        type=Path,
        default=".",
    )

    args = parser.parse_args(ARGS)
    args.features = ['REC_Age', 'TES_PRO_AllCommitersExperience', 'TES_PRO_OwnersExperience', 'TES_COM_CountLineCodeDecl', 'REC_TotalMaxExeTime', 'TES_PRO_CommitCount', 'TES_COM_CountStmtDecl', 'TES_COM_CountLine', 'TES_PRO_OwnersContribution', 'TES_COM_CountLineBlank', 'TES_COM_RatioCommentToCode', 'TES_COM_CountStmt', 'REC_TotalAvgExeTime', 'REC_RecentAvgExeTime', 'TES_COM_CountStmtExe', 'TES_COM_CountLineCodeExe', 'REC_LastExeTime', 'TES_COM_CountLineCode', 'REC_RecentMaxExeTime', 'TES_COM_CountLineComment']
    args.oversampling = True
    args.replay_ratio = 0
    args.policy = "LSTM"

    args.output_path.mkdir(parents=True, exist_ok=True)
    args.unique_separator = "\t"
    args.best_ranker = 8
    args.best_ranker_params = {
        "rtype": 0,
        "srate": 0.5,
        "bag": 150,
        "frate": 0.3,
        "tree": 5,
        "leaf": 200,
        "shrinkage": 0.2,
    }
    args.func(args)


if __name__ == "__main__":
    workdir = "C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\datasets"
    ARGS = ["rl", "-t", "50", "-o", workdir + "\\spring-cloud@spring-cloud-dataflow"]
    main()
    # subjects = []
    # for item in os.listdir(workdir):
    #     item_path = os.path.join(workdir, item)
    #     if os.path.isdir(item_path):
    #         subjects.append(item_path)
    
    # for subject in subjects:
    #     ARGS = ["rfe", "-t", "30", "-o", subject]
    #     main()
