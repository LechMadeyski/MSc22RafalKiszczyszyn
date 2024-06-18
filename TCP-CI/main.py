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

K = None

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
EXP_DATA = {
  "oversampling": False,
  "K": None,
  "rlenv": "PAIRWISE",
  "exp_name": None  
}


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

    FEATURES = {
        15: ['REC_Age', 'TES_PRO_OwnersExperience', 'TES_PRO_AllCommitersExperience', 'REC_TotalMaxExeTime', 'REC_TotalAvgExeTime', 'REC_RecentAvgExeTime', 'TES_COM_CountLine', 'TES_COM_CountLineCodeDecl', 'REC_LastExeTime', 'REC_RecentMaxExeTime', 'TES_PRO_OwnersContribution', 'TES_PRO_CommitCount', 'TES_COM_CountStmtExe', 'TES_COM_RatioCommentToCode', 'TES_COM_CountStmt'],
        30: ['TES_PRO_CommitCount', 'TES_COM_CountLine', 'TES_PRO_OwnersExperience', 'TES_PRO_AllCommitersExperience', 'TES_COM_CountLineBlank', 'REC_Age', 'TES_COM_CountLineCodeDecl', 'REC_TotalMaxExeTime', 'TES_COM_CountStmtDecl', 'TES_COM_CountStmt', 'TES_COM_CountStmtExe', 'REC_LastExeTime', 'REC_TotalAvgExeTime', 'TES_COM_CountLineCodeExe', 'TES_COM_RatioCommentToCode', 'REC_RecentMaxExeTime', 'TES_COM_CountLineCode', 'REC_RecentAvgExeTime', 'COD_COV_PRO_IMP_AllCommitersExperience', 'TES_PRO_OwnersContribution', 'COD_COV_PRO_IMP_OwnersExperience', 'REC_TotalFailRate', 'REC_LastTransitionAge', 'COV_ImpScoreSum', 'COD_COV_COM_IMP_RatioCommentToCode', 'REC_TotalTransitionRate', 'COD_COV_PRO_IMP_CommitCount', 'TES_COM_CountLineComment', 'TES_COM_SumCyclomaticStrict', 'TES_COM_CountDeclMethodPublic'],
        50: ['TES_COM_CountLineCodeExe', 'TES_COM_CountStmtDecl', 'TES_PRO_CommitCount', 'TES_COM_CountStmt', 'TES_COM_CountStmtExe', 'TES_COM_CountLine', 'TES_PRO_OwnersExperience', 'TES_PRO_AllCommitersExperience', 'TES_COM_SumCyclomaticStrict', 'TES_COM_CountLineBlank', 'TES_COM_RatioCommentToCode', 'REC_Age', 'TES_COM_CountLineCode', 'TES_COM_CountLineComment', 'TES_COM_CountLineCodeDecl', 'REC_TotalMaxExeTime', 'REC_TotalAvgExeTime', 'TES_PRO_OwnersContribution', 'REC_LastExeTime', 'REC_TotalFailRate', 'REC_RecentMaxExeTime', 'REC_RecentAvgExeTime', 'TES_COM_SumCyclomaticModified', 'TES_COM_SumCyclomatic', 'COD_COV_PRO_IMP_AllCommitersExperience', 'REC_TotalTransitionRate', 'TES_COM_CountDeclMethodPublic', 'COD_COV_PRO_IMP_CommitCount', 'COD_COV_COM_IMP_RatioCommentToCode', 'REC_LastFailureAge', 'COD_COV_PRO_IMP_OwnersExperience', 'COV_ImpScoreSum', 'COD_COV_PRO_IMP_OwnersContribution', 'REC_LastTransitionAge', 'TES_COM_CountDeclInstanceMethod', 'REC_TotalExcRate', 'COD_COV_COM_IMP_CountDeclClassVariable', 'COD_COV_COM_IMP_CountLineComment', 'COD_COV_COM_IMP_CountDeclInstanceVariable', 'TES_COM_SumEssential', 'COD_COV_COM_C_RatioCommentToCode', 'COD_COV_CHN_C_LinesAdded', 'TES_COM_CountDeclMethod', 'COD_COV_COM_IMP_MaxCyclomaticStrict', 'REC_MaxTestFileTransitionRate', 'TES_COM_CountDeclExecutableUnit', 'REC_MaxTestFileFailRate', 'COD_COV_COM_IMP_CountDeclMethodDefault', 'COD_COV_PRO_C_OwnersExperience', 'COD_COV_COM_IMP_CountLineCodeDecl'],
        80: ['TES_COM_SumCyclomaticModified', 'TES_COM_CountLineCodeExe', 'TES_COM_CountStmtDecl', 'TES_PRO_CommitCount', 'TES_COM_CountStmt', 'TES_COM_SumCyclomatic', 'TES_COM_CountStmtExe', 'COD_COV_PRO_IMP_AllCommitersExperience', 'TES_COM_CountLine', 'TES_PRO_OwnersExperience', 'TES_PRO_AllCommitersExperience', 'TES_COM_SumCyclomaticStrict', 'TES_COM_CountDeclMethodPublic', 'TES_COM_CountLineBlank', 'TES_COM_RatioCommentToCode', 'REC_Age', 'TES_COM_CountLineCode', 'TES_COM_CountLineComment', 'REC_RecentAvgExeTime', 'TES_COM_CountLineCodeDecl', 'COD_COV_COM_IMP_RatioCommentToCode', 'REC_TotalMaxExeTime', 'REC_TotalAvgExeTime', 'COV_ImpScoreSum', 'TES_PRO_OwnersContribution', 'REC_TotalTransitionRate', 'TES_COM_SumEssential', 'REC_LastExeTime', 'REC_LastTransitionAge', 'REC_LastFailureAge', 'REC_TotalFailRate', 'COD_COV_PRO_IMP_OwnersExperience', 'TES_COM_CountDeclFunction', 'TES_COM_CountDeclExecutableUnit', 'COD_COV_COM_IMP_CountLineComment', 'TES_COM_CountDeclInstanceMethod', 'REC_RecentMaxExeTime', 'COD_COV_PRO_IMP_CommitCount', 'TES_COM_MaxCyclomaticModified', 'COD_COV_COM_IMP_CountDeclInstanceVariable', 'TES_COM_CountDeclMethod', 'COD_COV_PRO_IMP_OwnersContribution', 'COD_COV_COM_IMP_CountDeclInstanceMethod', 'TES_COM_MaxCyclomatic', 'COV_ImpCount', 'REC_TotalExcRate', 'COD_COV_COM_IMP_CountLineCodeExe', 'REC_MaxTestFileTransitionRate', 'TES_COM_MaxCyclomaticStrict', 'TES_PRO_DistinctDevCount', 'COD_COV_COM_IMP_CountDeclClassVariable', 'COD_COV_COM_IMP_CountDeclClassMethod', 'REC_TotalAssertRate', 'COD_COV_COM_IMP_CountDeclMethodPublic', 'COD_COV_PRO_IMP_MinorContributorCount', 'COD_COV_COM_IMP_MaxCyclomaticStrict', 'TES_COM_MaxNesting', 'DET_COV_IMP_Faults', 'COD_COV_COM_IMP_MaxEssential', 'COD_COV_PRO_IMP_DistinctDevCount', 'REC_MaxTestFileFailRate', 'COD_COV_COM_IMP_MaxNesting', 'COD_COV_COM_IMP_CountLineCodeDecl', 'COD_COV_COM_IMP_CountStmtDecl', 'COV_ChnScoreSum', 'COD_COV_CHN_C_LinesAdded', 'TES_PRO_MinorContributorCount', 'TES_COM_CountDeclClassVariable', 'COD_COV_COM_IMP_MaxCyclomatic', 'TES_COM_CountDeclInstanceVariable', 'COD_COV_COM_IMP_CountDeclMethodDefault', 'COD_COV_PRO_C_OwnersContribution', 'REC_RecentTransitionRate', 'COD_COV_PRO_C_OwnersExperience', 'COD_COV_CHN_C_LinesDeleted', 'COD_COV_PRO_C_AllCommitersExperience', 'COD_COV_CHN_C_AddedChangeScattering', 'COD_COV_COM_IMP_MaxCyclomaticModified', 'COD_COV_COM_IMP_CountDeclMethodProtected', 'REC_LastVerdict']
    }

    FEATURES_OS = {
        15: ['REC_Age', 'REC_TotalMaxExeTime', 'TES_PRO_OwnersExperience', 'TES_PRO_AllCommitersExperience', 'REC_LastExeTime', 'REC_RecentAvgExeTime', 'REC_RecentMaxExeTime', 'REC_TotalAvgExeTime', 'TES_COM_CountLine', 'TES_PRO_OwnersContribution', 'TES_PRO_CommitCount', 'TES_COM_CountLineCodeDecl', 'TES_COM_CountStmtDecl', 'REC_TotalFailRate', 'COD_COV_PRO_IMP_AllCommitersExperience'],
        30: ['TES_PRO_AllCommitersExperience', 'REC_TotalMaxExeTime', 'TES_PRO_CommitCount', 'REC_Age', 'REC_LastExeTime', 'REC_RecentMaxExeTime', 'TES_COM_CountLine', 'TES_PRO_OwnersExperience', 'TES_COM_RatioCommentToCode', 'TES_COM_CountLineCodeDecl', 'TES_COM_CountStmtDecl', 'REC_RecentAvgExeTime', 'TES_COM_CountStmt', 'TES_COM_CountLineBlank', 'REC_TotalAvgExeTime', 'TES_COM_CountStmtExe', 'TES_COM_CountLineCodeExe', 'TES_COM_CountLineCode', 'REC_TotalTransitionRate', 'TES_PRO_OwnersContribution', 'REC_TotalFailRate', 'COD_COV_PRO_IMP_AllCommitersExperience', 'COD_COV_PRO_IMP_OwnersExperience', 'REC_LastTransitionAge', 'REC_TotalExcRate', 'TES_COM_CountLineComment', 'COD_COV_COM_IMP_RatioCommentToCode', 'REC_LastFailureAge', 'COV_ImpScoreSum', 'TES_COM_CountDeclMethodPublic']
    }

    args = parser.parse_args(ARGS)
    
    args.exp_name = EXP_DATA["exp_name"]
    args.oversampling = EXP_DATA["oversampling"]
    K = EXP_DATA["K"]
    if args.oversampling:
        args.features = FEATURES_OS[K] if K else None
    else:
        args.features = FEATURES[K] if K else None
    
    args.replay_ratio = 0
    args.policy = "MLP"
    args.rlenv = EXP_DATA["rlenv"]

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


SELECTED = ['S2', 'S8', 'S9', 'S12', 'S13', 'S14', 'S16', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25']


def run(data, args: list):
    global EXP_DATA
    global ARGS
    
    db_path = "C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\datasets.csv"
    workdir = "C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\datasets\\"
    db = pd.read_csv(db_path)
    
    for _, subject in db.iterrows():
        sid = subject['SID']
        if sid not in SELECTED:
            continue

        EXP_DATA = data[sid] 
        subject_path = subject['Subject'].replace("/", "@")
        args.extend(["-o", workdir + subject_path])
        ARGS = args 
        # ["learn", "-t", "50", "-r", "best", "-e", "FULL", "-o", workdir + subject_path]
        # ARGS = ["rfe", "-t", "50", "-o", workdir + subject_path]
        # ARGS = ["rl", "-t", "50", "-o", workdir + subject_path]
        main()
    pass


def exp1():
    data = {sid: {"oversampling": False, "K": None, "rlenv": "PAIRWISE", "exp_name": None} for sid in SELECTED}
    run(data, ["learn", "-t", "50", "-r", "best", "-e", "FULL"])
    run(data, ["rl", "-t", "50"])

def exp2_1():
    data = {sid: {"oversampling": False, "K": None, "rlenv": "PAIRWISE", "exp_name": None} for sid in SELECTED}
    run(data, ["rfe", "-t", "50"])

def exp2_2():
    for k in [80, 50, 30, 15]:
        data = {sid: {"oversampling": False, "K": k, "rlenv": "PAIRWISE", "exp_name": None} for sid in SELECTED}
        run(data, ["learn", "-t", "50", "-r", "best", "-e", "FULL"])
        run(data, ["rl", "-t", "50"])

def exp3_1():
    data = {sid: {"oversampling": True, "K": None, "rlenv": "PAIRWISE", "exp_name": None} for sid in SELECTED}
    run(data, ["learn", "-t", "50", "-r", "best", "-e", "FULL"])
    run(data, ["rl", "-t", "50"])

def exp3_2():
    data = {sid: {"oversampling": True, "K": None, "rlenv": "PAIRWISE", "exp_name": None} for sid in SELECTED}
    run(data, ["rfe", "-t", "50"])

def exp3_3():
    for k in [30, 15]:
        data = {sid: {"oversampling": True, "K": k, "rlenv": "PAIRWISE", "exp_name": None} for sid in SELECTED}
        run(data, ["rl", "-t", "50"])

def exp4():
    # data = {sid: {"oversampling": False, "K": 15, "rlenv": "DIFF", "exp_name": "rl-diff-f15"} for sid in SELECTED}
    # run(data, ["rl", "-t", "50"])
    data = {sid: {"oversampling": True, "K": 15, "rlenv": "DIFF", "exp_name": f"rl-diff-os-f15"} for sid in ['S2', 'S8', 'S9']}
    run(data, ["rl", "-t", "50"])
    

if __name__ == "__main__":
    import pandas as pd
    import random
    import numpy as np

    random.seed(44)
    np.random.seed(44)

    exp4()
