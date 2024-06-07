import argparse
import pandas as pd
import numpy as np
import math
import os
from datetime import datetime
from statistics import mean


from .TPAgentUtil import TPAgentUtil
from .PairWiseEnv import CIPairWiseEnv
from .TPPairWiseDQNAgent import TPPairWiseDQNAgent
from .ci_cycle import CICycleLog
from .Config import Config
from .TestcaseExecutionDataLoader import TestCaseExecutionDataLoader
from .CustomCallback import CustomCallback
from stable_baselines.bench import Monitor
from pathlib import Path
from .CIListWiseEnvMultiAction import CIListWiseEnvMultiAction
from .CIListWiseEnv import CIListWiseEnv
from .PointWiseEnv import CIPointWiseEnv
import sys


# find the cycle with maximum number of test cases
def get_max_test_cases_count(cycle_logs:[]):
    max_test_cases_count = 0
    for cycle_log in cycle_logs:
        if cycle_log.get_test_cases_count() > max_test_cases_count:
            max_test_cases_count = cycle_log.get_test_cases_count()
    return max_test_cases_count


def experiment(mode, algo, train_ds, test_ds, start_cycle, end_cycle, episodes, model_path, dataset_name, conf, verbos=False):
    algo, algo_params = algo
    results = {"build": [], "time": []}

    log_dir = os.path.dirname(conf.log_file)
#    -- fix end cycle issue
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if start_cycle <= 0:
        start_cycle = 0

    if end_cycle >= len(train_ds)-1:
        end_cycle = len(train_ds)
    # check for max cycle and end_cycle and set end_cycle to max if it is larger than max
    first_round: bool = True
    if start_cycle > 0:
        first_round = False
        previous_model_path = model_path + "/" + mode + "_" + algo + dataset_name + "_" + str(
            0) + "_" + str(start_cycle-1)
    model_save_path = None

    for i in range(start_cycle, end_cycle - 1):
        if (train_ds[i].get_test_cases_count() < 6) or \
                ( (conf.dataset_type == "simple") and
                  (train_ds[i].get_failed_test_cases_count() < 1)):
            continue
        if mode.upper() == 'PAIRWISE':
            N = train_ds[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N,2)+1)))
            env = CIPairWiseEnv(train_ds[i], conf)
        elif mode.upper() == 'POINTWISE':
            N = train_ds[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N,2)+1)))
            env = CIPointWiseEnv(train_ds[i], conf)
        elif mode.upper() == 'LISTWISE':
            conf.max_test_cases_count = get_max_test_cases_count(train_ds)
            N = train_ds[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N,2)+1)))
            env = CIListWiseEnv(train_ds[i], conf)
        elif mode.upper() == 'LISTWISE2':
            conf.max_test_cases_count = get_max_test_cases_count(train_ds)
            N = train_ds[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N,2)+1)))
            env = CIListWiseEnvMultiAction(train_ds[i], conf)
        
        start = datetime.now()
        print("Training agent with replaying of cycle " + str(i) + " with steps " + str(steps))

        if model_save_path:
            previous_model_path = model_save_path
        model_save_path = model_path + "/" + mode + "_" + algo + dataset_name + "_" + str(
            start_cycle) + "_" + str(i)
        env = Monitor(env, model_save_path +"_monitor.csv")
        callback_class = CustomCallback(svae_path=model_save_path,
                                        check_freq=int(steps/episodes), log_dir=log_dir, verbose=verbos)

        if first_round:
            tp_agent = TPAgentUtil.create_model(algo, env, params=algo_params)
            tp_agent.learn(total_timesteps=steps, reset_num_timesteps=True, callback=callback_class)
            first_round = False
        else:
            tp_agent = TPAgentUtil.load_model(algo=algo, env=env, path=previous_model_path+".zip")
            tp_agent.learn(total_timesteps=steps, reset_num_timesteps=True, callback=callback_class)
            
        print("Training agent with replaying of cycle " + str(i) + " is finished")

        j = i+1   # test trained agent on next cycles
        while (((test_ds[j].get_test_cases_count() < 6)
               or ((conf.dataset_type == "simple") and (test_ds[j].get_failed_test_cases_count() == 0) ))
               and (j < end_cycle)):
            # or test_case_data[j].get_failed_test_cases_count() == 0) \
            j = j+1
            if j > end_cycle-1:
                break
        if j > end_cycle - 1:
            break

        if mode.upper() == 'PAIRWISE':
            env_test = CIPairWiseEnv(test_ds[j], conf)
        elif mode.upper() == 'POINTWISE':
            env_test = CIPointWiseEnv(test_ds[j], conf)
        elif mode.upper() == 'LISTWISE':
            env_test = CIListWiseEnv(test_ds[j], conf)
        elif mode.upper() == 'LISTWISE2':
            env_test = CIListWiseEnvMultiAction(test_ds[j], conf)

        test_case_vector = TPAgentUtil.test_agent(env=env_test, algo=algo, model_path=model_save_path+".zip", mode=mode)
        test_case_id_vector = []


        for test_case in test_case_vector:
            test_case_id_vector.append(str(test_case['test_id']))
            cycle_id_text = test_case['cycle_id']
        
        stop = datetime.now()
        results["build"].append(cycle_id_text)
        results["time"].append(stop - start)
        
        if test_ds[j].get_failed_test_cases_count() != 0:
            ranking = pd.DataFrame(test_case_vector)
            ranking[['verdict', 'last_exec_time']].to_csv(conf.output_path + f"\\{cycle_id_text}.csv")
        
        pd.DataFrame(results).to_csv(os.path.join(conf.output_path, "results.csv"))
    

def reportDatasetInfo(test_case_data:list):
    cycle_cnt = 0
    failed_test_case_cnt = 0
    test_case_cnt = 0
    failed_cycle = 0
    for cycle in test_case_data:
        if cycle.get_test_cases_count() > 5:
            cycle_cnt = cycle_cnt+1
            test_case_cnt = test_case_cnt + cycle.get_test_cases_count()
            failed_test_case_cnt = failed_test_case_cnt+cycle.get_failed_test_cases_count()
            if cycle.get_failed_test_cases_count() > 0:
                failed_cycle = failed_cycle + 1
    print(f"# of cycle: {cycle_cnt}, # of test case: {test_case_cnt}, # of failed test case: {failed_test_case_cnt}, "
          f" failure rate:{failed_test_case_cnt/test_case_cnt}, # failed test cycle: {failed_cycle}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN debugger')
    old_limit = sys.getrecursionlimit()
    print("Recursion limit:" + str(old_limit))
    sys.setrecursionlimit(1000000)
    # parser.add_argument('--traningData',help='tranind data folder',required=False)
    parser.add_argument('-m', '--mode', help='[pairwise,pointwise,listwise] ', required=True)
    parser.add_argument('-a', '--algo', help='[a2c,dqn,..]', required=True)
    parser.add_argument('-d', '--dataset_type', help='simple, enriched', required=False, default="simple")
    parser.add_argument('-e', '--episodes', help='Training episodes ', required=True)
    parser.add_argument('-w', '--win_size', help='Windows size of the history', required=False)
    parser.add_argument('-t', '--train_data', help='Train set folder', required=True)
    parser.add_argument('-f', '--first_cycle', help='first cycle used for training', required=False)
    parser.add_argument('-c', '--cycle_count', help='Number of cycle used for training', required=False)
    parser.add_argument('-l', '--list_size', help='Maximum number of test case per cycle', required=False)
    parser.add_argument('-o', '--output_path', help='Output path of the agent model', required=False)


    # parser.add_argument('-f','--flags',help='Input csv file containing testing result',required=False)
    supported_formalization = ['PAIRWISE', 'POINTWISE', 'LISTWISE','LISTWISE2']
    supported_algo = ['DQN', 'PPO2', "A2C", "ACKTR", "DDPG", "ACER", "GAIL", "HER", "PPO1", "SAC", "TD3", "TRPO"]
    args = parser.parse_args()
    assert supported_formalization.count(args.mode.upper()) == 1, "The formalization mode is not set correctly"
    assert supported_algo.count(args.algo.upper()) == 1, "The formalization mode is not set correctly"

    conf = Config()
    conf.train_data = args.train_data
    conf.dataset_name = Path(args.train_data).stem
    if not args.win_size:
        conf.win_size = 10
    else:
        conf.win_size = int(args.win_size)
    if not args.first_cycle:
        conf.first_cycle = 0
    else:
        conf.first_cycle = int(args.first_cycle)
    if not args.cycle_count:
        conf.cycle_count = 9999999


    if not args.output_path:
        conf.output_path = '../experiments/' + args.mode + "/" + args.algo + "/" + conf.dataset_name + "_" \
                           + str(conf.win_size) + "/"
        conf.log_file = conf.output_path + args.mode + "_" + args.algo + "_" + \
                        conf.dataset_name + "_" + args.episodes + "_" + str(conf.win_size) + "_log.txt"
    else:
        conf.output_path = args.output_path + "/" + args.mode + "/" + args.algo + "/" + conf.dataset_name + "_" \
                           + str(conf.win_size) + "/"
        conf.log_file = conf.output_path + args.mode + "_" + args.algo + "_" + \
                        conf.dataset_name + "_" + args.episodes + "_" + str(conf.win_size) + "_log.txt"

    test_data_loader = TestCaseExecutionDataLoader(conf.train_data, args.dataset_type)
    test_data = test_data_loader.load_data()
    ci_cycle_logs = test_data_loader.pre_process()
    ### open data

    reportDatasetInfo(test_case_data=ci_cycle_logs)

    #training using n cycle staring from start cycle
    conf.dataset_type = args.dataset_type
    experiment(mode=args.mode, algo=(args.algo.upper(), {"policy": "MLP", "replay_ratio": 0}), train_ds=ci_cycle_logs, test_ds=ci_cycle_logs, episodes=int(args.episodes),
            start_cycle=conf.first_cycle, verbos=False,
            end_cycle=conf.first_cycle + conf.cycle_count - 1, model_path=conf.output_path, dataset_name="", conf=conf)
    # .. lets test this tommorow by passing args
