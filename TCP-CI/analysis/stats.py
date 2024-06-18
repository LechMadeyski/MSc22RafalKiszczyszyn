from settings import SELECTED, SUBJECTS
from collectors import \
  RandomForestResultsCollector, \
  RandomApproachResultsCollector, \
  AcerResultsCollector, \
  SimpleHeuristicResultsCollector, \
  GenericResultsCollector, \
  TestingDurationCollector, \
  BuildsCollector
from metrics import compare, cles, wilcoxon

import random
import numpy as np
import pandas as pd

random.seed(44)
np.random.seed(44)


def rl(exp_name):
    return lambda subjects: AcerResultsCollector(subjects, exp_name)

def sl(exp_name):
    return lambda subjects: RandomForestResultsCollector(subjects, exp_name)

def rnd():
    return lambda subjects: RandomApproachResultsCollector(subjects, 'full-outliers')

def h():
    return lambda subjects: SimpleHeuristicResultsCollector(subjects, 'full-outliers', 'REC_TotalFailRate', ascending=False)

def find_best_experiment(subjects, builds, sid, candidates):
    avg_max, std_min, b = 0, 1, None
    b_name = ""
    for candidate in candidates[sid]:
        rcollector = GenericResultsCollector(
            candidate(subjects),
            builds=builds
        )
        results = rcollector.collect_into_df([sid])
        
        avg = results['APFDc (avg)'].to_list()[0]
        std = results['APFDc (std)'].to_list()[0]

        if avg > avg_max or (avg == avg_max and std <= std_min):
            avg_max = avg
            std_min = std
            b = rcollector.collected(sid)
            b_name = rcollector._collector._experiment_name

    print(f"{sid}: found best {avg_max} ± {std_min} from {b_name}")

    return b, b_name


def evaluate_experiments(subjects, builds, sid, experiments, measure="APFDc"):
    avg_max, std_min, = 0, 1
    indices_best = []
    all = []
    for index, experiment in enumerate(experiments[sid]):
        rcollector = GenericResultsCollector(
            experiment(subjects),
            builds=builds
        )
        results = rcollector.collect_into_df([sid])
        
        avg = results[f'{measure} (avg)'].to_list()[0]
        std = results[f'{measure} (std)'].to_list()[0]

        all.append((avg, std))

        if avg > avg_max or (avg == avg_max and std <= std_min):
            if avg == avg_max and std == std_min:
                indices_best.append(index)
            else:
                indices_best = [index]
                
            avg_max = avg
            std_min = std

    print(f"{sid}: found best {avg_max} ± {std_min} for indices {indices_best}")

    return all, indices_best


def stats_comparison(subjects, selected, left, right):
    builds = BuildsCollector(subjects, 'full-outliers').collect(selected)
    
    pairs = []
    for sid in selected:
        lcollector = GenericResultsCollector(
            left[sid](subjects),
            builds=builds
        )
        lcollector.collect_into_df([sid])
        l = lcollector.collected(sid)
        
        rcollector = GenericResultsCollector(
            right[sid](subjects),
            builds=builds
        )
        rcollector.collect_into_df([sid])
        r = rcollector.collected(sid)
        
        pairs.append((sid, l, r))

    comparison = compare(pairs, 'apfdc')
    print(comparison)

    for _, row in comparison.iterrows():
        sid = row['SID']
        p_value = row['p-value']
        cl = row['CL']

        sid = f"$S_{{{sid[1:]}}}$"

        if p_value < 0.01:
            p_value = "\\bm{$<0.01$}"
        elif p_value < 0.05:
            p_value = f"{p_value:.2f}"
            p_value = f"\\bm{{${p_value}$}}"
        else:
            p_value = f"{p_value:.2f}"

        cl = f"{cl:.2f}"

        print(f"{sid.ljust(10)} & {p_value.ljust(15)} & {cl.ljust(5)}")


def stats_comparison_onevsbest(subjects, selected, left, candidates):
    builds = BuildsCollector(subjects, 'full-outliers').collect(selected)
    
    pairs = []
    for sid in selected:
        lcollector = GenericResultsCollector(
            left[sid](subjects),
            builds=builds
        )
        lcollector.collect_into_df([sid])
        l = lcollector.collected(sid)
        
        r, _ = find_best_experiment(subjects, builds, sid, candidates)
        
        pairs.append((sid, l, r))

    comparison = compare(pairs, 'apfdc')
    print(comparison)

    for _, row in comparison.iterrows():
        sid = row['SID']
        p_value = row['p-value']
        cl = row['CL']

        sid = f"$S_{{{sid[1:]}}}$"

        if p_value < 0.01:
            p_value = "\\bm{$<0.01$}"
        elif p_value < 0.05:
            p_value = f"{p_value:.2f}"
            p_value = f"\\bm{{${p_value}$}}"
        else:
            p_value = f"{p_value:.2f}"

        cl = f"{cl:.2f}"

        print(f"{sid.ljust(10)} & {p_value.ljust(15)} & {cl.ljust(5)}")


def stats_comparison_bestvsbest(subjects, selected, lcandidantes, rcandidates):
    builds = BuildsCollector(subjects, 'full-outliers').collect(selected)
    
    pairs = []
    for sid in selected:
        l, _ = find_best_experiment(subjects, builds, sid, lcandidantes)
        r, _ = find_best_experiment(subjects, builds, sid, rcandidates)
        
        pairs.append((sid, l, r))

    comparison = compare(pairs, 'apfdc')
    print(comparison)

    for _, row in comparison.iterrows():
        sid = row['SID']
        p_value = row['p-value']
        cl = row['CL']

        sid = f"$S_{{{sid[1:]}}}$"

        if p_value < 0.01:
            p_value = "\\bm{$<0.01$}"
        elif p_value < 0.05:
            p_value = f"{p_value:.2f}"
            p_value = f"\\bm{{${p_value}$}}"
        else:
            p_value = f"{p_value:.2f}"

        cl = f"{cl:.2f}"

        print(f"{sid.ljust(10)} & {p_value.ljust(15)} & {cl.ljust(5)}")


def print_comparison_table(comparison):
    for sid in comparison:
        row = []
        for index, r in enumerate(comparison[sid]["all"]):
            avg, std = r
            avg = f"{avg:.2f}"
            std = f"{std:.2f}"
            
            if index in comparison[sid]["best"]:
                text = f"\\bm{{${avg} \pm {std}$}}"
            else:
                text = f"${avg} \pm {std}$"
            
            row.append(text.ljust(20))

        sid = f"$S_{{{sid[1:]}}}$"
        row = " & ".join(row)
        print(f"{sid.ljust(10)} & {row} \\\\")


def results_comparison(subjects, selected, experiments, measure='APFDc'):
    builds = BuildsCollector(subjects, 'full-outliers').collect(selected)
    
    comparison = {sid: None for sid in selected}
    for sid in selected:
        all, best = evaluate_experiments(subjects, builds, sid, experiments, measure)
        comparison[sid] = {"all": all, "best": best}

    print_comparison_table(comparison)


# stats_comparison_best(
#     SUBJECTS,
#     SELECTED,
#     {sid: sl('full-outliers') for sid in SELECTED},
#     {sid: [sl(f'full-outliers-f{k}') for k in [80, 50, 30, 15]] for sid in SELECTED}
# )

def ComparisonOnFullSet():
    print("Gathering APFDc")
    results_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: [sl('full-outliers'), rl('rl'), rnd()] for sid in SELECTED}
    )
    
    print("SL vs RL")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: sl('full-outliers') for sid in SELECTED},
        {sid: rl('rl') for sid in SELECTED}    
    )

    print("SL vs RND")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: sl('full-outliers') for sid in SELECTED},
        {sid: rnd() for sid in SELECTED}    
    )

    print("RL vs RND")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: rl('rl') for sid in SELECTED},
        {sid: rnd() for sid in SELECTED}    
    )

def FeatureSelection_RandomForest():
    experiments = [sl('full-outliers')]
    experiments.extend([sl(f'full-outliers-f{k}') for k in [80, 50, 30, 15]])

    print("Gathering APFDc")
    results_comparison(
        SUBJECTS,
        SELECTED,
        {sid: experiments for sid in SELECTED}
    )

    print("Statistically comparing results")
    stats_comparison_onevsbest(
        SUBJECTS,
        SELECTED,
        {sid: sl('full-outliers') for sid in SELECTED},
        {sid: [sl(f'full-outliers-f{k}') for k in [80, 50, 30, 15]] for sid in SELECTED}
    )

def FeatureSelection_Acer():
    baseline = rl('rl')
    fselection = [rl(f'rl-f{k}') for k in [80, 50, 30, 15]]
    
    experiments = [baseline]
    experiments.extend(fselection)
    
    print("Gathering APFDc")
    results_comparison(
        SUBJECTS,
        SELECTED,
        {sid: experiments for sid in SELECTED}
    )

    print("Statistically comparing results")
    stats_comparison_onevsbest(
        SUBJECTS, 
        SELECTED,
        {sid: baseline for sid in SELECTED},
        {sid: fselection for sid in SELECTED}
    )

    print("Statistically comparing results RND vs ACER")
    stats_comparison_onevsbest(
        SUBJECTS, 
        SELECTED,
        {sid: rnd() for sid in SELECTED},
        {sid: fselection for sid in SELECTED}
    )

    print("Statistically comparing results RF vs ACER")
    stats_comparison_onevsbest(
        SUBJECTS, 
        SELECTED,
        {sid: sl('full-outliers') for sid in SELECTED},
        {sid: fselection for sid in SELECTED}
    )

def Oversampling_ComparisonOnFullSet():
    print("Comparison of APFDc: SL vs SL (os)")
    results_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: [sl('full-outliers'), sl('full-outliers-os')] for sid in SELECTED}
    )

    print("Statistical comparison: SL vs SL (os)")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: sl('full-outliers') for sid in SELECTED},
        {sid: sl('full-outliers-os') for sid in SELECTED}
    )

    print("Comparison of APFDc: SL vs SL (os)")
    results_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: [rl('rl'), rl('rl-os')] for sid in SELECTED}
    )

    print("Statistical comparison: SL vs SL (os)")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: rl('rl') for sid in SELECTED},
        {sid: rl('rl-os') for sid in SELECTED}
    )

def Oversampling_AcerComparisonWithFeatureSelection():
    print("Comparison of APFDc: 30")
    results_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: [rl('rl-f30'), rl('rl-os-f30')] for sid in SELECTED}
    )

    print("Statistical comparison: 30")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: rl('rl-f30') for sid in SELECTED},
        {sid: rl('rl-os-f30') for sid in SELECTED},
    )

    print("Comparison of APFDc: 15")
    results_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: [rl('rl-f15'), rl('rl-os-f15')] for sid in SELECTED}
    )

    print("Statistical comparison: 15")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: rl('rl-f15') for sid in SELECTED},
        {sid: rl('rl-os-f15') for sid in SELECTED},
    )

    print("Statistical comparison: RF vs best")
    stats_comparison_onevsbest(
        SUBJECTS, 
        SELECTED, 
        {sid: sl('full-outliers') for sid in SELECTED},
        {sid: [rl('rl-os-f15'), rl('rl-os-f30')] for sid in SELECTED}
    )

def SampleEncoding():
    print("Comparison of APFDc: 15")
    results_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: [rl('rl-f15'), rl('rl-diff-f15')] for sid in SELECTED}
    )

    print("Statistical comparison: 15")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: rl('rl-f15') for sid in SELECTED},
        {sid: rl('rl-diff-f15') for sid in SELECTED},
    )

    print("Comparison of APFDc: 15 (os)")
    results_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: [rl('rl-os-f15'), rl('rl-diff-os-f15')] for sid in SELECTED}
    )

    print("Statistical comparison: 15 (os)")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        {sid: rl('rl-os-f15') for sid in SELECTED},
        {sid: rl('rl-diff-os-f15') for sid in SELECTED},
    )

def FinalComparison():
    builds = BuildsCollector(SUBJECTS, 'full-outliers').collect(SELECTED)
    
    print("Finding best SL experiment")
    sl_best = {}
    for sid in SELECTED:
        _, exp_name = find_best_experiment(
            SUBJECTS,
            builds,
            sid,
            {sid: [
                sl('full-outliers'), 
                sl('full-outliers-f15'), 
                sl('full-outliers-f30'), 
                sl('full-outliers-f50'), 
                sl('full-outliers-f80'),
                sl('full-outliers-os')
            ]}
        )
        sl_best[sid] = sl(exp_name)
    
    print("Finding best RL experiment")
    rl_best = {}
    for sid in SELECTED:
        _, exp_name = find_best_experiment(
            SUBJECTS,
            builds,
            sid,
            {sid: [
                rl('rl'), 
                rl('rl-os'),
                rl('rl-diff-f15'),
                rl('rl-diff-os-f15'),
                rl('rl-f15'),
                rl('rl-f30'),
                rl('rl-f50'),
                rl('rl-f80'),
                rl('rl-os-f15'),
                rl('rl-os-f30')
            ]}
        )
        rl_best[sid] = rl(exp_name)

    results_comparison(
        SUBJECTS,
        SELECTED,
        {sid: [sl_best[sid], rl_best[sid], rnd()] for sid in SELECTED}
    )

    print("SL vs RL")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        sl_best,
        rl_best    
    )

    print("SL vs RND")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        sl_best,
        {sid: rnd() for sid in SELECTED}    
    )

    print("RL vs RND")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        rl_best,
        {sid: rnd() for sid in SELECTED}    
    )


def Comparison_MLvsHeuristic():
    builds = BuildsCollector(SUBJECTS, 'full-outliers').collect(SELECTED)
    
    print("Finding best SL experiment (FULL)")
    rf150_best = {}
    for sid in SELECTED:
        _, exp_name = find_best_experiment(
            SUBJECTS,
            builds,
            sid,
            {sid: [
                sl('full-outliers'), 
                sl('full-outliers-os')
            ]}
        )
        rf150_best[sid] = sl(exp_name)

    rf15_best = {sid: sl('full-outliers-f15') for sid in SELECTED}
    
    print("Finding best RL15 experiment")
    rl15_best = {}
    for sid in SELECTED:
        _, exp_name = find_best_experiment(
            SUBJECTS,
            builds,
            sid,
            {sid: [
                rl('rl-diff-f15'),
                rl('rl-diff-os-f15'),
                rl('rl-f15'),
                rl('rl-os-f15')
            ]}
        )
        rl15_best[sid] = rl(exp_name)

    results_comparison(
        SUBJECTS,
        SELECTED,
        {sid: [rf150_best[sid], rf15_best[sid], rl15_best[sid], h()] for sid in SELECTED}
    )

    print("RF150 vs H")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        rf150_best,
        {sid: h() for sid in SELECTED}    
    )

    print("RF15 vs H")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        rf15_best,
        {sid: h() for sid in SELECTED}    
    )

    print("RL15 vs H")
    stats_comparison(
        SUBJECTS, 
        SELECTED, 
        rl15_best,
        {sid: h() for sid in SELECTED}    
    )


def get_fcollection_times():
    from io import StringIO
    data = """SID,COV,TES,REC,TC,TT,C/T
S2,61.4,38.2,0.5,1.1,6.0,18
S8,73.9,17.9,0.6,0.9,22.4,4
S9,74.4,11.6,0.3,0.5,20.7,3
S12,70.9,10.1,1.3,0.5,9.5,5
S13,56.6,43.3,1.1,1.2,7.5,16
S14,77.2,20.6,0.6,1.4,7.7,18
S16,63.9,9.4,0.8,0.2,6.3,4
S20,63.6,16.4,1.2,0.7,17.7,4
S21,58.4,9.0,1.9,0.2,12.3,2
S22,67.3,6.3,0.8,0.3,48.8,1
S23,64.0,7.3,1.0,0.2,68.1,<1
S24,67.0,3.3,1.8,0.2,13.7,1
S25,61.5,5.8,0.6,0.1,15.5,1"""
    
    return pd.read_csv(StringIO(data))


def evaluate(measurements):
    avg = np.average(measurements)
    std = np.std(measurements)
    min = np.min(measurements)
    max = np.max(measurements)
    return (round(avg, 1), round(std, 1), round(min, 1), round(max, 1))


def statcompare(x, y):
    _, p_value = wilcoxon(x, y)
    cles_ = cles(x, y)
    if p_value >= 0.05 or (0.45 < cles_ < 0.55):
        return 0

    return 1 if cles_ - 0.5 >= 0 else -1


def TimeReduction():
    builds = BuildsCollector(SUBJECTS, 'full-outliers').collect(SELECTED)
    data = TestingDurationCollector(SUBJECTS, 'full-outliers').collect(SELECTED)
    fcollection_times = get_fcollection_times()
    
    print("Finding best SL experiment (FULL)")
    rf150_best = {}
    for sid in SELECTED:
        _, exp_name = find_best_experiment(
            SUBJECTS,
            builds,
            sid,
            {sid: [
                sl('full-outliers'), 
                sl('full-outliers-os')
            ]}
        )
        rf150_best[sid] = sl(exp_name)

    print("Finding best RL15 experiment")
    rl15_best = {}
    for sid in SELECTED:
        _, exp_name = find_best_experiment(
            SUBJECTS,
            builds,
            sid,
            {sid: [
                rl('rl-diff-f15'),
                rl('rl-diff-os-f15'),
                rl('rl-f15'),
                rl('rl-os-f15')
            ]}
        )
        rl15_best[sid] = rl(exp_name)

    print("Collecting Heuristic results")
    heuristic = h()(SUBJECTS).collect(SELECTED)
    
    print("Collecting Random results")
    rand = rnd()(SUBJECTS).collect(SELECTED)

    print("Collecting RF FULL results")
    rf_full = {}
    for sid in rf150_best:
      rf_full[sid] = rf150_best[sid](SUBJECTS).collect([sid])[sid]
    
    print("Collecting RF 15 results")
    rf_15 = sl('full-outliers-f15')(SUBJECTS).collect(SELECTED)
    
    print("Collecting ACER results")
    acer = {}
    for sid in rl15_best:
        acer[sid] = rl15_best[sid](SUBJECTS).collect([sid])[sid]

    measurements = {
        "RF150": {sid: [] for sid in SELECTED},
        "RF150c": {sid: [] for sid in SELECTED},
        "RF15": {sid: [] for sid in SELECTED},
        "RF15c": {sid: [] for sid in SELECTED},
        "ACER": {sid: [] for sid in SELECTED},
        "ACERc": {sid: [] for sid in SELECTED},
        "H": {sid: [] for sid in SELECTED},
        "Hc": {sid: [] for sid in SELECTED},
        "RND": {sid: [] for sid in SELECTED},
    }

    results = {
        "SID": SELECTED,
        "RF150": [],
        "RF150c": [],
        "RF15": [],
        "RF15c": [],
        "ACER": [],
        "ACERc": [],
        "H": [],
        "Hc": [],
        "RND": [],
    }

    comparisons = {    
        "H vs RND": [],
        "Hc vs RND": [],

        "RF150 vs RND": [],
        "RF150c vs RND": [],
        "RF150 vs H": [],
        "RF150c vs Hc": [],
        
        "RF15 vs RND": [],
        "RF15c vs RND": [],
        "RF15 vs H": [],
        "RF15c vs Hc": [],
        
        "ACER vs RND": [],
        "ACERc vs RND": [],
        "ACER vs H": [],
        "ACERc vs Hc": [],
    }

    for sid in SELECTED:    
        fcollection = fcollection_times[fcollection_times['SID'] == sid]
        fcollection_total = fcollection["TC"].values[0] * 60 * 1000 
        
        print(sid, "Saving time measurements")
        for build in data[sid]["f_builds"]:
            ratio = data[sid]["ts_size"][build] / data[sid]["avg_ts_size"]

            c_rec = fcollection["REC"].values[0] / 100 * fcollection_total * ratio
            c_tes = fcollection["TES"].values[0] / 100 * fcollection_total * ratio
            c_cov = fcollection["COV"].values[0] / 100 * fcollection_total * ratio
            c_t = c_rec + c_tes + c_cov
            
            total = data[sid]["durations"][build]       

            # Random
            ttf = np.average(rand[sid][build]["ttf"]) 
            reduction = 100 - (ttf / total * 100)
            measurements["RND"][sid].append(reduction)

            # RF FULL
            ttf = rf_full[sid][build]["ttf"]
            reduction = 100 - (ttf / total * 100)
            measurements["RF150"][sid].append(reduction)

            reduction = 100 - ((ttf + c_t) / total * 100)
            measurements["RF150c"][sid].append(reduction)

            # RF 15
            ttf = rf_15[sid][build]["ttf"]
            reduction = 100 - (ttf / total * 100)
            measurements["RF15"][sid].append(reduction)

            reduction = 100 - ((ttf + c_rec + c_tes) / total * 100)
            measurements["RF15c"][sid].append(reduction)

            # ACER
            ttf = acer[sid][build]["ttf"]
            reduction = 100 - (ttf / total * 100)
            measurements["ACER"][sid].append(reduction)

            reduction = 100 - ((ttf + c_rec + c_tes) / total * 100)
            measurements["ACERc"][sid].append(reduction)

            # Heuristic
            ttf = np.average(heuristic[sid][build]["ttf"]) 
            
            reduction = 100 - (ttf / total * 100)
            measurements["H"][sid].append(reduction)

            reduction = 100 - ((ttf + c_rec) / total * 100)
            measurements["Hc"][sid].append(reduction)

        print(sid, "Evaluating measurements")
        for n in comparisons:
            x = n.split(" vs ")
            comparisons[n].append(statcompare(measurements[x[0]][sid], measurements[x[1]][sid]))

        for n in measurements:
            results[n].append(evaluate(measurements[n][sid]))

    oogies = {m: [] for m in measurements}

    def cell(row, column):
        avg, std, min, _ = row[column]
        if min == 0:
            oogies[column].append(row["SID"])
        
        just = ""
        if std < 10:
            just = "\\,\\,\\,"
        
        return f"${round(avg, 1)} \pm {just} {round(std, 1)}$"

    df = pd.DataFrame(results)
    for _, row in df.iterrows():
        sid = row["SID"]
        print(f"$S_{{{sid[1:]}}}$", cell(row, "RF150"), cell(row, "RF15"), cell(row, "ACER"), cell(row, "H"), cell(row, "RND") + " \\\\", sep=" & ")

    print(oogies)
        
    df = pd.DataFrame(results)
    for _, row in df.iterrows():
        sid = row["SID"]
        print(f"$S_{{{sid[1:]}}}$", cell(row, "RF150c"), cell(row, "RF15c"), cell(row, "ACERc"), cell(row, "Hc"), cell(row, "RND") + " \\\\", sep=" & ")

    print(" & " + " & ".join(map(lambda x: f"$S_{{{x[1:]}}}$", SELECTED)))
    for n in comparisons:
      positive = "\\textcolor{darkgreen}{1}"
      negative = "\\textcolor{darkred}{-1}"
      def t(x):
        if x == 0:
            return "0"
        return positive if x > 0 else negative    
      print(f"{n} & {' & '.join(map(t, comparisons[n]))} \\\\")


if __name__ == '__main__':
    # ComparisonOnFullSet()
    # FeatureSelection_RandomForest()
    # FeatureSelection_Acer()
    # Oversampling_ComparisonOnFullSet()
    # Oversampling_AcerComparisonWithFeatureSelection()
    SampleEncoding()
    # FinalComparison()
    # Comparison_MLvsHeuristic()
    # TimeReduction()
