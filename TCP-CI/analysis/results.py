import pandas as pd
from settings import SELECTED, SUBJECTS
from collectors import \
  RandomForestResultsCollector, \
  RandomApproachResultsCollector, \
  AcerResultsCollector, \
  GenericResultsCollector, \
  BuildsCollector
from metrics import compare
from typing import Dict

import random
import numpy as np

random.seed(44)
np.random.seed(44)

def join(df: pd.DataFrame, origin: pd.DataFrame, ds_name, measure):
    df[f'{ds_name} {measure}'] = origin[f'{measure} (avg)'].astype(str) + ' ± ' + origin[f'{measure} (std)'].astype(str)


def get_comparison_pairs(x: GenericResultsCollector, y: GenericResultsCollector, selected):    
    pairs = []
    for sid in selected:
        pairs.append((sid, x.collected(sid), y.collected(sid)))
    
    return pairs


def comparison(subjects, selected):
    sl_exp_name = 'full-outliers'
    rl_exp_name = 'rl'
    overview = pd.DataFrame({'SID': selected})
    
    builds = BuildsCollector(subjects, sl_exp_name).collect(selected)

    rf_c = GenericResultsCollector(
        RandomForestResultsCollector(subjects, sl_exp_name),
        builds=builds
    )
    rf = rf_c.collect_into_df(selected)

    join(overview, rf, 'RF', 'APFDc')

    acer_c = GenericResultsCollector(
        AcerResultsCollector(subjects, rl_exp_name),
        builds=builds
    )
    acer = acer_c.collect_into_df(selected)

    join(overview, acer, 'ACER-PA', 'APFDc')

    rnd_c = GenericResultsCollector(
        RandomApproachResultsCollector(subjects, 'full-outliers')
    )
    rnd = rnd_c.collect_into_df(selected)

    join(overview, rnd, 'RND', 'APFDc')

    print(overview)

    pairs = get_comparison_pairs(rf_c, rnd_c, selected)
    comparison = compare(pairs, 'apfdc')
    
    comparison["RND"] = overview["RND APFDc"]
    comparison["RF"] = overview["RF APFDc"]

    print(comparison[["RF", "RND", "p-value", "CL"]])

    pairs = get_comparison_pairs(acer_c, rnd_c, selected)
    comparison = compare(pairs, 'apfdc')
    
    comparison["RND"] = overview["RND APFDc"]
    comparison["ACER-PA"] = overview["ACER-PA APFDc"]

    print(comparison[["ACER-PA", "RND", "p-value", "CL"]])


def comparison_w_oversampling(subjects, selected):
    sl_exp_name = 'full-outliers'
    rl_exp_name = 'rl'
    overview = pd.DataFrame({'SID': selected})
    
    builds = BuildsCollector(subjects, sl_exp_name).collect(selected)

    rf_c = GenericResultsCollector(
        RandomForestResultsCollector(subjects, sl_exp_name),
        builds=builds
    )
    rf = rf_c.collect_into_df(selected)

    join(overview, rf, 'RF', 'APFDc')

    rf_os_c = GenericResultsCollector(
        RandomForestResultsCollector(subjects, sl_exp_name + '-os2'),
        builds=builds
    )
    rf_os = rf_os_c.collect_into_df(selected)

    join(overview, rf_os, 'RF (os)', 'APFDc')

    acer_c = GenericResultsCollector(
        AcerResultsCollector(subjects, rl_exp_name),
        builds=builds
    )
    acer = acer_c.collect_into_df(selected)

    join(overview, acer, 'ACER-PA', 'APFDc')

    acer_os_c = GenericResultsCollector(
        AcerResultsCollector(subjects, rl_exp_name + '-os'),
        builds=builds
    )
    acer_os = acer_os_c.collect_into_df(selected)

    join(overview, acer_os, 'ACER-PA (os)', 'APFDc')

    rnd_c = GenericResultsCollector(
        RandomApproachResultsCollector(subjects, 'full-outliers')
    )
    rnd = rnd_c.collect_into_df(selected)

    join(overview, rnd, 'RND', 'APFDc')

    print(overview)

    # pairs = get_comparison_pairs(acer_os_c, rnd_c, selected)
    # comparison = compare(pairs, 'apfdc')
    
    # comparison["RND"] = overview["RND APFDc"]
    # comparison["ACER-PA"] = overview["ACER-PA (os) APFDc"]

    # print(comparison[["ACER-PA", "RND", "p-value", "CL"]])


def rl_fselection(subjects, selected, K, exp_name='rl'):
    overview = pd.DataFrame({'SID': selected})
    
    builds = BuildsCollector(subjects, 'full-outliers').collect(selected)

    for k in K:
        name = f"{exp_name}-f{k}" if k < 150 else exp_name
        acer_c = GenericResultsCollector(
            AcerResultsCollector(subjects, name),
            builds=builds
        )
        acer = acer_c.collect_into_df(selected)
        
        join(overview, acer, k, 'APFDc')

    print(overview)
    return overview


def sl_fselection(subjects, selected, K):
    overview = pd.DataFrame({'SID': selected})
    
    for k in K:
        name = f"full-outliers-f{k}" if k < 150 else 'full-outliers'
        acer_c = GenericResultsCollector(
            RandomForestResultsCollector(subjects, name),
        )
        acer = acer_c.collect_into_df(selected)
        
        join(overview, acer, k, 'APFDc')

    print(overview)
    return overview


def rl_fselection_w_oversampling(subjects, selected):
    overview = pd.DataFrame({'SID': selected})
    fselection = rl_fselection(subjects, selected, [30, 15])
    fselection_os = rl_fselection(subjects, selected, [30, 15], exp_name='rl-os')   

    overview['30'] = fselection['30 APFDc']
    overview['30 (os)'] = fselection_os['30 APFDc']
    overview['15'] = fselection['15 APFDc']
    overview['15 (os)'] = fselection_os['15 APFDc']
    print(overview)


def rl_environment(subjects, selected):
    builds = BuildsCollector(subjects, 'full-outliers').collect(list(selected))
    overview = pd.DataFrame({'SID': selected})

    acer_c = GenericResultsCollector(
        AcerResultsCollector(subjects, "rl-os-f15"),
        builds=builds
    )
    acer = acer_c.collect_into_df(selected)

    join(overview, acer, "Pair", 'APFDc')

    acer_c = GenericResultsCollector(
        AcerResultsCollector(subjects, "rl-diff-os-f15"),
        builds=builds
    )
    acer = acer_c.collect_into_df(selected)

    join(overview, acer, "Diff", 'APFDc')

    print(overview)


# comparison_wo_fselection(SUBJECTS, SELECTED)
# comparison_w_oversampling(SUBJECTS, SELECTED)
# rl_fselection(SUBJECTS, SELECTED, [30, 15])
# sl_fselection(SUBJECTS, SELECTED,  [150, 80, 50, 30, 15])
# rl_fselection_w_oversampling(SUBJECTS, SELECTED)
rl_environment(SUBJECTS, SELECTED)
