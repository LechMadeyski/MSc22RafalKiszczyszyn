import pandas as pd
from settings import SELECTED, SUBJECTS
from collectors import \
  RandomForestResultsCollector, \
  RandomApproachResultsCollector, \
  AcerResultsCollector, \
  GenericResultsCollector, \
  BuildsCollector
from metrics import compare


def join(df: pd.DataFrame, origin: pd.DataFrame, ds_name, measure):
    df[f'{ds_name} {measure}'] = origin[f'{measure} (avg)'].astype(str) + ' ± ' + origin[f'{measure} (std)'].astype(str)


def get_comparison_pairs(x: GenericResultsCollector, y: GenericResultsCollector, selected):    
    pairs = []
    for sid in selected:
        pairs.append((sid, x.collected(sid), y.collected(sid)))
    
    return pairs


def comparison_wo_fselection(subjects, selected):
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


def comparison_w_fselection(subjects, selected):
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
        RandomForestResultsCollector(subjects, sl_exp_name + '-os'),
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

    pairs = get_comparison_pairs(acer_os_c, rnd_c, selected)
    comparison = compare(pairs, 'apfdc')
    
    comparison["RND"] = overview["RND APFDc"]
    comparison["ACER-PA"] = overview["ACER-PA (os) APFDc"]

    print(comparison[["ACER-PA", "RND", "p-value", "CL"]])


def experiment2(subjects, selected, baseline_exp_name, K):
    overview = pd.DataFrame({'SID': selected})
    
    builds = BuildsCollector(subjects, baseline_exp_name).collect(selected)

    for k in K:
        name = f"rl-f{k}" if k < 150 else "rl"
        acer_c = GenericResultsCollector(
            AcerResultsCollector(subjects, name),
            builds=builds
        )
        acer = acer_c.collect_into_df(selected)
        
        join(overview, acer, k, 'APFDc')

    print(overview)


# experiment_0(SUBJECTS, [f'S{i+1}' for i in range(25)])
# rl_experiment(SUBJECTS, ['S20'], 'full-outliers', 'rl')
# rl_experiment(SUBJECTS, ['S20'], 'full-outliers', 'rl-f20')
# experiment1(SUBJECTS, SELECTED, 'full-outliers', 'rl')
# experiment2(SUBJECTS, SELECTED, 'full-outliers', [150, 80, 50, 30, 15])

comparison_wo_fselection(SUBJECTS, SELECTED)
# comparison_w_fselection(SUBJECTS, SELECTED)
