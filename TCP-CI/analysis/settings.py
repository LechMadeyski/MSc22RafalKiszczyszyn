from enumerators import SubjectEnumerator

DATASETS = "C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\datasets"

SUBJECTS = SubjectEnumerator(
    'C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\datasets.csv', 
    DATASETS
)

SELECTED = ['S2', 'S8',  'S9', 'S12', 'S13', 'S14', 'S16', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25']

BEST_RF150 = {
    "S2": "full-outliers",
    "S8": "full-outliers",
    "S9": "full-outliers-os",
    "S12": "full-outliers-os",
    "S13": "full-outliers",
    "S14": "full-outliers-os",
    "S16": "full-outliers",
    "S20": "full-outliers-os",
    "S21": "full-outliers",
    "S22": "full-outliers",
    "S23": "full-outliers-os",
    "S24": "full-outliers",
    "S25": "full-outliers"
}

BEST_ACER = {
    "S2": "rl-diff-os-f15",
    "S8": "rl-diff-os-f15",
    "S9": "rl-f15",
    "S12": "rl-os-f15",
    "S13": "rl-os-f15",
    "S14": "rl-os-f15",
    "S16": "rl-os-f15",
    "S20": "rl-diff-f15",
    "S21": "rl-os-f15",
    "S22": "rl-f15",
    "S23": "rl-os-f15",
    "S24": "rl-diff-os-f15",
    "S25": "rl-f15"
}