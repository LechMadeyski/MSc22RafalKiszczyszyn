from enumerators import SubjectEnumerator

DATASETS = "C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\datasets"

SUBJECTS = SubjectEnumerator(
    'C:\\Users\\rafal\\MT\\repos\\MSc22RafalKiszczyszyn\\TCP-CI\\analysis\\datasets.csv', 
    DATASETS
)

SELECTED = ['S2', 'S8',  'S9', 'S12', 'S13', 'S14', 'S16', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25']