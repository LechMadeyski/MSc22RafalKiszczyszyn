import pandas as pd


class SubjectEnumerator:

    def __init__(self, path, workdir):
        self._subjects: pd.DataFrame = pd.read_csv(path)
        self._workdir = workdir

    def enumerate(self, filter=None, process=None):
        for index, row in self._subjects.iterrows():
            if filter is not None and row['SID'] not in filter:
                continue
            
            if process is not None:
                ds_path = self.get_ds_path(row['Subject'])
                process(row['SID'], ds_path)
        pass  

    def get_ds_path(self, subject_name: str):
        ds_name = subject_name.replace("/", "@")
        return self._workdir + f"\\{ds_name}"
