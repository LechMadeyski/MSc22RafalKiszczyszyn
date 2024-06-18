### Datasets
Datasets are available on [Zenodo](https://zenodo.org/records/6415365). Download the file `TCP-CI-main-dataset.tar.gz`, and place the uncompressed datasets folder under `./TCP-CI/datasets`.

### Environment 
Experiments have been run with Python 3.7. Other versions are not guaranteed to work. Create a virtual environment using `python -m venv .env`, activate it, and install the requirements with `python -m pip install -r ./TCP-CI/requirements.txt`.

### Experiments
You can run an experiment using `python ./TCP-CI/main.py RQ`, where `RQ` corresponds to the Research Question to reproduce (e.g., RQ1.1, RQ1.2, RQ1.3, RQ1.4, RQ1.5, RQ2.1, RQ2.2). The script will execute all respective experiments and run an analysis on the results, which will be printed on the console.
