### 1. Understand software licence
Academic licence must be requrested. 

### 2. Python version
Python 3.12 (latest at the time) did not work. Tried python 3.7 which did not work either, although this version is claimed to work. It may be Windows related issue. Ultimately, version 3.9 is used. 

### 3. Conflicts in requirements
When installing requirements first time, following error occurred:
```
ERROR: typer 0.3.2 has requirement click<7.2.0,>=7.1.1, but you'll have click 8.1.7 which is incompatible.
```
File `requirements.txt` had to be manually modified (refer to commit 314733c8).

### 4. No module named `numpy.distutils._msvccompiler` in numpy.distutils
VS2022 Build Tools must be installed on machine.

### 5. Module understand does not exist
Add environment variable `SCITOOLS_HOME` holding path to the directory with binaries (it should be `C:\Program Files\Scitools\bin\pc-win64`). 

Add line `os.add_dll_directory(os.environ["SCITOOLS_HOME"])` in the main.py (refer to commit d922dd99).

Add line `$env:PYTHONPATH = "$env:SCITOOLS_HOME\Python"` in `.venv/Scripts/Active.ps1`

### 6. Can't find model 'en_core_web_sm' (spacy)
Run `python -m spacy download en_core_web_sm` inside project environment.

### 7. Could not find or load main class ciir.umass.edu.features.FeatureManager
Change classpath delimiter in `ranklib_learner.py` (refer to commit a51e5929).
