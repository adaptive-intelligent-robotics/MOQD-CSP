# QD4CSP: Quality Diversity for Crystal Structure Prediction

`QD4CSP` is the first of its kind implementation combining the strengths of Quality Diversity algorithms
for inorganic crystal structure prediction. 

This project is the result of the _MSc Thesis_ project completed as part of the _MSc Artificial Intelligence_.
It was supervised by Dr Antoine Cully, with expert materials science input provided by Professor Aron Walsh, 
Chair of Materials Design at the Department of Materials at Imperial College London. 


### Getting started with the package
To get started with this package clone this repo:

```bash
git clone https://gitlab.doc.ic.ac.uk/AIRL/students_projects/2022-2023/marta_wolinska/csp-elites.git
```
Then enter the correct directory on your machine:
```bash
cd csp-elites
```

We provide two installation methods, one using poetry (preferred) and using standard `requirements.txt`.

#### Poetry
This package uses [poetry](https://python-poetry.org) dependency manager. 
To install all dependencies run:
```bash
poetry install
```

#### Python virtual environments
Once you have cloned the repository, create and activate your virtual environment:
```shell
python3 -m venv ./venv
source venv/bin/activate
```
Then install the requirements:
```shell script
pip3 install -r requirements.txt
```
### Using the package
To run a demo experiment run:
```shell
 python3 csp_scripts/experiment_from_config.py experiment_configs/demo.json
```
This will run a very simple demo with 2 TiO2 structures. All results will be saved under the `experiments_folder`.

Experiments are most conveniently defined using a configuration file. These files can be generated individually or in batches using the directions below 

## Generating Configuration Files
To generate a configuration file for your experiment simply run 

```shell
python3 csp_scripts/generate_pre_filled_config.py <config_filename> <config_folder>
```
Passing the `<config_folder>` parameter will create a subfolder within `experiment_configs`.
If it is not passed the config file will save directly within `experiment_configs`.

This will be filled with some default values and the resulting json should then be updated directly.

### Mass generating configs
However, if you are running many experiments the above is not suitable. 
Therefore, you can generate multiple config files from a csv file. 
This method will also generate hpc scripts required to run batch jobs

You can generate multiple config files at a time for an array job on the hpc by running 
```shell
python3 csp_scripts/automation/generate_configs_from_csv.py
```

By default, this will create the framework for an array job with 5 experiments.
It will read from `automation_scripts/experiment_list.csv` to produce 5 configuration files
saved within the folder with the dat and time.
It will also create a folder with the same and `_scripts` which will have the required jobs scripts for the hpc. 
The templates are stored within `automation_scripts/hpc/job_templates`

To add a new template, create a new file in this repository. 
Then add a new Enum to `JobsEnum` class within `csp_elites/utils/csv_loading.py`

To copy the desired_folders to the home directory of the hpc amend the `automation_scripts/hpc/copy_configs_and_scripts.sh`
with the desired folders. 
Then run
```shell
bash automation_scripts/hpc/copy_configs_and_scripts.sh
```

## Running an Experiment 
You can run an experiment from a configuration file or directly from a file. 
The latter is recommended for debugging new features. 

### Running from a configuration file
To run your job simply run

```shell
python3 csp_scripts/experiment_from_config.py configs/<your-config-name>.json
```

Or if you prefer to change parameters directly in a python script you can amend them in `csp_scripts/`
### Running Feature Debugging Script

```shell
python3 csp_scripts/run_experiment.py  
```


### Configuring cython to use fast algorithm conversion in \texttt{CHGNet}
As guidelines were not available on the package at the time of writing we provide our method to ensure
cython is set up correctly to be used with \texttt{CHGNet}


First clone the CHGNet repository and enter the folder
```shell
git clone https://github.com/CederGroupHub/chgnet.git
cd chgnet
```
Then run:
```shell
python3 setup.py build_ext --inplace
```
Now we will need to copy the generated filed into our virtual environment 

```shell
cd chgnet/graph
copy *.c venv/lib/chgnet/graph
copy *.pyx venv/lib/chgnet/graph
```

You can verify this by running a script containing the following:
```python
from chgnet.model import CHGNet

if __name__ == '__main__':
    model = CHGNet.load()
    print(model.graph_converter.algorithm)
```
