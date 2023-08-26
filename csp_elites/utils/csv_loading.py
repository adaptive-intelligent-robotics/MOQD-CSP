import json
import pathlib

import pandas as pd

from csp_elites.utils.experiment_parameters import ExperimentParameters


def write_bash_script(config_name, job_name):
    with open(pathlib.Path(__file__).parent.parent.parent / "experiments/hpc_template.pbs", "r") as file:
        text = file.readlines()

    with open(pathlib.Path(__file__).parent.parent.parent / f"experiments/{job_name}.pbs", "w") as file:
        for line in text:

            if 'REPLACE_ME' in line:
                line = line.replace('REPLACE_ME', config_name)

            file.write(line)

if __name__ == '__main__':
    write_bash_script("0826/test.json", "test_job")

    # df = pd.read_csv(pathlib.Path(__file__).parent.parent.parent / "experiments/experiment_list.csv")
    # run_params_start_col = df.columns.get_loc("cvt_samples")
    # number_of_columns = df.shape[1]
    # run_params = df.iloc[:, run_params_start_col:number_of_columns]
    # df["cvt_run_parameters"] = run_params.to_dict(orient='records')
    # df.drop(df.iloc[:, run_params_start_col:number_of_columns], inplace=True, axis=1)
    #
    #
    # config_names = pd.Series.to_frame(df.loc[:, "config_name"])
    # config_names = config_names.to_dict(orient="list")
    # config_names = config_names["config_name"]
    #
    # df.drop(df.iloc[:, 0:2], inplace=True, axis=1)
    #
    # list_of_experiment_params_as_dict = df.to_dict(orient="records")
    #
    #
    # for i, el in enumerate(list_of_experiment_params_as_dict):
    #     exp_params = ExperimentParameters(**el)
    #     with open(pathlib.Path(__file__).parent.parent.parent / f"experiments/{config_names[i]}.json", "w") as file:
    #         json.dump(el, file)
    #
    #     with open(pathlib.Path(__file__).parent.parent.parent / f"experiments/{config_names[i]}.json", "r") as file:
    #         exp_params_2 = json.load(file)
    #         exp_params_2 = ExperimentParameters(**exp_params_2)
    #
    #     print(config_names[i])
    #     print(exp_params_2)
    print()
