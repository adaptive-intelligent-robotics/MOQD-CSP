import json
import pathlib
from enum import Enum
from math import isnan

import pandas as pd

class JobsEnum(Enum):
    MEDIUM = "hpc_template_medium.pbs"
    THROUGHPUT_30_MEMORY = "hpc_template.pbs"
    GPU = "hpc_template_gpu.pbs"
    THROUGHPUT_20_MEMORY = "hpc_template_20gb.pbs"

def write_bash_script(
        config_name,
        job_name,
        n_jobs_in_array: int,
        path_to_save: pathlib.Path,
        template_name: JobsEnum
):
    with open(pathlib.Path(__file__).parent.parent.parent / f"automation_scripts/hpc/job_templates/{template_name.value}", "r") as file:
        text = file.readlines()

    with open(path_to_save / f"{job_name}.pbs", "w") as file:
        for line in text:
            if 'CONFIG_NAME' in line:
                line = line.replace('CONFIG_NAME', config_name + "_${PBS_ARRAY_INDEX}" + ".json")
            elif "N_JOBS" in line:
                line = line.replace('N_JOBS', str(n_jobs_in_array))
            file.write(line)


def write_configs_from_csv(path_to_cofnig_csv: pathlib.Path, number_for_array_job: int, path_to_save: pathlib.Path,
                           bash_template: JobsEnum):
    df = pd.read_csv(path_to_cofnig_csv)
    run_params_start_col = df.columns.get_loc("cvt_samples")
    number_of_columns = df.shape[1]
    run_params = df.iloc[:, run_params_start_col:number_of_columns]
    df["cvt_run_parameters"] = run_params.to_dict(orient='records')
    df.drop(df.iloc[:, run_params_start_col:number_of_columns], inplace=True, axis=1)

    config_names = pd.Series.to_frame(df.loc[:, "config_name"])
    config_names = config_names.to_dict(orient="list")
    config_names = config_names["config_name"]

    df.drop(df.iloc[:, 0:2], inplace=True, axis=1)

    list_of_experiment_params_as_dict = df.to_dict(orient="records")
    path_to_save.mkdir(exist_ok=True)
    for i, el in enumerate(list_of_experiment_params_as_dict):
        if isinstance(el["blocks"], str):
            el["blocks"] = eval(el["blocks"])
        if isinstance(el["operator_probabilities"], str):
            el["operator_probabilities"] = eval(el["operator_probabilities"])
        elif isinstance(el["operator_probabilities"], float):
            if isnan(el["operator_probabilities"]):
                el["operator_probabilities"] = "none"

        el["fitness_min_max_values"] = eval(el["fitness_min_max_values"])

        new_cvt_parameters = {}
        for k, v in el["cvt_run_parameters"].items():
            if isinstance(v, float):
                if isnan(v):
                    continue
                else:
                    new_cvt_parameters[k] = v
            elif isinstance(v, int) or isinstance(v, bool):
                new_cvt_parameters[k] = v
            else:
                new_cvt_parameters[k] = eval(v)

        el["cvt_run_parameters"] = new_cvt_parameters

        for j in range(1, number_for_array_job + 1):
            el["experiment_tag"] = \
            el["experiment_tag"] + f"_{j}"
            with open(path_to_save / f"{config_names[i]}_{j}.json", "w") as file:
                json.dump(el, file)

        scripts_directory = path_to_save.name + "_scripts"
        (path_to_save.parent / scripts_directory).mkdir(exist_ok=True)
        write_bash_script(
            config_name=f"{path_to_save.name}/{config_names[i]}",
            job_name=config_names[i],
            path_to_save=path_to_save.parent / scripts_directory,
            n_jobs_in_array=number_for_array_job,
            template_name=bash_template
        )


if __name__ == '__main__':
    write_configs_from_csv(
        path_to_cofnig_csv=pathlib.Path(__file__).parent.parent.parent / "experiments/experiment_list.csv",
        number_for_array_job=5,
        path_to_save=pathlib.Path(__file__).parent.parent.parent / "configs/0903",
        bash_template=JobsEnum.THROUGHPUT_20_MEMORY
    )
