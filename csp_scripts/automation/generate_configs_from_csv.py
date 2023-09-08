import pathlib

from csp_elites.map_elites.elites_utils import make_current_time_string
from csp_elites.utils.csv_loading import write_configs_from_csv, JobsEnum

if __name__ == '__main__':
    source_file = "automation_scripts/experiment_list.csv"
    number_of_array_jobs = 5
    subfolder_to_save = make_current_time_string(with_time=True)


    write_configs_from_csv(
        path_to_cofnig_csv=pathlib.Path(__file__).parent.parent.parent / source_file,
        number_for_array_job=number_of_array_jobs,
        path_to_save=pathlib.Path(__file__).parent.parent.parent / f"experiment_configs/{subfolder_to_save}",
        bash_template=JobsEnum.THROUGHPUT_20_MEMORY
    )
