import json
import os
import pathlib
from json import JSONDecodeError

import pandas as pd

if __name__ == "__main__":
    # path_to_config = pathlib.Path(__file__).parent.parent.parent / ".experiment.nosync" / "experiments" / "20230727_03_43_TiO2_test" / "config.json"
    # with open(path_to_config, "r") as file:
    #     data = json.load(file)

    all_paths_to_configs = [
        pathlib.Path(__file__).parent.parent.parent / "configs",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0808",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0809",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0812",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0813",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0814",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0815",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0820",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0821",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0822",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0823",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0824",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0826",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0827",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0828",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0830",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0831",
        pathlib.Path(__file__).parent.parent.parent / "configs" / "0903",
    ]

    all_configs = []
    for path in all_paths_to_configs:
        new_configs = [
            os.path.join(path, o)
            for o in os.listdir(path)
            if os.path.isfile(os.path.join(path, o))
        ]
        all_configs += new_configs

    config_names = [
        config.lstrip(
            "/Users/marta/Documents/MSc Artificial Intelligence/Thesis/csp-elites/configs/"
        ).rstrip(".json")
        for config in all_configs
    ]

    data = {}

    for i, config in enumerate(all_configs):
        try:
            with open(config, "r") as file:
                config_data = json.load(file)
            data[config_names[i]] = config_data
        except (UnicodeDecodeError, JSONDecodeError) as e:
            print(config)
            continue

    df = pd.DataFrame(data)
    df = df.transpose()
    df = pd.concat([df, df["cvt_run_parameters"].apply(pd.Series)], axis=1)
    df.drop(columns="cvt_run_parameters", inplace=True)
    df.to_csv(
        pathlib.Path(__file__).parent.parent.parent / "configs" / "list_of_configs.csv"
    )
    print()
