import os
import pathlib

import imageio


def plot_gif(experiment_directory_path: str):

    plot_list = [name for name in os.listdir(f"{experiment_directory_path}") if
                     not os.path.isdir(name) and "cvt_plot_" in name and ".png" in name]
    sorted_plot_list = sorted(plot_list, key=lambda x:int(x.lstrip("cvt_plot_").rstrip(".png")))

    print()
    frames = []
    for plot_name in sorted_plot_list:
        image = imageio.v2.imread(f"{experiment_directory_path}/{plot_name}")
        frames.append(image)

    imageio.mimsave(f"{experiment_directory_path}/cvt_plot_gif.gif",  # output gif
                    frames,)  # array of input frames)



if __name__ == '__main__':
    experiment_dicrectory = pathlib.Path(__file__). resolve(). parent. parent. parent / "experiment" / "experiments"
    experiment_label = "20230727_03_43_TiO2_test"
    directory_path = experiment_dicrectory / experiment_label
    plot_gif(str(directory_path))
