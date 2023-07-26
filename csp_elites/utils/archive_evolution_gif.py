import os

import imageio


def plot_gif(experiment_directory_path: str):

    plot_list = [name for name in os.listdir(f"{experiment_directory_path}") if
                     not os.path.isdir(name) and "cvt_plot_" in name and ".png" in name]
    sorted_plot_list = sorted(plot_list, key=lambda x:int(x.lstrip("cvt_plot_").rstrip(".png")))
    #
    # png_list = []
    # for plot_name in sorted_plot_list:
    #     plot_number = plot_name.lstrip("cvt_plot_").rstrip(".png")
    #     cairosvg.svg2pdf(url=f"{experiment_directory_path}/{plot_name}", write_to=f"{experiment_directory_path}/{plot_number}.png")
    #     png_list.append(f"{plot_number}.png")

    print()
    frames = []
    for plot_name in sorted_plot_list:
        image = imageio.v2.imread(f"{experiment_directory_path}/{plot_name}")
        frames.append(image)

    imageio.mimsave(f"{experiment_directory_path}/cvt_plot_gif.gif",  # output gif
                    frames,)  # array of input frames)



if __name__ == '__main__':
    directory_path = "../../experiments/20230722_10_09_TiO2_100k_1000_niches"
    plot_gif(directory_path)
