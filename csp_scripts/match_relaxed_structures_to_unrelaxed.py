import pathlib
import pickle
from typing import List

import imageio
import numpy as np
from ase import Atoms
from matplotlib import pyplot as plt
import matplotlib as mpl

from csp_elites.map_elites.archive import Archive
from csp_elites.utils.plot import load_archive_from_pickle


class RelaxedToUnrelaxedIndividualMatching:
    def __init__(self):
        pass

    def plot_gif_of_archive_relaxation(self,
            relaxed_archive_path: pathlib.Path,
            unrelaxed_archive_path: pathlib.Path,
            target_data_path: pathlib.Path,
            plot_with_arrows: bool,
            save_fig: bool,
            plot_references: bool,
            archive_number: int,
            experiment_tag: str,
    ):
        unrelaxed_archive = Archive.from_archive(unrelaxed_archive_path)
        relaxed_archive = Archive.from_relaxed_archive(relaxed_archive_path)

        if plot_references:
            target_archive = Archive.from_archive(target_data_path)
            plt.scatter(target_archive.descriptors[:, 0], target_archive.descriptors[:, 1], color="red",
                        label="comparison data (exp&theory)")
            for i in range(len(unrelaxed_archive.descriptors)):
                plt.annotate(i, (unrelaxed_archive.descriptors[i, 0], unrelaxed_archive.descriptors[i, 1]))

        font_size = 12
        params = {
            "axes.labelsize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "text.usetex": False,
            "figure.figsize": [12, 10],
        }

        mpl.rcParams.update(params)

        assert (self.match_confid_to_individual_index(unrelaxed_archive.individuals) == \
               self.match_confid_to_individual_index(relaxed_archive.individuals)).all()
        fig, ax = plt.subplots()
        ax.scatter(unrelaxed_archive.descriptors[:, 0], unrelaxed_archive.descriptors[:, 1], s=2, c="b", label="archive generated with QD")
        ax.scatter(relaxed_archive.descriptors[:, 0], relaxed_archive.descriptors[:, 1], s=2, c="g",
                    label="relaxed archive")

        if plot_with_arrows:
            descriptor_difference = relaxed_archive.descriptors - unrelaxed_archive.descriptors
            for i in range(len(unrelaxed_archive.descriptors[:, 0])):
                plt.arrow(unrelaxed_archive.descriptors[i, 0], unrelaxed_archive.descriptors[i, 1], descriptor_difference[i, 0],
                          descriptor_difference[i, 1], color="black", alpha=0.5)
            self.plot_relaxation_trajectory_gif(
                staring_points=unrelaxed_archive.descriptors,
                end_points=relaxed_archive.descriptors,
                experiment_folder=relaxed_archive_path.parent,
                archive_number=archive_number,
                experiment_tag=experiment_tag,
            )

        ax.set_xlabel("BD1 - Band Gap")
        ax.set_ylabel("BD2 - Shear Modulus")
        ax.set_title(
            f"Change beteween unrelaxed and relaxed archive {experiment_tag} {archive_number}")
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncols=3)
        if plot_with_arrows:
            arrows = "with_arrows"
        else:
            arrows = "no_arrows"

        if save_fig:
            plt.savefig(
                str(directory / experiment_tag / f"relaxed_vs_unrelaxed_{archive_number}_{arrows}.png"),
                format="png")
        else:
            plt.show()

    def match_confid_to_individual_index(self, individuals: List[Atoms]):
        configuration_ids = []
        for individual in individuals:
            configuration_ids.append([individual.info["confid"]])

        return np.array(configuration_ids).reshape(-1)

    def plot_relaxation_trajectory_gif(
        self,
        staring_points: np.ndarray,
        end_points: np.ndarray,
        experiment_folder: pathlib.Path,
        archive_number: int,
        experiment_tag: str,
        number_of_frames: int = 50,
    ):
        temp_dir = experiment_folder / "tempdir"
        temp_dir.mkdir(exist_ok=False)

        trajectory_x = np.linspace(staring_points[:, 0], end_points[:, 0], number_of_frames, axis=1)
        trajectory_y = np.linspace(staring_points[:, 1], end_points[:, 1], number_of_frames, axis=1)

        for i in range(number_of_frames):
            fig, ax = plt.subplots()
            ax.scatter(trajectory_x[:, i], trajectory_y[:, i], s=4, color="blue")
            ax.set_xlim(0, 100)
            ax.set_ylim(0,100)
            ax.set_xlabel("BD1 - Band Gap")
            ax.set_xlabel("BD2 - Shear Modulus")
            ax.set_title(
                f"Change beteween unrelaxed and relaxed archive {experiment_tag} {archive_number}")
            fig.savefig(str(temp_dir / f"{i}.png"), format="png")

        frames = []
        for plot_name in range(number_of_frames):
            image = imageio.v2.imread(str(temp_dir / f"{plot_name}.png"))
            frames.append(image)

        imageio.mimsave(f"{experiment_folder}/unrelaxed_to_relaxed_{archive_number}.gif",  # output gif
                        frames, duration=5)  # array of input frames)
        for plot_name in range(number_of_frames):
            image_path = temp_dir / f"{plot_name}.png"
            image_path.unlink()
        temp_dir.rmdir()
        # todo: ;add x and y lims dynamically



if __name__ == '__main__':
    experiment_tag = "20230730_05_02_TiO2_200_niches_10_relaxation_steps"
    archive_number = 25030

    directory = pathlib.Path(__file__).resolve().parent.parent  / ".experiment.nosync" / "experiments"
    archive_filename = directory / experiment_tag / f"archive_{archive_number}.pkl"
    relaxed_archive_filename  = directory / experiment_tag / f"relaxed_archive_{archive_number}.pkl"
    comparison_data = directory / "target_data" / "ti02_band_gap_shear_modulus.pkl"

    RelaxedToUnrelaxedIndividualMatching().plot_gif_of_archive_relaxation(
        relaxed_archive_path=relaxed_archive_filename,
        unrelaxed_archive_path=archive_filename,
        target_data_path=comparison_data,
        plot_with_arrows=True,
        save_fig=False,
        plot_references=False,
        archive_number=archive_number,
        experiment_tag=experiment_tag
    )
