import pathlib

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from csp_elites.map_elites.archive import Archive

if __name__ == '__main__':
    experiment_labels = [
        "20230813_01_48_TiO2_200_niches_for benchmark_100_relax_2"
    ]
    centroid_filepaths = [
        "centroids_200_2_band_gap_0_100_shear_modulus_0_120.dat",
    ]
    archive_number = 5079
    experiment_directory_path = pathlib.Path(__file__).parent.parent / ".experiment.nosync" / "experiments" /experiment_labels[0]
    centroid_full_path = pathlib.Path(__file__).parent.parent / ".experiment.nosync" / "experiments" / "centroids" / centroid_filepaths[0]

    unrelaxed_archive_location = experiment_directory_path / f"archive_{archive_number}.pkl"

    archive = Archive.from_archive(unrelaxed_archive_location, centroid_filepath=centroid_full_path)

    structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in archive.individuals]

    groups = StructureMatcher().group_structures(structures)
    ids_by_group = []
    for group in groups:
        id_in_group = []
        for el in group:
            match = [structures[i] == el for i in range(len(structures))]
            match_id = np.argwhere(np.array(match)).reshape(-1)
            id_in_group.append(archive.centroid_ids[match_id[0]])
        ids_by_group.append(id_in_group)

    print()
