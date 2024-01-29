# MOQD-CSP: Multi-Objective Quality-Diversity for Crystal Structure Prediction

`MOQD-CSP` contains the code for the [Multi-Objective Quality-Diversity for Crystal Structure Prediction]() paper. This builds on top of the [QD4CSP repo]() in order to apply Quality-Diversity Algorithms to Crystal Structure Prediction. This method (illustrated below) uses domain-specific mutations and surrogate models for evaluation in order to generate a diverse collection of crystal structures that achieve different trade-offs on objectives:

![method](https://github.com/adaptive-intelligent-robotics/MOQD-CSP/assets/49594227/52c63b87-9525-451d-8d41-689d79006dda)


## Installation

To run this code, you need to install the necessary libraries as listed in `requirements.txt` via:

```bash
pip install -r requirements.txt
```

However, we recommend using a containerised environment such as Docker, Singularity or conda to use the repository. Further details are provided in the last section. 

## Basic API Usage

To run the MOME-X algorithm, or any other baseline algorithm mentioned in the paper, you just need to run the `main.py` script and specify the algorithm and system you wish to run. For example, to run MOME-X on Carbon, you can run:

```bash
python3 main.py -—algo=mome-x –system=C
```

The hyperparameters of the algorithms can be modified by changing their values in the `configs` directory of the repository. Alternatively, they can be modified directly in the command line. For example, to decrease the `pareto_front_max_length` parameter from 50 to 20 in MOME-P2C, you can run:

```bash
python3 main.py --algo=mome-x pareto_front_max_length=20
```


## Analysis 

Running each algorithm automatically saves metrics, visualisations and plots of performance into a `results` directory. However, you can compare performance between algorithms once they have been run using the `analysis.py` script. To do this, you need to edit the list of the algorithms and environments you wish to compare and the metrics you wish to compute (at the bottom of `analysis.py`). Then, the relevant plots and performance metrics will be computed by running:

```bash
python3 analysis.py
```

Similarly, you can run:

```bash
python3 plot_moqd_illumination.py
```

Or:
```bash
python3 plot_reference_illumination.py
```

To generate additional visualisation plots of the final results.


## Generating Reference Data
By default, this repo contains the data for *Carbon, Silicon, Silicon Carbide, Silicon Dioxide* and *Titanium Dioxide*. However, you can generate reference data for other systems by modifying the `reference_data/prepare_reference_data.py` scrupt and then running it.


## Singularity Usage

To build a final container (an executable file) using Singularity make sure you are in the root of the repository and then run:

```bash
singularity build --fakeroot --force singularity/[FINAL CONTAINER NAME].sif singularity/singularity.def
```

where you can replace '[FINAL CONTAINER NAME]' by your desired file name. When you get the final image, you can execute it via:

```bash
singularity -d run --cleanenv --containall --no-home --nv [FINAL CONTAINER NAME].sif [EXTRA ARGUMENTS]
```

where 
- [FINAL CONTAINER NAME].sif is the final image built
- [EXTRA ARGUMENTS] is a list of any futher arguments that you want to add. For example, you may want to change the random seed or Brax environment.
