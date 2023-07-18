#!/bin/bash

rsync -azP /Users/marta/Documents/MSc\ Artificial\ Intelligence/Thesis/csp-elites/csp_elites msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
scp run_experiment.py msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
ssh -J msw16@shell5.doc.ic.ac.uk msw16@gpucluster2.doc.ic.ac.uk << EOF
cd csp_experiments/
sbatch script_job.sh
EOF
