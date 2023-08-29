rsync -azP /Users/marta/Documents/MSc\ Artificial\ Intelligence/Thesis/csp-elites/csp_elites msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
rsync -azP /Users/marta/Documents/MSc\ Artificial\ Intelligence/Thesis/csp-elites/configs msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
rsync -azP /Users/marta/Documents/MSc\ Artificial\ Intelligence/Thesis/csp-elites/.experiment.nosync/experiments/centroids msw16@shell5.doc.ic.ac.uk:~/csp_experiments/experiments/
rsync -azP /Users/marta/Documents/MSc\ Artificial\ Intelligence/Thesis/csp-elites/mp_reference_analysis msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
rsync -azP /Users/marta/Documents/MSc\ Artificial\ Intelligence/Thesis/csp-elites/retrieve_results msw16@shell5.doc.ic.ac.uk:~/csp_experiments/

scp run_experiment.py msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
scp main.py msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
scp experiment_from_config.py msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
scp experiment_from_config_cma.py msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
scp script_job.sh msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
scp script_job_2.sh msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
scp script_job_3.sh msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
scp script_job_4.sh msw16@shell5.doc.ic.ac.uk:~/csp_experiments/
scp script_job_5.sh msw16@shell5.doc.ic.ac.uk:~/csp_experiments/

ssh -J msw16@shell5.doc.ic.ac.uk msw16@gpucluster2.doc.ic.ac.uk << EOF
cd csp_experiments/
sbatch script_job.sh
sbatch script_job_2.sh
sbatch script_job_3.sh
sbatch script_job_4.sh
sbatch script_job_5.sh
EOF
