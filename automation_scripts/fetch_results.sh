#!/bin/bash

rsync -azP msw16@login.hpc.imperial.ac.uk:~/csp-elites/experiments /Users/marta/Documents/MSc\ Artificial\ Intelligence/Thesis/csp-elites/.experiment.nosync
rsync -azP msw16@shell5.doc.ic.ac.uk:~/csp_experiments/experiments /Users/marta/Documents/MSc\ Artificial\ Intelligence/Thesis/csp-elites/.experiment.nosync

#python3 process_all_results.py
