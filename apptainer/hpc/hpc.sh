#!/bin/bash

# Check if container path is valid
if [ ! -f $1 ]; then
	echo ERROR: invalid container path.
	exit 1
fi

# Parse hpc.yaml configuration file
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
source $SCRIPT_DIR/parse_yaml.sh
eval $(parse_yaml apptainer/hpc.yaml)

# Define additional shell variables
project_name=${PWD##*/}
container_path=$1
container_name=${container_path##*/}
container_directory=${container_name%.*}
commit=$(echo $container_directory | cut -d "_" -f 4)

# Check parsed configuration
if [ -z "$walltime" ]; then
	echo ERROR: walltime not defined in hpc.yaml.
	exit 1
fi
if [ -z "$nnodes" ]; then
	echo ERROR: nnodes not defined in hpc.yaml.
	exit 1
fi
if [ -z "$ncpus" ]; then
	echo ERROR: ncpus not defined in hpc.yaml.
	exit 1
fi
if [ -z "$mem" ]; then
	echo ERROR: mem not defined in hpc.yaml.
	exit 1
fi
if [ -z "$ngpus" ]; then
	echo ERROR: ngpus not defined in hpc.yaml.
	exit 1
fi
if [ -z "$gpu_type" ]; then
	echo ERROR: gpu_type not defined in hpc.yaml.
	exit 1
fi
if [ ! $njobs -gt 0 ]; then
	echo ERROR: njobs needs to be a positive integer.
	exit 1
fi

# Create select shell variable
if [ $ngpus -gt 0 ]; then
	select="$nnodes:ncpus=$ncpus:mem=$mem:ngpus=$ngpus:gpu_type=$gpu_type"
else
	select="$nnodes:ncpus=$ncpus:mem=$mem"
fi

# Create array job PBS directive
if [ $njobs -gt 1 ]; then
	pbs_array="#PBS -J 1-$njobs"
fi

# Create queue shell variable
if [ "$queue" == "null" ]; then
	queue=""
fi

# Create temporary directory
TMP_DIR=$(mktemp -d -p apptainer/)

# Save git log output to track container
git log --decorate --color -10 $commit > $TMP_DIR/git-log.txt

# Send container and git log to the HPC
ssh hpc "mkdir -p ~/projects/$project_name/$container_directory/"
rsync --verbose --ignore-existing --progress -e ssh $container_path $TMP_DIR/git-log.txt hpc:~/projects/$project_name/$container_directory/

# Create jobscripts
table="Job ID,Job Name,Job Script,Status,args\n"
for args in $args_; do
	# Expand args
	args=$(eval echo \$${args})

	# Build jobscript from template
	TMP_JOBSCRIPT=$(mktemp -p $TMP_DIR)
	sed "s/@job_name/$job_name/g" $SCRIPT_DIR/template.job > $TMP_JOBSCRIPT
	sed -i "s/@walltime/$walltime/g" $TMP_JOBSCRIPT
	sed -i "s/@select/$select/g" $TMP_JOBSCRIPT
	sed -i "s/@pbs_array/$pbs_array/g" $TMP_JOBSCRIPT
	sed -i "s/@project_name/$project_name/g" $TMP_JOBSCRIPT
	sed -i "s/@container_directory/$container_directory/g" $TMP_JOBSCRIPT
	sed -i "s/@container_name/$container_name/g" $TMP_JOBSCRIPT
	sed -i "s/@commit/$commit/g" $TMP_JOBSCRIPT
	sed -i "s/@args/$args/g" $TMP_JOBSCRIPT
	sed -i "s/@wandb_api_key/$WANDB_API_KEY/g" $TMP_JOBSCRIPT

	# Send jobscript to the HPC
	rsync --quiet --progress -e ssh $TMP_JOBSCRIPT hpc:~/projects/$project_name/$container_directory/
	jobid=$(ssh hpc "cd ~/projects/$project_name/$container_directory/ && /opt/pbs/bin/qsub $queue ~/projects/$project_name/$container_directory/${TMP_JOBSCRIPT##*/} 2> /dev/null")

	# Rename jobscript to $jobid.job
	if [ $? == 0 ]; then
		ssh hpc "mv ~/projects/$project_name/$container_directory/${TMP_JOBSCRIPT##*/} ~/projects/$project_name/$container_directory/${job_name}_${jobid%.*}.job"
		table+="${jobid%.*},$job_name,${job_name}_${jobid%.*}.job,Queued,$args\n"
	else
		ssh hpc "rm ~/projects/$project_name/$container_directory/${TMP_JOBSCRIPT##*/}"
		table+="-,-,-,Failed,$args\n"
	fi
done

rm -rf $TMP_DIR
echo
echo -e $table | column -s, -t
