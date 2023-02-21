# Running fMRIPrep on HPC

*Written by Mijin Kwon, Era Wu, & Luke Chang*

[fmriprep](https://fmriprep.readthedocs.io/en/stable/) is a data preprocessing pipeline for preprocessing fMRI data that was developed by a team at the [Center for Reproducible Research](http://reproducibility.stanford.edu/) led by Russ Poldrack and Chris Gorgolewski. Fmriprep was designed to provide an easily accessible, state-of-the-art interface that is robust to variations in scan acquisition protocols, requires minimal user input, and provides easily interpretable and comprehensive error and output reporting. Fmriprep performs basic processing steps (coregistration, normalization, unwarping, noise component extraction, segmentation, skullstripping etc.) providing outputs that are ready for data analysis. Additional information about using fmriprep can be found on [Andy Jahn's Brain Book](https://andysbrainbook.readthedocs.io/en/latest/OpenScience/OS/fMRIPrep.html#fmriprep) and also [neurostars](https://neurostars.org/tag/fmriprep).

In this tutorial, we will discuss how to run fmriprep on a high performance computing system (HPC) using the [Discovery system](https://rc.dartmouth.edu/index.php/discovery-overview/) at Dartmouth as an example. Dartmouth students will need to [request an account](https://rcweb.dartmouth.edu/accounts/) if you don't yet already have access to the system. We also recommend being added to the DBIC group to access our department nodes.

## Basics

### Accessing Data in Rolando

At the Dartmouth Brain Imaging Center (DBIC), all data collected on the scanner is copied to Rolando (rolando.cns.dartmouth.edu). To gain access to rolando, you will need to request an account from the DBIC systems administrator (Andrew Connolly andrew.c.connolly@dartmouth.edu). More details can be found here: https://dbic-handbook.readthedocs.io/en/latest/mri/dataaccess.html. Converting your dataset into the [BIDS format](https://bids.neuroimaging.io/) can be done using [heudiconv](https://heudiconv.readthedocs.io/en/latest/) after speaking with [Yarik Halchenko](https://centerforopenneuroscience.org/whoweare). This is still in the process of being fully automated.

### Download Data with DataLad

For this tutorial, we can use download sample data using [datalad](http://handbook.datalad.org/en/latest/) with the [Download Data](Download_Data.ipynb) tutorial. 

### Using Discovery

[Discovery](https://rc.dartmouth.edu/index.php/discovery-overview/) is a High Performance Computing system at Dartmouth. You will log on to a headnode and then will submit jobs as a scheduled batch job or an interactive job via the [SLURM] scheduler. Questions about Discovery can be directed to [Research Computing](https://rc.dartmouth.edu/index.php/contact-us/).

You will need to be on the Dartmouth network to have permission to login in to the Discovery head nodes. If you are on campus, use the eduroam wifi. Otherwise you will need to use [VPN](https://services.dartmouth.edu/TDClient/1806/Portal/KB/?CategoryID=17668).

#### Logging on to Discovery

The discovery system can be accessed via `ssh` using a terminal. There are a number of options depending on what you intend to do. For example, the `-X` or `-Y` flag can can enable X11 forwarding in case you would like to have graphics enabled. This can be useful for using GUIs such as FSL or matlab. You can also tunnel into a [Jupyter server](https://www.askpbs.org/t/how-do-i-run-a-jupyter-notebook-server-on-discovery/37/3) launched on an [interactive job](https://www.askpbs.org/t/how-do-i-submit-an-interactive-job/36).

  ● Linux: ssh -X username@discovery.dartmouth.edu  

  ● Mac: ssh -Y username@discovery.dartmouth.edu  
          - Install Xquartz for graphical interface

  ● Windows:
          - MobaXterm built in Xserver and sftp (free and recommended)
          - Ssh secure shell or putty

### Preprocessing with fmriprep

fmriprep is an fMRI data preprocessing pipeline that is discussed throughout the course in the [fmriprep](fmriprep) subsection of the [preprocessing](Preprocessing.ipynb) and also the [download data](Download_Data.ipynb) tutorial.

## Set-up before running fMRIPrep

fmriprep requires numerous software dependencies, which can be tricky to install. We recommend using containers created by the fmriprep developers to avoid dealing with software installation issues. If you are working on your own computer and have root access to your machine, we recommend using the [docker container](https://fmriprep.org/en/1.5.9/docker.html). If you are running your preprocessing on discovery or a computer where you do not have root access, we recommend using the [singularity container](https://fmriprep.org/en/1.5.9/singularity.html).

### Singularity

In this example, we will use singularity to run fmriprep on Discovery as Docker containers are not permitted on HPC systems due to required root access permissions. Singularity is already available on Disocver, so you shouldn't need to install anything else to get the container to work. If you have a docker container, you can convert this into a singularity container following this [tutorial](https://www.askpbs.org/t/how-do-i-use-docker-on-discovery/51). 

You should also ask other members of your lab as there is a high likilihood that there is already a working container in your lab's shared storage folder.

For this course, we have fmriprep singularity containers available in the DBIC folder `/dartfs/rc/lab/D/DBIC/DBIC/psych160/resources/fmriprep/fmriprep-21.0.1.simg`.

### FreeSurfer licence
Once installing or locating (if your lab already has a singularitly container shared across the lab) fMRIPrep, you also need to register on the FreeSurfer website, download the FreeSurfer license key (`license.txt` file) and save it under `/YOUR-DATA-DIRECTORY/derivative`. Again, if your lab is already using fMRIPrep, this is also likely to be already saved in your lab space along with the singularity container. 

For this course, we have a freesurfer license available in the DBIC folder `/dartfs/rc/lab/D/DBIC/DBIC/psych160/resources/fmriprep/license.txt`.

## Run fMRIPrep

Once you have finished setting everything, you are ready to run fMRIPrep on Discovery.

### Log on to Discovery
First, you will need to log on to a discovery headnode. The headnode is a way for you to browse, edit, copy, and move files, edit scripts and submit jobs to the slurm scheduler. **You should never be running fmriprep on a headnode!**.

```ssh #netID#@discovery.dartmouth.edu```

### Decide Job Type
Second, you will need to decide if you are going to run your fmriprep job as an interactive job or as a scheduled batch job on slurm. [Interactive jobs](https://www.askpbs.org/t/how-do-i-submit-an-interactive-job/36) means that you request computational resources from slurm that you can directly interact with. This is useful if you want to develop or test your script on Discovery. You can also interactively work with jupyter notebooks using interactive jobs following [this](https://www.askpbs.org/t/how-do-i-run-a-jupyter-notebook-server-on-discovery/37) tutorial. Alternatively, you can submit a batch job, in which you will tell slurm to run a specific batch script when resources are available. This job will have access to resources available in your shell such as your python distribution. It is important to remember that all batch jobs need to include loading data, what you want to do with the data, and explicitly writing out anything you want to save during the job for later. fmriprep is designed to run a job for a single subject. You could choose to loop over subjects within your batchscript, which means that subjects will be run serially. You could submit multiple jobs for each subject. Alternatively, you could submit a [job array](https://services.dartmouth.edu/TDClient/1806/Portal/KB/ArticleDet?ID=132181), which will run multiple jobs in parallel depending on how many resources are available. We recommend using job arrays for the fastest and most efficient use of Discovery resources. 


### Create Batchscript
Third, you will need to create your batchscript. Below is an example bash script (`example_slurm_fmriprep.sh`) that you can customize according to what you need for you data. You can find more about different options that can be modified on [this page](https://fmriprep.org/en/stable/usage.html) in the offical fMRIPrep documentation.

One suggestion is to have a folder with your container and freesurfer license file. For example, `FMRIPREP_RESOURCES_PATH=/dartfs/rc/lab/D/DBIC/DBIC/psych160/resources/fmriprep`.


```
#!/bin/bash

# Name of the job (* specify job name)
#SBATCH --job-name= YOUR-JOB-NAME

# Number of compute nodes
#SBATCH --nodes=1

# Number of CPUs per task
#SBATCH --cpus-per-task=8

# Request memory
#SBATCH --mem-per-cpu=4gb

# save logs (change YOUR-DIRECTORY to where you want to save logs)
#SBATCH --output=/YOUR-DIRECTORY/fmriprep_log.txt
#SBATCH --error=/YOUR-DIRECTORY/fmriprep_error.txt

# Walltime (job duration)
#SBATCH --time=24:00:00

# Array jobs (* change the range according to # of subject; % = number of active job array tasks)
#SBATCH --array=1-10%5

# Email notifications (*comma-separated options: BEGIN,END,FAIL)
#SBATCH --mail-type=BEGIN,END,FAIL

# Account to use (*change to any other account you are affiliated with)
#SBATCH --account=dbic

# Parameters
participants=(01 02 03 04 05 06 07 08 09 10)
PARTICIPANT_LABEL=${participants[(${SLURM_ARRAY_TASK_ID} - 1)]}
BIDS_DIR=/PATH-TO-YOUR-DATA-DIRECTORY/
OUTPUT_DIR=/PATH-TO-YOUR-OUTPUT-DIRECTORY/
WORK_DIR=/PATH-TO-YOUR-WORKING-DIRECTORY/
FMRIPREP_RESOURCES_PATH=/PATH-TO-YOUR-DIRECTORY-WITH_CONTAINER_AND_LICENSE/

echo "array id: " ${SLURM_ARRAY_TASK_ID}, "subject id: " ${PARTICIPANT_LABEL}

singularity run \
                --cleanenv \
                -B ${FMRIPREP_RESOURCES_PATH}:/resources \
                -B ${BIDS_DIR}:/data \
                -B ${WORK_DIR}:/work \
                -B ${OUTPUT_DIR}:/output \
        ${FMRIPREP_RESOURCES_PATH}/fmriprep-21.0.1.simg /data /output \
        participant --participant_label $PARTICIPANT_LABEL \
        -w /work \
        --nprocs 8 \
        --write-graph \
        --fs-license-file /resources/license.txt \
        --ignore slicetiming \
        --fs-no-reconall \
```

There might be other flags to customize your preprocessing depending on what you want to do.  For example, you might want discard the first 10 acquisitions (i.e., disdaqs) `--dummy-scans 10`, or skip the bids validation `--skip-bids-validation`.


### Submit Job
Fourth, you will want to submit a job to slurm to run your batchscript as job array. 

```sbatch example_slurm_fmriprep.sh```

You can monitor the status of your job with `squeue`, modify your job with `scontrol`, or cancel your job with `scancel`.

Here is a helpful list of [slurm commands](https://services.dartmouth.edu/TDClient/1806/Portal/KB/ArticleDet?ID=132625) that can be used on discovery. 

### Inspect Results
Finally, you should receive an email when your job completes and you can you can see the output of your preprocessing in `/YOUR-DATA-FOLDER/bids/derivatives/fmriprep/`.


## Reference and resources

This tutorial is an output based on a lot of existing resources related to this topic. Here are the list of references and resources that we referred to:

* [fMRIPrep](https://fmriprep.org/en/stable/index.html)
* [NeuroImaging PREProcessing toolS (NiPreps)](https://www.nipreps.org/)
* [Research computing at Dartmouth](https://rc.dartmouth.edu/)
* [Discovery overview](https://rc.dartmouth.edu/index.php/discovery-overview/)
* [DBIC Handbook](https://dbic-handbook.readthedocs.io/en/latest/)
* [Slurm overview at Darmouth Service Portal](https://services.dartmouth.edu/TDClient/1806/Portal/KB/ArticleDet?ID=132625)
* [DartBrain](https://dartbrains.org/)
* [Datalad Handbook](http://handbook.datalad.org/en/latest/)
* [Andy Jahn's Brain Book](https://andysbrainbook.readthedocs.io/en/latest/OpenScience/OS/fMRIPrep.html#fmriprep)
* [Slurm documentation](https://slurm.schedmd.com/)
* [Slurm key command list](https://slurm.schedmd.com/pdfs/summary.pdf)
