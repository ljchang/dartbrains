# Download Data

*Written by Luke Chang*

Many of the imaging tutorials throughout this course will use open data from the Pinel Localizer task.

The Pinel Localizer task was designed to probe several different types of basic cognitive processes, such as visual perception, finger tapping, language, and math. Several of the tasks are cued by reading text on the screen (i.e., visual modality) and also by hearing auditory instructions (i.e., auditory modality). The trials are randomized across conditions and have been optimized to maximize efficiency for a rapid event related design. There are 100 trials in total over a 5-minute scanning session. Read the original [paper](https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-8-91) for more specific details about the task and the [dataset paper](https://doi.org/10.1016/j.neuroimage.2015.09.052). 

This dataset is well suited for these tutorials as it is (a) publicly available to anyone in the world, (b) relatively small (only about 5min), and (c) provides many options to create different types of contrasts.

There are a total of 94 subjects available, but we will primarily only be working with a smaller subset of about 30.

Downloading the data is very easy as it is currently available on the [OSF website](https://osf.io/vhtf6/files/) and also 

We will use the `osfclient` [package](https://github.com/osfclient/osfclient) to download the entire dataset. Note, that the entire dataset is fairly large (~5.25gb), so make sure you have space on your computer. At some point, we will make a smaller version for the dartbrain course available for download.

If you are taking the Psych60 course at Dartmouth, we have already made the download available on the jupyterhub server.

Let's first make sure the `osfclient` package is installed in our python environment.

In this notebook, we will walk through how to access the datset using DataLad. 

## DataLad

The easist way to access the data is using [DataLad](https://www.datalad.org/), which is an open source version control system for data built on top of [git-annex](https://git-annex.branchable.com/). Think of it like git for data. It provides a handy command line interface for downloading data, tracking changes, and sharing it with others.

While DataLad offers a number of useful features for working with datasets, there are three in particular that we think make it worth the effort to install for this course.

1) Cloning a DataLad Repository can be completed with a single line of code `datalad clone <repository>` and provides the full directory structure in the form of symbolic links. This allows you to explore all of the files in the dataset, without having to download the entire dataset at once.

2) Specific files can be easily downloaded using `datalad get <filename>`, and files can be removed from your computer at any time using `datalad drop <filename>`. As these datasets are large, this will allow you to only work with the data that you need for a specific tutorial and you can drop the rest when you are done with it.

3) All of the DataLad commands can be run within Python using the datalad [python api](http://docs.datalad.org/en/latest/modref.html).

We will only be covering a few basic DataLad functions to get and drop data. We encourage the interested reader to read the very comprehensive DataLad [User Handbook](http://handbook.datalad.org/en/latest/) for more details and troubleshooting.

### Installing Datalad

DataLad can be easily installed using [pip](https://pip.pypa.io/en/stable/).

`pip install datalad`

Unfortunately, it currently requires manually installing the [git-annex](https://git-annex.branchable.com/) dependency, which is not automatically installed using pip.

If you are using OSX, we recommend installing git-annex using [homebrew](https://brew.sh/) package manager.

`brew install git-annex`

If you are on Debian/Ubuntu we recommend enabling the [NeuroDebian](http://neuro.debian.net/) repository and installing with apt-get.

`sudo apt-get install datalad`

For more installation options, we recommend reading the DataLad [installation instructions](https://git-annex.branchable.com/).


!pip install datalad

### Download Data with DataLad

The Pinel localizer dataset can be accessed at the following location https://gin.g-node.org/ljchang/Localizer/. To download the Localizer dataset run `datalad install https://gin.g-node.org/ljchang/Localizer` in a terminal in the location where you would like to install the dataset. Don't forget to change the directory to a folder on your local computer. The full dataset is approximately 42gb.

You can run this from the notebook using the `!` cell magic.

%cd ~/Dropbox/Dartbrains/data

!datalad install https://gin.g-node.org/ljchang/Localizer

## Datalad Basics

You might be surprised to find that after cloning the dataset that it barely takes up any space `du -sh`. This is because cloning only downloads the metadata of the dataset to see what files are included.

You can check to see how big the entire dataset would be if you downloaded everything using `datalad status`.

%cd ~/Dropbox/Dartbrains/data/Localizer

!datalad status --annex

### Getting Data
One of the really nice features of datalad is that you can see all of the data without actually storing it on your computer. When you want a specific file you use `datalad get <filename>` to download that specific file. Importantly, you do not need to download all of the dat at once, only when you need it.

Now that we have cloned the repository we can grab individual files. For example, suppose we wanted to grab the first subject's confound regressors generated by fmriprep.

!datalad get participants.tsv

Now we can check and see how much of the total dataset we have downloaded using `datalad status`

!datalad status --annex all

If you would like to download all of the files you can use `datalad get .`. Depending on the size of the dataset and the speed of your internet connection, this might take awhile. One really nice thing about datalad is that if your connection is interrupted you can simply run `datalad get .` again, and it will resume where it left off.

You can also install the dataset and download all of the files with a single command `datalad install -g https://gin.g-node.org/ljchang/Localizer`. You may want to do this if you have a lot of storage available and a fast internet connection. For most people, we recommend only downloading the files you need for a specific tutorial.

### Dropping Data
Most people do not have unlimited space on their hard drives and are constantly looking for ways to free up space when they are no longer actively working with files. Any file in a dataset can be removed using `datalad drop`. Importantly, this does not delete the file, but rather removes it from your computer. You will still be able to see file metadata after it has been dropped in case you want to download it again in the future.

As an example, let's drop the Localizer participants .tsv file.

!datalad drop participants.tsv

## Datalad has a Python API!
One particularly nice aspect of datalad is that it has a Python API, which means that anything you would like to do with datalad in the commandline, can also be run in Python. See the details of the datalad [Python API](http://docs.datalad.org/en/latest/modref.html).

For example, suppose you would like to clone a data repository, such as the Localizer dataset. You can run `dl.clone(source=url, path=location)`. Make sure you set `localizer_path` to the location where you would like the Localizer repository installed.

import os
import glob
import datalad.api as dl
import pandas as pd

localizer_path = '/Users/lukechang/Dropbox/Dartbrains/data/Localizer'

dl.clone(source='https://gin.g-node.org/ljchang/Localizer', path=localizer_path)


We can now create a dataset instance using `dl.Dataset(path_to_data)`.

ds = dl.Dataset(localizer_path)

How much of the dataset have we downloaded?  We can check the status of the annex using `ds.status(annex='all')`.

results = ds.status(annex='all')

Looks like it's empty, which makes sense since we only cloned the dataset. 

Now we need to get some data. Let's start with something small to play with first.

Let's use `glob` to find all of the tab-delimited confound data generated by fmriprep. 

file_list = glob.glob(os.path.join(localizer_path, '*', 'fmriprep', '*', 'func', '*tsv'))
file_list.sort()
file_list[:10]

glob can search the filetree and see all of the relevant data even though none of it has been downloaded yet.

Let's now download the first subjects confound regressor file and load it using pandas.

result = ds.get(file_list[0])

confounds = pd.read_csv(file_list[0], sep='\t')
confounds.head()

What if we wanted to drop that file? Just like the CLI, we can use `ds.drop(file_name)`.

result = ds.drop(file_list[0])

To confirm that it is actually removed, let's try to load it again with pandas.

confounds = pd.read_csv(file_list[0], sep='\t')


Looks like it was successfully removed.

We can also load the entire dataset in one command if want using `ds.get(dataset='.', recursive=True)`. We are not going to do it right now as this will take awhile and require lots of free hard disk space.

Let's actually download one of the files we will be using in the tutorial. First, let's use glob to get a list of all of the functional data that has been preprocessed by fmriprep, denoised, and smoothed.

file_list = glob.glob(os.path.join(localizer_path, 'derivatives', 'fmriprep', '*', 'func', '*task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))
file_list.sort()
file_list

Now let's download the first subject's file using `ds.get()`. This file is 825mb, so this might take a few minutes depending on your internet speed.

result = ds.get(file_list[0])

How much of the dataset have we downloaded?  We can check the status of the annex using `ds.status(annex='all')`.

result = ds.status(annex='all')

Now let's download the preprocessed data for the first 15 subjects including the fmriprep reports.

file_list = glob.glob(os.path.join(localizer_path, '*', 'fmriprep', 'sub*'))
file_list.sort()
for f in file_list[:30]:
    result = ds.get(f)

Ok, that concludes our tutorial for how to download data for this course with datalad using both the command line interface and also the Python API.