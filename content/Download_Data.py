# /// script
# dependencies = ["datalad"]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Interactive version:** [Open this notebook in molab](https://molab.marimo.io/github/ljchang/dartbrains/blob/v2-marimo-migration/content/Download_Data.py) to run code, interact with widgets, and modify examples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (download-data) =
    # Download Data

    *Written by Luke Chang & Kevin Ortego*

    Many of the imaging tutorials throughout this course will use open data from the Pinel Localizer task.

    The Pinel Localizer task was designed to probe several different types of basic cognitive processes, such as visual perception, finger tapping, language, and math. Several of the tasks are cued by reading text on the screen (i.e., visual modality) and also by hearing auditory instructions (i.e., auditory modality). The trials are randomized across conditions and have been optimized to maximize efficiency for a rapid event related design. There are 100 trials in total over a 5-minute scanning session. Read the original [paper](https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-8-91) for more specific details about the task and the [dataset paper](https://doi.org/10.1016/j.neuroimage.2015.09.052).

    This dataset is well suited for these tutorials as it is (a) publicly available to anyone in the world, (b) relatively small (only about 5min), and (c) provides many options to create different types of contrasts.

    There are a total of 94 subjects available, but we will primarily only be working with a smaller subset of about 15.

    Though the data is being shared on the [OSF website](https://osf.io/vhtf6/files/), we recommend downloading it from our [g-node repository](https://gin.g-node.org/ljchang/Localizer) as we have fixed a few issues with BIDS formatting and have also performed preprocessing using fmriprep.

    In this notebook, we will walk through how to access the datset using DataLad. Note, that the entire dataset is fairly large (~42gb), but the tutorials will mostly only be working with a small portion of the data (5.8gb), so there is no need to download the entire thing. If you are taking the Psych60 course at Dartmouth, we have already made the data available on the jupyterhub server.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Downloading Data from HuggingFace (Recommended)

    The Pinel Localizer dataset is hosted on [HuggingFace](https://huggingface.co/datasets/dartbrains/localizer). This is the recommended way to access the data for this course. Files are downloaded automatically and cached locally — no extra tools needed.

    ### Using the Course Helper Module

    The `Code.data` module provides convenient functions to download and access any file in the dataset:
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Code.data import get_file, get_subjects, load_events, get_tr, REPO_ID

    # List all subjects
    print(f"Subjects: {get_subjects()}")
    print(f"TR: {get_tr()} seconds")

    # Download a preprocessed BOLD file (cached after first download)
    bold_path = get_file('S01', 'derivatives', 'bold')
    print(f"\nBOLD file path: {bold_path}")
    return (REPO_ID, get_file, get_subjects, load_events)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loading Event Timing Data

    Each subject's task events (stimulus onsets, durations, and conditions) can be loaded directly as a DataFrame:
    """)
    return


@app.cell
def _(load_events):
    events = load_events('S01')
    events.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Direct File Access

    You can also download any file directly from HuggingFace using `hf_hub_download`:
    """)
    return


@app.cell
def _(REPO_ID):
    from huggingface_hub import hf_hub_download

    # Download a specific beta map
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename="derivatives/betas/S01_betas.nii.gz",
        repo_type="dataset",
    )
    print(f"Downloaded to: {path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Browsing the Dataset as a BIDS Tree

    `hf_hub_download` and `get_file()` cache files in `~/.cache/huggingface/hub/datasets--dartbrains--localizer/`, but the cache uses a content-addressed layout (`blobs/` for raw bytes, `snapshots/<commit>/` for symlinks back to those blobs with their original filenames). The `snapshots/` folder *does* preserve the original BIDS tree exactly, but the path is awkward to type.

    If you'd rather browse the dataset like a normal BIDS directory — `cd` into it, `ls` subjects, drag it into a file explorer, point external tools at it — the cleanest pattern is to download a full snapshot and symlink it to a friendly location of your choice.

    First, pull the snapshot. Files you've already cached with `get_file()` or `hf_hub_download` are reused, so this is fast on a second call:
    """)
    return


@app.cell
def _(REPO_ID):
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Snapshot lives at:\n  {snapshot_path}")
    return (snapshot_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now create a symlink from somewhere convenient (e.g. `~/data/localizer`) pointing at the snapshot. The symlink takes ~no disk space and lets you treat the cached data as if it lived in `~/data/localizer`:
    """)
    return


@app.cell
def _(snapshot_path):
    from pathlib import Path

    bids_root = Path.home() / "data" / "localizer"
    bids_root.parent.mkdir(parents=True, exist_ok=True)

    if bids_root.exists() or bids_root.is_symlink():
        bids_root.unlink()  # replace any stale symlink
    bids_root.symlink_to(snapshot_path)

    print(f"Browse the BIDS tree at: {bids_root}")
    return (bids_root,)


@app.cell
def _(bids_root):
    # Sanity check — list the top-level entries
    for _entry in sorted(bids_root.iterdir())[:10]:
        print(_entry.name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A few practical notes:

    - **Reuses the cache.** Both the snapshot folder and your symlink target ultimately point at the same content-addressed blobs. Files aren't duplicated, and `huggingface_hub` won't re-download anything you already have.
    - **Lazy fetch + symlink in one step.** If you'd rather have `huggingface_hub` materialize the tree directly at your chosen path (without going through `snapshot_download`'s default cache location), pass `local_dir=` and `local_dir_use_symlinks=True`:
      ```python
      snapshot_download(
          repo_id="dartbrains/localizer", repo_type="dataset",
          local_dir="~/data/localizer", local_dir_use_symlinks=True,
      )
      ```
      With `local_dir_use_symlinks=True` (the default on macOS/Linux), `~/data/localizer` will contain symlinks to the cached blobs — same end state, no extra disk usage.
    - **Windows caveat.** Symlinks on Windows require either developer mode enabled or admin privileges. If you hit a permission error, pass `local_dir_use_symlinks=False` to copy the bytes instead (uses ~the size of the snapshot in extra disk space).
    - **Updating.** If the dataset is updated on HuggingFace, re-run `snapshot_download` — it pulls only the changed blobs and updates the snapshot folder. Your symlink still points to the right place.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bulk Loading with the `datasets` Library

    For loading all beta maps or events at once, use the `datasets` library:
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    ds = load_dataset("dartbrains/localizer", "betas")
    print(f"Loaded {len(ds['train'])} beta maps")
    print(f"First entry: subject={ds['train'][0]['subject']}, condition={ds['train'][0]['condition']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Downloading Data with DataLad (Legacy)

    The dataset is also available via [DataLad](https://www.datalad.org/) from the GIN repository. This was the original download method and still works as an alternative.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DataLad

    The easist way to access the data is using [DataLad](https://www.datalad.org/), which is an open source version control system for data built on top of [git-annex](https://git-annex.branchable.com/). Think of it like git for data. It provides a handy command line interface for downloading data, tracking changes, and sharing it with others.

    While DataLad offers a number of useful features for working with datasets, there are three in particular that we think make it worth the effort to install for this course.

    1) Cloning a DataLad Repository can be completed with a single line of code `datalad clone <repository>` and provides the full directory structure in the form of symbolic links. This allows you to explore all of the files in the dataset, without having to download the entire dataset at once.

    2) Specific files can be easily downloaded using `datalad get <filename>`, and files can be removed from your computer at any time using `datalad drop <filename>`. As these datasets are large, this will allow you to only work with the data that you need for a specific tutorial and you can drop the rest when you are done with it.

    3) All of the DataLad commands can be run within Python using the datalad [python api](http://docs.datalad.org/en/latest/modref.html).

    We will only be covering a few basic DataLad functions to get and drop data. We encourage the interested reader to read the very comprehensive DataLad [User Handbook](http://handbook.datalad.org/en/latest/) for more details and troubleshooting.

    ### Installing Datalad on Mac and Unix Operating Systems

    DataLad can be easily installed using [pip](https://pip.pypa.io/en/stable/).

    `pip install datalad`

    Unfortunately, it currently requires manually installing the [git-annex](https://git-annex.branchable.com/) dependency, which is not automatically installed using pip.

    If you are using OSX, we recommend installing git-annex using [homebrew](https://brew.sh/) package manager.

    `brew install git-annex`

    If you are on Debian/Ubuntu we recommend enabling the [NeuroDebian](http://neuro.debian.net/) repository and installing with apt-get.

    `sudo apt-get install datalad`

    For more installation options, we recommend reading the DataLad [installation instructions](https://git-annex.branchable.com/).
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: datalad !pip install datalad
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Installing Datalad on Windows Operating Systems

    Installing Datalad on Windows can be a little more tricky compared to Unix based operating systems and there are limited tutorials available. Hopefully, windows users will find this tutorial useful.

    DataLad requires several components to work:
    1. **Python**
    2. **Git**
    3. **GitAnnex**
    4. **Datalad**

    There is a good chance you may already have Python or Git installed on your computer. However, this may be problematic as DataLad requires specific configurations for both Python and Git installations in order to work. These are detailed on the DataLad website, but it can be easy to miss or skip over, especially if you already have some of these packages installed. Here isa summary of what you should check, as well as how to potentially resolve problems without having to reinstall things.  If you don't have Python or Git installed yet, you can follow these instructions and installation should be relatively straightforward.

    #### 1) Python

    **If you need to install Python:**
    The [Anaconda Distribution](https://www.anaconda.com/products/distribution) has the most relevant packages for scientific computing already included and is widely recommended. Be sure to get Python 3, and the default installer options are generally safe, except **be sure to select the *ADD PYTHON TO PATH* option**, otherwise Datalad will not work. After you're done, proceed to Step 2 on Git.

    **If you already have Python installed on your computer:**
    + You may run into problems when installing DataLad if you did not add Python to your Windows path when installing Python.
    + This is especially likely because the Anaconda distribution installer **strongly discourages** you from adding Python to the path when navigating through the installation dialogue.
    + You can check if Python is on your path by doing the following:
        + Press WindowsKey + x and click "System" in the menu that pops up
        + Scroll down to "Related settings" and click "Advanced system settings"
        + Under the "Advanced" tab, click "Environment Variables"
        + In the "User Variables" pane you should see a variable called "Path" with some values corresponding to the path of your Python installation.
    + If Python is on your path, you *should* be good to go.
    + If Python is not on your path, you have two options:
        1. Uninstall Python, then reinstall, being sure to select the add Python to Windows path option this time (this option is recommended and guaranteed to work)
        2. Try adding Python to your Windows path manually:
            + If you already installed DataLad, you should uninstall it and reinstall after doing this
            + Instructions for adding Python to your path can be found [here](https://datatofish.com/add-python-to-windows-path/), but what you need to add to the path may differ for different distributions.  For instance, my Anaconda distribution has several other folders listed in its path entry that are not listed for the more basic Python distribution used at the link above.  For completeness if you want to try on your own, my Path has these elements:

                ```
                C:\Users\MyUserName\anaconda3
                C:\Users\MyUserName\anaconda3\Library\mingw-w64\bin
                C:\Users\MyUserName\anaconda3\Library\usr\bin
                C:\Users\MyUserName\anaconda3\Library\bin
                C:\Users\MyUserName\anaconda3\Scripts
                ```

    #### 2) Git

    **If you don't have Git already installed**, it can be found [here](https://git-scm.com/download/win). The default installation options are recommended for most things, but be sure to configure the following options when installing:
    - Enable *Use a TrueType font in all console windows*
    - Select *Git from the command line and also from 3rd-party software*
    - *Enable file system caching*
    - *Enable symbolic links*

    **If you already have Git installed** you should check your configuration settings. You can do so by opening the command prompt and typing:

        > git config --list

    Somewhere in the list of variables that pops up you should see:

        core.fscache = true
        core.symlinks = true

    If not, run the following commands from command prompt to change those settings:

        > git config --global core.symlinks true
        > git config --global core.fscache true

    The ***Git from the command line and also from 3rd-party software*** option is the recommended setting during installation. To check, you can do one of two things:
    1. Navigate to C:\Program Files\Git\etc\install-options, and check for the line "Path Option: Cmd" within that file, **OR**
    2. You can check your Windows path to see if Git is on the path (follow the steps described above for checking if Python is on your Windows path). Git will appear under the "System variables" pane under "Path" instead of under the "User variables" pane.
        + If it isn't there, instructions for adding Git to the path can be found [here](https://www.delftstack.com/howto/git/add-git-to-path-on-windows/#:~:text=Click%20Environment%20Variables%20under%20System,%5Cbin%5Cgit.exe%20.) but this is untested as to whether it will work correctly, especially if you've already installed git-annex and DataLad.

    Unfortunately, you cannot check whether the ***Use a TrueType font in all console windows*** was selected as far as I'm aware, but it is unclear what the implications of not doing that are and whether it would cause DataLad to not work.  If DataLad doesn't work for you once you get there, it is possible that you will need to reinstall git.

    #### 3) Git Annex
    **DO NOT INSTALL GIT ANNEX directly from their website**, because this does not seem to seem to currently work. The Windows installer is still in beta and that there are some known issues. Luckily you can use the git-annex installer provided by DataLad, which does work.

    Run these three commands from the command line to install git-annex:

        > pip install datalad-installer
        > datalad-installer git-annex -m datalad/packages
        > git config --global filter.annex.process "git-annex filter-process"

    #### 4) Datalad
    Installing datalad itself is easy too. Run the following in the command line:

        > pip install datalad

    You are now ready to get started with DataLad! (after you read this [general Warning for windows users from DataLad](https://handbook.datalad.org/en/latest/intro/windows.html#ohnowindows)) And as a final tip, DataLad seems to work best on Windows when used via its [Python API](http://docs.datalad.org/en/latest/modref.html) which can be easily accessed in Python as follows:

        import datalad.api as dl

    #### Windows Path Separators

    When using the DataLad via the command line you will need to first navigate to the folder where the data was installed before you can download the data (this doesn't matter when using the Python API).

    The **cd** command is used in the command prompt to **c**hange **d**irectory. You might notice how the path separators (`/` and `\\` ) are different in the first and second commands. This is a potential issue you might run into when using Windows vs Mac/Unix with Python. The backslash Windows file separator `\`  is different from the forward slash `/` used in other operating systems, or on URLs.  If you're ever using Python to run DataLad commands, or to load and save any kind of data in Python more generally, you may run into problems if you copy folder paths from Windows File Explorer into Python because they'll have the wrong separator. You can fix this in your Python scripts by switching all the `/` to `\`, or you can use a double `\\` which also works.

    + When you open the command prompt, you are in a default directory, which is displayed on the command line. Likely this is `C:\Users\YourUserName\\` and that will show up in the command line as:

            C:\Users\YourUserName> _
            (which is where the ">" before all the code lines comes from)
    + If your installed dataset lives at `C:\Users\YourUserName\ClassData\Localizer\\` you need to navigate to that directory using the cd command before the DataLad `get` command will work, which would look like this in the command line:

            C:\Users\YourUserName> cd ClassData\Localizer

    + Once you are in the data directory, you don't have to type the entire filepath, and you can run a command like

            datalad get sub-S01`

    + And in the command line that whole thing would be rendered like this:

            C:\Users\YourUserName\ClassData\Localizer> datalad get sub-S01
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download Data with DataLad

    The Pinel localizer dataset can be accessed at the following location https://gin.g-node.org/ljchang/Localizer/. To download the Localizer dataset run `datalad install https://gin.g-node.org/ljchang/Localizer` in a terminal in the location where you would like to install the dataset. Don't forget to change the directory to a folder on your local computer. The full dataset is approximately 42gb.

    You can run this from the notebook using the `!` cell magic.
    """)
    return


@app.cell
def _(subprocess):
    import os
    os.chdir('~/Dropbox/Dartbrains/data')

    #! datalad install https://gin.g-node.org/ljchang/Localizer
    subprocess.call(['datalad', 'install', 'https://gin.g-node.org/ljchang/Localizer'])
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Datalad Basics

    You might be surprised to find that after cloning the dataset that it barely takes up any space `du -sh`. This is because cloning only downloads the metadata of the dataset to see what files are included.

    You can check to see how big the entire dataset would be if you downloaded everything using `datalad status`.
    """)
    return


@app.cell
def _(subprocess):
    import os
    os.chdir('~/Dropbox/Dartbrains/data/Localizer')

    #! datalad status --annex
    subprocess.call(['datalad', 'status', '--annex'])
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Getting Data
    One of the really nice features of datalad is that you can see all of the data without actually storing it on your computer. When you want a specific file you use `datalad get <filename>` to download that specific file. Importantly, you do not need to download all of the dat at once, only when you need it.

    Now that we have cloned the repository we can grab individual files. For example, suppose we wanted to grab the first subject's confound regressors generated by fmriprep.
    """)
    return


@app.cell
def _(subprocess):
    #! datalad get participants.tsv
    subprocess.call(['datalad', 'get', 'participants.tsv'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we can check and see how much of the total dataset we have downloaded using `datalad status`
    """)
    return


@app.cell
def _(subprocess):
    #! datalad status --annex all
    subprocess.call(['datalad', 'status', '--annex', 'all'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you would like to download all of the files you can use `datalad get .`. Depending on the size of the dataset and the speed of your internet connection, this might take awhile. One really nice thing about datalad is that if your connection is interrupted you can simply run `datalad get .` again, and it will resume where it left off.

    You can also install the dataset and download all of the files with a single command `datalad install -g https://gin.g-node.org/ljchang/Localizer`. You may want to do this if you have a lot of storage available and a fast internet connection. For most people, we recommend only downloading the files you need for a specific tutorial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dropping Data
    Most people do not have unlimited space on their hard drives and are constantly looking for ways to free up space when they are no longer actively working with files. Any file in a dataset can be removed using `datalad drop`. Importantly, this does not delete the file, but rather removes it from your computer. You will still be able to see file metadata after it has been dropped in case you want to download it again in the future.

    As an example, let's drop the Localizer participants .tsv file.
    """)
    return


@app.cell
def _(subprocess):
    #! datalad drop participants.tsv
    subprocess.call(['datalad', 'drop', 'participants.tsv'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Datalad has a Python API!
    One particularly nice aspect of datalad is that it has a Python API, which means that anything you would like to do with datalad in the commandline, can also be run in Python. See the details of the datalad [Python API](http://docs.datalad.org/en/latest/modref.html).

    For example, suppose you would like to clone a data repository, such as the Localizer dataset. You can run `dl.clone(source=url, path=location)`. Make sure you set `localizer_path` to the location where you would like the Localizer repository installed.
    """)
    return


@app.cell
def _():
    import os
    import glob
    import datalad.api as dl
    import pandas as pd

    localizer_path = '/Users/lukechang/Dropbox/Dartbrains/data/Localizer'

    dl.clone(source='https://gin.g-node.org/ljchang/Localizer', path=localizer_path)
    return dl, glob, localizer_path, os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now create a dataset instance using `dl.Dataset(path_to_data)`.
    """)
    return


@app.cell
def _(dl, localizer_path):
    ds = dl.Dataset(localizer_path)
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How much of the dataset have we downloaded?  We can check the status of the annex using `ds.status(annex='all')`.
    """)
    return


@app.cell
def _(ds):
    results = ds.status(annex='all')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looks like it's empty, which makes sense since we only cloned the dataset.

    Now we need to get some data. Let's start with something small to play with first.

    Let's use `glob` to find all of the tab-delimited confound data generated by fmriprep.
    """)
    return


@app.cell
def _(glob, localizer_path, os):
    file_list = glob.glob(os.path.join(localizer_path, '*', 'fmriprep', '*', 'func', '*tsv'))
    file_list.sort()
    file_list[:10]
    return (file_list,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    glob can search the filetree and see all of the relevant data even though none of it has been downloaded yet.

    Let's now download the first subjects confound regressor file and load it using pandas.
    """)
    return


@app.cell
def _(ds, file_list, pd):
    result = ds.get(file_list[0])

    confounds = pd.read_csv(file_list[0], sep='\t')
    confounds.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What if we wanted to drop that file? Just like the CLI, we can use `ds.drop(file_name)`.
    """)
    return


@app.cell
def _(ds, file_list):
    result_1 = ds.drop(file_list[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To confirm that it is actually removed, let's try to load it again with pandas.
    """)
    return


@app.cell
def _(file_list, pd):
    confounds_1 = pd.read_csv(file_list[0], sep='\t')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looks like it was successfully removed.

    We can also load the entire dataset in one command if want using `ds.get(dataset='.', recursive=True)`. We are not going to do it right now as this will take awhile and require lots of free hard disk space.

    Let's actually download one of the files we will be using in the tutorial. First, let's use glob to get a list of all of the functional data that has been preprocessed by fmriprep, denoised, and smoothed.
    """)
    return


@app.cell
def _(glob, localizer_path, os):
    file_list_1 = glob.glob(os.path.join(localizer_path, 'derivatives', 'fmriprep', '*', 'func', '*task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))
    file_list_1.sort()
    file_list_1
    return (file_list_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's download the first subject's file using `ds.get()`. This file is 825mb, so this might take a few minutes depending on your internet speed.
    """)
    return


@app.cell
def _(ds, file_list_1):
    result_2 = ds.get(file_list_1[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How much of the dataset have we downloaded?  We can check the status of the annex using `ds.status(annex='all')`.
    """)
    return


@app.cell
def _(ds):
    result_3 = ds.status(annex='all')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download Data for Course
    Now let's download the data we will use for the course. We will download:
    - `sub-S01`'s raw data
    - experimental metadata
    - preprocessed data for the first 20 subjects including the fmriprep QC reports.
    """)
    return


@app.cell
def _(ds, glob, localizer_path, os):
    result_4 = ds.get(os.path.join(localizer_path, 'sub-S01'))
    result_4 = ds.get(glob.glob(os.path.join(localizer_path, '*.json')))
    result_4 = ds.get(glob.glob(os.path.join(localizer_path, '*.tsv')))
    result_4 = ds.get(glob.glob(os.path.join(localizer_path, 'phenotype')))
    return


@app.cell
def _(ds, glob, localizer_path, os):
    file_list_2 = glob.glob(os.path.join(localizer_path, '*', 'fmriprep', 'sub*'))
    file_list_2.sort()
    for f in file_list_2[:20]:
        result_5 = ds.get(f)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To get the python packages for the course, install the dependencies listed in the [pyproject.toml](https://github.com/ljchang/dartbrains/blob/v2-marimo-migration/pyproject.toml) using `uv sync`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (run-preprocessing)=
    ## Preprocessing
    The data has already been preprocessed using [fmriprep](https://fmriprep.readthedocs.io/en/stable/), which is a robust, but opinionated automated preprocessing pipeline developed by [Russ Poldrack's group at Stanford University](https://poldracklab.stanford.edu/). The developer's have made a number of choices about how to preprocess your fMRI data using best practices and have created an automated pipeline using multiple software packages that are all distributed via a [docker container](https://fmriprep.org/en/1.5.9/docker.html).

    Though, you are welcome to just start working right away with the preprocessed data, here are the steps to run it yourself:

     - 1. Install [Docker](https://www.docker.com/) and download image

         `docker pull poldracklab/fmriprep:<latest-version>`

     - 2. Run a single command in the terminal specifying the location of the data, the location of the output, the participant id, and a few specific flags depending on specific details of how you want to run the preprocessing.

        `fmriprep-docker /Users/lukechang/Dropbox/Dartbrains/Data/localizer /Users/lukechang/Dropbox/Dartbrains/Data/preproc participant --participant_label sub-S01 --write-graph --fs-no-reconall --notrack --fs-license-file ~/Dropbox/Dartbrains/License/license.txt --work-dir /Users/lukechang/Dropbox/Dartbrains/Data/work`

    In practice, it's alway a little bit finicky to get everything set up on a particular system. Sometimes you might run into issues with a specific missing file like the [freesurfer license](https://fmriprep.readthedocs.io/en/stable/usage.html#the-freesurfer-license) even if you're not using it. You might also run into issues with the format of the data that might have some conflicts with the [bids-validator](https://github.com/bids-standard/bids-validator). In our experience, there is always some frustrations getting this to work, but it's very nice once it's done.
    """)
    return


if __name__ == "__main__":
    app.run()
