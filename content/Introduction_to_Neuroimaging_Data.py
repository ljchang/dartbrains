import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from pathlib import Path
    import os
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Code.data import get_file, get_subjects, get_tr, load_events, load_confounds, REPO_ID, CONDITIONS
    from huggingface_hub import hf_hub_download
    import nibabel as nib
    import matplotlib.pyplot as plt
    from nilearn.plotting import view_img, plot_glass_brain, plot_anat, plot_epi, plot_stat_map
    from nltools.data import Brain_Data
    from nltools.utils import get_anatomical

    IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "brain_data"
    return (
        Brain_Data,
        IMG_DIR,
        get_anatomical,
        get_file,
        get_subjects,
        load_events,
        mo,
        nib,
        plot_anat,
        plot_glass_brain,
        plot_stat_map,
        plt,
        view_img,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to Neuroimaging Data

    In this tutorial we will learn the basics of the organization of data folders, and how to load, plot, and manipulate neuroimaging data in Python.

    To introduce the basics of fMRI data structures, watch this short video by Martin Lindquist.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Interactive version:** [Open this notebook in molab](https://molab.marimo.io/github/ljchang/dartbrains/blob/v2-marimo-migration/content/Introduction_to_Neuroimaging_Data.py) to run code, interact with widgets, and modify examples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""<iframe
          width="560" height="315"
          src="https://www.youtube.com/embed/OuRdQJMU5ro"
          frameborder="0" allowfullscreen>
      </iframe>
      """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Software Packages
    There are many different software packages to analyze neuroimaging data. Most of them are open source and free to use (with the exception of [BrainVoyager](https://www.brainvoyager.com/)). The most popular ones ([SPM](https://www.fil.ion.ucl.ac.uk/spm/), [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki), & [AFNI](https://afni.nimh.nih.gov/)) have been around a long time and are where many new methods are developed and distributed. These packages have focused on implementing what they believe are the best statistical methods, ease of use, and computational efficiency. They have very large user bases so many bugs have been identified and fixed over the years. There are also lots of publicly available documentation, listserves, and online tutorials, which makes it very easy to get started using these tools.

    There are also many more boutique packages that focus on specific types of preprocessing step and analyses such as spatial normalization with [ANTs](http://stnava.github.io/ANTs/), connectivity analyses with the [conn-toolbox](https://web.conn-toolbox.org/), representational similarity analyses with the [rsaToolbox](https://github.com/rsagroup/rsatoolbox), and prediction/classification with [pyMVPA](http://www.pymvpa.org/).

    Many packages have been developed within proprietary software such as [Matlab](https://www.mathworks.com/products/matlab.html) (e.g., SPM, Conn, RSAToolbox, etc). Unfortunately, this requires that your university has site license for Matlab and many individual add-on toolboxes. If you are not affiliated with a University, you may have to pay for Matlab, which can be fairly expensive. There are free alternatives such as [octave](https://www.gnu.org/software/octave/), but octave does not include many of the add-on toolboxes offered by matlab that may be required for a specific package. Because of this restrictive licensing, it is difficult to run matlab on cloud computing servers and to use with free online courses such as dartbrains. Other packages have been written in C/C++/C# and need to be compiled to run on your specific computer and operating system. While these tools are typically highly computationally efficient, it can sometimes be challenging to get them to install and work on specific computers and operating systems.

    There has been a growing trend to adopt the open source Python framework in the data science and scientific computing communities, which has lead to an explosion in the number of new packages available for statistics, visualization, machine learning, and web development. [pyMVPA](http://www.pymvpa.org/) was an early leader in this trend, and there are many great tools that are being actively developed such as [nilearn](https://nilearn.github.io/), [brainiak](https://brainiak.org/), [neurosynth](https://github.com/neurosynth/neurosynth), [nipype](https://nipype.readthedocs.io/en/latest/), [fmriprep](https://fmriprep.readthedocs.io/en/stable/), and many more. One exciting thing is that these newer developments have built on the expertise of decades of experience with imaging analyses, and leverage changes in high performance computing. There is also a very tight integration with many cutting edge developments in adjacent communities such as machine learning with [scikit-learn](https://scikit-learn.org/stable/), [tensorflow](https://www.tensorflow.org/), and [pytorch](https://pytorch.org/), which has made new types of analyses much more accessible to the neuroimaging community. There has also been an influx of younger contributors with software development expertise. You might be surprised to know that many of the popular tools being used had core contributors originating from the neuroimaging community (e.g., scikit-learn, seaborn, and many more).

    For this course, I have chosen to focus on tools developed in Python as it is an easy to learn programming language, has excellent tools, works well on distributed computing systems, has great ways to disseminate information (e.g., jupyter notebooks, jupyter-book, etc), and is free! If you are just getting started, I would spend some time working with [NiLearn](https://nilearn.github.io/) and [Brainiak](https://brainiak.org/), which have a lot of functionality, are very well tested, are reasonably computationally efficient, and most importantly have lots of documentation and tutorials to get started.

    We will be using many packages throughout the course such as [fmriprep](https://fmriprep.readthedocs.io/en/stable/) to perform preprocessing, and [nltools](https://nltools.org/), which is a package developed in my lab, to do basic data manipulation and analysis. NLtools is built using many other toolboxes such as [nibabel](https://nipy.org/nibabel/) and [nilearn](https://nilearn.github.io/), and we will also be using these frequently throughout the course.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BIDS: Brain Imaging Dataset Specification

    Recently, there has been growing interest to share datasets across labs and even on public repositories such as [openneuro](https://openneuro.org/). In order to make this a successful enterprise, it is necessary to have some standards in how the data are named and organized. Historically, each lab has used their own idiosyncratic conventions, which can make it difficult for outsiders to analyze. In the past few years, there have been heroic efforts by the neuroimaging community to create a standardized file organization and naming practices. This specification is called **BIDS** for [Brain Imaging Dataset Specification](http://bids.neuroimaging.io/).

    As you can imagine, individuals have their own distinct method of organizing their files. Think about how you keep track of your files on your personal laptop (versus your friend). This may be okay in the personal realm, but in science, it's best if anyone (especially yourself 6 months from now!) can follow your work and know *which* files mean *what* by their titles.

    Our course dataset — the [dartbrains/localizer](https://huggingface.co/datasets/dartbrains/localizer) dataset on HuggingFace — follows the BIDS layout. Here's the top-level structure of the raw side:

    ```
    localizer/
    ├── dataset_description.json     # dataset name, BIDS version, authors
    ├── participants.tsv             # one row per subject (age, sex, …)
    ├── participants.json            # column descriptions for participants.tsv
    ├── task-localizer_bold.json     # task-level acquisition params (TR, slice timing, …)
    ├── README.md
    ├── sub-S01/
    │   ├── anat/
    │   │   └── metadata.csv
    │   └── func/
    │       ├── sub-S01_task-localizer_events.tsv   # stimulus onsets, durations, conditions
    │       └── metadata.csv
    ├── sub-S02/ …
    ├── sub-S20/
    └── derivatives/                 # processed outputs (see next section)
    ```

    A few things to notice:

    1. **Files are in NIfTI format**, not raw DICOMs. (In this dataset the raw `.nii.gz` files aren't hosted to keep the download small — only the `events.tsv` per subject lives under raw, with the preprocessed scans available under `derivatives/`. A complete BIDS dataset would include `sub-S01/anat/sub-S01_T1w.nii.gz` and `sub-S01/func/sub-S01_task-localizer_bold.nii.gz` here.)
    2. **Scans are broken up by modality** — `anat/`, `func/`, `dwi/`, `fmap/` — for each subject.
    3. **Filenames carry metadata** as `key-value` *entities* separated by underscores: `sub-S01_task-localizer_events.tsv` tells you the subject, task, and content type at a glance.
    4. **Sidecar JSON files** describe acquisition parameters in a machine-readable format (echo time, slice timing, phase encoding direction, …), either alongside each scan or "inherited" from a top-level file like `task-localizer_bold.json`.

    Not only does this specification standardize within labs, it also makes collaboration, software development, and data publishing dramatically easier. Because the format is consistent, tools like [pybids](https://github.com/bids-standard/pybids) can programmatically index and query an entire BIDS directory. In this course, we use lightweight helper functions in `Code.data` that download individual files on demand from HuggingFace Hub.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The `derivatives/` folder

    BIDS makes a strict separation between **raw data** (what came off the scanner) and **derivatives** (anything produced by running a pipeline on that raw data). Derived files live in a sibling `derivatives/` directory, with one subfolder per pipeline. Here's the actual layout for our dataset:

    ```
    localizer/derivatives/
    ├── fmriprep/
    │   ├── dataset_description.json
    │   ├── sub-S01.html             # per-subject QC report
    │   ├── sub-S01/
    │   │   ├── anat/
    │   │   │   ├── sub-S01_desc-preproc_T1w.nii.gz                        # T1 in native space
    │   │   │   ├── sub-S01_desc-brain_mask.nii.gz                         # brain mask, native
    │   │   │   ├── sub-S01_dseg.nii.gz                                    # tissue segmentation
    │   │   │   ├── sub-S01_label-{GM,WM,CSF}_probseg.nii.gz               # tissue probabilities
    │   │   │   ├── sub-S01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5   # forward transform
    │   │   │   ├── sub-S01_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5   # inverse transform
    │   │   │   ├── sub-S01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz   # T1 in MNI space
    │   │   │   └── sub-S01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
    │   │   ├── func/
    │   │   │   ├── sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
    │   │   │   ├── sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
    │   │   │   ├── sub-S01_task-localizer_space-MNI152NLin2009cAsym_boldref.nii.gz
    │   │   │   └── sub-S01_task-localizer_desc-confounds_regressors.tsv  # motion + physio regressors
    │   │   └── figures/             # QC SVGs (carpetplot, flirtbbr, dseg, …)
    │   ├── sub-S02/ …
    │   └── logs/CITATION.{bib,html,md,tex}
    └── betas/                       # condition-level GLM estimates
        ├── S01_beta_audio_computation.nii.gz
        ├── S01_beta_audio_left_hand.nii.gz
        │   …  (10 conditions per subject)
        ├── S01_betas.nii.gz         # stacked 4D image (10 conditions)
        ├── S02_beta_…
        └── …
    ```

    Each pipeline gets its own subfolder under `derivatives/` (here: `fmriprep/` for preprocessing and `betas/` for our first-level GLM outputs; other common ones are `freesurfer/`, `mriqc/`, `xcp_d/`). This means you can run multiple pipelines on the same dataset without them colliding, and deleting and re-running a pipeline never risks the raw data.

    Derivative files follow BIDS naming conventions but add **entities** that describe the processing variant. The most important ones to recognize:

    - `desc-` describes *what kind of derivative* — `desc-preproc_bold` is the preprocessed BOLD timeseries; `desc-brain_mask` is a brain mask; `desc-confounds_regressors` is the confounds TSV.
    - `space-` identifies the *coordinate space* — `space-MNI152NLin2009cAsym` means the file has been warped into the MNI152 nonlinear 2009c asymmetric template; absence of `space-` means native subject space.
    - `from-`/`to-` on `xfm.h5` files describe the *direction of a transform* (T1w → MNI for forward warps, MNI → T1w for inverse).
    - `label-` distinguishes *tissue classes* on segmentation outputs (GM, WM, CSF).

    These conventions keep filenames self-describing: `sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz` tells you it's subject S01's localizer task, preprocessed and resampled into MNI space — without opening the file.

    In this course, our `Code.data.get_file()` helper takes a `scope` argument that distinguishes raw from derivative data: `scope='raw'` pulls from `sub-S01/`, `scope='derivatives'` pulls from `derivatives/fmriprep/sub-S01/`, and `scope='betas'` pulls from `derivatives/betas/`. The helper downloads on demand from HuggingFace and caches locally, so you don't need the full directory structure on disk.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Accessing the Dataset

    The Localizer dataset is hosted on [HuggingFace](https://huggingface.co/datasets/dartbrains/localizer) in BIDS format. We provide helper functions in `Code.data` that download files on demand and cache them locally:

    ```python
    from Code.data import get_file, get_subjects, load_events

    # Get the preprocessed BOLD file for subject S01
    bold_path = get_file('S01', 'derivatives', 'bold')

    # Get a list of all subjects
    subjects = get_subjects()  # ['S01', 'S02', ..., 'S20']

    # Load event timing for a subject
    events = load_events('S01')
    ```

    Files are downloaded from HuggingFace Hub the first time you request them and cached locally for subsequent use.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With a BIDS dataset, we often want to know which subjects are available, and retrieve specific files by subject, data type, and scope (raw vs. derivatives). Let's start by listing the subjects in the dataset.
    """)
    return


@app.cell
def _(get_subjects):
    subjects = get_subjects()
    subjects[:10]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also retrieve the path to a specific file. For example, let's get the preprocessed BOLD file for the first 10 subjects. The `get_file` function downloads the file from HuggingFace Hub on first access and returns the local cached path.
    """)
    return


@app.cell
def _(get_file, get_subjects):
    bold_files = [get_file(sub, 'derivatives', 'bold') for sub in get_subjects()[:10]]
    bold_files
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In a BIDS dataset, each file follows a structured naming convention. For example, a preprocessed BOLD file is named:

    `sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`

    The key-value pairs (`sub-S01`, `task-localizer`, `space-...`, `desc-preproc`) are called **entities** and they encode metadata directly in the filename. This is one of the core design principles of BIDS: you can understand what a file contains just by reading its name.

    Common BIDS entities include:
    - `sub-<label>`: Subject identifier
    - `task-<label>`: Task name
    - `space-<label>`: Reference space (e.g., MNI152NLin2009cAsym)
    - `desc-<label>`: Description (e.g., preproc for preprocessed)
    - `suffix`: The type of data (bold, T1w, events, etc.)

    Let's look at the path for a single file to see this structure.
    """)
    return


@app.cell
def _(get_file):
    f = get_file('S01', 'derivatives', 'bold')
    f
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This dataset contains a single task called *localizer*. Look at the [Download Data](Download_Data.ipynb) page for more information about this task.

    We can also retrieve event files that describe the experimental conditions and their timing. Let's load the events for the first subject.
    """)
    return


@app.cell
def _(load_events):
    events_df = load_events('S01')
    events_df.head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading Data with Nibabel
    Neuroimaging data is often stored in the format of nifti files `.nii` which can also be compressed using gzip `.nii.gz`.  These files store both 3D and 4D data and also contain structured metadata in the image **header**.

    There is a very nice tool to access nifti data stored on your file system in python called [nibabel](http://nipy.org/nibabel/).  If you don't already have nibabel installed on your computer it is easy via `pip`. First, tell the jupyter cell that you would like to access the unix system outside of the notebook and then install nibabel using pip `!pip install nibabel`. You only need to run this once (unless you would like to update the version).

    nibabel objects can be initialized by simply pointing to a nifti file even if it is compressed through gzip.  First, we will import the nibabel module as `nib` (short and sweet so that we don't have to type so much when using the tool).  I'm also including a path to where the data file is located so that I don't have to constantly type this.  It is easy to change this on your own computer.

    We will be loading an anatomical image from subject S01 from the localizer [dataset](../content/Download_Data).  See this [paper](https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-8-91) for more information about this dataset.

    We will use our `get_file` helper to grab subject S01's T1 image.
    """)
    return


@app.cell
def _(get_file, nib):
    data = nib.load(get_file('S01', 'derivatives', 'T1w'))
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we want to get more help on how to work with the nibabel data object we can either consult the [documentation](https://nipy.org/nibabel/tutorials.html#tutorials) or add a `?`.
    """)
    return


@app.cell
def _(data):
    help(data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The imaging data is stored in either a 3D or 4D numpy array. Just like numpy, it is easy to get the dimensions of the data using `shape`.
    """)
    return


@app.cell
def _(data):
    data.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looks like there are 3 dimensions (x,y,z) that is the number of voxels in each dimension. If we know the voxel size, we could convert this into millimeters.

    We can also directly access the data and plot a single slice using standard matplotlib functions.
    """)
    return


@app.cell
def _(data, plt):

    plt.imshow(data.get_fdata()[:,:,50], cmap='RdBu_r')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try slicing different dimensions (x,y,z) yourself to get a feel for how the data is represented in this anatomical image.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also access data from the image header. Let's assign the header of an image to a variable and print it to view it's contents.
    """)
    return


@app.cell
def _(data):
    header = data.header
    print(header)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Some of the important information in the header is information about the orientation of the image in space. This can be represented as the affine matrix, which can be used to transform images between different spaces.
    """)
    return


@app.cell
def _(data):
    data.affine
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will dive deeper into affine transformations in the preprocessing tutorial.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting Data with Nilearn
    There are many useful tools from the [nilearn](https://nilearn.github.io/index.html) library to help manipulate and visualize neuroimaging data. See their [documentation](https://nilearn.github.io/dev/plotting/index.html#different-plotting-functions) for an example.

    In this section, we will explore a few of their different plotting functions, which can work directly with nibabel instances.

    ### A note on displaying plots in marimo

    Marimo renders the **last expression** of each cell. Unlike Jupyter, it
    doesn't automatically call `_repr_html_` on opaque return objects, so
    `plot_anat(data)` or `plt.imshow(...)` alone will just print the object's
    repr string (e.g. `<OrthoSlicer object at 0x...>`) instead of a figure.

    No `%matplotlib inline` equivalent is needed — marimo always renders real
    `Figure` objects inline. You just have to make sure the last line of the
    cell **is** a Figure (or an HTML component). Three patterns, in order of
    recommendation:

    **1. Create the figure, pass it in, return it** *(preferred)*

    ```python
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_anat(data, axes=ax)
    fig
    ```

    - Explicit figure handle — safe across reactive re-runs
    - You control size, DPI, subplot layout
    - A couple extra lines per cell

    **2. Call the plotting function, then plt.gcf() (quick fix)**

    ```python
    plot_anat(data)
    plt.gcf()
    ```

    - One-line escape hatch — easy to retrofit
    - gcf() returns whichever figure pyplot touched most recently, which
    can be surprising when cells re-execute out of order in a reactive
    notebook
    - No control over figure dimensions

    **3. mo.Html(view.get_iframe()) for nilearn interactive views**

    ```python
    mo.Html(view_img(data).get_iframe())
    ```

    Used for view_img, view_connectome, view_surf — these return a
    nilearn HTMLDocument with embedded JavaScript. The get_iframe() call
    sandboxes the viewer into its own iframe so its JS doesn't collide with
    marimo's. Use .html instead of .get_iframe() if you want it inline in
    the main DOM and are sure there are no JS conflicts.
    """)
    return


@app.cell
def _(data, plot_anat, plt):
    _fig, _ax = plt.subplots(figsize=(12, 4))
    plot_anat(data, axes=_ax)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nilearn plotting functions are very flexible and allow us to easily customize our plots
    """)
    return


@app.cell
def _(data, plot_anat, plt):
    plot_anat(data, draw_cross=False, display_mode='z')
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    try to get more information how to use the function with `?` and try to add different commands to change the plot.

    nilearn also has a neat interactive viewer called `view_img` for examining images directly in the notebook.
    """)
    return


@app.cell
def _(data, view_img):
    view_img(data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `view_img` function is particularly useful for overlaying statistical maps over an anatomical image so that we can interactively examine where the results are located.

    As an example, let's load a mask of the amygdala and try to find where it is located. We will download it from [Neurovault](https://neurovault.org/images/18632/) using a function from `nltools`.
    """)
    return


@app.cell
def _(Brain_Data, data, view_img):
    amygdala_mask = Brain_Data('https://neurovault.org/media/images/1290/FSL_BAmyg_thr0.nii.gz').to_nifti()

    view_img(amygdala_mask, data)
    return (amygdala_mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also plot a glass brain which allows us to see through the brain from different slice orientations. In this example, we will plot the binary amygdala mask.
    """)
    return


@app.cell
def _(amygdala_mask, plot_glass_brain, plt):
    plot_glass_brain(amygdala_mask)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manipulating Data with Nltools
    Ok, we've now learned how to use nibabel to load imaging data and nilearn to plot it.

    Next we are going to learn how to use the `nltools` package that tries to make loading, plotting, and manipulating data easier. It uses many functions from nibabel, nilearn, and other python libraries. The bulk of the nltools toolbox is built around the `Brain_Data()` class. The concept behind the class is to have a similar feel to a pandas dataframe, which means that it should feel intuitive to manipulate the data.

    The `Brain_Data()` class has several attributes that may be helpful to know about. First, it stores imaging data in `.data` as a vectorized features by observations matrix. Each image is an observation and each voxel is a feature. Space is flattened using `nifti_masker` from nilearn. This object is also stored as an attribute in `.nifti_masker` to allow transformations from 2D to 3D/4D matrices. In addition, a brain_mask is stored in `.mask`. Finally, there are attributes to store either class labels for prediction/classification analyses in `.Y` and design matrices in `.X`. These are both expected to be pandas `DataFrames`.

    We will give a quick overview of basic Brain_Data operations, but we encourage you to see our [documentation](https://nltools.org/) for more details.

    ### Brain_Data basics
    To get a feel for `Brain_Data`, let's load an example anatomical overlay image that comes packaged with the toolbox.
    """)
    return


@app.cell
def _(Brain_Data, get_anatomical):
    anat = Brain_Data(get_anatomical())
    anat
    return (anat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To view the attributes of `Brain_Data` use the `vars()` function.
    """)
    return


@app.cell
def _(anat):
    print(vars(anat))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Brain_Data` has many methods to help manipulate, plot, and analyze imaging data. We can use the `dir()` function to get a quick list of all of the available methods that can be used on this class.

    To learn more about how to use these tools either use the `?` function, or look up the function in the [api documentation](https://nltools.org/api.html).
    """)
    return


@app.cell
def _(anat):
    print(dir(anat))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ok, now let's load a single subject's functional data from the localizer dataset. We will load one that has already been preprocessed with fmriprep and is stored in the derivatives folder.

    Loading data can be a little bit slow especially if the data need to be resampled to the template, which is set at $2mm^3$ by default. However, once it's loaded into the workspace it should be relatively fast to work with it.
    """)
    return


@app.cell
def _(Brain_Data, get_file):
    data_1 = Brain_Data(get_file('S01', 'derivatives', 'bold'))
    return (data_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here are a few quick basic data operations.

    Find number of images in Brain_Data() instance
    """)
    return


@app.cell
def _(data_1):
    print(len(data_1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Find the dimensions of the data (images x voxels)
    """)
    return


@app.cell
def _(data_1):
    print(data_1.shape())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use any type of indexing to slice the data such as integers, lists of integers, slices, or boolean vectors.
    """)
    return


@app.cell
def _(data_1):
    import numpy as np
    print(data_1[5].shape())
    print(data_1[[1, 6, 2]].shape())
    print(data_1[0:10].shape())
    index = np.zeros(len(data_1), dtype=bool)
    index[[1, 5, 9, 16, 20, 22]] = True
    print(data_1[index].shape())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simple Arithmetic Operations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calculate the mean for every voxel over images
    """)
    return


@app.cell
def _(data_1):
    data_1.mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calculate the standard deviation for every voxel over images
    """)
    return


@app.cell
def _(data_1):
    data_1.std()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Methods can be chained.  Here we get the shape of the mean.
    """)
    return


@app.cell
def _(data_1):
    print(data_1.mean().shape())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Brain_Data instances can be added and subtracted
    """)
    return


@app.cell
def _(data_1):
    new = data_1[1] + data_1[2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Brain_Data instances can be manipulated with basic arithmetic operations.

    Here we add 10 to every voxel and scale by 2
    """)
    return


@app.cell
def _(data_1):
    data2 = (data_1 + 10) * 2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Brain_Data instances can be copied
    """)
    return


@app.cell
def _(data_1):
    new_1 = data_1.copy()
    return (new_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Brain_Data instances can be easily converted to nibabel instances, which store the data in a 3D/4D matrix.  This is useful for interfacing with other python toolboxes such as [nilearn](http://nilearn.github.io)
    """)
    return


@app.cell
def _(data_1):
    data_1.to_nifti()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Brain_Data instances can be concatenated using the append method
    """)
    return


@app.cell
def _(data_1, new_1):
    new_2 = new_1.append(data_1[4])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lists of `Brain_Data` instances can also be concatenated by recasting as a `Brain_Data` object.
    """)
    return


@app.cell
def _(Brain_Data, data_1):
    print(type([x for x in data_1[:4]]))
    type(Brain_Data([x for x in data_1[:4]]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Any Brain_Data object can be written out to a nifti file.
    """)
    return


@app.cell
def _(data_1):
    data_1.write('Tmp_Data.nii.gz')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Images within a Brain_Data() instance are iterable.  Here we use a list comprehension to calculate the overall mean across all voxels within an image.
    """)
    return


@app.cell
def _(data_1):
    [x.mean() for x in data_1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Though, we could also do this with the `mean` method by setting `axis=1`.
    """)
    return


@app.cell
def _(data_1):
    data_1.mean(axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's plot the mean to see how the global signal changes over time.
    """)
    return


@app.cell
def _(data_1, plt):
    plt.plot(data_1.mean(axis=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the slow linear drift over time, where the global signal intensity gradually decreases. We will learn how to remove this with a high pass filter in future tutorials.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting
    There are multiple ways to plot your data.

    For a very quick plot, you can return a montage of axial slices with the `.plot()` method. As an example, we will plot the mean of each voxel over time.
    """)
    return


@app.cell
def _(data_1, plot_stat_map, plt):
    f_2 = plot_stat_map(data_1.mean().to_nifti())
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There is an interactive `.iplot()` method based on nilearn `view_img`.
    """)
    return


@app.cell
def _(data_1):
    data_1.mean().iplot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Brain_Data() instances can be converted to a nibabel instance and plotted using any nilearn plot method such as glass brain.
    """)
    return


@app.cell
def _(data_1, plot_glass_brain, plt):
    plot_glass_brain(data_1.mean().to_nifti())
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ok, that's the basics. `Brain_Data` can do much more!

    Check out some of our [tutorials](https://nltools.org/auto_examples/index.html) for more detailed examples.

    We'll be using this tool throughout the course.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    For homework, let's practice our skills in working with data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1
    A few subjects have already been preprocessed with fMRI prep.

    Use `get_subjects()` to figure out which subjects are available in the dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2

    One question we are often interested in is where in the brain do we have an adequate signal to noise ratio (SNR). There are many different metrics, here we will use temporal SNR, which the voxel mean over time divided by it's standard deviation.

    $$\text{tSNR} = \frac{\text{mean}(\text{voxel}_{i})}{\text{std}(\text{voxel}_i)}$$

    In Exercise 2, calculate the SNR for S01 and plot this so we can figure which regions have high and low SNR.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3

    We are often interested in identifying outliers in our data. In this exercise, find any image that is outside 95% of all images based on global intensity (i.e., zscore greater than 2) from 'S01' and plot each one.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
