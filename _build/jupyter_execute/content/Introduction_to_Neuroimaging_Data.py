#!/usr/bin/env python
# coding: utf-8

# # Introduction to Neuroimaging Data 
# 
# In this tutorial we will learn the basics of the organization of data folders, and how to load, plot, and manipulate neuroimaging data in Python.
# 
# To introduce the basics of fMRI data structures, watch this short video by Martin Lindquist.

# In[1]:


from IPython.display import YouTubeVideo

YouTubeVideo('OuRdQJMU5ro')


# ## Software Packages
# There are many different software packages to analyze neuroimaging data. Most of them are open source and free to use (with the exception of [BrainVoyager](https://www.brainvoyager.com/)). The most popular ones ([SPM](https://www.fil.ion.ucl.ac.uk/spm/), [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki), & [AFNI](https://afni.nimh.nih.gov/)) have been around a long time and are where many new methods are developed and distributed. These packages have focused on implementing what they believe are the best statistical methods, ease of use, and computational efficiency. They have very large user bases so many bugs have been identified and fixed over the years. There are also lots of publicly available documentation, listserves, and online tutorials, which makes it very easy to get started using these tools.
# 
# There are also many more boutique packages that focus on specific types of preprocessing step and analyses such as spatial normalization with [ANTs](http://stnava.github.io/ANTs/), connectivity analyses with the [conn-toolbox](https://web.conn-toolbox.org/), representational similarity analyses with the [rsaToolbox](https://github.com/rsagroup/rsatoolbox), and prediction/classification with [pyMVPA](http://www.pymvpa.org/).
# 
# Many packages have been developed within proprietary software such as [Matlab](https://www.mathworks.com/products/matlab.html) (e.g., SPM, Conn, RSAToolbox, etc). Unfortunately, this requires that your university has site license for Matlab and many individual add-on toolboxes. If you are not affiliated with a University, you may have to pay for Matlab, which can be fairly expensive. There are free alternatives such as [octave](https://www.gnu.org/software/octave/), but octave does not include many of the add-on toolboxes offered by matlab that may be required for a specific package. Because of this restrictive licensing, it is difficult to run matlab on cloud computing servers and to use with free online courses such as dartbrains. Other packages have been written in C/C++/C# and need to be compiled to run on your specific computer and operating system. While these tools are typically highly computationally efficient, it can sometimes be challenging to get them to install and work on specific computers and operating systems.
# 
# There has been a growing trend to adopt the open source Python framework in the data science and scientific computing communities, which has lead to an explosion in the number of new packages available for statistics, visualization, machine learning, and web development. [pyMVPA](http://www.pymvpa.org/) was an early leader in this trend, and there are many great tools that are being actively developed such as [nilearn](https://nilearn.github.io/), [brainiak](https://brainiak.org/), [neurosynth](https://github.com/neurosynth/neurosynth), [nipype](https://nipype.readthedocs.io/en/latest/), [fmriprep](https://fmriprep.readthedocs.io/en/stable/), and many more. One exciting thing is that these newer developments have built on the expertise of decades of experience with imaging analyses, and leverage changes in high performance computing. There is also a very tight integration with many cutting edge developments in adjacent communities such as machine learning with [scikit-learn](https://scikit-learn.org/stable/), [tensorflow](https://www.tensorflow.org/), and [pytorch](https://pytorch.org/), which has made new types of analyses much more accessible to the neuroimaging community. There has also been an influx of younger contributors with software development expertise. You might be surprised to know that many of the popular tools being used had core contributors originating from the neuroimaging community (e.g., scikit-learn, seaborn, and many more).
# 
# For this course, I have chosen to focus on tools developed in Python as it is an easy to learn programming language, has excellent tools, works well on distributed computing systems, has great ways to disseminate information (e.g., jupyter notebooks, jupyter-book, etc), and is free! If you are just getting started, I would spend some time working with [NiLearn](https://nilearn.github.io/) and [Brainiak](https://brainiak.org/), which have a lot of functionality, are very well tested, are reasonably computationally efficient, and most importantly have lots of documentation and tutorials to get started.
# 
# We will be using many packages throughout the course such as [PyBids](https://bids-standard.github.io/pybids/) to navigate neuroimaging datasets, [fmriprep](https://fmriprep.readthedocs.io/en/stable/) to perform preprocessing, and [nltools](https://nltools.org/), which is a package developed in my lab, to do basic data manipulation and analysis. NLtools is built using many other toolboxes such as [nibabel](https://nipy.org/nibabel/) and [nilearn](https://nilearn.github.io/), and we will also be using these frequently throughout the course.

# ## BIDS: Brain Imaging Dataset Specification
# 
# Recently, there has been growing interest to share datasets across labs and even on public repositories such as [openneuro](https://openneuro.org/). In order to make this a successful enterprise, it is necessary to have some standards in how the data are named and organized. Historically, each lab has used their own idiosyncratic conventions, which can make it difficult for outsiders to analyze. In the past few years, there have been heroic efforts by the neuroimaging community to create a standardized file organization and naming practices. This specification is called **BIDS** for [Brain Imaging Dataset Specification](http://bids.neuroimaging.io/).
# 
# As you can imagine, individuals have their own distinct method of organizing their files. Think about how you keep track of your files on your personal laptop (versus your friend). This may be okay in the personal realm, but in science, it's best if anyone (especially  yourself 6 months from now!) can follow your work and know *which* files mean *what* by their titles.
# 
# Here's an example of non-Bids versus BIDS dataset found in [this paper](https://www.nature.com/articles/sdata201644):
# 
# ![file_tree](../images/brain_data/file_tree.jpg)
# 
# Here are a few major differences between the two datasets:
# 
# 1. In BIDS, files are in nifti format (not dicoms).
# 2. In BIDS, scans are broken up into separate folders by type of scan(functional versus anatomical versus diffusion weighted) for each subject.
# 3. In BIDS, JSON files are included that contain descriptive information about the scans (e.g., acquisition parameters)
# 
# Not only can using this specification be useful within labs to have a set way of structuring data, but it can also be useful when collaborating across labs, developing and utilizing software, and publishing data.
# 
# In addition, because this is a consistent format, it is possible to have a python package to make it easy to query a dataset. We recommend using [pybids](https://github.com/bids-standard/pybids).
# 
# The dataset we will be working with has already been converted to the BIDS format (see download localizer tutorial). 
# 
# You may need to install [pybids]() to query the BIDS datasets using following command `!pip install pybids`.

# ### The `BIDSLayout`
# [Pybids](https://github.com/bids-standard/pybids) is a package to help query and navigate a neuroimaging dataset that is in the BIDs format. At the core of pybids is the `BIDSLayout` object. A `BIDSLayout` is a lightweight Python class that represents a BIDS project file tree and provides a variety of helpful methods for querying and manipulating BIDS files. While the BIDSLayout initializer has a large number of arguments you can use to control the way files are indexed and accessed, you will most commonly initialize a BIDSLayout by passing in the BIDS dataset root location as a single argument.
# 
# Notice we are setting `derivatives=True`. This means the layout will also index the derivatives sub folder, which might contain preprocessed data, analyses, or other user generated files. 

# In[11]:


from bids import BIDSLayout, BIDSValidator
import os

data_dir = '../data/localizer'
layout = BIDSLayout(data_dir, derivatives=True)
layout


# When we initialize a BIDSLayout, all of the files and metadata found under the specified root folder are indexed. This can take a few seconds (or, for very large datasets, a minute or two). Once initialization is complete, we can start querying the BIDSLayout in various ways. The main query method is `.get()`. If we call .`get()` with no additional arguments, we get back a list of all the BIDS files in our dataset.
# 
# Let's return the first 10 files

# In[12]:


layout.get()[:10]


# As you can see, just a generic `.get()` call gives us *all* of the files. We will definitely want to be a bit more specific. We can specify the type of data we would like to query. For example, suppose we want to return the first 10 subject ids.

# In[14]:


layout.get(target='subject', return_type='id', scope='derivatives')[:10]


# Or perhaps, we would like to get the file names for the raw bold functional nifti images for the first 10 subjects. We can filter files in the `raw` or `derivatives`, using `scope` keyword.`scope='raw'`, to only query raw bold nifti files.

# In[15]:


layout.get(target='subject', scope='raw', suffix='bold', return_type='file')[:10]


# When you call .get() on a BIDSLayout, the default returned values are objects of class BIDSFile. A BIDSFile is a lightweight container for individual files in a BIDS dataset. 
# 
# Here are some of the attributes and methods available to us in a BIDSFile (note that some of these are only available for certain subclasses of BIDSFile; e.g., you can't call get_image() on a BIDSFile that doesn't correspond to an image file!):
# 
# - .path: The full path of the associated file
# - .filename: The associated file's filename (without directory)
# - .dirname: The directory containing the file
# - .get_entities(): Returns information about entities associated with this BIDSFile (optionally including metadata)
# - .get_image(): Returns the file contents as a nibabel image (only works for image files)
# - .get_df(): Get file contents as a pandas DataFrame (only works for TSV files)
# - .get_metadata(): Returns a dictionary of all metadata found in associated JSON files
# - .get_associations(): Returns a list of all files associated with this one in some way
# 
# Let's explore the first file in a little more detail.

# In[16]:


f = layout.get()[0]
f


# If we wanted to get the path of the file, we can use `.path`.

# In[17]:


f.path


# Suppose we were interested in getting a list of tasks included in the dataset.

# In[18]:


layout.get_task()


# We can query all of the files associated with this task.

# In[19]:


layout.get(task='localizer', suffix='bold', scope='raw')[:10]


# Notice that there are nifti and event files. We can get the filename for the first particant's functional run

# In[140]:


f = layout.get(task='localizer')[0].filename
f


# If you want a summary of all the files in your BIDSLayout, but don't want to have to iterate BIDSFile objects and extract their entities, you can get a nice bird's-eye view of your dataset using the `to_df()` method.

# In[20]:


layout.to_df()


# ## Loading Data with Nibabel
# Neuroimaging data is often stored in the format of nifti files `.nii` which can also be compressed using gzip `.nii.gz`.  These files store both 3D and 4D data and also contain structured metadata in the image **header**.
# 
# There is an very nice tool to access nifti data stored on your file system in python called [nibabel](http://nipy.org/nibabel/).  If you don't already have nibabel installed on your computer it is easy via `pip`. First, tell the jupyter cell that you would like to access the unix system outside of the notebook and then install nibabel using pip `!pip install nibabel`. You only need to run this once (unless you would like to update the version).
# 
# nibabel objects can be initialized by simply pointing to a nifti file even if it is compressed through gzip.  First, we will import the nibabel module as `nib` (short and sweet so that we don't have to type so much when using the tool).  I'm also including a path to where the data file is located so that I don't have to constantly type this.  It is easy to change this on your own computer.
# 
# We will be loading an anatomical image from subject S01 from the localizer [dataset](../content/Download_Data).  See this [paper](https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-8-91) for more information about this dataset.
# 
# We will use pybids to grab subject S01's T1 image.

# In[26]:


import nibabel as nib

data = nib.load(layout.get(subject='S01', scope='derivatives', suffix='T1w', return_type='file', extension='nii.gz')[1])


# If we want to get more help on how to work with the nibabel data object we can either consult the [documentation](https://nipy.org/nibabel/tutorials.html#tutorials) or add a `?`.

# In[27]:


get_ipython().run_line_magic('pinfo', 'data')


# The imaging data is stored in either a 3D or 4D numpy array. Just like numpy, it is easy to get the dimensions of the data using `shape`. 

# In[28]:


data.shape


# Looks like there are 3 dimensions (x,y,z) that is the number of voxels in each dimension. If we know the voxel size, we could convert this into millimeters.
# 
# We can also directly access the data and plot a single slice using standard matplotlib functions.

# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

plt.imshow(data.get_fdata()[:,:,50])


# Try slicing different dimensions (x,y,z) yourself to get a feel for how the data is represented in this anatomical image.

# In[ ]:





# We can also access data from the image header. Let's assign the header of an image to a variable and print it to view it's contents.

# In[30]:


header = data.header
print(header)      


# Some of the important information in the header is information about the orientation of the image in space. This can be represented as the affine matrix, which can be used to transform images between different spaces.

# In[31]:


data.affine


# We will dive deeper into affine transformations in the preprocessing tutorial.

# ## Plotting Data with Nilearn
# There are many useful tools from the [nilearn](https://nilearn.github.io/index.html) library to help manipulate and visualize neuroimaging data. See their [documentation](https://nilearn.github.io/plotting/index.html#different-plotting-functions) for an example.
# 
# In this section, we will explore a few of their different plotting functions, which can work directly with nibabel instances.

# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')

from nilearn.plotting import view_img, plot_glass_brain, plot_anat, plot_epi


# In[33]:


plot_anat(data)


# Nilearn plotting functions are very flexible and allow us to easily customize our plots

# In[34]:


plot_anat(data, draw_cross=False, display_mode='z')


# try to get more information how to use the function with `?` and try to add different commands to change the plot.
# 
# nilearn also has a neat interactive viewer called `view_img` for examining images directly in the notebook. 

# In[35]:


view_img(data)


# The `view_img` function is particularly useful for overlaying statistical maps over an anatomical image so that we can interactively examine where the results are located.
# 
# As an example, let's load a mask of the amygdala and try to find where it is located. We will download it from [Neurovault](https://neurovault.org/images/18632/) using a function from `nltools`.

# In[41]:


from nltools.data import Brain_Data
amygdala_mask = Brain_Data('https://neurovault.org/media/images/1290/FSL_BAmyg_thr0.nii.gz').to_nifti()

view_img(amygdala_mask, data)


# We can also plot a glass brain which allows us to see through the brain from different slice orientations. In this example, we will plot the binary amygdala mask.

# In[42]:


plot_glass_brain(amygdala_mask)


# ## Manipulating Data with Nltools
# Ok, we've now learned how to use nibabel to load imaging data and nilearn to plot it.
# 
# Next we are going to learn how to use the `nltools` package that tries to make loading, plotting, and manipulating data easier. It uses many functions from nibabel, nilearn, and other python libraries. The bulk of the nltools toolbox is built around the `Brain_Data()` class. The concept behind the class is to have a similar feel to a pandas dataframe, which means that it should feel intuitive to manipulate the data.
# 
# The `Brain_Data()` class has several attributes that may be helpful to know about. First, it stores imaging data in `.data` as a vectorized features by observations matrix. Each image is an observation and each voxel is a feature. Space is flattened using `nifti_masker` from nilearn. This object is also stored as an attribute in `.nifti_masker` to allow transformations from 2D to 3D/4D matrices. In addition, a brain_mask is stored in `.mask`. Finally, there are attributes to store either class labels for prediction/classification analyses in `.Y` and design matrices in `.X`. These are both expected to be pandas `DataFrames`.
# 
# We will give a quick overview of basic Brain_Data operations, but we encourage you to see our [documentation](https://nltools.org/) for more details.
# 
# ### Brain_Data basics
# To get a feel for `Brain_Data`, let's load an example anatomical overlay image that comes packaged with the toolbox.

# In[43]:


from nltools.data import Brain_Data
from nltools.utils import get_anatomical

anat = Brain_Data(get_anatomical())
anat


# To view the attributes of `Brain_Data` use the `vars()` function.

# In[44]:


print(vars(anat))


# `Brain_Data` has many methods to help manipulate, plot, and analyze imaging data. We can use the `dir()` function to get a quick list of all of the available methods that can be used on this class.
# 
# To learn more about how to use these tools either use the `?` function, or look up the function in the [api documentation](https://nltools.org/api.html).
# 

# In[45]:


print(dir(anat))


# Ok, now let's load a single subject's functional data from the localizer dataset. We will load one that has already been preprocessed with fmriprep and is stored in the derivatives folder.
# 
# Loading data can be a little bit slow especially if the data need to be resampled to the template, which is set at $2mm^3$ by default. However, once it's loaded into the workspace it should be relatively fast to work with it.
# 

# In[48]:


sub = 'sub-S01'

data = Brain_Data(os.path.join(data_dir, 'derivatives', 'fmriprep', sub, 'func', f'{sub}_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))


# Here are a few quick basic data operations.
# 
# Find number of images in Brain_Data() instance

# In[49]:


print(len(data))


# Find the dimensions of the data (images x voxels)

# In[50]:


print(data.shape())


# We can use any type of indexing to slice the data such as integers, lists of integers, slices, or boolean vectors.

# In[51]:


import numpy as np

print(data[5].shape())

print(data[[1,6,2]].shape())

print(data[0:10].shape())

index = np.zeros(len(data), dtype=bool)
index[[1,5,9, 16, 20, 22]] = True

print(data[index].shape())


# ### Simple Arithmetic Operations

# Calculate the mean for every voxel over images

# In[52]:


data.mean()


# Calculate the standard deviation for every voxel over images

# In[53]:


data.std()


# Methods can be chained.  Here we get the shape of the mean.

# In[54]:


print(data.mean().shape())


# Brain_Data instances can be added and subtracted

# In[55]:


new = data[1]+data[2]


# Brain_Data instances can be manipulated with basic arithmetic operations.
# 
# Here we add 10 to every voxel and scale by 2

# In[56]:


data2 = (data + 10) * 2


# Brain_Data instances can be copied

# In[57]:


new = data.copy()


# Brain_Data instances can be easily converted to nibabel instances, which store the data in a 3D/4D matrix.  This is useful for interfacing with other python toolboxes such as [nilearn](http://nilearn.github.io)
# 

# In[58]:


data.to_nifti()


# Brain_Data instances can be concatenated using the append method

# In[59]:


new = new.append(data[4])


# Lists of `Brain_Data` instances can also be concatenated by recasting as a `Brain_Data` object.

# In[60]:


print(type([x for x in data[:4]]))

type(Brain_Data([x for x in data[:4]]))


# Any Brain_Data object can be written out to a nifti file.

# In[203]:


data.write('Tmp_Data.nii.gz')


# Images within a Brain_Data() instance are iterable.  Here we use a list comprehension to calculate the overall mean across all voxels within an image.

# In[61]:


[x.mean() for x in data]


# Though, we could also do this with the `mean` method by setting `axis=1`.

# In[62]:


data.mean(axis=1)


# Let's plot the mean to see how the global signal changes over time.

# In[63]:


plt.plot(data.mean(axis=1))


# Notice the slow linear drift over time, where the global signal intensity gradually decreases. We will learn how to remove this with a high pass filter in future tutorials.

# ### Plotting
# There are multiple ways to plot your data.
# 
# For a very quick plot, you can return a montage of axial slices with the `.plot()` method. As an example, we will plot the mean of each voxel over time.

# In[64]:


f = data.mean().plot()


# There is an interactive `.iplot()` method based on nilearn `view_img`.

# In[65]:


data.mean().iplot()


# Brain_Data() instances can be converted to a nibabel instance and plotted using any nilearn plot method such as glass brain.
# 

# In[66]:


plot_glass_brain(data.mean().to_nifti())


# Ok, that's the basics. `Brain_Data` can do much more!
# 
# Check out some of our [tutorials](https://nltools.org/auto_examples/index.html) for more detailed examples.
# 
# We'll be using this tool throughout the course.

# ## Exercises
# 
# For homework, let's practice our skills in working with data.

# ### Exercise 1
# A few subjects have already been preprocessed with fMRI prep.
# 
# Use pybids to figure out which subjects have been preprocessed.

# In[ ]:





# ### Exercise 2
# 
# One question we are often interested in is where in the brain do we have an adequate signal to noise ratio (SNR). There are many different metrics, here we will use temporal SNR, which the voxel mean over time divided by it's standard deviation.
# 
# $$\text{tSNR} = \frac{\text{mean}(\text{voxel}_{i})}{\text{std}(\text{voxel}_i)}$$
# 
# In Exercise 2, calculate the SNR for S01 and plot this so we can figure which regions have high and low SNR.

# In[ ]:





# ### Exercise 3
# 
# We are often interested in identifying outliers in our data. In this exercise, find any image that is outside 95% of all images based on global intensity (i.e., zscore greater than 2) from 'S01' and plot each one.

# In[ ]:




