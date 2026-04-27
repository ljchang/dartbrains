# Glossary

Throughout this course we will use a variety of different functions available in the base Python library, but also many other libraries in the scientific computing stack. Here we provide a list of all of the functions that are used across the various notebooks. It can be a helpful reference when you are learning Python about the types of things you can do with various packages. Remember you can always view the docstrings for any function by adding a `?` to the end of the function name.

## Jupyter Cell Magic

Magics are specific to and provided by the IPython kernel. Whether Magics are available on a kernel is a decision that is made by the kernel developer on a per-kernel basis. To work properly, Magics must use a syntax element which is not valid in the underlying language. For example, the IPython kernel uses the `%` syntax element for Magics as `%` is not a valid unary operator in Python. However, `%` might have meaning in other languages.

:::{glossary}
%conda
: Run the conda package manager within the current kernel. [API docs](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-conda)

%debug
: Activate the interactive debugger. This magic command supports two ways of activating debugger. One is to activate debugger before executing code. This way, you can set a break point, to step through the code from the point. The other one is to activate debugger in post-mortem mode. You can activate this mode simply running %debug without any argument. If an exception has just occurred, this lets you inspect its stack frames interactively. [API docs](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-debug)

%matplotlib
: Set up matplotlib to work interactively. Example: `%matplotlib inline`. This function lets you activate matplotlib interactive support at any point during an IPython session. It does not import anything into the interactive namespace. [API docs](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-matplotlib)

%timeit
: Time execution of a Python statement or expression using the timeit module. This function can be used both as a line and cell magic. [API docs](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-timeit)

! (shell execute)
: Shell execute — run shell command and capture output (`!!` is short-hand). Example: `!pip`. [API docs](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-system)
:::

## Base Python Functions

These functions are all bundled with Python.

:::{glossary}
any
: Test if any of the elements are true. [API docs](https://docs.python.org/2/library/stdtypes.html#truth-value-testing)

bool
: Cast as boolean type. [API docs](https://docs.python.org/3/library/functions.html#bool)

dict
: Cast as dictionary type. [API docs](https://docs.python.org/3/library/functions.html#func-dict)

enumerate
: Return an enumerate object. iterable must be a sequence, an iterator, or some other object which supports iteration. The `__next__()` method of the iterator returned by `enumerate()` returns a tuple containing a count (from start which defaults to 0) and the values obtained from iterating over iterable. [API docs](https://docs.python.org/3/library/functions.html#enumerate)

float
: Return a floating point number constructed from a number or string x. [API docs](https://docs.python.org/3/library/functions.html#float)

import
: Import python module into namespace. [API docs](https://docs.python.org/3/reference/import.html)

int
: Cast as integer type. [API docs](https://docs.python.org/3/library/functions.html#int)

len
: Return the length (the number of items) of an object. The argument may be a sequence (such as a string, bytes, tuple, list, or range) or a collection (such as a dictionary, set, or frozen set). [API docs](https://docs.python.org/3/library/functions.html#len)

glob.glob
: The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. No tilde expansion is done, but `*`, `?`, and character ranges expressed with `[]` will be correctly matched. [API docs](https://docs.python.org/3/library/glob.html)

list
: Cast as list type. [API docs](https://docs.python.org/3/library/functions.html#func-list)

max
: Return the largest item in an iterable or the largest of two or more arguments. [API docs](https://docs.python.org/3/library/functions.html#max)

min
: Return the smallest item in an iterable or the smallest of two or more arguments. [API docs](https://docs.python.org/3/library/functions.html#min)

os.path.basename
: Return the base name of pathname path. This is the second element of the pair returned by passing path to the function `split()`. [API docs](https://docs.python.org/2/library/os.path.html#os.path.basename)

os.path.join
: Join one or more path components intelligently. The return value is the concatenation of path and any members of paths with exactly one directory separator (`os.sep`) following each non-empty part except the last. [API docs](https://docs.python.org/2/library/os.path.html#os.path.join)

print
: Print strings. Recommend using f-strings formatting. Example: `print(f'Results: {variable}')`. [API docs](https://docs.python.org/3/tutorial/inputoutput.html)

pwd
: Print current working directory. [API docs](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-pwd)

sorted
: Return a new sorted list from the items in iterable. [API docs](https://docs.python.org/3/library/functions.html#sorted)

str
: Cast as string type. [API docs](https://docs.python.org/3/library/functions.html#func-str)

range
: Rather than being a function, range is actually an immutable sequence type, as documented in Ranges and Sequence Types — list, tuple, range. [API docs](https://docs.python.org/3/library/functions.html#func-range)

tuple
: Cast as tuple type. [API docs](https://docs.python.org/3/library/functions.html#func-tuple)

type
: Return the type of the object. [API docs](https://docs.python.org/3/library/functions.html#type)

zip
: Make an iterator that aggregates elements from each of the iterables. [API docs](https://docs.python.org/3/library/functions.html#zip)
:::

## Pandas

[pandas](https://pandas.pydata.org/) is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

`import pandas as pd`

:::{glossary}
pd.read_csv
: Read a comma-separated values (csv) file into DataFrame. [API docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

pd.concat
: Concatenate pandas objects along a particular axis with optional set logic along the other axes. [API docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html)

pd.DataFrame.isnull
: Detect missing values. [API docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isnull.html)

pd.DataFrame.mean
: Return the mean of the values for the requested axis. [API docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html)

pd.DataFrame.std
: Return sample standard deviation over requested axis. [API docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.std.html)

pd.DataFrame.plot
: Plot data using matplotlib. [API docs](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html)

pd.DataFrame.map
: Map values of Series according to input correspondence. [API docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html)

pd.DataFrame.groupby
: Group DataFrame or Series using a mapper or by a Series of columns. [API docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html)

pd.DataFrame.fillna
: Fill NA/NaN values using the specified method. [API docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html)

pd.DataFrame.replace
: Replace values given in to_replace with value. [API docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html)
:::

## NumPy

[NumPy](https://numpy.org/) is the fundamental package for scientific computing with Python. It contains among other things:
- a powerful N-dimensional array object
- sophisticated (broadcasting) functions
- tools for integrating C/C++ and Fortran code
- useful linear algebra, Fourier transform, and random number capabilities

Besides its obvious scientific uses, NumPy can also be used as an efficient multi-dimensional container of generic data. Arbitrary data-types can be defined. This allows NumPy to seamlessly and speedily integrate with a wide variety of databases.

`import numpy as np`

:::{glossary}
np.arange
: Return evenly spaced values within a given interval. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html)

np.array
: Create an array. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html)

np.convolve
: Returns the discrete, linear convolution of two one-dimensional sequences. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html)

np.cos
: Trigonometric cosine element-wise. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cos.html)

np.diag
: Extract a diagonal or construct a diagonal array. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.diag.html)

np.diag_indices
: Return the indices to access the main diagonal of an array. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.diag_indices.html)

np.dot
: Dot product of two arrays. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html)

np.exp
: Calculate the exponential of all elements in the input array. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html)

np.fft.fft
: Compute the one-dimensional discrete Fourier Transform. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html)

np.fft.ifft
: Compute the one-dimensional inverse discrete Fourier Transform. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.ifft.html)

np.hstack
: Stack arrays in sequence horizontally (column wise). [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html)

np.linalg.pinv
: Compute the (Moore-Penrose) pseudo-inverse of a matrix. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html)

np.mean
: Compute the arithmetic mean along the specified axis. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html)

np.nan
: IEEE 754 floating point representation of Not a Number (NaN). [API docs](https://docs.scipy.org/doc/numpy/reference/constants.html#numpy.NaN)

np.ones
: Return a new array of given shape and type, filled with ones. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html)

np.pi
: Return pi 3.1415926535897932384626433... [API docs](https://docs.scipy.org/doc/numpy/reference/constants.html#numpy.pi)

np.random.randint
: Return random integers from low (inclusive) to high (exclusive). [API docs](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randint.html)

np.random.randn
: Return a sample (or samples) from the "standard normal" distribution. [API docs](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randn.html)

np.real
: Return the real part of the complex argument. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.real.html)

np.sin
: Trigonometric sine, element-wise. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sin.html)

np.sqrt
: Return the non-negative square-root of an array, element-wise. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html)

np.squeeze
: Remove single-dimensional entries from the shape of an array. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html)

np.std
: Compute the standard deviation along the specified axis. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html#numpy.std)

np.vstack
: Stack arrays in sequence vertically (row wise). [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html#numpy.vstack)

np.zeros
: Return a new array of given shape and type, filled with zeros. [API docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html)
:::

## SciPy

[SciPy](https://docs.scipy.org/doc/scipy/) is one of the core packages that make up the SciPy stack. It provides many user-friendly and efficient numerical routines, such as routines for numerical integration, interpolation, optimization, linear algebra, and statistics.

:::{glossary}
scipy.stats.binom
: A binomial discrete random variable. [API docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)

scipy.signal.butter
: Butterworth digital and analog filter design. [API docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)

scipy.signal.filtfilt
: Apply a digital filter forward and backward to a signal. [API docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html)

scipy.signal.freqz
: Compute the frequency response of a digital filter. [API docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html)

scipy.signal.sosfreqz
: Compute the frequency response of a digital filter in SOS format. [API docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfreqz.html)

scipy.stats.ttest_1samp
: Calculate the T-test for the mean of ONE group of scores. [API docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html)
:::

## Matplotlib

[Matplotlib](https://matplotlib.org/) is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter notebook, web application servers, and four graphical user interface toolkits.

`import matplotlib.pyplot as plt`

:::{glossary}
plt.bar
: Make a bar plot. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.bar.html)

plt.figure
: Create a new figure. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html)

plt.hist
: Plot a histogram. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html)

plt.imshow
: Display an image, i.e. data on a 2D regular raster. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.imshow.html)

plt.legend
: Place a legend on the axes. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html)

plt.savefig
: Save the current figure. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html)

plt.scatter
: A scatter plot of y vs x with varying marker size and/or color. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html)

plt.subplots
: Create a figure and a set of subplots. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots.html)

plt.tight_layout
: Automatically adjust subplot parameters to give specified padding. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.tight_layout.html)

ax.axvline
: Add a vertical line across the axes. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.axvline.html)

ax.set_xlabel
: Set the label for the x-axis. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html)

ax.set_xlim
: Set the x-axis view limits. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_xlim.html)

ax.set_xticklabels
: Set the x-tick labels with list of string labels. [API docs](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html)

ax.set_ylim
: Set the y-axis view limits. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_ylim.html)

ax.set_yticklabels
: Set the y-tick labels with list of string labels. [API docs](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html)

ax.set_ylabel
: Set the label for the y-axis. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html)

ax.set_title
: Set a title for the axes. [API docs](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_title.html)
:::

## Seaborn

[Seaborn](https://seaborn.pydata.org/) is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

`import seaborn as sns`

:::{glossary}
sns.heatmap
: Plot rectangular data as a color-encoded matrix. [API docs](https://seaborn.pydata.org/generated/seaborn.heatmap.html)

sns.catplot
: Figure-level interface for drawing categorical plots onto a FacetGrid. [API docs](https://seaborn.pydata.org/generated/seaborn.catplot.html)

sns.jointplot
: Draw a plot of two variables with bivariate and univariate graphs. [API docs](https://seaborn.pydata.org/generated/seaborn.jointplot.html)

sns.regplot
: Plot data and a linear regression model fit. [API docs](https://seaborn.pydata.org/generated/seaborn.regplot.html)
:::

## scikit-learn

[Scikit-learn](https://scikit-learn.org/) is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities.

:::{glossary}
sklearn.metrics.pairwise_distances
: This method takes either a vector array or a distance matrix, and returns a distance matrix. If the input is a vector array, the distances are computed. If the input is a distances matrix, it is returned instead. [API docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)

sklearn.metrics.balanced_accuracy_score
: Compute the balanced accuracy. The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class. [API docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
:::

## networkx

[NetworkX](https://networkx.github.io/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

`import networkx as nx`

:::{glossary}
nx.draw_kamada_kawai
: Draw the graph G with a Kamada-Kawai force-directed layout. [API docs](https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_kamada_kawai.html)

nx.degree
: Return the degree of a node or nodes. The node degree is the number of edges adjacent to that node. [API docs](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.DiGraph.degree.html)
:::

## NiBabel

[nibabel](https://nipy.org/nibabel/) is a package to help read/write access to some common neuroimaging file formats, including: ANALYZE (plain, SPM99, SPM2 and later), GIFTI, NIfTI1, NIfTI2, CIFTI-2, MINC1, MINC2, AFNI BRIK/HEAD, MGH and ECAT as well as Philips PAR/REC. We can read and write FreeSurfer geometry, annotation and morphometry files. There is some very limited support for DICOM. NiBabel is the successor of PyNIfTI.

The various image format classes give full or selective access to header (meta) information and access to the image data is made available via NumPy arrays.

`import nibabel as nib`

:::{glossary}
nib.load
: Load file given filename, guessing at file type. [API docs](https://nipy.org/nibabel/reference/nibabel.loadsave.html#module-nibabel.loadsave)

nib.save
: Save an image to file adapting format to filename. [API docs](https://nipy.org/nibabel/reference/nibabel.loadsave.html#nibabel.loadsave.save)

data.get_data
: Return image data from image with any necessary scaling applied. [API docs](https://nipy.org/nibabel/reference/nibabel.dataobj_images.html)

data.get_shape
: Return shape for image. [API docs](https://nipy.org/nibabel/reference/nibabel.dataobj_images.html)

data.header
: The header of an image contains the image metadata. The information in the header will differ between different image formats. For example, the header information for a NIfTI1 format file differs from the header information for a MINC format file. [API docs](https://nipy.org/nibabel/nibabel_images.html)

data.affine
: Homogenous affine giving relationship between voxel coordinates and world coordinates. Affine can also be None. In this case, `obj.affine` also returns None, and the affine as written to disk will depend on the file format. [API docs](https://nipy.org/nibabel/reference/nibabel.nifti1.html#nibabel.nifti1.Nifti1Image)
:::

## NiLearn

[nilearn](https://nilearn.github.io/) is a Python module for fast and easy statistical learning on NeuroImaging data.

It leverages the scikit-learn Python toolbox for multivariate statistics with applications such as predictive modelling, classification, decoding, or connectivity analysis.

:::{glossary}
nilearn.plotting.plot_anat
: Plot cuts of an anatomical image (by default 3 cuts: Frontal, Axial, and Lateral). [API docs](https://nilearn.github.io/dev/modules/generated/nilearn.plotting.plot_anat.html)

nilearn.plotting.view_img
: Interactive html viewer of a statistical map, with optional background. [API docs](https://nilearn.github.io/dev/modules/generated/nilearn.plotting.view_img.html)

nilearn.plotting.plot_glass_brain
: Plot 2d projections of an ROI/mask image (by default 3 projections: Frontal, Axial, and Lateral). The brain glass schematics are added on top of the image. [API docs](https://nilearn.github.io/dev/modules/generated/nilearn.plotting.plot_glass_brain.html)

nilearn.plotting.plot_stat_map
: Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and Lateral). [API docs](https://nilearn.github.io/dev/modules/generated/nilearn.plotting.plot_stat_map.html)
:::

## nltools

[NLTools](https://nltools.org/) is a Python package for analyzing neuroimaging data. It is the analysis engine powering neuro-learn. There are tools to perform data manipulation and analyses such as univariate GLMs, predictive multivariate modeling, and representational similarity analyses.

### Data Classes

#### Adjacency

:::{glossary}
Adjacency
: A class to represent Adjacency matrices as a vector rather than a 2-dimensional matrix. This makes it easier to perform data manipulation and analyses. This tool is particularly useful for performing Representational Similarity Analyses. [API docs](https://nltools.org/api.html#nltools.data.Adjacency)

Adjacency.distance
: Calculate distance between images within an Adjacency() instance. [API docs](https://nltools.org/api.html#nltools.data.Adjacency.distance)

Adjacency.distance_to_similarity
: Convert distance matrix to similarity matrix. [API docs](https://nltools.org/api.html#nltools.data.Adjacency.distance_to_similarity)

Adjacency.plot
: Create Heatmap of Adjacency Matrix. [API docs](https://nltools.org/api.html#nltools.data.Adjacency.plot)

Adjacency.plot_mds
: Plot Multidimensional Scaling. [API docs](https://nltools.org/api.html#nltools.data.Adjacency.plot_mds)

Adjacency.to_graph
: Convert Adjacency into networkx graph. Only works on single_matrix for now. [API docs](https://nltools.org/api.html#nltools.data.Adjacency.to_graph)
:::

#### Brain_Data

:::{glossary}
Brain_Data
: A class to represent neuroimaging data in python as a vector rather than a 3-dimensional matrix. This makes it easier to perform data manipulation and analyses. This is the main tool for working with neuroimaging data. [API docs](https://nltools.org/api.html#nltools-data-data-types)

Brain_Data.append
: Append data to Brain_Data instance. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.append)

Brain_Data.apply_mask
: Mask Brain_Data instance. [API docs](https://nltools.org/api.html?highlight=apply+mask#nltools.data.Brain_Data.apply_mask)

Brain_Data.copy
: Create a copy of a Brain_Data instance. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.copy)

Brain_Data.decompose
: Decompose Brain_Data object. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.decompose)

Brain_Data.distance
: Calculate distance between images within a Brain_Data() instance. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.distance)

Brain_Data.extract_roi
: Extract activity from mask. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.extract_roi)

Brain_Data.find_spikes
: Function to identify spikes from Time Series Data. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.find_spikes)

Brain_Data.iplot
: Create an interactive brain viewer for the current brain data instance. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.iplot)

Brain_Data.mean
: Get mean of each voxel across images. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.mean)

Brain_Data.plot
: Create a quick plot of self.data. Will plot each image separately. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.plot)

Brain_Data.predict
: Run prediction. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.predict)

Brain_Data.regress
: Run a mass-univariate regression across voxels. Three types of regressions can be run: 1) Standard OLS (default) 2) Robust OLS (heteroscedasticity and/or auto-correlation robust errors) 3) ARMA (auto-regressive and moving-average lags = 1 by default; experimental). [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.regress)

Brain_Data.shape
: Get images by voxels shape. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.shape)

Brain_Data.similarity
: Calculate similarity of Brain_Data() instance with single Brain_Data or Nibabel image. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.similarity)

Brain_Data.smooth
: Apply spatial smoothing using nilearn smooth_img(). [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.smooth)

Brain_Data.std
: Get standard deviation of each voxel across images. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.std)

Brain_Data.threshold
: Threshold Brain_Data instance. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.threshold)

Brain_Data.to_nifti
: Convert Brain_Data Instance into Nifti Object. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.to_nifti)

Brain_Data.ttest
: Calculate one sample t-test across each voxel (two-sided). [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.ttest)

Brain_Data.write
: Write out Brain_Data object to Nifti or HDF5 File. [API docs](https://nltools.org/api.html#nltools.data.Brain_Data.write)
:::

#### Design_Matrix

:::{glossary}
Design_Matrix
: A class to represent design matrices with special methods for data processing (e.g. convolution, upsampling, downsampling) and also intelligent and flexible appending (e.g. automatically keep certain columns or polynomial terms separated during concatenation). It plays nicely with Brain_Data and can be used to build an experimental design to pass to Brain_Data's X attribute. It is essentially an enhanced pandas DataFrame, with extra attributes and methods. [API docs](https://nltools.org/api.html#nltools.data.Design_Matrix)

Design_Matrix.add_dct_basis
: Adds unit scaled cosine basis functions to Design_Matrix columns, based on spm-style discrete cosine transform for use in high-pass filtering. Does not add intercept/constant. Care is recommended if using this along with `.add_poly()`, as some columns will be highly correlated. [API docs](https://nltools.org/api.html#nltools.data.Design_Matrix.add_dct_basis)

Design_Matrix.add_poly
: Add nth order Legendre polynomial terms as columns to design matrix. Good for adding constant/intercept to model (order = 0) and accounting for slow-frequency nuisance artifacts e.g. linear, quadratic, etc drifts. [API docs](https://nltools.org/api.html#nltools.data.Design_Matrix.add_poly)

Design_Matrix.clean
: Method to fill NaNs in Design Matrix and remove duplicate columns based on data values, NOT names. Columns are dropped if they are correlated >= the requested threshold (default = .95). [API docs](https://nltools.org/api.html#nltools.data.Design_Matrix.clean)

Design_Matrix.convolve
: Perform convolution using an arbitrary function. [API docs](https://nltools.org/api.html#nltools.data.Design_Matrix.convolve)

Design_Matrix.heatmap
: Visualize Design Matrix spm style. Use `.plot()` for typical pandas plotting functionality. Can pass optional keyword args to seaborn heatmap. [API docs](https://nltools.org/api.html#nltools.data.Design_Matrix.heatmap)

Design_Matrix.head
: This function returns the first n rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it. [API docs](https://nltools.org/api/pandas.DataFrame.head.html)

Design_Matrix.info
: Print a concise summary of a DataFrame. [API docs](https://nltools.org/api/pandas.DataFrame.info.html)

Design_Matrix.vif
: Compute variance inflation factor amongst columns of design matrix, ignoring polynomial terms. Much faster than statsmodel and more reliable too. Uses the same method as Matlab and R (diagonal elements of the inverted correlation matrix). [API docs](https://nltools.org/api.html#nltools.data.Design_Matrix.vif)

Design_Matrix.zscore
: Z-score the design matrix, but ensures that returned object is a design matrix. [API docs](https://nltools.org/api.html#nltools.data.Design_Matrix.zscore)
:::

### Statistics Functions

:::{glossary}
stats.fdr
: Determine FDR threshold given a p value array and desired false discovery rate q. [API docs](https://nltools.org/api.html#nltools.stats.fdr)

stats.find_spikes
: Function to identify spikes from fMRI Time Series Data. [API docs](https://nltools.org/api.html#nltools.stats.find_spikes)

stats.fisher_r_to_z
: Use Fisher transformation to convert correlation to z score. [API docs](https://nltools.org/api.html#nltools.stats.fisher_r_to_z)

stats.one_sample_permutation
: One sample permutation test using randomization. [API docs](https://nltools.org/api.html#nltools.stats.one_sample_permutation)

stats.threshold
: Threshold test image by p-value from p image. [API docs](https://nltools.org/api.html#nltools.stats.threshold)

stats.regress
: This is a flexible function to run several types of regression models provided X and Y numpy arrays. Y can be a 1d numpy array or 2d numpy array. In the latter case, results will be output with shape 1 x Y.shape[1], in other words fitting a separate regression model to each column of Y. [API docs](https://nltools.org/api.html#nltools.stats.regress)

stats.zscore
: Z-score every column in a pandas dataframe or series. [API docs](https://nltools.org/api.html#nltools.stats.zscore)
:::

### Miscellaneous Functions

:::{glossary}
SimulateGrid
: A class to simulate signal and noise within 2D grid. [API docs](https://github.com/cosanlab/nltools/blob/master/nltools/simulator.py)

external.hrf.glover_hrf
: Implementation of the Glover hemodynamic response function (HRF) model. [API docs](https://nistats.github.io/modules/generated/nistats.hemodynamic_models.glover_hrf.html)

datasets.fetch_pain
: Download and loads pain dataset from neurovault. [API docs](https://nltools.org/api.html#nltools.datasets.fetch_pain)
:::
