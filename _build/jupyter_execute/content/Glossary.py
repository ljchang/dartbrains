# Glossary
*Written by Luke Chang*

Throughout this course we will use a variety of different functions available in the base Python library, but also many other libraries in the scientific computing stack. Here we provide a list of all of the functions that are used across the various notebooks. It can be a helpful reference when you are learning Python about the types of things you can do with various packages. Remember you can always view the docstrings for any function by adding a `?` to the end of the function name.

## Jupyter Cell Magic
Magics are specific to and provided by the IPython kernel. Whether Magics are available on a kernel is a decision that is made by the kernel developer on a per-kernel basis. To work properly, Magics must use a syntax element which is not valid in the underlying language. For example, the IPython kernel uses the `%` syntax element for Magics as `%` is not a valid unary operator in Python. However, `%` might have meaning in other languages.

[%conda](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-conda): Run the conda package manager within the current kernel.

[%debug](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-debug): Activate the interactive debugger. This magic command support two ways of activating debugger. One is to activate debugger before executing code. This way, you can set a break point, to step through the code from the point. You can use this mode by giving statements to execute and optionally a breakpoint. The other one is to activate debugger in post-mortem mode. You can activate this mode simply running %debug without any argument. If an exception has just occurred, this lets you inspect its stack frames interactively. Note that this will always work only on the last traceback that occurred, so you must call this quickly after an exception that you wish to inspect has fired, because if another one occurs, it clobbers the previous one.

[%matplotlib](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-matplotlib): Set up matplotlib to work interactively. Example: ```%matplotlib inline```

This function lets you activate matplotlib interactive support at any point during an IPython session. It does not import anything into the interactive namespace.

[%timeit](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-timeit): Time execution of a Python statement or expression using the timeit module. This function can be used both as a line and cell magic

[!](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-system): Shell execute - run shell command and capture output (!! is short-hand). Example: `!pip`.

## Base Python Functions
These functions are all bundled with Python

[any](https://docs.python.org/2/library/stdtypes.html#truth-value-testing): Test if any of the elements are true.

[bool](https://docs.python.org/3/library/functions.html#bool): Cast as boolean type

[dict](https://docs.python.org/3/library/functions.html#func-dict): Cast as dictionary type

[enumerate](https://docs.python.org/3/library/functions.html#enumerate): Return an enumerate object. iterable must be a sequence, an iterator, or some other object which supports iteration. The __next__() method of the iterator returned by enumerate() returns a tuple containing a count (from start which defaults to 0) and the values obtained from iterating over iterable.

[float](https://docs.python.org/3/library/functions.html#float): Return a floating point number constructed from a number or string x.

[import](https://docs.python.org/3/reference/import.html): Import python module into namespace. 

[int](https://docs.python.org/3/library/functions.html#int): Cast as integer type

[len](https://docs.python.org/3/library/functions.html#len): Return the length (the number of items) of an object. The argument may be a sequence (such as a string, bytes, tuple, list, or range) or a collection (such as a dictionary, set, or frozen set).

[glob.glob](https://docs.python.org/3/library/glob.html): The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. No tilde expansion is done, but `*`, `?`, and character ranges expressed with `[]` will be correctly matched. This is done by using the `os.scandir()` and `fnmatch.fnmatch()` functions in concert, and not by actually invoking a subshell. 

[list](https://docs.python.org/3/library/functions.html#func-list): Cast as list type

[max](https://docs.python.org/3/library/functions.html#max): Return the largest item in an iterable or the largest of two or more arguments.

[min](https://docs.python.org/3/library/functions.html#min): Return the smallest item in an iterable or the smallest of two or more arguments.

[os.path.basename](https://docs.python.org/2/library/os.path.html#os.path.basename): Return the base name of pathname path. This is the second element of the pair returned by passing path to the function split(). Note that the result of this function is different from the Unix basename program; where basename for '/foo/bar/' returns 'bar', the basename() function returns an empty string ('').

[os.path.join](https://docs.python.org/2/library/os.path.html#os.path.join): Join one or more path components intelligently. The return value is the concatenation of path and any members of paths with exactly one directory separator (os.sep) following each non-empty part except the last, meaning that the result will only end in a separator if the last part is empty. If a component is an absolute path, all previous components are thrown away and joining continues from the absolute path component.

[print](https://docs.python.org/3/tutorial/inputoutput.html): Print strings. Recommend using f-strings formatting. Example, `print(f'Results: {variable}')`.

[pwd](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-pwd): Print current working directory

[sorted](https://docs.python.org/3/library/functions.html#sorted): Return a new sorted list from the items in iterable.

[str](https://docs.python.org/3/library/functions.html#func-str): For more information on static methods, see [The standard type hierarchy](https://docs.python.org/3/reference/datamodel.html#types).

[range](https://docs.python.org/3/library/functions.html#func-range): Rather than being a function, range is actually an immutable sequence type, as documented in Ranges and Sequence Types — list, tuple, range.

[tuple](https://docs.python.org/3/library/functions.html#func-tuple): Cast as tuple type

[type](https://docs.python.org/3/library/functions.html#type): Return the type of the object.

[zip](https://docs.python.org/3/library/functions.html#zip): Make an iterator that aggregates elements from each of the iterables.


## Pandas
[pandas](https://pandas.pydata.org/) is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

```import pandas as pd```

***

[pd.read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html): Read a comma-separated values (csv) file into DataFrame.

[pd.concat](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html): Concatenate pandas objects along a particular axis with optional set logic along the other axes.

[pd.DataFrame.isnull](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isnull.html): Detect missing values.

[pd.DataFrame.mean](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html): Return the mean of the values for the requested axis.

[pd.DataFrame.std](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.std.html): Return sample standard deviation over requested axis.

[pd.DataFrame.plot](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html): Plot data using matplotlib

[pd.DataFrame.map](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html): Map values of Series according to input correspondence.

[pd.DataFrame.groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html): Group DataFrame or Series using a mapper or by a Series of columns.

[pd.DataFrame.fillna](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html): Fill NA/NaN values using the specified method.

[pd.DataFrame.replace](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html): Replace values given in to_replace with value.

## NumPy
[NumPy](https://numpy.org/) is the fundamental package for scientific computing with Python. It contains among other things:
- a powerful N-dimensional array object
- sophisticated (broadcasting) functions
- tools for integrating C/C++ and Fortran code
- useful linear algebra, Fourier transform, and random number capabilities

Besides its obvious scientific uses, NumPy can also be used as an efficient multi-dimensional container of generic data. Arbitrary data-types can be defined. This allows NumPy to seamlessly and speedily integrate with a wide variety of databases.

```import numpy as np```

***


[np.arange](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html): Return evenly spaced values within a given interval.

[np.array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html): Create an array

[np.convolve](https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html): Returns the discrete, linear convolution of two one-dimensional sequences.

[np.cos](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cos.html): Trigonometric cosine element-wise.

[np.diag](https://docs.scipy.org/doc/numpy/reference/generated/numpy.diag.html): Extract a diagonal or construct a diagonal array.

[np.diag_indices](https://docs.scipy.org/doc/numpy/reference/generated/numpy.diag_indices.html): Return the indices to access the main diagonal of an array.

[np.dot](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html): Dot product of two arrays.

[np.exp](https://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html): Calculate the exponential of all elements in the input array.

[np.fft.fft](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html): Compute the one-dimensional discrete Fourier Transform.

[np.fft.ifft](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.ifft.html): Compute the one-dimensional inverse discrete Fourier Transform.

[np.hstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html): Stack arrays in sequence horizontally (column wise).

[np.linalg.pinv](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html): Compute the (Moore-Penrose) pseudo-inverse of a matrix.

[np.mean](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html): Compute the arithmetic mean along the specified axis.

[np.nan](https://docs.scipy.org/doc/numpy/reference/constants.html#numpy.NaN): IEEE 754 floating point representation of Not a Number (NaN).

[np.ones](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html): Return a new array of given shape and type, filled with ones.

[np.pi](https://docs.scipy.org/doc/numpy/reference/constants.html#numpy.pi): Return pi 3.1415926535897932384626433...

[np.random.randint](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randint.html): Return random integers from low (inclusive) to high (exclusive).

[np.random.randn](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randn.html): Return a sample (or samples) from the “standard normal” distribution.

[np.real](https://docs.scipy.org/doc/numpy/reference/generated/numpy.real.html): Return the real part of the complex argument.

[np.sin](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sin.html): Trigonometric sine, element-wise.

[np.sqrt](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html): Return the non-negative square-root of an array, element-wise.

[np.squeeze](https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html): Remove single-dimensional entries from the shape of an array.

[np.std](https://docs.scipy.org/doc/numpy/reference/generated/numpy.std.html#numpy.std): Compute the standard deviation along the specified axis.

[np.vstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html#numpy.vstack): Stack arrays in sequence vertically (row wise).

[np.zeros](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html): Return a new array of given shape and type, filled with zeros.

## SciPy 
[SciPy](https://www.scipy.org/scipylib/index.html) is one of the core packages that make up the SciPy stack. It provides many user-friendly and efficient numerical routines, such as routines for numerical integration, interpolation, optimization, linear algebra, and statistics.

***

[scipy.stats.binom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html): A binomial discrete random variable.

[scipy.signal.butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html): Butterworth digital and analog filter design.

[scipy.signal.filtfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html): Apply a digital filter forward and backward to a signal.

[scipy.signal.freqz](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html): Compute the frequency response of a digital filter.

[scipy.signal.sosfreqz](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfreqz.html): Compute the frequency response of a digital filter in SOS format.

[scipy.stats.ttest_1samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html): Calculate the T-test for the mean of ONE group of scores.

## Matplotlib
[Matplotlib](https://matplotlib.org/) is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter notebook, web application servers, and four graphical user interface toolkits.

```import matplotlib.pyplot as plt```

***


[plt.bar](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.bar.html): Make a bar plot.

[plt.figure](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.figure.html): Create a new figure.

[plt.hist](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html): Plot a histogram.

[plt.imshow](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.imshow.html): Display an image, i.e. data on a 2D regular raster.

[plt.legend](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html): Place a legend on the axes.

[plt.savefig](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html): Save the current figure.

[plt.scatter](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html): A scatter plot of y vs x with varying marker size and/or color.

[plt.subplots](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots.html): Create a figure and a set of subplots.

[plt.tight_layout](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.tight_layout.html): Automatically adjust subplot parameters to give specified padding.

[ax.axvline](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.axvline.html): Add a vertical line across the axes.

[ax.set_xlabel](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html): Set the label for the x-axis.
 
[ax.set_xlim](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_xlim.html): Set the x-axis view limits.

[ax.set_xticklabels](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html): Set the x-tick labels with list of string labels.

[ax.set_ylim](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_ylim.html): Set the y-axis view limits.

[ax.set_yticklabels](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html): Set the y-tick labels with list of string labels.

[ax.set_ylabel](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html): Set the label for the y-axis.

[ax.set_title](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.set_title.html): Set a title for the axes.

## Seaborn
[Seaborn](https://seaborn.pydata.org/) is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

```import seaborn as sns```

***

[sns.heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html): Plot rectangular data as a color-encoded matrix.

[sns.catplot](https://seaborn.pydata.org/generated/seaborn.catplot.html): Figure-level interface for drawing categorical plots onto a FacetGrid.

[sns.jointplot](https://seaborn.pydata.org/generated/seaborn.jointplot.html): Draw a plot of two variables with bivariate and univariate graphs.

[sns.regplot](https://seaborn.pydata.org/generated/seaborn.regplot.html): Plot data and a linear regression model fit.

## scikit-learn
[Scikit-learn](https://scikit-learn.org/) is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities.

***

[sklearn.metrics.pairwise_distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html): This method takes either a vector array or a distance matrix, and returns a distance matrix. If the input is a vector array, the distances are computed. If the input is a distances matrix, it is returned instead.

[sklearn.metrics.balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html): Compute the balanced accuracy. The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.


## networkx
[NetworkX](https://networkx.github.io/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

```import networkx as nx```

***
 
[nx.draw_kamada_kawai](https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_kamada_kawai.html): Draw the graph G with a Kamada-Kawai force-directed layout.

[nx.degree](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.DiGraph.degree.html): Return the degree of a node or nodes. The node degree is the number of edges adjacent to that node.

## NiBabel 
[nibabel](https://nipy.org/nibabel/) is a package to help Read / write access to some common neuroimaging file formats, including: ANALYZE (plain, SPM99, SPM2 and later), GIFTI, NIfTI1, NIfTI2, CIFTI-2, MINC1, MINC2, AFNI BRIK/HEAD, MGH and ECAT as well as Philips PAR/REC. We can read and write FreeSurfer geometry, annotation and morphometry files. There is some very limited support for DICOM. NiBabel is the successor of PyNIfTI.

The various image format classes give full or selective access to header (meta) information and access to the image data is made available via NumPy arrays.

```import nibabel as nib```

***

[nib.load](https://nipy.org/nibabel/reference/nibabel.loadsave.html#module-nibabel.loadsave): Load file given filename, guessing at file type

[nib.save](https://nipy.org/nibabel/reference/nibabel.loadsave.html#nibabel.loadsave.save): Save an image to file adapting format to filename

[data.get_data](https://nipy.org/nibabel/reference/nibabel.dataobj_images.html): Return image data from image with any necessary scaling applied

[data.get_shape](https://nipy.org/nibabel/reference/nibabel.dataobj_images.html): Return shape for image


[data.header](https://nipy.org/nibabel/nibabel_images.html): The header of an image contains the image metadata. The information in the header will differ between different image formats. For example, the header information for a NIfTI1 format file differs from the header information for a MINC format file.

[data.affine](https://nipy.org/nibabel/reference/nibabel.nifti1.html#nibabel.nifti1.Nifti1Image): homogenous affine giving relationship between voxel coordinates and world coordinates. Affine can also be None. In this case, obj.affine also returns None, and the affine as written to disk will depend on the file format.

## NiLearn
[nilearn](https://nilearn.github.io/) is a Python module for fast and easy statistical learning on NeuroImaging data.

It leverages the scikit-learn Python toolbox for multivariate statistics with applications such as predictive modelling, classification, decoding, or connectivity analysis.

***

[nilearn.plotting.plot_anat](https://nilearn.github.io/modules/generated/nilearn.plotting.plot_anat.html): Plot cuts of an anatomical image (by default 3 cuts: Frontal, Axial, and Lateral)

[nilearn.plotting.view_img](https://nilearn.github.io/modules/generated/nilearn.plotting.view_img.html):Interactive html viewer of a statistical map, with optional background

[nilearn.plotting.plot_glass_brain](https://nilearn.github.io/modules/generated/nilearn.plotting.plot_glass_brain.html):  Plot 2d projections of an ROI/mask image (by default 3 projections: Frontal, Axial, and Lateral). The brain glass schematics are added on top of the image.

[nilearn.plotting.plot_stat_map](https://nilearn.github.io/modules/generated/nilearn.plotting.plot_stat_map.html): Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and Lateral)

## nltools
[NLTools](https://neurolearn.readthedocs.io/en/latest/) is a Python package for analyzing neuroimaging data. It is the analysis engine powering neuro-learn There are tools to perform data manipulation and analyses such as univariate GLMs, predictive multivariate modeling, and representational similarity analyses.

***

### Data Classes
#### Adjacency
[Adjacency](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Adjacency) is a class to represent Adjacency matrices as a vector rather than a 2-dimensional matrix. This makes it easier to perform data manipulation and analyses. This tool is particularly useful for performing Representational Similarity Analyses.

[Adjacency.distance](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Adjacency.distance): Calculate distance between images within an Adjacency() instance.

[Adjacency.distance_to_similarity](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Adjacency.distance_to_similarity): Convert distance matrix to similarity matrix

[Adjacency.plot](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Adjacency.plot): Create Heatmap of Adjacency Matrix

[Adjacency.plot_mds](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Adjacency.plot_mds): Plot Multidimensional Scaling

[Adjacency.to_graph](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Adjacency.to_graph): Convert Adjacency into networkx graph. only works on single_matrix for now.


### Brain_Data
[Brain_Data](https://neurolearn.readthedocs.io/en/latest/api.html#nltools-data-data-types) is a class to represent neuroimaging data in python as a vector rather than a 3-dimensional matrix.This makes it easier to perform data manipulation and analyses. This is the main tool for working with neuroimaging data.

[Brain_Data.append](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.append): Append data to Brain_Data instance

[apply_mask](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.apply_mask): Mask Brain_Data instance

[Brain_Data.copy](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.copy): Create a copy of a Brain_Data instance.

[Brain_Data.decompose](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.decompose): Decompose Brain_Data object

[Brain_Data.distance](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.distance): Calculate distance between images within a Brain_Data() instance.

[Brain_Data.extract_roi](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.extract_roi): Extract activity from mask

[Brain_Data.find_spikes](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.find_spikes): Function to identify spikes from Time Series Data

[Brain_Data.iplot](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.iplot): Create an interactive brain viewer for the current brain data instance.

[Brain_Data.mean](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.mean): Get mean of each voxel across images.

[Brain_Data.plot](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.plot): Create a quick plot of self.data. Will plot each image separately

[Brain_Data.predict](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.predict): Run prediction

[Brain_Data.regress](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.regress
): Run a mass-univariate regression across voxels. Three types of regressions can be run: 1) Standard OLS (default) 2) Robust OLS (heteroscedasticty and/or auto-correlation robust errors), i.e. OLS with “sandwich estimators” 3) ARMA (auto-regressive and moving-average lags = 1 by default; experimental)

[Brain_Data.shape](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.shape): Get images by voxels shape.

[Brain_Data.similarity](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.similaritye): Calculate similarity of Brain_Data() instance with single Brain_Data or Nibabel image

[Brain_Data.smooth](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.smooth): Apply spatial smoothing using nilearn smooth_img()

[Brain_Data.std](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.std): Get standard deviation of each voxel across images.

[Brain_Data.threshold](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.threshold): Threshold Brain_Data instance. 

[Brain_Data.to_nifti](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.to_nifti): Convert Brain_Data Instance into Nifti Object

[Brain_Data.ttest](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.ttest): Calculate one sample t-test across each voxel (two-sided)

[Brain_Data.write](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Brain_Data.write): Write out Brain_Data object to Nifti or HDF5 File.

### Design_Matrix
[Design_Matrix](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Design_Matrix) is a class to represent design matrices with special methods for data processing (e.g. convolution, upsampling, downsampling) and also intelligent and flexible and intelligent appending (e.g. auto-matically keep certain columns or polynomial terms separated during concatentation). It plays nicely with Brain_Data and can be used to build an experimental design to pass to Brain_Data’s X attribute. It is essentially an enhanced pandas df, with extra attributes and methods. Methods always return a new design matrix instance (copy). Column names are always string types. Inherits most methods on pandas DataFrames.

[Design_Matrix.add_dct_basis](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Design_Matrix.add_dct_basis): Adds unit scaled cosine basis functions to Design_Matrix columns, based on spm-style discrete cosine transform for use in high-pass filtering. Does not add intercept/constant. Care is recommended if using this along with .add_poly(), as some columns will be highly-correlated.

[Design_Matrix.add_poly](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Design_Matrix.add_poly): Add nth order Legendre polynomial terms as columns to design matrix. Good for adding constant/intercept to model (order = 0) and accounting for slow-frequency nuisance artifacts e.g. linear, quadratic, etc drifts. Care is recommended when using this with .add_dct_basis() as some columns will be highly correlated.

[Design_Matrix.clean](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Design_Matrix.clean): Method to fill NaNs in Design Matrix and remove duplicate columns based on data values, NOT names. Columns are dropped if they are correlated >= the requested threshold (default = .95). In this case, only the first instance of that column will be retained and all others will be dropped.

[Design_Matrix.convolve](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Design_Matrix.convolve): Perform convolution using an arbitrary function.

[Design_Matrix.heatmap](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Design_Matrix.heatmap): Visualize Design Matrix spm style. Use .plot() for typical pandas plotting functionality. Can pass optional keyword args to seaborn heatmap.

[Design_Matrix.head](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html): This function returns the first n rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it.

[Design_Matrix.info](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html): Print a concise summary of a DataFrame.

[Design_Matrix.vif](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Design_Matrix.vif): Compute variance inflation factor amongst columns of design matrix, ignoring polynomial terms. Much faster that statsmodel and more reliable too. Uses the same method as Matlab and R (diagonal elements of the inverted correlation matrix).

[Design_Matrix.zscore](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.data.Design_Matrix.zscore
): nltools.stats.downsample, but ensures that returned object is a design matrix.


### Statistics Functions


[stats.fdr](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.stats.fdr): Determine FDR threshold given a p value array and desired false discovery rate q.

[stats.find_spikes](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.stats.find_spikes): Function to identify spikes from fMRI Time Series Data

[stats.fisher_r_to_z](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.stats.fisher_r_to_z): Use Fisher transformation to convert correlation to z score

[stats.one_sample_permutation](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.stats.one_sample_permutation): One sample permutation test using randomization.

[stats.threshold](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.stats.threshold): Threshold test image by p-value from p image

[stats.regress](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.stats.regress): This is a flexible function to run several types of regression models provided X and Y numpy arrays. Y can be a 1d numpy array or 2d numpy array. In the latter case, results will be output with shape 1 x Y.shape[1], in other words fitting a separate regression model to each column of Y.

[stats.zscore](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.stats.zscore): zscore every column in a pandas dataframe or series.

### Miscellaneous Functions

[SimulateGrid](https://github.com/cosanlab/nltools/blob/master/nltools/simulator.py): A class to simulate signal and noise within 2D grid. Need to update link to nltools documentation once it is built.

[external.hrf.glover_hrf](https://nistats.github.io/modules/generated/nistats.hemodynamic_models.glover_hrf.html): Implementation of the Glover hemodynamic response function (HRF) model.

[datasets.fetch_pain](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.datasets.fetch_pain): Download and loads pain dataset from neurovault

[datasets.fetch_localizer](https://neurolearn.readthedocs.io/en/latest/api.html#nltools.datasets.fetch_localizer): Download and load Brainomics Localizer dataset (94 subjects).



