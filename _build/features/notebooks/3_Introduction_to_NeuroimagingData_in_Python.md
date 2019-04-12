---
redirect_from:
  - "/features/notebooks/3-introduction-to-neuroimagingdata-in-python"
interact_link: content/features/notebooks/3_Introduction_to_NeuroimagingData_in_Python.ipynb
kernel_name: python3
title: 'Introduction to Neuroimaging Data'
prev_page:
  url: /features/notebooks/2_Introduction_to_Dataframes_Plotting
  title: 'Introduction to Dataframes and Plotting'
next_page:
  url: /features/notebooks/4_ICA
  title: 'Separating Signal from Noise with ICA'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# Introduction to neuroimaging data with Python

In this tutorial we will learn how to load, plot, and manipulate neuroimaging data in Python

Written by Luke Chang

## Nibabel

Neuroimaging data is often stored in the format of nifti files `.nii` which can also be compressed using gzip `.nii.gz`.  These files store both 3D and 4D data and also contain structured metadata in the image **header**.

There is an very nice tool to access nifti data stored on your file system in python called [nibabel](http://nipy.org/nibabel/).  If you don't already have nibabel installed on your computer it is easy via `pip`. First, tell the jupyter cell that you would like to access the unix system outside of the notebook and then install nibabel using pip. You only need to run this once (unless you would like to update the version).



{:.input_area}
```python
!pip install nibabel
```


nibabel objects can be initialized by simply pointing to a nifti file even if it is compressed through gzip.  First, we will import the nibabel module as `nib` (short and sweet so that we don't have to type so much when using the tool).  I'm also including a path to where the data file is located so that I don't have to constantly type this.  It is easy to change this on your own computer.

We will be loading an anatomical image from subject S01 from the open localizer [dataset](http://brainomics.cea.fr/localizer/).  See this [paper](https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-8-91) for more information about this dataset.



{:.input_area}
```python
import os
import nibabel as nib

base_dir = '/Users/lukechang/Github/dartbrains/Tutorials'
```




{:.input_area}
```python
data = nib.load(os.path.join(base_dir, 'normalized_T1_anat_defaced.nii.gz'))
```




{:.input_area}
```python
data.get_data()
```


If we want to get more help on how to work with the nibabel data object we can either consult the [documentation](https://nipy.org/nibabel/tutorials.html#tutorials) or add a `?`.



{:.input_area}
```python
data?
```


The imaging data is stored in either a 3D or 4D numpy array. Just like numpy, it is easy to get the dimensions of the data using `shape`



{:.input_area}
```python
data.shape
```


We can also directly access the data and plot a single slice using standard matplotlib functions.



{:.input_area}
```python
%matplotlib inline

import matplotlib.pyplot as plt

plt.imshow(data.get_data()[:,:,50])
```


Try slicing different dimensions (x,y,z) yourself to get a feel for how the data is represented in this anatomical image.

We can also access data from the image header. Let's assign the header of an image to a variable and print it to view it's contents.



{:.input_area}
```python
header = data.header
print(header)      
```


Some of the important information in the header is information about the orientation of the image in space. This can be represented as the affine matrix.



{:.input_area}
```python
data.affine
```


The affine matrix is a way to transform images between spaces.

Here is a short tutorial on affine transformations from the nibabel [documentation](https://nipy.org/nibabel/coordinate_systems.html).  

Here is another nice [tutorial](https://nilearn.github.io/auto_examples/04_manipulating_images/plot_affine_transformation.html#sphx-glr-auto-examples-04-manipulating-images-plot-affine-transformation-py) from nilearn in 2D space.

We have voxel coordinates (in voxel space).  We want to get scanner RAS+
coordinates corresponding to the voxel coordinates.  We need a *coordinate
transform* to take us from voxel coordinates to scanner RAS+ coordinates.

In general, we have some voxel space coordinate $(i, j, k)$, and we want to
generate the reference space coordinate $(x, y, z)$.

Imagine we had solved this, and we had a coordinate transform function $f$
that accepts a voxel coordinate and returns a coordinate in the reference
space:

$(x, y, z) = f(i, j, k)$

$f$ accepts a coordinate in the *input* space and returns a coordinate in the
*output* space.  In our case the input space is voxel space and the output
space is scanner RAS+.

In theory $f$ could be a complicated non-linear function, but in practice, we
know that the scanner collects data on a regular grid.  This means that the
relationship between $(i, j, k)$ and $(x, y, z)$ is linear (actually
*affine*), and can be encoded with linear (actually affine) transformations
comprising translations, rotations and zooms [wikipedia linear transform](https://en.wikipedia.org/wiki/Linear_map)

Scaling (zooming) in three dimensions can be represented by a diagonal 3 by 3
matrix.  Here's how to zoom the first dimension by $p$, the second by $q$ and
the third by $r$ units:

$
\begin{bmatrix}
x\\
y\\
z\\
\end{bmatrix} =
\begin{bmatrix}
p i\\
q j\\
r k\\
\end{bmatrix} =
\begin{bmatrix}
p & 0 & 0 \\
0 & q & 0 \\
0 & 0 & r \\
\end{bmatrix}
\begin{bmatrix}
i\\
j\\
k\\
\end{bmatrix}
$

A rotation in three dimensions can be represented as a 3 by 3 *rotation
matrix* [wikipedia rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix).  For example, here is a rotation by
$\theta$ radians around the third array axis:

$
\begin{bmatrix}
x\\
y\\
z\\
\end{bmatrix} =
\begin{bmatrix}
\cos(\theta) &  -\sin(\theta) & 0 \\
\sin(\theta) & \cos(\theta) & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
i\\
j\\
k\\
\end{bmatrix}
$

This is a rotation by $\phi$ radians around the second array axis:

$
\begin{bmatrix}
x\\
y\\
z\\
\end{bmatrix} =
\begin{bmatrix}
\cos(\phi) & 0 & \sin(\phi) \\
0 & 1 & 0 \\
-\sin(\phi) & 0 & \cos(\phi) \\
\end{bmatrix}
\begin{bmatrix}
i\\
j\\
k\\
\end{bmatrix}
$

A rotation of $\gamma$ radians around the first array axis:

$
\begin{bmatrix}
x\\
y\\
z\\
\end{bmatrix} =
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos(\gamma) & -\sin(\gamma) \\
0 & \sin(\gamma) & \cos(\gamma) \\
\end{bmatrix}
\begin{bmatrix}
i\\
j\\
k\\
\end{bmatrix}
$

Zoom and rotation matrices can be combined by matrix multiplication.

Here's a scaling of $p, q, r$ units followed by a rotation of $\theta$ radians
around the third axis followed by a rotation of $\phi$ radians around the
second axis:

$
\begin{bmatrix}
x\\
y\\
z\\
\end{bmatrix} =
\begin{bmatrix}
\cos(\phi) & 0 & \sin(\phi) \\
0 & 1 & 0 \\
-\sin(\phi) & 0 & \cos(\phi) \\
\end{bmatrix}
\begin{bmatrix}
\cos(\theta) &  -\sin(\theta) & 0 \\
\sin(\theta) & \cos(\theta) & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
p & 0 & 0 \\
0 & q & 0 \\
0 & 0 & r \\
\end{bmatrix}
\begin{bmatrix}
i\\
j\\
k\\
\end{bmatrix}
$

This can also be written:


$
M =
\begin{bmatrix}
\cos(\phi) & 0 & \sin(\phi) \\
0 & 1 & 0 \\
-\sin(\phi) & 0 & \cos(\phi) \\
\end{bmatrix}
\begin{bmatrix}
\cos(\theta) &  -\sin(\theta) & 0 \\
\sin(\theta) & \cos(\theta) & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
p & 0 & 0 \\
0 & q & 0 \\
0 & 0 & r \\
\end{bmatrix}
$

$
\begin{bmatrix}
x\\
y\\
z\\
\end{bmatrix} = M
\begin{bmatrix}
i\\
j\\
k\\
\end{bmatrix}
$

This might be obvious because the matrix multiplication is the result of
applying each transformation in turn on the coordinates output from the
previous transformation. Combining the transformations into a single matrix
$M$ works because matrix multiplication is associative -- $ABCD = (ABC)D$.

A translation in three dimensions can be represented as a length 3 vector to
be added to the length 3 coordinate.  For example, a translation of $a$ units
on the first axis, $b$ on the second and $c$ on the third might be written
as:

$
\begin{bmatrix}
x\\
y\\
z\\
\end{bmatrix} =
\begin{bmatrix}
i\\
j\\
k\\
\end{bmatrix} +
\begin{bmatrix}
a \\
b \\
c \\
\end{bmatrix}
$

We can write our function $f$ as a combination of matrix multiplication by
some 3 by 3 rotation / zoom matrix $M$ followed by addition of a 3 by 1
translation vector $(a, b, c)$

$
\begin{bmatrix}
x\\
y\\
z\\
\end{bmatrix} = M
\begin{bmatrix}
i\\
j\\
k\\
\end{bmatrix} +
\begin{bmatrix}
a\\
b\\
c\\
\end{bmatrix}
$

We could record the parameters necessary for $f$ as the 3 by 3 matrix, $M$
and the 3 by 1 vector $(a, b, c)$.

In fact, the 4 by 4 image *affine array* does include exactly this
information. If $m_{i,j}$ is the value in row $i$ column $j$ of matrix $M$,
then the image affine matrix $A$ is:

$
A =
\begin{bmatrix}
m_{1,1} & m_{1,2} & m_{1,3} & a \\
m_{2,1} & m_{2,2} & m_{2,3} & b \\
m_{3,1} & m_{3,2} & m_{3,3} & c \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$

Why the extra row of $[0, 0, 0, 1]$?  We need this row because we have
rephrased the combination of rotations / zooms and translations as a
transformation in *homogenous coordinates* (see [wikipedia homogenous
coordinates](https://en.wikipedia.org/wiki/Homogeneous_coordinates)).  This is a trick that allows us to put the translation part into the same matrix as the rotations / zooms, so that both translations and
rotations / zooms can be applied by matrix multiplication.  In order to make
this work, we have to add an extra 1 to our input and output coordinate
vectors:

$
\begin{bmatrix}
x\\
y\\
z\\
1\\
\end{bmatrix} =
\begin{bmatrix}
m_{1,1} & m_{1,2} & m_{1,3} & a \\
m_{2,1} & m_{2,2} & m_{2,3} & b \\
m_{3,1} & m_{3,2} & m_{3,3} & c \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
i\\
j\\
k\\
1\\
\end{bmatrix}
$

This results in the same transformation as applying $M$ and $(a, b, c)$
separately. One advantage of encoding transformations this way is that we can
combine two sets of [rotations, zooms, translations] by matrix multiplication
of the two corresponding affine matrices.

In practice, although it is common to combine 3D transformations using 4 by 4
affine matrices, we usually *apply* the transformations by breaking up the
affine matrix into its component $M$ matrix and $(a, b, c)$ vector and doing:

$
\begin{bmatrix}
x\\
y\\
z\\
\end{bmatrix} = M
\begin{bmatrix}
i\\
j\\
k\\
\end{bmatrix} +
\begin{bmatrix}
a\\
b\\
c\\
\end{bmatrix}
$

As long as the last row of the 4 by 4 is $[0, 0, 0, 1]$, applying the
transformations in this way is mathematically the same as using the full 4 by
4 form, without the inconvenience of adding the extra 1 to our input and
output vectors.


You can think of the image affine as a combination of a series of
transformations to go from voxel coordinates to mm coordinates in terms of the
magnet isocenter.  Here is the EPI affine broken down into a series of
transformations, with the results shown on the localizer image:

<img src="https://nipy.org/nibabel/_images/illustrating_affine.png" />

Applying different affine transformations allows us to rotate, reflect, scale, and shear the image.

For example, let's try to reflect the image so that it is facing the opposite direction.



{:.input_area}
```python
import numpy as np
from nibabel.affines import apply_affine, from_matvec, to_matvec

reflect = np.array([[-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

reflect_affine = from_matvec(reflect)
reflect_affine
```


Now maybe we would like to shift the image 10 units in the x direction.



{:.input_area}
```python
translate_affine = from_matvec(reflect, [10, 0, 0])
translate_affine
```




{:.input_area}
```python
transformed = np.dot(data.get_data(),translate_affine)
plt.imshow(transformed[:,:,50])
```


What if we wanted to make the brain smaller by applying a scaling transformation?



{:.input_area}
```python
scaling_affine = np.array([[3, 0, 0, 0],
                           [0, 3, 0, 0],
                           [0, 0, 3, 0],
                           [0, 0, 0, 1]])

scaling_affine


```


How about applying a rotation of 0.3 radians?



{:.input_area}
```python
cos_gamma = np.cos(0.3)
sin_gamma = np.sin(0.3)
rotation_affine = np.array([[1, 0, 0, 0],
                            [0, cos_gamma, -sin_gamma, 0],
                            [0, sin_gamma, cos_gamma, 0],
                            [0, 0, 0, 1]])
```


## Nilearn
There are many useful tools from the [nilearn](https://nilearn.github.io/index.html) library to help manipulate and visualize neuroimaging data. See their [documentation](https://nilearn.github.io/plotting/index.html#different-plotting-functions) for an example.

Let's make sure it is installed first.



{:.input_area}
```python
!pip install nilearn
```


Now let's load a few different plotting functions from their plotting module



{:.input_area}
```python
from nilearn.plotting import view_img, glass_brain, plot_anat, plot_epi
```




{:.input_area}
```python
we can make simple 
```




{:.input_area}
```python
%matplotlib inline

plot_anat(data)
```


Nilearn plotting functions are very flexible and allow us to easily customize our plots



{:.input_area}
```python
plot_anat(data, draw_cross=False, display_mode='z')
```


try to get more information how to use the function with `?` and try to add different commands to change the plot

nilearn also has a neat interactive viewer called `view_img` for examining images directly in the notebook. 



{:.input_area}
```python
view_img(data)
```


The `view_img` function is particularly useful for overlaying statistical maps over an anatomical image so that we can interactively examine where the results are located.

As an example, let's load a mask of the amygdala and try to find where it is located.



{:.input_area}
```python
amygdala_mask = nib.load(os.path.join(base_dir,  'FSL_BAmyg_thr0.nii.gz'))
view_img(amygdala_mask, data)
```


We can also load 4D data such as a series of epi images.  Here we will load a short functional run from the same particiapnts.



{:.input_area}
```python
epi = nib.load(os.path.join(base_dir, 'raw_fMRI_raw_bold.nii.gz'))
print(epi.shape)
```


For some reason nibabel is reading this nifti as a 5D image.  Let's quickly remove the extra dimension.



{:.input_area}
```python
epi_data = epi.get_fdata()
epi_data = epi_data.squeeze()
print(epi_data.shape)
```


Ok, looks like this fixed the dimensions.

Now, let's plot the average voxel signal intensity across the whole brain for each of the 128 TRs.



{:.input_area}
```python
plt.plot(np.mean(epi_data,axis=(0,1,2)))
```


Notice the slow linear drift over time, where the global signal intensity gradually decreases.
