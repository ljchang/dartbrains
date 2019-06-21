---
redirect_from:
  - "/features/notebooks/download-data"
interact_link: content/features/notebooks/Download_Data.ipynb
kernel_name: conda-env-py36-py
title: 'Download Localizer Data'
prev_page:
  url: /features/notebooks/2_Introduction_to_Dataframes_Plotting
  title: 'Introduction to Dataframes and Plotting'
next_page:
  url: /features/notebooks/3_Introduction_to_NeuroimagingData_in_Python
  title: 'Introduction to Neuroimaging Data'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# Download Localizer Data

Many of the imaging tutorials will use open data from the Pinel Localizer task.

The Pinel Localizer task was designed to probe several different types of basic cognitive processes, such as visual perception, finger tapping, language, and math. Several of the tasks are cued by reading text on the screen (i.e., visual modality) and also by hearing auditory instructions (i.e., auditory modality). The trials are randomized across conditions and have been optimized to maximize efficiency for a rapid event related design. There are 100 trials in total over a 5-minute scanning session. Read the [paper](https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-8-91) for more specific details about the task. You can also view the [Data](http://brainomics.cea.fr/localizer/dataset?rql=Any%20X%2C%20XT%2C%20XL%2C%20XI%2C%20XF%2C%20XD%20WHERE%20X%20is%20Scan%2C%20X%20type%20XT%2C%20X%20label%20XL%2C%20X%20identifier%20XI%2C%20X%20format%20XF%2C%20X%20description%20XD) on the author’s cubicweb interface.

This dataset is well suited for these tutorials as it is (a) publicly available to anyone in the world, (b) relatively small (only about 5min), and (c) provides many options to create different types of contrasts.

There are a total of 94 subjects available, but we will primarily only be working with a smaller subset of about 30.

Downloading the data is very easy.  All we need to do is specify a directory to download and unpack it.



{:.input_area}
```python
import pandas as pd
from nltools.data import Brain_Data
from nltools.datasets import fetch_localizer

data_dir = '/Users/lukechang/nilearn_data/brainomics_localizer/'
```


We first will create a list of subject identification numbers from 's01' to 's30'.  Then we will download all of the subjects one by one using the `fetch_localizer` function from `nltools.datasets`.  You can easily modify this code to download more or less subjects.

We can choose to download either the 'raw' or 'preprocessed' data. Feel free to download both, but you will have to change the `data_type` and rerun the function. 

We can also choose if we want to download the anatomical images. This is useful for running preprocessing later. 

If you are eager to get going and don't plan on preprocessing the data yourself, feel free to set `get_anats=False, data_type='preprocessed'` to download the bare minimum. 

Don't forget to set `data_dir=data_dir` if you want to put the data in a specific folder.  Otherwise it will default to `~/nilearn_data/brainomics_localizer/`.




{:.input_area}
```python
sub_ids = [f'S{x:02}' for x in np.arange(1,31)]

f = fetch_localizer(subject_ids=sub_ids, get_anats=True, data_type='raw')
```


{:.output .output_stream}
```
Downloading data from http://brainomics.cea.fr/localizer/dataset/cubicwebexport.csv ...
Downloading data from http://brainomics.cea.fr/localizer/dataset/cubicwebexport2.csv ...

```

Ok, we have now downloaded all of the data!

The `f` variable contains a dictionary of all of the relevant data.  

 - `f['functional']`: contains a list of the functional data.
 - `f['structural']`: contains a list of the structural data.
 - `f['ext_vars']`: contains all of the subject metadata
 - `f['description']`: contains a brief description of the dataset.



{:.input_area}
```python
print(f['description'].decode())
```


{:.output .output_stream}
```
Brainomics Localizer


Notes
-----
A protocol that captures the cerebral bases of auditory and
visual perception, motor actions, reading, language comprehension
and mental calculation at an individual level. Individual functional
maps are reliable and quite precise.


Content
-------
    :'func': Nifti images of the neural activity maps
    :'cmaps': Nifti images of contrast maps
    :'tmaps': Nifti images of corresponding t-maps
    :'masks': Structural images of the mask used for each subject.
    :'anats': Structural images of anatomy of each subject

References
----------
For more information about this dataset's structure:
http://brainomics.cea.fr/localizer/

To cite this dataset:
Papadopoulos Orfanos, Dimitri, et al.
"The Brainomics/Localizer database."
NeuroImage 144.B (2017): 309.

For an example of scientific results obtained using this dataset:
Pinel, Philippe, et al.
"Fast reproducible identification and large-scale databasing of
 individual functional cognitive networks."
BMC Neuroscience 8.1 (2007): 91.

Licence: usage is unrestricted for non-commercial research purposes.


```

Now let's take a look inside the metadata file.

It is a pandas data frame with info about each participant.





{:.input_area}
```python
f['ext_vars'].head()
```


Ok, now let's try to load one of the functional datasets using `Brain_Data`.



{:.input_area}
```python
data = Brain_Data(f['functional'][0])
```


{:.output .output_traceback_line}
```

    ---------------------------------------------------------------------------

    DimensionError                            Traceback (most recent call last)

    <ipython-input-56-d493f847f938> in <module>
    ----> 1 data = Brain_Data(f['functional'][0])
    

    ~/anaconda3/envs/py36/lib/python3.6/site-packages/nltools-0.3.13-py3.6.egg/nltools/data/brain_data.py in __init__(self, data, Y, X, mask, output_file, **kwargs)
        130                             if isinstance(i, six.string_types):
        131                                 self.data.append(self.nifti_masker.fit_transform(
    --> 132                                                  nib.load(i)))
        133                             elif isinstance(i, nib.Nifti1Image):
        134                                 self.data.append(self.nifti_masker.fit_transform(i))


    ~/anaconda3/envs/py36/lib/python3.6/site-packages/nilearn/input_data/base_masker.py in fit_transform(self, X, y, confounds, **fit_params)
        205                                 ).transform(X, confounds=confounds)
        206             else:
    --> 207                 return self.fit(**fit_params).transform(X, confounds=confounds)
        208         else:
        209             # fit method of arity 2 (supervised transformation)


    ~/anaconda3/envs/py36/lib/python3.6/site-packages/nilearn/input_data/base_masker.py in transform(self, imgs, confounds)
        175         self._check_fitted()
        176 
    --> 177         return self.transform_single_imgs(imgs, confounds)
        178 
        179     def fit_transform(self, X, y=None, confounds=None, **fit_params):


    ~/anaconda3/envs/py36/lib/python3.6/site-packages/nilearn/input_data/nifti_masker.py in transform_single_imgs(self, imgs, confounds, copy)
        306             confounds=confounds,
        307             copy=copy,
    --> 308             dtype=self.dtype
        309         )
        310 


    ~/anaconda3/envs/py36/lib/python3.6/site-packages/sklearn/externals/joblib/memory.py in __call__(self, *args, **kwargs)
        340 
        341     def __call__(self, *args, **kwargs):
    --> 342         return self.func(*args, **kwargs)
        343 
        344     def call_and_shelve(self, *args, **kwargs):


    ~/anaconda3/envs/py36/lib/python3.6/site-packages/nilearn/input_data/nifti_masker.py in filter_and_mask(imgs, mask_img_, parameters, memory_level, memory, verbose, confounds, copy, dtype)
         36                     copy=True,
         37                     dtype=None):
    ---> 38     imgs = _utils.check_niimg(imgs, atleast_4d=True, ensure_ndim=4)
         39 
         40     # Check whether resampling is truly necessary. If so, crop mask


    ~/anaconda3/envs/py36/lib/python3.6/site-packages/nilearn/_utils/niimg_conversions.py in check_niimg(niimg, ensure_ndim, atleast_4d, dtype, return_iterator, wildcards)
        282 
        283     if ensure_ndim is not None and len(niimg.shape) != ensure_ndim:
    --> 284         raise DimensionError(len(niimg.shape), ensure_ndim)
        285 
        286     if return_iterator:


    DimensionError: Input data has incompatible dimensionality: Expected dimension is 4D and you provided a 5D image. See http://nilearn.github.io/manipulating_images/input_output.html.


```

Uh, oh...  Looks like this data was created using an older version of SPM and isn't being read correctly by `Brain_Data`.

Let's do our first bit of debugging and fix this.

We will use nibabel to load the data and inspect the shape of the data file.



{:.input_area}
```python
import nibabel as nib

dat = nib.load(f['functional'][0][0])
dat.shape
```





{:.output .output_data_text}
```
(64, 64, 40, 1, 128)
```



The first 3 dimensions correspond to X, Y, & Z planes.  The last dimension is time. It looks like there is an extra dimension that we will need to get rid of so that we can load the data correctly.

We will get the data into a numpy array and use the `squeeze()` method to collapse the extra dimension.



{:.input_area}
```python
dat.get_data().squeeze().shape
```





{:.output .output_data_text}
```
(64, 64, 40, 128)
```



See how we now have the correct shape?

Let's now create a new nibabel instance with the correct dimensions and write over the file.

To create a new nibabel instance, we just need to pass in the 4-D numpy array with the affine matrix.



{:.input_area}
```python
dat.affine
```





{:.output .output_data_text}
```
array([[ -3.,   0.,   0.,  96.],
       [  0.,   3.,   0., -96.],
       [  0.,   0.,   3., -60.],
       [  0.,   0.,   0.,   1.]])
```





{:.input_area}
```python
dat_fixed = nib.Nifti1Image(dat.get_data().squeeze(), dat.affine)
nib.save(dat_fixed, f['functional'][0][0])
```


Let's check and see if we can correctly load the file now.



{:.input_area}
```python
data = Brain_Data(f['functional'][0])
data
```





{:.output .output_data_text}
```
nltools.data.brain_data.Brain_Data(data=(128, 238955), Y=0, X=(0, 0), mask=MNI152_T1_2mm_brain_mask.nii.gz, output_file=[])
```



Great!  Looks like we fixed it.  Let's now fix all of the rest of the images so that we won't have this problem again.



{:.input_area}
```python
for file_name in f['functional']:
    dat = nib.load(file_name[0])
    if len(dat.shape) > 4:
        dat_fixed = nib.Nifti1Image(dat.get_data().squeeze(), dat.affine)
        nib.save(dat_fixed, file_name[0])
```

