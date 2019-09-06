---
redirect_from:
  - "/features/notebooks/7-nipype-preprocessing"
interact_link: content/features/notebooks/7_Nipype_Preprocessing.ipynb
kernel_name: conda-env-py36-py
has_widgets: false
title: 'Building Preprocessing Workflows with Nipype'
prev_page:
  url: /features/notebooks/6_Nipype_Quickstart.html
  title: 'Preprocessing with Nipype Quickstart'
next_page:
  url: /features/notebooks/8_fmriprep_tutorial.html
  title: 'Introduction to Automated Preprocessing with fmriprep'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# Hands-on 1: How to create a fMRI preprocessing workflow
*Written by Michael Notter & Luke Chang*

The purpose of this section is that you set-up a complete fMRI analysis workflow yourself. So that in the end, you are able to perform the analysis from A-Z, i.e. from preprocessing to group analysis. This section will cover the preprocessing part, and the section [Hands-on 2: Analysis](handson_analysis.ipynb) will handle the analysis part.

We will use this opportunity to show you some nice additional interfaces/nodes that might not be relevant to your usual analysis. But it's always nice to know that they exist. And hopefully, this will encourage you to investigate all other interfaces that Nipype can bring to the tip of your finger.

This notebook was taken from a more comprehensive [tutorial](https://github.com/miykael/nipype_tutorial) on nipype. I encourage you to check it out to learn about this tool in more depth.

## Preparation

Before we can start with anything we first need to download the data. For this hands-on, we will only use the right-handed subjects 2-4 and 7-9. This can be done very quickly with the following `datalad` command.

**Note:** This might take a while, as datalad needs to download ~200MB of data

**Note: You only need to download the data if you are running this locally on your laptop.  The data is already available on jupyterhub!**



{:.input_area}
```bash
%%bash
datalad get -J 4 /data/ds000114/sub-0[234789]/ses-test/anat/sub-0[234789]_ses-test_T1w.nii.gz \
                /data/ds000114/sub-0[234789]/ses-test/func/*fingerfootlips*
```


# Preprocessing Workflow Structure

So let's get our hands dirty. First things first, it's always good to know which interfaces you want to use in your workflow and in which order you want to execute them. For the preprocessing workflow, I recommend that we use the following nodes:

     1. Gunzip (Nipype)
     2. Drop Dummy Scans (FSL)
     3. Slice Time Correction (SPM)
     4. Motion Correction (SPM)
     5. Artifact Detection
     6. Segmentation (SPM)
     7. Coregistration (FSL)
     8. Smoothing (FSL)
     9. Apply Binary Mask (FSL)
    10. Remove Linear Trends (Nipype)
    
**Note:** This workflow might be overkill concerning data manipulation, but it hopefully serves as a good Nipype exercise.

## Imports

It's always best to have all relevant module imports at the beginning of your script. So let's import what we most certainly need.



{:.input_area}
```python
%matplotlib inline

from nilearn import plotting

# Get the Node and Workflow object
from nipype import Node, Workflow

# Specify which SPM to use
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('/optnfs/el7/spm/spm12')

from os.path import join

your_netid = 'f00275v'
data_dir = '/dartfs/rc/lab/P/Psych60/data/ds000114'
output_dir = '/dartfs/rc/lab/P/Psych60/students_output/%s/7_nipype_preprocessing' % your_netid
# data_dir = '/dartfs-hpc/rc/home/v/f00275v/Psych60/data/brainomics_data'
```


**Note:** Ideally you would also put the imports of all the interfaces that you use here at the top. But as we will develop the workflow step by step, we can also import the relevant modules as we go.

## Create Nodes and Workflow connections

Let's create all the nodes that we need! Make sure to specify all relevant inputs and keep in mind which ones you later on need to connect in your pipeline.

### Workflow

We recommend to create the workflow and establish all its connections at a later place in your script. This helps to have everything nicely together. But for this hands-on example, it makes sense to establish the connections between the nodes as we go.

And for this, we first need to create a workflow:



{:.input_area}
```python
# Create the workflow here
# Hint: use 'base_dir' to specify where to store the working directory
```




{:.input_area}
```python
preproc = Workflow(name='work_preproc', base_dir=join(output_dir, 'output/'))
```


### Gunzip

I've already created the `Gunzip` node as a template for the other nodes. Also, we've specified an `in_file` here so that we can directly test the nodes without worrying about the Input/Output data stream to the workflow. This will be taken care of in a later section.



{:.input_area}
```python
from nipype.algorithms.misc import Gunzip
```




{:.input_area}
```python
# Specify example input file
func_file = join(data_dir, 'sub-07/ses-test/func/sub-07_ses-test_task-fingerfootlips_bold.nii.gz')

# Initiate Gunzip node
gunzip_func = Node(Gunzip(in_file=func_file), name='gunzip_func')
```


### Drop Dummy Scans

The functional images of this dataset were recorded with 4 dummy scans at the beginning (see the [corresponding publication](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3641991/)). But those dummy scans were not yet taken out from the functional images.

To better illustrate this, let's plot the time course of a random voxel of the just defined `func_file`:



{:.input_area}
```python
%matplotlib inline

import nibabel as nb
import matplotlib.pyplot as plt
plt.plot(nb.load(func_file).get_fdata()[32, 32, 15, :]);
```



{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_14_0.png)



In the figure above, we see that at the very beginning there are extreme values, which hint to the fact that steady state wasn't reached yet. Therefore, we want to exclude the dummy scans from the original data. This can be achieved with FSL's `ExtractROI`.



{:.input_area}
```python
from nipype.interfaces.fsl import ExtractROI
```




{:.input_area}
```python
extract = Node(ExtractROI(t_min=4, t_size=-1, output_type='NIFTI'),
               name="extract")
```


This `ExtractROI` node can now be connected to the `gunzip_func` node from above. To do this, we use the following command:



{:.input_area}
```python
preproc.connect([(gunzip_func, extract, [('out_file', 'in_file')])])
```


{:.output .output_traceback_line}
```

    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-15-eee3193e21b8> in <module>
    ----> 1 preproc.connect([(gunzip_func, extract, [('out_file', 'in_file')])])
    

    /optnfs/el7/jupyterhub/envs/Psych60-Chang/lib/python3.7/site-packages/nipype/pipeline/engine/workflows.py in connect(self, *args, **kwargs)
        178 Trying to connect %s:%s to %s:%s but input '%s' of node '%s' is already
        179 connected.
    --> 180 """ % (srcnode, source, destnode, dest, dest, destnode))
        181                 if not (hasattr(destnode, '_interface') and
        182                         ('.io' in str(destnode._interface.__class__) or any([


    Exception: Trying to connect work_preproc.gunzip_func:out_file to work_preproc.extract:in_file but input 'in_file' of node 'work_preproc.extract' is already
    connected.



```

### Slice Time Correction

Now to the next step. Let's us SPM's `SliceTiming` to correct for slice wise acquisition of the volumes. As a reminder, the tutorial dataset was recorded...
- with a time repetition (TR) of 2.5 seconds
- with 30 slices per volume
- in an interleaved fashion, i.e. slice order is [1, 3, 5, 7, ..., 2, 4, 6, ..., 30]
- with a time acquisition (TA) of 2.4167 seconds, i.e. `TR-(TR/num_slices)`



{:.input_area}
```python
from nipype.interfaces.spm import SliceTiming
```




{:.input_area}
```python
slice_order = list(range(1, 31, 2)) + list(range(2, 31, 2))
print(slice_order)
```


{:.output .output_stream}
```
[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

```



{:.input_area}
```python
# Initiate SliceTiming node here
```




{:.input_area}
```python
slicetime = Node(SliceTiming(num_slices=30,
                             ref_slice=15,
                             slice_order=slice_order,
                             time_repetition=2.5,
                             time_acquisition=2.5-(2.5/30)),
                 name='slicetime')
```


Now the next step is to connect the `SliceTiming` node to the rest of the workflow, i.e. the `ExtractROI` node.



{:.input_area}
```python
# Connect SliceTiming node to the other nodes here
```




{:.input_area}
```python
preproc.connect([(extract, slicetime, [('roi_file', 'in_files')])])
```


### Motion Correction

To correct for motion in the scanner, we will be using FSL's `MCFLIRT`.



{:.input_area}
```python
from nipype.interfaces.fsl import MCFLIRT
```




{:.input_area}
```python
# Initate MCFLIRT node here
```




{:.input_area}
```python
mcflirt = Node(MCFLIRT(mean_vol=True,
                       save_plots=True),
               name="mcflirt")
```


Connect the `MCFLIRT` node to the rest of the workflow.



{:.input_area}
```python
# Connect MCFLIRT node to the other nodes here
```




{:.input_area}
```python
preproc.connect([(slicetime, mcflirt, [('timecorrected_files', 'in_file')])])
```


### Artifact Detection

We will use the really cool and useful `ArtifactDetection` tool from Nipype to detect motion and intensity outliers in the functional images. The interface is initiated as follows:



{:.input_area}
```python
from nipype.algorithms.rapidart import ArtifactDetect
```




{:.input_area}
```python
art = Node(ArtifactDetect(norm_threshold=2,
                          zintensity_threshold=3,
                          mask_type='spm_global',
                          parameter_source='FSL',
                          use_differences=[True, False],
                          plot_type='svg'),
           name="art")
```


The parameters above mean the following:
- `norm_threshold` - Threshold to use to detect motion-related outliers when composite motion is being used
- `zintensity_threshold` - Intensity Z-threshold use to detection images that deviate from the mean
- `mask_type` - Type of mask that should be used to mask the functional data. *spm_global* uses an spm_global like calculation to determine the brain mask
- `parameter_source` - Source of movement parameters
- `use_differences` - If you want to use differences between successive motion (first element) and intensity parameter (second element) estimates in order to determine outliers

And this is how you connect this node to the rest of the workflow:



{:.input_area}
```python
preproc.connect([(mcflirt, art, [('out_file', 'realigned_files'),
                                 ('par_file', 'realignment_parameters')])
                 ])
```


### Segmentation of anatomical image

Now let's work on the anatomical image. In particular, let's use SPM's `NewSegment` to create probability maps for the gray matter, white matter tissue and CSF.



{:.input_area}
```python
from nipype.interfaces.spm import NewSegment
```




{:.input_area}
```python
# Use the following tissue specification to get a GM and WM probability map

tpm_img ='/optnfs/el7/spm/spm12/tpm/TPM.nii'
tissue1 = ((tpm_img, 1), 1, (True,False), (False, False))
tissue2 = ((tpm_img, 2), 1, (True,False), (False, False))
tissue3 = ((tpm_img, 3), 2, (True,False), (False, False))
tissue4 = ((tpm_img, 4), 3, (False,False), (False, False))
tissue5 = ((tpm_img, 5), 4, (False,False), (False, False))
tissue6 = ((tpm_img, 6), 2, (False,False), (False, False))
tissues = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]
```




{:.input_area}
```python
# Initiate NewSegment node here
```




{:.input_area}
```python
segment = Node(NewSegment(tissues=tissues), name='segment')
```


We will again be using a `Gunzip` node to unzip the anatomical image that we then want to use as input to the segmentation node. We again also need to specify the anatomical image that we want to use in this case. As before, this will later also be handled directly by the Input/Output stream.



{:.input_area}
```python
# Specify example input file
anat_file = join(data_dir, 'sub-07/ses-test/anat/sub-07_ses-test_T1w.nii.gz')

# Initiate Gunzip node
gunzip_anat = Node(Gunzip(in_file=anat_file), name='gunzip_anat')
```


Now we can connect the `NewSegment` node to the rest of the workflow.



{:.input_area}
```python
# Connect NewSegment node to the other nodes here
```




{:.input_area}
```python
preproc.connect([(gunzip_anat, segment, [('out_file', 'channel_files')])])
```


### Compute Coregistration Matrix

As a next step, we will make sure that the functional images are coregistered to the anatomical image. For this, we will use FSL's `FLIRT` function. As we just created a white matter probability map, we can use this together with the Boundary-Based Registration (BBR) cost function to optimize the image coregistration. As some helpful notes...
- use a degree of freedom of 6
- specify the cost function as `bbr`
- use the `schedule='/usr/share/fsl/5.0/etc/flirtsch/bbr.sch'`



{:.input_area}
```python
from nipype.interfaces.fsl import FLIRT
```




{:.input_area}
```python
# Initiate FLIRT node here
```




{:.input_area}
```python
coreg = Node(FLIRT(dof=6,
                   cost='bbr',
                   schedule='/optnfs/el7/fsl/5.0.10/etc/flirtsch/bbr.sch',
                   output_type='NIFTI'),
             name="coreg")
```




{:.input_area}
```python
# Connect FLIRT node to the other nodes here
```




{:.input_area}
```python
preproc.connect([(gunzip_anat, coreg, [('out_file', 'reference')]),
                 (mcflirt, coreg, [('mean_img', 'in_file')])
                 ])
```


As mentioned above, the `bbr` routine can use the subject-specific white matter probability map to guide the coregistration. But for this, we need to create a binary mask out of the WM probability map. This can easily be done by FSL's `Threshold` interface.



{:.input_area}
```python
from nipype.interfaces.fsl import Threshold

# Threshold - Threshold WM probability image
threshold_WM = Node(Threshold(thresh=0.5,
                              args='-bin',
                              output_type='NIFTI'),
                name="threshold_WM")
```


Now, to select the WM probability map that the `NewSegment` node created, we need some helper function. Because the output field `partial_volume_files` form the segmentation node, will give us a list of files, i.e. `[[GM_prob], [WM_prob], [], [], [], []]`. Therefore, using the following function, we can select only the last element of this list.



{:.input_area}
```python
# Select WM segmentation file from segmentation output
def get_wm(files):
    return files[1][0]

# Connecting the segmentation node with the threshold node
preproc.connect([(segment, threshold_WM, [(('native_class_images', get_wm),
                                           'in_file')])])
```


Now we can just connect this `Threshold` node to the coregistration node from above.



{:.input_area}
```python
# Connect Threshold node to coregistration node above here
```




{:.input_area}
```python
preproc.connect([(threshold_WM, coreg, [('out_file', 'wm_seg')])])
```


### Apply Coregistration Matrix to functional image

Now that we know the coregistration matrix to correctly overlay the functional mean image on the subject-specific anatomy, we need to apply to coregistration to the whole time series. This can be achieved with FSL's `FLIRT` as follows:



{:.input_area}
```python
# Specify the isometric voxel resolution you want after coregistration
desired_voxel_iso = 4

# Apply coregistration warp to functional images
applywarp = Node(FLIRT(interp='spline',
                       apply_isoxfm=desired_voxel_iso,
                       output_type='NIFTI'),
                 name="applywarp")
```


**<span style="color:red">Important</span>**: As you can see above, we also specified a variable `desired_voxel_iso`. This is very important at this stage, otherwise `FLIRT` will transform your functional images to a resolution of the anatomical image, which will dramatically increase the file size (e.g. to 1-10GB per file). If you don't want to change the voxel resolution, use the additional parameter `no_resample=True`. Important, for this to work, you still need to define `apply_isoxfm`.



{:.input_area}
```python
# Connecting the ApplyWarp node to all the other nodes
preproc.connect([(mcflirt, applywarp, [('out_file', 'in_file')]),
                 (coreg, applywarp, [('out_matrix_file', 'in_matrix_file')]),
                 (gunzip_anat, applywarp, [('out_file', 'reference')])
                 ])
```


### Smoothing

Next step is image smoothing. The most simple way to do this is to use FSL's or SPM's `Smooth` function. But for learning purposes, let's use FSL's `SUSAN` workflow as it is implemented in Nipype. Note that this time, we are importing a workflow instead of an interface.



{:.input_area}
```python
from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth
```


If you type `create_susan_smooth?` you can see how to specify the input variables to the susan workflow. In particular, they are...
- `fwhm`: set this value to 4 (or whichever value you want)
- `mask_file`: will be created in a later step
- `in_file`: will be handled while connection to other nodes in the preproc workflow



{:.input_area}
```python
# Initiate SUSAN workflow here
```




{:.input_area}
```python
susan = create_susan_smooth(name='susan')
susan.inputs.inputnode.fwhm = 4
```




{:.input_area}
```python
# Connect Threshold node to coregistration node above here
```




{:.input_area}
```python
preproc.connect([(applywarp, susan, [('out_file', 'inputnode.in_files')])])
```


### Create Binary Mask

There are many possible approaches on how you can mask your functional images. One of them is not at all, one is with a simple brain mask and one that only considers certain kind of brain tissue, e.g. gray matter.

For the current example, we want to create a dilated gray matter mask. For this purpose we need to:
1. Resample the gray matter probability map to the same resolution as the functional images
2. Threshold this resampled probability map at a specific value
3. Dilate this mask by some voxels to make the mask less conservative and more inclusive

The first step can be done in many ways (eg. using freesurfer's `mri_convert`, `nibabel`) but in our case, we will use FSL's `FLIRT`. The trick is to use the probability mask, as input file and a reference file.



{:.input_area}
```python
from nipype.interfaces.fsl import FLIRT

# Initiate resample node
resample = Node(FLIRT(apply_isoxfm=desired_voxel_iso,
                      output_type='NIFTI'),
                name="resample")
```


The second and third step can luckily be done with just one node. We can take almost the same `Threshold` node as above. We just need to add another additional argument: `-dilF` - which applies a maximum filtering of all voxels.



{:.input_area}
```python
from nipype.interfaces.fsl import Threshold

# Threshold - Threshold GM probability image
mask_GM = Node(Threshold(thresh=0.5,
                         args='-bin -dilF',
                         output_type='NIFTI'),
                name="mask_GM")

# Select GM segmentation file from segmentation output
def get_gm(files):
    return files[0][0]
```


Now we can connect the resample and the gray matter mask node to the segmentation node and each other.



{:.input_area}
```python
preproc.connect([(segment, resample, [(('native_class_images', get_gm), 'in_file'),
                                      (('native_class_images', get_gm), 'reference')
                                      ]),
                 (resample, mask_GM, [('out_file', 'in_file')])
                 ])
```


This should do the trick.

### Apply the binary mask

Now we can connect this dilated gray matter mask to the susan node, as well as actually applying this to the resulting smoothed images.



{:.input_area}
```python
# Connect gray matter Mask node to the susan workflow here
```




{:.input_area}
```python
preproc.connect([(mask_GM, susan, [('out_file', 'inputnode.mask_file')])])
```


To apply the mask to the smoothed functional images, we will use FSL's `ApplyMask` interface.



{:.input_area}
```python
from nipype.interfaces.fsl import ApplyMask
```


**Important:** The susan workflow gives out a list of files, i.e. `[smoothed_func.nii]` instead of just the filename directly. If we would use a normal `Node` for `ApplyMask` this would lead to the following error:

    TraitError: The 'in_file' trait of an ApplyMaskInput instance must be an existing file name, but a value of ['/output/work_preproc/susan/smooth/mapflow/_smooth0/asub-07_ses-test_task-fingerfootlips_bold_mcf_flirt_smooth.nii.gz'] <class 'list'> was specified.


To prevent this we will be using a `MapNode` and specify the `in_file` as it's iterfield. Like this, the node is capable to handle a list of inputs as it will know that it has to apply itself iteratively to the list of inputs.



{:.input_area}
```python
from nipype import MapNode
```




{:.input_area}
```python
# Initiate ApplyMask node here
```




{:.input_area}
```python
mask_func = MapNode(ApplyMask(output_type='NIFTI'),
                    name="mask_func", 
                    iterfield=["in_file"])
```




{:.input_area}
```python
# Connect smoothed susan output file to ApplyMask node here
```




{:.input_area}
```python
preproc.connect([(susan, mask_func, [('outputnode.smoothed_files', 'in_file')]),
                 (mask_GM, mask_func, [('out_file', 'mask_file')])
                 ])
```


### Remove linear trends in functional images

Last but not least. Let's use Nipype's `TSNR` module to remove linear and quadratic trends in the functionally smoothed images. For this, you only have to specify the `regress_poly` parameter in the node initiation.



{:.input_area}
```python
from nipype.algorithms.confounds import TSNR
```




{:.input_area}
```python
# Initiate TSNR node here
```




{:.input_area}
```python
detrend = Node(TSNR(regress_poly=2), name="detrend")
```




{:.input_area}
```python
# Connect the detrend node to the other nodes here
```




{:.input_area}
```python
preproc.connect([(mask_func, detrend, [('out_file', 'in_file')])])
```


## Datainput with `SelectFiles` and `iterables`

This is all nice and well. But so far we still had to specify the input values for `gunzip_anat` and `gunzip_func` ourselves. How can we scale this up to multiple subjects and/or multiple functional images and make the workflow take the input directly from the BIDS dataset?

For this, we need [`SelectFiles`](../../nipype_tutorial/notebooks/basic_data_input.ipynb#SelectFiles) and [`iterables`](../../nipype_tutorial/notebooks/basic_iteration.ipynb)! It's rather simple, specify a template and fill-up the placeholder variables.



{:.input_area}
```python
# Import the SelectFiles
from nipype import SelectFiles

# String template with {}-based strings
templates = {'anat': 'sub-{subject_id}/ses-{ses_id}/anat/'
                     'sub-{subject_id}_ses-test_T1w.nii.gz',
             'func': 'sub-{subject_id}/ses-{ses_id}/func/'
                     'sub-{subject_id}_ses-{ses_id}_task-{task_id}_bold.nii.gz'}

# Create SelectFiles node
sf = Node(SelectFiles(templates,
                      base_directory=data_dir,
                      sort_filelist=True),
          name='selectfiles')
sf.inputs.ses_id='test'
sf.inputs.task_id='fingerfootlips'
```


Now we can specify over which subjects the workflow should iterate. To test the workflow, let's still just look at subject 7.



{:.input_area}
```python
subject_list = ['07']
sf.iterables = [('subject_id', subject_list)]
```




{:.input_area}
```python
# Connect SelectFiles node to the other nodes here
```




{:.input_area}
```python
preproc.connect([(sf, gunzip_anat, [('anat', 'in_file')]),
                 (sf, gunzip_func, [('func', 'in_file')])])
```


## Visualize the workflow

Now that we're done. Let's look at the workflow that we just created.



{:.input_area}
```python
# Create preproc output graph
preproc.write_graph(graph2use='colored', format='png', simple_form=True)

# Visualize the graph
from IPython.display import Image
Image(filename=join(output_dir, 'output/work_preproc/graph.png'), width=750)
```


{:.output .output_stream}
```
190416-10:45:49,342 nipype.workflow INFO:
	 Generated workflow graph: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/graph.png (graph2use=colored, simple_form=True).

```




{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_106_1.png)




##  Run the Workflow

Now we are ready to run the workflow! Be careful about the `n_procs` parameter if you run a workflow in `'MultiProc'` mode. `n_procs` specifies the number of jobs/cores your computer will use to run the workflow. If this number is too high your computer will try to execute too many things at once and will most likely crash.

**Note**: If  you're using a Docker container and FLIRT fails to run without any good reason, you might need to change memory settings in the Docker preferences (6 GB should be enough for this workflow).



{:.input_area}
```python
preproc.run('MultiProc', plugin_args={'n_procs': 4})
```


{:.output .output_stream}
```
190416-10:45:58,544 nipype.workflow INFO:
	 Workflow work_preproc settings: ['check', 'execution', 'logging', 'monitoring']
190416-10:45:58,736 nipype.workflow INFO:
	 Running in parallel.
190416-10:45:58,746 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-10:45:58,914 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.selectfiles" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/selectfiles".
190416-10:45:58,972 nipype.workflow INFO:
	 [Node] Running "selectfiles" ("nipype.interfaces.io.SelectFiles")
190416-10:45:59,16 nipype.workflow INFO:
	 [Node] Finished "work_preproc.selectfiles".
190416-10:46:00,746 nipype.workflow INFO:
	 [Job 0] Completed (work_preproc.selectfiles).
190416-10:46:00,754 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 2 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-10:46:00,921 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.gunzip_func" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/gunzip_func".
190416-10:46:00,925 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.gunzip_anat" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/gunzip_anat".
190416-10:46:00,981 nipype.workflow INFO:
	 [Node] Running "gunzip_func" ("nipype.algorithms.misc.Gunzip")
190416-10:46:00,994 nipype.workflow INFO:
	 [Node] Running "gunzip_anat" ("nipype.algorithms.misc.Gunzip")
190416-10:46:02,748 nipype.workflow INFO:
	 [MultiProc] Running 2 tasks, and 0 jobs ready. Free memory (GB): 55.98/56.38, Free processors: 2/4.
                     Currently running:
                       * work_preproc.gunzip_anat
                       * work_preproc.gunzip_func
190416-10:46:12,741 nipype.workflow INFO:
	 [Node] Finished "work_preproc.gunzip_anat".
190416-10:46:12,756 nipype.workflow INFO:
	 [Job 6] Completed (work_preproc.gunzip_anat).
190416-10:46:12,761 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 1 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.gunzip_func
190416-10:46:12,884 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.segment" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/segment".
190416-10:46:12,951 nipype.workflow INFO:
	 [Node] Running "segment" ("nipype.interfaces.spm.preprocess.NewSegment")
190416-10:46:14,760 nipype.workflow INFO:
	 [MultiProc] Running 2 tasks, and 0 jobs ready. Free memory (GB): 55.98/56.38, Free processors: 2/4.
                     Currently running:
                       * work_preproc.segment
                       * work_preproc.gunzip_func
190416-10:46:14,798 nipype.workflow INFO:
	 [Node] Finished "work_preproc.gunzip_func".
190416-10:46:16,760 nipype.workflow INFO:
	 [Job 1] Completed (work_preproc.gunzip_func).
190416-10:46:16,766 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 1 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.segment
190416-10:46:16,879 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.extract" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/extract".
190416-10:46:16,932 nipype.workflow INFO:
	 [Node] Running "extract" ("nipype.interfaces.fsl.utils.ExtractROI"), a CommandLine Interface with command:
fslroi /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/gunzip_func/sub-07_ses-test_task-fingerfootlips_bold.nii /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/extract/sub-07_ses-test_task-fingerfootlips_bold_roi.nii 4 -1
190416-10:46:18,764 nipype.workflow INFO:
	 [MultiProc] Running 2 tasks, and 0 jobs ready. Free memory (GB): 55.98/56.38, Free processors: 2/4.
                     Currently running:
                       * work_preproc.extract
                       * work_preproc.segment
190416-10:46:29,492 nipype.workflow INFO:
	 [Node] Finished "work_preproc.extract".
190416-10:46:30,775 nipype.workflow INFO:
	 [Job 2] Completed (work_preproc.extract).
190416-10:46:30,780 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 1 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.segment
190416-10:46:30,891 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.slicetime" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/slicetime".
190416-10:46:30,942 nipype.workflow INFO:
	 [Node] Running "slicetime" ("nipype.interfaces.spm.preprocess.SliceTiming")
190416-10:46:32,778 nipype.workflow INFO:
	 [MultiProc] Running 2 tasks, and 0 jobs ready. Free memory (GB): 55.98/56.38, Free processors: 2/4.
                     Currently running:
                       * work_preproc.slicetime
                       * work_preproc.segment
190416-10:48:28,71 nipype.workflow INFO:
	 [Node] Finished "work_preproc.slicetime".
190416-10:48:28,892 nipype.workflow INFO:
	 [Job 3] Completed (work_preproc.slicetime).
190416-10:48:28,898 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 1 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.segment
190416-10:48:29,12 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.mcflirt" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/mcflirt".
190416-10:48:29,72 nipype.workflow INFO:
	 [Node] Running "mcflirt" ("nipype.interfaces.fsl.preprocess.MCFLIRT"), a CommandLine Interface with command:
mcflirt -in /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/slicetime/asub-07_ses-test_task-fingerfootlips_bold_roi.nii -meanvol -out /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/mcflirt/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz -plots
190416-10:48:30,895 nipype.workflow INFO:
	 [MultiProc] Running 2 tasks, and 0 jobs ready. Free memory (GB): 55.98/56.38, Free processors: 2/4.
                     Currently running:
                       * work_preproc.mcflirt
                       * work_preproc.segment
190416-10:50:49,344 nipype.workflow INFO:
	 [Node] Finished "work_preproc.mcflirt".
190416-10:50:51,33 nipype.workflow INFO:
	 [Job 4] Completed (work_preproc.mcflirt).
190416-10:50:51,39 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 1 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.segment
190416-10:50:51,156 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.art" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/art".
190416-10:50:51,212 nipype.workflow INFO:
	 [Node] Running "art" ("nipype.algorithms.rapidart.ArtifactDetect")
190416-10:50:52,402 nipype.workflow INFO:
	 [Node] Finished "work_preproc.art".
190416-10:50:53,35 nipype.workflow INFO:
	 [Job 5] Completed (work_preproc.art).
190416-10:50:53,40 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 0 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.segment
190416-10:51:55,769 nipype.workflow INFO:
	 [Node] Finished "work_preproc.segment".
190416-10:51:57,101 nipype.workflow INFO:
	 [Job 7] Completed (work_preproc.segment).
190416-10:51:57,106 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 2 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-10:51:57,228 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.resample" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/resample".
190416-10:51:57,238 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.threshold_WM" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/threshold_WM".
190416-10:51:57,286 nipype.workflow INFO:
	 [Node] Running "resample" ("nipype.interfaces.fsl.preprocess.FLIRT"), a CommandLine Interface with command:
flirt -in /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/segment/c1sub-07_ses-test_T1w.nii -ref /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/segment/c1sub-07_ses-test_T1w.nii -out c1sub-07_ses-test_T1w_flirt.nii -omat c1sub-07_ses-test_T1w_flirt.mat -applyisoxfm 4.000000
190416-10:51:57,292 nipype.workflow INFO:
	 [Node] Running "threshold_WM" ("nipype.interfaces.fsl.maths.Threshold"), a CommandLine Interface with command:
fslmaths /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/segment/c2sub-07_ses-test_T1w.nii -thr 0.5000000000 -bin /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/threshold_WM/c2sub-07_ses-test_T1w_thresh.nii
190416-10:51:59,105 nipype.workflow INFO:
	 [MultiProc] Running 2 tasks, and 0 jobs ready. Free memory (GB): 55.98/56.38, Free processors: 2/4.
                     Currently running:
                       * work_preproc.threshold_WM
                       * work_preproc.resample
190416-10:52:04,325 nipype.workflow INFO:
	 [Node] Finished "work_preproc.resample".
190416-10:52:05,108 nipype.workflow INFO:
	 [Job 8] Completed (work_preproc.resample).
190416-10:52:05,113 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 1 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.threshold_WM
190416-10:52:05,231 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.mask_GM" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/mask_GM".
190416-10:52:05,298 nipype.workflow INFO:
	 [Node] Running "mask_GM" ("nipype.interfaces.fsl.maths.Threshold"), a CommandLine Interface with command:
fslmaths /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/resample/c1sub-07_ses-test_T1w_flirt.nii -thr 0.5000000000 -bin -dilF /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/mask_GM/c1sub-07_ses-test_T1w_flirt_thresh.nii
190416-10:52:06,372 nipype.workflow INFO:
	 [Node] Finished "work_preproc.mask_GM".
190416-10:52:07,110 nipype.workflow INFO:
	 [Job 9] Completed (work_preproc.mask_GM).
190416-10:52:07,115 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 0 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.threshold_WM
190416-10:52:09,801 nipype.workflow INFO:
	 [Node] Finished "work_preproc.threshold_WM".
190416-10:52:11,114 nipype.workflow INFO:
	 [Job 10] Completed (work_preproc.threshold_WM).
190416-10:52:11,119 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-10:52:11,249 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.coreg" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/coreg".
190416-10:52:11,306 nipype.workflow INFO:
	 [Node] Running "coreg" ("nipype.interfaces.fsl.preprocess.FLIRT"), a CommandLine Interface with command:
flirt -in /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/mcflirt/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg.nii.gz -ref /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/gunzip_anat/sub-07_ses-test_T1w.nii -out asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.nii -omat asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.mat -cost bbr -dof 6 -schedule /optnfs/el7/fsl/5.0.10/etc/flirtsch/bbr.sch -wmseg /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/threshold_WM/c2sub-07_ses-test_T1w_thresh.nii
190416-10:52:13,118 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 0 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.coreg
190416-10:54:56,754 nipype.workflow INFO:
	 [Node] Finished "work_preproc.coreg".
190416-10:54:57,282 nipype.workflow INFO:
	 [Job 11] Completed (work_preproc.coreg).
190416-10:54:57,288 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-10:54:57,439 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.applywarp" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/applywarp".
190416-10:54:57,495 nipype.workflow INFO:
	 [Node] Running "applywarp" ("nipype.interfaces.fsl.preprocess.FLIRT"), a CommandLine Interface with command:
flirt -in /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/mcflirt/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz -ref /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/gunzip_anat/sub-07_ses-test_T1w.nii -out asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.nii -omat asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.mat -applyisoxfm 4.000000 -init /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/coreg/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.mat -interp spline
190416-10:54:59,285 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 0 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.applywarp
190416-10:55:35,680 nipype.workflow INFO:
	 [Node] Finished "work_preproc.applywarp".
190416-10:55:37,324 nipype.workflow INFO:
	 [Job 12] Completed (work_preproc.applywarp).
190416-10:55:37,329 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 2 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-10:55:37,463 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.susan.mask" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/mask".
190416-10:55:37,475 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.susan.median" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/median".
190416-10:55:37,559 nipype.workflow INFO:
	 [Node] Setting-up "_median0" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/median/mapflow/_median0".
190416-10:55:37,561 nipype.workflow INFO:
	 [Node] Setting-up "_mask0" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/mask/mapflow/_mask0".
190416-10:55:37,631 nipype.workflow INFO:
	 [Node] Running "_mask0" ("nipype.interfaces.fsl.utils.ImageMaths"), a CommandLine Interface with command:
fslmaths /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/applywarp/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.nii -mas /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/mask_GM/c1sub-07_ses-test_T1w_flirt_thresh.nii /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/mask/mapflow/_mask0/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt_mask.nii.gz
190416-10:55:37,634 nipype.workflow INFO:
	 [Node] Running "_median0" ("nipype.interfaces.fsl.utils.ImageStats"), a CommandLine Interface with command:
fslstats /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/applywarp/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.nii -k /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/mask_GM/c1sub-07_ses-test_T1w_flirt_thresh.nii -p 50 
190416-10:55:39,327 nipype.workflow INFO:
	 [MultiProc] Running 2 tasks, and 0 jobs ready. Free memory (GB): 55.98/56.38, Free processors: 2/4.
                     Currently running:
                       * work_preproc.susan.median
                       * work_preproc.susan.mask
190416-10:55:39,887 nipype.workflow INFO:
	 [Node] Finished "_median0".
190416-10:55:39,929 nipype.workflow INFO:
	 [Node] Finished "work_preproc.susan.median".
190416-10:55:41,327 nipype.workflow INFO:
	 [Job 15] Completed (work_preproc.susan.median).
190416-10:55:41,333 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 0 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.susan.mask
190416-10:55:46,899 nipype.workflow INFO:
	 [Node] Finished "_mask0".
190416-10:55:46,927 nipype.workflow INFO:
	 [Node] Finished "work_preproc.susan.mask".
190416-10:55:47,334 nipype.workflow INFO:
	 [Job 13] Completed (work_preproc.susan.mask).
190416-10:55:47,339 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-10:55:47,462 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.susan.meanfunc2" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/meanfunc2".
190416-10:55:47,518 nipype.workflow INFO:
	 [Node] Setting-up "_meanfunc20" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/meanfunc2/mapflow/_meanfunc20".
190416-10:55:47,575 nipype.workflow INFO:
	 [Node] Running "_meanfunc20" ("nipype.interfaces.fsl.utils.ImageMaths"), a CommandLine Interface with command:
fslmaths /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/mask/mapflow/_mask0/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt_mask.nii.gz -Tmean /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/meanfunc2/mapflow/_meanfunc20/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt_mask_mean.nii.gz
190416-10:55:49,337 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 0 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.susan.meanfunc2
190416-10:55:50,388 nipype.workflow INFO:
	 [Node] Finished "_meanfunc20".
190416-10:55:50,442 nipype.workflow INFO:
	 [Node] Finished "work_preproc.susan.meanfunc2".
190416-10:55:51,338 nipype.workflow INFO:
	 [Job 14] Completed (work_preproc.susan.meanfunc2).
190416-10:55:51,343 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-10:55:51,462 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.susan.merge" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/merge".
190416-10:55:51,527 nipype.workflow INFO:
	 [Node] Running "merge" ("nipype.interfaces.utility.base.Merge")
190416-10:55:51,575 nipype.workflow INFO:
	 [Node] Finished "work_preproc.susan.merge".
190416-10:55:53,340 nipype.workflow INFO:
	 [Job 16] Completed (work_preproc.susan.merge).
190416-10:55:53,345 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-10:55:53,468 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.susan.multi_inputs" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/multi_inputs".
190416-10:55:53,515 nipype.workflow INFO:
	 [Node] Running "multi_inputs" ("nipype.interfaces.utility.wrappers.Function")
190416-10:55:53,563 nipype.workflow INFO:
	 [Node] Finished "work_preproc.susan.multi_inputs".
190416-10:55:55,342 nipype.workflow INFO:
	 [Job 17] Completed (work_preproc.susan.multi_inputs).
190416-10:55:55,347 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-10:55:55,479 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.susan.smooth" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/smooth".
190416-10:55:55,530 nipype.workflow INFO:
	 [Node] Setting-up "_smooth0" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/smooth/mapflow/_smooth0".
190416-10:55:55,594 nipype.workflow INFO:
	 [Node] Running "_smooth0" ("nipype.interfaces.fsl.preprocess.SUSAN"), a CommandLine Interface with command:
susan /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/applywarp/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.nii 1046.2500000000 1.6986436006 3 1 1 /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/meanfunc2/mapflow/_meanfunc20/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt_mask_mean.nii.gz 1046.2500000000 /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/smooth/mapflow/_smooth0/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt_smooth.nii.gz
190416-10:55:57,345 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 0 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.susan.smooth
190416-10:57:19,545 nipype.workflow INFO:
	 [Node] Finished "_smooth0".
190416-10:57:19,595 nipype.workflow INFO:
	 [Node] Finished "work_preproc.susan.smooth".
190416-10:57:21,429 nipype.workflow INFO:
	 [Job 18] Completed (work_preproc.susan.smooth).
190416-10:57:21,434 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-10:57:21,563 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.mask_func" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/mask_func".
190416-10:57:21,616 nipype.workflow INFO:
	 [Node] Setting-up "_mask_func0" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/mask_func/mapflow/_mask_func0".
190416-10:57:21,673 nipype.workflow INFO:
	 [Node] Running "_mask_func0" ("nipype.interfaces.fsl.maths.ApplyMask"), a CommandLine Interface with command:
fslmaths /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/susan/_subject_id_07/smooth/mapflow/_smooth0/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt_smooth.nii.gz -mas /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/mask_GM/c1sub-07_ses-test_T1w_flirt_thresh.nii /dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/mask_func/mapflow/_mask_func0/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt_smooth_masked.nii
190416-10:57:23,432 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 0 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.mask_func
190416-10:58:01,229 nipype.workflow INFO:
	 [Node] Finished "_mask_func0".
190416-10:58:01,271 nipype.workflow INFO:
	 [Node] Finished "work_preproc.mask_func".
190416-10:58:01,470 nipype.workflow INFO:
	 [Job 19] Completed (work_preproc.mask_func).
190416-10:58:01,475 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-10:58:01,590 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.detrend" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/detrend".
190416-10:58:01,648 nipype.workflow INFO:
	 [Node] Running "detrend" ("nipype.algorithms.confounds.TSNR")
190416-10:58:03,474 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 0 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.detrend
190416-10:58:14,807 nipype.workflow INFO:
	 [Node] Finished "work_preproc.detrend".
190416-10:58:15,485 nipype.workflow INFO:
	 [Job 20] Completed (work_preproc.detrend).
190416-10:58:15,490 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 0 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.

```




{:.output .output_data_text}
```
<networkx.classes.digraph.DiGraph at 0x2ad265de0710>
```



## Inspect output

What did we actually do? Let's look at all the data that was created.



{:.input_area}
```python
!tree -L 3 {join(output_dir, 'output/work_preproc/')} -I '*js|*json|*pklz|_report|*dot|*html|*txt|*.m'

```


{:.output .output_stream}
```
/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/
|-- graph.png
|-- _subject_id_02
|   |-- applywarp
|   |   |-- asub-02_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.mat
|   |   `-- asub-02_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.nii
|   |-- art
|   |   `-- plot.asub-02_ses-test_task-fingerfootlips_bold_roi_mcf.svg
|   |-- coreg
|   |   |-- asub-02_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.mat
|   |   `-- asub-02_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.nii
|   |-- datasink
|   |-- detrend
|   |   `-- detrend.nii.gz
|   |-- extract
|   |   `-- sub-02_ses-test_task-fingerfootlips_bold_roi.nii
|   |-- gunzip_anat
|   |   `-- sub-02_ses-test_T1w.nii
|   |-- gunzip_func
|   |   `-- sub-02_ses-test_task-fingerfootlips_bold.nii
|   |-- mask_func
|   |   `-- mapflow
|   |-- mask_GM
|   |   `-- c1sub-02_ses-test_T1w_flirt_thresh.nii
|   |-- mcflirt
|   |   |-- asub-02_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz
|   |   |-- asub-02_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg.nii.gz
|   |   `-- asub-02_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par
|   |-- resample
|   |   |-- c1sub-02_ses-test_T1w_flirt.mat
|   |   `-- c1sub-02_ses-test_T1w_flirt.nii
|   |-- segment
|   |   |-- c1sub-02_ses-test_T1w.nii
|   |   |-- c2sub-02_ses-test_T1w.nii
|   |   `-- c3sub-02_ses-test_T1w.nii
|   |-- selectfiles
|   |-- slicetime
|   |   `-- asub-02_ses-test_task-fingerfootlips_bold_roi.nii
|   `-- threshold_WM
|       `-- c2sub-02_ses-test_T1w_thresh.nii
|-- _subject_id_03
|   |-- applywarp
|   |   |-- asub-03_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.mat
|   |   `-- asub-03_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.nii
|   |-- art
|   |   `-- plot.asub-03_ses-test_task-fingerfootlips_bold_roi_mcf.svg
|   |-- coreg
|   |   |-- asub-03_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.mat
|   |   `-- asub-03_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.nii
|   |-- datasink
|   |-- detrend
|   |   `-- detrend.nii.gz
|   |-- extract
|   |   `-- sub-03_ses-test_task-fingerfootlips_bold_roi.nii
|   |-- gunzip_anat
|   |   `-- sub-03_ses-test_T1w.nii
|   |-- gunzip_func
|   |   `-- sub-03_ses-test_task-fingerfootlips_bold.nii
|   |-- mask_func
|   |   `-- mapflow
|   |-- mask_GM
|   |   `-- c1sub-03_ses-test_T1w_flirt_thresh.nii
|   |-- mcflirt
|   |   |-- asub-03_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz
|   |   |-- asub-03_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg.nii.gz
|   |   `-- asub-03_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par
|   |-- resample
|   |   |-- c1sub-03_ses-test_T1w_flirt.mat
|   |   `-- c1sub-03_ses-test_T1w_flirt.nii
|   |-- segment
|   |   |-- c1sub-03_ses-test_T1w.nii
|   |   |-- c2sub-03_ses-test_T1w.nii
|   |   `-- c3sub-03_ses-test_T1w.nii
|   |-- selectfiles
|   |-- slicetime
|   |   `-- asub-03_ses-test_task-fingerfootlips_bold_roi.nii
|   `-- threshold_WM
|       `-- c2sub-03_ses-test_T1w_thresh.nii
|-- _subject_id_04
|   |-- applywarp
|   |   |-- asub-04_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.mat
|   |   `-- asub-04_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.nii
|   |-- art
|   |   `-- plot.asub-04_ses-test_task-fingerfootlips_bold_roi_mcf.svg
|   |-- coreg
|   |   |-- asub-04_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.mat
|   |   `-- asub-04_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.nii
|   |-- datasink
|   |-- detrend
|   |   `-- detrend.nii.gz
|   |-- extract
|   |   `-- sub-04_ses-test_task-fingerfootlips_bold_roi.nii
|   |-- gunzip_anat
|   |   `-- sub-04_ses-test_T1w.nii
|   |-- gunzip_func
|   |   `-- sub-04_ses-test_task-fingerfootlips_bold.nii
|   |-- mask_func
|   |   `-- mapflow
|   |-- mask_GM
|   |   `-- c1sub-04_ses-test_T1w_flirt_thresh.nii
|   |-- mcflirt
|   |   |-- asub-04_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz
|   |   |-- asub-04_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg.nii.gz
|   |   `-- asub-04_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par
|   |-- resample
|   |   |-- c1sub-04_ses-test_T1w_flirt.mat
|   |   `-- c1sub-04_ses-test_T1w_flirt.nii
|   |-- segment
|   |   |-- c1sub-04_ses-test_T1w.nii
|   |   |-- c2sub-04_ses-test_T1w.nii
|   |   `-- c3sub-04_ses-test_T1w.nii
|   |-- selectfiles
|   |-- slicetime
|   |   `-- asub-04_ses-test_task-fingerfootlips_bold_roi.nii
|   `-- threshold_WM
|       `-- c2sub-04_ses-test_T1w_thresh.nii
|-- _subject_id_07
|   |-- applywarp
|   |   |-- asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.mat
|   |   `-- asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.nii
|   |-- art
|   |   `-- plot.asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.svg
|   |-- coreg
|   |   |-- asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.mat
|   |   `-- asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.nii
|   |-- datasink
|   |-- detrend
|   |   `-- detrend.nii.gz
|   |-- extract
|   |   `-- sub-07_ses-test_task-fingerfootlips_bold_roi.nii
|   |-- gunzip_anat
|   |   `-- sub-07_ses-test_T1w.nii
|   |-- gunzip_func
|   |   `-- sub-07_ses-test_task-fingerfootlips_bold.nii
|   |-- mask_func
|   |   `-- mapflow
|   |-- mask_GM
|   |   `-- c1sub-07_ses-test_T1w_flirt_thresh.nii
|   |-- mcflirt
|   |   |-- asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz
|   |   |-- asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg.nii.gz
|   |   `-- asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par
|   |-- resample
|   |   |-- c1sub-07_ses-test_T1w_flirt.mat
|   |   `-- c1sub-07_ses-test_T1w_flirt.nii
|   |-- segment
|   |   |-- c1sub-07_ses-test_T1w.nii
|   |   |-- c2sub-07_ses-test_T1w.nii
|   |   `-- c3sub-07_ses-test_T1w.nii
|   |-- selectfiles
|   |-- slicetime
|   |   `-- asub-07_ses-test_task-fingerfootlips_bold_roi.nii
|   `-- threshold_WM
|       `-- c2sub-07_ses-test_T1w_thresh.nii
|-- _subject_id_08
|   |-- applywarp
|   |   |-- asub-08_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.mat
|   |   `-- asub-08_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.nii
|   |-- art
|   |   |-- mask.asub-08_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz
|   |   `-- plot.asub-08_ses-test_task-fingerfootlips_bold_roi_mcf.svg
|   |-- coreg
|   |   |-- asub-08_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.mat
|   |   `-- asub-08_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.nii
|   |-- detrend
|   |   `-- detrend.nii.gz
|   |-- extract
|   |   `-- sub-08_ses-test_task-fingerfootlips_bold_roi.nii
|   |-- gunzip_anat
|   |   `-- sub-08_ses-test_T1w.nii
|   |-- gunzip_func
|   |   `-- sub-08_ses-test_task-fingerfootlips_bold.nii
|   |-- mask_func
|   |   `-- mapflow
|   |-- mask_GM
|   |   `-- c1sub-08_ses-test_T1w_flirt_thresh.nii
|   |-- mcflirt
|   |   |-- asub-08_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz
|   |   |-- asub-08_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg.nii.gz
|   |   `-- asub-08_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par
|   |-- resample
|   |   |-- c1sub-08_ses-test_T1w_flirt.mat
|   |   `-- c1sub-08_ses-test_T1w_flirt.nii
|   |-- segment
|   |   |-- c1sub-08_ses-test_T1w.nii
|   |   |-- c2sub-08_ses-test_T1w.nii
|   |   `-- c3sub-08_ses-test_T1w.nii
|   |-- selectfiles
|   |-- slicetime
|   |   `-- asub-08_ses-test_task-fingerfootlips_bold_roi.nii
|   `-- threshold_WM
|       `-- c2sub-08_ses-test_T1w_thresh.nii
|-- _subject_id_09
|   |-- applywarp
|   |   |-- asub-09_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.mat
|   |   `-- asub-09_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.nii
|   |-- art
|   |   `-- plot.asub-09_ses-test_task-fingerfootlips_bold_roi_mcf.svg
|   |-- coreg
|   |   |-- asub-09_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.mat
|   |   `-- asub-09_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg_flirt.nii
|   |-- datasink
|   |-- detrend
|   |   `-- detrend.nii.gz
|   |-- extract
|   |   `-- sub-09_ses-test_task-fingerfootlips_bold_roi.nii
|   |-- gunzip_anat
|   |   `-- sub-09_ses-test_T1w.nii
|   |-- gunzip_func
|   |   `-- sub-09_ses-test_task-fingerfootlips_bold.nii
|   |-- mask_func
|   |   `-- mapflow
|   |-- mask_GM
|   |   `-- c1sub-09_ses-test_T1w_flirt_thresh.nii
|   |-- mcflirt
|   |   |-- asub-09_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz
|   |   |-- asub-09_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg.nii.gz
|   |   `-- asub-09_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par
|   |-- resample
|   |   |-- c1sub-09_ses-test_T1w_flirt.mat
|   |   `-- c1sub-09_ses-test_T1w_flirt.nii
|   |-- segment
|   |   |-- c1sub-09_ses-test_T1w.nii
|   |   |-- c2sub-09_ses-test_T1w.nii
|   |   `-- c3sub-09_ses-test_T1w.nii
|   |-- selectfiles
|   |-- slicetime
|   |   `-- asub-09_ses-test_task-fingerfootlips_bold_roi.nii
|   `-- threshold_WM
|       `-- c2sub-09_ses-test_T1w_thresh.nii
`-- susan
    |-- _subject_id_02
    |   |-- mask
    |   |-- meanfunc2
    |   |-- median
    |   |-- merge
    |   |-- multi_inputs
    |   `-- smooth
    |-- _subject_id_03
    |   |-- mask
    |   |-- meanfunc2
    |   |-- median
    |   |-- merge
    |   |-- multi_inputs
    |   `-- smooth
    |-- _subject_id_04
    |   |-- mask
    |   |-- meanfunc2
    |   |-- median
    |   |-- merge
    |   |-- multi_inputs
    |   `-- smooth
    |-- _subject_id_07
    |   |-- mask
    |   |-- meanfunc2
    |   |-- median
    |   |-- merge
    |   |-- multi_inputs
    |   `-- smooth
    |-- _subject_id_08
    |   |-- mask
    |   |-- meanfunc2
    |   |-- median
    |   |-- merge
    |   |-- multi_inputs
    |   `-- smooth
    `-- _subject_id_09
        |-- mask
        |-- meanfunc2
        |-- median
        |-- merge
        |-- multi_inputs
        `-- smooth

150 directories, 122 files

```

But what did we do specifically? Well, let's investigate.

### Motion Correction and Artifact Detection

How much did the subject move in the scanner and where there any outliers in the functional images?



{:.input_area}
```python
%matplotlib inline
```




{:.input_area}
```python
# Plot the motion paramters
import numpy as np
import matplotlib.pyplot as plt
par = np.loadtxt(join(output_dir, 'output/work_preproc/_subject_id_07/mcflirt/'
                 'asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par'))
fig, axes = plt.subplots(2, 1, figsize=(15, 5))
axes[0].set_ylabel('rotation (radians)')
axes[0].plot(par[0:, :3])
axes[1].plot(par[0:, 3:])
axes[1].set_xlabel('time (TR)')
axes[1].set_ylabel('translation (mm)');
```



{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_114_0.png)



The motion parameters seems to look ok. What about the detection of artifacts?



{:.input_area}
```python
# Showing the artifact detection output
from IPython.display import SVG
SVG(filename=join(output_dir, 'output/work_preproc/_subject_id_07/art/',
    'plot.asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.svg'))
```





![svg](../../images/features/notebooks/7_Nipype_Preprocessing_116_0.svg)



Which volumes are problematic?



{:.input_area}
```python
outliers = np.loadtxt(join(output_dir, 'output/work_preproc/_subject_id_07/art/',
                      'art.asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_outliers.txt'))
list(outliers.astype('int'))
```





{:.output .output_data_text}
```
[]
```



### Masks and Probability maps

Let's see what all the masks and probability maps look like. For this, we will use `nilearn`'s `plot_anat` function.



{:.input_area}
```python
%matplotlib inline

from nilearn import image as nli
from nilearn.plotting import plot_stat_map

output = join(output_dir, 'output/work_preproc/_subject_id_07/')
```


First, let's look at the tissue probability maps.



{:.input_area}
```python
anat = join(output, 'gunzip_anat/sub-07_ses-test_T1w.nii')
```




{:.input_area}
```python
plot_stat_map(
    join(output, 'segment/c1sub-07_ses-test_T1w.nii'), title='GM prob. map',  cmap=plt.cm.magma,
    threshold=0.5, bg_img=anat, display_mode='z', cut_coords=range(-35, 15, 10), dim=-1);
```



{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_123_0.png)





{:.input_area}
```python
plot_stat_map(
    join(output, 'segment/c2sub-07_ses-test_T1w.nii'), title='WM prob. map', cmap=plt.cm.magma,
    threshold=0.5, bg_img=anat, display_mode='z', cut_coords=range(-35, 15, 10), dim=-1);
```



{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_124_0.png)





{:.input_area}
```python
plot_stat_map(
    join(output, 'segment/c3sub-07_ses-test_T1w.nii'), title='CSF prob. map', cmap=plt.cm.magma,
    threshold=0.5, bg_img=anat, display_mode='z', cut_coords=range(-35, 15, 10), dim=-1);
```



{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_125_0.png)



And how does the gray matter mask look like that we used on the functional images?



{:.input_area}
```python
plot_stat_map(
    join(output, 'mask_GM/c1sub-07_ses-test_T1w_flirt_thresh.nii'), title='dilated GM Mask', cmap=plt.cm.magma,
    threshold=0.5, bg_img=anat, display_mode='z', cut_coords=range(-35, 15, 10), dim=-1);
```



{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_127_0.png)



### Functional Image transformations

Let's also investigate the transformation that we applied to the functional images.



{:.input_area}
```python
%matplotlib inline

from nilearn import image as nli
from nilearn.plotting import plot_epi

output = join(output_dir, 'output/work_preproc/_subject_id_07/')
```




{:.input_area}
```python
plot_epi(join(output, 'mcflirt/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz_mean_reg.nii.gz'),
         title='Motion Corrected mean image', display_mode='z', cut_coords=range(-40, 21, 15),
         cmap=plt.cm.viridis);
```



{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_130_0.png)





{:.input_area}
```python
mean = nli.mean_img(join(output, 'applywarp/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.nii'))
plot_epi(mean, title='Coregistred mean image', display_mode='z', cut_coords=range(-40, 21, 15),
         cmap=plt.cm.viridis);
```



{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_131_0.png)





{:.input_area}
```python
mean = nli.mean_img(join(output_dir, 'output/work_preproc/susan/_subject_id_07/smooth/mapflow/_smooth0/',
                    'asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt_smooth.nii.gz'))
plot_epi(mean, title='Smoothed mean image', display_mode='z', cut_coords=range(-40, 21, 15),
         cmap=plt.cm.viridis);
```



{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_132_0.png)





{:.input_area}
```python
mean = nli.mean_img(join(output, 'mask_func/mapflow/_mask_func0/'
                    'asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt_smooth_masked.nii'))
plot_epi(mean, title='Masked mean image', display_mode='z', cut_coords=range(-40, 21, 15),
         cmap=plt.cm.viridis);
```



{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_133_0.png)





{:.input_area}
```python
plot_epi(join(output, 'detrend/mean.nii.gz'), title='Detrended mean image', display_mode='z',
         cut_coords=range(-40, 21, 15), cmap=plt.cm.viridis);
```



{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_134_0.png)



That's all nice and beautiful, but what did smoothing and detrending actually do to the data?



{:.input_area}
```python
%matplotlib inline

import nibabel as nb

output = join(output_dir, 'output/work_preproc/_subject_id_07/')

# Load the relevant datasets
mc = nb.load(join(output, 'applywarp/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt.nii'))
smooth = nb.load(join(output_dir, 'output/work_preproc/susan/_subject_id_07/smooth/mapflow/',
                 '_smooth0/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_flirt_smooth.nii.gz'))
detrended_data = nb.load(output + 'detrend/detrend.nii.gz')

# Plot a representative voxel
x, y, z = 32, 34, 43
fig = plt.figure(figsize=(12, 4))
plt.plot(mc.get_data()[x, y, z, :])
plt.plot(smooth.get_data()[x, y, z, :])
plt.plot(detrended_data.get_data()[x, y, z, :])
plt.legend(['motion corrected', 'smoothed', 'detrended']);
```



{:.output .output_png}
![png](../../images/features/notebooks/7_Nipype_Preprocessing_136_0.png)



## Data output with `DataSink`

The results look fine, but we don't need all those temporary files. So let's use Datasink to keep only those files that we actually need for the 1st and 2nd level analysis.



{:.input_area}
```python
from nipype.interfaces.io import DataSink

# Initiate the datasink node
output_folder = 'datasink_handson'
datasink = Node(DataSink(base_directory=join(output_dir, 'output/'),
                         container=output_folder),
                name="datasink")
```


Now the next step is to specify all the output that we want to keep in our output folder `output`. Make sure to keep:
- from the artifact detection node the outlier file as well as the outlier plot
- from the motion correction node the motion parameters
- from the last node, the detrended functional image



{:.input_area}
```python
# Connect nodes to datasink here
```




{:.input_area}
```python
preproc.connect([(art, datasink, [('outlier_files', 'preproc.@outlier_files'),
                                  ('plot_files', 'preproc.@plot_files')]),
                 (mcflirt, datasink, [('par_file', 'preproc.@par')]),
                 (detrend, datasink, [('detrended_file', 'preproc.@func')]),
                 ])
```


## Run the workflow

After adding the datasink folder, let's run the preprocessing workflow again.



{:.input_area}
```python
preproc.run('MultiProc', plugin_args={'n_procs': 4})
```


{:.output .output_stream}
```
190416-11:30:54,227 nipype.workflow INFO:
	 Workflow work_preproc settings: ['check', 'execution', 'logging', 'monitoring']
190416-11:30:54,427 nipype.workflow INFO:
	 Running in parallel.
190416-11:30:54,434 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:30:54,647 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.selectfiles" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/selectfiles".
190416-11:30:54,760 nipype.workflow INFO:
	 [Node] Running "selectfiles" ("nipype.interfaces.io.SelectFiles")
190416-11:30:54,807 nipype.workflow INFO:
	 [Node] Finished "work_preproc.selectfiles".
190416-11:30:56,437 nipype.workflow INFO:
	 [Job 0] Completed (work_preproc.selectfiles).
190416-11:30:56,445 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 2 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:30:56,627 nipype.workflow INFO:
	 [Job 1] Cached (work_preproc.gunzip_func).
190416-11:30:56,637 nipype.workflow INFO:
	 [Job 6] Cached (work_preproc.gunzip_anat).
190416-11:30:58,572 nipype.workflow INFO:
	 [Job 2] Cached (work_preproc.extract).
190416-11:30:58,586 nipype.workflow INFO:
	 [Job 7] Cached (work_preproc.segment).
190416-11:31:00,440 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 3 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:31:00,581 nipype.workflow INFO:
	 [Job 3] Cached (work_preproc.slicetime).
190416-11:31:00,597 nipype.workflow INFO:
	 [Job 8] Cached (work_preproc.resample).
190416-11:31:00,612 nipype.workflow INFO:
	 [Job 10] Cached (work_preproc.threshold_WM).
190416-11:31:02,442 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 2 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:31:02,583 nipype.workflow INFO:
	 [Job 4] Cached (work_preproc.mcflirt).
190416-11:31:02,596 nipype.workflow INFO:
	 [Job 9] Cached (work_preproc.mask_GM).
190416-11:31:04,590 nipype.workflow INFO:
	 [Node] Outdated cache found for "work_preproc.art".
190416-11:31:04,598 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.art" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/art".
190416-11:31:04,605 nipype.workflow INFO:
	 [Node] Outdated cache found for "work_preproc.art".
190416-11:31:04,610 nipype.workflow INFO:
	 [Job 11] Cached (work_preproc.coreg).
190416-11:31:04,706 nipype.workflow INFO:
	 [Node] Running "art" ("nipype.algorithms.rapidart.ArtifactDetect")
190416-11:31:05,938 nipype.workflow INFO:
	 [Node] Finished "work_preproc.art".
190416-11:31:06,445 nipype.workflow INFO:
	 [Job 5] Completed (work_preproc.art).
190416-11:31:06,450 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:31:06,603 nipype.workflow INFO:
	 [Job 12] Cached (work_preproc.applywarp).
190416-11:31:08,448 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 2 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:31:08,597 nipype.workflow INFO:
	 [Job 13] Cached (work_preproc.susan.mask).
190416-11:31:08,617 nipype.workflow INFO:
	 [Job 15] Cached (work_preproc.susan.median).
190416-11:31:10,450 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:31:10,595 nipype.workflow INFO:
	 [Job 14] Cached (work_preproc.susan.meanfunc2).
190416-11:31:12,597 nipype.workflow INFO:
	 [Job 16] Cached (work_preproc.susan.merge).
190416-11:31:14,596 nipype.workflow INFO:
	 [Job 17] Cached (work_preproc.susan.multi_inputs).
190416-11:31:16,606 nipype.workflow INFO:
	 [Job 18] Cached (work_preproc.susan.smooth).
190416-11:31:18,602 nipype.workflow INFO:
	 [Job 19] Cached (work_preproc.mask_func).
190416-11:31:20,599 nipype.workflow INFO:
	 [Node] Outdated cache found for "work_preproc.detrend".
190416-11:31:20,606 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.detrend" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/detrend".
190416-11:31:20,613 nipype.workflow INFO:
	 [Node] Outdated cache found for "work_preproc.detrend".
190416-11:31:20,728 nipype.workflow INFO:
	 [Node] Running "detrend" ("nipype.algorithms.confounds.TSNR")
190416-11:31:22,464 nipype.workflow INFO:
	 [MultiProc] Running 1 tasks, and 0 jobs ready. Free memory (GB): 56.18/56.38, Free processors: 3/4.
                     Currently running:
                       * work_preproc.detrend
190416-11:31:33,367 nipype.workflow INFO:
	 [Node] Finished "work_preproc.detrend".
190416-11:31:34,473 nipype.workflow INFO:
	 [Job 20] Completed (work_preproc.detrend).
190416-11:31:34,479 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:31:34,661 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.datasink" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/datasink".
190416-11:31:34,710 nipype.workflow INFO:
	 [Node] Running "datasink" ("nipype.interfaces.io.DataSink")
190416-11:31:34,805 nipype.workflow INFO:
	 [Node] Finished "work_preproc.datasink".
190416-11:31:36,475 nipype.workflow INFO:
	 [Job 21] Completed (work_preproc.datasink).
190416-11:31:36,481 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 0 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.

```




{:.output .output_data_text}
```
<networkx.classes.digraph.DiGraph at 0x2ad267fa7400>
```



Let's look now at the output of this datasink folder.





{:.input_area}
```python
!tree -L 3 {join(output_dir, 'output/datasink_handson/')} -I '*js|*json|*pklz|_report|*dot|*html|*txt|*.m'
```


{:.output .output_stream}
```
/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/
`-- preproc
    |-- plot.sub-02.svg
    |-- plot.sub-03.svg
    |-- plot.sub-04.svg
    |-- plot.sub-07.svg
    |-- plot.sub-09.svg
    |-- sub-02_detrend.nii.gz
    |-- sub-02.par
    |-- sub-03_detrend.nii.gz
    |-- sub-03.par
    |-- sub-04_detrend.nii.gz
    |-- sub-04.par
    |-- sub-07_detrend.nii.gz
    |-- sub-07.par
    |-- sub-09_detrend.nii.gz
    `-- sub-09.par

1 directory, 15 files

```

Much better! But we're still not there yet. There are many unnecessary file specifiers that we can get rid off. To do so, we can use `DataSink`'s `substitutions` parameter. For this, we create a list of tuples: on the left, we specify the string that we want to replace and on the right, with what we want to replace it with.



{:.input_area}
```python
## Use the following substitutions for the DataSink output
substitutions = [('asub', 'sub'),
                 ('_ses-test_task-fingerfootlips_bold_roi_mcf', ''),
                 ('.nii.gz.par', '.par'),
                 ]

# To get rid of the folder '_subject_id_07' and renaming detrend
substitutions += [('_subject_id_%s/detrend' % s,
                   '_subject_id_%s/sub-%s_detrend' % (s, s)) for s in subject_list]
substitutions += [('_subject_id_%s/' % s, '') for s in subject_list]
datasink.inputs.substitutions = substitutions
```


Before we run the preprocessing workflow again, let's first delete the current output folder:



{:.input_area}
```python
# Delets the current output folder
!rm -rf {join(output_dir, 'output/datasink_handson')}

```




{:.input_area}
```python
# Runs the preprocessing workflow again, this time with substitutions
preproc.run('MultiProc', plugin_args={'n_procs': 4})
```


{:.output .output_stream}
```
190416-11:55:58,604 nipype.workflow INFO:
	 Workflow work_preproc settings: ['check', 'execution', 'logging', 'monitoring']
190416-11:55:58,809 nipype.workflow INFO:
	 Running in parallel.
190416-11:55:58,816 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:55:59,63 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.selectfiles" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/selectfiles".
190416-11:55:59,147 nipype.workflow INFO:
	 [Node] Running "selectfiles" ("nipype.interfaces.io.SelectFiles")
190416-11:55:59,194 nipype.workflow INFO:
	 [Node] Finished "work_preproc.selectfiles".
190416-11:56:00,820 nipype.workflow INFO:
	 [Job 0] Completed (work_preproc.selectfiles).
190416-11:56:00,828 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 2 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:56:01,63 nipype.workflow INFO:
	 [Job 1] Cached (work_preproc.gunzip_func).
190416-11:56:01,74 nipype.workflow INFO:
	 [Job 6] Cached (work_preproc.gunzip_anat).
190416-11:56:02,960 nipype.workflow INFO:
	 [Job 2] Cached (work_preproc.extract).
190416-11:56:02,972 nipype.workflow INFO:
	 [Job 7] Cached (work_preproc.segment).
190416-11:56:04,822 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 3 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:56:04,962 nipype.workflow INFO:
	 [Job 3] Cached (work_preproc.slicetime).
190416-11:56:04,977 nipype.workflow INFO:
	 [Job 8] Cached (work_preproc.resample).
190416-11:56:04,989 nipype.workflow INFO:
	 [Job 10] Cached (work_preproc.threshold_WM).
190416-11:56:06,824 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 2 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:56:06,963 nipype.workflow INFO:
	 [Job 4] Cached (work_preproc.mcflirt).
190416-11:56:06,974 nipype.workflow INFO:
	 [Job 9] Cached (work_preproc.mask_GM).
190416-11:56:08,973 nipype.workflow INFO:
	 [Job 5] Cached (work_preproc.art).
190416-11:56:08,992 nipype.workflow INFO:
	 [Job 11] Cached (work_preproc.coreg).
190416-11:56:10,828 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:56:10,973 nipype.workflow INFO:
	 [Job 12] Cached (work_preproc.applywarp).
190416-11:56:12,830 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 2 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:56:13,11 nipype.workflow INFO:
	 [Job 13] Cached (work_preproc.susan.mask).
190416-11:56:13,27 nipype.workflow INFO:
	 [Job 15] Cached (work_preproc.susan.median).
190416-11:56:14,832 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.
190416-11:56:14,989 nipype.workflow INFO:
	 [Job 14] Cached (work_preproc.susan.meanfunc2).
190416-11:56:16,978 nipype.workflow INFO:
	 [Job 16] Cached (work_preproc.susan.merge).
190416-11:56:18,977 nipype.workflow INFO:
	 [Job 17] Cached (work_preproc.susan.multi_inputs).
190416-11:56:20,986 nipype.workflow INFO:
	 [Job 18] Cached (work_preproc.susan.smooth).
190416-11:56:22,982 nipype.workflow INFO:
	 [Job 19] Cached (work_preproc.mask_func).
190416-11:56:24,981 nipype.workflow INFO:
	 [Job 20] Cached (work_preproc.detrend).
190416-11:56:27,11 nipype.workflow INFO:
	 [Node] Outdated cache found for "work_preproc.datasink".
190416-11:56:27,21 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.datasink" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/datasink".
190416-11:56:27,28 nipype.workflow INFO:
	 [Node] Outdated cache found for "work_preproc.datasink".
190416-11:56:27,113 nipype.workflow INFO:
	 [Node] Running "datasink" ("nipype.interfaces.io.DataSink")
190416-11:56:27,124 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_07/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-07.par
190416-11:56:27,141 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_07/art.asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_outliers.txt -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/art.sub-07_outliers.txt
190416-11:56:27,147 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_07/plot.asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.svg -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/plot.sub-07.svg
190416-11:56:27,155 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_07/detrend.nii.gz -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-07_detrend.nii.gz
190416-11:56:27,206 nipype.workflow INFO:
	 [Node] Finished "work_preproc.datasink".
190416-11:56:28,845 nipype.workflow INFO:
	 [Job 21] Completed (work_preproc.datasink).
190416-11:56:28,850 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 0 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 4/4.

```




{:.output .output_data_text}
```
<networkx.classes.digraph.DiGraph at 0x2ad2656a97f0>
```





{:.input_area}
```python
!tree {join(output_dir, 'output/datasink_handson/')} -I '*js|*json|*pklz|_report|*dot|*html|*txt|*.m'

```


{:.output .output_stream}
```
/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/
`-- preproc
    |-- plot.sub-02.svg
    |-- plot.sub-03.svg
    |-- plot.sub-04.svg
    |-- plot.sub-07.svg
    |-- plot.sub-09.svg
    |-- sub-02_detrend.nii.gz
    |-- sub-02.par
    |-- sub-03_detrend.nii.gz
    |-- sub-03.par
    |-- sub-04_detrend.nii.gz
    |-- sub-04.par
    |-- sub-07_detrend.nii.gz
    |-- sub-07.par
    |-- sub-09_detrend.nii.gz
    `-- sub-09.par

1 directory, 15 files

```

# Run Preprocessing workflow on 6 right-handed subjects

Perfect! 

Now we might want to run this pipeline for many subjects.  In this example, we will run all right-handed subjects. For this, you just need to change the `subject_list` variable and run again the places where this variable is used (i.e. `sf.iterables` and in `DataSink` `substitutions`.

**NOTE**: For today due to the limited resources of our jupyter hub server, we will be just looking at the results that have already been run.  However, feel free to run this on your after class.

The full output can be viewed at the following location:

`/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing`




{:.input_area}
```python
# Update 'subject_list' and its dependencies here
```




{:.input_area}
```python
subject_list = ['02', '03', '04', '07', '08', '09']

sf.iterables = [('subject_id', subject_list)]
```




{:.input_area}
```python
# To get rid of the folder '_subject_id_02' and renaming detrend
substitutions += [('_subject_id_%s/detrend' % s,
                   '_subject_id_%s/sub-%s_detrend' % (s, s)) for s in subject_list]
substitutions += [('_subject_id_%s/' % s, '') for s in subject_list]
datasink.inputs.substitutions = substitutions
```


Now we can run the workflow again, this time for all right handed subjects in parallel.



{:.input_area}
```python
# Runs the preprocessing workflow again, this time with substitutions
preproc.run('MultiProc', plugin_args={'n_procs': 8})
```


{:.output .output_stream}
```
190416-12:41:17,521 nipype.workflow INFO:
	 Workflow work_preproc settings: ['check', 'execution', 'logging', 'monitoring']
190416-12:41:17,943 nipype.workflow INFO:
	 Running in parallel.
190416-12:41:17,958 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 6 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:18,272 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.selectfiles" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_09/selectfiles".
190416-12:41:18,279 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.selectfiles" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_08/selectfiles".
190416-12:41:18,283 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.selectfiles" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/selectfiles".
190416-12:41:18,288 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.selectfiles" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_04/selectfiles".
190416-12:41:18,292 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.selectfiles" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_03/selectfiles".
190416-12:41:18,299 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.selectfiles" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_02/selectfiles".
190416-12:41:18,433 nipype.workflow INFO:
	 [Node] Running "selectfiles" ("nipype.interfaces.io.SelectFiles")
190416-12:41:18,436 nipype.workflow INFO:
	 [Node] Running "selectfiles" ("nipype.interfaces.io.SelectFiles")
190416-12:41:18,441 nipype.workflow INFO:
	 [Node] Running "selectfiles" ("nipype.interfaces.io.SelectFiles")
190416-12:41:18,445 nipype.workflow INFO:
	 [Node] Running "selectfiles" ("nipype.interfaces.io.SelectFiles")
190416-12:41:18,450 nipype.workflow INFO:
	 [Node] Running "selectfiles" ("nipype.interfaces.io.SelectFiles")
190416-12:41:18,452 nipype.workflow INFO:
	 [Node] Running "selectfiles" ("nipype.interfaces.io.SelectFiles")
190416-12:41:18,498 nipype.workflow INFO:
	 [Node] Finished "work_preproc.selectfiles".
190416-12:41:18,500 nipype.workflow INFO:
	 [Node] Finished "work_preproc.selectfiles".
190416-12:41:18,501 nipype.workflow INFO:
	 [Node] Finished "work_preproc.selectfiles".
190416-12:41:18,508 nipype.workflow INFO:
	 [Node] Finished "work_preproc.selectfiles".
190416-12:41:18,510 nipype.workflow INFO:
	 [Node] Finished "work_preproc.selectfiles".
190416-12:41:18,510 nipype.workflow INFO:
	 [Node] Finished "work_preproc.selectfiles".
190416-12:41:19,960 nipype.workflow INFO:
	 [Job 0] Completed (work_preproc.selectfiles).
190416-12:41:19,966 nipype.workflow INFO:
	 [Job 22] Completed (work_preproc.selectfiles).
190416-12:41:19,970 nipype.workflow INFO:
	 [Job 44] Completed (work_preproc.selectfiles).
190416-12:41:19,973 nipype.workflow INFO:
	 [Job 66] Completed (work_preproc.selectfiles).
190416-12:41:19,976 nipype.workflow INFO:
	 [Job 88] Completed (work_preproc.selectfiles).
190416-12:41:19,980 nipype.workflow INFO:
	 [Job 110] Completed (work_preproc.selectfiles).
190416-12:41:19,985 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 12 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:20,173 nipype.workflow INFO:
	 [Job 1] Cached (work_preproc.gunzip_func).
190416-12:41:20,182 nipype.workflow INFO:
	 [Job 6] Cached (work_preproc.gunzip_anat).
190416-12:41:20,190 nipype.workflow INFO:
	 [Job 23] Cached (work_preproc.gunzip_func).
190416-12:41:20,198 nipype.workflow INFO:
	 [Job 28] Cached (work_preproc.gunzip_anat).
190416-12:41:20,206 nipype.workflow INFO:
	 [Job 45] Cached (work_preproc.gunzip_func).
190416-12:41:20,214 nipype.workflow INFO:
	 [Job 50] Cached (work_preproc.gunzip_anat).
190416-12:41:20,222 nipype.workflow INFO:
	 [Job 67] Cached (work_preproc.gunzip_func).
190416-12:41:20,230 nipype.workflow INFO:
	 [Job 72] Cached (work_preproc.gunzip_anat).
190416-12:41:22,101 nipype.workflow INFO:
	 [Job 2] Cached (work_preproc.extract).
190416-12:41:22,114 nipype.workflow INFO:
	 [Job 7] Cached (work_preproc.segment).
190416-12:41:22,125 nipype.workflow INFO:
	 [Job 24] Cached (work_preproc.extract).
190416-12:41:22,138 nipype.workflow INFO:
	 [Job 29] Cached (work_preproc.segment).
190416-12:41:22,148 nipype.workflow INFO:
	 [Job 46] Cached (work_preproc.extract).
190416-12:41:22,157 nipype.workflow INFO:
	 [Job 51] Cached (work_preproc.segment).
190416-12:41:22,166 nipype.workflow INFO:
	 [Job 68] Cached (work_preproc.extract).
190416-12:41:22,180 nipype.workflow INFO:
	 [Job 73] Cached (work_preproc.segment).
190416-12:41:23,964 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 16 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:24,109 nipype.workflow INFO:
	 [Job 3] Cached (work_preproc.slicetime).
190416-12:41:24,125 nipype.workflow INFO:
	 [Job 8] Cached (work_preproc.resample).
190416-12:41:24,142 nipype.workflow INFO:
	 [Job 10] Cached (work_preproc.threshold_WM).
190416-12:41:24,155 nipype.workflow INFO:
	 [Job 25] Cached (work_preproc.slicetime).
190416-12:41:24,171 nipype.workflow INFO:
	 [Job 30] Cached (work_preproc.resample).
190416-12:41:24,179 nipype.workflow INFO:
	 [Job 32] Cached (work_preproc.threshold_WM).
190416-12:41:24,188 nipype.workflow INFO:
	 [Job 47] Cached (work_preproc.slicetime).
190416-12:41:24,208 nipype.workflow INFO:
	 [Job 52] Cached (work_preproc.resample).
190416-12:41:25,965 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 14 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:26,101 nipype.workflow INFO:
	 [Job 4] Cached (work_preproc.mcflirt).
190416-12:41:26,112 nipype.workflow INFO:
	 [Job 9] Cached (work_preproc.mask_GM).
190416-12:41:26,120 nipype.workflow INFO:
	 [Job 26] Cached (work_preproc.mcflirt).
190416-12:41:26,129 nipype.workflow INFO:
	 [Job 31] Cached (work_preproc.mask_GM).
190416-12:41:26,137 nipype.workflow INFO:
	 [Job 48] Cached (work_preproc.mcflirt).
190416-12:41:26,146 nipype.workflow INFO:
	 [Job 53] Cached (work_preproc.mask_GM).
190416-12:41:26,154 nipype.workflow INFO:
	 [Job 54] Cached (work_preproc.threshold_WM).
190416-12:41:26,164 nipype.workflow INFO:
	 [Job 69] Cached (work_preproc.slicetime).
190416-12:41:27,966 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 13 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:28,108 nipype.workflow INFO:
	 [Job 5] Cached (work_preproc.art).
190416-12:41:28,125 nipype.workflow INFO:
	 [Job 11] Cached (work_preproc.coreg).
190416-12:41:28,144 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.art" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_08/art".
190416-12:41:28,154 nipype.workflow INFO:
	 [Job 33] Cached (work_preproc.coreg).
190416-12:41:28,171 nipype.workflow INFO:
	 [Job 49] Cached (work_preproc.art).
190416-12:41:28,188 nipype.workflow INFO:
	 [Job 55] Cached (work_preproc.coreg).
190416-12:41:28,196 nipype.workflow INFO:
	 [Job 70] Cached (work_preproc.mcflirt).
190416-12:41:28,209 nipype.workflow INFO:
	 [Job 74] Cached (work_preproc.resample).
190416-12:41:28,246 nipype.workflow INFO:
	 [Node] Running "art" ("nipype.algorithms.rapidart.ArtifactDetect")
190416-12:41:29,479 nipype.workflow INFO:
	 [Node] Finished "work_preproc.art".
190416-12:41:29,966 nipype.workflow INFO:
	 [Job 27] Completed (work_preproc.art).
190416-12:41:29,973 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 10 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:30,115 nipype.workflow INFO:
	 [Job 12] Cached (work_preproc.applywarp).
190416-12:41:30,130 nipype.workflow INFO:
	 [Job 34] Cached (work_preproc.applywarp).
190416-12:41:30,145 nipype.workflow INFO:
	 [Job 56] Cached (work_preproc.applywarp).
190416-12:41:30,158 nipype.workflow INFO:
	 [Job 71] Cached (work_preproc.art).
190416-12:41:30,167 nipype.workflow INFO:
	 [Job 75] Cached (work_preproc.mask_GM).
190416-12:41:30,175 nipype.workflow INFO:
	 [Job 76] Cached (work_preproc.threshold_WM).
190416-12:41:30,183 nipype.workflow INFO:
	 [Job 89] Cached (work_preproc.gunzip_func).
190416-12:41:30,190 nipype.workflow INFO:
	 [Job 94] Cached (work_preproc.gunzip_anat).
190416-12:41:31,969 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 11 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:32,112 nipype.workflow INFO:
	 [Job 13] Cached (work_preproc.susan.mask).
190416-12:41:32,128 nipype.workflow INFO:
	 [Job 15] Cached (work_preproc.susan.median).
190416-12:41:32,144 nipype.workflow INFO:
	 [Job 35] Cached (work_preproc.susan.mask).
190416-12:41:32,158 nipype.workflow INFO:
	 [Job 37] Cached (work_preproc.susan.median).
190416-12:41:32,171 nipype.workflow INFO:
	 [Job 57] Cached (work_preproc.susan.mask).
190416-12:41:32,183 nipype.workflow INFO:
	 [Job 59] Cached (work_preproc.susan.median).
190416-12:41:32,198 nipype.workflow INFO:
	 [Job 77] Cached (work_preproc.coreg).
190416-12:41:32,207 nipype.workflow INFO:
	 [Job 90] Cached (work_preproc.extract).
190416-12:41:33,971 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 8 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:34,172 nipype.workflow INFO:
	 [Job 14] Cached (work_preproc.susan.meanfunc2).
190416-12:41:34,196 nipype.workflow INFO:
	 [Job 36] Cached (work_preproc.susan.meanfunc2).
190416-12:41:34,210 nipype.workflow INFO:
	 [Job 58] Cached (work_preproc.susan.meanfunc2).
190416-12:41:34,227 nipype.workflow INFO:
	 [Job 78] Cached (work_preproc.applywarp).
190416-12:41:34,237 nipype.workflow INFO:
	 [Job 91] Cached (work_preproc.slicetime).
190416-12:41:34,248 nipype.workflow INFO:
	 [Job 95] Cached (work_preproc.segment).
190416-12:41:34,255 nipype.workflow INFO:
	 [Job 111] Cached (work_preproc.gunzip_func).
190416-12:41:34,262 nipype.workflow INFO:
	 [Job 116] Cached (work_preproc.gunzip_anat).
190416-12:41:35,973 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 10 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:36,136 nipype.workflow INFO:
	 [Job 16] Cached (work_preproc.susan.merge).
190416-12:41:36,150 nipype.workflow INFO:
	 [Job 38] Cached (work_preproc.susan.merge).
190416-12:41:36,163 nipype.workflow INFO:
	 [Job 60] Cached (work_preproc.susan.merge).
190416-12:41:36,181 nipype.workflow INFO:
	 [Job 79] Cached (work_preproc.susan.mask).
190416-12:41:36,197 nipype.workflow INFO:
	 [Job 81] Cached (work_preproc.susan.median).
190416-12:41:36,208 nipype.workflow INFO:
	 [Job 92] Cached (work_preproc.mcflirt).
190416-12:41:36,224 nipype.workflow INFO:
	 [Job 96] Cached (work_preproc.resample).
190416-12:41:36,234 nipype.workflow INFO:
	 [Job 98] Cached (work_preproc.threshold_WM).
190416-12:41:37,974 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 9 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:38,122 nipype.workflow INFO:
	 [Job 17] Cached (work_preproc.susan.multi_inputs).
190416-12:41:38,138 nipype.workflow INFO:
	 [Job 39] Cached (work_preproc.susan.multi_inputs).
190416-12:41:38,152 nipype.workflow INFO:
	 [Job 61] Cached (work_preproc.susan.multi_inputs).
190416-12:41:38,167 nipype.workflow INFO:
	 [Job 80] Cached (work_preproc.susan.meanfunc2).
190416-12:41:38,181 nipype.workflow INFO:
	 [Job 93] Cached (work_preproc.art).
190416-12:41:38,194 nipype.workflow INFO:
	 [Job 97] Cached (work_preproc.mask_GM).
190416-12:41:38,210 nipype.workflow INFO:
	 [Job 99] Cached (work_preproc.coreg).
190416-12:41:38,219 nipype.workflow INFO:
	 [Job 112] Cached (work_preproc.extract).
190416-12:41:39,977 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 7 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:40,161 nipype.workflow INFO:
	 [Job 18] Cached (work_preproc.susan.smooth).
190416-12:41:40,187 nipype.workflow INFO:
	 [Job 40] Cached (work_preproc.susan.smooth).
190416-12:41:40,208 nipype.workflow INFO:
	 [Job 62] Cached (work_preproc.susan.smooth).
190416-12:41:40,239 nipype.workflow INFO:
	 [Job 82] Cached (work_preproc.susan.merge).
190416-12:41:40,256 nipype.workflow INFO:
	 [Job 100] Cached (work_preproc.applywarp).
190416-12:41:40,266 nipype.workflow INFO:
	 [Job 113] Cached (work_preproc.slicetime).
190416-12:41:40,277 nipype.workflow INFO:
	 [Job 117] Cached (work_preproc.segment).
190416-12:41:41,980 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 9 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:42,164 nipype.workflow INFO:
	 [Job 19] Cached (work_preproc.mask_func).
190416-12:41:42,181 nipype.workflow INFO:
	 [Job 41] Cached (work_preproc.mask_func).
190416-12:41:42,198 nipype.workflow INFO:
	 [Job 63] Cached (work_preproc.mask_func).
190416-12:41:42,215 nipype.workflow INFO:
	 [Job 83] Cached (work_preproc.susan.multi_inputs).
190416-12:41:42,231 nipype.workflow INFO:
	 [Job 101] Cached (work_preproc.susan.mask).
190416-12:41:42,245 nipype.workflow INFO:
	 [Job 103] Cached (work_preproc.susan.median).
190416-12:41:42,258 nipype.workflow INFO:
	 [Job 114] Cached (work_preproc.mcflirt).
190416-12:41:42,272 nipype.workflow INFO:
	 [Job 118] Cached (work_preproc.resample).
190416-12:41:43,981 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 8 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:44,129 nipype.workflow INFO:
	 [Job 20] Cached (work_preproc.detrend).
190416-12:41:44,139 nipype.workflow INFO:
	 [Job 42] Cached (work_preproc.detrend).
190416-12:41:44,150 nipype.workflow INFO:
	 [Job 64] Cached (work_preproc.detrend).
190416-12:41:44,174 nipype.workflow INFO:
	 [Job 84] Cached (work_preproc.susan.smooth).
190416-12:41:44,189 nipype.workflow INFO:
	 [Job 102] Cached (work_preproc.susan.meanfunc2).
190416-12:41:44,204 nipype.workflow INFO:
	 [Job 115] Cached (work_preproc.art).
190416-12:41:44,213 nipype.workflow INFO:
	 [Job 119] Cached (work_preproc.mask_GM).
190416-12:41:44,223 nipype.workflow INFO:
	 [Job 120] Cached (work_preproc.threshold_WM).
190416-12:41:45,982 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 6 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:46,160 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.datasink" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_09/datasink".
190416-12:41:46,180 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.datasink" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_08/datasink".
190416-12:41:46,205 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.datasink" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_07/datasink".
190416-12:41:46,222 nipype.workflow INFO:
	 [Job 85] Cached (work_preproc.mask_func).
190416-12:41:46,230 nipype.workflow INFO:
	 [Node] Running "datasink" ("nipype.interfaces.io.DataSink")
190416-12:41:46,235 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_08/asub-08_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-08.par
190416-12:41:46,241 nipype.workflow INFO:
	 [Node] Running "datasink" ("nipype.interfaces.io.DataSink")
190416-12:41:46,242 nipype.workflow INFO:
	 [Job 104] Cached (work_preproc.susan.merge).
190416-12:41:46,245 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_09/asub-09_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-09.par
190416-12:41:46,246 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_08/art.asub-08_ses-test_task-fingerfootlips_bold_roi_mcf_outliers.txt -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/art.sub-08_outliers.txt
190416-12:41:46,248 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_09/art.asub-09_ses-test_task-fingerfootlips_bold_roi_mcf_outliers.txt -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/art.sub-09_outliers.txt
190416-12:41:46,252 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_08/plot.asub-08_ses-test_task-fingerfootlips_bold_roi_mcf.svg -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/plot.sub-08.svg
190416-12:41:46,251 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_09/plot.asub-09_ses-test_task-fingerfootlips_bold_roi_mcf.svg -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/plot.sub-09.svg
190416-12:41:46,255 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_09/detrend.nii.gz -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-09_detrend.nii.gz
190416-12:41:46,266 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_08/detrend.nii.gz -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-08_detrend.nii.gz
190416-12:41:46,269 nipype.workflow INFO:
	 [Job 121] Cached (work_preproc.coreg).
190416-12:41:46,297 nipype.workflow INFO:
	 [Node] Running "datasink" ("nipype.interfaces.io.DataSink")
190416-12:41:46,301 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_07/asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-07.par
190416-12:41:46,304 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_07/art.asub-07_ses-test_task-fingerfootlips_bold_roi_mcf_outliers.txt -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/art.sub-07_outliers.txt
190416-12:41:46,306 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_07/plot.asub-07_ses-test_task-fingerfootlips_bold_roi_mcf.svg -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/plot.sub-07.svg
190416-12:41:46,309 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_07/detrend.nii.gz -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-07_detrend.nii.gz
190416-12:41:46,311 nipype.workflow INFO:
	 [Node] Finished "work_preproc.datasink".
190416-12:41:46,316 nipype.workflow INFO:
	 [Node] Finished "work_preproc.datasink".
190416-12:41:46,352 nipype.workflow INFO:
	 [Node] Finished "work_preproc.datasink".
190416-12:41:47,983 nipype.workflow INFO:
	 [Job 21] Completed (work_preproc.datasink).
190416-12:41:47,986 nipype.workflow INFO:
	 [Job 43] Completed (work_preproc.datasink).
190416-12:41:47,989 nipype.workflow INFO:
	 [Job 65] Completed (work_preproc.datasink).
190416-12:41:47,994 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 3 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:48,140 nipype.workflow INFO:
	 [Job 86] Cached (work_preproc.detrend).
190416-12:41:48,157 nipype.workflow INFO:
	 [Job 105] Cached (work_preproc.susan.multi_inputs).
190416-12:41:48,173 nipype.workflow INFO:
	 [Job 122] Cached (work_preproc.applywarp).
190416-12:41:49,987 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 4 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:50,186 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.datasink" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_04/datasink".
190416-12:41:50,205 nipype.workflow INFO:
	 [Job 106] Cached (work_preproc.susan.smooth).
190416-12:41:50,232 nipype.workflow INFO:
	 [Job 123] Cached (work_preproc.susan.mask).
190416-12:41:50,250 nipype.workflow INFO:
	 [Job 125] Cached (work_preproc.susan.median).
190416-12:41:50,291 nipype.workflow INFO:
	 [Node] Running "datasink" ("nipype.interfaces.io.DataSink")
190416-12:41:50,296 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_04/asub-04_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-04.par
190416-12:41:50,300 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_04/art.asub-04_ses-test_task-fingerfootlips_bold_roi_mcf_outliers.txt -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/art.sub-04_outliers.txt
190416-12:41:50,303 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_04/plot.asub-04_ses-test_task-fingerfootlips_bold_roi_mcf.svg -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/plot.sub-04.svg
190416-12:41:50,306 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_04/detrend.nii.gz -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-04_detrend.nii.gz
190416-12:41:50,346 nipype.workflow INFO:
	 [Node] Finished "work_preproc.datasink".
190416-12:41:51,987 nipype.workflow INFO:
	 [Job 87] Completed (work_preproc.datasink).
190416-12:41:51,993 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 2 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:52,164 nipype.workflow INFO:
	 [Job 107] Cached (work_preproc.mask_func).
190416-12:41:52,197 nipype.workflow INFO:
	 [Job 124] Cached (work_preproc.susan.meanfunc2).
190416-12:41:54,160 nipype.workflow INFO:
	 [Job 108] Cached (work_preproc.detrend).
190416-12:41:54,177 nipype.workflow INFO:
	 [Job 126] Cached (work_preproc.susan.merge).
190416-12:41:56,189 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.datasink" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_03/datasink".
190416-12:41:56,207 nipype.workflow INFO:
	 [Job 127] Cached (work_preproc.susan.multi_inputs).
190416-12:41:56,283 nipype.workflow INFO:
	 [Node] Running "datasink" ("nipype.interfaces.io.DataSink")
190416-12:41:56,288 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_03/asub-03_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-03.par
190416-12:41:56,292 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_03/art.asub-03_ses-test_task-fingerfootlips_bold_roi_mcf_outliers.txt -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/art.sub-03_outliers.txt
190416-12:41:56,295 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_03/plot.asub-03_ses-test_task-fingerfootlips_bold_roi_mcf.svg -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/plot.sub-03.svg
190416-12:41:56,298 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_03/detrend.nii.gz -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-03_detrend.nii.gz
190416-12:41:56,338 nipype.workflow INFO:
	 [Node] Finished "work_preproc.datasink".
190416-12:41:57,991 nipype.workflow INFO:
	 [Job 109] Completed (work_preproc.datasink).
190416-12:41:57,997 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 1 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.
190416-12:41:58,153 nipype.workflow INFO:
	 [Job 128] Cached (work_preproc.susan.smooth).
190416-12:42:00,140 nipype.workflow INFO:
	 [Job 129] Cached (work_preproc.mask_func).
190416-12:42:02,143 nipype.workflow INFO:
	 [Job 130] Cached (work_preproc.detrend).
190416-12:42:04,164 nipype.workflow INFO:
	 [Node] Setting-up "work_preproc.datasink" in "/dartfs/rc/lab/P/Psych60/students_output/f00275v/7_nipype_preprocessing/output/work_preproc/_subject_id_02/datasink".
190416-12:42:04,259 nipype.workflow INFO:
	 [Node] Running "datasink" ("nipype.interfaces.io.DataSink")
190416-12:42:04,265 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_02/asub-02_ses-test_task-fingerfootlips_bold_roi_mcf.nii.gz.par -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-02.par
190416-12:42:04,269 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_02/art.asub-02_ses-test_task-fingerfootlips_bold_roi_mcf_outliers.txt -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/art.sub-02_outliers.txt
190416-12:42:04,272 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_02/plot.asub-02_ses-test_task-fingerfootlips_bold_roi_mcf.svg -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/plot.sub-02.svg
190416-12:42:04,276 nipype.interface INFO:
	 sub: /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/_subject_id_02/detrend.nii.gz -> /dartfs-hpc/rc/home/v/f00275v/Psych60/students_output/f00275v/7_nipype_preprocessing/output/datasink_handson/preproc/sub-02_detrend.nii.gz
190416-12:42:04,327 nipype.workflow INFO:
	 [Node] Finished "work_preproc.datasink".
190416-12:42:05,999 nipype.workflow INFO:
	 [Job 131] Completed (work_preproc.datasink).
190416-12:42:06,4 nipype.workflow INFO:
	 [MultiProc] Running 0 tasks, and 0 jobs ready. Free memory (GB): 56.38/56.38, Free processors: 8/8.

```




{:.output .output_data_text}
```
<networkx.classes.digraph.DiGraph at 0x2ad278abffd0>
```


