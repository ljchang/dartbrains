
Results included in this manuscript come from preprocessing
performed using *fMRIPrep* 20.0.6
(@fmriprep1; @fmriprep2; RRID:SCR_016216),
which is based on *Nipype* 1.4.2
(@nipype1; @nipype2; RRID:SCR_002502).

Anatomical data preprocessing

: The T1-weighted (T1w) image was corrected for intensity non-uniformity (INU)
with `N4BiasFieldCorrection` [@n4], distributed with ANTs 2.2.0 [@ants, RRID:SCR_004757], and used as T1w-reference throughout the workflow.
The T1w-reference was then skull-stripped with a *Nipype* implementation of
the `antsBrainExtraction.sh` workflow (from ANTs), using OASIS30ANTs
as target template.
Brain tissue segmentation of cerebrospinal fluid (CSF),
white-matter (WM) and gray-matter (GM) was performed on
the brain-extracted T1w using `fast` [FSL 5.0.9, RRID:SCR_002823,
@fsl_fast].
Volume-based spatial normalization to one standard space (MNI152NLin2009cAsym) was performed through
nonlinear registration with `antsRegistration` (ANTs 2.2.0),
using brain-extracted versions of both T1w reference and the T1w template.
The following template was selected for spatial normalization:
*ICBM 152 Nonlinear Asymmetrical template version 2009c* [@mni152nlin2009casym, RRID:SCR_008796; TemplateFlow ID: MNI152NLin2009cAsym], 

Functional data preprocessing

: For each of the 1 BOLD runs found per subject (across all
tasks and sessions), the following preprocessing was performed.
First, a reference volume and its skull-stripped version were generated
using a custom methodology of *fMRIPrep*.
Susceptibility distortion correction (SDC) was omitted.
The BOLD reference was then co-registered to the T1w reference using
`flirt` [FSL 5.0.9, @flirt] with the boundary-based registration [@bbr]
cost-function.
Co-registration was configured with nine degrees of freedom to account
for distortions remaining in the BOLD reference.
Head-motion parameters with respect to the BOLD reference
(transformation matrices, and six corresponding rotation and translation
parameters) are estimated before any spatiotemporal filtering using
`mcflirt` [FSL 5.0.9, @mcflirt].
The BOLD time-series (including slice-timing correction when applied)
were resampled onto their original, native space by applying
the transforms to correct for head-motion.
These resampled BOLD time-series will be referred to as *preprocessed
BOLD in original space*, or just *preprocessed BOLD*.
The BOLD time-series were resampled into standard space,
generating a *preprocessed BOLD run in MNI152NLin2009cAsym space*.
First, a reference volume and its skull-stripped version were generated
using a custom methodology of *fMRIPrep*.
Several confounding time-series were calculated based on the
*preprocessed BOLD*: framewise displacement (FD), DVARS and
three region-wise global signals.
FD and DVARS are calculated for each functional run, both using their
implementations in *Nipype* [following the definitions by @power_fd_dvars].
The three global signals are extracted within the CSF, the WM, and
the whole-brain masks.
Additionally, a set of physiological regressors were extracted to
allow for component-based noise correction [*CompCor*, @compcor].
Principal components are estimated after high-pass filtering the
*preprocessed BOLD* time-series (using a discrete cosine filter with
128s cut-off) for the two *CompCor* variants: temporal (tCompCor)
and anatomical (aCompCor).
tCompCor components are then calculated from the top 5% variable
voxels within a mask covering the subcortical regions.
This subcortical mask is obtained by heavily eroding the brain mask,
which ensures it does not include cortical GM regions.
For aCompCor, components are calculated within the intersection of
the aforementioned mask and the union of CSF and WM masks calculated
in T1w space, after their projection to the native space of each
functional run (using the inverse BOLD-to-T1w transformation). Components
are also calculated separately within the WM and CSF masks.
For each CompCor decomposition, the *k* components with the largest singular
values are retained, such that the retained components' time series are
sufficient to explain 50 percent of variance across the nuisance mask (CSF,
WM, combined, or temporal). The remaining components are dropped from
consideration.
The head-motion estimates calculated in the correction step were also
placed within the corresponding confounds file.
The confound time series derived from head motion estimates and global
signals were expanded with the inclusion of temporal derivatives and
quadratic terms for each [@confounds_satterthwaite_2013].
Frames that exceeded a threshold of 0.5 mm FD or 1.5 standardised DVARS
were annotated as motion outliers.
All resamplings can be performed with *a single interpolation
step* by composing all the pertinent transformations (i.e. head-motion
transform matrices, susceptibility distortion correction when available,
and co-registrations to anatomical and output spaces).
Gridded (volumetric) resamplings were performed using `antsApplyTransforms` (ANTs),
configured with Lanczos interpolation to minimize the smoothing
effects of other kernels [@lanczos].
Non-gridded (surface) resamplings were performed using `mri_vol2surf`
(FreeSurfer).


Many internal operations of *fMRIPrep* use
*Nilearn* 0.6.2 [@nilearn, RRID:SCR_001362],
mostly within the functional processing workflow.
For more details of the pipeline, see [the section corresponding
to workflows in *fMRIPrep*'s documentation](https://fmriprep.readthedocs.io/en/latest/workflows.html "FMRIPrep's documentation").


### Copyright Waiver

The above boilerplate text was automatically generated by fMRIPrep
with the express intention that users should copy and paste this
text into their manuscripts *unchanged*.
It is released under the [CC0](https://creativecommons.org/publicdomain/zero/1.0/) license.

### References

