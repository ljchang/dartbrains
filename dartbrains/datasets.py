# -*- coding: utf-8 -*-

'''
dartbrains datasets
===================

functions to help download datasets

'''

import os
import numpy as np
from sklearn.datasets.base import Bunch
import numbers
from nilearn.datasets.utils import (_get_dataset_dir, _fetch_files, _get_dataset_descr)
from nilearn._utils.compat import _urllib


def fetch_localizer(subject_ids=None, get_anats=False, data_type='raw', data_dir=None, url=None, resume=True, verbose=1):
    """Download and load Brainomics Localizer dataset (94 subjects).
    "The Functional Localizer is a simple and fast acquisition
    procedure based on a 5-minute functional magnetic resonance
    imaging (fMRI) sequence that can be run as easily and as
    systematically as an anatomical scan. This protocol captures the
    cerebral bases of auditory and visual perception, motor actions,
    reading, language comprehension and mental calculation at an
    individual level. Individual functional maps are reliable and
    quite precise. The procedure is decribed in more detail on the
    Functional Localizer page." This code is modified from
    `fetch_localizer_contrasts` from nilearn.datasets.funcs.py.
    (see http://brainomics.cea.fr/localizer/)
    "Scientific results obtained using this dataset are described in
    Pinel et al., 2007" [1]

    Notes:
    It is better to perform several small requests than a big one because the
    Brainomics server has no cache (can lead to timeout while the archive
    is generated on the remote server).  For example, download
    n_subjects=np.array(1,10), then n_subjects=np.array(10,20), etc.

    Parameters
    ----------
    subject_ids: list
        List of Subject IDs (e.g., ['S01','S02']. If None is given,
        all 94 subjects are used.
    get_anats: boolean
        Whether individual structural images should be fetched or not.\
    data_type: string
        type of data to download. Valid values are ['raw','preprocessed']
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location.
    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).
    resume: bool
        Whether to resume download of a partly-downloaded file.
    verbose: int
        Verbosity level (0 means no message).
    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        - 'functional': string list
            Paths to nifti contrast maps
        - 'structural' string
            Path to nifti files corresponding to the subjects structural images
    References
    ----------
    Pinel, Philippe, et al.
    "Fast reproducible identification and large-scale databasing of
    individual functional cognitive networks."
    BMC neuroscience 8.1 (2007): 91.

    """

    if subject_ids is None:
        subject_ids = ['S%02d' % x for x in np.arange(1,95)]
    elif not isinstance(subject_ids, (list)):
        raise ValueError("subject_ids must be a list of subject ids (e.g., ['S01','S02'])")

    if data_type == 'raw':
        dat_type = "raw fMRI"
        dat_label = "raw bold"
        anat_type = "raw T1"
        anat_label = "raw anatomy"
    elif data_type == 'preprocessed':
        dat_type = "preprocessed fMRI"
        dat_label = "bold"
        anat_type = "normalized T1"
        anat_label = "anatomy"
    else:
        raise ValueError("Only ['raw','preprocessed'] data_types are currently supported.")

    root_url = "http://brainomics.cea.fr/localizer/"
    dataset_name = 'brainomics_localizer'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    fdescr = _get_dataset_descr(dataset_name)
    opts = {'uncompress': True}

    bold_files = []; anat_files = [];
    for subject_id in subject_ids:
        base_query = ("Any X,XT,XL,XI,XF,XD WHERE X is Scan, X type XT, "
                  "X concerns S, "
                  "X label XL, X identifier XI, "
                  "X format XF, X description XD, "
                  'S identifier = "%s", ' % (subject_id, ) +
                  'X type IN(%(types)s), X label "%(label)s"')

        file_tarball_url = "%sbrainomics_data.zip?rql=%s&vid=data-zip" % (root_url, _urllib.parse.quote(base_query % {"types": "\"%s\"" % dat_type,  "label": dat_label}, safe=',()'))
        name_aux = str.replace(str.join('_', [dat_type, dat_label]), ' ', '_')
        file_path = os.path.join("brainomics_data", subject_id, "%s.nii.gz" % name_aux)
        bold_files.append(_fetch_files(data_dir, [(file_path, file_tarball_url, opts)], verbose=verbose))

        if get_anats:
            file_tarball_url = "%sbrainomics_data_anats.zip?rql=%s&vid=data-zip" % (root_url, _urllib.parse.quote(base_query % {"types": "\"%s\"" % anat_type, "label": anat_label}, safe=',()'))
            if data_type == 'raw':
                anat_name_aux = "raw_T1_raw_anat_defaced.nii.gz"
            elif data_type == 'preprocessed':
                anat_name_aux = "normalized_T1_anat_defaced.nii.gz"
            file_path = os.path.join("brainomics_data", subject_id, anat_name_aux)
            anat_files.append(_fetch_files(data_dir, [(file_path, file_tarball_url, opts)], verbose=verbose))

    # Fetch subject characteristics (separated in two files)
    if url is None:
        url_csv = ("%sdataset/cubicwebexport.csv?rql=%s&vid=csvexport"
                   % (root_url, _urllib.parse.quote("Any X WHERE X is Subject")))
        url_csv2 = ("%sdataset/cubicwebexport2.csv?rql=%s&vid=csvexport"
                    % (root_url,
                       _urllib.parse.quote("Any X,XI,XD WHERE X is QuestionnaireRun, "
                                    "X identifier XI, X datetime "
                                    "XD", safe=',')))
    else:
        url_csv = "%s/cubicwebexport.csv" % url
        url_csv2 = "%s/cubicwebexport2.csv" % url

    filenames = [("cubicwebexport.csv", url_csv, {}),("cubicwebexport2.csv", url_csv2, {})]
    csv_files = _fetch_files(data_dir, filenames, verbose=verbose)
    metadata = pd.merge(pd.read_csv(csv_files[0], sep=';'), pd.read_csv(csv_files[1], sep=';'), on='"subject_id"')
    metadata.to_csv(os.path.join(data_dir,'metadata.csv'))
    for x in ['cubicwebexport.csv','cubicwebexport2.csv']:
        os.remove(os.path.join(data_dir, x))

    if not get_anats:
        anat_files = None

    return Bunch(functional=bold_files, structural=anat_files, ext_vars=metadata, description=fdescr)
