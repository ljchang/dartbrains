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

def fetch_localizer_raw(n_subjects=None, get_anats=False, data_dir=None, url=None, resume=True, verbose=1):
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
    is generated on the remote server).  For example, download n_subjects=10, then n_subjects=20, etc.

    Parameters
    ----------
    n_subjects: int or list, optional
        The number or list of subjects to load. If None is given,
        all 94 subjects are used.
    get_anats: boolean
        Whether individual structural images should be fetched or not.
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
        - 'cmaps': string list
            Paths to nifti contrast maps
        - 'tmaps' string list (if 'get_tmaps' set to True)
            Paths to nifti t maps
        - 'masks': string list
            Paths to nifti files corresponding to the subjects individual masks
        - 'anats': string
            Path to nifti files corresponding to the subjects structural images
    References
    ----------
    Pinel, Philippe, et al.
    "Fast reproducible identification and large-scale databasing of
    individual functional cognitive networks."
    BMC neuroscience 8.1 (2007): 91.
    See Also
    ---------
    nilearn.datasets.fetch_localizer_calculation_task
    nilearn.datasets.fetch_localizer_button_task
    """

    if n_subjects is None:
        n_subjects = 94  # 94 subjects available
    if (isinstance(n_subjects, numbers.Number) and
                    ((n_subjects > 94) or (n_subjects < 1))):
        warnings.warn("Wrong value for \'n_subjects\' (%d). The maximum "
                      "value will be used instead (\'n_subjects=94\')")
        n_subjects = 94  # 94 subjects available

    opts = {'uncompress': True}

    if isinstance(n_subjects, numbers.Number):
        subject_mask = np.arange(1, n_subjects + 1)
        subject_id_max = "S%02d" % n_subjects
    else:
        subject_mask = np.array(n_subjects)
        subject_id_max = "S%02d" % np.max(n_subjects)
        n_subjects = len(n_subjects)
    subject_ids = ["S%02d" % s for s in subject_mask]
    data_type = "raw fMRI"
    label = "raw bold"
    rql_types = str.join(", ", ["\"%s\"" % x for x in [data_type]])
    root_url = "http://brainomics.cea.fr/localizer/"

    base_query = ("Any X,XT,XL,XI,XF,XD WHERE X is Scan, X type XT, "
              "X concerns S, "
              "X label XL, X identifier XI, "
              "X format XF, X description XD, "
              'S identifier <= "%s", ' % (subject_id_max, ) +
              'X type IN(%(types)s), X label "%(label)s"')

    urls = ["%sbrainomics_data.zip?rql=%s&vid=data-zip"
            % (root_url, _urllib.parse.quote(base_query % {"types": rql_types,
                                          "label": label}, safe=',()'))]
    filenames = []
    for subject_id in subject_ids:
        name_aux = str.replace(
                    str.join('_', [data_type, label]), ' ', '_')
        file_path = os.path.join("brainomics_data", subject_id, "%s.nii.gz" % name_aux)
        file_tarball_url = urls[0]
        filenames.append((file_path, file_tarball_url, opts))

    # Fetch anats if asked by user
    if get_anats:
        urls.append("%sbrainomics_data_anats.zip?rql=%s&vid=data-zip"
                    % (root_url,
                       _urllib.parse.quote(base_query % {"types": '"raw T1"',
                                                  "label": "raw anatomy"}, safe=',()')))
        for subject_id in subject_ids:
            file_path = os.path.join("brainomics_data", subject_id, "raw_T1_raw_anat_defaced.nii.gz")
            file_tarball_url = urls[-1]
            filenames.append((file_path, file_tarball_url, opts))

    # Fetch subject characteristics (separated in two files)
    if url is None:
        url_csv = ("%sdataset/cubicwebexport.csv?rql=%s&vid=csvexport"
                   % (root_url, _urllib.parse.quote("Any X WHERE X is Subject")))
        url_csv2 = ("%sdataset/cubicwebexport2.csv?rql=%s&vid=csvexport"
                    % (root_url,
                       _urllib.parse.quote("Any X,XI,XD WHERE X is QuestionnaireRun, "
                                    "X identifier XI, X datetime "
                                    "XD", safe=',')
                       ))
    else:
        url_csv = "%s/cubicwebexport.csv" % url
        url_csv2 = "%s/cubicwebexport2.csv" % url
    filenames += [("cubicwebexport.csv", url_csv, {}),
                  ("cubicwebexport2.csv", url_csv2, {})]

    # Actual data fetching
    dataset_name = 'brainomics_localizer'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    fdescr = _get_dataset_descr(dataset_name)
    files = _fetch_files(data_dir, filenames, verbose=verbose)
    anats = None

    # combine data from both covariates files into one single recarray
    from numpy.lib.recfunctions import join_by
    ext_vars_file2 = files[-1]
    csv_data2 = np.recfromcsv(ext_vars_file2, delimiter=';')
    files = files[:-1]
    ext_vars_file = files[-1]
    csv_data = np.recfromcsv(ext_vars_file, delimiter=';')
    files = files[:-1]
    # join_by sorts the output along the key
    csv_data = join_by('subject_id', csv_data, csv_data2,
                       usemask=False, asrecarray=True)[subject_mask - 1]
    if get_anats:
        anats = files[-n_subjects:]
        files = files[:-n_subjects]

    return Bunch(functional=files, structural=anats, ext_vars=csv_data, description=fdescr)
