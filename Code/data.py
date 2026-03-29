"""
DartBrains Localizer Dataset Access
====================================

Helper functions to download and access the Pinel Localizer dataset
from HuggingFace Hub (dartbrains/localizer).

Files are downloaded on first access and cached locally by huggingface_hub.
"""

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "dartbrains/localizer"

SUBJECTS = [f"S{i:02d}" for i in range(1, 21)]

TR = 2.4  # seconds, from task-localizer_bold.json

CONDITIONS = [
    "audio_computation",
    "audio_left_hand",
    "audio_right_hand",
    "audio_sentence",
    "horizontal_checkerboard",
    "vertical_checkerboard",
    "video_computation",
    "video_left_hand",
    "video_right_hand",
    "video_sentence",
]


def _download(filename: str) -> str:
    """Download a file from the dartbrains/localizer dataset. Returns local cached path."""
    return hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")


def get_subjects() -> list[str]:
    """Return list of subject IDs (S01-S20)."""
    return list(SUBJECTS)


def get_tr() -> float:
    """Return the repetition time in seconds."""
    return TR


def get_file(subject: str, scope: str, suffix: str, extension: str = ".nii.gz") -> str:
    """Download and return the local path to a dataset file.

    Args:
        subject: Subject ID, e.g. "S01"
        scope: One of "raw", "derivatives", or "betas"
        suffix: BIDS suffix -- "bold", "T1w", "events", "confounds", "mask",
                or a condition name for betas (e.g. "audio_computation"),
                or "all" for the stacked betas file
        extension: File extension including dot, e.g. ".nii.gz", ".tsv"

    Returns:
        Local filesystem path to the cached file.
    """
    s = subject
    sub = f"sub-{s}"

    if scope == "betas":
        if suffix == "all":
            filename = f"derivatives/betas/{s}_betas{extension}"
        else:
            filename = f"derivatives/betas/{s}_beta_{suffix}{extension}"

    elif scope == "raw":
        if suffix == "events":
            filename = f"{sub}/func/{sub}_task-localizer_events.tsv"
        elif suffix == "bold":
            # No raw bold on HF -- use preprocessed
            filename = f"derivatives/fmriprep/{sub}/func/{sub}_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold{extension}"
        else:
            raise ValueError(f"Unknown raw suffix: {suffix}")

    elif scope == "derivatives":
        if suffix == "bold":
            filename = f"derivatives/fmriprep/{sub}/func/{sub}_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold{extension}"
        elif suffix == "T1w":
            filename = f"derivatives/fmriprep/{sub}/anat/{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w{extension}"
        elif suffix == "confounds":
            filename = f"derivatives/fmriprep/{sub}/func/{sub}_task-localizer_desc-confounds_regressors.tsv"
        elif suffix == "mask":
            filename = f"derivatives/fmriprep/{sub}/func/{sub}_task-localizer_space-MNI152NLin2009cAsym_desc-brain_mask{extension}"
        else:
            raise ValueError(f"Unknown derivatives suffix: {suffix}")
    else:
        raise ValueError(f"Unknown scope: {scope}. Use 'raw', 'derivatives', or 'betas'.")

    return _download(filename)


def load_events(subject: str) -> pd.DataFrame:
    """Download and load the events TSV for a subject as a DataFrame."""
    path = get_file(subject, scope="raw", suffix="events", extension=".tsv")
    return pd.read_csv(path, sep="\t")


def load_confounds(subject: str) -> pd.DataFrame:
    """Download and load the fmriprep confounds TSV for a subject."""
    path = get_file(subject, scope="derivatives", suffix="confounds")
    return pd.read_csv(path, sep="\t")
