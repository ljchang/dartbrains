"""Populate DARTBRAINS_STORAGE_ROOT with the files chapters read from the class copy.

The static site is built without anyone signed in, so chapters that use
``storage.course()`` render against a local directory shaped like the R2
bucket. The class copy mirrors Hugging Face's layout, so staging it is a
matter of downloading the same public files into the same relative paths.
Add to FILES whenever a chapter starts reading a new file from ``/course``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

FILES = {
    # GLM_Single_Subject_Model: one participant, preprocessed
    "dartbrains/localizer": [
        "derivatives/fmriprep/sub-S01/func/sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        "derivatives/fmriprep/sub-S01/func/sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
        "derivatives/fmriprep/sub-S01/func/sub-S01_task-localizer_desc-confounds_regressors.tsv",
        "sub-S01/func/sub-S01_task-localizer_events.tsv",
    ],
}


def main() -> None:
    root = Path(os.environ["DARTBRAINS_STORAGE_ROOT"])
    for repo, files in FILES.items():
        name = repo.split("/", 1)[1]
        for rel in files:
            dest = root / "course" / name / rel
            if dest.is_file():
                continue
            src = hf_hub_download(repo, rel, repo_type="dataset")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            print(f"staged {dest.relative_to(root)}")


if __name__ == "__main__":
    main()
