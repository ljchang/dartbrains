# HF Dataset Config Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generator CLI that emits uniform path-index `load_dataset` configs (one comma-CSV per config) across the `localizer`, `sherlock`, and `paranoia` HuggingFace datasets, normalize sherlock/paranoia to `derivatives/fmriprep/`, and update loaders + notebook examples in lockstep.

**Architecture:** Pure, unit-tested core in `dartbrains-tools/scripts/hf_configs/` (BIDS/label parsing → index rows → README `configs:` YAML) wired by a thin argparse CLI (`scripts/generate_hf_configs.py`). Network I/O (list files, upload, verify) is isolated in one thin module. Layout normalization is a one-time reviewed git-mv procedure per repo. Loader path strings and the `Download_Data.py` examples follow.

**Tech Stack:** Python 3.11+, `huggingface_hub>=1.0`, `datasets` (for `--check`), `pytest`, `fnmatch`/`re` (no new runtime deps).

## Global Constraints

- Generator code lives in `dartbrains-tools/scripts/` and is **NOT** shipped in the installed package (ops tool, not a runtime dependency). No new entries under `src/`.
- No new runtime dependencies in `pyproject.toml [project].dependencies`. `datasets` is used only by the CLI's `--check` path and the tests that need it — invoke via `uv run --with datasets`.
- Every generated config is a **comma-separated `.csv`** loaded by the default HF `csv` builder. No `niftifolder`, no per-config `sep`.
- Every per-file config row is `{path (repo-relative), <labels...>}`. `path` column is always first.
- `subject` label = the raw BIDS entity value after `sub-` (e.g. `S01`, `01`, `tb2994`) — do not normalize across datasets.
- HF writes go **to a branch** (`refs/pr/*` or a named branch) for review, never directly to `main`.
- Commit messages: no Claude attribution, no `Co-Authored-By` (per repo CLAUDE.md).
- Datasets and their canonical layout (post-normalization): all three use `derivatives/fmriprep/**`. Extra configs: localizer has `events`, `betas`, `participants`; sherlock has `onsets`; paranoia has `participants`.

---

## File Structure

**dartbrains-tools (generator + loaders):**
- Create `scripts/hf_configs/__init__.py` — package marker.
- Create `scripts/hf_configs/labels.py` — `parse_bids_entities`, `extract_beta_labels`, `extract_onset_kind` (pure).
- Create `scripts/hf_configs/specs.py` — `DATASETS` per-dataset config literals (pure data).
- Create `scripts/hf_configs/index.py` — `build_index`, `render_readme_configs`, `rows_to_csv` (pure).
- Create `scripts/hf_configs/hub.py` — network wrappers (`list_files`, `upload_text`, `read_text`) (thin, not unit-tested).
- Create `scripts/generate_hf_configs.py` — argparse CLI (`index`, `check` subcommands).
- Create `tests/conftest.py` — put `scripts/` on `sys.path` so tests can `import hf_configs.*`.
- Create `tests/test_hf_config_labels.py`, `tests/test_hf_config_index.py`.
- Modify `src/dartbrains_tools/data/sherlock.py`, `src/dartbrains_tools/data/paranoia.py` — `fmriprep/` → `derivatives/fmriprep/`.
- Modify `tests/test_data_sherlock.py`, `tests/test_data_paranoia.py` — assert new paths.
- Modify `pyproject.toml` — version bump.

**HF dataset repos (localizer, sherlock, paranoia):** normalized layout, generated `*.csv` index files, rewritten `README.md`.

**dartbrains (course):**
- Modify `content/Download_Data.py` — `betas` example to path-index shape.

---

## Phase 1 — Generator core (offline, TDD)

### Task 1: BIDS + label parsing

**Files:**
- Create: `dartbrains-tools/scripts/hf_configs/__init__.py`
- Create: `dartbrains-tools/scripts/hf_configs/labels.py`
- Create: `dartbrains-tools/tests/conftest.py`
- Test: `dartbrains-tools/tests/test_hf_config_labels.py`

**Interfaces:**
- Produces:
  - `parse_bids_entities(path: str) -> dict[str, str]` — returns present keys among `subject, task, run, space, desc`; `subject` is the value after `sub-`.
  - `extract_beta_labels(path: str) -> dict[str, str]` — `{subject, condition, type}` where `type` ∈ `{individual, stacked}` (stacked omits `condition`).
  - `extract_onset_kind(path: str) -> dict[str, str]` — `{kind}` ∈ `{watch, recall, crop}`.

- [ ] **Step 1: Create the package marker and conftest**

Create `dartbrains-tools/scripts/hf_configs/__init__.py`:

```python
"""Offline core for the HF dataset config generator (not shipped in the package)."""
```

Create `dartbrains-tools/tests/conftest.py`:

```python
import sys
from pathlib import Path

# Make scripts/hf_configs importable in tests without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
```

- [ ] **Step 2: Write the failing test**

Create `dartbrains-tools/tests/test_hf_config_labels.py`:

```python
"""Pure filename-parsing tests for the config generator -- no network."""

from hf_configs.labels import (
    extract_beta_labels,
    extract_onset_kind,
    parse_bids_entities,
)


def test_parse_bids_entities_bold():
    p = ("derivatives/fmriprep/sub-S01/func/"
         "sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    assert parse_bids_entities(p) == {
        "subject": "S01",
        "task": "localizer",
        "space": "MNI152NLin2009cAsym",
        "desc": "preproc",
    }


def test_parse_bids_entities_with_run():
    p = "derivatives/fmriprep/sub-tb2994/func/sub-tb2994_task-story_run-2_desc-confounds_regressors.tsv"
    ent = parse_bids_entities(p)
    assert ent["subject"] == "tb2994"
    assert ent["task"] == "story"
    assert ent["run"] == "2"


def test_parse_bids_entities_sherlock_numeric_subject():
    p = "derivatives/fmriprep/sub-01/func/sub-01_task-sherlockPart1_desc-brain_mask.nii.gz"
    assert parse_bids_entities(p)["subject"] == "01"


def test_extract_beta_labels_individual():
    assert extract_beta_labels("derivatives/betas/S01_beta_audio_computation.nii.gz") == {
        "subject": "S01",
        "condition": "audio_computation",
        "type": "individual",
    }


def test_extract_beta_labels_stacked():
    assert extract_beta_labels("derivatives/betas/S07_betas.nii.gz") == {
        "subject": "S07",
        "type": "stacked",
    }


def test_extract_onset_kind():
    assert extract_onset_kind("onsets/Sherlock_Watch_Scene_N50_Onsets.csv") == {"kind": "watch"}
    assert extract_onset_kind("onsets/Sherlock_Recall_Scene_n50_Onsets.csv") == {"kind": "recall"}
    assert extract_onset_kind("onsets/Sherlock_Crop_Onsets.csv") == {"kind": "crop"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd dartbrains-tools && uv run pytest tests/test_hf_config_labels.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hf_configs.labels'`

- [ ] **Step 4: Write minimal implementation**

Create `dartbrains-tools/scripts/hf_configs/labels.py`:

```python
"""Extract label columns from HF dataset file paths.

The shared BIDS parser handles the fmriprep-style names; two small custom
extractors handle the non-BIDS beta and onset filenames.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

# BIDS `key-value` tokens we care about, mapped to output column names.
_BIDS_KEYS = {"sub": "subject", "task": "task", "run": "run",
              "space": "space", "desc": "desc"}


def parse_bids_entities(path: str) -> dict[str, str]:
    """Return BIDS entity values parsed from the basename of *path*.

    Only the keys in ``_BIDS_KEYS`` are returned, and only when present.
    ``subject`` is the raw value after ``sub-`` (e.g. ``S01``, ``01``, ``tb2994``).
    """
    stem = PurePosixPath(path).name
    out: dict[str, str] = {}
    for token in stem.split("_"):
        if "-" not in token:
            continue
        key, _, value = token.partition("-")
        col = _BIDS_KEYS.get(key)
        if col is not None:
            out[col] = value
    return out


_BETA_INDIVIDUAL = re.compile(r"(?P<subject>S\d+)_beta_(?P<condition>.+)\.nii\.gz$")
_BETA_STACKED = re.compile(r"(?P<subject>S\d+)_betas\.nii\.gz$")


def extract_beta_labels(path: str) -> dict[str, str]:
    """Labels for localizer beta maps (filenames are not BIDS-encoded)."""
    name = PurePosixPath(path).name
    m = _BETA_INDIVIDUAL.match(name)
    if m:
        return {"subject": m["subject"], "condition": m["condition"], "type": "individual"}
    m = _BETA_STACKED.match(name)
    if m:
        return {"subject": m["subject"], "type": "stacked"}
    raise ValueError(f"Unrecognized beta filename: {name!r}")


_ONSET_KINDS = {"watch": "watch", "recall": "recall", "crop": "crop"}


def extract_onset_kind(path: str) -> dict[str, str]:
    """Label for sherlock onset CSVs -- matched by keyword in the filename."""
    name = PurePosixPath(path).name.lower()
    for needle, kind in _ONSET_KINDS.items():
        if needle in name:
            return {"kind": kind}
    raise ValueError(f"Unrecognized onset filename: {name!r}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd dartbrains-tools && uv run pytest tests/test_hf_config_labels.py -q`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
cd dartbrains-tools
git add scripts/hf_configs/__init__.py scripts/hf_configs/labels.py tests/conftest.py tests/test_hf_config_labels.py
git commit -m "feat(configs): add BIDS + beta/onset label parsers for HF config generator"
```

---

### Task 2: Index building + README rendering

**Files:**
- Create: `dartbrains-tools/scripts/hf_configs/index.py`
- Test: `dartbrains-tools/tests/test_hf_config_index.py`

**Interfaces:**
- Consumes: `hf_configs.labels.parse_bids_entities` and the optional per-config `labels` callable.
- Produces:
  - `glob_to_regex(glob: str) -> re.Pattern` — `**` matches across `/`, `*` matches within a path segment.
  - `build_index(files: list[str], config: dict) -> list[dict]` — filter `files` by `config["glob"]`, apply `parse_bids_entities` plus optional `config["labels"]` (a callable overriding/augmenting entity keys), return rows `{"path": f, **labels}` sorted by path.
  - `rows_to_csv(rows: list[dict]) -> str` — comma-CSV text; header is `path` followed by the union of label keys in first-seen order; missing cells empty.
  - `render_readme_configs(dataset: dict) -> str` — the YAML `configs:` block; index configs point `data_files[0].path` at `f"{name}.csv"`; the config named in `dataset["default"]` gets `default: true`.

- [ ] **Step 1: Write the failing test**

Create `dartbrains-tools/tests/test_hf_config_index.py`:

```python
"""Pure index-building + README-rendering tests -- no network."""

from hf_configs.index import (
    build_index,
    glob_to_regex,
    render_readme_configs,
    rows_to_csv,
)
from hf_configs.labels import extract_beta_labels, parse_bids_entities

FILES = [
    "derivatives/fmriprep/sub-S01/func/sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "derivatives/fmriprep/sub-S02/func/sub-S02_task-localizer_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "derivatives/fmriprep/sub-S01/func/sub-S01_task-localizer_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    "derivatives/betas/S01_beta_audio_computation.nii.gz",
    "derivatives/betas/S01_betas.nii.gz",
    "derivatives/betas/metadata.csv",           # must be excluded by the nifti glob
    "sub-S01/func/sub-S01_task-localizer_events.tsv",
]


def test_glob_star_stops_at_slash():
    rx = glob_to_regex("derivatives/fmriprep/*/func/*_bold.nii.gz")
    # single * does NOT cross a directory boundary
    assert not rx.match(
        "derivatives/fmriprep/sub-S01/anat/extra/sub-S01_desc-preproc_bold.nii.gz"
    )


def test_glob_doublestar_crosses_slash():
    rx = glob_to_regex("derivatives/fmriprep/**/*_desc-preproc_bold.nii.gz")
    assert rx.match(FILES[0])


def test_build_index_bold_uses_bids_labels():
    cfg = {"glob": "derivatives/fmriprep/**/*_desc-preproc_bold.nii.gz"}
    rows = build_index(FILES, cfg)
    assert [r["path"] for r in rows] == [FILES[0], FILES[1]]
    assert rows[0]["subject"] == "S01"
    assert rows[0]["task"] == "localizer"


def test_build_index_betas_excludes_metadata_csv():
    cfg = {"glob": "derivatives/betas/*.nii.gz", "labels": extract_beta_labels}
    rows = build_index(FILES, cfg)
    paths = [r["path"] for r in rows]
    assert "derivatives/betas/metadata.csv" not in paths
    assert {r["type"] for r in rows} == {"individual", "stacked"}


def test_rows_to_csv_header_and_blanks():
    rows = [
        {"path": "a.nii.gz", "subject": "S01", "type": "individual", "condition": "x"},
        {"path": "b.nii.gz", "subject": "S01", "type": "stacked"},
    ]
    csv = rows_to_csv(rows)
    lines = csv.strip().splitlines()
    assert lines[0] == "path,subject,type,condition"
    assert lines[2] == "b.nii.gz,S01,stacked,"   # missing condition -> empty cell


def test_render_readme_configs_marks_default_and_points_at_csv():
    dataset = {
        "repo": "dartbrains/localizer",
        "default": "betas",
        "configs": {
            "bold": {"glob": "x/**"},
            "betas": {"glob": "y/*.nii.gz", "labels": extract_beta_labels},
        },
    }
    yaml = render_readme_configs(dataset)
    assert "config_name: bold" in yaml
    assert "path: bold.csv" in yaml
    assert "config_name: betas" in yaml
    assert "default: true" in yaml
    # exactly one default
    assert yaml.count("default: true") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dartbrains-tools && uv run pytest tests/test_hf_config_index.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hf_configs.index'`

- [ ] **Step 3: Write minimal implementation**

Create `dartbrains-tools/scripts/hf_configs/index.py`:

```python
"""Build path-index rows and render the README `configs:` block (pure)."""

from __future__ import annotations

import io
import csv as _csv
import re

from .labels import parse_bids_entities


def glob_to_regex(glob: str) -> re.Pattern[str]:
    """Translate a glob to a full-match regex.

    ``**`` matches any characters including ``/``; ``*`` matches any run of
    characters except ``/``; ``?`` matches a single non-``/`` character.
    """
    out = []
    i = 0
    while i < len(glob):
        c = glob[i]
        if glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def build_index(files: list[str], config: dict) -> list[dict]:
    """Select files matching ``config['glob']`` and attach label columns.

    Base labels come from :func:`parse_bids_entities`; if ``config['labels']``
    is provided it is called per path and its keys override/augment the base.
    Rows are ``{'path': f, **labels}`` sorted by path.
    """
    rx = glob_to_regex(config["glob"])
    labeler = config.get("labels")
    rows = []
    for f in sorted(files):
        if not rx.match(f):
            continue
        labels = parse_bids_entities(f)
        if labeler is not None:
            labels = {**labels, **labeler(f)}
        rows.append({"path": f, **labels})
    return rows


def rows_to_csv(rows: list[dict]) -> str:
    """Serialize rows to comma-CSV. Header = 'path' + union of label keys
    (first-seen order); missing cells are empty."""
    fields = ["path"]
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    buf = io.StringIO()
    w = _csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def render_readme_configs(dataset: dict) -> str:
    """Render the README frontmatter `configs:` block for a dataset spec."""
    lines = ["configs:"]
    default = dataset.get("default")
    for name, cfg in dataset["configs"].items():
        lines.append(f"  - config_name: {name}")
        if name == default:
            lines.append("    default: true")
        target = cfg["content_out"] if "content" in cfg else f"{name}.csv"
        lines.append("    data_files:")
        lines.append("      - split: train")
        lines.append(f"        path: {target}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dartbrains-tools && uv run pytest tests/test_hf_config_index.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
cd dartbrains-tools
git add scripts/hf_configs/index.py tests/test_hf_config_index.py
git commit -m "feat(configs): add index builder + README configs renderer"
```

---

### Task 3: Per-dataset specs

**Files:**
- Create: `dartbrains-tools/scripts/hf_configs/specs.py`
- Test: extend `dartbrains-tools/tests/test_hf_config_index.py`

**Interfaces:**
- Consumes: `extract_beta_labels`, `extract_onset_kind`.
- Produces: `DATASETS: dict[str, dict]` keyed by repo id. Each dataset dict has `repo`, `default`, and `configs`. A config is either an index config (`{"glob": ..., "labels"?: callable}`) or a content config (`{"content": "<src path>", "content_out": "<out csv>"}`).

- [ ] **Step 1: Write the failing test** (append to `tests/test_hf_config_index.py`)

```python
def test_specs_cover_expected_configs():
    from hf_configs.specs import DATASETS

    assert set(DATASETS) == {
        "dartbrains/localizer",
        "dartbrains/sherlock",
        "dartbrains/paranoia",
    }
    loc = DATASETS["dartbrains/localizer"]["configs"]
    assert set(loc) == {"bold", "confounds", "mask", "events", "betas", "participants"}
    assert set(DATASETS["dartbrains/sherlock"]["configs"]) == {
        "bold", "confounds", "mask", "onsets",
    }
    assert set(DATASETS["dartbrains/paranoia"]["configs"]) == {
        "bold", "confounds", "mask", "participants",
    }
    # participants is a content config; bold is an index config
    assert "content" in loc["participants"]
    assert "glob" in loc["bold"]


def test_all_three_share_the_fmriprep_globs():
    from hf_configs.specs import DATASETS

    globs = {
        repo: DATASETS[repo]["configs"]["bold"]["glob"]
        for repo in DATASETS
    }
    assert len(set(globs.values())) == 1  # identical bold glob everywhere
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dartbrains-tools && uv run pytest tests/test_hf_config_index.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hf_configs.specs'`

- [ ] **Step 3: Write minimal implementation**

Create `dartbrains-tools/scripts/hf_configs/specs.py`:

```python
"""Per-dataset config specs. The only place dataset-specific knowledge lives.

After layout normalization all three repos use ``derivatives/fmriprep/``, so
the bold/confounds/mask globs are identical everywhere.
"""

from __future__ import annotations

from .labels import extract_beta_labels, extract_onset_kind

_BOLD = "derivatives/fmriprep/**/*_desc-preproc_bold.nii.gz"
_CONFOUNDS = "derivatives/fmriprep/**/*_desc-confounds_*.tsv"
_MASK = "derivatives/fmriprep/**/*_desc-brain_mask.nii.gz"

_FMRIPREP = {
    "bold": {"glob": _BOLD},
    "confounds": {"glob": _CONFOUNDS},
    "mask": {"glob": _MASK},
}

DATASETS: dict[str, dict] = {
    "dartbrains/localizer": {
        "repo": "dartbrains/localizer",
        "default": "bold",
        "configs": {
            **_FMRIPREP,
            "events": {"glob": "sub-*/func/*_events.tsv"},
            "betas": {"glob": "derivatives/betas/*.nii.gz", "labels": extract_beta_labels},
            "participants": {"content": "participants.tsv", "content_out": "participants.csv"},
        },
    },
    "dartbrains/sherlock": {
        "repo": "dartbrains/sherlock",
        "default": "bold",
        "configs": {
            **_FMRIPREP,
            "onsets": {"glob": "onsets/*.csv", "labels": extract_onset_kind},
        },
    },
    "dartbrains/paranoia": {
        "repo": "dartbrains/paranoia",
        "default": "bold",
        "configs": {
            **_FMRIPREP,
            "participants": {"content": "participants.tsv", "content_out": "participants.csv"},
        },
    },
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dartbrains-tools && uv run pytest tests/test_hf_config_index.py -q`
Expected: PASS (8 tests total in file)

- [ ] **Step 5: Commit**

```bash
cd dartbrains-tools
git add scripts/hf_configs/specs.py tests/test_hf_config_index.py
git commit -m "feat(configs): add per-dataset config specs"
```

---

### Task 4: Network wrappers + CLI (dry-run)

**Files:**
- Create: `dartbrains-tools/scripts/hf_configs/hub.py`
- Create: `dartbrains-tools/scripts/generate_hf_configs.py`

**Interfaces:**
- Consumes: `DATASETS`, `build_index`, `rows_to_csv`, `render_readme_configs`.
- Produces (`hub.py`):
  - `list_files(repo: str) -> list[str]`
  - `read_text(repo: str, path: str) -> str`
  - `upload_files(repo: str, files: dict[str, str], branch: str, message: str) -> str` — `files` maps repo-relative path → text content; returns the commit/PR URL.
- Produces (CLI): `python scripts/generate_hf_configs.py index --repo <id> [--dry-run] [--branch <name>]` and `... check --repo <id>`.

- [ ] **Step 1: Write `hub.py`**

Create `dartbrains-tools/scripts/hf_configs/hub.py`:

```python
"""Thin HuggingFace Hub I/O for the generator. Network only -- no logic."""

from __future__ import annotations

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

_api = HfApi()


def list_files(repo: str) -> list[str]:
    return list_repo_files(repo, repo_type="dataset")


def read_text(repo: str, path: str) -> str:
    local = hf_hub_download(repo, path, repo_type="dataset")
    with open(local, encoding="utf-8") as fh:
        return fh.read()


def upload_files(repo: str, files: dict[str, str], branch: str, message: str) -> str:
    """Upload each {repo_path: text} on *branch*, creating the branch/PR."""
    from huggingface_hub import CommitOperationAdd

    ops = [
        CommitOperationAdd(path_in_repo=p, path_or_fileobj=text.encode("utf-8"))
        for p, text in files.items()
    ]
    info = _api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=ops,
        commit_message=message,
        create_pr=True,
    )
    return info.pr_url or ""
```

- [ ] **Step 2: Write the CLI**

Create `dartbrains-tools/scripts/generate_hf_configs.py`:

```python
#!/usr/bin/env python
"""Generate uniform path-index load_dataset configs for the dartbrains datasets.

Usage:
    python scripts/generate_hf_configs.py index --repo dartbrains/localizer --dry-run
    python scripts/generate_hf_configs.py index --repo dartbrains/localizer --branch add-configs
    python scripts/generate_hf_configs.py check --repo dartbrains/localizer
"""

from __future__ import annotations

import argparse
import sys

from hf_configs import hub
from hf_configs.index import build_index, render_readme_configs, rows_to_csv
from hf_configs.specs import DATASETS


def _build_outputs(repo: str) -> dict[str, str]:
    """Return {repo_path: text} for every generated file (CSVs + participants)."""
    spec = DATASETS[repo]
    files = hub.list_files(repo)
    out: dict[str, str] = {}
    for name, cfg in spec["configs"].items():
        if "content" in cfg:
            raw = hub.read_text(repo, cfg["content"])
            out[cfg["content_out"]] = _tsv_to_csv(raw)
            continue
        rows = build_index(files, cfg)
        if not rows:
            print(f"  WARNING: config {name!r} matched 0 files", file=sys.stderr)
        out[f"{name}.csv"] = rows_to_csv(rows)
    return out


def _tsv_to_csv(text: str) -> str:
    import csv
    import io

    reader = csv.reader(io.StringIO(text), delimiter="\t")
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in reader:
        writer.writerow(row)
    return buf.getvalue()


def _rewrite_readme(repo: str, configs_yaml: str) -> str:
    """Replace the frontmatter `configs:` block in the repo README with *configs_yaml*."""
    readme = hub.read_text(repo, "README.md")
    import re

    # Frontmatter is delimited by the first two `---` lines.
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", readme, flags=re.DOTALL)
    if not m:
        raise ValueError(f"{repo} README has no YAML frontmatter")
    front, body = m.group(1), m.group(2)
    # Drop any existing configs: block (until the next top-level key or end).
    front = re.sub(r"(?ms)^configs:\n(?:[ \t]+.*\n?)*", "", front)
    front = front.rstrip("\n") + "\n" + configs_yaml.rstrip("\n") + "\n"
    return f"---\n{front}---\n{body}"


def cmd_index(args):
    repo = args.repo
    outputs = _build_outputs(repo)
    outputs["README.md"] = _rewrite_readme(repo, render_readme_configs(DATASETS[repo]))
    if args.dry_run:
        for path, text in outputs.items():
            preview = text if path.endswith("README.md") else "\n".join(text.splitlines()[:4])
            print(f"\n===== {path} ({len(text.splitlines())} lines) =====")
            print(preview)
        return
    url = hub.upload_files(
        repo, outputs, branch=args.branch,
        message="Generate uniform path-index load_dataset configs",
    )
    print(f"Opened PR: {url}")


def cmd_check(args):
    from datasets import load_dataset

    repo = args.repo
    files = hub.list_files(repo)
    ok = True
    for name, cfg in DATASETS[repo]["configs"].items():
        if "content" in cfg:
            continue
        expected = len(build_index(files, cfg))
        got = len(load_dataset(repo, name, split="train"))
        status = "OK" if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        print(f"  {name:12s} expected={expected:4d} got={got:4d} {status}")
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True)
    pi = sub.add_parser("index", help="generate + (optionally) upload configs")
    pi.add_argument("--repo", required=True, choices=list(DATASETS))
    pi.add_argument("--dry-run", action="store_true")
    pi.add_argument("--branch", default="add-configs")
    pi.set_defaults(func=cmd_index)
    pc = sub.add_parser("check", help="verify row counts via load_dataset")
    pc.add_argument("--repo", required=True, choices=list(DATASETS))
    pc.set_defaults(func=cmd_check)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the CLI parses and dry-runs against localizer**

localizer is already on `derivatives/fmriprep/`, so `index --dry-run` works before any migration.

Run: `cd dartbrains-tools && uv run python scripts/generate_hf_configs.py index --repo dartbrains/localizer --dry-run`
Expected: prints a `bold.csv` / `confounds.csv` / `mask.csv` / `events.csv` / `betas.csv` / `participants.csv` preview and a rewritten `README.md`; `betas.csv` header is `path,subject,condition,type`; no `metadata.csv` row; no `0 files` warnings for `bold`/`betas`/`events`.

- [ ] **Step 4: Commit**

```bash
cd dartbrains-tools
git add scripts/hf_configs/hub.py scripts/generate_hf_configs.py
git commit -m "feat(configs): add hub I/O wrappers + generator CLI (index/check)"
```

---

## Phase 2 — Layout normalization (sherlock, paranoia)

One-time, reviewed git operation per repo. Nifti files are Git-LFS pointers, so `git mv` rewrites only pointer text — no blob transfer. **Run per repo; open on a branch; review the diff (it must be pure renames); merge via the HF PR UI.**

### Task 5: Move `fmriprep/ → derivatives/fmriprep/` and delete `.datalad/`

**Files:** (remote HF repos `dartbrains/sherlock`, `dartbrains/paranoia`) — no files in this repo.

- [ ] **Step 1: Clone sherlock pointers-only**

```bash
cd /tmp && rm -rf sherlock && \
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/dartbrains/sherlock && \
cd sherlock && git checkout -b normalize-derivatives
```

- [ ] **Step 2: Move the tree and delete cruft**

```bash
cd /tmp/sherlock
git mv fmriprep derivatives/fmriprep 2>/dev/null || (mkdir -p derivatives && git mv fmriprep derivatives/fmriprep)
git rm -r --quiet .datalad
git status --short | head
```

Expected: every line is `R  fmriprep/... -> derivatives/fmriprep/...` (renames) or `D  .datalad/...`. No `A`/`M` on binary blobs.

- [ ] **Step 3: Verify no blob was smudged (pointers intact)**

Run: `cd /tmp/sherlock && git show :derivatives/fmriprep/$(ls derivatives/fmriprep | head -1)/func/*bold.nii.gz 2>/dev/null | head -3`
Expected: shows a `version https://git-lfs.github.com/spec/v1` pointer, not binary.

- [ ] **Step 4: Push the branch**

```bash
cd /tmp/sherlock
git commit -m "Normalize layout: move fmriprep under derivatives/, drop .datalad"
git push -u origin normalize-derivatives
```

Then open the branch as a PR in the HF web UI and confirm the diff is renames-only.

- [ ] **Step 5: Repeat Steps 1-4 for paranoia**

Substitute `paranoia` for `sherlock` throughout. (Paranoia has no `onsets/`; otherwise identical.)

- [ ] **Step 6: Add the OpenNeuro raw-data note to each README**

In each repo's `README.md` body (via the HF web editor or the same branch), add under Dataset Structure:

```markdown
> **Raw BIDS data** for this dataset is not hosted here — only fmriprep
> derivatives. Fetch raw data from OpenNeuro.
```

- [ ] **Step 7: Merge both PRs in the HF UI** after confirming renames-only diffs.

---

## Phase 3 — Publish configs (all three repos)

### Task 6: Generate + upload configs, then verify

**Files:** remote HF repos.

- [ ] **Step 1: Dry-run each repo** (sherlock/paranoia only valid AFTER Phase 2 merges)

```bash
cd dartbrains-tools
for r in dartbrains/localizer dartbrains/sherlock dartbrains/paranoia; do
  echo "##### $r"; uv run python scripts/generate_hf_configs.py index --repo $r --dry-run 2>&1 | grep -E "=====|WARNING"
done
```

Expected: each repo lists its expected configs; **no `WARNING: ... 0 files`** lines (a 0-file config means a wrong glob or the migration didn't land).

- [ ] **Step 2: Upload each repo on a branch**

```bash
cd dartbrains-tools
for r in dartbrains/localizer dartbrains/sherlock dartbrains/paranoia; do
  uv run python scripts/generate_hf_configs.py index --repo $r --branch add-configs
done
```

Expected: prints an `Opened PR:` URL per repo.

- [ ] **Step 3: Verify row counts against each open PR**

For each repo, after the PR ref exists, run `check` (it loads from `main`; run it after merge, or point `load_dataset(revision=...)` at the PR ref manually). Minimal gate — run after merge in Step 5:

```bash
cd dartbrains-tools
uv run --with datasets python scripts/generate_hf_configs.py check --repo dartbrains/localizer
```

Expected: every config prints `OK` (got == expected).

- [ ] **Step 4: Update README prose** (Quick Start) in each PR to the path-index shape, replacing any `ds[0]["nifti"]` example with:

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import nibabel as nib

ds = load_dataset("dartbrains/localizer", "betas")     # index of {path, subject, condition, type}
row = ds["train"][0]
img = nib.load(hf_hub_download("dartbrains/localizer", row["path"], repo_type="dataset"))
```

- [ ] **Step 5: Merge each PR** in the HF UI, then re-run Step 3 `check` for all three — all `OK`.

---

## Phase 4 — Loaders + notebook (dartbrains-tools, dartbrains)

### Task 7: Update sherlock/paranoia loader paths

**Files:**
- Modify: `dartbrains-tools/src/dartbrains_tools/data/sherlock.py`
- Modify: `dartbrains-tools/src/dartbrains_tools/data/paranoia.py`
- Modify: `dartbrains-tools/tests/test_data_sherlock.py`, `tests/test_data_paranoia.py`

**Interfaces:**
- Produces: loader helpers that resolve to `derivatives/fmriprep/...`. No signature changes.

- [ ] **Step 1: Update the failing tests first**

In `tests/test_data_paranoia.py`, change every expected path from `fmriprep/...` to `derivatives/fmriprep/...`. Example:

```python
def test_get_file_bold(monkeypatch):
    captured = {}
    monkeypatch.setattr(paranoia, "_download", lambda f: captured.setdefault("f", f))
    paranoia.get_file("sub-tb2994", run=1, suffix="bold")
    assert captured["f"] == (
        "derivatives/fmriprep/sub-tb2994/func/"
        "sub-tb2994_task-story_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
```

Apply the same `derivatives/fmriprep/` prefix to `test_get_file_bold_denoised_smoothed`, `test_get_file_confounds`, and any mask/boldref/T1w/ROI expectations. Do the equivalent edits in `tests/test_data_sherlock.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dartbrains-tools && uv run pytest tests/test_data_paranoia.py tests/test_data_sherlock.py -q`
Expected: FAIL — captured path still `fmriprep/...`, expected now `derivatives/fmriprep/...`.

- [ ] **Step 3: Update the loaders**

In `src/dartbrains_tools/data/paranoia.py`, change:

```python
    func = f"fmriprep/{sub}/func"
    anat = f"fmriprep/{sub}/anat"
```
to:
```python
    func = f"derivatives/fmriprep/{sub}/func"
    anat = f"derivatives/fmriprep/{sub}/anat"
```

And in `load_roi_timeseries`:
```python
    filename = f"derivatives/fmriprep/{subject}/func/{subject}_run-{run}_nodeTimeSeries.csv"
```

In `src/dartbrains_tools/data/sherlock.py`, apply the same `derivatives/fmriprep/` prefix to `func`, `anat`, and the `load_roi_timeseries` filename (`{subject}_Part{part}_Average_ROI_n50.csv`). Leave `onsets/`, `stimuli/` paths unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dartbrains-tools && uv run pytest tests/test_data_paranoia.py tests/test_data_sherlock.py -q`
Expected: PASS

- [ ] **Step 5: Full suite + commit**

```bash
cd dartbrains-tools
uv run pytest -q
git add src/dartbrains_tools/data/sherlock.py src/dartbrains_tools/data/paranoia.py tests/test_data_sherlock.py tests/test_data_paranoia.py
git commit -m "fix(data): point sherlock/paranoia loaders at derivatives/fmriprep"
```

---

### Task 8: Version bump + release

**Files:**
- Modify: `dartbrains-tools/pyproject.toml`

- [ ] **Step 1: Bump the version**

In `pyproject.toml`, change `version = "0.1.6"` to `version = "0.1.7"`.

- [ ] **Step 2: Commit + tag**

```bash
cd dartbrains-tools
git add pyproject.toml
git commit -m "chore: release 0.1.7 (derivatives/fmriprep loader paths)"
```

(Publishing to PyPI follows the repo's existing release process.)

---

### Task 9: Update the Download_Data notebook betas example

**Files:**
- Modify: `dartbrains/content/Download_Data.py` (the `load_dataset` example cell, ~line 235)

**Interfaces:**
- Consumes: the new `betas` path-index config on `dartbrains/localizer`.

- [ ] **Step 1: Update the example cell body**

Replace the betas `load_dataset` example so it reflects the path-index shape (no `["nifti"]` auto-decode). The cell should read a row and load the volume explicitly:

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import nibabel as nib

ds = load_dataset("dartbrains/localizer", "betas")
_first = ds["train"][0]                      # {'path', 'subject', 'condition', 'type'}
_img = nib.load(hf_hub_download("dartbrains/localizer", _first["path"], repo_type="dataset"))
print(f"{_first['subject']} {_first['condition']}  ->  {_img.shape}")
```

Keep any surrounding `mo.md` prose consistent (describe it as an *index* of beta maps, loaded on demand). Follow CLAUDE.md marimo rules: this cell executes (mode:cached notebook) and its last expression is a `print`, which is fine for a non-figure cell.

- [ ] **Step 2: Verify the cell runs**

Run: `cd dartbrains && uv run marimo export ipynb content/Download_Data.py --include-outputs -o /tmp/dd_check.ipynb 2>&1 | tail -5`
Expected: export completes with no error from the betas cell (it downloads one beta file and prints shape).

- [ ] **Step 3: Commit**

```bash
cd dartbrains
git add content/Download_Data.py
git commit -m "docs: update betas load_dataset example to path-index shape"
```

---

## Self-Review

**Spec coverage:**
- Uniform path-index data model → Tasks 1-3 (labels, index, specs), verified Task 6.
- `participants` as content (emitted as `participants.csv`) → specs Task 3 + CLI `_tsv_to_csv` Task 4.
- Layout normalization `fmriprep → derivatives/fmriprep` + `.datalad` delete + OpenNeuro note → Task 5.
- Generator CLI in `scripts/`, `list_repo_files` discovery, shared BIDS parser + per-dataset globs, publish-to-branch → Tasks 1-4, 6.
- Custom label extractors (betas, onsets) → Task 1.
- `--check` verification → Task 4 (`cmd_check`), run in Task 6.
- Coordinated loaders + notebook updates → Tasks 7, 9.
- Config taxonomy table → asserted in Task 3 tests.
- Out-of-scope items (no niftifolder, onsets/stimuli stay top-level, localizer raw untouched) → respected throughout.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step states expected output.

**Type consistency:** `build_index(files, config)`, `rows_to_csv(rows)`, `render_readme_configs(dataset)`, `glob_to_regex(glob)`, `parse_bids_entities(path)`, `extract_beta_labels(path)`, `extract_onset_kind(path)`, `DATASETS`, `hub.list_files/read_text/upload_files` — names and signatures match across Tasks 1-4 and the CLI.

**Deviation from spec (noted):** spec said `participants` config points "straight at participants.tsv" with the csv builder + `sep="\t"`. HF's frontmatter `configs:` block can't pass a `sep` kwarg per config, so the generator instead emits a comma-separated `participants.csv` from the tab-separated source. Same content, uniformly comma-CSV, default builder — strictly simpler. All other decisions unchanged.
