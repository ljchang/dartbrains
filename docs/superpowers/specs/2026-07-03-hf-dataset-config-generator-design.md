# HF Dataset Config Generator — Design

**Date:** 2026-07-03
**Status:** Approved (design)
**Repos affected:** `dartbrains/localizer`, `dartbrains/sherlock`, `dartbrains/paranoia` (HuggingFace datasets); `dartbrains-tools` (loaders + generator CLI); `dartbrains` (notebooks referencing paths).

## Problem

The three HuggingFace datasets expose `load_dataset(...)` configs inconsistently. `localizer` gained a working `betas` config (via the `niftifolder` builder that auto-decodes volumes), but the other configs and the other two datasets are ad hoc or missing. We want **one consistent, maintainable way** to define and interact with dataset configs across all three, driven by a **generator tool** that exploits the BIDS commonality among them.

Two independent problems, solved together:

1. **Layout divergence** — `localizer` follows the BIDS convention (`derivatives/fmriprep/`); `sherlock` and `paranoia` dump fmriprep output at the top level (`fmriprep/`). This forces per-dataset special-casing everywhere.
2. **Config inconsistency** — no uniform interaction model or config-authoring mechanism across datasets.

## Ground truth (verified via `list_repo_files`, 2026-07-03)

| Repo | Raw BIDS? | fmriprep location | Other | Notes |
|---|---|---|---|---|
| `localizer` | Yes (`sub-S01`…`sub-S94`, `phenotype/`) | `derivatives/fmriprep/` ✅ | `derivatives/betas/` | Full BIDS dataset (3642 files) |
| `sherlock` | **No** | `fmriprep/` ❌ (top-level) | `onsets/`, `stimuli/`, `.datalad/` | Preprocessed-only (1341 files); raw on OpenNeuro |
| `paranoia` | **No** | `fmriprep/` ❌ (top-level) | `stimuli/`, `.datalad/` | Preprocessed-only (958 files); raw on OpenNeuro |

## Decision 1 — Data model: uniform path-index

**Every per-file config is a CSV index with one identical row shape: `{ path (repo-relative), + labels }`.** The consumer always does the same thing — filter the table, then load the files they want:

```python
df = load_dataset("dartbrains/localizer", "bold").to_pandas()
for p in df[df.subject == "S01"].path:
    img = nib.load(hf_hub_download("dartbrains/localizer", p, repo_type="dataset"))
# identical pattern for confounds/events, just pd.read_csv instead of nib.load
```

Rationale:

- **Uniform interaction** across nifti and tabular configs (filter → fetch → load). One mental model.
- **Lightweight** — indexing a multi-GB nifti repo downloads nothing; `load_dataset` returns only the index. Users fetch volumes on demand.
- **Simple + cheap generator** — its whole job is *parse labels from filenames → write a CSV*. Trivial to test (no large downloads); sidesteps the `niftifolder` requirement that `metadata.csv` sit adjacent to volumes (which collides with the fmriprep tree's per-subject metadata files — the original `betas` bug).
- **Clean separation of concerns** — `load_dataset` configs become the *index/manifest*; the `dartbrains_tools.data` helpers (`load_events`, `load_confounds`, …) remain the *content loaders* that return DataFrames. Tabular configs no longer inline content, and that's fine — the helpers already do that.

Trade-off accepted: `betas` stops auto-decoding into a `Nifti1Image`. The notebook never used the decoded array (only `subject`/`condition` labels), and `nib.load(hf_hub_download(repo, row["path"]))` is one line for anyone who wants it.

**One exception — `participants`** is not per-file; it's a single dataset-level demographics table. It stays *content*: its config `data_files` points straight at `participants.tsv` (the `csv` builder with `sep="\t"`). A 1-row path-index there would be silly.

## Decision 2 — Normalize layout first

Make **`derivatives/fmriprep/`** canonical on all three repos:

- **sherlock + paranoia:** move top-level `fmriprep/ → derivatives/fmriprep/`. The ROI / node-timeseries CSVs ride along (they live inside `func/`). Delete `.datalad/` cruft.
- **onsets/** (sherlock) and **stimuli/** (both): **leave at top level** — stimulus metadata / raw stimuli, not preprocessing derivatives.
- **localizer:** unchanged — already correct.
- **README** (sherlock/paranoia): note that raw BIDS data lives on OpenNeuro.

Result: `bold` / `confounds` / `mask` are the **same glob** (`derivatives/fmriprep/**`) across all three. localizer is already there, so this achieves 3-way consistency *without contorting localizer* — localizer simply carries extra configs (`events`, `betas`) the naturalistic datasets don't.

The move is cheap: nifti files are Git-LFS pointers, so a path rename rewrites small pointer files, not the multi-GB blobs — no re-upload, one commit per repo.

## Decision 3 — Generator architecture

**Location:** `dartbrains-tools/scripts/generate_hf_configs.py` — a maintenance CLI, **not** shipped in the installed package (ops/build tool, not a runtime dependency). Co-located with the loaders so path knowledge stays in one repo.

**Four cheap stages:**

1. **Discover** — `huggingface_hub.list_repo_files(repo, repo_type="dataset")` → full file listing, no downloads.
2. **Match & label** — a shared **BIDS-entity parser** fills `subject` (`sub-XX`), `task`, `run`, `space`, `desc` automatically. Each config supplies a **glob** to select files and optionally a **custom label extractor** for non-BIDS filenames.
3. **Write index** — one `<config>.csv` (`{path, labels…}`) per config; rewrite the README `configs:` YAML so each config's `data_files.path` points at its CSV.
4. **Publish** — upload CSVs + README on a **branch** for review (standard gate).

**Per-dataset spec** — one small literal per dataset; this is the *only* place divergence lives:

```python
LOCALIZER = {
    "repo": "dartbrains/localizer",
    "configs": {
        "bold":      {"glob": "derivatives/fmriprep/**/*_desc-preproc_bold.nii.gz"},
        "confounds": {"glob": "derivatives/fmriprep/**/*_desc-confounds_*.tsv"},
        "mask":      {"glob": "derivatives/fmriprep/**/*_desc-brain_mask.nii.gz"},
        "events":    {"glob": "sub-*/func/*_events.tsv"},
        "betas":     {"glob": "derivatives/betas/*.nii.gz", "labels": extract_beta_labels},
        "participants": {"content": "participants.tsv"},  # non-index, csv builder sep=\t
    },
}
SHERLOCK = {  # bold/confounds/mask identical globs; plus:
    "onsets": {"glob": "onsets/*.csv", "labels": extract_onset_kind},
}
PARANOIA = {  # bold/confounds/mask identical globs; plus participants (content)
}
```

**Custom label extractors** (the only non-BIDS naming):

- `extract_beta_labels`: `S01_beta_audio_computation.nii.gz` → `{subject: S01, condition: audio_computation, type: individual}`; `S01_betas.nii.gz` → `{subject: S01, type: stacked}`.
- `extract_onset_kind`: `Sherlock_Watch_Scene_N50_Onsets.csv` → `{kind: watch}` (also `recall`, `crop`).

## Config taxonomy (final)

| Config | localizer | sherlock | paranoia | Kind |
|---|:-:|:-:|:-:|---|
| `bold` | ✅ | ✅ | ✅ | path-index |
| `confounds` | ✅ | ✅ | ✅ | path-index |
| `mask` | ✅ | ✅ | ✅ | path-index |
| `events` | ✅ | — | — | path-index |
| `betas` | ✅ | — | — | path-index (custom labels) |
| `onsets` | — | ✅ | — | path-index (custom labels) |
| `participants` | ✅ | — | ✅ | content (csv, sep=\t) |

## Verification

The CLI's `--check` mode: after writing, for each config run `load_dataset(repo, config)` and assert (a) row count equals the discovered file count, and (b) expected label columns are present. Guards against a bad glob or a metadata-placement regression before publish.

## Coordinated change set (per repo, on a branch)

1. **HF repo:** `fmriprep/ → derivatives/fmriprep/` rename + `.datalad/` delete (sherlock/paranoia only); upload `<config>.csv` index files; rewrite README `configs:` block.
2. **`dartbrains-tools` loaders:** `sherlock.py` / `paranoia.py` `func = f"fmriprep/..."` → `derivatives/fmriprep/...` (and the ROI/node-timeseries paths). Bump version.
3. **`dartbrains` notebooks:** any `snapshot_download(allow_patterns=["fmriprep/..."])` → `derivatives/fmriprep/...`; update `Download_Data.py` `load_dataset` examples to the new configs.

All three land together so nothing breaks between the repo move and the loader/notebook updates.

## Out of scope (YAGNI)

- Auto-decoding nifti into `Nifti1Image` via `load_dataset` (path-index + `nib.load` replaces it).
- Moving `onsets/` / `stimuli/` under `derivatives/`.
- Restructuring localizer's raw BIDS tree.
- Any change to the `dartbrains_tools.data` helper API surface (only internal path strings change).
