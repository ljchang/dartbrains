# DartBrains v2: Marimo Migration & Site Redesign -- Design Spec

## Context

DartBrains (dartbrains.org) is an open-access neuroimaging course built with Jupyter Book 1, containing ~30 Jupyter notebooks (.ipynb) covering Python fundamentals through advanced fMRI analysis. The project needs a major update to:

1. Modernize the notebook authoring format (Jupyter → marimo)
2. Add interactive visualizations (anywidget + Three.js/Plotly)
3. Upgrade the site framework (Jupyter Book 1 → Jupyter Book 2)
4. Incorporate new MR Physics content already drafted in Dropbox

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Authoring format | Marimo (.py notebooks) | Reactive execution, cleaner diffs, better version control, native anywidget support |
| Site framework | Jupyter Book 2 (MyST-MD engine) | Natural evolution from JB1, built-in TOC/nav/search/theming, migration tool available |
| Interactive access | molab (molab.marimo.io) | Real Python backend, supports nibabel/nilearn/4GB+ fMRI data, free, GitHub integration |
| Static rendering | `marimo export ipynb --include-outputs` → JB2 build | Readers can browse without accounts; plots/figures render as static snapshots |
| Data hosting | HuggingFace (Localizer dataset already uploaded) | Large fMRI datasets downloaded on-demand in notebooks, not stored in repo |
| Package management | uv + pyproject.toml | Modern, fast, reproducible dependency resolution (already set up in Dropbox draft) |
| Python version | 3.13 | Latest stable (already configured in Dropbox draft) |

## Architecture

### Repository Structure

```
dartbrains/
├── content/                        # marimo .py notebooks (source of truth)
│   ├── intro.md                    # landing page (MyST markdown)
│   ├── MR_Physics_1_Magnetism_and_Resonance.py
│   ├── MR_Physics_2_Signal_and_Contrast.py
│   ├── MR_Physics_3_Imaging_and_fMRI.py
│   ├── Preprocessing.py
│   ├── Signal_Processing.py
│   ├── Introduction_to_Programming.py
│   ├── Introduction_to_Pandas.py
│   ├── Introduction_to_Plotting.py
│   ├── Introduction_to_Neuroimaging_Data.py
│   ├── GLM.py
│   ├── GLM_Single_Subject_Model.py
│   ├── Group_Analysis.py
│   ├── Thresholding_Group_Analyses.py
│   ├── Connectivity.py
│   ├── Introduction_to_ICA.py
│   ├── Multivariate_Prediction.py
│   ├── RSA.py
│   ├── Parcellations.py
│   ├── Resampling_Statistics.py
│   ├── Syllabus.md
│   ├── Schedule.md
│   ├── Instructors.md
│   ├── Contributing.md
│   └── Glossary.md                 # or .py if interactive
├── Code/                           # shared Python modules
│   ├── mr_simulations.py           # Bloch equation solver, tissue constants, Plotly helpers
│   ├── mr_widgets.py               # anywidget wrapper classes
│   └── js/                         # Three.js/Canvas 2D widget implementations
│       ├── precession_widget.js
│       ├── spin_ensemble_widget.js
│       ├── kspace_widget.js
│       ├── compass_widget.js
│       ├── convolution_widget.js
│       ├── encoding_widget.js
│       ├── transform_cube_widget.js
│       ├── smoothing_widget.js
│       ├── net_magnetization_widget.js
│       └── cost_function_widget.js
├── images/                         # static image assets
├── data/                           # small data files or download scripts
├── _build/                         # generated site output (gitignored)
├── myst.yml                        # Jupyter Book 2 config (replaces _config.yml + _toc.yml)
├── pyproject.toml                  # Python dependencies (uv)
├── uv.lock                         # dependency lock file
├── .python-version                 # Python 3.13
├── .github/workflows/
│   └── deploy-book.yml             # CI: export marimo → ipynb → JB2 build → deploy
├── scripts/
│   └── export_notebooks.py         # batch export marimo .py → .ipynb with outputs
└── README.md
```

### Two-Layer Experience

**Layer 1: Static site (dartbrains.org)** — the primary reading experience
- Built with Jupyter Book 2 from exported .ipynb files
- Fully rendered outputs: plots, figures, tables, text
- Navigation, table of contents, search, theming all handled by JB2
- No account required, fast loading
- Each notebook page includes an "Open in molab" button

**Layer 2: Interactive notebooks (molab)** — opt-in for hands-on learners
- Linked from the static site via `molab.marimo.io/github/ljchang/dartbrains/blob/v2-marimo-migration/content/{notebook}.py`
- Full marimo reactivity: sliders, anywidgets, editable code
- Real Python backend: nibabel, nilearn, scipy all available
- fMRI datasets download from HuggingFace on demand

### Build Pipeline

```
marimo .py files (source of truth)
        │
        ▼
scripts/export_notebooks.py
  marimo export ipynb <file>.py -o <file>.ipynb --include-outputs
        │
        ▼
Jupyter Book 2 build
  jupyter book build --site
        │
        ▼
GitHub Pages deployment
  .github/workflows/deploy-book.yml
```

### Data Strategy

- Large fMRI datasets (Localizer, etc.) hosted on HuggingFace
- Notebooks download data on demand via `huggingface_hub` or `datasets` library
- Small supporting files (CSV, masks) can live in `data/`
- No large binary data in the git repo

## Content Migration Plan

### New content (from Dropbox draft — copy directly)
- MR_Physics_1_Magnetism_and_Resonance.py (complete draft)
- MR_Physics_2_Signal_and_Contrast.py (complete draft)
- MR_Physics_3_Imaging_and_fMRI.py (complete draft)
- Preprocessing.py (complete draft with widgets)
- Code/mr_simulations.py
- Code/mr_widgets.py
- Code/js/*.js (10 widget files)

### Existing content (convert from .ipynb → marimo .py)
- Introduction_to_Programming.ipynb
- Introduction_to_Pandas.ipynb
- Introduction_to_Plotting.ipynb
- Introduction_to_Neuroimaging_Data.ipynb
- Signal_Processing.ipynb
- GLM.ipynb
- GLM_Single_Subject_Model.ipynb
- Group_Analysis.ipynb
- Thresholding_Group_Analyses.ipynb
- Connectivity.ipynb
- Introduction_to_ICA.ipynb
- ICA.ipynb (advanced)
- Multivariate_Prediction.ipynb
- RSA.ipynb
- Parcellations.ipynb
- Resampling_Statistics.ipynb
- Download_Data.ipynb
- Glossary.ipynb

### Static content (keep as markdown, update as needed)
- intro.md
- Intro_to_Neuroimaging.md
- Syllabus.md
- Schedule.md
- Instructors.md
- Contributing.md
- fmriprep_on_discovery.md

### Content to retire or replace
- Signal_Measurement.ipynb → replaced by MR Physics 1-3 series
- Introduction_to_JupyterHub.ipynb → update for molab/marimo workflow
- Introduction_to_Discovery.ipynb → evaluate if still relevant
- Project gallery pages (2019-2022) → archive or carry forward

## Site Configuration (myst.yml)

The JB2 migration tool will generate this from existing _config.yml + _toc.yml. Key settings to preserve:

- Site title, author, logo, favicon
- Google Analytics (UA-138270939-1)
- Repository links (edit page, issues)
- Hypothesis comments
- Launch buttons → replace with "Open in molab" buttons
- CNAME: dartbrains.org
- TOC structure updated for new content organization

## Table of Contents (Draft)

```
- intro.md
- Part: Course Overview
    - Instructors.md
    - Syllabus.md
    - Schedule.md
- Part: Getting Started
    - Introduction_to_Programming.py
    - Introduction_to_Pandas.py
    - Introduction_to_Plotting.py
    - Download_Data.py
    - Glossary.md
- Part: MR Physics & Imaging (NEW)
    - MR_Physics_1_Magnetism_and_Resonance.py
    - MR_Physics_2_Signal_and_Contrast.py
    - MR_Physics_3_Imaging_and_fMRI.py
- Part: Neuroimaging Analysis
    - Intro_to_Neuroimaging.md
    - Introduction_to_Neuroimaging_Data.py
    - Signal_Processing.py
    - Preprocessing.py
    - GLM.py
    - GLM_Single_Subject_Model.py
    - Group_Analysis.py
    - Thresholding_Group_Analyses.py
- Part: Advanced Methods
    - Connectivity.py
    - Introduction_to_ICA.py
    - Multivariate_Prediction.py
    - RSA.py
    - Parcellations.py
    - Resampling_Statistics.py
- Part: Additional Resources
    - fmriprep_on_discovery.md
    - url: http://naturalistic-data.org/ (Naturalistic Data Analysis)
- Part: Contributing
    - Contributing.md
    - url: https://github.com/ljchang/dartbrains
```

## Dependencies

### Build dependencies (CI)
- marimo (>=0.21.1)
- nbformat
- jupyter-book (v2)

### Notebook runtime dependencies (pyproject.toml)
- marimo>=0.21.1
- anywidget>=0.9.21
- numpy>=2.4.3
- scipy>=1.17.1
- plotly>=6.6.0
- matplotlib>=3.10.8
- pandas
- seaborn
- nibabel>=5.4.2
- nilearn
- nltools
- scikit-learn
- networkx
- huggingface-hub
- datasets
- datalad>=1.3.4

## Verification Plan

1. **Feasibility test**: Load a full fMRI dataset (~4GB) on molab — confirm RAM is sufficient
2. **Export pipeline test**: Verify `marimo export ipynb --include-outputs` produces valid .ipynb with rendered plots and static widget snapshots for each notebook type (pure code, plotly, anywidget)
3. **JB2 build test**: Confirm exported .ipynb files render correctly in Jupyter Book 2
4. **Molab GitHub integration test**: Verify notebooks open correctly via `molab.marimo.io/github/...` URLs
5. **Cross-reference test**: Confirm links between notebooks and to external resources work in both static site and molab
6. **Content review**: Walk through each migrated notebook to ensure no content was lost or broken in conversion
7. **Widget rendering**: Verify anywidgets appear as meaningful static snapshots on the site (not blank boxes)
8. **CI pipeline**: End-to-end test of the GitHub Actions workflow: push → export → build → deploy
