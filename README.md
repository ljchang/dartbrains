[![DOI](https://zenodo.org/badge/171529794.svg#left)](https://zenodo.org/badge/latestdoi/171529794)
# DartBrains

An open-access introduction to functional neuroimaging analysis in Python.

**Website:** [dartbrains.org](https://dartbrains.org)

## About

DartBrains provides interactive tutorials covering fMRI data analysis, from MR physics fundamentals through advanced analysis techniques. All content is authored as [marimo](https://marimo.io/) notebooks with interactive visualizations.

## Reading the tutorials

**Browse online:** Visit [dartbrains.org](https://dartbrains.org) to read all tutorials with rendered outputs.

**Run interactively:** Each tutorial page has an "Open in molab" link to launch the notebook with live code, interactive widgets, and editable cells.

## Local development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
# Clone the repo
git clone https://github.com/ljchang/dartbrains.git
cd dartbrains

# Install dependencies
uv sync

# Edit a notebook
uv run marimo edit content/MR_Physics_1_Magnetism_and_Resonance.py

# Build the static site
uv run python scripts/export_notebooks.py
uv run jupyter book build --site
```

## License

All content is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Acknowledgements

Created by [Luke Chang](http://www.lukejchang.com/) and supported by NSF CAREER Award 1848370 and the [Dartmouth Center for the Advancement of Learning](https://dcal.dartmouth.edu/about/impact/experiential-learning).
