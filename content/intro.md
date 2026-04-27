[![DOI](https://zenodo.org/badge/171529794.svg)](https://zenodo.org/records/171529794)
# Course Overview

How can we understand how the brain works? This course provides an introduction to in vivo neuroimaging in humans using functional magnetic resonance imaging (fMRI). The goal of the class is to introduce: (1) how the scanner generates data, (2) how psychological states can be probed in the scanner, and (3) how this data can be processed and analyzed. Students will be expected to analyze brain imaging data using the opensource Python programming language. We will be using several packages such as numpy, matplotlib, nibabel, nilearn, fmriprep, and nltools. This course will be useful for students working in neuroimaging labs, completing a neuroimaging thesis, or interested in pursuing graduate training in fields related to cognitive neuroscience.

## Goals

1) Learn the basics of fMRI signals
2) Introduce standard data preprocessing techniques
3) Introduce the general linear model
4) Introduce advanced analysis techniques

## Overview
This course is designed primarily around learning the basics of fMRI data analysis using the Python programming language. We will cover a lot of ground from introducing the Python programming language, to signal processing, to working with opensource packages from the Python Scientific Computing community. 

## Version 2.0
The original version of Dartbrains v1.0 was created in 2019 as an attempt to create a flipped classroom experience for teaching an undergraduate laboratory course on fMRI. These materials were expanded in 2020 to better accommodate remote teaching during the global pandemic. In 2026, we decided to update the materials to reflect changes to the Python ecosystem that have dramatically improved performance and user experiences. Rust has enabled improved performance of many Python packages. For example, [uv](https://docs.astral.sh/uv/) has provided a fresh perspective on python package management and allowed us to finally part ways from [Anaconda](https://www.anaconda.com/). [Polars](https://pola.rs/) has emerged as a fast alternative to Pandas. Our data hosting became antiquated and we migrated our data to [huggingface](https://huggingface.co/dartbrains).[Marimo notebooks](https://marimo.io/) have dramatically improved upon jupyter notebooks with renewed minimalism, elegant design, and the incorporation of reactivity and DAGs. We were early adopters and big fans of [jupyter-book](https://jupyterbook.org/), which we used to build v1.0 of DartBrains. However, switching to marimo notebooks left us wanting a combination of the static page experience of jupyter-book enhanced with the reactivity potential of marimo notebooks. This led us to create [marimo-book](https://marimobook.org/) to accomodate our vision for DartBrains v2.0. 

## Questions
Please post any questions that arise from the material in this course on our [Discourse Page](https://www.askpbs.org/c/dartbrains)

## License for this book
All content is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0) license.

## Acknowledgements
Dartbrains was created by [Luke Chang](http://www.lukejchang.com/) and supported by an NSF CAREER Award 1848370, a grant from the NIMH R01MH116026, and an award from the [Dartmouth Center for the Advancement of Learning](https://dcal.dartmouth.edu/about/impact/experiential-learning). We were the first course at Dartmouth to incorporate hosted jupyter notebooks into a classroom experience. Our original [jupyterhub server](https://jhub.dartmouth.edu/hub/home?ticket=ST-124603-DFlvO4E7nc4b-FeVN5Ml4KO-M5o-localhost) was built and maintained by the [Research Computing staff at Dartmouth](https://rc.dartmouth.edu/). Special thanks to Arnold Song, William Hamblen, Christian Darabos, and John Hudson.