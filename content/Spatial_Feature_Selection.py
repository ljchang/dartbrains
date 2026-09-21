# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dartbrains-tools>=0.3.0",
#     "matplotlib",
#     "nltools==0.6.0.dev2",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "seaborn",
# ]
# 
# [tool.grader]
# server = "https://grader.dartbrains.org"
# course = "neuroimaging"
# term = "2026-fall"
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from dartbrains_tools.data import localizer
    from nltools.data import BrainData
    from nltools.mask import expand_mask
    from nltools.templates import fetch_resource
    from sklearn.model_selection import LeaveOneGroupOut


    return BrainData, fetch_resource, localizer, mo, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spatial Feature Selection

    *Written by Luke Chang*

    In the previous tutorial we trained a classifier on the whole brain and asked
    whether it could tell left-hand from right-hand movements. It could. But that
    result leaves a more interesting question untouched: **where in the brain is
    that information?**

    Answering it means choosing a *spatial scale* to analyze at, and that choice is
    not a technicality — it changes what question you are asking. Multivariate
    analyses live on a spectrum:

    - **Whole brain** — one model over every voxel. Maximally sensitive to
      information spread thinly across many regions, but it tells you little about
      location.
    - **Region of interest (ROI)** — one model per parcel. Localizes the effect, but
      inherits whatever assumptions the atlas was built with.
    - **Searchlight** — one model per small sphere, centered on every voxel in turn.
      Localizes without committing to an atlas, at a real computational cost and
      with some interpretational traps.

    A natural instinct is to run all three and keep whichever gives the highest
    accuracy. **Resist it.** As Jolly & Chang (2021) argue, the goal is not to
    optimize for the "best" spatial scale — it is to match the scale of your
    *analysis* to the scale of the *representation* you think you are studying.
    Comparing scales as though they were competing models is not even well defined,
    since they are answering different questions.

    In this tutorial we will:

    - Run the same classification at all three spatial scales on the same data
    - See why cross-validation must respect subject boundaries for the numbers to
      mean anything
    - Learn why a searchlight map cannot give you voxel-level feature importance
    - Develop intuition for which scale suits which kind of representation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The data

    We will reuse the localizer dataset from the previous tutorial: 20 participants,
    each contributing one beta map for a **video-cued left-hand** movement and one
    for a **video-cued right-hand** movement. The classification problem is the same
    one as before — left vs right — which means any difference we see across the
    three analyses comes from the spatial scale and nothing else.
    """)
    return


@app.cell
def _(BrainData, localizer, np):
    _sub_list = localizer.get_subjects()
    left_files = [localizer.get_file(sub, "betas", "video_left_hand") for sub in _sub_list]
    right_files = [localizer.get_file(sub, "betas", "video_right_hand") for sub in _sub_list]

    data = BrainData(left_files).append(BrainData(right_files))

    # left = 1, right = 0, matching the previous tutorial.
    Y = np.hstack([np.ones(len(left_files)), np.zeros(len(right_files))])

    # Each participant contributes one left and one right image, so subject labels
    # repeat across the two halves of the stack.
    subject_id = np.hstack([_sub_list, _sub_list])

    f"{len(_sub_list)} subjects -> {data.shape[0]} images x {data.shape[1]} voxels"
    return Y, data, subject_id


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First, a word about cross-validation

    Before comparing anything, we have to be able to trust the numbers.

    Each participant contributes **two** images to this dataset — one left, one
    right. If we split the data into folds at random, a participant's left-hand map
    can land in the training set while their right-hand map lands in the test set.
    The classifier then gets to see that person's brain during training and is
    scored on that same brain. Whatever it learned about *them* — their anatomy,
    their alignment, their noise — is not knowledge that generalizes to a new
    participant, but it still shows up in the score.

    The fix is to cross-validate over **participants**, not images: every image
    belonging to a held-out participant goes in the test set together. In `nltools`
    you do that by passing a grouping variable:

    ```python
    data.predict(y=Y, groups=subject_id, cv=5)
    ```

    Supplying `groups=` switches the fold assignment to a group-aware scheme, so a
    participant is never split across the train/test boundary.

    This matters more than it might seem for what follows. The three analyses below
    are only comparable to each other if they all use the *same* cross-validation
    scheme — otherwise differences between them reflect how the folds were drawn
    rather than anything about spatial scale.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scale 1: Whole brain

    The whole-brain model treats every voxel as a feature in a single classifier.
    Nothing is excluded in advance, so if the discriminating signal is spread thinly
    across many regions the model can still pick it up by combining weak evidence
    from all of them.

    This is the scale at which whole-brain "decoders" or "biomarkers" are built. It
    supports a strong claim about **generalization** — the model either predicts a
    new participant or it doesn't — but only a diffuse claim about **location**.
    """)
    return


@app.cell
def _(Y, data, subject_id):
    whole_brain = data.predict(
        y=Y, spatial_scale="whole_brain", groups=subject_id, cv=5
    )

    f"Whole-brain accuracy: {whole_brain.mean_score:.3f}  (chance = 0.5)"
    return (whole_brain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A single number, and a good one. But notice what it does *not* tell us: which
    part of the brain carried the signal.

    It is tempting to read the model's weight map as an answer to that. Be careful —
    a large weight does not mean a voxel represents the information. Jolly & Chang
    (2021) put it directly:

    > reliable weight maps do not indicate that a voxel explicitly represents
    > psychological information but that in concert with other voxels it can
    > effectively predict an outcome

    A voxel can earn a large weight simply by *denoising* its neighbors — by
    carrying correlated noise the model subtracts away. That is a useful thing for a
    predictor to do and a misleading thing to interpret anatomically.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scale 2: Regions of interest

    Instead of one model over all 238,955 voxels, we can fit **one model per
    region** and ask which regions support the classification. That requires a
    parcellation — here the k=50 whole-brain atlas you met in the Parcellations
    tutorial.

    This buys us localization. The price is that the answer is now conditional on
    the atlas: if a functional region is split across two parcels, or blurred
    together with its neighbor, no model can recover it. The parcellation encodes an
    assumption about where the boundaries are, and the results inherit it.
    """)
    return


@app.cell
def _(Y, data, fetch_resource, np, subject_id):
    # Pass the atlas as a file path (or Nifti). nltools resamples it into the
    # data's mask space itself, so wrapping it in BrainData first is unnecessary.
    atlas = fetch_resource("masks/k50_2mm.nii.gz")

    roi = data.predict(
        y=Y, spatial_scale="roi", roi_mask=atlas,
        groups=subject_id, cv=5, n_jobs=-1,
    )

    f"{len(roi.roi_labels)} parcels | best {np.nanmax(roi.mean_score):.3f} | median {np.nanmedian(roi.mean_score):.3f}"
    return (roi,)


@app.cell
def _(roi):
    roi_above_chance = roi.score_map - 0.5

    roi_above_chance.plot(
        method="glass",
        cmap="RdBu_r",
        title="ROI decoding accuracy above chance (k=50)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two things are worth noticing.

    **The best parcel beats the whole-brain model.** Restricting the classifier to
    one well-chosen region did better than giving it every voxel in the brain. That
    is not a paradox — most voxels carry no signal for this contrast, and feeding
    them to the model adds noise and dimensionality without adding evidence.

    **Most parcels are far from the best one.** The median parcel sits well below
    the top parcel. The information is not distributed evenly; it is concentrated
    in a handful of regions, and the map shows where.

    Note also that we asked for 50 parcels and got 46. Parcels that fall entirely
    outside this dataset's brain mask have no voxels to model, so they drop out.
    Anything the atlas does not cover is simply invisible to this analysis — which
    is the atlas assumption made concrete.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scale 3: Searchlight

    The ROI analysis needed an atlas. A **searchlight** removes that requirement:
    center a small sphere on a voxel, fit a model using only the voxels inside the
    sphere, record the accuracy at the center, then slide the sphere to the next
    voxel and repeat. The result is an accuracy map with the same resolution as the
    data, built without committing to any parcellation.

    The catch is cost. A whole-brain searchlight fits *one model per voxel* — here
    that would be 238,955 cross-validated models, about **11 minutes**. That is
    affordable but not free, and it grows quickly with the number of participants
    and folds.

    To keep this tutorial responsive we will restrict the searchlight to
    sensorimotor cortex (pre- and postcentral gyrus). **Be aware of what that
    costs us conceptually:** we just used an anatomical prior to decide where to
    look, which is exactly the assumption a searchlight was supposed to let us
    avoid. For a real analysis you would run the whole brain. We are trading the
    method's main selling point for a faster tutorial, and it is worth being honest
    about that rather than letting it pass unnoticed.
    """)
    return


@app.cell
def _(BrainData, data, fetch_resource, pd):
    labels = pd.read_csv(fetch_resource("atlases/labels_harvard_oxford.csv"))
    sensorimotor_rows = labels[
        labels["name"].str.contains("central_Gyrus", case=False, na=False)
    ]

    # Harvard-Oxford is a 4D *probabilistic* atlas: one volume per region, values
    # 0-100. Take every voxel where any of these regions exceeds 25% probability.
    ho = BrainData(fetch_resource("atlases/atlas_harvard_oxford.nii.gz"), mask=data.mask)
    sensorimotor = BrainData(
        (ho[list(sensorimotor_rows["index"])].data.max(axis=0) > 25).astype(float),
        mask=data.mask,
    )

    data_sm = data.apply_mask(sensorimotor)

    f"{list(sensorimotor_rows['name'])} -> {data_sm.shape[1]:,} voxels"
    return (data_sm,)


@app.cell
def _(Y, data_sm, np, subject_id):
    searchlight = data_sm.predict(
        y=Y, spatial_scale="searchlight", radius=10,
        groups=subject_id, cv=5, n_jobs=-1,
    )

    _acc = searchlight.score_map.data
    f"peak {np.nanmax(_acc):.3f} | median {np.nanmedian(_acc):.3f} | above 0.7: {np.nanmean(_acc > 0.7):.1%} of spheres"
    return (searchlight,)


@app.cell
def _(searchlight):
    sl_above_chance = searchlight.score_map - 0.5

    sl_above_chance.plot(
        method="glass",
        cmap="RdBu_r",
        title="Searchlight accuracy above chance (10mm, sensorimotor cortex)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What a searchlight map can and cannot tell you

    Searchlight maps are seductive. They look like activation maps, so it is easy to
    read them the same way. Three cautions, following Jolly & Chang (2021).

    **1. The blobs are smeared, by construction.** Neighboring spheres overlap
    heavily, so a single highly informative voxel raises the accuracy of every
    sphere that contains it — a whole 10mm neighborhood lights up around it. As the
    paper puts it, accuracy scores are *"smeared over spatial extents because
    searchlights are overlapping"*, which makes it *"nearly impossible to identify
    which voxels are most important for prediction."* The peak of a searchlight blob
    is not necessarily where the information is.

    **2. You cannot get voxel-level feature importance.** Because each voxel belongs
    to many overlapping spheres, *"each voxel has a different feature weight
    depending upon the particular searchlight it belongs to"* — so there is no
    single well-defined weight per voxel, and traditional importance testing is
    infeasible. This is not a missing feature in the software; it is a property of
    the method. Notice that `nltools` returns a `weight_map` for the ROI analysis,
    where each voxel belongs to exactly one parcel, but not for the searchlight.
    The API asymmetry is the conceptual point made concrete.

    **3. Every sphere is a hypothesis test.** We just computed thousands of
    accuracies. Some will look impressive by chance alone, exactly as in the
    mass-univariate case — so a searchlight map needs the same multiple-comparisons
    machinery you met in the Thresholding tutorial. An uncorrected searchlight map
    is not a result.

    One more caveat specific to how searchlights are usually run. They are most
    often computed **within each participant** and the resulting accuracies
    aggregated across the group. Under that design, rejecting the null *"only
    suggests that some individuals demonstrate an effect not that the effect is
    typical."* Our searchlight here is cross-validated **across** participants, so
    it makes a group-level generalization claim instead — a different, and in some
    ways stronger, kind of statement. It is worth knowing which of the two any given
    paper did.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Putting the three together

    Same participants, same contrast, same classifier, same cross-validation. Only
    the spatial scale changed.
    """)
    return


@app.cell
def _(np, pd, plt, roi, searchlight, sns, whole_brain):
    _sl_acc = searchlight.score_map.data
    _sl_acc = _sl_acc[np.isfinite(_sl_acc)]
    _roi_acc = roi.mean_score[np.isfinite(roi.mean_score)]

    comparison = pd.DataFrame(
        {
            "accuracy": np.concatenate([_roi_acc, _sl_acc]),
            "scale": ["ROI (46 parcels)"] * _roi_acc.size
            + ["Searchlight (spheres)"] * _sl_acc.size,
        }
    )

    _order = ["ROI (46 parcels)", "Searchlight (spheres)"]
    _fig, _ax = plt.subplots(figsize=(9, 3.5))
    sns.boxplot(
        data=comparison, x="accuracy", y="scale", order=_order, ax=_ax,
        whis=(5, 95), fliersize=0, color="lightsteelblue",
    )

    # The peak matters as much as the middle here, and a boxplot hides it.
    for _i, _s in enumerate(_order):
        _mx = comparison.loc[comparison["scale"] == _s, "accuracy"].max()
        _ax.scatter([_mx], [_i], color="darkorange", s=70, marker="D", zorder=5)
        _ax.annotate(
            f"best {_mx:.2f}", (_mx, _i), color="darkorange",
            textcoords="offset points", xytext=(0, 14), ha="center",
        )

    _ax.axvline(0.5, color="grey", linestyle=":", label="chance")
    _ax.axvline(
        whole_brain.mean_score, color="crimson", linestyle="--",
        label=f"whole brain ({whole_brain.mean_score:.2f})",
    )
    _ax.set_xlim(0.35, 1.0)
    _ax.set_xlabel("Classification accuracy")
    _ax.set_ylabel("")
    _ax.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The figure makes the real lesson visible, and it is **not** "smaller scales win."

    The whole-brain model (red line) lands in the middle — above the median of both
    local analyses, but below the best of either. Meanwhile the bulk of parcels and
    the bulk of spheres sit well below those peaks.

    Read together, those facts say something the whole-brain number alone could not:
    **the information is focal, not diffuse.** A handful of sensorimotor regions
    carry nearly all of it, and the whole-brain model reaches its score by pooling that
    concentrated signal with thousands of uninformative voxels — doing well despite
    the noise rather than because the signal is everywhere.

    Two cautions about reading the figure too literally:

    - The searchlight distribution only covers **sensorimotor cortex**, because we
      restricted it. Its median is therefore much higher than a whole-brain
      searchlight's would be. The ROI and searchlight medians are not measuring the
      same thing.
    - Comparing peaks across scales is not a fair contest either. A searchlight
      reports the best of thousands of overlapping spheres, so its maximum is
      selected from far more opportunities than the best of 46 parcels.

    This is why Jolly & Chang (2021) explicitly **caution against treating the
    choice as an optimization**:

    > we caution researchers against framing the issue as attempting to "optimize"
    > for the "best" spatial scale

    Formal model comparison between these analyses is, in their words, *"not trivial
    or even feasible."* They are not competing estimators of one quantity — they are
    different questions.

    ## So which scale should you use?

    Match the analysis to the kind of representation you expect:

    | If you think the representation is... | Reach for |
    |---|---|
    | fine-grained and locally organized (perceptual, retinotopic, somatotopic) | **searchlight** |
    | organized around known anatomy or function, or you have a strong prior | **ROI** |
    | abstract and distributed (emotion, social cognition, pain) | **whole brain** |

    Our example is a **motor** contrast — about as locally organized as fMRI effects
    get — so it is unsurprising that the local scales found sharp, strong peaks. Do
    not generalize from it to "local beats global." For a more abstract target the
    picture typically reverses: whole-brain models have *"been more successful than
    searchlights in identifying sensitive and specific predictive models of more
    abstract psychological processes."*

    The honest answer to "what spatial scale should I use?" is a question back:
    *what spatial scale do you think the brain is using?*

    ## Recap

    - Spatial scale is a modeling choice that changes the question, not just the
      resolution of the answer
    - Cross-validation must respect participant boundaries or the scores are leaked
    - ROI analyses localize but inherit the atlas's assumptions
    - Searchlight maps are spatially smeared, cannot give voxel-level importance,
      and need multiple-comparison correction like any other map
    - Choose the scale that matches the representation, and resist ranking scales
      against each other

    ## Further reading

    - Jolly, E., & Chang, L. J. (2021). Multivariate spatial feature selection in
      fMRI. *Social Cognitive and Affective Neuroscience*, 16(8), 795-806.
      [doi:10.1093/scan/nsab010](https://doi.org/10.1093/scan/nsab010)
    - The **Parcellations** tutorial, for where atlases come from
    - The **Thresholding Group Analyses** tutorial, for correcting the maps produced here
    - The **RSA** tutorial, which asks the same "at what scale?" question of
      representational geometry rather than decoding accuracy
    """)
    return


if __name__ == "__main__":
    app.run()
