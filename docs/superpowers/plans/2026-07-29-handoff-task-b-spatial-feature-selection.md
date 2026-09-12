# Handoff: Task B — spatial feature selection chapter

**Written**: 2026-07-29
**Updated**: 2026-08-01 — §1a blocker **fixed**, §1b remeasured (~130x wrong), §4 paper **read**
**Branch**: `nltools-0.6-migration` (dartbrains) — now **pushed** to origin, 14 commits ahead of `master`, tree clean
**Status**: Task A (thresholding) is **done and verified**. Task B is the last item from the original handoff.

Supersedes the Task-B sections of
[`2026-07-28-handoff-thresholding-and-spatial-scale.md`](./2026-07-28-handoff-thresholding-and-spatial-scale.md).
Read §1 before designing anything — **two of that document's load-bearing claims
about Task B are wrong**, and one of them was a hard blocker.

**Bottom line as of 2026-08-01: the chapter is unblocked — go write it.**

- All three scales work and all three are cheap: ROI ~5 s, restricted
  searchlight ~20–40 s, whole-brain searchlight 11 min. Both of §1's blocking
  claims are resolved and compute is no longer a design constraint.
- The paper is read (§4). **Framing:** match method to representation; do *not*
  stage a scale bake-off (the paper explicitly cautions against it); do *not*
  use bias–variance language (the paper doesn't).
- **One task remains before relocking:** land the nltools fix upstream (§5).
  It is written and green locally but not on `origin/master`, so dartbrains
  cannot pin it yet. Chapter prose does not depend on this.

---

## 0. Environment

The stale-HF-token problem is **gone**. `~/.cache/huggingface/token` was deleted;
anonymous access to the public `nltools/niftis` dataset works. **No
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` prefix is needed anywhere any more.** (`huggingface-cli`
is retired — the CLI is `hf auth login` if you ever need write access.)

Verification harness is unchanged and still the right one:

```bash
MPLBACKEND=Agg uv run marimo export ipynb --include-outputs content/<NB>.py -o /tmp/x.ipynb
python3 -c "
import json; nb=json.load(open('/tmp/x.ipynb'))
errs=[o for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='error']
print(f'cells: {len(nb[\"cells\"])}, errors: {len(errs)}')
for e in errs[:4]: print(' -', e.get('ename'), str(e.get('evalue'))[:200])
"
```

`marimo export script` only checks that the dataflow graph resolves — it will
happily pass a notebook that dies at runtime. Use `export ipynb --include-outputs`.

**Check figures too, not just the error count.** A cell can "pass" and emit a
blank figure. Extract `image/png` outputs and check byte size; anything under
~12 KB is almost certainly empty.

---

## 1. Corrections to the previous handoff — read this first

### (a) ✅ `predict(spatial_scale='roi')` was BROKEN — fixed 2026-08-01

**Resolved.** The chapter is no longer blocked on ROI scale. Skip to
"What the fix was" unless you want the history.

The previous handoff states all three scales were verified:

> | `'roi'` | `roi_mask=atlas` | `Predict.scores` = per-parcel accuracy vector (46 parcels on k50) |

**It raised on the real localizer data.** Every form of `roi_mask` failed:

```
CASE expanded, no groups : IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
CASE expanded, groups    : IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
CASE label img, no groups: IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
CASE expanded nifti      : IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
```

Minimal repro:

```python
import numpy as np
from dartbrains_tools.data import localizer
from nltools.data import BrainData
from nltools.mask import expand_mask
from nltools.templates import fetch_resource

subs  = localizer.get_subjects()
left  = BrainData([localizer.get_file(s, "betas", "audio_left_hand")  for s in subs])
right = BrainData([localizer.get_file(s, "betas", "audio_right_hand") for s in subs])
dat   = left.append(right)                      # (40, 238955)
y     = np.array([0]*len(subs) + [1]*len(subs))

atlas = BrainData(fetch_resource("masks/k50_2mm.nii.gz"), mask=dat.mask)  # (238955,)
ex    = expand_mask(atlas)                                                # (50, 238955)

dat.predict(y=y, spatial_scale="roi", roi_mask=ex, cv=5, n_jobs=1)        # IndexError
```

The traceback bottoms out in `BrainData.__getitem__`:

```
nltools/data/braindata/__init__.py:231 in __getitem__
    new.data = np.array(self.data[index, :]).squeeze()
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
```

#### What the fix was

The atlas-resolution block — coerce `roi_mask` to a Nifti, resample to
`bd.mask`, `apply_mask` to a label vector — existed in **four copies** in
nltools, and they had drifted apart:

| copy | handled `BrainData`? | handled stacked/4-D? |
|---|---|---|
| `analysis.py::_resolve_atlas_label_vec` (used by `align`) | yes | no |
| `analysis.py::reduce_per_roi` (`mean`/`std`/`median`) | yes | no |
| `analysis.py::_distance_roi` (`distance`) | yes | no |
| `prediction.py::_run_roi` (`predict`) | **no** | no |

So the observed failures were two separate defects:

- **`predict` alone dropped the `BrainData` branch.** A `BrainData` atlas fell
  through to `resample_to_img()`, nilearn treated it as an *iterable of images*,
  called `__getitem__` on it, and you got that `IndexError` from
  `braindata/__init__.py:231`. Nothing to do with `expand_mask`, which was fine.
- **No entry point accepted a stacked binary mask at all** — which is what
  `expand_mask` returns, and what a 4-D atlas file is. That one hit `distance`
  and `mean`/`std`/`median` too, so RSA at ROI scale had the same latent bug.

The fix collapses all four copies into one resolver,
`nltools/data/braindata/utils.py::resolve_roi_atlas`, which accepts every
idiomatic form: 3-D label Nifti, path, `BrainData` label vector, stacked binary
`BrainData` from `expand_mask`, and 4-D Nifti. Stacked forms are collapsed by a
new pure helper `nltools.mask.collapse_label_stack`. `predict` also picks up the
named "no nonzero labels" error the other three already had (it used to die with
`ValueError: need at least one array to concatenate`).

**Verified on the real localizer data**, 20 subjects, left vs right hand, k50:

```
expanded BrainData, cv=5          50 parcels    4.6 s   best parcel 11 acc=0.875   median 0.550
expanded BrainData, LeaveOneGroupOut  50 parcels   21.6 s   best parcel 11 acc=0.975   median 0.675
label BrainData,    cv=5          50 parcels    4.8 s   best parcel 11 acc=0.875   median 0.550
expanded 4-D Nifti, cv=5          50 parcels    5.0 s   best parcel 11 acc=0.875   median 0.550
```

All forms agree exactly, and ROI scale is **cheap** — ~5 s for 50 parcels, ~22 s
with subject-wise CV. It is comfortably a live (non-cached) chapter cell.
Parcel 11 at 0.975 with leave-one-subject-out is a strong, real motor effect —
a good payoff for the "where is the information?" narrative.

**Caveat on labels from a stacked mask:** a binary stack carries no label
*values*, only stack order, so parcels come back numbered 1..n. Round-tripping a
1..n atlas through `expand_mask` preserves its labels; an atlas labeled e.g.
{3, 7} comes back as {1, 2}. If you need the atlas's own IDs, pass the label
image or `BrainData` label vector rather than the expanded stack.

**Status of the fix:** written and green in the local nltools working tree
(`/Users/lukechang/Github/nltools`, uncommitted as of 2026-08-01) — full suite
1682 passed, `uv run poe lint` clean. **It is not on `origin/master` yet**, so
relocking dartbrains will *not* pick it up until it lands (see §5).

### (b) ~~Whole-brain searchlight is ~24 hours~~ — it is **11 minutes** (remeasured 2026-08-01)

**This section's original claim was wrong by ~130x.** It read:

> ```
> searchlight, radius_mm=10, 20 subjects, cv=5, n_jobs=-1
>   367 ms per voxel
>   whole brain (238,955 voxels) -> ~1,464 min = ~24 hours
> ```

Remeasured on an idle machine (load average ~4), same data, same parameters,
one job at a time — **directly, not extrapolated**:

```
searchlight, radius_mm=10, 20 subjects, cv=5, n_jobs=-1

  precentral (bilateral)      9,518 vox     18.4 s   1.9 ms/vox   peak acc 0.900
  sensorimotor (pre+post)    16,174 vox     39.6 s   2.4 ms/vox   peak acc 0.875
  WHOLE BRAIN               238,955 vox    661.1 s   2.8 ms/vox   peak acc 0.950
                                          = 11.0 min             median 0.475
                                                       only 1.9% of voxels > 0.70
```

The original 367 ms/voxel was contaminated by exactly the CPU contention this
section warns about two paragraphs down. The real rate is ~2–3 ms/voxel and it
is **essentially linear** in mask size — scaling from a 272-voxel sphere
predicted 7.7–9.7 min for whole brain against 11.0 min actual, so extrapolation
here is mildly optimistic but not dangerous.

**What this changes for the chapter:**

- A restricted a-priori searchlight is a **live cell** (18–40 s). No `mode:
  cached`, no hard restriction needed. The premise that the chapter must choose
  between a tiny mask and precomputation is gone.
- A **real whole-brain searchlight map is affordable** at 11 min. That is ~15%
  of the ~75 min build, so it is a judgment call rather than a blocker: run it
  live if the chapter wants one honest whole-brain map, or put that single
  chapter on `mode: cached` (§3) if 11 min is too much of the budget. Either is
  defensible; nothing is forced.

**The result itself is a good chapter payoff**: peak searchlight accuracy 0.950,
but median 0.475 and only 1.9% of searchlights above 0.70. So the information is
real and highly focal, while whole-brain decoding gets 0.675 from distributed
signal. Note this means the three scales do **not** produce a clean ordering —
see the warning in §4.

The methodological lesson from Task A still stands and now cuts both ways:
**measure on an idle machine, one job at a time.** Running two `n_jobs=-1` jobs
concurrently drove load average to 147 during Task A and inflated timings ~100x
— which is how this section acquired a 24-hour figure for an 11-minute job.
Distrust any timing in these handoffs that was not taken on a quiet machine.

---

## 2. What IS verified working

```
whole_brain predict, left vs right hand, 20 subjects, cv=5, n_jobs=-1
  1.8 s        mean_score = 0.675

roi predict (k50), same data, cv=5, n_jobs=1        (§1a, after the fix)
  4.6 s        50 parcels, best parcel 11 acc=0.875, median 0.550
roi predict (k50), same data, LeaveOneGroupOut, n_jobs=1
 21.6 s        50 parcels, best parcel 11 acc=0.975, median 0.675

searchlight, same data, radius_mm=10, cv=5, n_jobs=-1   (§1b, remeasured)
 18.4 s        precentral bilateral (9,518 vox), peak 0.900
 11.0 min      WHOLE BRAIN (238,955 vox), peak 0.950, median 0.475
```

All three scales verified on the real data, all three affordable — the two
blocking claims in §1 are resolved. Read §1b's warning about the median-vs-peak
result before building the comparison narrative.

`predict()` signature at the pinned rev (note it is already fully keyword-only
after `self`):

```python
BrainData.predict(*, y=None, X=None, spatial_scale='whole_brain', model='svm',
                  cv=5, standardize=True, reduce=None, n_components=None,
                  scoring='auto', groups=None, roi_mask=None, radius_mm=10.0,
                  inplace=False, n_jobs=1, random_state=None, progress_bar=False)
```

**`n_jobs` defaults to 1** here — pass `n_jobs=-1` explicitly.

### Available localizer contrasts (20 subjects)

```
audio_computation     audio_left_hand    audio_right_hand    audio_sentence
video_computation     video_left_hand    video_right_hand    video_sentence
horizontal_checkerboard                  vertical_checkerboard
```

So the plan's left-vs-right motor design is viable, and you can cross cue
modality (audio vs video) with effector (left vs right) — a nice way to show
that *what* the classifier separates depends on where you look.

---

## 3. Where the chapter goes

`book.yml`, in the `Neuroimaging Analysis` section, between
`Multivariate_Prediction.py` and `RSA.py`:

```yaml
      - file: content/Multivariate_Prediction.py
      - file: content/<NEW>.py            # <- here
      - file: content/RSA.py
        title: Representational Similarity Analysis
```

`RSA.py:407` already forward-references it, so either keep that pointer accurate
or update it:

> …at what spatial scale should this analysis be run? We will come back to that
> idea in more depth when we look at spatial feature selection.

### The `mode: cached` escape hatch

Two chapters already use it — `Download_Data.py` (46 GB snapshot) and
`Parcellations.py` (network fetches with bad TLS chains). Rendered once locally,
committed to `_rendered/`, so CI never re-executes them:

```yaml
      - file: content/Parcellations.py
        mode: cached
```

This is the obvious home for an expensive searchlight, if you decide the chapter
needs a real whole-brain map rather than a restricted one.

`book.yml` sets `defaults: execution_timeout: null` — there is **no** per-notebook
cap (marimo-book 0.1.27 added a 600 s default; it is deliberately disabled). The
full build has always been ~75 min. So nothing will *kill* a slow notebook; it
will just make the build unusable.

---

## 4. ✅ The paper — read 2026-08-01

[Jolly & Chang, 2021, *SCAN*](https://doi.org/10.1093/scan/nsab010),
"Multivariate spatial feature selection in fMRI". **Open access, no paywall** —
full text at [PMC8343556](https://pmc.ncbi.nlm.nih.gov/articles/PMC8343556).
(Note the senior author is the repo owner, so this chapter is effectively the
narrative documentation for his own framework.)

The `spatial_scale` vocabulary in nltools (`'whole_brain' | 'roi' | 'searchlight'`)
follows this paper, per nltools' CLAUDE.md.

### 🚨 The previous handoff's framing question was based on a false premise

It asked "how it frames the **bias–variance trade-off** across spatial scales."
**The paper does not use bias–variance language at all.** Do not write the
chapter around that axis.

What it actually argues is a **representational–methodological mismatch**:
psychological processes are organized at different spatial scales, and the
analytic method must match the scale of the representation being investigated.
The spectrum runs searchlight (locally distributed) → ROI → whole brain
(diffuse), and what is traded off is **sensitivity to local vs. distributed
information**, not error decomposition.

### The paper's own caution — this shapes the whole chapter

> we caution researchers against framing the issue as attempting to "optimize"
> for the "best" spatial scale

and it notes "model comparison between searchlights and whole brain models are
not trivial or even feasible." **So the chapter must not stage a bake-off with a
winner.** This converges exactly with what the data shows (§1b): the three
scales do not produce a clean ordering. The paper's argument and the measured
numbers tell the same story, which is a gift — write the chapter as
*matching method to representation*, not as *finding the best method*.

### Searchlight's interpretational pitfalls (outline point 4)

Three distinct criticisms, all quotable and all with a concrete hook in our data:

1. **Group inference is weaker than it looks.** Searchlights are "most often
   computed on individual brains and performance metrics are aggregated at the
   group level," so "rejecting the null-hypothesis only suggests that some
   individuals demonstrate an effect not that the effect is typical."
2. **Feature importance is ill-defined.** Because neighborhoods overlap, "each
   voxel has a different feature weight depending upon the particular searchlight
   it belongs to," making importance testing "infeasible." *Hook:* this is
   precisely why nltools' ROI runner can return a `weight_map` (each voxel
   belongs to exactly one parcel — disjoint reassembly) and the searchlight
   runner cannot. The API asymmetry is the paper's point made executable.
3. **Accuracy is spatially smeared.** "accuracy scores are 'smeared' over spatial
   extents because searchlights are overlapping," making it "nearly impossible to
   identify which voxels are most important for prediction." *Hook:* our measured
   map — peak 0.950, median 0.475, only 1.9% of spheres above 0.70 — is exactly
   the smeared-blob picture.

### What it says about whole brain

Strengths: whole-brain models "permit strong inferences about generalization,
based on model performance, and diffuse inferences about the spatial location of
representations based on feature weights." Popular as translational
"biomarkers."

Critical caveat for the chapter: "reliable weight maps do not indicate that a
voxel explicitly represents psychological information but that in concert with
other voxels it can effectively predict an outcome" — some voxels "serve to
denoise other voxels which share correlated noise." That is a strong,
teachable warning against reading a whole-brain weight map as a localization map.

### Goal-aligned heuristic (the chapter's practical takeaway)

| representation | scale the paper points to |
|---|---|
| fine-grained, locally organized (perceptual, visual cortex) | searchlight |
| in between, or when anatomy/function gives a prior | ROI |
| abstract, diffuse (emotion, social cognition) | whole brain |

Whole-brain models "have been more successful than searchlights in identifying
sensitive and specific predictive models of more abstract psychological
processes."

⚠️ Note our worked example — left vs right hand — is a **motor/perceptual**
effect, so the paper predicts searchlight and ROI should do well on it, and they
do (peak 0.950 / 0.975). Be explicit in the prose that this example sits at the
fine-grained end, and that an emotion or social-cognition target would likely
reverse the picture. Otherwise the chapter accidentally teaches "focal beats
distributed" as a general law, which is the opposite of the paper's argument.

### Draft outline (from the previous handoff, still reasonable)

1. The question: *at what spatial scale is the information?* Same data, same
   classifier, three answers.
2. Whole-brain — maximal sensitivity to distributed codes, no localization.
3. ROI — localizes, but inherits the atlas's assumptions. **(unblocked, §1a —
   ~5 s for k50, runs live)**
4. Searchlight — assumption-free localization at a compute cost, plus the
   interpretational pitfalls the paper raises. **(unblocked, §1b — restricted
   ~20–40 s runs live; whole brain 11 min if the chapter wants one.)**
5. Compare all three on the same left-vs-right motor data — the payoff.

   ⚠️ **The measured numbers do not give a clean scale-ordering**, so do not
   write the prose around one. On left-vs-right hand, 20 subjects, `cv=5`:
   whole brain 0.675; ROI (k50) median 0.550 with the best parcel at 0.875;
   searchlight median 0.475 with a peak of 0.950 and only 1.9% of spheres above
   0.70. The honest story is *not* "finer scale is better" — it is that
   whole-brain decoding aggregates weak distributed signal into a decent score,
   while ROI and searchlight reveal the information is actually focal and most
   of the brain is at chance. Median-vs-peak is the real contrast to teach, and
   it lands directly on the paper's interpretational-pitfalls argument (§4).
6. Cross-links: RSA (`distance(spatial_scale=)`), Parcellations, and Thresholding
   — searchlight maps need the same correction machinery, which is now a real
   chapter to point at.

Point 6 is stronger than it was: the thresholding chapter now actually teaches
max-statistic and cluster FWE, so "your searchlight accuracy map needs
correcting too" has somewhere concrete to land.

---

## 5. nltools state — matters for this work

dartbrains pins nltools by git rev in `uv.lock` (`rev = "master"` only controls
*resolution*; the lock freezes a SHA). Currently **`0fe8333f`**, which is
**8 commits behind `origin/master`** and predates the merged #469 h5py fix — the
explicit `[h5]` extra in `pyproject.toml` is covering for that.

Relocking (`uv lock --upgrade-package nltools`) is still worth doing early in
Task B, but **not** for the ROI blocker — that fix is not on `origin/master`
yet (§1a). Relock as a migration test: the breaking `iplot` niivue rewrite
(`b3603484`) landed after the pin and has never been exercised by these
notebooks. Sequence it as: land the ROI fix upstream → relock once → re-run the
11 nltools-bearing notebooks, so one relock covers both changes.

**Open nltools PRs** (all green, none merged):

| PR | What |
|---|---|
| [#473](https://github.com/cosanlab/nltools/pull/473) | `progress_bar=` across the inference family (breaking: bars now off by default) |
| [#475](https://github.com/cosanlab/nltools/pull/475) | keyword-only options in the inference layer (stacked on #473) |
| [#476](https://github.com/cosanlab/nltools/pull/476) | dependabot: mcp override + pymdown ignore |
| [#477](https://github.com/cosanlab/nltools/pull/477) | nilearn skill: cluster-forming threshold scale correction |
| [#470–#472](https://github.com/cosanlab/nltools/pulls) | held deliberately — `fit()` cleaning RFC, `add_poly` heuristic, `find_spikes` hygiene |
| [#474](https://github.com/cosanlab/nltools/issues/474) | **issue** — `nltools.stats` vs `algorithms.inference` export 13 duplicate names; design discussion for Eshin |

If #473 merges before Task B starts, note that progress bars are **off by
default** now — `one_sample_permutation_test` and friends will be silent unless
you pass `progress_bar=True`.

---

## 6. API facts worth not rediscovering

Carried forward, plus what this session added.

- **`BrainData.shape` is a property, not a method** — `dat.shape()` raises
  `TypeError: 'tuple' object is not callable`.
- **`predict()` defaults to `n_jobs=1`.** Pass `-1` explicitly.
- **`atlases/atlas_harvard_oxford.nii.gz` is a 4D *probabilistic* atlas**
  (151×194×159×113) — one volume per region, probabilities 0–100, in row order
  matching `atlases/labels_harvard_oxford.csv`. No label-value offset to worry
  about. Build a mask with
  `BrainData(path, mask=data.mask)[rows].data.max(axis=0) > 25`.
- **`nilearn.non_parametric_inference(threshold=)` is a P-VALUE**, not a t-value.
  A t-like value silently yields all-zero cluster maps (see the Task A handoff
  §3.1a, and nltools PR #477). nilearn is inconsistent about this internally:
  `threshold_stats_img` and `cluster_level_inference` take **z-scale** thresholds
  (both default 3.0).
- **`threshold_stats_img` assumes z-scaled input** — pass `stats['z']`, not
  `stats['t']`.
- **`zscore` accepts pandas but always returns Polars.** `z ** 2` raises `TypeError`.
- **`DesignMatrix.plot()` defaults to a heatmap now** — time courses need
  `method='timeseries'`; `method='corr'` replaces seaborn correlation heatmaps.
- **`predict(cv=<int>, groups=...)` used to silently ignore `groups`** and run
  StratifiedKFold. **This now raises** (fixed 2026-08-01, same working tree as
  §1a) — the guard also catches an explicitly-passed `KFold`/`StratifiedKFold`/
  `ShuffleSplit`/`LeaveOneOut`, which is how `BrainCollection.predict(cv=5,
  groups=...)` leaked one level up. Subject-wise CV needs a group-aware splitter
  (`GroupKFold`/`LeaveOneGroupOut`); custom splitters pass through untouched.
  `Multivariate_Prediction.py` already does this correctly, so no dartbrains
  notebook changes.

  ⚠️ **Correction to the previous claim** that ignoring groups "silently inflates
  accuracy." On this data it does the opposite: `cv=5` gives median parcel
  accuracy 0.550, `LeaveOneGroupOut` gives 0.675. Each subject contributes one
  sample per class, so random folds leak little, while LOGO trains on 38 samples
  instead of 32. **The direction of the bias depends on the design** — don't
  teach "grouping lowers your accuracy" from this example. What is true, and
  what still matters for Task B, is that all three scales must use the *same*
  CV scheme or their numbers are not comparable.
- **`BrainData` picks a template by voxel resolution** — pass `mask=<data>.mask`
  so a 1 mm atlas lands in 2 mm space.
- **`fetch_resource` returns a path**, so it feeds nilearn directly.
- **`fdr()` returns `-1` when nothing survives.** Always branch; thresholding at
  a negative p keeps every voxel.
- **Stale marimo caches survive API changes and fail confusingly.**
  `content/__marimo__/cache/` holds pickled objects; they are untracked build
  artifacts — `rm -rf` and let them regenerate. Cache plain arrays, never
  library objects.
- **`mo.persistent_cache` is incompatible with matplotlib figure rendering** —
  on a cache hit the plotting code does not re-run and `plt.gcf()` returns an
  empty figure. Wrap only the computation; plot in a downstream cell.

---

## 7. Suggested order of attack

1. ~~Re-test the ROI blocker~~ — **done, fixed** (§1a). Land the nltools fix
   upstream (commit + PR from the local nltools tree), since dartbrains cannot
   pin it until it is on `origin/master`.
2. **Relock nltools** (`uv lock --upgrade-package nltools`) once that fix lands,
   re-run the 11 nltools-bearing notebooks through the harness. Catches the
   `iplot` rewrite in the same pass.
3. ~~Read Jolly & Chang (2021)~~ — **done** (§4). Framing settled: match method
   to representation, do NOT stage a scale bake-off, and do not use
   bias-variance language (the paper doesn't).
4. ~~Pick the searchlight budget~~ — **done, measured** (§1b). Restricted
   searchlight runs live; whole brain is 11 min, so the only open call is
   whether to spend 15% of the build on one whole-brain map or put the chapter
   on `mode: cached`. Decide while writing, not before.
5. **Write the chapter**, verify with `export ipynb --include-outputs`, and check
   figure byte sizes, not just the error count.
6. Update the `RSA.py:407` forward-reference and the `book.yml` TOC.
