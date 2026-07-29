# Handoff: Task B — spatial feature selection chapter

**Written**: 2026-07-29
**Branch**: `nltools-0.6-migration` (dartbrains) — now **pushed** to origin, 14 commits ahead of `master`, tree clean
**Status**: Task A (thresholding) is **done and verified**. Task B is the last item from the original handoff.

Supersedes the Task-B sections of
[`2026-07-28-handoff-thresholding-and-spatial-scale.md`](./2026-07-28-handoff-thresholding-and-spatial-scale.md).
Read §1 before designing anything — **two of that document's load-bearing claims
about Task B are wrong**, and one of them is a hard blocker.

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

### (a) 🚨 `predict(spatial_scale='roi')` is BROKEN — this is a blocker

The previous handoff states all three scales were verified:

> | `'roi'` | `roi_mask=atlas` | `Predict.scores` = per-parcel accuracy vector (46 parcels on k50) |

**It raises on the real localizer data.** Every form of `roi_mask` fails:

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

i.e. something in the ROI path iterates a `BrainData` and then re-indexes the
1-D slices it gets back. `expand_mask` itself is **fine** — it returns
`(50, 238955)` correctly; the failure is downstream in `predict`.

**Note this is the *pinned* nltools** (`uv.lock` → `0fe8333f`), the same rev the
previous handoff verified against — so this is not a regression from a newer
nltools. Either the original verification used a call shape not tried here, or
the claim was wrong.

**Before designing the chapter around ROI scale:**
1. Re-test against current `origin/master` — nltools has moved on since `0fe8333f`
   (see §5); this may already be fixed.
2. If it is still broken, it needs an upstream fix, and the chapter's ROI
   section is blocked until then. `BrainCollection.predict(spatial_scale='roi')`
   is a separate code path and untested here — worth trying as a workaround.

### (b) Whole-brain searchlight is ~24 hours, not "may blow the build"

The previous handoff says whole-brain searchlight "may blow the ~75 min build."
Measured, it is not close:

```
searchlight, radius_mm=10, 20 subjects, cv=5, n_jobs=-1
  367 ms per voxel
  whole brain (238,955 voxels) -> ~1,464 min = ~24 hours
```

(Measured with some CPU contention, so treat as an upper bound — but even a
4x-optimistic reading is ~6 hours. It is definitively not runnable at build time.)

Practical consequence: a searchlight over even a **2,000-voxel** mask costs
~12 min, and **500 voxels** ~3 min. So the chapter must either restrict hard to
a small a-priori mask, or precompute and use `mode: cached`, or both. Decide this
*before* writing prose — it shapes what the chapter can honestly show.

This mirrors the TFCE lesson from Task A exactly: the cheap-looking extrapolation
was off by two orders of magnitude. **Measure on an idle machine, one job at a
time** — running two `n_jobs=-1` jobs concurrently drove load average to 147 and
inflated timings ~100x during Task A.

---

## 2. What IS verified working

```
whole_brain predict, left vs right hand, 20 subjects, cv=5, n_jobs=-1
  1.8 s        mean_score = 0.675
```

Cheap, and a real effect — a good anchor for the chapter's opening.

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

## 4. Outstanding prerequisite: read the paper

[Jolly & Chang, 2021, *SCAN*](https://doi.org/10.1093/scan/nsab010) is **still
unread**, and the previous handoff is explicit that the chapter should reflect
the paper's actual argument rather than a generic summary of the scale axis.

Two questions to settle from it:
- how it frames the bias–variance trade-off across spatial scales
- what it concludes about searchlight's interpretational pitfalls

The `spatial_scale` vocabulary in nltools (`'whole_brain' | 'roi' | 'searchlight'`)
follows this paper, per nltools' CLAUDE.md, so the chapter is effectively the
narrative documentation for that design choice.

### Draft outline (from the previous handoff, still reasonable)

1. The question: *at what spatial scale is the information?* Same data, same
   classifier, three answers.
2. Whole-brain — maximal sensitivity to distributed codes, no localization.
3. ROI — localizes, but inherits the atlas's assumptions. **(blocked, §1a)**
4. Searchlight — assumption-free localization at a compute cost, plus the
   interpretational pitfalls the paper raises. **(restrict hard, §1b)**
5. Compare all three on the same left-vs-right motor data — the payoff.
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

Relocking (`uv lock --upgrade-package nltools`) is worth doing early in Task B,
for two reasons: it may fix the ROI blocker in §1a, and it is itself a migration
test — the breaking `iplot` niivue rewrite (`b3603484`) landed after the pin and
has never been exercised by these notebooks.

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
- **`predict(cv=<int>, groups=...)` silently ignores `groups`** and runs
  StratifiedKFold. Subject-wise CV needs an explicit splitter
  (`GroupKFold`/`LeaveOneGroupOut`). **This silently inflates accuracy — and it
  matters directly for Task B**, where every scale must use subject-wise CV or
  the three numbers are not comparable.
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

1. **Relock nltools** (`uv lock --upgrade-package nltools`), re-run the 11
   nltools-bearing notebooks through the harness. Catches the `iplot` rewrite,
   and may resolve §1a for free.
2. **Re-test the ROI blocker.** If still broken, file it upstream; decide whether
   the chapter ships with two scales plus a note, or waits.
3. **Read Jolly & Chang (2021).** Settle the framing before any prose.
4. **Pick the searchlight budget** from §1b — restricted mask, `mode: cached`, or
   both. Measure the actual mask you choose on an idle machine.
5. **Write the chapter**, verify with `export ipynb --include-outputs`, and check
   figure byte sizes, not just the error count.
6. Update the `RSA.py:407` forward-reference and the `book.yml` TOC.
