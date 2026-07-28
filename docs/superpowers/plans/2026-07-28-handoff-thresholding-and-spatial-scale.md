# Handoff: thresholding chapter + spatial feature selection chapter

**Written**: 2026-07-28
**Branch**: `nltools-0.6-migration` (dartbrains) — 12 commits ahead of `master`, tree clean
**Status**: nltools 0.6 API migration is **complete**. The two remaining tasks are new content.

Read this first, then [`2026-07-27-thresholding-collection-spatial-scale.md`](./2026-07-27-thresholding-collection-spatial-scale.md)
for the full design rationale and the validated statistics. This document is the
operational handoff: what's done, what's next, and what will bite you.

---

## 0. Start here — environment and verification

**The HF token on this machine is stale** and 401s against the *public*
`nltools/niftis` dataset. Every command below needs:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run ...
```

Permanent fix: `huggingface-cli logout` and re-login, or clear
`~/.cache/huggingface/token`. Worth checking CI doesn't carry a stale one too.

**Verification harness.** Use this, not `marimo export script`:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 MPLBACKEND=Agg \
  uv run marimo export ipynb --include-outputs content/<NB>.py -o /tmp/x.ipynb
python3 -c "
import json; nb=json.load(open('/tmp/x.ipynb'))
errs=[o for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='error']
print(f'cells: {len(nb[\"cells\"])}, errors: {len(errs)}')
for e in errs[:4]: print(' -', e.get('ename'), str(e.get('evalue'))[:200])
"
```

`marimo export script` only checks that the dataflow graph resolves — it
linearizes into one namespace, so it will happily pass a notebook that fails at
runtime (e.g. a variable used downstream but never returned from its defining
cell). `export ipynb --include-outputs` actually executes every cell and is
what marimo-book's build gate uses. Three real bugs in this migration were
caught only by execution.

**Stale marimo caches survive an API migration and fail confusingly.**
`content/__marimo__/cache/threshold_*` held objects pickled against the pre-0.6
`nltools.simulator` path and raised `ModuleNotFoundError` before any cell code
ran. They're untracked build artifacts — `rm -rf` and let them regenerate.

---

## 1. What's done

All 11 nltools-bearing notebooks migrated to 0.6 and verified executing:

| Notebook | Cells | Errors |
|---|---:|---:|
| `Group_Analysis` | 65 | 0 |
| `Connectivity` | 68 | 0 |
| `Thresholding_Group_Analyses` | 50 | 0 |
| `Multivariate_Prediction` | 46 | 0 |
| `RSA` | 41 | 0 |
| `Introduction_to_ICA` | 31 | 0 |
| `Parcellations` | 31 | 0 |
| `ICA` | 15 | 0 |
| `GLM_Single_Subject_Model`, `GLM`, `Introduction_to_Neuroimaging_Data` | — | 0 |

Not affected: `MR_Physics`, `Signal_Processing`, `Preprocessing` (the three
`mode: wasm` notebooks — **they don't use nltools, so the Pyodide surface is
untouched**), plus the pandas/polars/plotting/programming intros and
`Download_Data`.

**Deployment is safe**: `deploy-marimo-book.yml` triggers only on `master` and
`v2-marimo-migration`. This branch cannot reach the live site.

---

## 2. Upstream PRs (blocking-ish)

Four PRs, all from bugs this migration surfaced.

| PR | Status | Effect on dartbrains |
|---|---|---|
| [#469](https://github.com/cosanlab/nltools/pull/469) h5py core dep | **merged** | Already reflected — `pyproject.toml` uses `nltools[h5,graph,interactive_plots]` |
| [#470](https://github.com/cosanlab/nltools/pull/470) `fit()` stops cleaning; warns on rank deficiency | open (RFC) | **Will make GLM chapter prose stale** — see §5 |
| [#471](https://github.com/cosanlab/nltools/pull/471) `add_poly`/`add_dct_basis` underscore heuristic | open | Lets `add_poly()` run after appending motion confounds; `Connectivity` currently works around it by ordering |
| [#472](https://github.com/cosanlab/nltools/pull/472) `find_spikes` duplicates + row count | open | Makes tutorials robust for students on their own data |

dartbrains pins nltools to git `master`, so **none of #470–#472 are in the
installed version yet**. Anything written now runs against pre-#470 behavior.

---

## 3. Task A — thresholding chapter: permutation + cluster inference

Target: `content/Thresholding_Group_Analyses.py` (already migrated; 50 cells green).

### The toolbox boundary (verified, don't re-litigate)

| Capability | Where |
|---|---|
| Voxelwise sign-flip permutation | ✅ nltools — `ttest(permutation=True)`, `one_sample_permutation_test`, `BrainCollection.permutation_test()`. GPU-capable, `return_null=True` |
| Cluster **extent thresholding** | ✅ nltools — `threshold(..., cluster_threshold=N)`. Descriptive; builds no null |
| Atlas-labeled cluster tables | ✅ nltools — `cluster_report(stat_threshold=, cluster_threshold=, atlas=)` → `.peaks`, `.clusters`, `.plot()`, `.to_csv()` |
| FDR / Holm-Bonferroni | ✅ `nltools.stats.corrections` (exactly 4 functions: `fdr`, `holm_bonf`, `threshold`, `multi_threshold`) |
| **FWE-calibrated** cluster correction, TFCE, max-stat | ❌ **absent from nltools** — comes from nilearn |

nilearn's `non_parametric_inference` returns all four corrections in one call
(verified): `threshold=` → `['t','logp_max_t','size','logp_max_size','mass','logp_max_mass']`;
`tfce=True` → `['t','logp_max_t','tfce','logp_max_tfce']`.

### Agreed approach

Hand-rolled from the null → nilearn as the production tool → `cluster_report()`
to label survivors. Decided with the user; don't revisit.

### Two findings that MUST shape the chapter

Both validated by simulation; scripts committed at
`docs/superpowers/plans/validation-2026-07-27/`.

**(a) Permutation p-values floor at ~`1/n_permute`.** With 500 permutations the
smallest attainable p is 0.002, so Bonferroni over 900 voxels (target 5.6e-5) is
*unreachable*. Classical corrections must be demonstrated on **parametric**
p-values and the FWE procedures on the **permutation** null. Mixing the two
families produces a nonsense 0% uncorrected rate (I hit this on the first pass).
This floor is also exactly what motivates max-statistic FWE, so it belongs in
the chapter as a teaching point rather than a footnote.

**(b) `SimulateGrid` noise is spatially independent** — `_create_noise()` is a
bare `randn(w, w, n_subjects) * sigma`, no smoothing. Cluster inference
presupposes spatial autocorrelation, so on iid noise the null cluster-size
distribution collapses and the test is degenerate:

```
smoothing sigma=0     cluster-extent FWE = 0.0%   median observed cluster = 1 vox
smoothing sigma=2.0   cluster-extent FWE = 6.0%   median observed cluster = 5 vox
```

The simulation must be smoothed before the cluster section, and **showing both
numbers side by side is the single best demonstration in the chapter** — it is
the mechanism behind Eklund, Nichols & Knutsson (2016).

### Measured calibration (100 null sims, 900 voxels, α=.05)

```
unc_001         56.0%  ±9.7%     <- theory: 1-(1-.001)^900 = 59.4%  ✓
bonferroni       6.0%  ±4.7%     ✓
fdr_05           6.0%  ±4.7%     ✓ (FDR controls FWER under the complete null)
perm_maxstat     4.0%  ±3.8%     ✓ calibrated
perm_cluster     6.0%             ✓ (smoothed; 0% unsmoothed — see (b))
```

### Verified building blocks

```python
def max_stat_threshold(null, alpha=0.05):
    return np.percentile(np.abs(null).max(axis=1), 100 * (1 - alpha))

def cluster_fwe_p(obs, null, w, forming):
    """p = P(max null cluster size >= largest observed cluster)."""
    null_max = np.array([largest_cluster((np.abs(p) > forming).reshape(w, w)) for p in null])
    obs_size = largest_cluster((np.abs(obs) > forming).reshape(w, w))
    if obs_size == 0:
        return 1.0, 0
    return (np.sum(null_max >= obs_size) + 1) / (len(null_max) + 1), obs_size
```

Use a **permutation p-value**, not a percentile cut — cluster sizes are integers
and heavily tied, so a strict `>` against a percentile is needlessly conservative.

### ⚠️ Memory guardrail

`return_null=True` returns `(n_permute, n_voxels)`. At the chapter's 100×100 grid
× 1000 perms that's ~78 MB — fine. **At whole-brain scale (238k voxels × 5000)
it is ~9.5 GB — never call it on real brain data.** Steps built from the null
stay on the simulated grid; real data goes through nilearn.

### Proposed structure

1. The problem — keep the existing simulation, now quantified: **56% of null maps
   contain ≥1 false positive at p<.001** vs theoretical 59.4%
2. Classical corrections — Bonferroni and FDR on parametric p-values
3. Permutation — sign-flipping; **teach the `1/n_permute` floor here**
4. Max-statistic FWE — three lines from `return_null=True`; verify calibration live
5. Cluster inference — smooth the grid first; show the 0% → 6% contrast; connect to Eklund 2016
6. Real data — `nilearn.non_parametric_inference(threshold=, tfce=True)` on localizer contrasts
7. Reporting — `cluster_report(atlas=...)`; be explicit this step is *descriptive*, not corrective

### Also fix while in there

- `fdr()` returns `-1` when nothing survives. The migrated notebook now branches
  on it, but re-check any new call sites — thresholding at a negative p keeps
  every voxel, a bad failure mode in this chapter specifically.
- `SimulateGrid.threshold_simulation(correction=)` only recognizes `'fdr'`. There
  is no permutation or cluster path in that class; don't assume one.
- `plot_grid_simulation()` internally re-runs with `_run_ttest`, i.e. parametric,
  regardless of what you did to `t_values`/`p_values`. Pre-existing
  inconsistency — decide whether to work around or avoid.

---

## 4. Task B — new chapter: spatial feature selection

Slots after `Multivariate_Prediction`, before `RSA`. Standalone chapter (decided).

**Read [Jolly & Chang, 2021, *SCAN*](https://doi.org/10.1093/scan/nsab010) before
drafting prose.** This is an outstanding prerequisite — the chapter should
reflect the paper's actual argument, not a generic summary of the scale axis.
Open questions to settle from it: how it frames the bias–variance trade-off
across scales, and what it concludes about searchlight's interpretational
pitfalls.

### Verified API — all three scales work

`predict(spatial_scale=...)` returns the natural output for each:

| scale | kwargs | returns |
|---|---|---|
| `'whole_brain'` | — | `Predict` with `.weight_map`, `.scores`, `.mean_score` |
| `'roi'` | `roi_mask=atlas` | `Predict.scores` = per-parcel accuracy vector (46 parcels on k50) |
| `'searchlight'` | `roi_mask=`, `radius_mm=` | `Predict.accuracy_map` = voxel-space `BrainData`, directly plottable |

`BrainCollection.predict()` takes the same kwargs plus `cv='loso'`.

### Draft outline

1. The question: *at what spatial scale is the information?* Same data, same
   classifier, three answers
2. Whole-brain — maximal sensitivity to distributed codes, no localization
3. ROI — localizes, but inherits the atlas's assumptions
4. Searchlight — assumption-free localization at a compute cost, plus the
   interpretational pitfalls the paper raises
5. Compare all three on the same localizer left-vs-right motor data — the payoff,
   and the reason it's standalone
6. Cross-links: RSA (`distance(spatial_scale=)`), Parcellations, Thresholding
   (searchlight maps need the same correction machinery — natural callback)

### ⚠️ Compute budget is the main risk

Searchlight over a 6×6×6-voxel restricted mask ran in 14 s. **Whole-brain
searchlight × ~25 subjects at build time may blow the ~75 min build.** Measure
before committing. Options in order: restrict to a gray-matter or ROI mask;
coarsen `radius_mm`; precompute and cache (`mode: cached`, as `Download_Data`
and `Parcellations` already do).

`RSA.py` already forward-references this chapter ("We will come back to that idea
in more depth when we look at spatial feature selection") — keep or update that
pointer.

---

## 5. Tracked follow-ups

**If #470 merges**, `content/GLM_Single_Subject_Model.py` needs updating (commit
`213db3b` flags this inline):
- The prose references `design_clean=False`, a kwarg #470 removes
- That design (DCT basis + `add_poly(order=2)`) will start emitting the new
  rank-deficiency warning instead of silently dropping `poly_1`/`poly_2`
- Per the user's position that **regularization should be preferred over
  deletion**, the better teaching move is probably to show `model='ridge'` as the
  response rather than `.clean()`

**Open pedagogical question, not yet decided**: the GLM chapter adds both a DCT
basis *and* polynomial drift — genuinely redundant, and the prose presents them
as alternatives ("we typically use this approach rather than applying a high pass
filter") before using both. Options: drop the redundancy; keep it and let the
rank warning be the lesson; or keep it and show ridge. The user leans toward
regularization generally.

**Possible cross-chapter thread**: if regularization is the preferred answer to
collinear designs, `Multivariate_Prediction`'s regularization section is a natural
place to make that argument for the *first-level* model too. Currently the two are
taught as unrelated topics.

**Unverified**: the RSA per-ROI `similarity(..., method=None)` NaN edge (a parcel
with no coverage raises rather than warns). It did **not** reproduce on the real
localizer betas, only on synthetic data. Watch for it if the atlas or dataset
changes.

---

## 6. API facts worth not rediscovering

Things that cost time this round and aren't obvious from signatures:

- **`zscore` accepts pandas but always returns Polars.** `z ** 2` on the result
  raises `TypeError`. Motion-covariate expansions must use Polars expressions.
- **`DesignMatrix.plot()` changed meaning** — was pandas' line plot, now defaults
  to a heatmap. Time courses need `method='timeseries'`; `method='corr'` replaces
  seaborn correlation heatmaps.
- **`fit()` drops near-collinear regressors** (pre-#470), so the fitted model can
  have fewer regressors than the design. Contrast *vectors* then fail on length;
  contrast *strings* keep working. `compute_contrasts` supports `+`, `-`, and
  scalar `0.25*name`, but **not** parentheses or division.
- **`predict(cv=<int>, groups=...)` silently ignores `groups`** and runs
  StratifiedKFold. Subject-wise CV requires an explicit splitter
  (`GroupKFold`/`LeaveOneGroupOut`). This silently inflates accuracy.
- **`BrainData` picks a template by voxel resolution**, so a 1mm anatomical mask
  loads into 1mm space and won't align with 2mm functional data. Pass
  `mask=<data>.mask`.
- **`find_spikes` returns a `DesignMatrix`** with spikes pre-marked as confounds
  and takes `TR=`. No `TR` index column to drop any more.
- **`BrainData(list_of_paths)` no longer flattens** — the `[BrainData(f) for f in …]`
  double-wrap workaround is obsolete.
- **`BrainCollection` is fully functional**; the upstream migration guide calling
  it a scaffold is stale (fixed in #469). `from_paths(design_paths=…)` passes
  paths through *unparsed*, and the `X=` callable receives a `_DesignContext`
  (`.bd`, `.dm`, `.confounds`, `.TR`, `.subject`, …), not a `DesignMatrix`.
- **`fetch_resource` returns a path**, so it feeds nilearn directly — no
  `BrainData(url).to_nifti()` round-trip. All five previously hard-coded
  Neurovault/GitHub atlas URLs are in the `nltools/niftis` HF dataset.
