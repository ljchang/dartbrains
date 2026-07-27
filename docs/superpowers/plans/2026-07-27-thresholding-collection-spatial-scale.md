# Plan: permutation/cluster inference, BrainCollection, and spatial feature selection

**Branch**: `nltools-0.6-migration`
**Date**: 2026-07-27
**Depends on**: [`2026-07-27-nltools-0.6-migration.md`](./2026-07-27-nltools-0.6-migration.md) — the API migration must land first; this plan writes new content against the 0.6 API.

Four workstreams, all decided:

1. Thresholding chapter — permutation + cluster inference: **hand-rolled from the null, then nilearn, then `cluster_report()` to label**
2. `BrainCollection` — replace subject loops **everywhere**, including RSA and Connectivity, keeping one explicit teaching loop
3. Spatial cleanup — masks/parcellations via `spatial_scale=`
4. Spatial feature selection — **new standalone chapter** based on [Jolly & Chang, 2021, *SCAN*](https://doi.org/10.1093/scan/nsab010)

---

## 0. What's verified, and where the toolbox boundary actually is

Everything below was run against the synced env (nltools `master` @ `0fe8333f`, nilearn 0.12.1), not taken from docs.

### Inference: what lives where

| Capability | nltools 0.6 | Evidence |
|---|---|---|
| Voxelwise sign-flip permutation | ✅ `ttest(permutation=True)`, `BrainCollection.permutation_test()`, `one_sample_permutation_test()`. GPU-capable, deterministic across backends, `return_null=True` exposes the full `(n_permute, n_voxels)` null | verified |
| Cluster **extent thresholding** | ✅ `threshold(..., cluster_threshold=N)` — changelog: *"Add cluster thresholding to Brain_Data.threshold()"*. Wraps `nilearn.image.threshold_img`. Descriptive: drops small clusters, builds no null | verified |
| Cluster **reporting** | ✅ `cluster_report(stat_threshold=, cluster_threshold=, atlas=)` → `ClusterReport` with `.peaks`, `.clusters` (polars), `.stat_img`, `.plot()`, `.to_csv()` | verified |
| FDR / Holm-Bonferroni | ✅ `nltools.stats.corrections` = `{fdr, holm_bonf, threshold, multi_threshold}` — all four functions in the module | verified |
| **FWE-calibrated** cluster correction, TFCE, max-stat | ❌ **absent.** `grep -riE "tfce\|family.?wise\|fwe\|max.?stat"` over the installed package returns zero hits outside a Mantel citation | verified |

**The correction comes from nilearn**, already a dependency. One call returns all four:

```python
from nilearn.glm.second_level import non_parametric_inference
out = non_parametric_inference(con_imgs, design_matrix=dm, mask=mask,
                               n_perm=10000, threshold=2.5, tfce=True)
# threshold=  -> ['t', 'logp_max_t', 'size', 'logp_max_size', 'mass', 'logp_max_mass']
# tfce=True   -> ['t', 'logp_max_t', 'tfce', 'logp_max_tfce']
```

| output | correction |
|---|---|
| `logp_max_t` | voxel-level FWE (max-statistic) |
| `logp_max_size` | cluster-extent FWE |
| `logp_max_mass` | cluster-mass FWE |
| `logp_max_tfce` | TFCE |

So the chapter teaches three genuinely-new tools across two packages. That boundary is worth stating explicitly to students — it's a realistic picture of how the Python neuroimaging stack composes.

### Statistical core — validated, not assumed

Ran 100 null simulations (30×30 = 900 voxels, 20 subjects, 500 permutations, α = .05), measuring the *family-wise* false-positive rate:

```
unc_001         56.0%  ±9.7%     <- theory: 1-(1-.001)^900 = 59.4%  ✓
bonferroni       6.0%  ±4.7%     ✓
fdr_05           6.0%  ±4.7%     ✓ (FDR controls FWER under the complete null)
perm_maxstat     4.0%  ±3.8%     ✓ calibrated
perm_cluster     0.0%             <- NOT a bug; see below
```

Two findings that must shape the chapter:

**(a) Permutation p-values have a resolution floor of ~`1/n_permute`.** With 500 permutations the smallest attainable p is 0.002, so a Bonferroni target of `.05/900 = 5.6e-5` is *unreachable* — you cannot Bonferroni-correct 900 voxels with 500 permutations. Max-statistic FWE sidesteps this exactly because it uses the null of the maximum rather than per-voxel p-values. Classical corrections must therefore be demonstrated on **parametric** t-test p-values and the FWE procedures on the **permutation** null; mixing the two families produces the nonsense 0% uncorrected rate I hit on the first pass.

**(b) `SimulateGrid` noise is spatially independent** — `_create_noise()` is a bare `randn(w, w, n_subjects) * sigma`, no smoothing. Cluster inference presupposes spatial autocorrelation, so on iid noise the null cluster-size distribution collapses to size 1–2 and the test becomes degenerate. Confirmed by adding a Gaussian filter:

```
smoothing sigma=0     cluster-extent FWE = 0.0%   median observed cluster = 1 vox
smoothing sigma=2.0   cluster-extent FWE = 6.0%   median observed cluster = 5 vox
```

This is a *feature* for the course, not an obstacle: it is the cleanest possible demonstration of why spatial smoothness underwrites cluster inference — the same mechanism behind Eklund, Nichols & Knutsson (2016). The simulation section must smooth the grid before the cluster demo, and should show both numbers side by side.

Also use a **permutation p-value** (`(#{null_max >= obs} + 1) / (n_permute + 1)`) rather than a percentile cut — cluster sizes are integers and heavily tied, so a strict `>` against a percentile is needlessly conservative.

Both simulations are committed alongside this plan and can be re-run:
`validation-2026-07-27/fwe_validate.py` (the five-method calibration table) and
`validation-2026-07-27/fwe_smooth.py` (the smoothing contrast).

### Verified building blocks

```python
# max-statistic FWE
def max_stat_threshold(null, alpha=0.05):
    return np.percentile(np.abs(null).max(axis=1), 100 * (1 - alpha))

# cluster-extent FWE (permutation p-value; handles integer ties)
def cluster_fwe_p(obs, null, w, forming):
    null_max = np.array([largest_cluster((np.abs(p) > forming).reshape(w, w)) for p in null])
    obs_size = largest_cluster((np.abs(obs) > forming).reshape(w, w))
    if obs_size == 0:
        return 1.0, 0
    return (np.sum(null_max >= obs_size) + 1) / (len(null_max) + 1), obs_size
```

Memory: `return_null=True` yields `(n_permute, n_voxels)`. At the chapter's 100×100 grid × 1000 permutations that's ~78 MB — fine. **At whole-brain scale (238 k voxels × 5000) it is ~9.5 GB — do not call `return_null=True` on real brain data.** The hand-rolled section stays on the simulated grid; real data goes through nilearn.

### `BrainCollection` — real, and the migration guide is stale

The upstream migration guide still says *"not yet available (scaffold)"*. **That is out of date.** Verified working end to end:

```python
bc  = BrainCollection.from_paths(bold_paths, mask=mask, design_paths=ev_paths, metadata=meta)
bc2 = bc.smooth(6).fit(model='glm', X=lambda ctx: DesignMatrix(ctx.dm, run_length=len(ctx.bd), TR=tr)
                                                 .add_poly(order=1, include_lower=True))
con = bc2.compute_contrasts('face_c0 - house_c0', statistic='beta')   # -> BrainCollection
con.ttest()                     # -> {'mean','t','z','p'}
con.permutation_test(n_permute=200)   # -> {'mean','p'}
con.predict(y=y, spatial_scale='whole_brain'|'roi', cv=4)
```

Constructors: `from_paths`, `from_glob`, `from_bids`. Other methods: `align`, `anova`, `isc`, `cv`, `map`, `apply`, `detrend`, `filter`, `standardize`, `resample`, `threshold`, `write`.

Three sharp edges found:

1. **`h5py` is required but ships in an optional extra.** `BrainCollection.fit()` writes HDF5 fit bundles; without it every fit dies as `ModuleNotFoundError: No module named 'h5py'` *inside a joblib worker*, which is a confusing traceback for students. Already fixed in `pyproject.toml` → `nltools[h5,graph,interactive_plots]` (`graph` for `Adjacency.to_graph()` in Connectivity, `interactive_plots` for `iplot`'s default ipywidgets slider).
2. **`from_paths(design_paths=...)` passes raw paths through.** The source comment is explicit: *"DesignMatrix has no read() classmethod yet, so callers that pass paths today are responsible for loading."* So `ctx.dm` is a `str`, not a `DesignMatrix`. The builder must construct it. Not a bug, but it will confuse students unless the notebook says so.
3. **The `X=` callable receives a `_DesignContext`, not a DesignMatrix.** Fields: `bd`, `dm`, `confounds`, `sample_mask`, `metadata`, `subject`, `session`, `run`, `task`, `TR`, `bold_path`, `events_path`, `confounds_path`, plus `__getitem__` falling through to metadata. This is *good* for teaching — the per-subject design builder gets everything it needs and reads declaratively — but the contract needs to be shown explicitly.

dartbrains' loader is a path accessor (`localizer.get_file(sub, 'derivatives', 'bold')`) over an HF cache, not a browsable BIDS tree, so **`from_paths` is the right constructor**, built from a comprehension over `localizer.get_subjects()`. (`dartbrains_tools.bids` exists and may expose a layout — check whether `from_bids` is viable before settling.)

### Spatial scale — all three points verified

`predict(spatial_scale=...)` returns the natural output for each scale:

| scale | kwargs | returns |
|---|---|---|
| `'whole_brain'` | — | `Predict` with `.weight_map`, `.scores`, `.mean_score` |
| `'roi'` | `roi_mask=atlas` | `Predict.scores` = per-parcel accuracy vector (46 parcels on k50) |
| `'searchlight'` | `roi_mask=`, `radius_mm=` | `Predict.accuracy_map` = voxel-space `BrainData`, directly plottable |

Searchlight over a 6×6×6-voxel restricted mask ran in 14 s; whole-brain will need a compute budget (see §5).

---

## 1. Thresholding chapter — the rewrite

Current state: `Thresholding_Group_Analyses.py` uses `SimulateGrid` for the simulation half and `ttest(threshold_dict={'unc':…, 'fdr':…})` for the real-data half. `threshold_dict` is gone in 0.6, and `SimulateGrid.threshold_simulation(correction=)` only recognizes `'fdr'` — no permutation, no cluster path. So the chapter needs real work regardless of the new content.

**Proposed structure:**

1. **The problem** — keep the existing simulation showing the uncorrected false-positive explosion. Now quantifiable: **56% of null maps contain ≥1 false positive at p<.001**, against a theoretical 59.4%. Much stronger than the current qualitative treatment.
2. **Classical corrections** — Bonferroni and FDR on parametric p-values (6% and 6% measured). Keep `SimulateGrid`'s `threshold_type='q'` path.
3. **Permutation** — introduce sign-flipping with `one_sample_permutation_test`. Show the null. **Teach the `1/n_permute` resolution floor here** — it motivates everything that follows.
4. **Max-statistic FWE** — build it from `return_null=True` in three lines. Verify calibration live (4% measured). This is the conceptual payoff: one null for the *whole map*, not per voxel.
5. **Cluster inference** — smooth the grid first, and show the 0% → 6% contrast explicitly. Cluster-forming threshold, null of the largest cluster, permutation p-value. Connect to Eklund et al. 2016.
6. **Real data, production tools** — `nilearn.non_parametric_inference(threshold=, tfce=True)` on the localizer contrast maps; compare `logp_max_t` / `logp_max_size` / `logp_max_mass` / `logp_max_tfce`.
7. **Reporting** — `nltools.cluster_report(atlas=...)` for the labeled peak table; `threshold(cluster_threshold=N)` for display. Be explicit that this step is *descriptive* — it labels what survived, it does not correct.

**Guardrail**: never `return_null=True` on whole-brain data (§0). Steps 3–5 stay on the grid; step 6 uses nilearn on real data.

**Also fix**: `fdr()` returns `-1` when nothing survives, and the notebook must branch on it — currently it doesn't, and a no-survivors run would silently threshold at -1 (keeping everything). Good teachable moment about honest reporting of null results.

## 2. BrainCollection everywhere

Pattern, per the decision: **explicit loop once as the teaching version, then the collection version.** The loop stays only in `Group_Analysis.py` (where first-level modeling is being taught for the first time); everywhere else goes straight to the collection.

| Notebook | Current | Target |
|---|---|---|
| `Group_Analysis.py` | `for sub in tqdm(...)`: load → smooth → build dm → regress → write betas | Keep the loop, annotated as "what the collection does for you", then the `BrainCollection.from_paths(...).smooth(6).fit(...)` equivalent + `compute_contrasts` + `ttest` |
| `Thresholding_Group_Analyses.py` | `BrainData([BrainData(f) for f in con_file_list])` | `BrainCollection.from_paths(...)` → `.permutation_test()` |
| `Multivariate_Prediction.py` | stacks per-subject betas; `cv_dict={'subject_id':…}` | `.predict(cv='loso')` — **fixes the silent `groups=` bug** flagged in the migration doc |
| `RSA.py` | per-subject loop building per-ROI RDMs | `BrainCollection` + `distance(spatial_scale='roi')` |
| `Connectivity.py` | per-subject denoising | `BrainCollection.fit()` with a design-builder callable |

Document the three sharp edges from §0 where students first meet them: the `_DesignContext` contract, `from_paths` handing through raw design paths, and the `h5` extra.

## 3. Spatial cleanup

Replace masking-and-looping with `spatial_scale=` wherever it appears:

- `RSA.py` — per-ROI loop → `distance(spatial_scale='roi', roi_mask=atlas)` → `similarity(project=True)` (already scoped in the migration doc §5.7; carries the NaN-parcel caveat)
- `Multivariate_Prediction.py` — `data.apply_mask(motor).predict(...)` → `predict(spatial_scale='roi', roi_mask=...)`
- `Connectivity.py` — `expand_mask` + `extract_roi` loop → `extract_roi(mask=atlas)` directly
- `Parcellations.py` — drop the `BrainData(url).to_nifti()` round-trip; `fetch_resource()` returns a path nilearn takes directly
- All hard-coded Neurovault/GitHub atlas URLs → `fetch_resource()` (migration doc §6.1)

## 4. New chapter: spatial feature selection

Slots after `Multivariate_Prediction`, before `RSA`. Framing follows Jolly & Chang (2021): searchlight, ROI, and whole-brain are three points on **one spatial-scale axis**, not three unrelated techniques — and the choice encodes an assumption about where the information lives.

**Read the paper before drafting the prose** — this section should reflect its actual argument and results, not a generic summary. Open questions to settle from it: how it frames the bias–variance trade-off across scales, and what it concludes about searchlight's interpretational pitfalls.

Draft outline:

1. The question: *at what spatial scale is the information?* Same data, same classifier, three answers.
2. **Whole-brain** — `predict(spatial_scale='whole_brain')`. One model, all voxels. Maximal sensitivity to distributed codes, no localization.
3. **ROI** — `predict(spatial_scale='roi', roi_mask=atlas)`. One model per parcel → accuracy vector → paint back to the brain. Localizes, but inherits the atlas's assumptions.
4. **Searchlight** — `predict(spatial_scale='searchlight', radius_mm=)` → `accuracy_map`. Assumption-free localization at the cost of compute, plus the interpretational pitfalls the paper raises.
5. **Compare** — all three on the same localizer left-vs-right motor data, side by side. This is the chapter's payoff and the reason it should be standalone.
6. Cross-links: RSA (`distance(spatial_scale=)`), Parcellations (where atlases come from), Thresholding (searchlight maps need the same correction machinery — a natural callback).

**Compute budget is the main risk.** Whole-brain searchlight × ~25 subjects at build time may exceed what the ~75 min build tolerates. Options in order of preference: restrict to a gray-matter or ROI mask; coarsen `radius_mm`; precompute and cache the map as an artifact (`mode: cached`, as `Download_Data`/`Parcellations` already do). Measure before committing.

---

## 5. Sequencing

Phase 1 is a hard prerequisite — the notebooks don't currently run on 0.6 at all.

1. **Phase 1** — the API migration (separate plan). Everything else builds on it.
2. **Thresholding chapter** (§1). Self-contained, needs no dataset download for steps 1–5, and the statistical core is already verified.
3. **`Group_Analysis.py` BrainCollection** (§2). Establishes the pattern the other notebooks copy.
4. **Propagate BrainCollection** to Thresholding, Multivariate_Prediction, RSA, Connectivity.
5. **Spatial cleanup** (§3) — largely mechanical once §4's vocabulary is settled.
6. **New spatial feature selection chapter** (§4). Last: it depends on the collection pattern and on the spatial vocabulary being consistent, and it needs a compute-budget decision.
7. **Full build**, read every figure. The failure mode throughout is *quiet*: a plausible-but-different figure, not an exception.

### Open items to resolve during implementation

- Read Jolly & Chang (2021) properly before drafting §4 prose.
- Check whether `dartbrains_tools.bids` exposes a layout that makes `BrainCollection.from_bids` viable (cleaner than `from_paths`).
- Measure whole-brain searchlight cost on real data; decide restrict vs. cache.
- Confirm nilearn's `calculate_cluster_measures` `assert clust_vals[0] == 0` doesn't trip on the localizer data — it fired on a small synthetic volume with a strong effect. Needs a realistic mask.
- Verify the RSA per-ROI `similarity(..., method=None)` NaN edge (migration doc §5.7) on real betas.
- Stale local HF token still breaks `fetch_resource` on this machine (migration doc §7) — fix before testing anything that fetches.
