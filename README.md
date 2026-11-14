---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

<!-- dataset-factory:links -->
**Dataset Factory · v-0d2e623e604ecf78**

[Dataset website](https://earino.github.io/factory-sbir-phase1-to-phase2-conversion/versions/v-0d2e623e604ecf78/) · [GitHub release](https://github.com/earino/factory-sbir-phase1-to-phase2-conversion/releases/tag/v-0d2e623e604ecf78) · [Hugging Face](https://huggingface.co/datasets/earino/factory-sbir-phase1-to-phase2-conversion/tree/v-0d2e623e604ecf78) · [Catalogue](https://earino.github.io/dataset-factory-site/)

Artifact: a1f22706a52ba521dcb070d0a85877c552f5c62e8752af292e4a31ec27fdbe72
<!-- /dataset-factory:links -->

# Stage-gate conversion of first-time SBIR/STTR Phase I awards to Phase II: will a new awardee firm advance within four years?

Will a firm's first-ever SBIR/STTR Phase I award convert into a Phase II award from the same agency within four fiscal years?

Derived from U.S. Small Business Administration (SBIR.gov). This is not its official product.

Prediction time: The Phase I award date: proposal_award_date when published and inside the federal fiscal year of award_year; otherwise the fiscal year end of award_year. Only fields on the index Phase I award record are read.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.6206483231470346,
    "brier": 0.23681453246956263
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.24661408213974287
  },
  "calendar_identity": {
    "auc": 0.5030937303126253,
    "brier": 0.2508945361309057
  }
}
```

## Limitations

- SBIR.gov is agency-reported and uneven across agencies and years; award_amount and number_employees are as-published values that may have been corrected after the award, so this is a current-vintage retrospective extract and not a snapshot of what an analyst saw on the award date.
- There is no parent key linking a Phase II to its Phase I, so conversion is inferred from firm identifier, agency and award-year window; a small share of labels will be misassigned, for example when a repeat awardee's later Phase II is credited to the wrong Phase I.
- Firm identity depends on UEI, DUNS or name matching, and DUNS is an optional field in the dictionary, so some firms may be merged or split across identifiers.
- HHS awards omit proposal_award_date by design ('Yes, Except HHS' in the dictionary), so the decision instant is a fiscal year rather than an exact date for NIH awardees.
- 17 U.S.C. §105(a) is territorial: it removes copyright within the United States only and does not establish public-domain status elsewhere; downstream reuse outside the US needs its own analysis.
- Award titles and abstracts are company-authored text and are excluded from the release rather than relicensed.
- Phase III commercialisation data sit in a government-only database under the Policy Directive, so the observable horizon stops at the Phase II stage gate.
- Firms whose first Phase I award falls after 2018 are excluded because their four-year outcome window is not closed at the freeze.
- Annual award records are a current-vintage administrative extract; later corrections can revise earlier years, so historical availability is not proven.
- The shipped time column award_date uses Proposal Award Date when it falls inside the award_year federal fiscal year; HHS awards do not publish that date, so the fiscal year end is used. Those rows share a fiscal-year-end event cluster and exact HHS award dates are not observed.
- The published state column mixes two-letter USPS codes with full state and territory names; the builder canonicalizes both to codes, so the state feature is a normalized location rather than a verbatim copy of the source field.
- No parent key links a Phase II award to its Phase I award; conversion is inferred by firm key and agency within four award years, so some labels may be misassigned.
- Firm identity uses UEI, else DUNS, else normalized company name plus state; optional identifiers and name changes can merge or split firms.
- HHS awards may omit proposal_award_date; the index time is the federal award year, and award_date falls back to fiscal year end rather than an exact date.
- Same-year multiple Phase I awards are included only when proposal dates identify a unique earliest record; otherwise the firm is excluded rather than given an arbitrary row.
- Phase II records with missing agency or an unknown or missing phase in a label window are excluded, not treated as negative.
- No label-derived group rates or persistence reference are shipped; each firm appears once.
- The target is an award-record event, not the underlying cause of non-conversion; voluntary exit and temporary suspension are not separately observed.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
