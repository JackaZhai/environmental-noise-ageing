# Package preparation

Prepared on 2026-10-18 from the supplied `Code_for_Peer_Review` package dated 2026-10-16.

The original download was preserved. This copy uses the repository name `environmental-noise-ageing`. The README was reorganized and includes the author-supplied Abstract, with only whitespace normalized for Markdown. A usage guide and script/export inventory were added, the Git ignore rules were expanded for local files, and the SHA-256 distribution manifest was regenerated with portable forward-slash paths.

The Python source, analysis configurations, saved prompt, R helpers, synthetic examples, existing tests, and review-only notice retain their original contents. The declared pandas and statsmodels dependencies now select the 2.x and 0.14.x series, respectively. A trial with statsmodels 0.15.0 failed at the original spline code's access to `model.data.design_info`; these version bounds retain the original statistical implementation. Tested direct dependency versions are recorded in `requirements-tested.txt`. The package retains its review-only distribution terms.

Validation results are recorded in `docs/validation.md`. Generated smoke outputs and test fixtures are local verification artifacts and are excluded from the distribution archive.

A subsequent UK Biobank data-safety audit added `docs/ukb_data_safety.md`, `scripts/audit_distribution.py`, stronger ignore rules, and a README data-protection section. These additions preserve the original analysis implementation.
