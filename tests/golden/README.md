# tests/golden

Reference data for regression checks.

| File | Used by |
|---|---|
| `coverage_baseline.xml` | The CI `coverage-guard` job |
| `<runner>.json` (optional) | [tools/golden_regression.py](../../tools/golden_regression.py) |

## Coverage baseline

CI runs `pytest`, which writes `coverage.xml`. The job fails if any file's line
coverage drops more than 2.0 percentage points below this baseline, or if a
file in the baseline is missing from the new report. If the drop is
intentional, update the baseline in the same PR:

```powershell
pytest
Copy-Item coverage.xml tests\golden\coverage_baseline.xml
```

## Output snapshots

`golden_regression.py` runs a `Run_*.py` headlessly with `--no-open` and
interactive windows suppressed. It then hashes every PNG under
`Data/outputs/<workflow>/<event>/plots/` and every deck the runner exports.

```powershell
python tools/golden_regression.py snapshot Run_Correlation   # writes tests/golden/Run_Correlation.json
python tools/golden_regression.py verify   Run_Correlation   # exit 0 = match, 1 = NEW / MISSING / CHANGED files listed
python tools/golden_regression.py list
```

No snapshot is committed at the moment. A snapshot depends on the local
`Data/` inputs, which git ignores, so it's only valid on the machine that
recorded it. Check that the input data hasn't changed before you trust a
`verify` result.

How files are hashed:

- **PNG**: the decoded RGBA pixel array, so matplotlib metadata and timestamps
  don't affect the hash.
- **PPTX**: each ZIP entry, excluding `docProps/core.xml` (save time) and
  `docProps/app.xml` (edit count). If a newer python-pptx writes other
  volatile parts, add them to `_PPTX_VOLATILE_PARTS`.

Two snapshots taken on the same machine with the same inputs must be identical.
If they aren't, fix the harness rather than loosening the comparison.
