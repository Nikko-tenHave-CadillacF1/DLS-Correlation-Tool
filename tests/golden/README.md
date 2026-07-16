# Golden-output snapshots

Byte-oriented regression records for `Run_*.py` workflows. Each snapshot
captures SHA-256 hashes of every plot PNG and PowerPoint file the workflow
produces, so subsequent refactors can be verified as pixel/byte-identical
without a human eye-balling every output.

## What lives here

One `<runner>.json` per workflow, committed to git. Example:
`Run_Correlation.json` covers `Run_Correlation.py`.

## How PNGs are hashed

**We hash the pixel array, not the file bytes.** Matplotlib PNG output
embeds a build-time timestamp in a `tIME` metadata chunk and other
metadata that varies across `matplotlib` versions (and sometimes between
runs on the same version). Raw-file-byte hashes would false-positive on
every dependency bump.

The tool loads each PNG with Pillow, converts to `RGBA`, and hashes the
resulting NumPy array — this is deterministic across matplotlib versions
as long as the actual pixels rendered are identical.

## How PPTX is hashed

**Content-hash of ZIP entries, excluding volatile metadata parts.** A
`.pptx` is an OOXML ZIP archive. python-pptx rewrites two parts on every
save regardless of whether the content changed:

- `docProps/core.xml` embeds `dcterms:modified` (wall-clock timestamp).
- `docProps/app.xml` embeds an edit-count.

Raw-byte hashing would therefore false-positive on every re-run. The
harness unzips the pptx, computes SHA-256 of each entry's bytes,
excludes those two parts, and hashes the sorted
`{filename → sha256}` table. This measures actual slide layout, media,
and text — which is what a golden-output check should care about.

If a future python-pptx version starts writing additional metadata
timestamps into other parts (e.g. custom XML in `docProps/custom.xml`),
add them to `_PPTX_VOLATILE_PARTS` in `tools/golden_regression.py`. The
symptom is a fresh snapshot immediately failing verification.

## Regeneration

Snapshots are committed to source control. Regenerating them is an
explicit, reviewable action:

```powershell
python tools/golden_regression.py snapshot Run_Correlation
```

The resulting JSON diff should be reviewed the same way a source-code
change would be — a hash change means a plot changed, and that change
needs to be intentional. If it's not, the refactor that caused it needs
investigation.

## Verifying

```powershell
python tools/golden_regression.py verify Run_Correlation
```

Exits 0 if every hash matches the snapshot; exits 1 with a list of NEW /
MISSING / CHANGED files otherwise.

## Reproducibility contract

Two consecutive `snapshot` invocations on the same host, same versions,
and same input files must produce byte-identical JSON. If they don't,
the regression harness itself is broken and needs fixing before it can
be used to gate refactors. Report any non-determinism rather than
loosening the hash comparison.
