"""Pure CSV/parquet loader helpers extracted from :mod:`engine.dataplotter`.

The functions here own no state: everything they need is passed explicitly
by the caller. :class:`DataPlotter` still exposes each function as a thin
method delegator so existing internal call sites (``self._load_run_data(...)``
etc.) keep working unchanged.

Extracted 2026-07 as part of the second refactor phase (context + loaders).
"""

from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd

from ..logger import log


def _normalize_parquet_column_aliases(df):
    raw_columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    existing = set(raw_columns)
    for col in raw_columns:
        if col.startswith("_") and len(col) > 1 and col[1].isalpha():
            canonical = col[1].upper() + col[2:]
            if canonical not in existing:
                rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    # Cast bool columns to float. Some parquet sources (notably AVL-TR OC
    # exports) store flag channels like `_bGripLimited` as dtype=bool. That
    # causes two silent bugs downstream:
    #   1) scipy.integrate.cumulative_trapezoid uses boolean arithmetic on
    #      bool inputs (True + True == True), which halves the integrated
    #      time in derived channels like `time_grip_limited`.
    #   2) pandas raises LossySetitemError when NaN is written into a bool
    #      Series (e.g. `mask_waveform_discontinuities`), breaking waveform
    #      plots that reference these channels.
    # Float (not int) is required because NaN cannot be represented in int.
    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    if bool_cols:
        df = df.astype({c: "float64" for c in bool_cols}, copy=False)
    return df


def _resolve_required_parquet_columns(
    schema_cols,
    columns_to_load,
    nrun=None,
    nlap=None,
    *,
    alias_cache: dict | None = None,
):
    raw_set = set(schema_cols)
    raw_lower = {c.lower(): c for c in schema_cols}
    schema_key = tuple(schema_cols)
    canonical_to_raw = None if alias_cache is None else alias_cache.get(schema_key)
    if canonical_to_raw is None:
        canonical_to_raw = {}
        for raw in schema_cols:
            if raw.startswith("_") and len(raw) > 1 and raw[1].isalpha():
                canonical = raw[1].upper() + raw[2:]
                # Also expose the camelCase form so users can write
                # 'nRun' when the parquet column is '_nRun'.
                camelcase = raw[1].lower() + raw[2:]
                canonical_to_raw.setdefault(camelcase, raw)
            else:
                canonical = raw
            canonical_to_raw.setdefault(canonical, raw)
            canonical_to_raw.setdefault(raw, raw)
        if alias_cache is not None:
            alias_cache[schema_key] = canonical_to_raw
    needed = set()
    for candidates, flag in [
        (["nRun", "nrun", "_nRun", "_nrun", "NRun"], nrun),
        (["nLap", "nlap", "_nLap", "_nlap", "NLap"], nlap),
    ]:
        if flag is not None:
            for c in candidates:
                if c in raw_set:
                    needed.add(c)
                    break
            else:
                target_lower = candidates[0].lower()
                if target_lower in raw_lower:
                    needed.add(raw_lower[target_lower])
    if columns_to_load:
        for logical in columns_to_load:
            if logical in canonical_to_raw:
                needed.add(canonical_to_raw[logical])
            elif logical in raw_set:
                needed.add(logical)
            else:
                lower = logical.lower()
                if lower in raw_lower:
                    needed.add(raw_lower[lower])
    return sorted(needed) if needed else None


def _load_parquet_with_fallback(
    file_path,
    columns_to_load: Iterable[str] | None = None,
    parquet_nrun=None,
    parquet_nlap=None,
    run_name: str = "",
    *,
    available_engines: list[str],
    get_schema_columns: Callable,
    apply_rank_value_filter: Callable,
    alias_cache: dict,
    verbose: bool,
):
    if not available_engines:
        raise ImportError("Parquet input requires 'pyarrow' or 'fastparquet', but neither is installed.")
    errors = []
    for engine in available_engines:
        try:
            schema_cols = get_schema_columns(file_path, engine)
            if schema_cols is not None and columns_to_load:
                col_subset = _resolve_required_parquet_columns(
                    schema_cols,
                    columns_to_load,
                    nrun=parquet_nrun,
                    nlap=parquet_nlap,
                    alias_cache=alias_cache,
                )
            else:
                col_subset = None
            read_kwargs = {"engine": engine}
            if col_subset:
                read_kwargs["columns"] = col_subset
            if parquet_nrun is not None and parquet_nlap is not None:
                log.info(
                    "Run '%s' provided both nrun and nlap; applying nrun filter and ignoring nlap.",
                    run_name.upper() if run_name else file_path.name,
                )
            df = pd.read_parquet(file_path, **read_kwargs)
            df.columns = [str(c).strip() for c in df.columns]
            df = _normalize_parquet_column_aliases(df)
            if parquet_nrun is not None:
                df = apply_rank_value_filter(
                    df,
                    filter_spec=parquet_nrun,
                    column_logical_name="nRun",
                    file_path=file_path,
                    run_name=run_name,
                    is_rank=True,
                    raise_on_missing_column=True,
                    raise_on_empty_result=True,
                )
            elif parquet_nlap is not None:
                df = apply_rank_value_filter(
                    df,
                    filter_spec=parquet_nlap,
                    column_logical_name="nLap",
                    file_path=file_path,
                    run_name=run_name,
                    is_rank=False,
                    raise_on_missing_column=False,
                    raise_on_empty_result=False,
                )
            if columns_to_load:
                requested = sorted(set(columns_to_load))
                df_cols = list(df.columns)
                df_cols_set = set(df_cols)
                df_cols_lower = {str(c).lower(): c for c in df_cols}
                available = []
                missing = []
                seen_avail = set()
                for c in requested:
                    hit = None
                    if c in df_cols_set:
                        hit = c
                    elif c.startswith("_") and len(c) > 1 and c[1].isalpha() and (c[1].upper() + c[2:]) in df_cols_set:
                        hit = c[1].upper() + c[2:]
                    elif len(c) > 1 and c[0].isalpha() and c[0].islower() and (c[0].upper() + c[1:]) in df_cols_set:
                        # e.g. user asked for 'nRun'; parquet column
                        # normalised from '_nRun' is 'NRun'.
                        hit = c[0].upper() + c[1:]
                    elif c.lower() in df_cols_lower:
                        # Case-insensitive fallback (e.g. user 'nRun',
                        # parquet column 'nrun' with no underscore).
                        hit = df_cols_lower[c.lower()]
                    if hit is None:
                        missing.append(c)
                        continue
                    if hit not in seen_avail:
                        available.append(hit)
                        seen_avail.add(hit)
                if missing and verbose:
                    log.debug(
                        "Parquet '%s' missing %d channel(s): %s%s",
                        file_path.name,
                        len(missing),
                        ", ".join(missing[:10]),
                        " ..." if len(missing) > 10 else "",
                    )
                if available:
                    df = df[available]
                else:
                    raise KeyError(f"No requested channels found in parquet. Requested: {requested[:10]}")
            return df
        except Exception as exc:
            errors.append(f"{engine}: {exc}")
    raise RuntimeError(
        f"Unable to load parquet '{file_path}' via engines {available_engines}. Errors: {' | '.join(errors)}"
    )


def _load_run_data(
    file_path,
    use_python_engine: bool = False,
    columns_to_load: Iterable[str] | None = None,
    parquet_nrun=None,
    parquet_nlap=None,
    run_name: str = "",
    *,
    load_parquet: Callable,
    make_unique: Callable,
):
    try:
        if file_path.suffix.lower() == ".parquet":
            df = load_parquet(
                file_path,
                columns_to_load=columns_to_load,
                parquet_nrun=parquet_nrun,
                parquet_nlap=parquet_nlap,
                run_name=run_name,
            )
            df.columns = make_unique([str(c).strip() for c in df.columns])
            units = {c: "" for c in df.columns}
            return df, df.columns, units
        with open(file_path) as f:
            lines = f.readlines()
        header = make_unique(lines[1].strip().split(","))
        units_row = lines[2].strip().split(",")
        units = dict(zip(header, units_row))
        if columns_to_load:
            cols_to_read = [c for c in header if c in set(columns_to_load)]
        else:
            cols_to_read = None
        kwargs = dict(
            sep=",",
            skiprows=3,
            header=None,
            names=header,
            on_bad_lines="skip",
            usecols=cols_to_read,
        )
        if use_python_engine:
            kwargs["engine"] = "python"
        else:
            kwargs["low_memory"] = False
        df = pd.read_csv(file_path, **kwargs)
        units = {c: units.get(c, "") for c in df.columns}
        return df, df.columns, units
    except Exception as e:
        log.error("Failed to load '%s': %s", file_path, e)
        raise
