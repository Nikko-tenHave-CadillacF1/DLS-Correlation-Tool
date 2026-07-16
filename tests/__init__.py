"""Test suite for DLS-Correlation-Tool.

These are characterization tests targeting the pure-Python cores of the
engine (dataclass validation, gate mask logic, standalone fit kernel).
They intentionally avoid touching `channel_config` or any real telemetry
files so they run in CI without the workflow layer.
"""
