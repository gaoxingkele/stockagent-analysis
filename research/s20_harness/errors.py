"""Explicit validity failures, distinct from failed infrastructure or scientific no-gain."""


class DataValidityError(ValueError):
    """An invoked data verifier rejected its inputs or reconstructed evidence."""
