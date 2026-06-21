"""Domain exceptions for the extraction package.

IncompatibleSchemaError: raised when a metrics.json file's `schema_version`
does not match a version this CLI build can consume. Per user decision
2026-06-21, schema breaks are non-silent — readers must fail loudly and
ask the user to re-extract with the current CLI rather than silently
degrading to a partial interpretation of the old schema.
"""

from __future__ import annotations


class IncompatibleSchemaError(Exception):
    """metrics.json schema_version is not supported by this CLI build.

    Example::

        raise IncompatibleSchemaError(
            f"schema_version={version!r} not supported; "
            f"re-extract with current CLI (this build reads 1.2)"
        )
    """
