"""Smoke test: verify the package is importable."""


def test_import_oidpipe():
    import oidpipe

    assert hasattr(oidpipe, "__doc__")


def test_import_subpackages():
    import oidpipe.ingest
    import oidpipe.db
    import oidpipe.features
    import oidpipe.anomaly
    import oidpipe.agent
    import oidpipe.pipeline

    assert oidpipe.pipeline.__doc__ is not None
