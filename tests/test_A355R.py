"""Gate: A355R present in experimental database."""


def test_a355r_present(stage0_result) -> None:
    _, meta = stage0_result
    checks = {c["name"]: c for c in meta["gate"]["checks"]}
    assert "A355R" in checks["required_control_mutations"]["found"]
    assert checks["required_control_mutations"]["passed"]
