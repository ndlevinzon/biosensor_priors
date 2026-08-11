"""Gate: Q324R present in experimental database."""


def test_q324r_present(stage0_result) -> None:
    _, meta = stage0_result
    checks = {c["name"]: c for c in meta["gate"]["checks"]}
    assert "Q324R" in checks["required_control_mutations"]["found"]
    assert checks["required_control_mutations"]["passed"]
