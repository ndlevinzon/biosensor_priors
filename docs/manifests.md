# Manifests and provenance

Every stage writes a small ``manifest.json`` under ``manifests/`` (and may
mirror a copy beside its primary outputs).

## Required fields

| Field | Description |
| --- | --- |
| ``stage`` | Stage name / version of the stage contract |
| ``created_at`` | UTC timestamp |
| ``inputs`` | Paths + content hashes of input artifacts |
| ``parameters`` | Resolved config snapshot (not just a path) |
| ``software`` | Package and external tool versions |
| ``random_seed`` | Seed(s) used |
| ``outputs`` | Paths + hashes of produced artifacts |
| ``gate`` | ``passed`` / ``failed`` / ``skipped`` + details |
| ``notes`` | Optional free text |

## Why this matters

- Reconstruct any figure or table from hashes + config
- Prove predictions were frozen before wet-lab return (Stage 5)
- Block silent use of failed Gate-2 physics in Stage 3
- Enable Stage 6 ablations with identical provenance trails

Implementation lives in ``biosensor_priors.common.provenance``.
