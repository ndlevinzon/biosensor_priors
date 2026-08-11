# Identifier system

One master identifier vocabulary is used across wet lab, structures, physics,
models, and search.

| Identifier | Meaning |
| --- | --- |
| ``construct_id`` | Unique experimental / design construct |
| ``version`` | Sequence background (e.g. ``V2.4``) |
| ``parent_version`` | Immediate parent background |
| ``canonical_position`` | Position in the canonical numbering (e.g. relative to ``V1.0``) |
| ``version_position`` | Position in a specific version's sequence |
| ``mutation`` | Mutation string (e.g. ``Q324R``) using canonical numbering unless stated |
| ``experimental_round`` | Wet-lab campaign round |
| ``structure_model_id`` | One predicted structure (method × seed × state × version) |
| ``physics_scan_id`` | One RIF/RPX scan run / batch |
| ``conformer_id`` | Permanent ligand conformer identity |
| ``model_run_id`` | One Stage-3 surrogate fit |
| ``candidate_id`` | One Stage-4 design-space member |
| ``split_id`` | Frozen train/test split file |

IDs appear in parquet tables, manifests, job scripts, and prediction freezes so
joins never rely on fragile filename parsing alone.
