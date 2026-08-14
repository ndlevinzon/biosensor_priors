# Stage 2 ligand inputs (durable - not wiped by clean_pipeline_artifacts).
#
# Layout (per ligand):
#   data/ligands/AcCoA/ligand.smi      # one SMILES string (first non-comment line)
#   data/ligands/AcCoA/starting.mol2   # optional 3D start (also .sdf / .mol / .pdb)
#   data/ligands/PropCoA/...
#
# Priority for 3D start:
#   1. configs/physics.yaml ligands.starting_structures.<name> (if path exists)
#   2. data/ligands/<name>/starting.{mol2,sdf,mol,pdb}
#
# Priority for SMILES:
#   1. configs/physics.yaml ligands.smiles.<name> (if non-null)
#   2. data/ligands/<name>/ligand.smi
#
# Drop your files here, then run biosensor-stage2 (backend: external for real RDKit).
#
# Note: some CoA ``starting.mol2`` files fail RDKit kekulization on the adenine
# ring. Conformer generation then falls back to ``ligand.smi`` (ETKDG still
# rebuilds 3D coordinates). Prefer a sanitized SDF if you need a specific pose.
