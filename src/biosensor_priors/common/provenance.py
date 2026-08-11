"""Stage manifests and provenance recording.

Each stage writes a ``manifest.json`` capturing inputs, content hashes,
parameters, software/tool versions, random seeds, output paths, and whether
its verification gate passed. Manifests make analyses reconstructable.
"""
