# Rule: Inter-Module Contract

## Scope
All code that passes data between modules or imports from other modules.

## Rules

### R3.1 — PipelinePacket is the only inter-module data format
Data between modules MUST pass through `PipelinePacket`. No module may bypass the packet and read another module's raw output files directly as its primary input path.

### R3.2 — Isolated imports for cross-module access
When `pipeline_main.py` imports from a module, it MUST use the `_isolated_import()` context manager. This prevents `config.py` namespace collision.

```python
# CORRECT
with _isolated_import(M3_DIR):
    from preprocessor import DataPreprocessor

# WRONG — config.py collision
sys.path.insert(0, str(M3_DIR))
from preprocessor import DataPreprocessor
```

### R3.3 — Respect the annotation routing
- `is_annotated=True` → training/evaluation path (Modules 3-8)
- `is_annotated=False` → deployment inference path (Module 9)

No code may send unannotated data into training or annotated data into deployment without explicit, logged override.

### R3.4 — Module output structure is a contract
Each module's output folder structure (documented in CLAUDE.md) is a contract that downstream modules depend on. Changing output file names, column names, or folder structure requires updating all downstream consumers.

### R3.5 — Each module must be independently runnable
Every module MUST work both as part of the pipeline (receiving PipelinePacket) AND standalone (receiving a folder path or dict). This enables isolated testing and debugging.

## Rationale
The 9-module sequential architecture depends on stable contracts between modules. Breaking a contract silently propagates errors through every downstream module, potentially corrupting months of analysis.
