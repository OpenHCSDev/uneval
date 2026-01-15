---
title: 'uneval: Declarative Python Source Serialization with Automatic Import Resolution'
tags:
  - Python
  - serialization
  - code generation
  - reproducibility
  - configuration
authors:
  - name: Tristan Simas
    orcid: 0000-0002-6526-3149
    affiliation: 1
affiliations:
  - name: McGill University
    index: 1
date: 13 January 2026
bibliography: paper.bib
---

# Summary

`uneval` converts in-memory Python objects into executable Python source code with correct imports. Given a dataclass with nested Enums, Paths, and callables, `uneval` produces a complete Python script:

```python
# Input: in-memory object
config = PipelineConfig(
    path_planning=PathPlanningConfig(output_dir_suffix="_custom"),
    dtype=DtypeConversion.PRESERVE_INPUT,
)

# Output: executable Python source
from openhcs.constants import DtypeConversion
from openhcs.core.config import PathPlanningConfig, PipelineConfig

config = PipelineConfig(
    path_planning=PathPlanningConfig(output_dir_suffix="_custom"),
    dtype=DtypeConversion.PRESERVE_INPUT,
)
```

The output is diffable, reviewable, and can be `exec()`'d to recreate the original object. This enables round-trip editing: a GUI serializes configuration to Python, users edit the code manually, then reload it into the GUI.

# Statement of Need

Binary serializers like `pickle` [@pickle] and `dill` [@dill] are compact but opaque and break across Python versions. JSON/YAML lose type information and require custom loaders. `repr()` produces partial code without imports. None handle name collisions (e.g., two classes named `Config` from different modules).

`uneval` addresses these gaps with a two-pass algorithm:

1. **Format pass**: Traverse the object, collecting code fragments and required imports
2. **Import resolution**: Detect collisions, generate aliases (e.g., `from module_a import Config as Config_a`)
3. **Regeneration pass**: Re-format with resolved aliases to produce final code

This two-pass design is essential—import aliases cannot be determined until all types are visited.

# State of the Field

The standard library `pickle` [@pickle] and `dill` [@dill] serialize to binary. `cloudpickle` [@cloudpickle] handles dynamic functions but remains non-diffable. Dataclasses [@pep557] simplify structured data but provide no code generation. `uneval` complements these by emitting executable source with deterministic imports.

# Software Design

The architecture uses a pluggable formatter registry with auto-registration:

```python
class EnumFormatter(SourceFormatter):
    priority = 70

    def can_format(self, value: Any) -> bool:
        return isinstance(value, Enum)

    def format(self, value: Enum, context: FormatContext) -> SourceFragment:
        cls = value.__class__
        import_pair = (cls.__module__, cls.__name__)
        name = context.name_mappings.get(import_pair, cls.__name__)
        return SourceFragment(f"{name}.{value.name}", frozenset([import_pair]))
```

Formatters register themselves via `__init_subclass__`—no manual registry updates required. Adding support for a domain type requires only defining a new formatter class.

**Clean mode** omits fields matching defaults, producing concise output. **Explicit mode** includes all fields for complete reproducibility.

Key components:

- `SourceFragment(code, imports)`: atomic serialization result
- `FormatContext`: tracks indentation, clean mode, and resolved name mappings
- `SourceFormatter`: ABC with priority-based selection and auto-registration
- `resolve_imports()`: deterministic collision handling via aliasing
- Helper nodes (`Assignment`, `CodeBlock`, `Comment`, `BlankLine`) for composing scripts

# Research Impact Statement

`uneval` powers code-based serialization in OpenHCS [@openhcs], enabling:

- **GUI round-trip editing**: Pipeline Editor serializes steps to Python; users edit code directly; changes reload into the GUI via `pyqt-formgen` [@pyqtformgen]
- **Remote execution**: ZMQ clients serialize pipeline configurations as Python code, avoiding pickle versioning issues across nodes
- **Reproducibility**: Pipeline scripts are human-readable records of exact processing parameters

The formatter registry enables domain extensions without modifying uneval's core. OpenHCS adds formatters for:

- **FunctionStep**: Pipeline step objects with function patterns and processing configuration
- **Virtual module rewrites**: External library functions (e.g., `skimage.filters.gaussian`) are rewritten to virtual module paths (`openhcs.skimage.filters.gaussian`) that include OpenHCS decorators
- **Lazy dataclass bypass**: For dataclasses with `__getattribute__` interception (used for hierarchical config inheritance), formatters use `object.__getattribute__` to access raw field values without triggering lazy resolution

# AI Usage Disclosure

Generative AI assisted with drafting documentation. All content was reviewed and tested by human developers.

# Acknowledgements

This work was supported by [TODO: Add funding sources].

# References
