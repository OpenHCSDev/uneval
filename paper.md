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

`uneval` converts in-memory Python objects into executable Python source code with correct imports. It targets configuration, reproducibility, and round-trip editing workflows where human-readable code is preferable to binary serialization. Given a Python object, `uneval` produces a full source fragment and the imports it requires, resolving name collisions in a second pass:

```python
from uneval import Assignment, generate_python_source

code = generate_python_source(Assignment("config", config_obj), clean_mode=True)
```

This enables tools to serialize complex nested structures (e.g., dataclasses with Enums and Paths) into editable Python scripts that can be executed to recreate the original objects.

# Statement of Need

Binary serializers such as `pickle` and `dill` [@pickle; @dill] are compact but opaque and can break across Python versions or library changes. Text-based formats (JSON/YAML) often lose type information and require custom loaders. Existing approaches like `repr()` or dataclass-specific helpers produce partial code, but do not track required imports, resolve name collisions, or support extension to arbitrary types.

`uneval` addresses these gaps by:

- Generating **valid Python source** with imports for all referenced types
- Providing a **pluggable formatter registry** to support domain-specific objects
- Supporting **clean vs. explicit** output modes for concise or fully-specified configs
- Producing code that is **diffable, reviewable, and executable**

# State of the Field

The standard library `pickle` module [@pickle] and `dill` [@dill] serialize Python objects into binary formats. `cloudpickle` [@cloudpickle] improves support for dynamically defined functions but remains binary and non-diffable. Dataclasses [@pep557] simplify structured data but do not provide code generation with import management. `uneval` complements these tools by emitting executable Python source with deterministic imports and a declarative extension mechanism.

# Software Design

`uneval` implements a two-pass source serialization pipeline:

1. **Format pass**: a `SourceFormatter` registry selects a formatter for each value and emits a `SourceFragment` containing code and required imports.
2. **Import resolution**: collisions are detected and resolved by aliasing, producing a name mapping.
3. **Regeneration pass**: formatters re-run with resolved names to produce final code.

Key components:

- `SourceFragment(code, imports)`: atomic serialization result
- `FormatContext`: tracks indentation, clean mode, and resolved name mappings
- `SourceFormatter`: ABC with auto-registration via `__init_subclass__`
- `resolve_imports()`: deterministic collision handling and import line generation
- Helper nodes (`Assignment`, `CodeBlock`, `Comment`, `BlankLine`) for composing full scripts

The registry makes it trivial to extend `uneval` for domain objects by adding a formatter class.

# Research Impact Statement

`uneval` powers code-based serialization in OpenHCS, enabling UI editors and remote execution to round-trip complex pipeline objects through editable Python scripts. The declarative formatter approach eliminated thousands of lines of repetitive code generation logic in favor of a small, extensible core. The library is applicable to any Python project that needs readable, version-resilient serialization with accurate imports.

# AI Usage Disclosure

Generative AI assisted with drafting documentation. All content was reviewed and tested by human developers.

# Acknowledgements

This work was supported by [TODO: Add funding sources].

# References
