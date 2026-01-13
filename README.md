# uneval

Declarative Python source code generation from objects.

## Quick start

```python
from uneval import Assignment, generate_python_source

code = generate_python_source(Assignment("config", config_obj), clean_mode=True)
```
