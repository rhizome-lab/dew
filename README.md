# Dew

Minimal expression language, multiple backends.

Part of the [Rhizome](https://rhizome-lab.github.io) ecosystem.

## Overview

Dew is a minimal expression language that compiles to multiple backends. Small, ephemeral, perfectly formed—like a droplet condensed from logic. Parse once, emit to WGSL (GPU shaders), Cranelift (native JIT), or Lua (scripting).

## Crates

| Crate | Description |
|-------|-------------|
| `rhizome-dew-core` | Core AST and parsing (feature-gated conditionals and functions) |
| `rhizome-dew-cond` | Conditional backend helpers for domain crates |
| `rhizome-dew-scalar` | Scalar math: sin, cos, exp, lerp, etc. |
| `rhizome-dew-linalg` | Linear algebra: Vec2-4, Mat2-4, dot, cross, etc. |
| `rhizome-dew-complex` | Complex numbers: exp, log, polar, conjugate, etc. |
| `rhizome-dew-quaternion` | Quaternions: rotation, slerp, axis-angle, etc. |

Each domain crate includes self-contained backends (feature flags):
- `wgsl`: WGSL shader code generation
- `lua`: Lua code generation + mlua execution
- `cranelift`: Cranelift JIT native compilation

## Architecture

```
dew-core               # Syntax only: AST, parsing
    |
    +-- dew-cond       # Conditional backend helpers
    |
    +-- dew-scalar     # Scalar domain: f32/f64 math functions
    |
    +-- dew-linalg     # Linalg domain: Vec2, Vec3, Mat2, Mat3
    |
    +-- dew-complex    # Complex numbers: [re, im]
    |
    +-- dew-quaternion # Quaternions: [x, y, z, w], Vec3
```

Domain crates are independent. Each has:
- Generic over numeric type `T: Float`
- Own `FunctionRegistry<T>` and `eval<T>()`
- Self-contained backend modules

## License

MIT
