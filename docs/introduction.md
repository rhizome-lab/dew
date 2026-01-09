# Introduction

Dew is an expression language for procedural generation. It provides composable math expressions that compile to multiple backends.

Part of the [Rhizome](https://rhizome-lab.github.io) ecosystem.

## Architecture

```
dew-core               # Syntax only: AST, parsing
    |
    +-- dew-scalar     # Scalar domain: f32/f64 math functions
    |
    +-- dew-linalg     # Linalg domain: Vec2, Vec3, Mat2, Mat3
    |
    +-- dew-complex    # Complex numbers: [re, im]
    |
    +-- dew-quaternion # Quaternions: [x, y, z, w], Vec3
```

All domain crates have WGSL, Lua, and Cranelift backends (feature flags).

**Core = syntax only, domains = semantics.** Each domain crate has its own:
- Value types and type system
- Function registry
- Self-contained backends (behind feature flags)

## Quick Example

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::{eval, scalar_registry};

// Parse an expression
let expr = Expr::parse("sin(x) + cos(y)").unwrap();

// Evaluate with variables
let mut vars = std::collections::HashMap::new();
vars.insert("x".to_string(), 0.5_f32);
vars.insert("y".to_string(), 1.0_f32);

let registry = scalar_registry();
let result = eval(expr.ast(), &vars, &registry).unwrap();
```

## Backends

Each domain crate includes three backends as optional features:

| Backend | Feature | Use case |
|---------|---------|----------|
| WGSL | `wgsl` | GPU shaders (WebGPU) |
| Lua | `lua` | Scripting, hot-reload |
| Cranelift | `cranelift` | Native JIT compilation |

Enable in `Cargo.toml`:

```toml
[dependencies]
rhizome-dew-scalar = { version = "0.1", features = ["wgsl", "lua", "cranelift"] }
```

## Crates

| Crate | Description |
|-------|-------------|
| `rhizome-dew-core` | Core AST and parsing |
| `rhizome-dew-scalar` | Scalar math: sin, cos, exp, lerp, etc. |
| `rhizome-dew-linalg` | Linear algebra: Vec2-4, Mat2-4, dot, cross, etc. |
| `rhizome-dew-complex` | Complex numbers: exp, log, polar, conjugate, etc. |
| `rhizome-dew-quaternion` | Quaternions: rotation, slerp, axis-angle, etc. |
