# Introduction

Dew is a minimal expression language that compiles to multiple backends. Parse once, emit to WGSL/GLSL (GPU shaders), Cranelift (native JIT), or Lua (scripting).

Part of the [Rhizome](https://rhizome-lab.github.io) ecosystem.

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

All domain crates have multiple backends (feature flags): WGSL, GLSL, OpenCL, CUDA, HIP, Rust, C, Lua, and Cranelift.

**Core = syntax only, domains = semantics.** Each domain crate has its own:
- Value types and type system
- Function registry
- Self-contained backends (behind feature flags)

## Feature Flags

dew-core uses feature flags to manage complexity:

| Feature | Description |
|---------|-------------|
| (none)  | Basic expressions: numbers, variables, arithmetic |
| `cond`  | Conditionals: `if`/`then`/`else`, comparisons (`<`, `>=`, `==`), boolean logic (`and`, `or`, `not`) |
| `func`  | Function calls: `name(args...)` with extensible registry |

Domain crates automatically enable `func` (they rely on function calls).

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

### With Conditionals

When dew-core is compiled with the `cond` feature:

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::{eval, scalar_registry};

// Conditional expression
let expr = Expr::parse("if x > 0 then sqrt(x) else 0").unwrap();

let mut vars = std::collections::HashMap::new();
vars.insert("x".to_string(), 9.0_f32);

let result = eval(expr.ast(), &vars, &scalar_registry()).unwrap();
// result = 3.0
```

## Backends

Each domain crate includes multiple backends as optional features:

| Backend | Feature | Use case |
|---------|---------|----------|
| WGSL | `wgsl` | GPU shaders (WebGPU) |
| GLSL | `glsl` | GPU shaders (OpenGL/Vulkan) |
| OpenCL | `opencl` | GPU compute kernels (cross-platform) |
| CUDA | `cuda` | GPU compute kernels (NVIDIA) |
| HIP | `hip` | GPU compute kernels (AMD ROCm, source-compatible with CUDA) |
| Rust | `rust` | Rust source code generation |
| C | `c` | C source code generation |
| TokenStream | `tokenstream` | Proc-macro code generation |
| Lua | `lua` | Scripting, hot-reload (includes `lua-codegen` for WASM) |
| Cranelift | `cranelift` | Native JIT compilation |

Enable in `Cargo.toml`:

```toml
[dependencies]
rhizome-dew-scalar = { version = "0.1", features = ["wgsl", "glsl", "lua", "cranelift"] }
```

### C Backend Limitations

The C backend generates code that assumes external type and function definitions:

**Required type definitions** (user-provided):
- Scalars: `float` (standard C)
- Vectors: `vec2`, `vec3`, `vec4` (structs with `.x`, `.y`, `.z`, `.w` fields)
- Matrices: `mat2`, `mat3`, `mat4`
- Complex: `complex_t` (struct with `.re`, `.im` fields)
- Quaternions: `quat_t` (struct with `.x`, `.y`, `.z`, `.w` fields)

**Required function definitions** (user-provided):
- Vector ops: `vec2_add()`, `vec2_dot()`, `vec2_scale()`, `vec2_normalize()`, etc.
- Matrix ops: `mat2_mul()`, `mat2_mul_vec2()`, etc.
- Complex ops: `complex_mul()`, `complex_conj()`, `complex_exp()`, etc.
- Quaternion ops: `quat_mul()`, `quat_slerp()`, `quat_from_axis_angle()`, etc.

**What it uses from standard C**:
- `<math.h>`: `sinf`, `cosf`, `tanf`, `expf`, `logf`, `sqrtf`, `fabsf`, `floorf`, `ceilf`, `fminf`, `fmaxf`, `powf`, `fmodf`, `atan2f`
- Ternary operator for conditionals: `cond ? then : else`
- Float literals with `f` suffix: `1.0f`, `3.14159f`

The generated code is designed to be embedded in projects with existing math libraries (e.g., cglm, HandmadeMath, custom implementations). It provides the expression logic while you provide the type system.

## Crates

| Crate | Description |
|-------|-------------|
| `rhizome-dew-core` | Core AST and parsing (feature-gated conditionals and functions) |
| `rhizome-dew-cond` | Conditional backend helpers for domain crates |
| `rhizome-dew-scalar` | Scalar math: sin, cos, exp, lerp, etc. |
| `rhizome-dew-linalg` | Linear algebra: Vec2-4, Mat2-4, dot, cross, etc. |
| `rhizome-dew-complex` | Complex numbers: exp, log, polar, conjugate, etc. |
| `rhizome-dew-quaternion` | Quaternions: rotation, slerp, axis-angle, etc. |
