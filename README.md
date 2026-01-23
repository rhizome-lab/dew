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

**GPU Shaders:**
- `wgsl`: WGSL (WebGPU)
- `glsl`: GLSL (OpenGL/Vulkan)

**GPU Kernels:**
- `opencl`: OpenCL (cross-platform GPU compute)
- `cuda`: CUDA (NVIDIA GPUs)
- `hip`: HIP (AMD ROCm, source-compatible with CUDA)

**Text Codegen:**
- `rust`: Rust source code
- `c`: C source code (embeddable, uses math.h)
- `tokenstream`: Rust TokenStream for proc-macros

**JIT & Scripting:**
- `lua`: Lua code generation + mlua execution (`lua-codegen` for WASM compatibility)
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
- Generic over numeric type `T: Numeric` (supports f32, f64, i32, i64)
- Own `FunctionRegistry<T>` and `eval<T>()`
- Self-contained backend modules

## Status & Roadmap

### Production Ready

**Core Language:**
- Expression AST with let bindings
- Conditionals (Compare, And, Or, If) - feature-gated
- Function calls - feature-gated
- Expression optimization (constant folding, algebraic simplification)
- Robust parser with property-based testing

**Domain Crates:**
- `dew-scalar` - Scalar math (sin, cos, exp, lerp, etc.)
- `dew-linalg` - Linear algebra (Vec2-4, Mat2-4, dot, cross, normalize, etc.)
- `dew-complex` - Complex numbers (exp, log, polar, conjugate, etc.)
- `dew-quaternion` - Quaternions (rotation, slerp, axis-angle, etc.)
- `dew-all` - Unified value type for domain composition

**Code Generation:**
- WGSL backend (all domain crates)
- GLSL backend (all domain crates)
- OpenCL backend (all domain crates)
- CUDA backend (all domain crates)
- HIP backend (all domain crates, source-compatible with CUDA)
- Rust text backend (all domain crates)
- C text backend (all domain crates)
- TokenStream backend for proc-macros (all domain crates)
- Lua backend with codegen + execution (all domain crates)
- Cranelift JIT backend (all domain crates)

**Tooling:**
- Editor support: VSCode, TextMate, Tree-sitter (Neovim, Helix, Zed, Emacs)
- VitePress documentation site
- WASM bindings with module profiles (core, linalg, graphics, signal, full)
- CI/CD with exhaustive backend parity tests

### In Progress

**Web Playground:**
- UI framework complete (SolidJS, editor, AST viewer)
- WASM integration pending (currently using mock data)
- Needs: real-time evaluation, variable input, feature toggles

### Future Work

**New Domains:**
- Dual numbers (automatic differentiation)
- Rotors/spinors (geometric algebra)

**External Backend Support:**
- Pattern for external codegen crates (e.g., `dew-linalg-metal`)
- Shared type inference utilities across backends

See `TODO.md` for detailed implementation tracking.

## License

MIT
