# TODO

## Backlog

### Port from resin-expr

- [x] Port `resin-expr` core expression types and AST
- [x] Port `resin-expr-std` standard library functions (excluding noise)

### Core

- [x] Define expression AST (functions, numeric values)
- [ ] Implement type system for expressions
- [ ] Add expression validation/normalization

### Backends

- [x] WGSL code generation (sap-wgsl)
- [x] Cranelift JIT compilation (sap-cranelift)
- [x] Lua code generation (sap-lua, mlua)

### Standard Library Backend Implementations

- [x] Add WGSL std functions to sap-std (behind feature flag)
- [x] Add Lua std functions to sap-std (behind feature flag)
- [x] Add Cranelift std functions to sap-std (behind feature flag)
  - Transcendental functions (sin, cos, exp, log, etc.) implemented via Rust callbacks
- [x] Add parity tests across backends (all 3 backends produce identical results)

### Infrastructure

- [x] Set up CI tests for all backends
- [x] Add integration tests (parity tests across backends)
- [x] Exhaustive test matrix for all functions and operations across all backends
- [ ] Documentation examples

## In Progress

### Linear Algebra (sap-linalg)

Design doc: `docs/linalg-design.md`

Architecture decision: **core = syntax only, domains = semantics**.

#### Design Decisions (resolved)

- [x] Type checking: no separate pass, types propagate during eval/emit
- [x] Function dispatch: runtime dispatch on `Value` variants
- [x] AST changes: none needed, core stays untyped
- [x] Literals: f32, type comes from context

#### Implementation Plan
- [x] Core scaffold: Value<T>, Type, eval, ops
- [x] Vec2, Mat2 types and operations
- [x] Vec3, Mat3 types and operations (feature = "3d", default)
- [x] Vec4, Mat4 (feature = "4d")
- [x] Operator dispatch (`*` for scale, matmul, etc.)
- [x] Generic over numeric type (f32, f64 via num-traits)
- [x] vec * mat (row vector convention) in addition to mat * vec
- [x] Common linalg functions: dot, cross, length, normalize, distance, reflect, hadamard, lerp, mix
- [ ] Backend implementations (WGSL, Lua, Cranelift)

#### Future Extensions (separate crates probably)

- Complex numbers (2D rotations)
- Quaternions (3D rotations)
- Dual numbers (autodiff)
- Rotors/spinors (geometric algebra)
