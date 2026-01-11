# TODO

## Backlog

### Port from resin-expr

- [x] Port `resin-expr` core expression types and AST
- [x] Port `resin-expr-std` standard library functions (excluding noise)

### Core

- [x] Define expression AST (functions, numeric values)

Note: No type system in core. Core = syntax only, domains = semantics.
Each domain crate handles its own types during eval/emit.

### Backends (now self-contained in domain crates)

- [x] dew-scalar: WGSL, Lua, Cranelift backends (self-contained)
- [x] dew-linalg: WGSL, Lua, Cranelift backends (self-contained)
- [x] dew-complex: WGSL, Lua, Cranelift backends (self-contained)
- [x] dew-quaternion: WGSL, Lua, Cranelift backends (self-contained)

Note: Old standalone backend crates removed.
Each domain crate now has self-contained backends behind feature flags.

### Conditionals (dew-cond)

- [x] Conditional AST nodes in dew-core (Compare, And, Or, If, UnaryOp::Not)
- [x] Feature-gated: `cond` feature in dew-core enables conditional syntax
- [x] Feature-gated: `func` feature in dew-core enables function calls
- [x] dew-cond crate with backend helpers (WGSL, Lua, Cranelift)
- [x] Scalar comparison/conditional support in dew-scalar
- [x] Scalar-only comparison passthrough in domain crates (complex, linalg, quaternion)

Note: Comparison semantics only for scalars. Complex/quaternion/vector types
don't have obvious comparison semantics, so they only support scalar comparisons
via passthrough.

### Standard Library (dew-scalar)

- [x] Generic over `T: Float` (works with f32, f64)
- [x] Own `ScalarFn<T>` trait and `FunctionRegistry<T>`
- [x] Own `eval<T>()` function
- [x] All standard functions: trig, exp/log, common math, interpolation
- [x] WGSL backend (feature = "wgsl")
- [x] Lua backend with mlua execution (feature = "lua")
- [x] Cranelift JIT backend (feature = "cranelift")

### Infrastructure

- [x] Set up CI tests for all backends
- [x] Add integration tests (parity tests across backends)
- [x] Exhaustive test matrix for all functions and operations across all backends
- [x] Documentation examples
- [x] Property-based testing (proptest) for parser robustness
- [x] Runnable examples in `examples/` directory
- [x] VitePress documentation in `docs/`

## Complexity Hotspots (38 functions >21)
- [ ] `crates/dew-quaternion/src/lua.rs:emit_function_call` (75)
- [ ] `crates/dew-linalg/src/lua.rs:emit_function_call` (65)
- [ ] `crates/dew-complex/src/wgsl.rs:emit_function_call` (59)
- [ ] `crates/dew-complex/src/lua.rs:emit_function_call` (59)
- [ ] `crates/dew-quaternion/src/wgsl.rs:emit_function_call` (59)
- [ ] `crates/dew-linalg/src/ops.rs:apply_mul` (56)
- [ ] `crates/dew-quaternion/src/funcs.rs:slerp_impl` (54)
- [ ] `crates/dew-quaternion/src/ops.rs:apply_mul` (53)
- [ ] `crates/dew-core/src/lib.rs:eval_ast` (52)
- [ ] `crates/dew-scalar/src/lib.rs:eval` (52)
- [ ] `crates/dew-core/src/lib.rs:eval_ast` (49)
- [ ] `crates/dew-complex/src/lib.rs:eval` (45)
- [ ] `crates/dew-linalg/src/lib.rs:eval` (44)
- [ ] `crates/dew-quaternion/src/lib.rs:eval` (44)
- [ ] `crates/dew-scalar/src/wgsl.rs:wgsl_func_name` (40)
- [ ] `crates/dew-scalar/src/lua.rs:emit_func` (40)
- [ ] `crates/dew-complex/src/cranelift.rs:compile_call` (39)
- [ ] `crates/dew-quaternion/src/funcs.rs:rotate_vec3_by_quat` (31)
- [ ] `crates/dew-quaternion/src/ops.rs:rotate_vec3_by_quat` (31)
- [ ] `crates/dew-linalg/src/ops.rs:mat4_mul_vec4` (29)
- [ ] `crates/dew-linalg/src/ops.rs:vec4_mul_mat4` (29)
- [ ] `crates/dew-linalg/src/cranelift.rs:compile_binop` (29)
- [ ] `crates/dew-scalar/src/cranelift.rs:compile_function` (29)
- [ ] `crates/dew-complex/src/ops.rs:apply_div` (27)
- [ ] `crates/dew-complex/src/wgsl.rs:emit_wgsl` (27)
- [ ] `crates/dew-complex/src/lua.rs:emit_lua` (27)
- [ ] `crates/dew-linalg/src/wgsl.rs:emit_wgsl` (27)
- [ ] `crates/dew-linalg/src/wgsl.rs:emit_function_call` (27)
- [ ] `crates/dew-linalg/src/lua.rs:emit_lua` (27)
- [ ] `crates/dew-linalg/src/cranelift.rs:compile_call` (27)
- [ ] `crates/dew-quaternion/src/wgsl.rs:emit_wgsl` (27)
- [ ] `crates/dew-quaternion/src/lua.rs:emit_lua` (27)
- [ ] `crates/dew-scalar/src/wgsl.rs:emit` (26)
- [ ] `crates/dew-linalg/src/lua.rs:emit_mul` (25)
- [ ] `crates/dew-quaternion/src/cranelift.rs:compile_binop` (24)
- [ ] `crates/dew-quaternion/src/cranelift.rs:compile_call` (23)
- [ ] `crates/dew-linalg/src/ops.rs:mat4_mul_mat4` (22)
- [ ] `crates/dew-complex/src/cranelift.rs:compile_binop` (21)

## Future Work

### Editor Support

- [x] TextMate grammar (enables VSCode + Shiki/VitePress)
- [x] VSCode extension (bundles TextMate grammar)
- [x] Tree-sitter grammar (enables Neovim, Helix, Zed, Emacs)
- [x] Shiki integration for VitePress docs

### Web Playground

- [ ] WASM build of dew crates (core + domain crates, no cranelift/lua backends)
- [ ] Parser + collapsible AST viewer
- [ ] Expression runner (interpreter)
- [ ] Codegen viewer (WGSL, Lua output)
- [ ] Feature toggle UI (cond, func) - requires runtime feature config or multiple WASM builds

Stack: SolidJS, Bun, Vite or Astro, BEM
Style reference: `~/git/lotus/packages/shared/src/index.css` (glassmorphic)

**WASM blockers:**
- `cranelift` feature: Can't JIT in browser. Skip for playground or show IR only.
- `lua` feature: Split into `lua-codegen` (pure Rust) and `lua` (mlua execution).
  Playground can use `lua-codegen` for viewing without C deps.
- Core + eval: No blockers. `std::sync::Arc` works in WASM.

**Feature toggles:**
- Currently compile-time (`#[cfg(feature = "...")]`)
- Options: (a) multiple WASM builds, (b) make features runtime-configurable
- Simplest: always enable all features in playground build

**Tooling ideas:**
- Monaco editor with dew language support (reuse TextMate grammar)
- wasm-bindgen for Rust-JS interop
- Consider zustand or nanostores for state management
- AST viewer: use a collapsible tree component (or build one)

### New Domain Crates

- [x] Complex numbers (2D rotations) - dew-complex
- [x] Quaternions (3D rotations) - dew-quaternion
- [ ] Dual numbers (autodiff)
- [ ] Rotors/spinors (geometric algebra)

### Nice to Have (maybe)

- Expression normalization/simplification (constant folding, algebraic simplification)
  - Would live in domain crates, not core
  - Could be useful for optimization before JIT compilation

## Backlog - Architecture

### Crate Composability

Problem: What if a user wants to use multiple domain crates together? E.g., linalg + rotors in the same expression.

Current state:
- dew-scalar: `T: Float` scalars
- dew-linalg: `Value<T>` enum (Scalar, Vec2, Vec3, Vec4, Mat2, Mat3, Mat4)
- dew-complex: `Value<T>` enum (Scalar, Complex)
- dew-quaternion: `Value<T>` enum (Scalar, Vec3, Quaternion)
- Future crates might add: Rotor, DualNumber, etc.

Each has its own `FunctionRegistry<T>` and `eval()` function.

Options to investigate:

1. **Dyn traits**: Common value trait, runtime dispatch
2. **Enum composition**: Generate combined value enum with auto-derived From/Into
   ```rust
   // Macro-generated
   enum CombinedValue<T> {
       Scalar(T),
       Vec2([T; 2]),
       Rotor(rotor::Rotor<T>),
       // ...
   }
   ```
3. **Generic over value type**: Each crate is generic over the value abstraction, users compose by providing their own combined type
4. **Extension trait pattern**: Shared base Value in dew-core, domain crates add methods via traits
   - **Problem**: Value must have ALL variants in dew-core upfront. Can't add new variants from external crates.
   - Only viable if dew-core is a monolith knowing all domains. Defeats modularity. **Not recommended.**

Trade-offs:

| Aspect | Option 2 (enum composition) | Option 3 (generic) |
|--------|---------------------------|-------------------|
| Standalone use | Easy: `LinalgValue<f32>` | Awkward: need concrete type |
| Trait bounds | None | Everywhere: `V: LinalgValue<T>` |
| Conversion cost | Small (enum moves cheap) | Zero |
| Compile time | Lower | Higher (monomorphization) |
| Implementation | From/Into macros | Trait impls for each combo |
| Flexibility | Closed (fixed variants) | Open (any type works) |

**Decision**: Option 3 (generic over value type) for open extension.

#### Implementation Status

Phase 1 (done):
- [x] Define `LinalgValue<T>` trait with construction/extraction methods
- [x] Implement trait for `Value<T>` (default concrete type)
- [x] Export trait from crate

Phase 2 (future, when needed):
- [ ] Make `LinalgFn<T>` generic: `LinalgFn<T, V: LinalgValue<T>>`
- [ ] Make `FunctionRegistry<T>` generic: `FunctionRegistry<T, V>`
- [ ] Make `eval()` generic: `eval<T, V: LinalgValue<T>>()`
- [ ] Update ops.rs to use trait extraction/construction
- [ ] Update funcs.rs to use trait

Current state: The trait is defined and users CAN implement it for their combined types.
But to actually USE combined types with eval/functions, Phase 2 is needed.

Each domain crate:
- Exposes a `DomainValue<T>` trait with construction/extraction methods
- Provides a default concrete type for standalone use (`Value<T>`)
- (Future) Has `eval<T, V: DomainValue<T>>()` generic over value type

Users composing multiple domains:
- Define their own combined enum
- Implement all domain traits for it
- (Future) Both crates work directly with it, zero conversion

Note: Phase 2 is low priority until we have 2+ domain crates that people want to compose.

### External Backend Support

How to create `dew-linalg-glsl` without modifying dew-linalg.

#### Pattern: Backends Don't Need Value, Only Type

Code generation backends (emit) work differently from evaluation:
- **Eval**: needs `Value<T>` to hold actual data
- **Emit**: needs `Type` for type inference, generates string output

Internal backends (wgsl.rs, lua.rs, cranelift.rs) show the pattern:
```rust
// Only needs Type, not Value
pub fn emit_wgsl(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<WgslExpr, WgslError>
```

#### Creating an External Backend

Example: `dew-linalg-glsl` crate

```toml
[dependencies]
rhizome-dew-core = "..."   # For Ast
rhizome-dew-linalg = "..." # For Type enum
```

```rust
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use rhizome_dew_linalg::Type;

pub fn emit_glsl(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<GlslExpr, GlslError> {
    match ast {
        Ast::Num(n) => /* ... */,
        Ast::Var(name) => /* lookup type from var_types */,
        Ast::BinOp(op, left, right) => {
            let left_expr = emit_glsl(left, var_types)?;
            let right_expr = emit_glsl(right, var_types)?;
            // Infer result type based on operand types
            // Generate GLSL-specific code
        }
        Ast::Call(name, args) => {
            // Map function names to GLSL equivalents
            // e.g., "lerp" -> "mix", "hadamard" -> element-wise multiply
        }
    }
}
```

#### What Domain Crates Must Expose

- [x] Public `Type` enum with public variants
- [x] Public `Value<T>` enum (for users who want both eval and emit in one project)
- [ ] Shared type inference rules (currently duplicated in each backend)
  - Consider: `pub fn infer_binop_type(op, left, right) -> Result<Type, TypeError>`
  - Would reduce duplication and ensure consistency

#### Testing External Backends

Pattern: parity tests against eval results
```rust
// Parse, emit to your backend, execute, compare with eval()
let expr = Expr::parse("dot(a, b)").unwrap();
let glsl_code = emit_glsl(expr.ast(), &var_types)?;
let glsl_result = execute_glsl(&glsl_code, &values)?;
let eval_result = eval(expr.ast(), &values, &registry)?;
assert_close(glsl_result, eval_result);
```
