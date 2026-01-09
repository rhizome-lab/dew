# dew-linalg Design Notes

## Core Architecture Decision

**Core = syntax only, domains = semantics.**

`dew-core` provides:
- AST representation
- Parsing
- Backend trait interfaces

Domain crates (like `dew-linalg`) provide:
- Value types
- Type checking/inference
- Evaluation (interpreter)
- Backend implementations

This allows extensibility without hardcoding types in core.

## Why Not Just Scalars?

Component-wise decomposition (Vec3 = 3 scalar Exprs) loses:
- Semantic information (can't emit `dot(a, b)` in WGSL)
- Hardware acceleration (SIMD, GPU vector ops)
- Sane function signatures (`mat4 * vec4` would be 20 scalar args)
- Operator overloading (`*` means different things for different types)

## Value Types

Each domain defines its own `Value` enum:

```rust
// dew-linalg
enum Value {
    Scalar(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Mat2([f32; 4]),
    Mat3([f32; 9]),
    Mat4([f32; 16]),
}
```

This is NOT in core. Other domains could define:
- `Stream(Box<dyn Iterator<Item=f32>>)`
- `Texture(Handle)`
- `Complex([f32; 2])`
- etc.

## Type Handling

No separate type inference pass. Types propagate during eval/emit.

**For evaluation:**
- Variables have known types (caller provides `HashMap<String, Value>`)
- Operators dispatch on runtime value types
- Errors caught when types don't match

**For backends:**
- Variables have known types (caller provides `HashMap<String, Type>`)
- Walk AST, propagate types bottom-up, emit code
- Single pass: returns `(code, type)` tuple

```rust
fn emit_wgsl(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<(String, Type), TypeError> {
    match ast {
        Ast::BinOp(Mul, l, r) => {
            let (l_code, l_typ) = emit_wgsl(l, var_types)?;
            let (r_code, r_typ) = emit_wgsl(r, var_types)?;
            let out_typ = mul_result_type(l_typ, r_typ)?;
            Ok((format!("({l_code} * {r_code})"), out_typ))
        }
        // ...
    }
}
```

## Numeric Type Genericity

`Value<T>` is generic over `T: num_traits::Float`.

- Works with f32, f64 out of the box
- Literals from AST (f32) are converted to T via `T::from(f32)`
- Key insight: **convert at boundaries** - expression is homogeneous, caller handles input/output conversion
- No mixed types within an expression

## Feature Gating Plan

```
dew-linalg
├── vec2, mat2          (always)
├── vec3, mat3          (default, feature = "3d")
├── vec4, mat4          (feature = "4d" or "homogeneous")
└── future: complex, quaternion, etc. (separate crates?)
```

Mat3/Mat4 extra row/col useful for affine transforms (translation).

## Open Questions

1. **Type checking**: Per-domain, or shared `dew-types` crate?
2. **Function dispatch**: How does `mul(mat4, vec4)` find the right implementation?
3. **AST changes**: Does core AST need type annotations, or is that domain-level?
4. **Literal parsing**: `3.14` - is type inferred or annotated?
