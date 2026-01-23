# Rust Backend

Generate Rust source code from dew expressions.

## Enable

```toml
rhizome-dew-scalar = { version = "0.1", features = ["rust"] }
rhizome-dew-linalg = { version = "0.1", features = ["rust"] }
```

## dew-scalar

### Generate Expression

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::rust::emit_rust;

let expr = Expr::parse("sin(x) + cos(y)").unwrap();
let rust = emit_rust(expr.ast()).unwrap();

println!("{}", rust.code);
// Output: (x.sin() + y.cos())
```

### Generate Function

```rust
use rhizome_dew_scalar::rust::emit_rust_fn;

let expr = Expr::parse("x * x + y * y").unwrap();
let rust = emit_rust_fn("distance_squared", expr.ast(), &["x", "y"]).unwrap();

println!("{}", rust);
// Output:
// fn distance_squared(x: f32, y: f32) -> f32 {
//     (x * x) + (y * y)
// }
```

## dew-linalg

### Generate with Types

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_linalg::rust::emit_rust;
use rhizome_dew_linalg::Type;
use std::collections::HashMap;

let expr = Expr::parse("dot(a, b)").unwrap();

let mut var_types: HashMap<String, Type> = HashMap::new();
var_types.insert("a".to_string(), Type::Vec3);
var_types.insert("b".to_string(), Type::Vec3);

let result = emit_rust(expr.ast(), &var_types).unwrap();

println!("{}", result.code);
// Output: a.dot(b)
```

## Function Mapping

| dew | Rust |
|-----|------|
| `sin(x)` | `x.sin()` |
| `cos(x)` | `x.cos()` |
| `sqrt(x)` | `x.sqrt()` |
| `abs(x)` | `x.abs()` |
| `floor(x)` | `x.floor()` |
| `x ^ y` | `x.powf(y)` |
| `min(a, b)` | `a.min(b)` |
| `max(a, b)` | `a.max(b)` |
| `lerp(a, b, t)` | `(a + (b - a) * t)` |
| `clamp(x, lo, hi)` | `x.clamp(lo, hi)` |

## Use Cases

- **Proc-macro code generation**: Generate Rust code at compile time
- **Debug output**: Inspect generated expressions as readable Rust
- **Documentation**: Show equivalent Rust code in examples
- **Integration**: Embed in Rust projects without runtime dependencies

## Comparison with TokenStream

| Feature | Rust text | TokenStream |
|---------|-----------|-------------|
| Output | `String` | `proc_macro2::TokenStream` |
| Use case | Debug, docs | Proc-macros |
| Parsing | Requires re-parse | Direct use in macros |
