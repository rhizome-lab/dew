# Lua Backend

Generate and execute Lua code from dew expressions.

## Enable

```toml
rhizome-dew-scalar = { version = "0.1", features = ["lua"] }
rhizome-dew-linalg = { version = "0.1", features = ["lua"] }
```

## dew-scalar

### Generate Lua Code

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::lua::emit_lua;

let expr = Expr::parse("sin(x) + cos(y)").unwrap();
let lua_code = emit_lua(expr.ast()).unwrap();

println!("{}", lua_code.code);
// Output: math.sin(x) + math.cos(y)
```

### Generate Function

```rust
use rhizome_dew_scalar::lua::emit_lua_fn;

let expr = Expr::parse("x * x + y * y").unwrap();
let lua_fn = emit_lua_fn("distance_squared", expr.ast(), &["x", "y"]).unwrap();

println!("{}", lua_fn);
// Output:
// function distance_squared(x, y)
//     return x * x + y * y
// end
```

### Execute Directly

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::lua::eval_lua;
use std::collections::HashMap;

let expr = Expr::parse("sin(x) * 2").unwrap();

let mut vars: HashMap<String, f32> = HashMap::new();
vars.insert("x".to_string(), 1.5);

let result = eval_lua(expr.ast(), &vars).unwrap();
println!("Result: {}", result);
```

## dew-linalg

### Generate with Types

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_linalg::lua::emit_lua;
use rhizome_dew_linalg::Type;
use std::collections::HashMap;

let expr = Expr::parse("dot(a, b)").unwrap();

let mut var_types: HashMap<String, Type> = HashMap::new();
var_types.insert("a".to_string(), Type::Vec2);
var_types.insert("b".to_string(), Type::Vec2);

let result = emit_lua(expr.ast(), &var_types).unwrap();
println!("{}", result.code);
// Output: (a[1] * b[1] + a[2] * b[2])
```

### Execute with Values

```rust
use rhizome_dew_linalg::lua::eval_lua;
use rhizome_dew_linalg::Value;

let expr = Expr::parse("length(v)").unwrap();

let mut vars: HashMap<String, Value<f32>> = HashMap::new();
vars.insert("v".to_string(), Value::Vec2([3.0, 4.0]));

let result = eval_lua(expr.ast(), &vars).unwrap();
// result = Scalar(5.0)
```

## Function Mapping

| dew | Lua |
|-----|-----|
| `sin(x)` | `math.sin(x)` |
| `floor(x)` | `math.floor(x)` |
| `min(a, b)` | `math.min(a, b)` |
| `x ^ y` | `x ^ y` (native) |
| `sinh(x)` | `(math.exp(x) - math.exp(-x)) / 2` |

Note: Lua lacks native hyperbolic functions, so they're computed via `exp`.

## Use Cases

- Hot-reloading expressions in game engines
- Scripting environments
- Prototyping before JIT compilation
