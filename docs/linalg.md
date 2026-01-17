# dew-linalg

Linear algebra types and operations for dew expressions.

## Installation

```toml
[dependencies]
rhizome-dew-core = "0.1"
rhizome-dew-linalg = "0.1"

# Features
rhizome-dew-linalg = { version = "0.1", features = ["3d", "4d", "wgsl"] }
```

### Features

| Feature | Description | Default |
|---------|-------------|---------|
| `3d` | Vec3, Mat3 types | Yes |
| `4d` | Vec4, Mat4 types | No |
| `wgsl` | WGSL backend | No |
| `lua` | Lua backend | No |
| `cranelift` | Cranelift JIT backend | No |

## Basic Usage

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_linalg::{Value, eval, linalg_registry};
use std::collections::HashMap;

// Parse an expression
let expr = Expr::parse("a + b").unwrap();

// Set up vector variables
let mut vars: HashMap<String, Value<f32>> = HashMap::new();
vars.insert("a".to_string(), Value::Vec2([1.0, 2.0]));
vars.insert("b".to_string(), Value::Vec2([3.0, 4.0]));

// Evaluate
let registry = linalg_registry();
let result = eval(expr.ast(), &vars, &registry).unwrap();
assert_eq!(result, Value::Vec2([4.0, 6.0]));
```

## Types

### Value Enum

```rust
pub enum Value<T> {
    Scalar(T),
    Vec2([T; 2]),
    Vec3([T; 3]),    // feature = "3d"
    Vec4([T; 4]),    // feature = "4d"
    Mat2([T; 4]),    // column-major
    Mat3([T; 9]),    // feature = "3d", column-major
    Mat4([T; 16]),   // feature = "4d", column-major
}
```

### Type Enum

```rust
pub enum Type {
    Scalar,
    Vec2,
    Vec3,  // feature = "3d"
    Vec4,  // feature = "4d"
    Mat2,
    Mat3,  // feature = "3d"
    Mat4,  // feature = "4d"
}
```

## Operators

### Arithmetic

| Expression | Result |
|------------|--------|
| `vec + vec` | Component-wise addition |
| `vec - vec` | Component-wise subtraction |
| `vec * scalar` | Scale vector |
| `scalar * vec` | Scale vector |
| `vec / scalar` | Divide components |
| `-vec` | Negate components |

### Matrix Operations

| Expression | Result |
|------------|--------|
| `mat * vec` | Matrix-vector multiply (column vector) |
| `vec * mat` | Vector-matrix multiply (row vector) |
| `mat * mat` | Matrix multiplication |
| `mat * scalar` | Scale matrix |

## Functions

### Vector Functions

| Function | Description | Result |
|----------|-------------|--------|
| `dot(a, b)` | Dot product | Scalar |
| `cross(a, b)` | Cross product (3D only) | Vec3 |
| `length(v)` | Vector length | Scalar |
| `normalize(v)` | Unit vector | Same as input |
| `distance(a, b)` | Distance between points | Scalar |
| `reflect(i, n)` | Reflect incident vector | Same as input |
| `hadamard(a, b)` | Element-wise multiply | Same as input |
| `lerp(a, b, t)` | Linear interpolation | Same as a, b |
| `mix(a, b, t)` | Linear interpolation (GLSL naming) | Same as a, b |

## Examples

### Vector Math

```rust
// Dot product
let expr = Expr::parse("dot(a, b)").unwrap();
vars.insert("a".to_string(), Value::Vec2([1.0, 0.0]));
vars.insert("b".to_string(), Value::Vec2([0.0, 1.0]));
let result = eval(expr.ast(), &vars, &registry).unwrap();
// result = Scalar(0.0) - perpendicular vectors
```

### Matrix Transform

```rust
// Transform a vector by a matrix
let expr = Expr::parse("m * v").unwrap();
vars.insert("m".to_string(), Value::Mat2([
    1.0, 0.0,  // column 0
    0.0, 1.0,  // column 1 (identity)
]));
vars.insert("v".to_string(), Value::Vec2([3.0, 4.0]));
let result = eval(expr.ast(), &vars, &registry).unwrap();
// result = Vec2([3.0, 4.0])
```

### Normalization

```rust
let expr = Expr::parse("normalize(v)").unwrap();
vars.insert("v".to_string(), Value::Vec2([3.0, 4.0]));
let result = eval(expr.ast(), &vars, &registry).unwrap();
// result = Vec2([0.6, 0.8]) - unit vector
```

## Generic Numeric Types

Works with floating-point (f32, f64) or integer (i32, i64) types:

```rust
use rhizome_dew_linalg::{eval, linalg_registry, linalg_registry_int, Value};

// f64 precision (full function set)
let mut vars: HashMap<String, Value<f64>> = HashMap::new();
vars.insert("v".to_string(), Value::Vec2([3.0, 4.0]));
let result: Value<f64> = eval(expr.ast(), &vars, &linalg_registry()).unwrap();

// Integer vectors (Numeric-only functions)
let mut int_vars: HashMap<String, Value<i32>> = HashMap::new();
int_vars.insert("a".to_string(), Value::Vec2([1, 2]));
int_vars.insert("b".to_string(), Value::Vec2([3, 4]));
let result: Value<i32> = eval(expr.ast(), &int_vars, &linalg_registry_int()).unwrap();
```

### Integer-specific Notes

- `linalg_registry_int()` provides functions that work with integers:
  - `dot`, `cross`, `hadamard`, `lerp`, `mix`
  - Constructors: `vec2`, `vec3`, `vec4`, `mat2`, `mat3`, `mat4`
  - Extractors: `x`, `y`, `z`, `w`
- Float-only functions (`length`, `normalize`, `distance`, `reflect`) are not available
- Use case: integer grid coordinates, euclidean rhythms, pixel operations

## Backends

See [WGSL](/backends/wgsl), [Lua](/backends/lua), [Cranelift](/backends/cranelift) for backend-specific usage.
