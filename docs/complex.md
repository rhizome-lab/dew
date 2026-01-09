# dew-complex

Complex number support for dew expressions. Useful for signal processing, 2D rotations, and general complex arithmetic.

## Installation

```toml
[dependencies]
rhizome-dew-core = "0.1"
rhizome-dew-complex = "0.1"

# Enable backends as needed
rhizome-dew-complex = { version = "0.1", features = ["wgsl", "lua", "cranelift"] }
```

## Basic Usage

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_complex::{Value, eval, complex_registry};
use std::collections::HashMap;

// Parse an expression
let expr = Expr::parse("a * b").unwrap();

// Set up complex variables
let mut vars: HashMap<String, Value<f32>> = HashMap::new();
vars.insert("a".to_string(), Value::Complex([1.0, 2.0]));  // 1 + 2i
vars.insert("b".to_string(), Value::Complex([3.0, 4.0]));  // 3 + 4i

// Evaluate
let registry = complex_registry();
let result = eval(expr.ast(), &vars, &registry).unwrap();
// (1+2i)(3+4i) = -5 + 10i
assert_eq!(result, Value::Complex([-5.0, 10.0]));
```

## Types

### Value Enum

```rust
pub enum Value<T> {
    Scalar(T),           // Real number
    Complex([T; 2]),     // [re, im]
}
```

### Type Enum

```rust
pub enum Type {
    Scalar,
    Complex,
}
```

## Operators

### Arithmetic

| Expression | Result |
|------------|--------|
| `z + w` | Complex addition |
| `z - w` | Complex subtraction |
| `z * w` | Complex multiplication |
| `z / w` | Complex division |
| `z ^ n` | Complex power (scalar exponent) |
| `-z` | Negation |
| `s * z` | Scalar-complex multiply |
| `z * s` | Complex-scalar multiply |

### Complex Multiplication

For `(a + bi)(c + di)`:
- Real: `ac - bd`
- Imag: `ad + bc`

### Complex Division

For `(a + bi) / (c + di)`:
- Multiply by conjugate: `(a + bi)(c - di) / (c² + d²)`

## Functions

### Component Access

| Function | Description | Result |
|----------|-------------|--------|
| `re(z)` | Real part | Scalar |
| `im(z)` | Imaginary part | Scalar |

### Properties

| Function | Description | Result |
|----------|-------------|--------|
| `abs(z)` | Magnitude \|z\| = √(a² + b²) | Scalar |
| `arg(z)` | Phase angle atan2(b, a) | Scalar |
| `norm(z)` | Squared magnitude a² + b² | Scalar |
| `conj(z)` | Conjugate a - bi | Complex |

### Exponential / Logarithmic

| Function | Description | Result |
|----------|-------------|--------|
| `exp(z)` | Complex exponential e^z | Complex |
| `log(z)` | Complex natural logarithm | Complex |
| `sqrt(z)` | Complex square root | Complex |
| `pow(z, n)` | Complex power | Complex |

### Construction

| Function | Description | Result |
|----------|-------------|--------|
| `polar(r, theta)` | From polar form r·e^(iθ) | Complex |

## Mathematical Notes

### Euler's Formula

```
exp(a + bi) = e^a · (cos(b) + i·sin(b))
```

### Complex Logarithm

```
log(z) = log|z| + i·arg(z)
```

### Complex Square Root

```
sqrt(z) = sqrt(|z|) · (cos(arg(z)/2) + i·sin(arg(z)/2))
```

## Examples

### Complex Multiplication

```rust
// (1 + 2i) * (3 + 4i) = -5 + 10i
let expr = Expr::parse("a * b").unwrap();
vars.insert("a".to_string(), Value::Complex([1.0, 2.0]));
vars.insert("b".to_string(), Value::Complex([3.0, 4.0]));
let result = eval(expr.ast(), &vars, &registry).unwrap();
assert_eq!(result, Value::Complex([-5.0, 10.0]));
```

### Magnitude

```rust
// |3 + 4i| = 5
let expr = Expr::parse("abs(z)").unwrap();
vars.insert("z".to_string(), Value::Complex([3.0, 4.0]));
let result = eval(expr.ast(), &vars, &registry).unwrap();
assert_eq!(result, Value::Scalar(5.0));
```

### 2D Rotation via Complex Multiply

```rust
// Rotate point (1, 0) by 90 degrees
// Multiply by i = (0 + 1i)
let expr = Expr::parse("point * rotation").unwrap();
vars.insert("point".to_string(), Value::Complex([1.0, 0.0]));
vars.insert("rotation".to_string(), Value::Complex([0.0, 1.0]));  // i = 90°
let result = eval(expr.ast(), &vars, &registry).unwrap();
// result ≈ (0, 1)
```

## Generic Numeric Types

Works with f32 or f64:

```rust
let mut vars: HashMap<String, Value<f64>> = HashMap::new();
vars.insert("z".to_string(), Value::Complex([3.0, 4.0]));
let result: Value<f64> = eval(expr.ast(), &vars, &complex_registry()).unwrap();
```

## Backends

See [WGSL](/backends/wgsl), [Lua](/backends/lua), [Cranelift](/backends/cranelift) for backend-specific usage.
