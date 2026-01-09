# dew-scalar

Standard scalar math functions for dew expressions.

## Installation

```toml
[dependencies]
rhizome-dew-core = "0.1"
rhizome-dew-scalar = "0.1"

# Enable backends as needed
rhizome-dew-scalar = { version = "0.1", features = ["wgsl", "lua", "cranelift"] }
```

## Basic Usage

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::{eval, scalar_registry};
use std::collections::HashMap;

// Parse an expression
let expr = Expr::parse("sin(x) * 2 + cos(y)").unwrap();

// Set up variables
let mut vars: HashMap<String, f32> = HashMap::new();
vars.insert("x".to_string(), 1.0);
vars.insert("y".to_string(), 0.5);

// Evaluate
let registry = scalar_registry();
let result = eval(expr.ast(), &vars, &registry).unwrap();
println!("Result: {}", result);
```

## Available Functions

### Constants

| Function | Description |
|----------|-------------|
| `pi()` | π ≈ 3.14159 |
| `e()` | Euler's number ≈ 2.71828 |
| `tau()` | τ = 2π ≈ 6.28318 |

### Trigonometry

| Function | Description |
|----------|-------------|
| `sin(x)` | Sine |
| `cos(x)` | Cosine |
| `tan(x)` | Tangent |
| `asin(x)` | Arc sine |
| `acos(x)` | Arc cosine |
| `atan(x)` | Arc tangent |
| `atan2(y, x)` | Two-argument arc tangent |
| `sinh(x)` | Hyperbolic sine |
| `cosh(x)` | Hyperbolic cosine |
| `tanh(x)` | Hyperbolic tangent |

### Exponential / Logarithmic

| Function | Description |
|----------|-------------|
| `exp(x)` | e^x |
| `exp2(x)` | 2^x |
| `ln(x)` | Natural logarithm |
| `log(x)` | Natural logarithm (alias) |
| `log2(x)` | Base-2 logarithm |
| `log10(x)` | Base-10 logarithm |
| `pow(x, y)` | x^y |
| `sqrt(x)` | Square root |
| `inversesqrt(x)` | 1 / sqrt(x) |

### Common Math

| Function | Description |
|----------|-------------|
| `abs(x)` | Absolute value |
| `sign(x)` | Sign (-1, 0, or 1) |
| `floor(x)` | Round down |
| `ceil(x)` | Round up |
| `round(x)` | Round to nearest |
| `trunc(x)` | Truncate toward zero |
| `fract(x)` | Fractional part |
| `min(a, b)` | Minimum |
| `max(a, b)` | Maximum |
| `clamp(x, lo, hi)` | Clamp to range |
| `saturate(x)` | Clamp to [0, 1] |

### Interpolation

| Function | Description |
|----------|-------------|
| `lerp(a, b, t)` | Linear interpolation |
| `mix(a, b, t)` | Linear interpolation (GLSL naming) |
| `step(edge, x)` | 0 if x < edge, else 1 |
| `smoothstep(e0, e1, x)` | Smooth Hermite interpolation |
| `inverse_lerp(a, b, v)` | Inverse of lerp: (v - a) / (b - a) |
| `remap(x, in_lo, in_hi, out_lo, out_hi)` | Remap from one range to another |

## Operators

| Operator | Description |
|----------|-------------|
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division |
| `^` | Power |
| `-x` | Negation |

## Generic Numeric Types

Works with any `T: Float` from num-traits:

```rust
use rhizome_dew_scalar::{eval, scalar_registry};

// f32
let result_f32: f32 = eval::<f32>(expr.ast(), &vars_f32, &scalar_registry()).unwrap();

// f64
let result_f64: f64 = eval::<f64>(expr.ast(), &vars_f64, &scalar_registry()).unwrap();
```

## Backends

See [WGSL](/backends/wgsl), [Lua](/backends/lua), [Cranelift](/backends/cranelift) for backend-specific usage.
