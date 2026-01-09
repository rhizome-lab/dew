# dew-quaternion

Quaternion support for dew expressions. Essential for 3D rotations, avoiding gimbal lock.

## Installation

```toml
[dependencies]
rhizome-dew-core = "0.1"
rhizome-dew-quaternion = "0.1"

# Enable backends as needed
rhizome-dew-quaternion = { version = "0.1", features = ["wgsl", "lua", "cranelift"] }
```

## Basic Usage

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_quaternion::{Value, eval, quaternion_registry};
use std::collections::HashMap;

// Parse an expression
let expr = Expr::parse("q * v").unwrap();

// Set up variables
let mut vars: HashMap<String, Value<f32>> = HashMap::new();
vars.insert("q".to_string(), Value::Quaternion([0.0, 0.0, 0.0, 1.0]));  // identity
vars.insert("v".to_string(), Value::Vec3([1.0, 0.0, 0.0]));

// Evaluate
let registry = quaternion_registry();
let result = eval(expr.ast(), &vars, &registry).unwrap();
assert_eq!(result, Value::Vec3([1.0, 0.0, 0.0]));  // unchanged
```

## Types

### Value Enum

```rust
pub enum Value<T> {
    Scalar(T),
    Vec3([T; 3]),         // [x, y, z]
    Quaternion([T; 4]),   // [x, y, z, w] - scalar last (GLM/glTF convention)
}
```

### Type Enum

```rust
pub enum Type {
    Scalar,
    Vec3,
    Quaternion,
}
```

### Component Order

Quaternions use **`[x, y, z, w]`** order (scalar-last), matching:
- GLM (OpenGL Mathematics)
- glTF format
- Most game engines

The imaginary components are `[x, y, z]`, the real/scalar component is `w`.

## Operators

### Arithmetic

| Expression | Result |
|------------|--------|
| `q + q` | Quaternion addition |
| `q - q` | Quaternion subtraction |
| `q * s` | Scale quaternion |
| `s * q` | Scale quaternion |
| `q / s` | Divide by scalar |
| `-q` | Negate quaternion |

### Quaternion Multiplication (Hamilton Product)

| Expression | Result |
|------------|--------|
| `q1 * q2` | Hamilton product (combines rotations) |

For `q1 = [x1, y1, z1, w1]` and `q2 = [x2, y2, z2, w2]`:

```
x = w1*x2 + x1*w2 + y1*z2 - z1*y2
y = w1*y2 - x1*z2 + y1*w2 + z1*x2
z = w1*z2 + x1*y2 - y1*x2 + z1*w2
w = w1*w2 - x1*x2 - y1*y2 - z1*z2
```

### Rotation

| Expression | Result |
|------------|--------|
| `q * v` | Rotate Vec3 by quaternion |

Uses optimized formula: `v' = v + 2w(q×v) + 2(q×(q×v))`

## Functions

### Properties

| Function | Description | Result |
|----------|-------------|--------|
| `length(q)` | Quaternion magnitude | Scalar |
| `length(v)` | Vector magnitude | Scalar |
| `normalize(q)` | Unit quaternion | Quaternion |
| `normalize(v)` | Unit vector | Vec3 |
| `dot(a, b)` | Dot product | Scalar |
| `conj(q)` | Conjugate [-x, -y, -z, w] | Quaternion |
| `inverse(q)` | Inverse (conj/norm) | Quaternion |

### Interpolation

| Function | Description | Result |
|----------|-------------|--------|
| `lerp(a, b, t)` | Linear interpolation | Same type |
| `slerp(q1, q2, t)` | Spherical linear interpolation | Quaternion |

### Construction

| Function | Description | Result |
|----------|-------------|--------|
| `axis_angle(axis, angle)` | From axis-angle | Quaternion |
| `rotate(v, q)` | Rotate vector by quaternion | Vec3 |

### Vector Operations

| Function | Description | Result |
|----------|-------------|--------|
| `cross(a, b)` | Cross product | Vec3 |

## Examples

### Identity Quaternion

```rust
// [0, 0, 0, 1] = no rotation
let identity = Value::Quaternion([0.0, 0.0, 0.0, 1.0]);
```

### 90° Rotation Around Y-Axis

```rust
use std::f32::consts::FRAC_PI_2;

let expr = Expr::parse("axis_angle(axis, angle)").unwrap();
vars.insert("axis".to_string(), Value::Vec3([0.0, 1.0, 0.0]));
vars.insert("angle".to_string(), Value::Scalar(FRAC_PI_2));
let q = eval(expr.ast(), &vars, &registry).unwrap();
// q ≈ [0, 0.707, 0, 0.707]
```

### Rotating a Vector

```rust
// Rotate (1, 0, 0) by 90° around Y
let expr = Expr::parse("q * v").unwrap();
vars.insert("q".to_string(), Value::Quaternion([0.0, 0.707, 0.0, 0.707]));
vars.insert("v".to_string(), Value::Vec3([1.0, 0.0, 0.0]));
let result = eval(expr.ast(), &vars, &registry).unwrap();
// result ≈ [0, 0, -1] (rotated to negative Z)
```

### Combining Rotations

```rust
// q_combined = q2 * q1 (q1 applied first, then q2)
let expr = Expr::parse("q2 * q1").unwrap();
// Order matters! This applies q1 first, then q2
```

### Smooth Rotation Interpolation

```rust
// Slerp between two rotations
let expr = Expr::parse("slerp(q_start, q_end, t)").unwrap();
vars.insert("t".to_string(), Value::Scalar(0.5));  // halfway
let result = eval(expr.ast(), &vars, &registry).unwrap();
// Smooth interpolation that maintains constant angular velocity
```

### Inverse Rotation

```rust
// To "undo" a rotation, multiply by inverse
let expr = Expr::parse("inverse(q) * v").unwrap();
// This rotates v by the opposite of q's rotation
```

## Mathematical Notes

### Unit Quaternions

For rotations, quaternions should be **normalized** (length = 1). Use `normalize(q)` to ensure this.

### Slerp

Spherical linear interpolation:
- Interpolates along the shortest arc on the 4D unit sphere
- Maintains constant angular velocity
- Handles the "short path" automatically (negates if dot product < 0)
- Falls back to lerp when quaternions are nearly parallel

### Rotation Formula

The optimized rotation `q * v` uses:
```
t = 2 * cross(q.xyz, v)
v' = v + q.w * t + cross(q.xyz, t)
```

This is equivalent to `q * v * conj(q)` but more efficient.

## Generic Numeric Types

Works with f32 or f64:

```rust
let mut vars: HashMap<String, Value<f64>> = HashMap::new();
vars.insert("q".to_string(), Value::Quaternion([0.0, 0.0, 0.0, 1.0]));
let result: Value<f64> = eval(expr.ast(), &vars, &quaternion_registry()).unwrap();
```

## Backends

See [WGSL](/backends/wgsl), [Lua](/backends/lua), [Cranelift](/backends/cranelift) for backend-specific usage.
