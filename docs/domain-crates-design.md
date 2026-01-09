# Domain Crates Design

Design doc for future domain crates: complex, quaternion, dual numbers.

## Overview

```
dew-core (syntax only)
    │
    ├── dew-scalar      # T: Float
    ├── dew-linalg      # Vec2, Vec3, Mat2, Mat3, etc.
    │
    ├── dew-complex     # Complex<T> - 2D rotations, signal processing
    ├── dew-quaternion  # Quaternion<T> - 3D rotations
    └── dew-dual        # Dual<T> - automatic differentiation
```

## dew-complex

2D complex numbers: `a + bi`

### Value Type

```rust
pub struct Complex<T> {
    pub re: T,  // real
    pub im: T,  // imaginary
}

// Or as array for consistency with linalg
pub struct Complex<T>(pub [T; 2]);  // [re, im]
```

### Operations

| Operation | Meaning |
|-----------|---------|
| `a + b` | Component-wise add |
| `a - b` | Component-wise sub |
| `a * b` | Complex multiply: (a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re) |
| `a / b` | Complex divide |
| `-a` | Negate |

### Functions

| Function | Description | Return |
|----------|-------------|--------|
| `conj(z)` | Conjugate (a - bi) | Complex |
| `abs(z)` | Magnitude √(a² + b²) | Scalar |
| `arg(z)` | Phase angle atan2(b, a) | Scalar |
| `norm(z)` | Squared magnitude a² + b² | Scalar |
| `exp(z)` | Complex exponential | Complex |
| `log(z)` | Complex logarithm | Complex |
| `pow(z, n)` | Complex power | Complex |
| `sqrt(z)` | Complex square root | Complex |
| `polar(r, θ)` | From polar: r*e^(iθ) | Complex |
| `re(z)` | Real part | Scalar |
| `im(z)` | Imaginary part | Scalar |

### Composability with linalg

Rotating a Vec2 by a complex number:
- Treat Vec2 as complex, multiply, extract back
- Need: `complex * vec2 → vec2` or explicit `rotate(vec2, complex) → vec2`

Options:
1. **Explicit function:** `rotate(v, z)` - clearer intent
2. **Operator overload:** `z * v` - more magical

### Standalone Use Cases

- Signal processing (FFT, filters)
- Fractals (Mandelbrot, Julia sets)
- Electrical engineering (phasors, impedance)
- 2D rotation without matrices

---

## dew-quaternion

3D rotations: `w + xi + yj + zk`

### Value Type

```rust
pub struct Quaternion<T>(pub [T; 4]);  // [w, x, y, z] or [x, y, z, w]?

// Convention: [w, x, y, z] (scalar first) or [x, y, z, w] (scalar last)?
// GLM/glTF use [x, y, z, w] (scalar last)
// Math texts use [w, x, y, z] (scalar first)
```

**Decision needed:** Component order convention.

### Operations

| Operation | Meaning |
|-----------|---------|
| `a + b` | Component-wise add |
| `a - b` | Component-wise sub |
| `a * b` | Hamilton product (quaternion multiply) |
| `a * s` | Scale by scalar |
| `-a` | Negate |

### Functions

| Function | Description | Return |
|----------|-------------|--------|
| `conj(q)` | Conjugate (w, -x, -y, -z) | Quaternion |
| `norm(q)` | Squared magnitude | Scalar |
| `length(q)` | Magnitude | Scalar |
| `normalize(q)` | Unit quaternion | Quaternion |
| `inverse(q)` | Multiplicative inverse | Quaternion |
| `dot(a, b)` | 4D dot product | Scalar |
| `slerp(a, b, t)` | Spherical interpolation | Quaternion |
| `lerp(a, b, t)` | Linear interpolation (then normalize) | Quaternion |
| `axis_angle(axis, angle)` | From axis-angle | Quaternion |
| `from_euler(roll, pitch, yaw)` | From Euler angles | Quaternion |
| `rotate(v, q)` | Rotate Vec3 by quaternion | Vec3 |

### Composability with linalg

Critical: `rotate(vec3, quaternion) → vec3`

Options:
1. **dew-quaternion depends on dew-linalg** - can use Vec3 directly
2. **Trait-based** - QuaternionValue trait, users compose
3. **Function takes array** - `rotate([T; 3], Quaternion<T>) -> [T; 3]`

Option 3 is simplest - no dependency, arrays are universal.

### Standalone Use Cases

- Orientation representation (IMU, sensors)
- Slerp interpolation for animations
- Avoiding gimbal lock

---

## dew-dual

Dual numbers for automatic differentiation: `a + bε` where `ε² = 0`

### Value Type

```rust
pub struct Dual<T> {
    pub val: T,  // value
    pub deriv: T,  // derivative
}

// Or
pub struct Dual<T>(pub [T; 2]);  // [val, deriv]
```

### Key Insight

Dual numbers propagate derivatives automatically:
- `dual(x, 1)` means "x with derivative 1" (i.e., d/dx)
- Operations preserve chain rule automatically

```
f(a + bε) = f(a) + f'(a)·b·ε
```

### Operations

| Operation | Result |
|-----------|--------|
| `(a, a') + (b, b')` | `(a + b, a' + b')` |
| `(a, a') - (b, b')` | `(a - b, a' - b')` |
| `(a, a') * (b, b')` | `(a*b, a'*b + a*b')` — product rule |
| `(a, a') / (b, b')` | `(a/b, (a'*b - a*b') / b²)` — quotient rule |
| `-(a, a')` | `(-a, -a')` |

### Functions

All scalar functions extended with chain rule:

| Function | Dual Extension |
|----------|----------------|
| `sin(a, a')` | `(sin(a), a' * cos(a))` |
| `cos(a, a')` | `(cos(a), -a' * sin(a))` |
| `exp(a, a')` | `(exp(a), a' * exp(a))` |
| `log(a, a')` | `(log(a), a' / a)` |
| `pow(a, n)` | `(a^n, a' * n * a^(n-1))` |
| `sqrt(a, a')` | `(√a, a' / (2√a))` |

### Usage Pattern

```rust
// Compute f(x) = x² + sin(x) and its derivative at x = 2
let x = Dual::var(2.0);  // (2.0, 1.0) — value 2, derivative 1
let result = x * x + sin(x);
// result.val = 4.0 + sin(2.0)
// result.deriv = 2*2 + cos(2.0) = derivative at x=2
```

### Composability

Dual numbers wrap scalars. Could extend to:
- `Dual<Complex<T>>` - complex autodiff
- `Dual<Vec3>` - gradient (but needs multiple ε)

For gradients (∂f/∂x, ∂f/∂y, ...), need dual vectors or multiple passes.

---

## Common Patterns

### Value Trait Pattern

Each crate defines a trait for composability:

```rust
// dew-complex
pub trait ComplexValue<T: Float>: Clone + PartialEq + Debug {
    fn from_complex(c: [T; 2]) -> Self;
    fn as_complex(&self) -> Option<[T; 2]>;
    // ...
}

// dew-quaternion
pub trait QuaternionValue<T: Float>: Clone + PartialEq + Debug {
    fn from_quaternion(q: [T; 4]) -> Self;
    fn as_quaternion(&self) -> Option<[T; 4]>;
    // ...
}
```

### Backends

Each crate has self-contained backends:
- `wgsl` feature
- `lua` feature
- `cranelift` feature

WGSL note: No native complex/quaternion, emit as vec2/vec4 with custom functions.

---

## Decisions

1. **Quaternion component order:** `[x, y, z, w]` (GLM/glTF convention, scalar last)
2. **Struct style:** Array `([T; 2])` for consistency with linalg
3. **Rotation API:** Both `rotate(vec, quat)` function AND `quat * vec` operator
4. **Implementation order:** dew-complex first, then dew-quaternion
