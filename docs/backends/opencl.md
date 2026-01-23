# OpenCL Backend

Generate OpenCL kernel code from dew expressions.

## Enable

```toml
rhizome-dew-scalar = { version = "0.1", features = ["opencl"] }
rhizome-dew-linalg = { version = "0.1", features = ["opencl"] }
```

## dew-scalar

### Generate Expression

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::opencl::emit_opencl;

let expr = Expr::parse("sin(x) + cos(y)").unwrap();
let ocl = emit_opencl(expr.ast()).unwrap();

println!("{}", ocl.code);
// Output: (sin(x) + cos(y))
```

### Generate Kernel Function

```rust
use rhizome_dew_scalar::opencl::emit_opencl_fn;

let expr = Expr::parse("x * x + y * y").unwrap();
let ocl = emit_opencl_fn("distance_squared", expr.ast(), &["x", "y"]).unwrap();

println!("{}", ocl);
// Output:
// float distance_squared(float x, float y) {
//     return ((x * x) + (y * y));
// }
```

## dew-linalg

### Generate with Types

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_linalg::opencl::emit_opencl;
use rhizome_dew_linalg::Type;
use std::collections::HashMap;

let expr = Expr::parse("dot(a, b)").unwrap();

let mut var_types: HashMap<String, Type> = HashMap::new();
var_types.insert("a".to_string(), Type::Vec3);
var_types.insert("b".to_string(), Type::Vec3);

let result = emit_opencl(expr.ast(), &var_types).unwrap();

println!("{}", result.code);
// Output: dot(a, b)
```

## OpenCL Features

### Built-in Vector Types

OpenCL provides native vector types:

| dew Type | OpenCL Type |
|----------|-------------|
| Scalar | `float` |
| Vec2 | `float2` |
| Vec3 | `float3` |
| Vec4 | `float4` |

### Built-in Vector Operations

OpenCL has built-in functions for common vector operations:

| dew | OpenCL |
|-----|--------|
| `dot(a, b)` | `dot(a, b)` |
| `cross(a, b)` | `cross(a, b)` |
| `length(v)` | `length(v)` |
| `normalize(v)` | `normalize(v)` |
| `distance(a, b)` | `distance(a, b)` |

### Vector Constructors

```c
// OpenCL uses cast-style constructors
(float2)(x, y)
(float3)(x, y, z)
(float4)(x, y, z, w)
```

## Function Mapping

| dew | OpenCL |
|-----|--------|
| `sin(x)` | `sin(x)` |
| `cos(x)` | `cos(x)` |
| `sqrt(x)` | `sqrt(x)` |
| `rsqrt(x)` | `rsqrt(x)` |
| `abs(x)` | `fabs(x)` |
| `floor(x)` | `floor(x)` |
| `x ^ y` | `pow(x, y)` |
| `min(a, b)` | `fmin(a, b)` |
| `max(a, b)` | `fmax(a, b)` |
| `pi()` | `M_PI_F` |
| `lerp(a, b, t)` | `mix(a, b, t)` |
| `clamp(x, lo, hi)` | `clamp(x, lo, hi)` |

## Comparison with CUDA/HIP

| Feature | OpenCL | CUDA/HIP |
|---------|--------|----------|
| Platform | Cross-platform | NVIDIA/AMD |
| Vector constructors | `(float2)(x, y)` | `make_float2(x, y)` |
| Built-in dot/cross | Yes | Requires helpers |
| Math functions | `sin()` | `sinf()` |
| Constants | `M_PI_F` | `M_PI` |

## Use Cases

- **Cross-platform GPU compute**: Run on NVIDIA, AMD, Intel, and mobile GPUs
- **Heterogeneous computing**: CPU + GPU on same platform
- **FPGA acceleration**: OpenCL targets some FPGAs
