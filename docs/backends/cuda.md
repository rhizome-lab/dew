# CUDA Backend

Generate CUDA kernel code from dew expressions for NVIDIA GPUs.

## Enable

```toml
rhizome-dew-scalar = { version = "0.1", features = ["cuda"] }
rhizome-dew-linalg = { version = "0.1", features = ["cuda"] }
```

## dew-scalar

### Generate Expression

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::cuda::emit_cuda;

let expr = Expr::parse("sin(x) + cos(y)").unwrap();
let cuda = emit_cuda(expr.ast()).unwrap();

println!("{}", cuda.code);
// Output: (sinf(x) + cosf(y))
```

### Generate Device Function

```rust
use rhizome_dew_scalar::cuda::emit_cuda_fn;

let expr = Expr::parse("x * x + y * y").unwrap();
let cuda = emit_cuda_fn("distance_squared", expr.ast(), &["x", "y"]).unwrap();

println!("{}", cuda);
// Output:
// __device__ float distance_squared(float x, float y) {
//     return ((x * x) + (y * y));
// }
```

## dew-linalg

### Generate with Types

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_linalg::cuda::emit_cuda;
use rhizome_dew_linalg::Type;
use std::collections::HashMap;

let expr = Expr::parse("dot(a, b)").unwrap();

let mut var_types: HashMap<String, Type> = HashMap::new();
var_types.insert("a".to_string(), Type::Vec3);
var_types.insert("b".to_string(), Type::Vec3);

let result = emit_cuda(expr.ast(), &var_types).unwrap();

println!("{}", result.code);
// Output: dot3(a, b)
```

## CUDA Features

### Built-in Vector Types

CUDA provides native vector types:

| dew Type | CUDA Type |
|----------|-----------|
| Scalar | `float` |
| Vec2 | `float2` |
| Vec3 | `float3` |
| Vec4 | `float4` |

### Vector Constructors

```cuda
// CUDA uses make_* functions
make_float2(x, y)
make_float3(x, y, z)
make_float4(x, y, z, w)
```

### Required Helper Functions

Unlike OpenCL, CUDA doesn't have built-in `dot`, `cross`, `normalize`, etc. You must provide these:

```cuda
__device__ float dot2(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float length3(float3 v) {
    return sqrtf(dot3(v, v));
}

__device__ float3 normalize3(float3 v) {
    float len = length3(v);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
```

Alternatively, include CUDA's `helper_math.h` for operator overloading.

## Function Mapping

| dew | CUDA |
|-----|------|
| `sin(x)` | `sinf(x)` |
| `cos(x)` | `cosf(x)` |
| `sqrt(x)` | `sqrtf(x)` |
| `rsqrt(x)` | `rsqrtf(x)` |
| `abs(x)` | `fabsf(x)` |
| `floor(x)` | `floorf(x)` |
| `x ^ y` | `powf(x, y)` |
| `x % y` | `fmodf(x, y)` |
| `min(a, b)` | `fminf(a, b)` |
| `max(a, b)` | `fmaxf(a, b)` |
| `pi()` | `M_PI` |
| `sign(x)` | `copysignf(1.0f, x)` |

## Use Cases

- **NVIDIA GPU compute**: High-performance parallel computing
- **Deep learning preprocessing**: Custom data transformations
- **Scientific computing**: Physics simulations, numerical methods
- **Real-time graphics**: Custom shader-like computations

## See Also

- [HIP Backend](./hip.md) - Source-compatible with CUDA for AMD GPUs
