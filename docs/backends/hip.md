# HIP Backend

Generate HIP kernel code from dew expressions for AMD GPUs (ROCm).

## Enable

```toml
rhizome-dew-scalar = { version = "0.1", features = ["hip"] }
rhizome-dew-linalg = { version = "0.1", features = ["hip"] }
```

## Overview

HIP (Heterogeneous-compute Interface for Portability) is AMD's GPU programming platform designed to be **source-compatible with CUDA**. The generated HIP code is virtually identical to CUDA output, enabling easy porting between NVIDIA and AMD GPU platforms.

## dew-scalar

### Generate Expression

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::hip::emit_hip;

let expr = Expr::parse("sin(x) + cos(y)").unwrap();
let hip = emit_hip(expr.ast()).unwrap();

println!("{}", hip.code);
// Output: (sinf(x) + cosf(y))
```

### Generate Device Function

```rust
use rhizome_dew_scalar::hip::emit_hip_fn;

let expr = Expr::parse("x * x + y * y").unwrap();
let hip = emit_hip_fn("distance_squared", expr.ast(), &["x", "y"]).unwrap();

println!("{}", hip);
// Output:
// __device__ float distance_squared(float x, float y) {
//     return ((x * x) + (y * y));
// }
```

## dew-linalg

### Generate with Types

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_linalg::hip::emit_hip;
use rhizome_dew_linalg::Type;
use std::collections::HashMap;

let expr = Expr::parse("dot(a, b)").unwrap();

let mut var_types: HashMap<String, Type> = HashMap::new();
var_types.insert("a".to_string(), Type::Vec3);
var_types.insert("b".to_string(), Type::Vec3);

let result = emit_hip(expr.ast(), &var_types).unwrap();

println!("{}", result.code);
// Output: dot3(a, b)
```

## CUDA Compatibility

HIP uses the same syntax as CUDA:

| Feature | HIP | CUDA |
|---------|-----|------|
| Vector types | `float2`, `float3`, `float4` | Same |
| Constructors | `make_float2()` | Same |
| Device qualifier | `__device__` | Same |
| Math functions | `sinf()`, `cosf()` | Same |
| Constants | `M_PI` | Same |

## Function Mapping

| dew | HIP |
|-----|-----|
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

## Required Helper Functions

Like CUDA, HIP doesn't have built-in vector operations. Provide these helpers:

```cpp
__device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize3(float3 v) {
    float len = sqrtf(dot3(v, v));
    return make_float3(v.x / len, v.y / len, v.z / len);
}
```

## Use Cases

- **AMD GPU compute**: ROCm platform for AMD GPUs
- **Multi-vendor support**: Same codebase for NVIDIA and AMD
- **HPC clusters**: Many supercomputers use AMD GPUs
- **Cost optimization**: AMD GPUs often offer better price/performance

## Porting from CUDA

Since HIP output is source-compatible with CUDA:

1. Generate code with the HIP backend
2. The same code works with both `hipcc` (AMD) and `nvcc` (NVIDIA)
3. Use `#ifdef __HIP_PLATFORM_AMD__` for platform-specific code if needed

## See Also

- [CUDA Backend](./cuda.md) - For NVIDIA GPUs
- [OpenCL Backend](./opencl.md) - Cross-platform alternative
