# C Backend

Generate C source code from dew expressions.

## Enable

```toml
rhizome-dew-scalar = { version = "0.1", features = ["c"] }
rhizome-dew-linalg = { version = "0.1", features = ["c"] }
```

## dew-scalar

### Generate Expression

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::c::emit_c;

let expr = Expr::parse("sin(x) + cos(y)").unwrap();
let c = emit_c(expr.ast()).unwrap();

println!("{}", c.code);
// Output: (sinf(x) + cosf(y))
```

### Generate Function

```rust
use rhizome_dew_scalar::c::emit_c_fn;

let expr = Expr::parse("x * x + y * y").unwrap();
let c = emit_c_fn("distance_squared", expr.ast(), &["x", "y"]).unwrap();

println!("{}", c);
// Output:
// float distance_squared(float x, float y) {
//     return ((x * x) + (y * y));
// }
```

## Function Mapping

| dew | C |
|-----|---|
| `sin(x)` | `sinf(x)` |
| `cos(x)` | `cosf(x)` |
| `sqrt(x)` | `sqrtf(x)` |
| `abs(x)` | `fabsf(x)` |
| `floor(x)` | `floorf(x)` |
| `x ^ y` | `powf(x, y)` |
| `x % y` | `fmodf(x, y)` |
| `min(a, b)` | `fminf(a, b)` |
| `max(a, b)` | `fmaxf(a, b)` |
| `pi()` | `M_PI` |
| `e()` | `M_E` |

## Limitations

The C backend generates code that assumes external type and function definitions.

### Required Type Definitions (User-Provided)

```c
// Vectors (structs with component fields)
typedef struct { float x, y; } vec2;
typedef struct { float x, y, z; } vec3;
typedef struct { float x, y, z, w; } vec4;

// Matrices (column-major)
typedef struct { vec2 cols[2]; } mat2;
typedef struct { vec3 cols[3]; } mat3;
typedef struct { vec4 cols[4]; } mat4;

// Complex numbers
typedef struct { float re, im; } complex_t;

// Quaternions
typedef struct { float x, y, z, w; } quat_t;
```

### Required Function Definitions (User-Provided)

```c
// Vector operations
vec2 vec2_add(vec2 a, vec2 b);
float vec2_dot(vec2 a, vec2 b);
vec2 vec2_scale(vec2 v, float s);
vec2 vec2_normalize(vec2 v);
float vec2_length(vec2 v);

// Matrix operations
mat2 mat2_mul(mat2 a, mat2 b);
vec2 mat2_mul_vec2(mat2 m, vec2 v);

// Complex operations
complex_t complex_mul(complex_t a, complex_t b);
complex_t complex_conj(complex_t z);

// Quaternion operations
quat_t quat_mul(quat_t a, quat_t b);
quat_t quat_slerp(quat_t a, quat_t b, float t);
```

### Standard C Functions Used

From `<math.h>`:
- `sinf`, `cosf`, `tanf`, `asinf`, `acosf`, `atanf`, `atan2f`
- `expf`, `logf`, `log2f`, `log10f`, `powf`, `sqrtf`
- `fabsf`, `floorf`, `ceilf`, `roundf`, `truncf`, `fmodf`
- `fminf`, `fmaxf`, `copysignf`

## Use Cases

- **Embedded systems**: Generate C for microcontrollers
- **Game engines**: Integrate with existing C/C++ math libraries (cglm, HandmadeMath)
- **FFI**: Create C code callable from other languages
- **Legacy integration**: Embed in existing C codebases
