//! Standard linalg functions: dot, cross, normalize, length, etc.

use crate::{LinalgFn, Signature, Type, Value};
use num_traits::Float;

// ============================================================================
// Dot product
// ============================================================================

/// Dot product: dot(a, b) -> scalar
pub struct Dot;

impl<T: Float> LinalgFn<T> for Dot {
    fn name(&self) -> &str {
        "dot"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2],
            ret: Type::Scalar,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Scalar,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4],
            ret: Type::Scalar,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec2(a), Value::Vec2(b)) => Value::Scalar(a[0] * b[0] + a[1] * b[1]),
            #[cfg(feature = "3d")]
            (Value::Vec3(a), Value::Vec3(b)) => {
                Value::Scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
            }
            #[cfg(feature = "4d")]
            (Value::Vec4(a), Value::Vec4(b)) => {
                Value::Scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Cross product (3D only)
// ============================================================================

/// Cross product: cross(a, b) -> vec3
#[cfg(feature = "3d")]
pub struct Cross;

#[cfg(feature = "3d")]
impl<T: Float> LinalgFn<T> for Cross {
    fn name(&self) -> &str {
        "cross"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec3(a), Value::Vec3(b)) => Value::Vec3([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Length
// ============================================================================

/// Vector length: length(v) -> scalar
pub struct Length;

impl<T: Float> LinalgFn<T> for Length {
    fn name(&self) -> &str {
        "length"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2],
            ret: Type::Scalar,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3],
            ret: Type::Scalar,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4],
            ret: Type::Scalar,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Vec2(v) => Value::Scalar((v[0] * v[0] + v[1] * v[1]).sqrt()),
            #[cfg(feature = "3d")]
            Value::Vec3(v) => Value::Scalar((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()),
            #[cfg(feature = "4d")]
            Value::Vec4(v) => {
                Value::Scalar((v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt())
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Normalize
// ============================================================================

/// Normalize vector: normalize(v) -> vec (same type, unit length)
pub struct Normalize;

impl<T: Float> LinalgFn<T> for Normalize {
    fn name(&self) -> &str {
        "normalize"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Vec2(v) => {
                let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
                Value::Vec2([v[0] / len, v[1] / len])
            }
            #[cfg(feature = "3d")]
            Value::Vec3(v) => {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                Value::Vec3([v[0] / len, v[1] / len, v[2] / len])
            }
            #[cfg(feature = "4d")]
            Value::Vec4(v) => {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
                Value::Vec4([v[0] / len, v[1] / len, v[2] / len, v[3] / len])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Distance
// ============================================================================

/// Distance between two points: distance(a, b) -> scalar
pub struct Distance;

impl<T: Float> LinalgFn<T> for Distance {
    fn name(&self) -> &str {
        "distance"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2],
            ret: Type::Scalar,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Scalar,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4],
            ret: Type::Scalar,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec2(a), Value::Vec2(b)) => {
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                Value::Scalar((dx * dx + dy * dy).sqrt())
            }
            #[cfg(feature = "3d")]
            (Value::Vec3(a), Value::Vec3(b)) => {
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                let dz = a[2] - b[2];
                Value::Scalar((dx * dx + dy * dy + dz * dz).sqrt())
            }
            #[cfg(feature = "4d")]
            (Value::Vec4(a), Value::Vec4(b)) => {
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                let dz = a[2] - b[2];
                let dw = a[3] - b[3];
                Value::Scalar((dx * dx + dy * dy + dz * dz + dw * dw).sqrt())
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Reflect
// ============================================================================

/// Reflect vector: reflect(incident, normal) -> vec
/// Returns incident - 2 * dot(normal, incident) * normal
pub struct Reflect;

impl<T: Float> LinalgFn<T> for Reflect {
    fn name(&self) -> &str {
        "reflect"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec2(i), Value::Vec2(n)) => {
                let d = i[0] * n[0] + i[1] * n[1];
                let two = T::from(2.0).unwrap();
                Value::Vec2([i[0] - two * d * n[0], i[1] - two * d * n[1]])
            }
            #[cfg(feature = "3d")]
            (Value::Vec3(i), Value::Vec3(n)) => {
                let d = i[0] * n[0] + i[1] * n[1] + i[2] * n[2];
                let two = T::from(2.0).unwrap();
                Value::Vec3([
                    i[0] - two * d * n[0],
                    i[1] - two * d * n[1],
                    i[2] - two * d * n[2],
                ])
            }
            #[cfg(feature = "4d")]
            (Value::Vec4(i), Value::Vec4(n)) => {
                let d = i[0] * n[0] + i[1] * n[1] + i[2] * n[2] + i[3] * n[3];
                let two = T::from(2.0).unwrap();
                Value::Vec4([
                    i[0] - two * d * n[0],
                    i[1] - two * d * n[1],
                    i[2] - two * d * n[2],
                    i[3] - two * d * n[3],
                ])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Hadamard (element-wise multiply)
// ============================================================================

/// Element-wise vector multiply: hadamard(a, b) -> vec
pub struct Hadamard;

impl<T: Float> LinalgFn<T> for Hadamard {
    fn name(&self) -> &str {
        "hadamard"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2([a[0] * b[0], a[1] * b[1]]),
            #[cfg(feature = "3d")]
            (Value::Vec3(a), Value::Vec3(b)) => {
                Value::Vec3([a[0] * b[0], a[1] * b[1], a[2] * b[2]])
            }
            #[cfg(feature = "4d")]
            (Value::Vec4(a), Value::Vec4(b)) => {
                Value::Vec4([a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Lerp (linear interpolation for vectors)
// ============================================================================

/// Linear interpolation: lerp(a, b, t) -> vec
/// Returns a + (b - a) * t
pub struct Lerp;

impl<T: Float> LinalgFn<T> for Lerp {
    fn name(&self) -> &str {
        "lerp"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2, Type::Scalar],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3, Type::Scalar],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4, Type::Scalar],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1], &args[2]) {
            (Value::Vec2(a), Value::Vec2(b), Value::Scalar(t)) => {
                Value::Vec2([a[0] + (b[0] - a[0]) * *t, a[1] + (b[1] - a[1]) * *t])
            }
            #[cfg(feature = "3d")]
            (Value::Vec3(a), Value::Vec3(b), Value::Scalar(t)) => Value::Vec3([
                a[0] + (b[0] - a[0]) * *t,
                a[1] + (b[1] - a[1]) * *t,
                a[2] + (b[2] - a[2]) * *t,
            ]),
            #[cfg(feature = "4d")]
            (Value::Vec4(a), Value::Vec4(b), Value::Scalar(t)) => Value::Vec4([
                a[0] + (b[0] - a[0]) * *t,
                a[1] + (b[1] - a[1]) * *t,
                a[2] + (b[2] - a[2]) * *t,
                a[3] + (b[3] - a[3]) * *t,
            ]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Mix (alias for lerp, GLSL naming)
// ============================================================================

/// Linear interpolation (GLSL naming): mix(a, b, t) -> vec
pub struct Mix;

impl<T: Float> LinalgFn<T> for Mix {
    fn name(&self) -> &str {
        "mix"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2, Type::Scalar],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3, Type::Scalar],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4, Type::Scalar],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1], &args[2]) {
            (Value::Vec2(a), Value::Vec2(b), Value::Scalar(t)) => {
                Value::Vec2([a[0] + (b[0] - a[0]) * *t, a[1] + (b[1] - a[1]) * *t])
            }
            #[cfg(feature = "3d")]
            (Value::Vec3(a), Value::Vec3(b), Value::Scalar(t)) => Value::Vec3([
                a[0] + (b[0] - a[0]) * *t,
                a[1] + (b[1] - a[1]) * *t,
                a[2] + (b[2] - a[2]) * *t,
            ]),
            #[cfg(feature = "4d")]
            (Value::Vec4(a), Value::Vec4(b), Value::Scalar(t)) => Value::Vec4([
                a[0] + (b[0] - a[0]) * *t,
                a[1] + (b[1] - a[1]) * *t,
                a[2] + (b[2] - a[2]) * *t,
                a[3] + (b[3] - a[3]) * *t,
            ]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Vector constructors
// ============================================================================

/// Construct Vec2 from two scalars: vec2(x, y) -> Vec2
pub struct Vec2Constructor;

impl<T: Float> LinalgFn<T> for Vec2Constructor {
    fn name(&self) -> &str {
        "vec2"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Scalar, Type::Scalar],
            ret: Type::Vec2,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Scalar(x), Value::Scalar(y)) => Value::Vec2([*x, *y]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Construct Vec3 from three scalars: vec3(x, y, z) -> Vec3
#[cfg(feature = "3d")]
pub struct Vec3Constructor;

#[cfg(feature = "3d")]
impl<T: Float> LinalgFn<T> for Vec3Constructor {
    fn name(&self) -> &str {
        "vec3"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Scalar, Type::Scalar, Type::Scalar],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1], &args[2]) {
            (Value::Scalar(x), Value::Scalar(y), Value::Scalar(z)) => Value::Vec3([*x, *y, *z]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Construct Vec4 from four scalars: vec4(x, y, z, w) -> Vec4
#[cfg(feature = "4d")]
pub struct Vec4Constructor;

#[cfg(feature = "4d")]
impl<T: Float> LinalgFn<T> for Vec4Constructor {
    fn name(&self) -> &str {
        "vec4"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Scalar, Type::Scalar, Type::Scalar, Type::Scalar],
            ret: Type::Vec4,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1], &args[2], &args[3]) {
            (Value::Scalar(x), Value::Scalar(y), Value::Scalar(z), Value::Scalar(w)) => {
                Value::Vec4([*x, *y, *z, *w])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Matrix constructors
// ============================================================================

/// Construct Mat2 from four scalars (column-major): mat2(c0r0, c0r1, c1r0, c1r1) -> Mat2
pub struct Mat2Constructor;

impl<T: Float> LinalgFn<T> for Mat2Constructor {
    fn name(&self) -> &str {
        "mat2"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Scalar, Type::Scalar, Type::Scalar, Type::Scalar],
            ret: Type::Mat2,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1], &args[2], &args[3]) {
            (Value::Scalar(a), Value::Scalar(b), Value::Scalar(c), Value::Scalar(d)) => {
                Value::Mat2([*a, *b, *c, *d])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Construct Mat3 from nine scalars (column-major): mat3(...) -> Mat3
#[cfg(feature = "3d")]
pub struct Mat3Constructor;

#[cfg(feature = "3d")]
impl<T: Float> LinalgFn<T> for Mat3Constructor {
    fn name(&self) -> &str {
        "mat3"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
            ],
            ret: Type::Mat3,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (
            &args[0], &args[1], &args[2], &args[3], &args[4], &args[5], &args[6], &args[7],
            &args[8],
        ) {
            (
                Value::Scalar(a),
                Value::Scalar(b),
                Value::Scalar(c),
                Value::Scalar(d),
                Value::Scalar(e),
                Value::Scalar(f),
                Value::Scalar(g),
                Value::Scalar(h),
                Value::Scalar(i),
            ) => Value::Mat3([*a, *b, *c, *d, *e, *f, *g, *h, *i]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Construct Mat4 from sixteen scalars (column-major): mat4(...) -> Mat4
#[cfg(feature = "4d")]
pub struct Mat4Constructor;

#[cfg(feature = "4d")]
impl<T: Float> LinalgFn<T> for Mat4Constructor {
    fn name(&self) -> &str {
        "mat4"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
                Type::Scalar,
            ],
            ret: Type::Mat4,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (
            &args[0], &args[1], &args[2], &args[3], &args[4], &args[5], &args[6], &args[7],
            &args[8], &args[9], &args[10], &args[11], &args[12], &args[13], &args[14], &args[15],
        ) {
            (
                Value::Scalar(a),
                Value::Scalar(b),
                Value::Scalar(c),
                Value::Scalar(d),
                Value::Scalar(e),
                Value::Scalar(f),
                Value::Scalar(g),
                Value::Scalar(h),
                Value::Scalar(i),
                Value::Scalar(j),
                Value::Scalar(k),
                Value::Scalar(l),
                Value::Scalar(m),
                Value::Scalar(n),
                Value::Scalar(o),
                Value::Scalar(p),
            ) => Value::Mat4([
                *a, *b, *c, *d, *e, *f, *g, *h, *i, *j, *k, *l, *m, *n, *o, *p,
            ]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Component extraction
// ============================================================================

/// Extract x component: x(v) -> Scalar
pub struct ExtractX;

impl<T: Float> LinalgFn<T> for ExtractX {
    fn name(&self) -> &str {
        "x"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2],
            ret: Type::Scalar,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3],
            ret: Type::Scalar,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4],
            ret: Type::Scalar,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Vec2(v) => Value::Scalar(v[0]),
            #[cfg(feature = "3d")]
            Value::Vec3(v) => Value::Scalar(v[0]),
            #[cfg(feature = "4d")]
            Value::Vec4(v) => Value::Scalar(v[0]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Extract y component: y(v) -> Scalar
pub struct ExtractY;

impl<T: Float> LinalgFn<T> for ExtractY {
    fn name(&self) -> &str {
        "y"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2],
            ret: Type::Scalar,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3],
            ret: Type::Scalar,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4],
            ret: Type::Scalar,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Vec2(v) => Value::Scalar(v[1]),
            #[cfg(feature = "3d")]
            Value::Vec3(v) => Value::Scalar(v[1]),
            #[cfg(feature = "4d")]
            Value::Vec4(v) => Value::Scalar(v[1]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Extract z component: z(v) -> Scalar (Vec3 and Vec4 only)
#[cfg(feature = "3d")]
pub struct ExtractZ;

#[cfg(feature = "3d")]
impl<T: Float> LinalgFn<T> for ExtractZ {
    fn name(&self) -> &str {
        "z"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec3],
            ret: Type::Scalar,
        }];
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4],
            ret: Type::Scalar,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Vec3(v) => Value::Scalar(v[2]),
            #[cfg(feature = "4d")]
            Value::Vec4(v) => Value::Scalar(v[2]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Extract w component: w(v) -> Scalar (Vec4 only)
#[cfg(feature = "4d")]
pub struct ExtractW;

#[cfg(feature = "4d")]
impl<T: Float> LinalgFn<T> for ExtractW {
    fn name(&self) -> &str {
        "w"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec4],
            ret: Type::Scalar,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Vec4(v) => Value::Scalar(v[3]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Vectorized math functions
// ============================================================================

macro_rules! define_vectorized_fn {
    ($name:ident, $fn_name:expr, $method:ident) => {
        pub struct $name;

        impl<T: Float> LinalgFn<T> for $name {
            fn name(&self) -> &str {
                $fn_name
            }

            fn signatures(&self) -> Vec<Signature> {
                let mut sigs = vec![Signature {
                    args: vec![Type::Vec2],
                    ret: Type::Vec2,
                }];
                #[cfg(feature = "3d")]
                sigs.push(Signature {
                    args: vec![Type::Vec3],
                    ret: Type::Vec3,
                });
                #[cfg(feature = "4d")]
                sigs.push(Signature {
                    args: vec![Type::Vec4],
                    ret: Type::Vec4,
                });
                sigs
            }

            fn call(&self, args: &[Value<T>]) -> Value<T> {
                match &args[0] {
                    Value::Vec2(v) => Value::Vec2([v[0].$method(), v[1].$method()]),
                    #[cfg(feature = "3d")]
                    Value::Vec3(v) => Value::Vec3([v[0].$method(), v[1].$method(), v[2].$method()]),
                    #[cfg(feature = "4d")]
                    Value::Vec4(v) => Value::Vec4([
                        v[0].$method(),
                        v[1].$method(),
                        v[2].$method(),
                        v[3].$method(),
                    ]),
                    _ => unreachable!("signature mismatch"),
                }
            }
        }
    };
}

define_vectorized_fn!(VecSin, "sin", sin);
define_vectorized_fn!(VecCos, "cos", cos);
define_vectorized_fn!(VecAbs, "abs", abs);
define_vectorized_fn!(VecFloor, "floor", floor);
define_vectorized_fn!(VecSqrt, "sqrt", sqrt);

/// Vectorized fract: fract(v) -> VecN (fractional part)
pub struct VecFract;

impl<T: Float> LinalgFn<T> for VecFract {
    fn name(&self) -> &str {
        "fract"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Vec2(v) => Value::Vec2([v[0].fract(), v[1].fract()]),
            #[cfg(feature = "3d")]
            Value::Vec3(v) => Value::Vec3([v[0].fract(), v[1].fract(), v[2].fract()]),
            #[cfg(feature = "4d")]
            Value::Vec4(v) => Value::Vec4([v[0].fract(), v[1].fract(), v[2].fract(), v[3].fract()]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Vectorized comparison functions
// ============================================================================

/// Vectorized min: min(a, b) -> VecN (component-wise minimum)
pub struct VecMin;

impl<T: Float> LinalgFn<T> for VecMin {
    fn name(&self) -> &str {
        "min"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2([a[0].min(b[0]), a[1].min(b[1])]),
            #[cfg(feature = "3d")]
            (Value::Vec3(a), Value::Vec3(b)) => {
                Value::Vec3([a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])])
            }
            #[cfg(feature = "4d")]
            (Value::Vec4(a), Value::Vec4(b)) => Value::Vec4([
                a[0].min(b[0]),
                a[1].min(b[1]),
                a[2].min(b[2]),
                a[3].min(b[3]),
            ]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Vectorized max: max(a, b) -> VecN (component-wise maximum)
pub struct VecMax;

impl<T: Float> LinalgFn<T> for VecMax {
    fn name(&self) -> &str {
        "max"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2([a[0].max(b[0]), a[1].max(b[1])]),
            #[cfg(feature = "3d")]
            (Value::Vec3(a), Value::Vec3(b)) => {
                Value::Vec3([a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])])
            }
            #[cfg(feature = "4d")]
            (Value::Vec4(a), Value::Vec4(b)) => Value::Vec4([
                a[0].max(b[0]),
                a[1].max(b[1]),
                a[2].max(b[2]),
                a[3].max(b[3]),
            ]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Vectorized clamp: clamp(x, min, max) -> VecN
pub struct VecClamp;

impl<T: Float> LinalgFn<T> for VecClamp {
    fn name(&self) -> &str {
        "clamp"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2, Type::Vec2],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3, Type::Vec3],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4, Type::Vec4],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1], &args[2]) {
            (Value::Vec2(x), Value::Vec2(lo), Value::Vec2(hi)) => {
                Value::Vec2([x[0].max(lo[0]).min(hi[0]), x[1].max(lo[1]).min(hi[1])])
            }
            #[cfg(feature = "3d")]
            (Value::Vec3(x), Value::Vec3(lo), Value::Vec3(hi)) => Value::Vec3([
                x[0].max(lo[0]).min(hi[0]),
                x[1].max(lo[1]).min(hi[1]),
                x[2].max(lo[2]).min(hi[2]),
            ]),
            #[cfg(feature = "4d")]
            (Value::Vec4(x), Value::Vec4(lo), Value::Vec4(hi)) => Value::Vec4([
                x[0].max(lo[0]).min(hi[0]),
                x[1].max(lo[1]).min(hi[1]),
                x[2].max(lo[2]).min(hi[2]),
                x[3].max(lo[3]).min(hi[3]),
            ]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Interpolation functions
// ============================================================================

/// Vectorized step: step(edge, x) -> VecN (0 if x < edge, 1 otherwise)
pub struct VecStep;

impl<T: Float> LinalgFn<T> for VecStep {
    fn name(&self) -> &str {
        "step"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        fn step<T: Float>(edge: T, x: T) -> T {
            if x < edge { T::zero() } else { T::one() }
        }
        match (&args[0], &args[1]) {
            (Value::Vec2(edge), Value::Vec2(x)) => {
                Value::Vec2([step(edge[0], x[0]), step(edge[1], x[1])])
            }
            #[cfg(feature = "3d")]
            (Value::Vec3(edge), Value::Vec3(x)) => Value::Vec3([
                step(edge[0], x[0]),
                step(edge[1], x[1]),
                step(edge[2], x[2]),
            ]),
            #[cfg(feature = "4d")]
            (Value::Vec4(edge), Value::Vec4(x)) => Value::Vec4([
                step(edge[0], x[0]),
                step(edge[1], x[1]),
                step(edge[2], x[2]),
                step(edge[3], x[3]),
            ]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Vectorized smoothstep: smoothstep(edge0, edge1, x) -> VecN
pub struct VecSmoothstep;

impl<T: Float> LinalgFn<T> for VecSmoothstep {
    fn name(&self) -> &str {
        "smoothstep"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2, Type::Vec2],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3, Type::Vec3],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4, Type::Vec4],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        fn smoothstep<T: Float>(edge0: T, edge1: T, x: T) -> T {
            let t = ((x - edge0) / (edge1 - edge0)).max(T::zero()).min(T::one());
            let three = T::from(3.0).unwrap();
            let two = T::from(2.0).unwrap();
            t * t * (three - two * t)
        }
        match (&args[0], &args[1], &args[2]) {
            (Value::Vec2(e0), Value::Vec2(e1), Value::Vec2(x)) => Value::Vec2([
                smoothstep(e0[0], e1[0], x[0]),
                smoothstep(e0[1], e1[1], x[1]),
            ]),
            #[cfg(feature = "3d")]
            (Value::Vec3(e0), Value::Vec3(e1), Value::Vec3(x)) => Value::Vec3([
                smoothstep(e0[0], e1[0], x[0]),
                smoothstep(e0[1], e1[1], x[1]),
                smoothstep(e0[2], e1[2], x[2]),
            ]),
            #[cfg(feature = "4d")]
            (Value::Vec4(e0), Value::Vec4(e1), Value::Vec4(x)) => Value::Vec4([
                smoothstep(e0[0], e1[0], x[0]),
                smoothstep(e0[1], e1[1], x[1]),
                smoothstep(e0[2], e1[2], x[2]),
                smoothstep(e0[3], e1[3], x[3]),
            ]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Transform functions
// ============================================================================

/// Rotate a 2D vector by an angle: rotate2d(v, angle) -> Vec2
pub struct Rotate2D;

impl<T: Float> LinalgFn<T> for Rotate2D {
    fn name(&self) -> &str {
        "rotate2d"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec2, Type::Scalar],
            ret: Type::Vec2,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec2(v), Value::Scalar(angle)) => {
                let c = angle.cos();
                let s = angle.sin();
                Value::Vec2([v[0] * c - v[1] * s, v[0] * s + v[1] * c])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Rotate a 3D vector around the X axis: rotate_x(v, angle) -> Vec3
#[cfg(feature = "3d")]
pub struct RotateX;

#[cfg(feature = "3d")]
impl<T: Float> LinalgFn<T> for RotateX {
    fn name(&self) -> &str {
        "rotate_x"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Scalar],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec3(v), Value::Scalar(angle)) => {
                let c = angle.cos();
                let s = angle.sin();
                // Rotation around X: [x, y*c - z*s, y*s + z*c]
                Value::Vec3([v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Rotate a 3D vector around the Y axis: rotate_y(v, angle) -> Vec3
#[cfg(feature = "3d")]
pub struct RotateY;

#[cfg(feature = "3d")]
impl<T: Float> LinalgFn<T> for RotateY {
    fn name(&self) -> &str {
        "rotate_y"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Scalar],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec3(v), Value::Scalar(angle)) => {
                let c = angle.cos();
                let s = angle.sin();
                // Rotation around Y: [x*c + z*s, y, -x*s + z*c]
                Value::Vec3([v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Rotate a 3D vector around the Z axis: rotate_z(v, angle) -> Vec3
#[cfg(feature = "3d")]
pub struct RotateZ;

#[cfg(feature = "3d")]
impl<T: Float> LinalgFn<T> for RotateZ {
    fn name(&self) -> &str {
        "rotate_z"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Scalar],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec3(v), Value::Scalar(angle)) => {
                let c = angle.cos();
                let s = angle.sin();
                // Rotation around Z: [x*c - y*s, x*s + y*c, z]
                Value::Vec3([v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Rotate a 3D vector around an arbitrary axis: rotate3d(v, axis, angle) -> Vec3
/// Uses Rodrigues' rotation formula. The axis should be normalized.
#[cfg(feature = "3d")]
pub struct Rotate3D;

#[cfg(feature = "3d")]
impl<T: Float> LinalgFn<T> for Rotate3D {
    fn name(&self) -> &str {
        "rotate3d"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Vec3, Type::Scalar],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1], &args[2]) {
            (Value::Vec3(v), Value::Vec3(axis), Value::Scalar(angle)) => {
                // Rodrigues' rotation formula:
                // v' = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
                let c = angle.cos();
                let s = angle.sin();
                let k = axis; // Assumed normalized

                // k · v (dot product)
                let k_dot_v = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];

                // k × v (cross product)
                let k_cross_v = [
                    k[1] * v[2] - k[2] * v[1],
                    k[2] * v[0] - k[0] * v[2],
                    k[0] * v[1] - k[1] * v[0],
                ];

                let one_minus_c = T::one() - c;

                Value::Vec3([
                    v[0] * c + k_cross_v[0] * s + k[0] * k_dot_v * one_minus_c,
                    v[1] * c + k_cross_v[1] * s + k[1] * k_dot_v * one_minus_c,
                    v[2] * c + k_cross_v[2] * s + k[2] * k_dot_v * one_minus_c,
                ])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Registry helper
// ============================================================================

use crate::FunctionRegistry;

/// Register all standard linalg functions.
pub fn register_linalg<T: Float + 'static>(registry: &mut FunctionRegistry<T>) {
    registry.register(Dot);
    #[cfg(feature = "3d")]
    registry.register(Cross);
    registry.register(Length);
    registry.register(Normalize);
    registry.register(Distance);
    registry.register(Reflect);
    registry.register(Hadamard);
    registry.register(Lerp);
    registry.register(Mix);

    // Vector constructors
    registry.register(Vec2Constructor);
    #[cfg(feature = "3d")]
    registry.register(Vec3Constructor);
    #[cfg(feature = "4d")]
    registry.register(Vec4Constructor);

    // Matrix constructors
    registry.register(Mat2Constructor);
    #[cfg(feature = "3d")]
    registry.register(Mat3Constructor);
    #[cfg(feature = "4d")]
    registry.register(Mat4Constructor);

    // Component extraction
    registry.register(ExtractX);
    registry.register(ExtractY);
    #[cfg(feature = "3d")]
    registry.register(ExtractZ);
    #[cfg(feature = "4d")]
    registry.register(ExtractW);

    // Vectorized math
    registry.register(VecSin);
    registry.register(VecCos);
    registry.register(VecAbs);
    registry.register(VecFloor);
    registry.register(VecFract);
    registry.register(VecSqrt);

    // Vectorized comparison
    registry.register(VecMin);
    registry.register(VecMax);
    registry.register(VecClamp);

    // Interpolation
    registry.register(VecStep);
    registry.register(VecSmoothstep);

    // Transform
    registry.register(Rotate2D);
    #[cfg(feature = "3d")]
    registry.register(RotateX);
    #[cfg(feature = "3d")]
    registry.register(RotateY);
    #[cfg(feature = "3d")]
    registry.register(RotateZ);
    #[cfg(feature = "3d")]
    registry.register(Rotate3D);
}

/// Create a new registry with all standard linalg functions.
pub fn linalg_registry<T: Float + 'static>() -> FunctionRegistry<T> {
    let mut registry = FunctionRegistry::new();
    register_linalg(&mut registry);
    registry
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;
    use std::collections::HashMap;

    fn eval_expr(expr: &str, vars: &[(&str, Value<f32>)]) -> Value<f32> {
        let expr = Expr::parse(expr).unwrap();
        let var_map: HashMap<String, Value<f32>> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let registry = linalg_registry();
        crate::eval(expr.ast(), &var_map, &registry).unwrap()
    }

    #[test]
    fn test_dot_vec2() {
        let result = eval_expr(
            "dot(a, b)",
            &[
                ("a", Value::Vec2([1.0, 2.0])),
                ("b", Value::Vec2([3.0, 4.0])),
            ],
        );
        assert_eq!(result, Value::Scalar(11.0)); // 1*3 + 2*4 = 11
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_dot_vec3() {
        let result = eval_expr(
            "dot(a, b)",
            &[
                ("a", Value::Vec3([1.0, 2.0, 3.0])),
                ("b", Value::Vec3([4.0, 5.0, 6.0])),
            ],
        );
        assert_eq!(result, Value::Scalar(32.0)); // 1*4 + 2*5 + 3*6 = 32
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_cross() {
        let result = eval_expr(
            "cross(a, b)",
            &[
                ("a", Value::Vec3([1.0, 0.0, 0.0])),
                ("b", Value::Vec3([0.0, 1.0, 0.0])),
            ],
        );
        assert_eq!(result, Value::Vec3([0.0, 0.0, 1.0])); // x cross y = z
    }

    #[test]
    fn test_length_vec2() {
        let result = eval_expr("length(v)", &[("v", Value::Vec2([3.0, 4.0]))]);
        assert_eq!(result, Value::Scalar(5.0)); // 3-4-5 triangle
    }

    #[test]
    fn test_normalize_vec2() {
        let result = eval_expr("normalize(v)", &[("v", Value::Vec2([3.0, 4.0]))]);
        if let Value::Vec2(v) = result {
            assert!((v[0] - 0.6).abs() < 0.001);
            assert!((v[1] - 0.8).abs() < 0.001);
        } else {
            panic!("expected Vec2");
        }
    }

    #[test]
    fn test_distance_vec2() {
        let result = eval_expr(
            "distance(a, b)",
            &[
                ("a", Value::Vec2([0.0, 0.0])),
                ("b", Value::Vec2([3.0, 4.0])),
            ],
        );
        assert_eq!(result, Value::Scalar(5.0));
    }

    #[test]
    fn test_reflect_vec2() {
        // Reflect (1, -1) off horizontal surface with normal (0, 1)
        let result = eval_expr(
            "reflect(i, n)",
            &[
                ("i", Value::Vec2([1.0, -1.0])),
                ("n", Value::Vec2([0.0, 1.0])),
            ],
        );
        if let Value::Vec2(v) = result {
            assert!((v[0] - 1.0).abs() < 0.001);
            assert!((v[1] - 1.0).abs() < 0.001);
        } else {
            panic!("expected Vec2");
        }
    }

    #[test]
    fn test_hadamard_vec2() {
        let result = eval_expr(
            "hadamard(a, b)",
            &[
                ("a", Value::Vec2([2.0, 3.0])),
                ("b", Value::Vec2([4.0, 5.0])),
            ],
        );
        assert_eq!(result, Value::Vec2([8.0, 15.0]));
    }

    #[test]
    fn test_lerp_vec2() {
        let result = eval_expr(
            "lerp(a, b, t)",
            &[
                ("a", Value::Vec2([0.0, 0.0])),
                ("b", Value::Vec2([10.0, 20.0])),
                ("t", Value::Scalar(0.5)),
            ],
        );
        assert_eq!(result, Value::Vec2([5.0, 10.0]));
    }

    #[test]
    fn test_mix_vec2() {
        let result = eval_expr(
            "mix(a, b, t)",
            &[
                ("a", Value::Vec2([0.0, 0.0])),
                ("b", Value::Vec2([10.0, 20.0])),
                ("t", Value::Scalar(0.25)),
            ],
        );
        assert_eq!(result, Value::Vec2([2.5, 5.0]));
    }
}
