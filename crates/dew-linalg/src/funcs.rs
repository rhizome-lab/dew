//! Standard linalg functions: dot, cross, normalize, length, etc.

// `mut` is needed when 4d feature is enabled for sigs.push()
#![allow(unused_mut)]

use crate::{LinalgFn, LinalgValue, Signature, Type};
use num_traits::Float;
use rhizome_dew_core::Numeric;

// ============================================================================
// Dot product
// ============================================================================

/// Dot product: dot(a, b) -> scalar
pub struct Dot;

impl<T, V> LinalgFn<T, V> for Dot
where
    T: Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        match (args[0].typ(), args[1].typ()) {
            (Type::Vec2, Type::Vec2) => {
                let a = args[0].as_vec2().unwrap();
                let b = args[1].as_vec2().unwrap();
                V::from_scalar(a[0] * b[0] + a[1] * b[1])
            }
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Vec3) => {
                let a = args[0].as_vec3().unwrap();
                let b = args[1].as_vec3().unwrap();
                V::from_scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
            }
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Vec4) => {
                let a = args[0].as_vec4().unwrap();
                let b = args[1].as_vec4().unwrap();
                V::from_scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
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
impl<T, V> LinalgFn<T, V> for Cross
where
    T: Numeric,
    V: LinalgValue<T>,
{
    fn name(&self) -> &str {
        "cross"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let a = args[0].as_vec3().unwrap();
        let b = args[1].as_vec3().unwrap();
        V::from_vec3([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])
    }
}

// ============================================================================
// Length
// ============================================================================

/// Vector length: length(v) -> scalar
pub struct Length;

impl<T, V> LinalgFn<T, V> for Length
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        match args[0].typ() {
            Type::Vec2 => {
                let v = args[0].as_vec2().unwrap();
                V::from_scalar((v[0] * v[0] + v[1] * v[1]).sqrt())
            }
            #[cfg(feature = "3d")]
            Type::Vec3 => {
                let v = args[0].as_vec3().unwrap();
                V::from_scalar((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            }
            #[cfg(feature = "4d")]
            Type::Vec4 => {
                let v = args[0].as_vec4().unwrap();
                V::from_scalar((v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt())
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

impl<T, V> LinalgFn<T, V> for Normalize
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        match args[0].typ() {
            Type::Vec2 => {
                let v = args[0].as_vec2().unwrap();
                let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
                V::from_vec2([v[0] / len, v[1] / len])
            }
            #[cfg(feature = "3d")]
            Type::Vec3 => {
                let v = args[0].as_vec3().unwrap();
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                V::from_vec3([v[0] / len, v[1] / len, v[2] / len])
            }
            #[cfg(feature = "4d")]
            Type::Vec4 => {
                let v = args[0].as_vec4().unwrap();
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
                V::from_vec4([v[0] / len, v[1] / len, v[2] / len, v[3] / len])
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

impl<T, V> LinalgFn<T, V> for Distance
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        match (args[0].typ(), args[1].typ()) {
            (Type::Vec2, Type::Vec2) => {
                let a = args[0].as_vec2().unwrap();
                let b = args[1].as_vec2().unwrap();
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                V::from_scalar((dx * dx + dy * dy).sqrt())
            }
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Vec3) => {
                let a = args[0].as_vec3().unwrap();
                let b = args[1].as_vec3().unwrap();
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                let dz = a[2] - b[2];
                V::from_scalar((dx * dx + dy * dy + dz * dz).sqrt())
            }
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Vec4) => {
                let a = args[0].as_vec4().unwrap();
                let b = args[1].as_vec4().unwrap();
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                let dz = a[2] - b[2];
                let dw = a[3] - b[3];
                V::from_scalar((dx * dx + dy * dy + dz * dz + dw * dw).sqrt())
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

impl<T, V> LinalgFn<T, V> for Reflect
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        let two = T::from(2.0).unwrap();
        match (args[0].typ(), args[1].typ()) {
            (Type::Vec2, Type::Vec2) => {
                let i = args[0].as_vec2().unwrap();
                let n = args[1].as_vec2().unwrap();
                let d = i[0] * n[0] + i[1] * n[1];
                V::from_vec2([i[0] - two * d * n[0], i[1] - two * d * n[1]])
            }
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Vec3) => {
                let i = args[0].as_vec3().unwrap();
                let n = args[1].as_vec3().unwrap();
                let d = i[0] * n[0] + i[1] * n[1] + i[2] * n[2];
                V::from_vec3([
                    i[0] - two * d * n[0],
                    i[1] - two * d * n[1],
                    i[2] - two * d * n[2],
                ])
            }
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Vec4) => {
                let i = args[0].as_vec4().unwrap();
                let n = args[1].as_vec4().unwrap();
                let d = i[0] * n[0] + i[1] * n[1] + i[2] * n[2] + i[3] * n[3];
                V::from_vec4([
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

impl<T, V> LinalgFn<T, V> for Hadamard
where
    T: Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        match (args[0].typ(), args[1].typ()) {
            (Type::Vec2, Type::Vec2) => {
                let a = args[0].as_vec2().unwrap();
                let b = args[1].as_vec2().unwrap();
                V::from_vec2([a[0] * b[0], a[1] * b[1]])
            }
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Vec3) => {
                let a = args[0].as_vec3().unwrap();
                let b = args[1].as_vec3().unwrap();
                V::from_vec3([a[0] * b[0], a[1] * b[1], a[2] * b[2]])
            }
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Vec4) => {
                let a = args[0].as_vec4().unwrap();
                let b = args[1].as_vec4().unwrap();
                V::from_vec4([a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]])
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

impl<T, V> LinalgFn<T, V> for Lerp
where
    T: Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        let t = args[2].as_scalar().unwrap();
        match (args[0].typ(), args[1].typ()) {
            (Type::Vec2, Type::Vec2) => {
                let a = args[0].as_vec2().unwrap();
                let b = args[1].as_vec2().unwrap();
                V::from_vec2([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
            }
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Vec3) => {
                let a = args[0].as_vec3().unwrap();
                let b = args[1].as_vec3().unwrap();
                V::from_vec3([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                ])
            }
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Vec4) => {
                let a = args[0].as_vec4().unwrap();
                let b = args[1].as_vec4().unwrap();
                V::from_vec4([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                    a[3] + (b[3] - a[3]) * t,
                ])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Mix (alias for lerp, GLSL naming)
// ============================================================================

/// Linear interpolation (GLSL naming): mix(a, b, t) -> vec
pub struct Mix;

impl<T, V> LinalgFn<T, V> for Mix
where
    T: Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        let t = args[2].as_scalar().unwrap();
        match (args[0].typ(), args[1].typ()) {
            (Type::Vec2, Type::Vec2) => {
                let a = args[0].as_vec2().unwrap();
                let b = args[1].as_vec2().unwrap();
                V::from_vec2([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
            }
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Vec3) => {
                let a = args[0].as_vec3().unwrap();
                let b = args[1].as_vec3().unwrap();
                V::from_vec3([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                ])
            }
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Vec4) => {
                let a = args[0].as_vec4().unwrap();
                let b = args[1].as_vec4().unwrap();
                V::from_vec4([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                    a[3] + (b[3] - a[3]) * t,
                ])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Vector constructors
// ============================================================================

/// Construct Vec2 from two scalars: vec2(x, y) -> Vec2
pub struct Vec2Constructor;

impl<T, V> LinalgFn<T, V> for Vec2Constructor
where
    T: Numeric,
    V: LinalgValue<T>,
{
    fn name(&self) -> &str {
        "vec2"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Scalar, Type::Scalar],
            ret: Type::Vec2,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let x = args[0].as_scalar().unwrap();
        let y = args[1].as_scalar().unwrap();
        V::from_vec2([x, y])
    }
}

/// Construct Vec3 from three scalars: vec3(x, y, z) -> Vec3
#[cfg(feature = "3d")]
pub struct Vec3Constructor;

#[cfg(feature = "3d")]
impl<T, V> LinalgFn<T, V> for Vec3Constructor
where
    T: Numeric,
    V: LinalgValue<T>,
{
    fn name(&self) -> &str {
        "vec3"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Scalar, Type::Scalar, Type::Scalar],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let x = args[0].as_scalar().unwrap();
        let y = args[1].as_scalar().unwrap();
        let z = args[2].as_scalar().unwrap();
        V::from_vec3([x, y, z])
    }
}

/// Construct Vec4 from four scalars: vec4(x, y, z, w) -> Vec4
#[cfg(feature = "4d")]
pub struct Vec4Constructor;

#[cfg(feature = "4d")]
impl<T, V> LinalgFn<T, V> for Vec4Constructor
where
    T: Numeric,
    V: LinalgValue<T>,
{
    fn name(&self) -> &str {
        "vec4"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Scalar, Type::Scalar, Type::Scalar, Type::Scalar],
            ret: Type::Vec4,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let x = args[0].as_scalar().unwrap();
        let y = args[1].as_scalar().unwrap();
        let z = args[2].as_scalar().unwrap();
        let w = args[3].as_scalar().unwrap();
        V::from_vec4([x, y, z, w])
    }
}

// ============================================================================
// Matrix constructors
// ============================================================================

/// Construct Mat2 from four scalars (column-major): mat2(c0r0, c0r1, c1r0, c1r1) -> Mat2
pub struct Mat2Constructor;

impl<T, V> LinalgFn<T, V> for Mat2Constructor
where
    T: Numeric,
    V: LinalgValue<T>,
{
    fn name(&self) -> &str {
        "mat2"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Scalar, Type::Scalar, Type::Scalar, Type::Scalar],
            ret: Type::Mat2,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let a = args[0].as_scalar().unwrap();
        let b = args[1].as_scalar().unwrap();
        let c = args[2].as_scalar().unwrap();
        let d = args[3].as_scalar().unwrap();
        V::from_mat2([a, b, c, d])
    }
}

/// Construct Mat3 from nine scalars (column-major): mat3(...) -> Mat3
#[cfg(feature = "3d")]
pub struct Mat3Constructor;

#[cfg(feature = "3d")]
impl<T, V> LinalgFn<T, V> for Mat3Constructor
where
    T: Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        V::from_mat3([
            args[0].as_scalar().unwrap(),
            args[1].as_scalar().unwrap(),
            args[2].as_scalar().unwrap(),
            args[3].as_scalar().unwrap(),
            args[4].as_scalar().unwrap(),
            args[5].as_scalar().unwrap(),
            args[6].as_scalar().unwrap(),
            args[7].as_scalar().unwrap(),
            args[8].as_scalar().unwrap(),
        ])
    }
}

/// Construct Mat4 from sixteen scalars (column-major): mat4(...) -> Mat4
#[cfg(feature = "4d")]
pub struct Mat4Constructor;

#[cfg(feature = "4d")]
impl<T, V> LinalgFn<T, V> for Mat4Constructor
where
    T: Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        V::from_mat4([
            args[0].as_scalar().unwrap(),
            args[1].as_scalar().unwrap(),
            args[2].as_scalar().unwrap(),
            args[3].as_scalar().unwrap(),
            args[4].as_scalar().unwrap(),
            args[5].as_scalar().unwrap(),
            args[6].as_scalar().unwrap(),
            args[7].as_scalar().unwrap(),
            args[8].as_scalar().unwrap(),
            args[9].as_scalar().unwrap(),
            args[10].as_scalar().unwrap(),
            args[11].as_scalar().unwrap(),
            args[12].as_scalar().unwrap(),
            args[13].as_scalar().unwrap(),
            args[14].as_scalar().unwrap(),
            args[15].as_scalar().unwrap(),
        ])
    }
}

// ============================================================================
// Component extraction
// ============================================================================

/// Extract x component: x(v) -> Scalar
pub struct ExtractX;

impl<T, V> LinalgFn<T, V> for ExtractX
where
    T: Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        match args[0].typ() {
            Type::Vec2 => V::from_scalar(args[0].as_vec2().unwrap()[0]),
            #[cfg(feature = "3d")]
            Type::Vec3 => V::from_scalar(args[0].as_vec3().unwrap()[0]),
            #[cfg(feature = "4d")]
            Type::Vec4 => V::from_scalar(args[0].as_vec4().unwrap()[0]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Extract y component: y(v) -> Scalar
pub struct ExtractY;

impl<T, V> LinalgFn<T, V> for ExtractY
where
    T: Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        match args[0].typ() {
            Type::Vec2 => V::from_scalar(args[0].as_vec2().unwrap()[1]),
            #[cfg(feature = "3d")]
            Type::Vec3 => V::from_scalar(args[0].as_vec3().unwrap()[1]),
            #[cfg(feature = "4d")]
            Type::Vec4 => V::from_scalar(args[0].as_vec4().unwrap()[1]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Extract z component: z(v) -> Scalar (Vec3 and Vec4 only)
#[cfg(feature = "3d")]
pub struct ExtractZ;

#[cfg(feature = "3d")]
impl<T, V> LinalgFn<T, V> for ExtractZ
where
    T: Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        match args[0].typ() {
            Type::Vec3 => V::from_scalar(args[0].as_vec3().unwrap()[2]),
            #[cfg(feature = "4d")]
            Type::Vec4 => V::from_scalar(args[0].as_vec4().unwrap()[2]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Extract w component: w(v) -> Scalar (Vec4 only)
#[cfg(feature = "4d")]
pub struct ExtractW;

#[cfg(feature = "4d")]
impl<T, V> LinalgFn<T, V> for ExtractW
where
    T: Numeric,
    V: LinalgValue<T>,
{
    fn name(&self) -> &str {
        "w"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec4],
            ret: Type::Scalar,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        V::from_scalar(args[0].as_vec4().unwrap()[3])
    }
}

// ============================================================================
// Vectorized math functions
// ============================================================================

macro_rules! define_vectorized_fn {
    ($name:ident, $fn_name:expr, $method:ident) => {
        pub struct $name;

        impl<T, V> LinalgFn<T, V> for $name
        where
            T: Float + Numeric,
            V: LinalgValue<T>,
        {
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

            fn call(&self, args: &[V]) -> V {
                match args[0].typ() {
                    Type::Vec2 => {
                        let v = args[0].as_vec2().unwrap();
                        V::from_vec2([v[0].$method(), v[1].$method()])
                    }
                    #[cfg(feature = "3d")]
                    Type::Vec3 => {
                        let v = args[0].as_vec3().unwrap();
                        V::from_vec3([v[0].$method(), v[1].$method(), v[2].$method()])
                    }
                    #[cfg(feature = "4d")]
                    Type::Vec4 => {
                        let v = args[0].as_vec4().unwrap();
                        V::from_vec4([
                            v[0].$method(),
                            v[1].$method(),
                            v[2].$method(),
                            v[3].$method(),
                        ])
                    }
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

impl<T, V> LinalgFn<T, V> for VecFract
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        match args[0].typ() {
            Type::Vec2 => {
                let v = args[0].as_vec2().unwrap();
                V::from_vec2([v[0].fract(), v[1].fract()])
            }
            #[cfg(feature = "3d")]
            Type::Vec3 => {
                let v = args[0].as_vec3().unwrap();
                V::from_vec3([v[0].fract(), v[1].fract(), v[2].fract()])
            }
            #[cfg(feature = "4d")]
            Type::Vec4 => {
                let v = args[0].as_vec4().unwrap();
                V::from_vec4([v[0].fract(), v[1].fract(), v[2].fract(), v[3].fract()])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Vectorized comparison functions
// ============================================================================

/// Vectorized min: min(a, b) -> VecN (component-wise minimum)
pub struct VecMin;

impl<T, V> LinalgFn<T, V> for VecMin
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        match (args[0].typ(), args[1].typ()) {
            (Type::Vec2, Type::Vec2) => {
                let a = args[0].as_vec2().unwrap();
                let b = args[1].as_vec2().unwrap();
                V::from_vec2([a[0].min(b[0]), a[1].min(b[1])])
            }
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Vec3) => {
                let a = args[0].as_vec3().unwrap();
                let b = args[1].as_vec3().unwrap();
                V::from_vec3([a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])])
            }
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Vec4) => {
                let a = args[0].as_vec4().unwrap();
                let b = args[1].as_vec4().unwrap();
                V::from_vec4([
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].min(b[2]),
                    a[3].min(b[3]),
                ])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Vectorized max: max(a, b) -> VecN (component-wise maximum)
pub struct VecMax;

impl<T, V> LinalgFn<T, V> for VecMax
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        match (args[0].typ(), args[1].typ()) {
            (Type::Vec2, Type::Vec2) => {
                let a = args[0].as_vec2().unwrap();
                let b = args[1].as_vec2().unwrap();
                V::from_vec2([a[0].max(b[0]), a[1].max(b[1])])
            }
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Vec3) => {
                let a = args[0].as_vec3().unwrap();
                let b = args[1].as_vec3().unwrap();
                V::from_vec3([a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])])
            }
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Vec4) => {
                let a = args[0].as_vec4().unwrap();
                let b = args[1].as_vec4().unwrap();
                V::from_vec4([
                    a[0].max(b[0]),
                    a[1].max(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Vectorized clamp: clamp(x, min, max) -> VecN
pub struct VecClamp;

impl<T, V> LinalgFn<T, V> for VecClamp
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        match (args[0].typ(), args[1].typ(), args[2].typ()) {
            (Type::Vec2, Type::Vec2, Type::Vec2) => {
                let x = args[0].as_vec2().unwrap();
                let lo = args[1].as_vec2().unwrap();
                let hi = args[2].as_vec2().unwrap();
                V::from_vec2([x[0].max(lo[0]).min(hi[0]), x[1].max(lo[1]).min(hi[1])])
            }
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Vec3, Type::Vec3) => {
                let x = args[0].as_vec3().unwrap();
                let lo = args[1].as_vec3().unwrap();
                let hi = args[2].as_vec3().unwrap();
                V::from_vec3([
                    x[0].max(lo[0]).min(hi[0]),
                    x[1].max(lo[1]).min(hi[1]),
                    x[2].max(lo[2]).min(hi[2]),
                ])
            }
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Vec4, Type::Vec4) => {
                let x = args[0].as_vec4().unwrap();
                let lo = args[1].as_vec4().unwrap();
                let hi = args[2].as_vec4().unwrap();
                V::from_vec4([
                    x[0].max(lo[0]).min(hi[0]),
                    x[1].max(lo[1]).min(hi[1]),
                    x[2].max(lo[2]).min(hi[2]),
                    x[3].max(lo[3]).min(hi[3]),
                ])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Interpolation functions
// ============================================================================

/// Vectorized step: step(edge, x) -> VecN (0 if x < edge, 1 otherwise)
pub struct VecStep;

impl<T, V> LinalgFn<T, V> for VecStep
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        fn step<T: Float>(edge: T, x: T) -> T {
            if x < edge { T::zero() } else { T::one() }
        }
        match (args[0].typ(), args[1].typ()) {
            (Type::Vec2, Type::Vec2) => {
                let edge = args[0].as_vec2().unwrap();
                let x = args[1].as_vec2().unwrap();
                V::from_vec2([step(edge[0], x[0]), step(edge[1], x[1])])
            }
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Vec3) => {
                let edge = args[0].as_vec3().unwrap();
                let x = args[1].as_vec3().unwrap();
                V::from_vec3([
                    step(edge[0], x[0]),
                    step(edge[1], x[1]),
                    step(edge[2], x[2]),
                ])
            }
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Vec4) => {
                let edge = args[0].as_vec4().unwrap();
                let x = args[1].as_vec4().unwrap();
                V::from_vec4([
                    step(edge[0], x[0]),
                    step(edge[1], x[1]),
                    step(edge[2], x[2]),
                    step(edge[3], x[3]),
                ])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

/// Vectorized smoothstep: smoothstep(edge0, edge1, x) -> VecN
pub struct VecSmoothstep;

impl<T, V> LinalgFn<T, V> for VecSmoothstep
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
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

    fn call(&self, args: &[V]) -> V {
        fn smoothstep<T: Float>(edge0: T, edge1: T, x: T) -> T {
            let t = ((x - edge0) / (edge1 - edge0)).max(T::zero()).min(T::one());
            let three = T::from(3.0).unwrap();
            let two = T::from(2.0).unwrap();
            t * t * (three - two * t)
        }
        match (args[0].typ(), args[1].typ(), args[2].typ()) {
            (Type::Vec2, Type::Vec2, Type::Vec2) => {
                let e0 = args[0].as_vec2().unwrap();
                let e1 = args[1].as_vec2().unwrap();
                let x = args[2].as_vec2().unwrap();
                V::from_vec2([
                    smoothstep(e0[0], e1[0], x[0]),
                    smoothstep(e0[1], e1[1], x[1]),
                ])
            }
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Vec3, Type::Vec3) => {
                let e0 = args[0].as_vec3().unwrap();
                let e1 = args[1].as_vec3().unwrap();
                let x = args[2].as_vec3().unwrap();
                V::from_vec3([
                    smoothstep(e0[0], e1[0], x[0]),
                    smoothstep(e0[1], e1[1], x[1]),
                    smoothstep(e0[2], e1[2], x[2]),
                ])
            }
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Vec4, Type::Vec4) => {
                let e0 = args[0].as_vec4().unwrap();
                let e1 = args[1].as_vec4().unwrap();
                let x = args[2].as_vec4().unwrap();
                V::from_vec4([
                    smoothstep(e0[0], e1[0], x[0]),
                    smoothstep(e0[1], e1[1], x[1]),
                    smoothstep(e0[2], e1[2], x[2]),
                    smoothstep(e0[3], e1[3], x[3]),
                ])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Transform functions
// ============================================================================

/// Rotate a 2D vector by an angle: rotate2d(v, angle) -> Vec2
pub struct Rotate2D;

impl<T, V> LinalgFn<T, V> for Rotate2D
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
    fn name(&self) -> &str {
        "rotate2d"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec2, Type::Scalar],
            ret: Type::Vec2,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let v = args[0].as_vec2().unwrap();
        let angle = args[1].as_scalar().unwrap();
        let c = angle.cos();
        let s = angle.sin();
        V::from_vec2([v[0] * c - v[1] * s, v[0] * s + v[1] * c])
    }
}

/// Rotate a 3D vector around the X axis: rotate_x(v, angle) -> Vec3
#[cfg(feature = "3d")]
pub struct RotateX;

#[cfg(feature = "3d")]
impl<T, V> LinalgFn<T, V> for RotateX
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
    fn name(&self) -> &str {
        "rotate_x"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Scalar],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let v = args[0].as_vec3().unwrap();
        let angle = args[1].as_scalar().unwrap();
        let c = angle.cos();
        let s = angle.sin();
        // Rotation around X: [x, y*c - z*s, y*s + z*c]
        V::from_vec3([v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c])
    }
}

/// Rotate a 3D vector around the Y axis: rotate_y(v, angle) -> Vec3
#[cfg(feature = "3d")]
pub struct RotateY;

#[cfg(feature = "3d")]
impl<T, V> LinalgFn<T, V> for RotateY
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
    fn name(&self) -> &str {
        "rotate_y"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Scalar],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let v = args[0].as_vec3().unwrap();
        let angle = args[1].as_scalar().unwrap();
        let c = angle.cos();
        let s = angle.sin();
        // Rotation around Y: [x*c + z*s, y, -x*s + z*c]
        V::from_vec3([v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c])
    }
}

/// Rotate a 3D vector around the Z axis: rotate_z(v, angle) -> Vec3
#[cfg(feature = "3d")]
pub struct RotateZ;

#[cfg(feature = "3d")]
impl<T, V> LinalgFn<T, V> for RotateZ
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
    fn name(&self) -> &str {
        "rotate_z"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Scalar],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let v = args[0].as_vec3().unwrap();
        let angle = args[1].as_scalar().unwrap();
        let c = angle.cos();
        let s = angle.sin();
        // Rotation around Z: [x*c - y*s, x*s + y*c, z]
        V::from_vec3([v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]])
    }
}

/// Rotate a 3D vector around an arbitrary axis: rotate3d(v, axis, angle) -> Vec3
/// Uses Rodrigues' rotation formula. The axis should be normalized.
#[cfg(feature = "3d")]
pub struct Rotate3D;

#[cfg(feature = "3d")]
impl<T, V> LinalgFn<T, V> for Rotate3D
where
    T: Float + Numeric,
    V: LinalgValue<T>,
{
    fn name(&self) -> &str {
        "rotate3d"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Vec3, Type::Scalar],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[V]) -> V {
        let v = args[0].as_vec3().unwrap();
        let k = args[1].as_vec3().unwrap(); // Assumed normalized
        let angle = args[2].as_scalar().unwrap();

        // Rodrigues' rotation formula:
        // v' = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
        let c = angle.cos();
        let s = angle.sin();

        // k · v (dot product)
        let k_dot_v = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];

        // k × v (cross product)
        let k_cross_v = [
            k[1] * v[2] - k[2] * v[1],
            k[2] * v[0] - k[0] * v[2],
            k[0] * v[1] - k[1] * v[0],
        ];

        let one_minus_c = T::one() - c;

        V::from_vec3([
            v[0] * c + k_cross_v[0] * s + k[0] * k_dot_v * one_minus_c,
            v[1] * c + k_cross_v[1] * s + k[1] * k_dot_v * one_minus_c,
            v[2] * c + k_cross_v[2] * s + k[2] * k_dot_v * one_minus_c,
        ])
    }
}

// ============================================================================
// Registry helper
// ============================================================================

use crate::{FunctionRegistry, Value};

/// Register all standard linalg functions.
pub fn register_linalg<T, V>(registry: &mut FunctionRegistry<T, V>)
where
    T: Float + Numeric + 'static,
    V: LinalgValue<T> + 'static,
{
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

/// Create a new registry with all standard linalg functions using the default Value type.
pub fn linalg_registry<T: Float + Numeric + 'static>() -> FunctionRegistry<T, Value<T>> {
    let mut registry = FunctionRegistry::new();
    register_linalg(&mut registry);
    registry
}

/// Register only Numeric-compatible linalg functions (no sqrt, trig, etc.).
///
/// This is useful for integer vector math where Float methods aren't available.
/// Includes: dot, cross, hadamard, lerp, mix, constructors, extractors.
pub fn register_linalg_numeric<T, V>(registry: &mut FunctionRegistry<T, V>)
where
    T: Numeric + 'static,
    V: LinalgValue<T> + 'static,
{
    // Basic operations
    registry.register(Dot);
    #[cfg(feature = "3d")]
    registry.register(Cross);
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
}

/// Create a registry with Numeric-compatible functions for integer vectors.
///
/// Use this for `i32` or `i64` vector math. For float vectors, use `linalg_registry()`.
pub fn linalg_registry_int<T: Numeric + 'static>() -> FunctionRegistry<T, Value<T>> {
    let mut registry = FunctionRegistry::new();
    register_linalg_numeric(&mut registry);
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

    #[test]
    fn test_integer_vectors() {
        use rhizome_dew_core::Expr;

        // Create integer registry
        let registry: crate::FunctionRegistry<i32, Value<i32>> = linalg_registry_int();

        // Test dot product with integers
        let expr = Expr::parse("dot(a, b)").unwrap();
        let vars: HashMap<String, Value<i32>> = [
            ("a".to_string(), Value::Vec2([1, 2])),
            ("b".to_string(), Value::Vec2([3, 4])),
        ]
        .into();
        let result = crate::eval(expr.ast(), &vars, &registry).unwrap();
        assert_eq!(result, Value::Scalar(11)); // 1*3 + 2*4 = 11

        // Test vec2 constructor
        let expr = Expr::parse("vec2(x, y)").unwrap();
        let vars: HashMap<String, Value<i32>> = [
            ("x".to_string(), Value::Scalar(5)),
            ("y".to_string(), Value::Scalar(7)),
        ]
        .into();
        let result = crate::eval(expr.ast(), &vars, &registry).unwrap();
        assert_eq!(result, Value::Vec2([5, 7]));

        // Test hadamard (element-wise multiply)
        let expr = Expr::parse("hadamard(a, b)").unwrap();
        let vars: HashMap<String, Value<i32>> = [
            ("a".to_string(), Value::Vec2([2, 3])),
            ("b".to_string(), Value::Vec2([4, 5])),
        ]
        .into();
        let result = crate::eval(expr.ast(), &vars, &registry).unwrap();
        assert_eq!(result, Value::Vec2([8, 15]));
    }
}
