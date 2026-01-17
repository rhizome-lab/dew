//! Combined domain crate for dew expressions.
//!
//! Provides a unified `Value` type that implements all domain traits,
//! enabling seamless composition of linalg, complex, and quaternion operations.
//!
//! # Quick Start
//!
//! ```
//! use rhizome_dew_all::Value;
//!
//! // The combined Value can hold any domain type
//! let scalar: Value<f32> = Value::Scalar(1.0);
//! let vec3: Value<f32> = Value::Vec3([1.0, 2.0, 3.0]);
//! let complex: Value<f32> = Value::Complex([1.0, 2.0]);
//! let quat: Value<f32> = Value::Quaternion([0.0, 0.0, 0.0, 1.0]);
//! ```
//!
//! # Using with domain evals
//!
//! The `Value<T>` type implements all domain traits, so you can pass it to
//! domain-specific `eval` functions:
//!
//! ```
//! use rhizome_dew_all::Value;
//! use rhizome_dew_core::Expr;
//! use std::collections::HashMap;
//!
//! // Use linalg eval with combined Value
//! # #[cfg(feature = "linalg")]
//! # {
//! use rhizome_dew_linalg::{eval, linalg_registry, LinalgValue};
//!
//! let expr = Expr::parse("dot(a, b)").unwrap();
//! let vars: HashMap<String, Value<f32>> = [
//!     ("a".into(), Value::Vec3([1.0, 0.0, 0.0])),
//!     ("b".into(), Value::Vec3([1.0, 0.0, 0.0])),
//! ].into();
//!
//! // Create a registry for our combined Value type
//! let mut registry = rhizome_dew_linalg::FunctionRegistry::new();
//! rhizome_dew_linalg::register_linalg(&mut registry);
//!
//! let result = eval(expr.ast(), &vars, &registry).unwrap();
//! assert_eq!(result, Value::Scalar(1.0));
//! # }
//! ```
//!
//! # Features
//!
//! | Feature      | Description                           |
//! |--------------|---------------------------------------|
//! | `scalar`     | Include scalar math functions         |
//! | `linalg`     | Include linear algebra types          |
//! | `complex`    | Include complex number operations     |
//! | `quaternion` | Include quaternion operations         |
//! | `wgsl`       | WGSL shader code generation           |
//! | `glsl`       | GLSL shader code generation           |
//! | `lua`        | Lua code generation                   |
//! | `cranelift`  | Cranelift JIT compilation             |
//! | `optimize`   | Expression optimization passes        |

use num_traits::Float;

// Re-export domain crates for convenience
#[cfg(feature = "scalar")]
pub use rhizome_dew_scalar;

#[cfg(feature = "linalg")]
pub use rhizome_dew_linalg;

#[cfg(feature = "complex")]
pub use rhizome_dew_complex;

#[cfg(feature = "quaternion")]
pub use rhizome_dew_quaternion;

// ============================================================================
// Combined Type enum
// ============================================================================

/// Type of a combined value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    /// Real scalar.
    Scalar,
    /// 2D vector [x, y].
    Vec2,
    /// 3D vector [x, y, z].
    Vec3,
    /// 4D vector [x, y, z, w].
    Vec4,
    /// 2x2 matrix.
    Mat2,
    /// 3x3 matrix.
    Mat3,
    /// 4x4 matrix.
    Mat4,
    /// Complex number [re, im].
    Complex,
    /// Quaternion [x, y, z, w].
    Quaternion,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Scalar => write!(f, "scalar"),
            Type::Vec2 => write!(f, "vec2"),
            Type::Vec3 => write!(f, "vec3"),
            Type::Vec4 => write!(f, "vec4"),
            Type::Mat2 => write!(f, "mat2"),
            Type::Mat3 => write!(f, "mat3"),
            Type::Mat4 => write!(f, "mat4"),
            Type::Complex => write!(f, "complex"),
            Type::Quaternion => write!(f, "quaternion"),
        }
    }
}

// ============================================================================
// Combined Value type
// ============================================================================

/// Combined value type for all domains.
///
/// This type implements all domain value traits (`LinalgValue`, `ComplexValue`,
/// `QuaternionValue`), enabling seamless interoperation between domains.
///
/// Vec3 is shared between linalg and quaternion domains, providing natural
/// composition for 3D graphics and physics applications.
#[derive(Debug, Clone, PartialEq)]
pub enum Value<T> {
    /// Real scalar.
    Scalar(T),
    /// 2D vector [x, y].
    Vec2([T; 2]),
    /// 3D vector [x, y, z] (shared by linalg + quaternion).
    Vec3([T; 3]),
    /// 4D vector [x, y, z, w].
    Vec4([T; 4]),
    /// 2x2 matrix (column-major).
    Mat2([T; 4]),
    /// 3x3 matrix (column-major).
    Mat3([T; 9]),
    /// 4x4 matrix (column-major).
    Mat4([T; 16]),
    /// Complex number [re, im].
    Complex([T; 2]),
    /// Quaternion [x, y, z, w] (scalar last).
    Quaternion([T; 4]),
}

impl<T> Value<T> {
    /// Returns the type of this value.
    pub fn typ(&self) -> Type {
        match self {
            Value::Scalar(_) => Type::Scalar,
            Value::Vec2(_) => Type::Vec2,
            Value::Vec3(_) => Type::Vec3,
            Value::Vec4(_) => Type::Vec4,
            Value::Mat2(_) => Type::Mat2,
            Value::Mat3(_) => Type::Mat3,
            Value::Mat4(_) => Type::Mat4,
            Value::Complex(_) => Type::Complex,
            Value::Quaternion(_) => Type::Quaternion,
        }
    }
}

impl<T: Copy> Value<T> {
    /// Try to get as scalar.
    pub fn as_scalar(&self) -> Option<T> {
        match self {
            Value::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as vec2.
    pub fn as_vec2(&self) -> Option<[T; 2]> {
        match self {
            Value::Vec2(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as vec3.
    pub fn as_vec3(&self) -> Option<[T; 3]> {
        match self {
            Value::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as vec4.
    pub fn as_vec4(&self) -> Option<[T; 4]> {
        match self {
            Value::Vec4(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as mat2.
    pub fn as_mat2(&self) -> Option<[T; 4]> {
        match self {
            Value::Mat2(m) => Some(*m),
            _ => None,
        }
    }

    /// Try to get as mat3.
    pub fn as_mat3(&self) -> Option<[T; 9]> {
        match self {
            Value::Mat3(m) => Some(*m),
            _ => None,
        }
    }

    /// Try to get as mat4.
    pub fn as_mat4(&self) -> Option<[T; 16]> {
        match self {
            Value::Mat4(m) => Some(*m),
            _ => None,
        }
    }

    /// Try to get as complex.
    pub fn as_complex(&self) -> Option<[T; 2]> {
        match self {
            Value::Complex(c) => Some(*c),
            _ => None,
        }
    }

    /// Try to get as quaternion.
    pub fn as_quaternion(&self) -> Option<[T; 4]> {
        match self {
            Value::Quaternion(q) => Some(*q),
            _ => None,
        }
    }
}

// ============================================================================
// LinalgValue implementation
// ============================================================================

#[cfg(feature = "linalg")]
impl<T: Float + std::fmt::Debug> rhizome_dew_linalg::LinalgValue<T> for Value<T> {
    fn typ(&self) -> rhizome_dew_linalg::Type {
        match self {
            Value::Scalar(_) => rhizome_dew_linalg::Type::Scalar,
            Value::Vec2(_) => rhizome_dew_linalg::Type::Vec2,
            Value::Vec3(_) => rhizome_dew_linalg::Type::Vec3,
            Value::Vec4(_) => rhizome_dew_linalg::Type::Vec4,
            Value::Mat2(_) => rhizome_dew_linalg::Type::Mat2,
            Value::Mat3(_) => rhizome_dew_linalg::Type::Mat3,
            Value::Mat4(_) => rhizome_dew_linalg::Type::Mat4,
            _ => rhizome_dew_linalg::Type::Scalar, // fallback for non-linalg types
        }
    }

    fn from_scalar(v: T) -> Self {
        Value::Scalar(v)
    }
    fn from_vec2(v: [T; 2]) -> Self {
        Value::Vec2(v)
    }
    fn from_vec3(v: [T; 3]) -> Self {
        Value::Vec3(v)
    }
    fn from_vec4(v: [T; 4]) -> Self {
        Value::Vec4(v)
    }
    fn from_mat2(m: [T; 4]) -> Self {
        Value::Mat2(m)
    }
    fn from_mat3(m: [T; 9]) -> Self {
        Value::Mat3(m)
    }
    fn from_mat4(m: [T; 16]) -> Self {
        Value::Mat4(m)
    }

    fn as_scalar(&self) -> Option<T> {
        Value::as_scalar(self)
    }
    fn as_vec2(&self) -> Option<[T; 2]> {
        Value::as_vec2(self)
    }
    fn as_vec3(&self) -> Option<[T; 3]> {
        Value::as_vec3(self)
    }
    fn as_vec4(&self) -> Option<[T; 4]> {
        Value::as_vec4(self)
    }
    fn as_mat2(&self) -> Option<[T; 4]> {
        Value::as_mat2(self)
    }
    fn as_mat3(&self) -> Option<[T; 9]> {
        Value::as_mat3(self)
    }
    fn as_mat4(&self) -> Option<[T; 16]> {
        Value::as_mat4(self)
    }
}

// ============================================================================
// ComplexValue implementation
// ============================================================================

#[cfg(feature = "complex")]
impl<T: Float + std::fmt::Debug> rhizome_dew_complex::ComplexValue<T> for Value<T> {
    fn typ(&self) -> rhizome_dew_complex::Type {
        match self {
            Value::Scalar(_) => rhizome_dew_complex::Type::Scalar,
            Value::Complex(_) => rhizome_dew_complex::Type::Complex,
            _ => rhizome_dew_complex::Type::Scalar, // fallback for non-complex types
        }
    }

    fn from_scalar(v: T) -> Self {
        Value::Scalar(v)
    }
    fn from_complex(c: [T; 2]) -> Self {
        Value::Complex(c)
    }

    fn as_scalar(&self) -> Option<T> {
        Value::as_scalar(self)
    }
    fn as_complex(&self) -> Option<[T; 2]> {
        Value::as_complex(self)
    }
}

// ============================================================================
// QuaternionValue implementation
// ============================================================================

#[cfg(feature = "quaternion")]
impl<T: Float + std::fmt::Debug> rhizome_dew_quaternion::QuaternionValue<T> for Value<T> {
    fn typ(&self) -> rhizome_dew_quaternion::Type {
        match self {
            Value::Scalar(_) => rhizome_dew_quaternion::Type::Scalar,
            Value::Vec3(_) => rhizome_dew_quaternion::Type::Vec3,
            Value::Quaternion(_) => rhizome_dew_quaternion::Type::Quaternion,
            _ => rhizome_dew_quaternion::Type::Scalar, // fallback for non-quaternion types
        }
    }

    fn from_scalar(v: T) -> Self {
        Value::Scalar(v)
    }
    fn from_vec3(v: [T; 3]) -> Self {
        Value::Vec3(v)
    }
    fn from_quaternion(q: [T; 4]) -> Self {
        Value::Quaternion(q)
    }

    fn as_scalar(&self) -> Option<T> {
        Value::as_scalar(self)
    }
    fn as_vec3(&self) -> Option<[T; 3]> {
        Value::as_vec3(self)
    }
    fn as_quaternion(&self) -> Option<[T; 4]> {
        Value::as_quaternion(self)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_types() {
        let scalar: Value<f32> = Value::Scalar(1.0);
        assert_eq!(scalar.typ(), Type::Scalar);
        assert_eq!(scalar.as_scalar(), Some(1.0));

        let vec3: Value<f32> = Value::Vec3([1.0, 2.0, 3.0]);
        assert_eq!(vec3.typ(), Type::Vec3);
        assert_eq!(vec3.as_vec3(), Some([1.0, 2.0, 3.0]));

        let complex: Value<f32> = Value::Complex([1.0, 2.0]);
        assert_eq!(complex.typ(), Type::Complex);
        assert_eq!(complex.as_complex(), Some([1.0, 2.0]));

        let quat: Value<f32> = Value::Quaternion([0.0, 0.0, 0.0, 1.0]);
        assert_eq!(quat.typ(), Type::Quaternion);
        assert_eq!(quat.as_quaternion(), Some([0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    #[cfg(feature = "linalg")]
    fn test_linalg_eval_with_combined_value() {
        use rhizome_dew_core::Expr;
        use std::collections::HashMap;

        let expr = Expr::parse("dot(a, b)").unwrap();
        let vars: HashMap<String, Value<f32>> = [
            ("a".into(), Value::Vec3([1.0, 0.0, 0.0])),
            ("b".into(), Value::Vec3([1.0, 0.0, 0.0])),
        ]
        .into();

        let mut registry = rhizome_dew_linalg::FunctionRegistry::new();
        rhizome_dew_linalg::register_linalg(&mut registry);

        let result = rhizome_dew_linalg::eval(expr.ast(), &vars, &registry).unwrap();
        assert_eq!(result, Value::Scalar(1.0));
    }

    #[test]
    #[cfg(feature = "complex")]
    fn test_complex_eval_with_combined_value() {
        use rhizome_dew_core::Expr;
        use std::collections::HashMap;

        let expr = Expr::parse("abs(z)").unwrap();
        let vars: HashMap<String, Value<f32>> = [("z".into(), Value::Complex([3.0, 4.0]))].into();

        let mut registry = rhizome_dew_complex::FunctionRegistry::new();
        rhizome_dew_complex::register_complex(&mut registry);

        let result = rhizome_dew_complex::eval(expr.ast(), &vars, &registry).unwrap();
        assert_eq!(result, Value::Scalar(5.0));
    }

    #[test]
    #[cfg(feature = "quaternion")]
    fn test_quaternion_eval_with_combined_value() {
        use rhizome_dew_core::Expr;
        use std::collections::HashMap;

        let expr = Expr::parse("rotate(v, q)").unwrap();
        let vars: HashMap<String, Value<f32>> = [
            ("v".into(), Value::Vec3([1.0, 0.0, 0.0])),
            ("q".into(), Value::Quaternion([0.0, 0.0, 0.0, 1.0])), // identity
        ]
        .into();

        let mut registry = rhizome_dew_quaternion::FunctionRegistry::new();
        rhizome_dew_quaternion::register_quaternion(&mut registry);

        let result = rhizome_dew_quaternion::eval(expr.ast(), &vars, &registry).unwrap();
        assert_eq!(result, Value::Vec3([1.0, 0.0, 0.0]));
    }
}
