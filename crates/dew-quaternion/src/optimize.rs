//! Quaternion optimization passes.
//!
//! This module provides optimization passes for quaternion operations.
//! When all arguments are constants, operations are evaluated at compile time.
//!
//! # Example
//!
//! ```
//! use rhizome_dew_core::Expr;
//! use rhizome_dew_core::optimize::{optimize, standard_passes};
//! use rhizome_dew_quaternion::optimize::QuaternionConstantFolding;
//!
//! let expr = Expr::parse("length(vec3(3, 4, 0))").unwrap();
//!
//! // Combine core passes with quaternion constant folding
//! let mut passes: Vec<&dyn rhizome_dew_core::optimize::Pass> = standard_passes();
//! passes.push(&QuaternionConstantFolding);
//!
//! let optimized = optimize(expr.ast().clone(), &passes);
//! // length(vec3(3, 4, 0)) = 5
//! assert_eq!(optimized.to_string(), "5");
//! ```
//!
//! # Limitations
//!
//! Quaternion constant folding uses **constructor-based type inference**:
//! only operations where types are visible in the AST can be folded.
//!
//! ## What works
//!
//! ```text
//! length(vec3(3, 4, 0))           // → 5
//! dot(vec3(1, 0, 0), vec3(0, 1, 0)) // → 0
//! normalize(quat(0, 0, 0, 2))     // → quat(0, 0, 0, 1)
//! conj(quat(1, 2, 3, 4))          // → quat(-1, -2, -3, 4)
//! ```
//!
//! ## What doesn't work (yet)
//!
//! ```text
//! length(v)       // v is a variable — type unknown at AST level
//! dot(a, b)       // variables — no type info in AST
//! ```
//!
//! For full type-aware optimization, a future `TypedPass` trait could receive
//! type information from the caller's context.

use rhizome_dew_core::Ast;
use rhizome_dew_core::optimize::Pass;

/// Constant folding for quaternion operations.
///
/// Evaluates quaternion function calls when all arguments are constants.
/// Uses constructor-based type inference via `vec3(x, y, z)` and `quat(x, y, z, w)`.
///
/// # Supported Functions
///
/// ## Scalar results
/// - `length(v)` → magnitude
/// - `dot(a, b)` → dot product
///
/// ## Vec3 results
/// - `normalize(v)` → unit vector
/// - `lerp(a, b, t)` → linear interpolation
/// - `rotate(v, q)` → rotated vector
///
/// ## Quaternion results
/// - `conj(q)` → conjugate
/// - `normalize(q)` → unit quaternion
/// - `inverse(q)` → multiplicative inverse
/// - `lerp(q1, q2, t)` → linear interpolation
/// - `slerp(q1, q2, t)` → spherical interpolation
/// - `axis_angle(axis, angle)` → quaternion from axis-angle
///
/// ## Construction
/// - `vec3(x, y, z)` → Vec3
/// - `quat(x, y, z, w)` → Quaternion
pub struct QuaternionConstantFolding;

impl Pass for QuaternionConstantFolding {
    fn transform(&self, ast: &Ast) -> Option<Ast> {
        let Ast::Call(name, args) = ast else {
            return None;
        };

        // Try to evaluate as constant quaternion expression
        let const_args: Option<Vec<ConstValue>> = args.iter().map(ConstValue::from_ast).collect();
        let const_args = const_args?;

        let result = evaluate_quaternion_function(name, &const_args)?;
        Some(result.to_ast())
    }
}

/// A constant value that can be a scalar, vec3, or quaternion.
#[derive(Debug, Clone, Copy)]
enum ConstValue {
    Scalar(f64),
    Vec3([f64; 3]),
    Quaternion([f64; 4]),
}

impl ConstValue {
    /// Try to extract a constant value from an AST node.
    fn from_ast(ast: &Ast) -> Option<Self> {
        match ast {
            Ast::Num(n) => Some(ConstValue::Scalar(*n)),
            Ast::Call(name, args) if name == "vec3" && args.len() == 3 => {
                let x = Self::extract_num(&args[0])?;
                let y = Self::extract_num(&args[1])?;
                let z = Self::extract_num(&args[2])?;
                Some(ConstValue::Vec3([x, y, z]))
            }
            Ast::Call(name, args) if name == "quat" && args.len() == 4 => {
                let x = Self::extract_num(&args[0])?;
                let y = Self::extract_num(&args[1])?;
                let z = Self::extract_num(&args[2])?;
                let w = Self::extract_num(&args[3])?;
                Some(ConstValue::Quaternion([x, y, z, w]))
            }
            _ => None,
        }
    }

    fn extract_num(ast: &Ast) -> Option<f64> {
        match ast {
            Ast::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// Convert back to AST.
    fn to_ast(self) -> Ast {
        match self {
            ConstValue::Scalar(s) => Ast::Num(s),
            ConstValue::Vec3([x, y, z]) => {
                Ast::Call("vec3".into(), vec![Ast::Num(x), Ast::Num(y), Ast::Num(z)])
            }
            ConstValue::Quaternion([x, y, z, w]) => Ast::Call(
                "quat".into(),
                vec![Ast::Num(x), Ast::Num(y), Ast::Num(z), Ast::Num(w)],
            ),
        }
    }

    fn as_scalar(self) -> Option<f64> {
        match self {
            ConstValue::Scalar(s) => Some(s),
            _ => None,
        }
    }

    fn as_vec3(self) -> Option<[f64; 3]> {
        match self {
            ConstValue::Vec3(v) => Some(v),
            _ => None,
        }
    }

    fn as_quaternion(self) -> Option<[f64; 4]> {
        match self {
            ConstValue::Quaternion(q) => Some(q),
            _ => None,
        }
    }
}

/// Evaluate a quaternion function with constant arguments.
fn evaluate_quaternion_function(name: &str, args: &[ConstValue]) -> Option<ConstValue> {
    match (name, args.len()) {
        // Constructors
        ("vec3", 3) => {
            let x = args[0].as_scalar()?;
            let y = args[1].as_scalar()?;
            let z = args[2].as_scalar()?;
            Some(ConstValue::Vec3([x, y, z]))
        }
        ("quat", 4) => {
            let x = args[0].as_scalar()?;
            let y = args[1].as_scalar()?;
            let z = args[2].as_scalar()?;
            let w = args[3].as_scalar()?;
            Some(ConstValue::Quaternion([x, y, z, w]))
        }

        // Length
        ("length", 1) => match args[0] {
            ConstValue::Vec3(v) => Some(ConstValue::Scalar(
                (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(),
            )),
            ConstValue::Quaternion(q) => Some(ConstValue::Scalar(
                (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt(),
            )),
            _ => None,
        },

        // Normalize
        ("normalize", 1) => match args[0] {
            ConstValue::Vec3(v) => {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                Some(ConstValue::Vec3([v[0] / len, v[1] / len, v[2] / len]))
            }
            ConstValue::Quaternion(q) => {
                let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                Some(ConstValue::Quaternion([
                    q[0] / len,
                    q[1] / len,
                    q[2] / len,
                    q[3] / len,
                ]))
            }
            _ => None,
        },

        // Conjugate
        ("conj", 1) => {
            let q = args[0].as_quaternion()?;
            Some(ConstValue::Quaternion([-q[0], -q[1], -q[2], q[3]]))
        }

        // Inverse
        ("inverse", 1) => {
            let q = args[0].as_quaternion()?;
            let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
            Some(ConstValue::Quaternion([
                -q[0] / norm_sq,
                -q[1] / norm_sq,
                -q[2] / norm_sq,
                q[3] / norm_sq,
            ]))
        }

        // Dot product
        ("dot", 2) => match (&args[0], &args[1]) {
            (ConstValue::Vec3(a), ConstValue::Vec3(b)) => {
                Some(ConstValue::Scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
            }
            (ConstValue::Quaternion(a), ConstValue::Quaternion(b)) => Some(ConstValue::Scalar(
                a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3],
            )),
            _ => None,
        },

        // Lerp
        ("lerp", 3) => {
            let t = args[2].as_scalar()?;
            match (&args[0], &args[1]) {
                (ConstValue::Vec3(a), ConstValue::Vec3(b)) => Some(ConstValue::Vec3([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                ])),
                (ConstValue::Quaternion(a), ConstValue::Quaternion(b)) => {
                    Some(ConstValue::Quaternion([
                        a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t,
                        a[2] + (b[2] - a[2]) * t,
                        a[3] + (b[3] - a[3]) * t,
                    ]))
                }
                _ => None,
            }
        }

        // Slerp (spherical linear interpolation)
        ("slerp", 3) => {
            let a = args[0].as_quaternion()?;
            let b = args[1].as_quaternion()?;
            let t = args[2].as_scalar()?;
            Some(ConstValue::Quaternion(slerp_impl(&a, &b, t)))
        }

        // Axis-angle construction
        ("axis_angle", 2) => {
            let axis = args[0].as_vec3()?;
            let angle = args[1].as_scalar()?;
            let half_angle = angle / 2.0;
            let s = half_angle.sin();
            let c = half_angle.cos();
            let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            Some(ConstValue::Quaternion([
                axis[0] / len * s,
                axis[1] / len * s,
                axis[2] / len * s,
                c,
            ]))
        }

        // Rotate vector by quaternion
        ("rotate", 2) => {
            let v = args[0].as_vec3()?;
            let q = args[1].as_quaternion()?;
            Some(ConstValue::Vec3(rotate_vec3_by_quat(&v, &q)))
        }

        _ => None,
    }
}

/// Spherical linear interpolation for quaternions.
fn slerp_impl(a: &[f64; 4], b: &[f64; 4], t: f64) -> [f64; 4] {
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

    // If dot < 0, negate one quaternion to take shorter path
    let mut b = *b;
    if dot < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
        dot = -dot;
    }

    // Clamp dot to valid range for acos
    if dot > 1.0 {
        dot = 1.0;
    }

    // If quaternions are very close, use linear interpolation
    if dot > 0.9995 {
        let result = [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ];
        // Normalize
        let len = (result[0] * result[0]
            + result[1] * result[1]
            + result[2] * result[2]
            + result[3] * result[3])
            .sqrt();
        return [
            result[0] / len,
            result[1] / len,
            result[2] / len,
            result[3] / len,
        ];
    }

    // Spherical interpolation
    let theta = dot.acos();
    let sin_theta = theta.sin();
    let s0 = ((1.0 - t) * theta).sin() / sin_theta;
    let s1 = (t * theta).sin() / sin_theta;

    [
        a[0] * s0 + b[0] * s1,
        a[1] * s0 + b[1] * s1,
        a[2] * s0 + b[2] * s1,
        a[3] * s0 + b[3] * s1,
    ]
}

/// Rotate a vec3 by a quaternion using the optimized formula.
fn rotate_vec3_by_quat(v: &[f64; 3], q: &[f64; 4]) -> [f64; 3] {
    let (qx, qy, qz, qw) = (q[0], q[1], q[2], q[3]);

    // t = 2 * (q_xyz × v)
    let tx = 2.0 * (qy * v[2] - qz * v[1]);
    let ty = 2.0 * (qz * v[0] - qx * v[2]);
    let tz = 2.0 * (qx * v[1] - qy * v[0]);

    // v' = v + w * t + (q_xyz × t)
    [
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    ]
}

/// Returns all quaternion optimization passes.
pub fn quaternion_passes() -> Vec<&'static dyn Pass> {
    vec![&QuaternionConstantFolding]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;
    use rhizome_dew_core::optimize::{optimize, standard_passes};

    fn optimized(input: &str) -> Ast {
        let expr = Expr::parse(input).unwrap();
        let mut passes: Vec<&dyn Pass> = standard_passes();
        passes.push(&QuaternionConstantFolding);
        optimize(expr.ast().clone(), &passes)
    }

    fn optimized_scalar(input: &str) -> f64 {
        match optimized(input) {
            Ast::Num(n) => n,
            other => panic!("expected Num, got {other:?}"),
        }
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.001
    }

    // Constructors
    #[test]
    fn test_vec3_construction() {
        let result = optimized("vec3(1, 2, 3)");
        if let Ast::Call(name, args) = result {
            assert_eq!(name, "vec3");
            assert_eq!(args.len(), 3);
        } else {
            panic!("expected Call");
        }
    }

    #[test]
    fn test_quat_construction() {
        let result = optimized("quat(0, 0, 0, 1)");
        if let Ast::Call(name, args) = result {
            assert_eq!(name, "quat");
            assert_eq!(args.len(), 4);
        } else {
            panic!("expected Call");
        }
    }

    // Length
    #[test]
    fn test_vec3_length() {
        let v = optimized_scalar("length(vec3(3, 4, 0))");
        assert!(approx_eq(v, 5.0));
    }

    #[test]
    fn test_quat_length() {
        let v = optimized_scalar("length(quat(0, 0, 3, 4))");
        assert!(approx_eq(v, 5.0));
    }

    // Dot product
    #[test]
    fn test_vec3_dot() {
        let v = optimized_scalar("dot(vec3(1, 0, 0), vec3(0, 1, 0))");
        assert!(approx_eq(v, 0.0));
    }

    #[test]
    fn test_vec3_dot_parallel() {
        let v = optimized_scalar("dot(vec3(1, 2, 3), vec3(1, 2, 3))");
        assert!(approx_eq(v, 14.0)); // 1+4+9
    }

    // Normalize
    #[test]
    fn test_vec3_normalize() {
        let result = optimized("normalize(vec3(3, 0, 0))");
        if let Ast::Call(name, args) = result {
            assert_eq!(name, "vec3");
            if let Ast::Num(x) = &args[0] {
                assert!(approx_eq(*x, 1.0));
            }
        }
    }

    #[test]
    fn test_quat_normalize() {
        let result = optimized("normalize(quat(0, 0, 0, 2))");
        if let Ast::Call(name, args) = result {
            assert_eq!(name, "quat");
            if let Ast::Num(w) = &args[3] {
                assert!(approx_eq(*w, 1.0));
            }
        }
    }

    // Conjugate
    #[test]
    fn test_quat_conj() {
        let result = optimized("conj(quat(1, 2, 3, 4))");
        if let Ast::Call(name, args) = result {
            assert_eq!(name, "quat");
            if let (Ast::Num(x), Ast::Num(y), Ast::Num(z), Ast::Num(w)) =
                (&args[0], &args[1], &args[2], &args[3])
            {
                assert!(approx_eq(*x, -1.0));
                assert!(approx_eq(*y, -2.0));
                assert!(approx_eq(*z, -3.0));
                assert!(approx_eq(*w, 4.0));
            }
        }
    }

    // Lerp
    #[test]
    fn test_vec3_lerp() {
        let result = optimized("lerp(vec3(0, 0, 0), vec3(10, 20, 30), 0.5)");
        if let Ast::Call(name, args) = result {
            assert_eq!(name, "vec3");
            if let (Ast::Num(x), Ast::Num(y), Ast::Num(z)) = (&args[0], &args[1], &args[2]) {
                assert!(approx_eq(*x, 5.0));
                assert!(approx_eq(*y, 10.0));
                assert!(approx_eq(*z, 15.0));
            }
        }
    }

    // Partial constant (should not fold)
    #[test]
    fn test_partial_not_folded() {
        let result = optimized("length(v)");
        assert!(matches!(result, Ast::Call(name, _) if name == "length"));
    }

    // Combined with core passes
    #[test]
    fn test_combined() {
        let v = optimized_scalar("length(vec3(3, 4, 0)) + 1");
        assert!(approx_eq(v, 6.0));
    }
}
