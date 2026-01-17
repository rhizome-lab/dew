//! Linear algebra optimization passes.
//!
//! This module provides optimization passes for vector and matrix operations.
//! When operands are constant, operations are evaluated at compile time.
//!
//! # Example
//!
//! ```
//! use rhizome_dew_core::Expr;
//! use rhizome_dew_core::optimize::{optimize, standard_passes};
//! use rhizome_dew_linalg::optimize::LinalgConstantFolding;
//!
//! let expr = Expr::parse("vec2(1, 2) + vec2(3, 4)").unwrap();
//!
//! // Combine core passes with linalg constant folding
//! let mut passes: Vec<&dyn rhizome_dew_core::optimize::Pass> = standard_passes();
//! passes.push(&LinalgConstantFolding);
//!
//! let optimized = optimize(expr.ast().clone(), &passes);
//! // vec2(1, 2) + vec2(3, 4) → vec2(4, 6)
//! assert_eq!(optimized.to_string(), "vec2(4, 6)");
//! ```
//!
//! # Current Limitations
//!
//! This pass uses **constructor-based type inference**: it can only fold operations
//! where vector types are evident from the AST structure itself (i.e., `vec2(...)`,
//! `vec3(...)` calls with constant arguments).
//!
//! **What works:**
//! ```text
//! vec2(1, 2) + vec2(3, 4)       → vec2(4, 6)      ✓ (constructors visible)
//! length(vec2(3, 4))            → 5               ✓ (constructor visible)
//! dot(vec2(1, 0), vec2(0, 1))   → 0               ✓ (constructors visible)
//! ```
//!
//! **What doesn't work (yet):**
//! ```text
//! a + b                          (a, b are Vec3 variables - no type info in AST)
//! dot(a, a)                      (can't infer a is a vector from AST alone)
//! ```
//!
//! # Future Extension: Type-Aware Optimization
//!
//! To optimize operations on typed variables, the `Pass` trait would need to accept
//! type context:
//!
//! ```ignore
//! // Future API (not yet implemented):
//! pub trait TypedPass {
//!     fn transform(&self, ast: &Ast, var_types: &HashMap<String, Type>) -> Option<Ast>;
//! }
//! ```
//!
//! This would enable:
//! - `dot(a, a)` → `length_squared(a)` when `a: Vec3`
//! - Algebraic identities on typed variables
//! - Dead branch elimination based on type constraints
//!
//! The infrastructure for this exists (backends already receive type maps), but
//! the `Pass` trait hasn't been extended yet to avoid premature abstraction.

use rhizome_dew_core::optimize::Pass;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};

/// Constant folding for linear algebra operations.
///
/// Evaluates vector and matrix operations when all operands are constants.
///
/// # Supported Operations
///
/// ## Vector Constructors
/// - `vec2(x, y)` with constant args
/// - `vec3(x, y, z)` with constant args (3d feature)
/// - `vec4(x, y, z, w)` with constant args (4d feature)
///
/// ## Binary Operations
/// - `vec + vec` → component-wise addition
/// - `vec - vec` → component-wise subtraction
/// - `vec * scalar` / `scalar * vec` → scalar multiplication
/// - `vec / scalar` → scalar division
/// - `-vec` → negation
///
/// ## Functions
/// - `dot(a, b)` → scalar
/// - `length(v)` → scalar
/// - `normalize(v)` → vector (if constant)
/// - `hadamard(a, b)` → component-wise multiply
/// - `cross(a, b)` → vec3 (3d feature)
pub struct LinalgConstantFolding;

impl Pass for LinalgConstantFolding {
    fn transform(&self, ast: &Ast) -> Option<Ast> {
        match ast {
            // Binary operations on vectors
            Ast::BinOp(op, left, right) => fold_binop(*op, left, right),

            // Unary negation
            Ast::UnaryOp(UnaryOp::Neg, inner) => fold_neg(inner),

            // Function calls (dot, length, normalize, etc.)
            Ast::Call(name, args) => fold_call(name, args),

            _ => None,
        }
    }
}

// ============================================================================
// Value representation for constant folding
// ============================================================================

/// A constant value that can be folded.
#[derive(Debug, Clone)]
enum ConstValue {
    Scalar(f32),
    Vec2([f32; 2]),
    #[cfg(feature = "3d")]
    Vec3([f32; 3]),
    #[cfg(feature = "4d")]
    Vec4([f32; 4]),
}

impl ConstValue {
    /// Try to interpret an AST as a constant value.
    fn from_ast(ast: &Ast) -> Option<Self> {
        match ast {
            Ast::Num(n) => Some(ConstValue::Scalar(*n)),

            Ast::Call(name, args) => match (name.as_str(), args.len()) {
                ("vec2", 2) => {
                    let x = Self::as_scalar(&args[0])?;
                    let y = Self::as_scalar(&args[1])?;
                    Some(ConstValue::Vec2([x, y]))
                }
                #[cfg(feature = "3d")]
                ("vec3", 3) => {
                    let x = Self::as_scalar(&args[0])?;
                    let y = Self::as_scalar(&args[1])?;
                    let z = Self::as_scalar(&args[2])?;
                    Some(ConstValue::Vec3([x, y, z]))
                }
                #[cfg(feature = "4d")]
                ("vec4", 4) => {
                    let x = Self::as_scalar(&args[0])?;
                    let y = Self::as_scalar(&args[1])?;
                    let z = Self::as_scalar(&args[2])?;
                    let w = Self::as_scalar(&args[3])?;
                    Some(ConstValue::Vec4([x, y, z, w]))
                }
                _ => None,
            },

            _ => None,
        }
    }

    /// Helper to extract a scalar from an AST.
    fn as_scalar(ast: &Ast) -> Option<f32> {
        match ast {
            Ast::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// Convert back to AST.
    fn to_ast(&self) -> Ast {
        match self {
            ConstValue::Scalar(n) => Ast::Num(*n),
            ConstValue::Vec2([x, y]) => Ast::Call("vec2".into(), vec![Ast::Num(*x), Ast::Num(*y)]),
            #[cfg(feature = "3d")]
            ConstValue::Vec3([x, y, z]) => Ast::Call(
                "vec3".into(),
                vec![Ast::Num(*x), Ast::Num(*y), Ast::Num(*z)],
            ),
            #[cfg(feature = "4d")]
            ConstValue::Vec4([x, y, z, w]) => Ast::Call(
                "vec4".into(),
                vec![Ast::Num(*x), Ast::Num(*y), Ast::Num(*z), Ast::Num(*w)],
            ),
        }
    }
}

// ============================================================================
// Binary operation folding
// ============================================================================

fn fold_binop(op: BinOp, left: &Ast, right: &Ast) -> Option<Ast> {
    let left_val = ConstValue::from_ast(left)?;
    let right_val = ConstValue::from_ast(right)?;

    let result = match (op, &left_val, &right_val) {
        // Vec2 + Vec2
        (BinOp::Add, ConstValue::Vec2(a), ConstValue::Vec2(b)) => {
            ConstValue::Vec2([a[0] + b[0], a[1] + b[1]])
        }
        // Vec2 - Vec2
        (BinOp::Sub, ConstValue::Vec2(a), ConstValue::Vec2(b)) => {
            ConstValue::Vec2([a[0] - b[0], a[1] - b[1]])
        }
        // Vec2 * Scalar
        (BinOp::Mul, ConstValue::Vec2(v), ConstValue::Scalar(s)) => {
            ConstValue::Vec2([v[0] * s, v[1] * s])
        }
        // Scalar * Vec2
        (BinOp::Mul, ConstValue::Scalar(s), ConstValue::Vec2(v)) => {
            ConstValue::Vec2([s * v[0], s * v[1]])
        }
        // Vec2 / Scalar
        (BinOp::Div, ConstValue::Vec2(v), ConstValue::Scalar(s)) => {
            ConstValue::Vec2([v[0] / s, v[1] / s])
        }

        #[cfg(feature = "3d")]
        // Vec3 + Vec3
        (BinOp::Add, ConstValue::Vec3(a), ConstValue::Vec3(b)) => {
            ConstValue::Vec3([a[0] + b[0], a[1] + b[1], a[2] + b[2]])
        }
        #[cfg(feature = "3d")]
        // Vec3 - Vec3
        (BinOp::Sub, ConstValue::Vec3(a), ConstValue::Vec3(b)) => {
            ConstValue::Vec3([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        }
        #[cfg(feature = "3d")]
        // Vec3 * Scalar
        (BinOp::Mul, ConstValue::Vec3(v), ConstValue::Scalar(s)) => {
            ConstValue::Vec3([v[0] * s, v[1] * s, v[2] * s])
        }
        #[cfg(feature = "3d")]
        // Scalar * Vec3
        (BinOp::Mul, ConstValue::Scalar(s), ConstValue::Vec3(v)) => {
            ConstValue::Vec3([s * v[0], s * v[1], s * v[2]])
        }
        #[cfg(feature = "3d")]
        // Vec3 / Scalar
        (BinOp::Div, ConstValue::Vec3(v), ConstValue::Scalar(s)) => {
            ConstValue::Vec3([v[0] / s, v[1] / s, v[2] / s])
        }

        #[cfg(feature = "4d")]
        // Vec4 + Vec4
        (BinOp::Add, ConstValue::Vec4(a), ConstValue::Vec4(b)) => {
            ConstValue::Vec4([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
        }
        #[cfg(feature = "4d")]
        // Vec4 - Vec4
        (BinOp::Sub, ConstValue::Vec4(a), ConstValue::Vec4(b)) => {
            ConstValue::Vec4([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])
        }
        #[cfg(feature = "4d")]
        // Vec4 * Scalar
        (BinOp::Mul, ConstValue::Vec4(v), ConstValue::Scalar(s)) => {
            ConstValue::Vec4([v[0] * s, v[1] * s, v[2] * s, v[3] * s])
        }
        #[cfg(feature = "4d")]
        // Scalar * Vec4
        (BinOp::Mul, ConstValue::Scalar(s), ConstValue::Vec4(v)) => {
            ConstValue::Vec4([s * v[0], s * v[1], s * v[2], s * v[3]])
        }
        #[cfg(feature = "4d")]
        // Vec4 / Scalar
        (BinOp::Div, ConstValue::Vec4(v), ConstValue::Scalar(s)) => {
            ConstValue::Vec4([v[0] / s, v[1] / s, v[2] / s, v[3] / s])
        }

        _ => return None,
    };

    Some(result.to_ast())
}

// ============================================================================
// Unary negation folding
// ============================================================================

fn fold_neg(inner: &Ast) -> Option<Ast> {
    let val = ConstValue::from_ast(inner)?;

    let result = match val {
        ConstValue::Scalar(n) => ConstValue::Scalar(-n),
        ConstValue::Vec2([x, y]) => ConstValue::Vec2([-x, -y]),
        #[cfg(feature = "3d")]
        ConstValue::Vec3([x, y, z]) => ConstValue::Vec3([-x, -y, -z]),
        #[cfg(feature = "4d")]
        ConstValue::Vec4([x, y, z, w]) => ConstValue::Vec4([-x, -y, -z, -w]),
    };

    Some(result.to_ast())
}

// ============================================================================
// Function call folding
// ============================================================================

fn fold_call(name: &str, args: &[Ast]) -> Option<Ast> {
    match (name, args.len()) {
        // dot(a, b) → scalar
        ("dot", 2) => {
            let a = ConstValue::from_ast(&args[0])?;
            let b = ConstValue::from_ast(&args[1])?;
            let result = match (&a, &b) {
                (ConstValue::Vec2(a), ConstValue::Vec2(b)) => a[0] * b[0] + a[1] * b[1],
                #[cfg(feature = "3d")]
                (ConstValue::Vec3(a), ConstValue::Vec3(b)) => {
                    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
                }
                #[cfg(feature = "4d")]
                (ConstValue::Vec4(a), ConstValue::Vec4(b)) => {
                    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
                }
                _ => return None,
            };
            Some(Ast::Num(result))
        }

        // length(v) → scalar
        ("length", 1) => {
            let v = ConstValue::from_ast(&args[0])?;
            let result = match &v {
                ConstValue::Vec2(v) => (v[0] * v[0] + v[1] * v[1]).sqrt(),
                #[cfg(feature = "3d")]
                ConstValue::Vec3(v) => (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(),
                #[cfg(feature = "4d")]
                ConstValue::Vec4(v) => {
                    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt()
                }
                _ => return None,
            };
            Some(Ast::Num(result))
        }

        // normalize(v) → unit vector
        ("normalize", 1) => {
            let v = ConstValue::from_ast(&args[0])?;
            let result = match &v {
                ConstValue::Vec2(v) => {
                    let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
                    ConstValue::Vec2([v[0] / len, v[1] / len])
                }
                #[cfg(feature = "3d")]
                ConstValue::Vec3(v) => {
                    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    ConstValue::Vec3([v[0] / len, v[1] / len, v[2] / len])
                }
                #[cfg(feature = "4d")]
                ConstValue::Vec4(v) => {
                    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
                    ConstValue::Vec4([v[0] / len, v[1] / len, v[2] / len, v[3] / len])
                }
                _ => return None,
            };
            Some(result.to_ast())
        }

        // hadamard(a, b) → component-wise multiply
        ("hadamard", 2) => {
            let a = ConstValue::from_ast(&args[0])?;
            let b = ConstValue::from_ast(&args[1])?;
            let result = match (&a, &b) {
                (ConstValue::Vec2(a), ConstValue::Vec2(b)) => {
                    ConstValue::Vec2([a[0] * b[0], a[1] * b[1]])
                }
                #[cfg(feature = "3d")]
                (ConstValue::Vec3(a), ConstValue::Vec3(b)) => {
                    ConstValue::Vec3([a[0] * b[0], a[1] * b[1], a[2] * b[2]])
                }
                #[cfg(feature = "4d")]
                (ConstValue::Vec4(a), ConstValue::Vec4(b)) => {
                    ConstValue::Vec4([a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]])
                }
                _ => return None,
            };
            Some(result.to_ast())
        }

        // cross(a, b) → vec3 (3d only)
        #[cfg(feature = "3d")]
        ("cross", 2) => {
            let a = ConstValue::from_ast(&args[0])?;
            let b = ConstValue::from_ast(&args[1])?;
            match (&a, &b) {
                (ConstValue::Vec3(a), ConstValue::Vec3(b)) => {
                    let result = ConstValue::Vec3([
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0],
                    ]);
                    Some(result.to_ast())
                }
                _ => None,
            }
        }

        // distance(a, b) → scalar
        ("distance", 2) => {
            let a = ConstValue::from_ast(&args[0])?;
            let b = ConstValue::from_ast(&args[1])?;
            let result = match (&a, &b) {
                (ConstValue::Vec2(a), ConstValue::Vec2(b)) => {
                    let dx = a[0] - b[0];
                    let dy = a[1] - b[1];
                    (dx * dx + dy * dy).sqrt()
                }
                #[cfg(feature = "3d")]
                (ConstValue::Vec3(a), ConstValue::Vec3(b)) => {
                    let dx = a[0] - b[0];
                    let dy = a[1] - b[1];
                    let dz = a[2] - b[2];
                    (dx * dx + dy * dy + dz * dz).sqrt()
                }
                #[cfg(feature = "4d")]
                (ConstValue::Vec4(a), ConstValue::Vec4(b)) => {
                    let dx = a[0] - b[0];
                    let dy = a[1] - b[1];
                    let dz = a[2] - b[2];
                    let dw = a[3] - b[3];
                    (dx * dx + dy * dy + dz * dz + dw * dw).sqrt()
                }
                _ => return None,
            };
            Some(Ast::Num(result))
        }

        // lerp(a, b, t), mix(a, b, t)
        ("lerp", 3) | ("mix", 3) => {
            let a = ConstValue::from_ast(&args[0])?;
            let b = ConstValue::from_ast(&args[1])?;
            let t = match ConstValue::from_ast(&args[2])? {
                ConstValue::Scalar(t) => t,
                _ => return None,
            };
            let result = match (&a, &b) {
                (ConstValue::Vec2(a), ConstValue::Vec2(b)) => {
                    ConstValue::Vec2([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
                }
                #[cfg(feature = "3d")]
                (ConstValue::Vec3(a), ConstValue::Vec3(b)) => ConstValue::Vec3([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                ]),
                #[cfg(feature = "4d")]
                (ConstValue::Vec4(a), ConstValue::Vec4(b)) => ConstValue::Vec4([
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                    a[3] + (b[3] - a[3]) * t,
                ]),
                (ConstValue::Scalar(a), ConstValue::Scalar(b)) => {
                    ConstValue::Scalar(a + (b - a) * t)
                }
                _ => return None,
            };
            Some(result.to_ast())
        }

        // Component accessors: x(v), y(v), z(v), w(v)
        ("x", 1) => {
            let v = ConstValue::from_ast(&args[0])?;
            match &v {
                ConstValue::Vec2(v) => Some(Ast::Num(v[0])),
                #[cfg(feature = "3d")]
                ConstValue::Vec3(v) => Some(Ast::Num(v[0])),
                #[cfg(feature = "4d")]
                ConstValue::Vec4(v) => Some(Ast::Num(v[0])),
                _ => None,
            }
        }
        ("y", 1) => {
            let v = ConstValue::from_ast(&args[0])?;
            match &v {
                ConstValue::Vec2(v) => Some(Ast::Num(v[1])),
                #[cfg(feature = "3d")]
                ConstValue::Vec3(v) => Some(Ast::Num(v[1])),
                #[cfg(feature = "4d")]
                ConstValue::Vec4(v) => Some(Ast::Num(v[1])),
                _ => None,
            }
        }
        #[cfg(feature = "3d")]
        ("z", 1) => {
            let v = ConstValue::from_ast(&args[0])?;
            match &v {
                ConstValue::Vec3(v) => Some(Ast::Num(v[2])),
                #[cfg(feature = "4d")]
                ConstValue::Vec4(v) => Some(Ast::Num(v[2])),
                _ => None,
            }
        }
        #[cfg(feature = "4d")]
        ("w", 1) => {
            let v = ConstValue::from_ast(&args[0])?;
            match &v {
                ConstValue::Vec4(v) => Some(Ast::Num(v[3])),
                _ => None,
            }
        }

        _ => None,
    }
}

/// Returns all linalg optimization passes.
pub fn linalg_passes() -> Vec<&'static dyn Pass> {
    vec![&LinalgConstantFolding]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;
    use rhizome_dew_core::optimize::{optimize, standard_passes};

    fn optimized(input: &str) -> String {
        let expr = Expr::parse(input).unwrap();
        let mut passes: Vec<&dyn Pass> = standard_passes();
        passes.push(&LinalgConstantFolding);
        let result = optimize(expr.ast().clone(), &passes);
        result.to_string()
    }

    // Vec2 operations
    #[test]
    fn test_vec2_add() {
        assert_eq!(optimized("vec2(1, 2) + vec2(3, 4)"), "vec2(4, 6)");
    }

    #[test]
    fn test_vec2_sub() {
        assert_eq!(optimized("vec2(5, 7) - vec2(2, 3)"), "vec2(3, 4)");
    }

    #[test]
    fn test_vec2_mul_scalar() {
        assert_eq!(optimized("vec2(1, 2) * 3"), "vec2(3, 6)");
        assert_eq!(optimized("3 * vec2(1, 2)"), "vec2(3, 6)");
    }

    #[test]
    fn test_vec2_div_scalar() {
        assert_eq!(optimized("vec2(6, 8) / 2"), "vec2(3, 4)");
    }

    #[test]
    fn test_vec2_neg() {
        assert_eq!(optimized("-vec2(1, 2)"), "vec2(-1, -2)");
    }

    // Vec3 operations (3d feature)
    #[cfg(feature = "3d")]
    #[test]
    fn test_vec3_add() {
        assert_eq!(optimized("vec3(1, 2, 3) + vec3(4, 5, 6)"), "vec3(5, 7, 9)");
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_vec3_mul_scalar() {
        assert_eq!(optimized("vec3(1, 2, 3) * 2"), "vec3(2, 4, 6)");
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_vec3_neg() {
        assert_eq!(optimized("-vec3(1, 2, 3)"), "vec3(-1, -2, -3)");
    }

    // Functions
    #[test]
    fn test_dot_vec2() {
        // dot([1,0], [0,1]) = 0 (perpendicular)
        assert_eq!(optimized("dot(vec2(1, 0), vec2(0, 1))"), "0");
        // dot([1,2], [3,4]) = 1*3 + 2*4 = 11
        assert_eq!(optimized("dot(vec2(1, 2), vec2(3, 4))"), "11");
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_dot_vec3() {
        // dot([1,0,0], [0,1,0]) = 0
        assert_eq!(optimized("dot(vec3(1, 0, 0), vec3(0, 1, 0))"), "0");
    }

    #[test]
    fn test_length_vec2() {
        // length([3,4]) = 5
        assert_eq!(optimized("length(vec2(3, 4))"), "5");
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_length_vec3() {
        // length([0,0,5]) = 5
        assert_eq!(optimized("length(vec3(0, 0, 5))"), "5");
    }

    #[test]
    fn test_normalize_vec2() {
        // normalize([3,4]) = [0.6, 0.8]
        let result = optimized("normalize(vec2(3, 4))");
        assert!(result.contains("0.6"));
        assert!(result.contains("0.8"));
    }

    #[test]
    fn test_hadamard() {
        assert_eq!(optimized("hadamard(vec2(2, 3), vec2(4, 5))"), "vec2(8, 15)");
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_cross() {
        // cross([1,0,0], [0,1,0]) = [0,0,1]
        assert_eq!(
            optimized("cross(vec3(1, 0, 0), vec3(0, 1, 0))"),
            "vec3(0, 0, 1)"
        );
    }

    #[test]
    fn test_distance() {
        // distance([0,0], [3,4]) = 5
        assert_eq!(optimized("distance(vec2(0, 0), vec2(3, 4))"), "5");
    }

    #[test]
    fn test_lerp_vec2() {
        // lerp([0,0], [10,10], 0.5) = [5,5]
        assert_eq!(
            optimized("lerp(vec2(0, 0), vec2(10, 10), 0.5)"),
            "vec2(5, 5)"
        );
    }

    #[test]
    fn test_component_accessors() {
        assert_eq!(optimized("x(vec2(3, 4))"), "3");
        assert_eq!(optimized("y(vec2(3, 4))"), "4");
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_component_z() {
        assert_eq!(optimized("z(vec3(1, 2, 3))"), "3");
    }

    // Combined with core passes
    #[test]
    fn test_combined_with_arithmetic() {
        // vec2(1+1, 2+2) should first fold 1+1→2, 2+2→4, then be vec2(2,4)
        assert_eq!(optimized("vec2(1 + 1, 2 + 2)"), "vec2(2, 4)");
    }

    #[test]
    fn test_nested_operations() {
        // length(vec2(1,2) + vec2(2,2)) = length(vec2(3,4)) = 5
        assert_eq!(optimized("length(vec2(1, 2) + vec2(2, 2))"), "5");
    }

    #[test]
    fn test_partial_constant() {
        // vec2(x, 1) + vec2(2, 3) should not fold (x is variable)
        assert_eq!(
            optimized("vec2(x, 1) + vec2(2, 3)"),
            "(vec2(x, 1) + vec2(2, 3))"
        );
    }
}
