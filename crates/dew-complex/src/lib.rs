//! Complex number support for dew expressions.
//!
//! Provides complex number types and operations for signal processing,
//! 2D rotations, and general complex arithmetic.
//!
//! # Quick Start
//!
//! ```
//! use rhizome_dew_core::Expr;
//! use rhizome_dew_complex::{Value, eval, complex_registry};
//! use std::collections::HashMap;
//!
//! let expr = Expr::parse("a * b").unwrap();
//!
//! let vars: HashMap<String, Value<f32>> = [
//!     ("a".into(), Value::Complex([1.0, 2.0])),  // 1 + 2i
//!     ("b".into(), Value::Complex([3.0, 4.0])),  // 3 + 4i
//! ].into();
//!
//! let result = eval(expr.ast(), &vars, &complex_registry()).unwrap();
//! // (1+2i)(3+4i) = -5 + 10i
//! assert_eq!(result, Value::Complex([-5.0, 10.0]));
//! ```
//!
//! # Features
//!
//! | Feature     | Description                    |
//! |-------------|--------------------------------|
//! | `wgsl`      | WGSL shader code generation    |
//! | `lua`       | Lua code generation            |
//! | `cranelift` | Cranelift JIT compilation      |
//!
//! # Types
//!
//! | Type      | Description                     |
//! |-----------|---------------------------------|
//! | `Scalar`  | Real number                     |
//! | `Complex` | Complex number [re, im]         |
//!
//! # Functions
//!
//! | Function          | Description                              |
//! |-------------------|------------------------------------------|
//! | `complex(re, im)` | From cartesian form → complex            |
//! | `polar(r, theta)` | From polar form → complex                |
//! | `re(z)`           | Real part → scalar                       |
//! | `im(z)`           | Imaginary part → scalar                  |
//! | `conj(z)`         | Conjugate (a - bi) → complex             |
//! | `abs(z)`          | Magnitude \|z\| → scalar                 |
//! | `arg(z)`          | Argument (angle) → scalar                |
//! | `norm(z)`         | Squared magnitude → scalar               |
//! | `exp(z)`          | Complex exponential → complex            |
//! | `log(z)`          | Complex logarithm → complex              |
//! | `sqrt(z)`         | Complex square root → complex            |
//! | `pow(z, n)`       | Complex power → complex                  |
//!
//! # Operators
//!
//! | Operation          | Result                          |
//! |--------------------|---------------------------------|
//! | `z1 + z2`          | Complex addition                |
//! | `z1 - z2`          | Complex subtraction             |
//! | `z1 * z2`          | Complex multiplication          |
//! | `z1 / z2`          | Complex division                |
//! | `z * scalar`       | Scalar multiplication           |
//! | `-z`               | Negation                        |

use num_traits::Float;
use rhizome_dew_core::{Ast, BinOp, CompareOp, UnaryOp};
use std::collections::HashMap;
use std::sync::Arc;

mod funcs;
pub mod ops;
#[cfg(test)]
mod parity_tests;

#[cfg(feature = "wgsl")]
pub mod wgsl;

#[cfg(feature = "glsl")]
pub mod glsl;

#[cfg(feature = "lua")]
pub mod lua;

#[cfg(feature = "cranelift")]
pub mod cranelift;

#[cfg(feature = "optimize")]
pub mod optimize;

pub use funcs::{
    Abs, Arg, Complex, Conj, Exp, Im, Log, Norm, Polar, Pow, Re, Sqrt, complex_registry,
    register_complex,
};

// ============================================================================
// Types
// ============================================================================

/// Type of a complex value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    /// Real scalar.
    Scalar,
    /// Complex number [re, im].
    Complex,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Scalar => write!(f, "scalar"),
            Type::Complex => write!(f, "complex"),
        }
    }
}

// ============================================================================
// ComplexValue trait (for composability)
// ============================================================================

/// Trait for values that support complex operations.
///
/// Implement this for combined value types when composing multiple domain crates.
pub trait ComplexValue<T: Float>: Clone + PartialEq + Sized + std::fmt::Debug {
    /// Returns the type of this value.
    fn typ(&self) -> Type;

    // Construction
    fn from_scalar(v: T) -> Self;
    fn from_complex(c: [T; 2]) -> Self;

    // Extraction
    fn as_scalar(&self) -> Option<T>;
    fn as_complex(&self) -> Option<[T; 2]>;
}

// ============================================================================
// Values
// ============================================================================

/// A complex value, generic over numeric type.
#[derive(Debug, Clone, PartialEq)]
pub enum Value<T> {
    /// Real scalar.
    Scalar(T),
    /// Complex number [re, im].
    Complex([T; 2]),
}

impl<T> Value<T> {
    /// Returns the type of this value.
    pub fn typ(&self) -> Type {
        match self {
            Value::Scalar(_) => Type::Scalar,
            Value::Complex(_) => Type::Complex,
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

    /// Try to get as complex.
    pub fn as_complex(&self) -> Option<[T; 2]> {
        match self {
            Value::Complex(c) => Some(*c),
            _ => None,
        }
    }
}

impl<T: Float + std::fmt::Debug> ComplexValue<T> for Value<T> {
    fn typ(&self) -> Type {
        Value::typ(self)
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
// Errors
// ============================================================================

/// Complex evaluation error.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Unknown variable.
    UnknownVariable(String),
    /// Unknown function.
    UnknownFunction(String),
    /// Type mismatch for binary operation.
    BinaryTypeMismatch { op: BinOp, left: Type, right: Type },
    /// Type mismatch for unary operation.
    UnaryTypeMismatch { op: UnaryOp, operand: Type },
    /// General type mismatch (for conditionals).
    TypeMismatch { expected: Type, got: Type },
    /// Wrong number of arguments to function.
    WrongArgCount {
        func: String,
        expected: usize,
        got: usize,
    },
    /// Type mismatch in function arguments.
    FunctionTypeMismatch {
        func: String,
        expected: Vec<Type>,
        got: Vec<Type>,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            Error::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            Error::BinaryTypeMismatch { op, left, right } => {
                write!(f, "cannot apply {op:?} to {left} and {right}")
            }
            Error::UnaryTypeMismatch { op, operand } => {
                write!(f, "cannot apply {op:?} to {operand}")
            }
            Error::TypeMismatch { expected, got } => {
                write!(f, "expected {expected}, got {got}")
            }
            Error::WrongArgCount {
                func,
                expected,
                got,
            } => {
                write!(f, "function '{func}' expects {expected} args, got {got}")
            }
            Error::FunctionTypeMismatch {
                func,
                expected,
                got,
            } => {
                write!(
                    f,
                    "function '{func}' expects types {expected:?}, got {got:?}"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

// ============================================================================
// Function Registry
// ============================================================================

/// A function signature.
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    pub args: Vec<Type>,
    pub ret: Type,
}

/// A function that can be called from complex expressions.
///
/// Generic over both the numeric type `T` and the value type `V`.
pub trait ComplexFn<T, V>: Send + Sync
where
    T: Float,
    V: ComplexValue<T>,
{
    /// Function name.
    fn name(&self) -> &str;

    /// Available signatures for this function.
    fn signatures(&self) -> Vec<Signature>;

    /// Call the function with typed arguments.
    fn call(&self, args: &[V]) -> V;
}

/// Registry of complex functions.
#[derive(Clone)]
pub struct FunctionRegistry<T, V>
where
    T: Float,
    V: ComplexValue<T>,
{
    funcs: HashMap<String, Arc<dyn ComplexFn<T, V>>>,
}

impl<T, V> Default for FunctionRegistry<T, V>
where
    T: Float,
    V: ComplexValue<T>,
{
    fn default() -> Self {
        Self {
            funcs: HashMap::new(),
        }
    }
}

impl<T, V> FunctionRegistry<T, V>
where
    T: Float,
    V: ComplexValue<T>,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<F: ComplexFn<T, V> + 'static>(&mut self, func: F) {
        self.funcs.insert(func.name().to_string(), Arc::new(func));
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn ComplexFn<T, V>>> {
        self.funcs.get(name)
    }
}

// ============================================================================
// Evaluation
// ============================================================================

/// Evaluate an AST with complex values.
///
/// Generic over both the numeric type `T` and the value type `V`.
pub fn eval<T, V>(
    ast: &Ast,
    vars: &HashMap<String, V>,
    funcs: &FunctionRegistry<T, V>,
) -> Result<V, Error>
where
    T: Float,
    V: ComplexValue<T>,
{
    match ast {
        Ast::Num(n) => Ok(V::from_scalar(T::from(*n).unwrap())),

        Ast::Var(name) => vars
            .get(name)
            .cloned()
            .ok_or_else(|| Error::UnknownVariable(name.clone())),

        Ast::BinOp(op, left, right) => {
            let left_val = eval(left, vars, funcs)?;
            let right_val = eval(right, vars, funcs)?;
            ops::apply_binop(*op, left_val, right_val)
        }

        Ast::UnaryOp(op, inner) => {
            let val = eval(inner, vars, funcs)?;
            ops::apply_unaryop(*op, val)
        }

        Ast::Compare(op, left, right) => {
            let left_val = eval(left, vars, funcs)?;
            let right_val = eval(right, vars, funcs)?;
            // Comparisons only supported for scalars
            match (left_val.as_scalar(), right_val.as_scalar()) {
                (Some(l), Some(r)) => {
                    let result = match op {
                        CompareOp::Lt => l < r,
                        CompareOp::Le => l <= r,
                        CompareOp::Gt => l > r,
                        CompareOp::Ge => l >= r,
                        CompareOp::Eq => l == r,
                        CompareOp::Ne => l != r,
                    };
                    Ok(V::from_scalar(if result { T::one() } else { T::zero() }))
                }
                _ => Err(Error::TypeMismatch {
                    expected: Type::Scalar,
                    got: Type::Complex,
                }),
            }
        }

        Ast::And(left, right) => {
            let left_val = eval(left, vars, funcs)?;
            if let Some(l) = left_val.as_scalar() {
                if l == T::zero() {
                    return Ok(V::from_scalar(T::zero()));
                }
                let right_val = eval(right, vars, funcs)?;
                if let Some(r) = right_val.as_scalar() {
                    Ok(V::from_scalar(if r != T::zero() {
                        T::one()
                    } else {
                        T::zero()
                    }))
                } else {
                    Err(Error::TypeMismatch {
                        expected: Type::Scalar,
                        got: Type::Complex,
                    })
                }
            } else {
                Err(Error::TypeMismatch {
                    expected: Type::Scalar,
                    got: Type::Complex,
                })
            }
        }

        Ast::Or(left, right) => {
            let left_val = eval(left, vars, funcs)?;
            if let Some(l) = left_val.as_scalar() {
                if l != T::zero() {
                    return Ok(V::from_scalar(T::one()));
                }
                let right_val = eval(right, vars, funcs)?;
                if let Some(r) = right_val.as_scalar() {
                    Ok(V::from_scalar(if r != T::zero() {
                        T::one()
                    } else {
                        T::zero()
                    }))
                } else {
                    Err(Error::TypeMismatch {
                        expected: Type::Scalar,
                        got: Type::Complex,
                    })
                }
            } else {
                Err(Error::TypeMismatch {
                    expected: Type::Scalar,
                    got: Type::Complex,
                })
            }
        }

        Ast::If(cond, then_expr, else_expr) => {
            let cond_val = eval(cond, vars, funcs)?;
            if let Some(c) = cond_val.as_scalar() {
                if c != T::zero() {
                    eval(then_expr, vars, funcs)
                } else {
                    eval(else_expr, vars, funcs)
                }
            } else {
                Err(Error::TypeMismatch {
                    expected: Type::Scalar,
                    got: Type::Complex,
                })
            }
        }

        Ast::Call(name, args) => {
            let func = funcs
                .get(name)
                .ok_or_else(|| Error::UnknownFunction(name.clone()))?;

            let arg_vals: Vec<V> = args
                .iter()
                .map(|a| eval(a, vars, funcs))
                .collect::<Result<_, _>>()?;

            let arg_types: Vec<Type> = arg_vals.iter().map(|v| v.typ()).collect();

            // Find matching signature
            let matched = func.signatures().iter().any(|sig| sig.args == arg_types);
            if !matched {
                return Err(Error::FunctionTypeMismatch {
                    func: name.clone(),
                    expected: func
                        .signatures()
                        .first()
                        .map(|s| s.args.clone())
                        .unwrap_or_default(),
                    got: arg_types,
                });
            }

            Ok(func.call(&arg_vals))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn eval_expr(expr: &str, vars: &[(&str, Value<f32>)]) -> Result<Value<f32>, Error> {
        let expr = Expr::parse(expr).unwrap();
        let var_map: HashMap<String, Value<f32>> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let registry = complex_registry();
        eval(expr.ast(), &var_map, &registry)
    }

    #[test]
    fn test_complex_add() {
        let result = eval_expr(
            "a + b",
            &[
                ("a", Value::Complex([1.0, 2.0])),
                ("b", Value::Complex([3.0, 4.0])),
            ],
        );
        assert_eq!(result.unwrap(), Value::Complex([4.0, 6.0]));
    }

    #[test]
    fn test_complex_mul() {
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        let result = eval_expr(
            "a * b",
            &[
                ("a", Value::Complex([1.0, 2.0])),
                ("b", Value::Complex([3.0, 4.0])),
            ],
        );
        assert_eq!(result.unwrap(), Value::Complex([-5.0, 10.0]));
    }

    #[test]
    fn test_scalar_complex_mul() {
        // 2 * (3+4i) = 6 + 8i
        let result = eval_expr(
            "s * z",
            &[("s", Value::Scalar(2.0)), ("z", Value::Complex([3.0, 4.0]))],
        );
        assert_eq!(result.unwrap(), Value::Complex([6.0, 8.0]));
    }

    #[test]
    fn test_complex_neg() {
        let result = eval_expr("-z", &[("z", Value::Complex([3.0, 4.0]))]);
        assert_eq!(result.unwrap(), Value::Complex([-3.0, -4.0]));
    }

    #[test]
    fn test_literal_becomes_scalar() {
        let result = eval_expr("2.0 + z", &[("z", Value::Complex([1.0, 1.0]))]);
        // 2 + (1+i) = 3 + i
        assert_eq!(result.unwrap(), Value::Complex([3.0, 1.0]));
    }
}
