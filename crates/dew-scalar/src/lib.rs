//! Standard scalar function library for dew expressions.
//!
//! This crate provides the foundation for numeric expressions: standard math functions
//! (sin, cos, sqrt, etc.), constants (pi, e, tau), and evaluation for scalar values.
//! All functions are generic over `T: Float`, supporting both `f32` and `f64`.
//!
//! # Quick Start
//!
//! ```
//! use rhizome_dew_core::Expr;
//! use rhizome_dew_scalar::{eval, scalar_registry};
//! use std::collections::HashMap;
//!
//! // Parse and evaluate an expression
//! let expr = Expr::parse("sin(x * pi()) + 1").unwrap();
//! let vars: HashMap<String, f32> = [("x".into(), 0.5)].into();
//! let result = eval(expr.ast(), &vars, &scalar_registry()).unwrap();
//! assert!((result - 2.0).abs() < 0.001); // sin(0.5 * π) + 1 = 2
//! ```
//!
//! # Features
//!
//! | Feature       | Description                                |
//! |---------------|--------------------------------------------|
//! | `wgsl`        | WGSL shader code generation                |
//! | `lua-codegen` | Lua code generation (pure Rust, WASM-safe) |
//! | `lua`         | Lua codegen + mlua execution               |
//! | `cranelift`   | Cranelift JIT compilation                  |
//!
//! # Available Functions
//!
//! ## Constants
//!
//! | Function | Description              |
//! |----------|--------------------------|
//! | `pi()`   | π ≈ 3.14159              |
//! | `e()`    | Euler's number ≈ 2.71828 |
//! | `tau()`  | τ = 2π ≈ 6.28318         |
//!
//! ## Trigonometric
//!
//! | Function       | Description                    |
//! |----------------|--------------------------------|
//! | `sin(x)`       | Sine                           |
//! | `cos(x)`       | Cosine                         |
//! | `tan(x)`       | Tangent                        |
//! | `asin(x)`      | Arcsine                        |
//! | `acos(x)`      | Arccosine                      |
//! | `atan(x)`      | Arctangent                     |
//! | `atan2(y, x)`  | Two-argument arctangent        |
//! | `sinh(x)`      | Hyperbolic sine                |
//! | `cosh(x)`      | Hyperbolic cosine              |
//! | `tanh(x)`      | Hyperbolic tangent             |
//!
//! ## Exponential & Logarithmic
//!
//! | Function        | Description                  |
//! |-----------------|------------------------------|
//! | `exp(x)`        | e^x                          |
//! | `exp2(x)`       | 2^x                          |
//! | `log(x)`        | Natural logarithm (alias ln) |
//! | `ln(x)`         | Natural logarithm            |
//! | `log2(x)`       | Base-2 logarithm             |
//! | `log10(x)`      | Base-10 logarithm            |
//! | `pow(x, y)`     | x^y                          |
//! | `sqrt(x)`       | Square root                  |
//! | `inversesqrt(x)`| 1 / sqrt(x)                  |
//!
//! ## Common Math
//!
//! | Function         | Description                    |
//! |------------------|--------------------------------|
//! | `abs(x)`         | Absolute value                 |
//! | `sign(x)`        | Sign (-1, 0, or 1)             |
//! | `floor(x)`       | Round down                     |
//! | `ceil(x)`        | Round up                       |
//! | `round(x)`       | Round to nearest               |
//! | `trunc(x)`       | Truncate toward zero           |
//! | `fract(x)`       | Fractional part                |
//! | `min(a, b)`      | Minimum of two values          |
//! | `max(a, b)`      | Maximum of two values          |
//! | `clamp(x, lo, hi)`| Clamp to range                |
//! | `saturate(x)`    | Clamp to [0, 1]                |
//!
//! ## Interpolation
//!
//! | Function                      | Description                           |
//! |-------------------------------|---------------------------------------|
//! | `lerp(a, b, t)`               | Linear interpolation: a + (b-a)*t     |
//! | `mix(a, b, t)`                | Alias for lerp (GLSL naming)          |
//! | `step(edge, x)`               | 0 if x < edge, else 1                 |
//! | `smoothstep(e0, e1, x)`       | Smooth Hermite interpolation          |
//! | `inverse_lerp(a, b, v)`       | Inverse of lerp: (v-a) / (b-a)        |
//! | `remap(x, i0, i1, o0, o1)`    | Remap from [i0,i1] to [o0,o1]         |
//!
//! # Custom Functions
//!
//! You can register custom functions by implementing the [`ScalarFn`] trait:
//!
//! ```
//! use rhizome_dew_scalar::{ScalarFn, FunctionRegistry, scalar_registry};
//!
//! struct Double;
//! impl ScalarFn<f32> for Double {
//!     fn name(&self) -> &str { "double" }
//!     fn arg_count(&self) -> usize { 1 }
//!     fn call(&self, args: &[f32]) -> f32 { args[0] * 2.0 }
//! }
//!
//! let mut registry = scalar_registry();
//! registry.register(Double);
//! ```
//!
//! # Using f64
//!
//! All functions work with `f64` by specifying the type parameter:
//!
//! ```
//! use rhizome_dew_core::Expr;
//! use rhizome_dew_scalar::{eval, scalar_registry, FunctionRegistry};
//! use std::collections::HashMap;
//!
//! let registry: FunctionRegistry<f64> = scalar_registry();
//! let expr = Expr::parse("sqrt(2)").unwrap();
//! let result: f64 = eval(expr.ast(), &HashMap::new(), &registry).unwrap();
//! assert!((result - std::f64::consts::SQRT_2).abs() < 1e-10);
//! ```

use num_traits::{Float, Num, NumCast, One, PrimInt, Zero};
use rhizome_dew_core::{Ast, BinOp, CompareOp, UnaryOp};

// ============================================================================
// Numeric Trait
// ============================================================================

/// Trait for types that can be used as numeric values in expressions.
///
/// This is a marker trait that combines the necessary bounds for basic
/// arithmetic operations. Both float and integer types implement this.
pub trait Numeric:
    Num + NumCast + Copy + PartialOrd + Zero + One + std::fmt::Debug + Send + Sync + 'static
{
    /// Whether this type supports bitwise operations.
    fn supports_bitwise() -> bool;

    /// Whether this type is a floating-point type.
    fn is_float() -> bool;
}

impl Numeric for f32 {
    fn supports_bitwise() -> bool {
        false
    }
    fn is_float() -> bool {
        true
    }
}

impl Numeric for f64 {
    fn supports_bitwise() -> bool {
        false
    }
    fn is_float() -> bool {
        true
    }
}

impl Numeric for i32 {
    fn supports_bitwise() -> bool {
        true
    }
    fn is_float() -> bool {
        false
    }
}

impl Numeric for i64 {
    fn supports_bitwise() -> bool {
        true
    }
    fn is_float() -> bool {
        false
    }
}

impl Numeric for u32 {
    fn supports_bitwise() -> bool {
        true
    }
    fn is_float() -> bool {
        false
    }
}

impl Numeric for u64 {
    fn supports_bitwise() -> bool {
        true
    }
    fn is_float() -> bool {
        false
    }
}
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "wgsl")]
pub mod wgsl;

#[cfg(feature = "glsl")]
pub mod glsl;

#[cfg(any(feature = "lua", feature = "lua-codegen"))]
pub mod lua;

#[cfg(feature = "cranelift")]
pub mod cranelift;

#[cfg(feature = "optimize")]
pub mod optimize;

// ============================================================================
// Errors
// ============================================================================

/// Scalar evaluation error.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Unknown variable.
    UnknownVariable(String),
    /// Unknown function.
    UnknownFunction(String),
    /// Wrong number of arguments to function.
    WrongArgCount {
        func: String,
        expected: usize,
        got: usize,
    },
    /// Operation not supported for this numeric type.
    UnsupportedOperation(String),
    /// Literal cannot be converted to target type (e.g., 3.14 to i32).
    InvalidLiteral(f64),
    /// Negative exponent not allowed for integer types.
    NegativeExponent,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            Error::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            Error::WrongArgCount {
                func,
                expected,
                got,
            } => {
                write!(f, "function '{func}' expects {expected} args, got {got}")
            }
            Error::UnsupportedOperation(op) => {
                write!(f, "operation '{op}' not supported for this numeric type")
            }
            Error::InvalidLiteral(n) => {
                write!(f, "literal {n} cannot be converted to integer type")
            }
            Error::NegativeExponent => {
                write!(f, "negative exponent not allowed for integer types")
            }
        }
    }
}

impl std::error::Error for Error {}

// ============================================================================
// Function Registry
// ============================================================================

/// A scalar function that can be called from expressions.
pub trait ScalarFn<T>: Send + Sync {
    /// Function name.
    fn name(&self) -> &str;

    /// Number of arguments.
    fn arg_count(&self) -> usize;

    /// Call the function with arguments.
    fn call(&self, args: &[T]) -> T;
}

/// Registry of scalar functions.
#[derive(Clone)]
pub struct FunctionRegistry<T> {
    funcs: HashMap<String, Arc<dyn ScalarFn<T>>>,
}

impl<T> Default for FunctionRegistry<T> {
    fn default() -> Self {
        Self {
            funcs: HashMap::new(),
        }
    }
}

impl<T> FunctionRegistry<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<F: ScalarFn<T> + 'static>(&mut self, func: F) {
        self.funcs.insert(func.name().to_string(), Arc::new(func));
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn ScalarFn<T>>> {
        self.funcs.get(name)
    }

    /// Returns an iterator over all function names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.funcs.keys().map(|s| s.as_str())
    }
}

// ============================================================================
// Evaluation
// ============================================================================

/// Evaluate an AST with scalar float values.
///
/// For integer types, use [`eval_int`].
pub fn eval<T: Float>(
    ast: &Ast,
    vars: &HashMap<String, T>,
    funcs: &FunctionRegistry<T>,
) -> Result<T, Error> {
    match ast {
        Ast::Num(n) => Ok(T::from(*n).unwrap()),

        Ast::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| Error::UnknownVariable(name.clone())),

        Ast::BinOp(op, left, right) => {
            let l = eval(left, vars, funcs)?;
            let r = eval(right, vars, funcs)?;
            match op {
                BinOp::Add => Ok(l + r),
                BinOp::Sub => Ok(l - r),
                BinOp::Mul => Ok(l * r),
                BinOp::Div => Ok(l / r),
                BinOp::Pow => Ok(l.powf(r)),
                BinOp::Rem => Ok(l % r),
                BinOp::BitAnd => Err(Error::UnsupportedOperation("&".into())),
                BinOp::BitOr => Err(Error::UnsupportedOperation("|".into())),
                BinOp::Shl => Err(Error::UnsupportedOperation("<<".into())),
                BinOp::Shr => Err(Error::UnsupportedOperation(">>".into())),
            }
        }

        Ast::UnaryOp(op, inner) => {
            let v = eval(inner, vars, funcs)?;
            match op {
                UnaryOp::Neg => Ok(-v),
                UnaryOp::BitNot => Err(Error::UnsupportedOperation("~".into())),
                UnaryOp::Not => {
                    if v == T::zero() {
                        Ok(T::one())
                    } else {
                        Ok(T::zero())
                    }
                }
            }
        }

        Ast::Compare(op, left, right) => {
            let l = eval(left, vars, funcs)?;
            let r = eval(right, vars, funcs)?;
            let result = match op {
                CompareOp::Lt => l < r,
                CompareOp::Le => l <= r,
                CompareOp::Gt => l > r,
                CompareOp::Ge => l >= r,
                CompareOp::Eq => l == r,
                CompareOp::Ne => l != r,
            };
            Ok(if result { T::one() } else { T::zero() })
        }

        Ast::And(left, right) => {
            let l = eval(left, vars, funcs)?;
            if l == T::zero() {
                Ok(T::zero()) // Short-circuit
            } else {
                let r = eval(right, vars, funcs)?;
                Ok(if r != T::zero() { T::one() } else { T::zero() })
            }
        }

        Ast::Or(left, right) => {
            let l = eval(left, vars, funcs)?;
            if l != T::zero() {
                Ok(T::one()) // Short-circuit
            } else {
                let r = eval(right, vars, funcs)?;
                Ok(if r != T::zero() { T::one() } else { T::zero() })
            }
        }

        Ast::If(cond, then_expr, else_expr) => {
            let c = eval(cond, vars, funcs)?;
            if c != T::zero() {
                eval(then_expr, vars, funcs)
            } else {
                eval(else_expr, vars, funcs)
            }
        }

        Ast::Call(name, args) => {
            let func = funcs
                .get(name)
                .ok_or_else(|| Error::UnknownFunction(name.clone()))?;

            if args.len() != func.arg_count() {
                return Err(Error::WrongArgCount {
                    func: name.clone(),
                    expected: func.arg_count(),
                    got: args.len(),
                });
            }

            let arg_vals: Vec<T> = args
                .iter()
                .map(|a| eval(a, vars, funcs))
                .collect::<Result<_, _>>()?;

            Ok(func.call(&arg_vals))
        }
    }
}

/// Evaluate an AST with integer values.
///
/// Supports bitwise operations and errors on:
/// - Fractional literals (e.g., 3.14)
/// - Negative exponents
/// - Float-only functions (sin, cos, etc.)
pub fn eval_int<T: PrimInt + NumCast>(
    ast: &Ast,
    vars: &HashMap<String, T>,
    funcs: &FunctionRegistry<T>,
) -> Result<T, Error> {
    match ast {
        Ast::Num(n) => {
            // Check if the literal is a whole number
            if n.fract() != 0.0 {
                return Err(Error::InvalidLiteral(*n));
            }
            T::from(*n).ok_or_else(|| Error::InvalidLiteral(*n))
        }

        Ast::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| Error::UnknownVariable(name.clone())),

        Ast::BinOp(op, left, right) => {
            let l = eval_int(left, vars, funcs)?;
            let r = eval_int(right, vars, funcs)?;
            match op {
                BinOp::Add => Ok(l + r),
                BinOp::Sub => Ok(l - r),
                BinOp::Mul => Ok(l * r),
                BinOp::Div => Ok(l / r),
                BinOp::Rem => Ok(l % r),
                BinOp::Pow => {
                    // Check for negative exponent
                    if r < T::zero() {
                        return Err(Error::NegativeExponent);
                    }
                    // Integer power via repeated multiplication
                    let mut result = T::one();
                    let mut exp = r;
                    let mut base = l;
                    while exp > T::zero() {
                        if exp & T::one() == T::one() {
                            result = result * base;
                        }
                        base = base * base;
                        exp = exp >> 1;
                    }
                    Ok(result)
                }
                BinOp::BitAnd => Ok(l & r),
                BinOp::BitOr => Ok(l | r),
                BinOp::Shl => {
                    // Convert shift amount to usize
                    let shift: u32 = r.to_u32().unwrap_or(0);
                    Ok(l << shift as usize)
                }
                BinOp::Shr => {
                    let shift: u32 = r.to_u32().unwrap_or(0);
                    Ok(l >> shift as usize)
                }
            }
        }

        Ast::UnaryOp(op, inner) => {
            let v = eval_int(inner, vars, funcs)?;
            match op {
                UnaryOp::Neg => Ok(T::zero() - v),
                UnaryOp::BitNot => Ok(!v),
                UnaryOp::Not => {
                    if v == T::zero() {
                        Ok(T::one())
                    } else {
                        Ok(T::zero())
                    }
                }
            }
        }

        Ast::Compare(op, left, right) => {
            let l = eval_int(left, vars, funcs)?;
            let r = eval_int(right, vars, funcs)?;
            let result = match op {
                CompareOp::Lt => l < r,
                CompareOp::Le => l <= r,
                CompareOp::Gt => l > r,
                CompareOp::Ge => l >= r,
                CompareOp::Eq => l == r,
                CompareOp::Ne => l != r,
            };
            Ok(if result { T::one() } else { T::zero() })
        }

        Ast::And(left, right) => {
            let l = eval_int(left, vars, funcs)?;
            if l == T::zero() {
                Ok(T::zero())
            } else {
                let r = eval_int(right, vars, funcs)?;
                Ok(if r != T::zero() { T::one() } else { T::zero() })
            }
        }

        Ast::Or(left, right) => {
            let l = eval_int(left, vars, funcs)?;
            if l != T::zero() {
                Ok(T::one())
            } else {
                let r = eval_int(right, vars, funcs)?;
                Ok(if r != T::zero() { T::one() } else { T::zero() })
            }
        }

        Ast::If(cond, then_expr, else_expr) => {
            let c = eval_int(cond, vars, funcs)?;
            if c != T::zero() {
                eval_int(then_expr, vars, funcs)
            } else {
                eval_int(else_expr, vars, funcs)
            }
        }

        Ast::Call(name, args) => {
            let func = funcs
                .get(name)
                .ok_or_else(|| Error::UnknownFunction(name.clone()))?;

            if args.len() != func.arg_count() {
                return Err(Error::WrongArgCount {
                    func: name.clone(),
                    expected: func.arg_count(),
                    got: args.len(),
                });
            }

            let arg_vals: Vec<T> = args
                .iter()
                .map(|a| eval_int(a, vars, funcs))
                .collect::<Result<_, _>>()?;

            Ok(func.call(&arg_vals))
        }
    }
}

// ============================================================================
// Standard Functions - Constants
// ============================================================================

/// Pi constant: pi() = 3.14159...
pub struct Pi;
impl<T: Float> ScalarFn<T> for Pi {
    fn name(&self) -> &str {
        "pi"
    }
    fn arg_count(&self) -> usize {
        0
    }
    fn call(&self, _args: &[T]) -> T {
        T::from(std::f64::consts::PI).unwrap()
    }
}

/// Euler's number: e() = 2.71828...
pub struct E;
impl<T: Float> ScalarFn<T> for E {
    fn name(&self) -> &str {
        "e"
    }
    fn arg_count(&self) -> usize {
        0
    }
    fn call(&self, _args: &[T]) -> T {
        T::from(std::f64::consts::E).unwrap()
    }
}

/// Tau constant: tau() = 2*pi = 6.28318...
pub struct Tau;
impl<T: Float> ScalarFn<T> for Tau {
    fn name(&self) -> &str {
        "tau"
    }
    fn arg_count(&self) -> usize {
        0
    }
    fn call(&self, _args: &[T]) -> T {
        T::from(std::f64::consts::TAU).unwrap()
    }
}

// ============================================================================
// Standard Functions - Trigonometric
// ============================================================================

pub struct Sin;
impl<T: Float> ScalarFn<T> for Sin {
    fn name(&self) -> &str {
        "sin"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].sin()
    }
}

pub struct Cos;
impl<T: Float> ScalarFn<T> for Cos {
    fn name(&self) -> &str {
        "cos"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].cos()
    }
}

pub struct Tan;
impl<T: Float> ScalarFn<T> for Tan {
    fn name(&self) -> &str {
        "tan"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].tan()
    }
}

pub struct Asin;
impl<T: Float> ScalarFn<T> for Asin {
    fn name(&self) -> &str {
        "asin"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].asin()
    }
}

pub struct Acos;
impl<T: Float> ScalarFn<T> for Acos {
    fn name(&self) -> &str {
        "acos"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].acos()
    }
}

pub struct Atan;
impl<T: Float> ScalarFn<T> for Atan {
    fn name(&self) -> &str {
        "atan"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].atan()
    }
}

pub struct Atan2;
impl<T: Float> ScalarFn<T> for Atan2 {
    fn name(&self) -> &str {
        "atan2"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        args[0].atan2(args[1])
    }
}

pub struct Sinh;
impl<T: Float> ScalarFn<T> for Sinh {
    fn name(&self) -> &str {
        "sinh"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].sinh()
    }
}

pub struct Cosh;
impl<T: Float> ScalarFn<T> for Cosh {
    fn name(&self) -> &str {
        "cosh"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].cosh()
    }
}

pub struct Tanh;
impl<T: Float> ScalarFn<T> for Tanh {
    fn name(&self) -> &str {
        "tanh"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].tanh()
    }
}

// ============================================================================
// Standard Functions - Exponential / Logarithmic
// ============================================================================

pub struct Exp;
impl<T: Float> ScalarFn<T> for Exp {
    fn name(&self) -> &str {
        "exp"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].exp()
    }
}

pub struct Exp2;
impl<T: Float> ScalarFn<T> for Exp2 {
    fn name(&self) -> &str {
        "exp2"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].exp2()
    }
}

pub struct Log;
impl<T: Float> ScalarFn<T> for Log {
    fn name(&self) -> &str {
        "log"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].ln()
    }
}

pub struct Ln;
impl<T: Float> ScalarFn<T> for Ln {
    fn name(&self) -> &str {
        "ln"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].ln()
    }
}

pub struct Log2;
impl<T: Float> ScalarFn<T> for Log2 {
    fn name(&self) -> &str {
        "log2"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].log2()
    }
}

pub struct Log10;
impl<T: Float> ScalarFn<T> for Log10 {
    fn name(&self) -> &str {
        "log10"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].log10()
    }
}

pub struct Pow;
impl<T: Float> ScalarFn<T> for Pow {
    fn name(&self) -> &str {
        "pow"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        args[0].powf(args[1])
    }
}

pub struct Sqrt;
impl<T: Float> ScalarFn<T> for Sqrt {
    fn name(&self) -> &str {
        "sqrt"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].sqrt()
    }
}

pub struct InverseSqrt;
impl<T: Float> ScalarFn<T> for InverseSqrt {
    fn name(&self) -> &str {
        "inversesqrt"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        T::one() / args[0].sqrt()
    }
}

// ============================================================================
// Standard Functions - Common Math
// ============================================================================

pub struct Abs;
impl<T: Float> ScalarFn<T> for Abs {
    fn name(&self) -> &str {
        "abs"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].abs()
    }
}

pub struct Sign;
impl<T: Float> ScalarFn<T> for Sign {
    fn name(&self) -> &str {
        "sign"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        let x = args[0];
        if x > T::zero() {
            T::one()
        } else if x < T::zero() {
            -T::one()
        } else {
            T::zero()
        }
    }
}

pub struct Floor;
impl<T: Float> ScalarFn<T> for Floor {
    fn name(&self) -> &str {
        "floor"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].floor()
    }
}

pub struct Ceil;
impl<T: Float> ScalarFn<T> for Ceil {
    fn name(&self) -> &str {
        "ceil"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].ceil()
    }
}

pub struct Round;
impl<T: Float> ScalarFn<T> for Round {
    fn name(&self) -> &str {
        "round"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].round()
    }
}

pub struct Trunc;
impl<T: Float> ScalarFn<T> for Trunc {
    fn name(&self) -> &str {
        "trunc"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].trunc()
    }
}

pub struct Fract;
impl<T: Float> ScalarFn<T> for Fract {
    fn name(&self) -> &str {
        "fract"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].fract()
    }
}

pub struct Min;
impl<T: Float> ScalarFn<T> for Min {
    fn name(&self) -> &str {
        "min"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        args[0].min(args[1])
    }
}

pub struct Max;
impl<T: Float> ScalarFn<T> for Max {
    fn name(&self) -> &str {
        "max"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        args[0].max(args[1])
    }
}

pub struct Clamp;
impl<T: Float> ScalarFn<T> for Clamp {
    fn name(&self) -> &str {
        "clamp"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[T]) -> T {
        args[0].max(args[1]).min(args[2])
    }
}

pub struct Saturate;
impl<T: Float> ScalarFn<T> for Saturate {
    fn name(&self) -> &str {
        "saturate"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].max(T::zero()).min(T::one())
    }
}

// ============================================================================
// Standard Functions - Interpolation
// ============================================================================

/// Linear interpolation: lerp(a, b, t) = a + (b - a) * t
pub struct Lerp;
impl<T: Float> ScalarFn<T> for Lerp {
    fn name(&self) -> &str {
        "lerp"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[T]) -> T {
        let (a, b, t) = (args[0], args[1], args[2]);
        a + (b - a) * t
    }
}

/// Alias for lerp (GLSL naming)
pub struct Mix;
impl<T: Float> ScalarFn<T> for Mix {
    fn name(&self) -> &str {
        "mix"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[T]) -> T {
        let (a, b, t) = (args[0], args[1], args[2]);
        a + (b - a) * t
    }
}

/// Step function: step(edge, x) = x < edge ? 0.0 : 1.0
pub struct Step;
impl<T: Float> ScalarFn<T> for Step {
    fn name(&self) -> &str {
        "step"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        if args[1] < args[0] {
            T::zero()
        } else {
            T::one()
        }
    }
}

/// Smooth Hermite interpolation
pub struct Smoothstep;
impl<T: Float> ScalarFn<T> for Smoothstep {
    fn name(&self) -> &str {
        "smoothstep"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[T]) -> T {
        let (edge0, edge1, x) = (args[0], args[1], args[2]);
        let t = ((x - edge0) / (edge1 - edge0)).max(T::zero()).min(T::one());
        let three = T::from(3.0).unwrap();
        let two = T::from(2.0).unwrap();
        t * t * (three - two * t)
    }
}

/// Inverse lerp: inverse_lerp(a, b, v) = (v - a) / (b - a)
pub struct InverseLerp;
impl<T: Float> ScalarFn<T> for InverseLerp {
    fn name(&self) -> &str {
        "inverse_lerp"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[T]) -> T {
        let (a, b, v) = (args[0], args[1], args[2]);
        (v - a) / (b - a)
    }
}

/// Remap: remap(x, in_lo, in_hi, out_lo, out_hi)
pub struct Remap;
impl<T: Float> ScalarFn<T> for Remap {
    fn name(&self) -> &str {
        "remap"
    }
    fn arg_count(&self) -> usize {
        5
    }
    fn call(&self, args: &[T]) -> T {
        let (x, in_lo, in_hi, out_lo, out_hi) = (args[0], args[1], args[2], args[3], args[4]);
        let t = (x - in_lo) / (in_hi - in_lo);
        out_lo + (out_hi - out_lo) * t
    }
}

// ============================================================================
// Standard Functions - Integer-specific
// ============================================================================

/// Bitwise XOR: xor(a, b)
pub struct Xor;
impl<T: PrimInt> ScalarFn<T> for Xor {
    fn name(&self) -> &str {
        "xor"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        args[0] ^ args[1]
    }
}

/// Integer abs: abs(x) for integer types
pub struct AbsInt;
impl<T: PrimInt> ScalarFn<T> for AbsInt {
    fn name(&self) -> &str {
        "abs"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        let x = args[0];
        if x < T::zero() { T::zero() - x } else { x }
    }
}

/// Integer min: min(a, b) for integer types
pub struct MinInt;
impl<T: PrimInt> ScalarFn<T> for MinInt {
    fn name(&self) -> &str {
        "min"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        if args[0] < args[1] { args[0] } else { args[1] }
    }
}

/// Integer max: max(a, b) for integer types
pub struct MaxInt;
impl<T: PrimInt> ScalarFn<T> for MaxInt {
    fn name(&self) -> &str {
        "max"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        if args[0] > args[1] { args[0] } else { args[1] }
    }
}

/// Integer clamp: clamp(x, lo, hi) for integer types
pub struct ClampInt;
impl<T: PrimInt> ScalarFn<T> for ClampInt {
    fn name(&self) -> &str {
        "clamp"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[T]) -> T {
        let (x, lo, hi) = (args[0], args[1], args[2]);
        if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        }
    }
}

/// Integer sign: sign(x) for integer types
pub struct SignInt;
impl<T: PrimInt> ScalarFn<T> for SignInt {
    fn name(&self) -> &str {
        "sign"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        let x = args[0];
        if x > T::zero() {
            T::one()
        } else if x < T::zero() {
            T::zero() - T::one()
        } else {
            T::zero()
        }
    }
}

// ============================================================================
// Registry
// ============================================================================

/// Registers all standard scalar functions into the given registry.
pub fn register_scalar<T: Float + 'static>(registry: &mut FunctionRegistry<T>) {
    // Constants
    registry.register(Pi);
    registry.register(E);
    registry.register(Tau);

    // Trigonometric
    registry.register(Sin);
    registry.register(Cos);
    registry.register(Tan);
    registry.register(Asin);
    registry.register(Acos);
    registry.register(Atan);
    registry.register(Atan2);
    registry.register(Sinh);
    registry.register(Cosh);
    registry.register(Tanh);

    // Exponential / logarithmic
    registry.register(Exp);
    registry.register(Exp2);
    registry.register(Log);
    registry.register(Ln);
    registry.register(Log2);
    registry.register(Log10);
    registry.register(Pow);
    registry.register(Sqrt);
    registry.register(InverseSqrt);

    // Common math
    registry.register(Abs);
    registry.register(Sign);
    registry.register(Floor);
    registry.register(Ceil);
    registry.register(Round);
    registry.register(Trunc);
    registry.register(Fract);
    registry.register(Min);
    registry.register(Max);
    registry.register(Clamp);
    registry.register(Saturate);

    // Interpolation
    registry.register(Lerp);
    registry.register(Mix);
    registry.register(Step);
    registry.register(Smoothstep);
    registry.register(InverseLerp);
    registry.register(Remap);
}

/// Creates a new registry with all standard scalar functions.
pub fn scalar_registry<T: Float + 'static>() -> FunctionRegistry<T> {
    let mut registry = FunctionRegistry::new();
    register_scalar(&mut registry);
    registry
}

/// Registers standard integer functions into the given registry.
///
/// Includes: abs, min, max, clamp, sign, xor
pub fn register_scalar_int<T: PrimInt + 'static>(registry: &mut FunctionRegistry<T>) {
    registry.register(AbsInt);
    registry.register(MinInt);
    registry.register(MaxInt);
    registry.register(ClampInt);
    registry.register(SignInt);
    registry.register(Xor);
}

/// Creates a new registry with standard integer functions.
pub fn scalar_registry_int<T: PrimInt + 'static>() -> FunctionRegistry<T> {
    let mut registry = FunctionRegistry::new();
    register_scalar_int(&mut registry);
    registry
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn eval_expr(expr: &str, vars: &[(&str, f32)]) -> f32 {
        let registry = scalar_registry();
        let expr = Expr::parse(expr).unwrap();
        let var_map: HashMap<String, f32> = vars.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        eval(expr.ast(), &var_map, &registry).unwrap()
    }

    #[test]
    fn test_constants() {
        assert!((eval_expr("pi()", &[]) - std::f32::consts::PI).abs() < 0.001);
        assert!((eval_expr("e()", &[]) - std::f32::consts::E).abs() < 0.001);
        assert!((eval_expr("tau()", &[]) - std::f32::consts::TAU).abs() < 0.001);
    }

    #[test]
    fn test_trig() {
        assert!(eval_expr("sin(0)", &[]).abs() < 0.001);
        assert!((eval_expr("cos(0)", &[]) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_exp_log() {
        assert!((eval_expr("exp(0)", &[]) - 1.0).abs() < 0.001);
        assert!((eval_expr("ln(1)", &[]) - 0.0).abs() < 0.001);
        assert!((eval_expr("sqrt(16)", &[]) - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_common() {
        assert_eq!(eval_expr("abs(-5)", &[]), 5.0);
        assert_eq!(eval_expr("floor(3.7)", &[]), 3.0);
        assert_eq!(eval_expr("ceil(3.2)", &[]), 4.0);
        assert_eq!(eval_expr("min(3, 7)", &[]), 3.0);
        assert_eq!(eval_expr("max(3, 7)", &[]), 7.0);
        assert_eq!(eval_expr("clamp(5, 0, 3)", &[]), 3.0);
        assert_eq!(eval_expr("saturate(1.5)", &[]), 1.0);
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(eval_expr("lerp(0, 10, 0.5)", &[]), 5.0);
        assert_eq!(eval_expr("mix(0, 10, 0.5)", &[]), 5.0);
        assert_eq!(eval_expr("step(0.5, 0.3)", &[]), 0.0);
        assert_eq!(eval_expr("step(0.5, 0.7)", &[]), 1.0);
        assert!((eval_expr("smoothstep(0, 1, 0.5)", &[]) - 0.5).abs() < 0.1);
        assert_eq!(eval_expr("inverse_lerp(0, 10, 5)", &[]), 0.5);
    }

    #[test]
    fn test_remap() {
        assert_eq!(eval_expr("remap(5, 0, 10, 0, 100)", &[]), 50.0);
    }

    #[test]
    fn test_with_variables() {
        let v = eval_expr("sin(x * pi())", &[("x", 0.5)]);
        assert!((v - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f64() {
        let registry: FunctionRegistry<f64> = scalar_registry();
        let expr = Expr::parse("sin(x) + 1").unwrap();
        let vars: HashMap<String, f64> = [("x".to_string(), 0.0)].into();
        let result = eval(expr.ast(), &vars, &registry).unwrap();
        assert!((result - 1.0).abs() < 0.001);
    }

    // Integer expression tests
    mod int_tests {
        use super::*;

        fn eval_int_expr(expr_str: &str, vars: &[(&str, i32)]) -> i32 {
            let registry = scalar_registry_int();
            let expr = Expr::parse(expr_str).unwrap();
            let var_map: HashMap<String, i32> =
                vars.iter().map(|(k, v)| (k.to_string(), *v)).collect();
            eval_int(expr.ast(), &var_map, &registry).unwrap()
        }

        #[test]
        fn test_int_arithmetic() {
            assert_eq!(eval_int_expr("5 + 3", &[]), 8);
            assert_eq!(eval_int_expr("10 - 4", &[]), 6);
            assert_eq!(eval_int_expr("6 * 7", &[]), 42);
            assert_eq!(eval_int_expr("15 / 4", &[]), 3); // Integer division
        }

        #[test]
        fn test_int_modulo() {
            assert_eq!(eval_int_expr("8 % 3", &[]), 2);
            assert_eq!(eval_int_expr("10 % 5", &[]), 0);
            assert_eq!(eval_int_expr("17 % 7", &[]), 3);
        }

        #[test]
        fn test_int_power() {
            assert_eq!(eval_int_expr("2 ^ 3", &[]), 8);
            assert_eq!(eval_int_expr("3 ^ 4", &[]), 81);
            assert_eq!(eval_int_expr("5 ^ 0", &[]), 1);
        }

        #[test]
        fn test_int_bitwise() {
            assert_eq!(eval_int_expr("5 & 3", &[]), 1); // 0101 & 0011 = 0001
            assert_eq!(eval_int_expr("5 | 3", &[]), 7); // 0101 | 0011 = 0111
            assert_eq!(eval_int_expr("xor(5, 3)", &[]), 6); // 0101 ^ 0011 = 0110
            assert_eq!(eval_int_expr("1 << 4", &[]), 16);
            assert_eq!(eval_int_expr("16 >> 2", &[]), 4);
        }

        #[test]
        fn test_int_bitnot() {
            // ~0 for i32 is -1
            assert_eq!(eval_int_expr("~0", &[]), -1);
        }

        #[test]
        fn test_int_functions() {
            assert_eq!(eval_int_expr("abs(-5)", &[]), 5);
            assert_eq!(eval_int_expr("min(3, 7)", &[]), 3);
            assert_eq!(eval_int_expr("max(3, 7)", &[]), 7);
            assert_eq!(eval_int_expr("clamp(5, 0, 3)", &[]), 3);
            assert_eq!(eval_int_expr("sign(-10)", &[]), -1);
            assert_eq!(eval_int_expr("sign(10)", &[]), 1);
            assert_eq!(eval_int_expr("sign(0)", &[]), 0);
        }

        #[test]
        fn test_int_with_variables() {
            assert_eq!(eval_int_expr("x + y", &[("x", 5), ("y", 3)]), 8);
            assert_eq!(
                eval_int_expr("steps % beats", &[("steps", 8), ("beats", 3)]),
                2
            );
        }

        #[test]
        fn test_int_fractional_literal_error() {
            let registry: FunctionRegistry<i32> = scalar_registry_int();
            let expr = Expr::parse("3.14 + 1").unwrap();
            let result = eval_int(expr.ast(), &HashMap::new(), &registry);
            assert!(matches!(result, Err(Error::InvalidLiteral(_))));
        }

        #[test]
        fn test_int_negative_exponent_error() {
            let registry: FunctionRegistry<i32> = scalar_registry_int();
            let expr = Expr::parse("2 ^ -1").unwrap();
            let vars: HashMap<String, i32> = HashMap::new();
            let result = eval_int(expr.ast(), &vars, &registry);
            assert!(matches!(result, Err(Error::NegativeExponent)));
        }
    }
}
