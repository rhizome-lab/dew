//! Scalar function optimization passes.
//!
//! This module provides optimization passes for scalar functions like `sin`, `cos`, `sqrt`, etc.
//! When all arguments are constants, the function is evaluated at compile time.
//!
//! # Example
//!
//! ```
//! use rhizome_dew_core::Expr;
//! use rhizome_dew_core::optimize::{optimize, standard_passes};
//! use rhizome_dew_scalar::optimize::ScalarConstantFolding;
//!
//! let expr = Expr::parse("sin(0) + cos(0)").unwrap();
//!
//! // Combine core passes with scalar constant folding
//! let mut passes: Vec<&dyn rhizome_dew_core::optimize::Pass> = standard_passes();
//! passes.push(&ScalarConstantFolding);
//!
//! let optimized = optimize(expr.ast().clone(), &passes);
//! // sin(0) + cos(0) → 0 + 1 → 1
//! assert_eq!(optimized.to_string(), "1");
//! ```

use rhizome_dew_core::Ast;
use rhizome_dew_core::optimize::Pass;

/// Constant folding for scalar functions.
///
/// Evaluates scalar function calls when all arguments are numeric literals.
///
/// # Supported Functions
///
/// ## Constants (0-arg)
/// - `pi()` → 3.14159...
/// - `e()` → 2.71828...
/// - `tau()` → 6.28318...
///
/// ## Trigonometric (1-arg)
/// - `sin(x)`, `cos(x)`, `tan(x)`
/// - `asin(x)`, `acos(x)`, `atan(x)`
/// - `sinh(x)`, `cosh(x)`, `tanh(x)`
///
/// ## Exponential/Logarithmic (1-arg)
/// - `exp(x)`, `exp2(x)`
/// - `log(x)`, `ln(x)`, `log2(x)`, `log10(x)`
/// - `sqrt(x)`, `inversesqrt(x)`
///
/// ## Common Math (1-arg)
/// - `abs(x)`, `sign(x)`
/// - `floor(x)`, `ceil(x)`, `round(x)`, `trunc(x)`, `fract(x)`
/// - `saturate(x)`
///
/// ## Two-argument
/// - `atan2(y, x)`, `pow(x, y)`
/// - `min(a, b)`, `max(a, b)`, `step(edge, x)`
///
/// ## Three-argument
/// - `clamp(x, lo, hi)`
/// - `lerp(a, b, t)`, `mix(a, b, t)`
/// - `smoothstep(e0, e1, x)`
/// - `inverse_lerp(a, b, v)`
///
/// ## Five-argument
/// - `remap(x, i0, i1, o0, o1)`
pub struct ScalarConstantFolding;

impl Pass for ScalarConstantFolding {
    fn transform(&self, ast: &Ast) -> Option<Ast> {
        let Ast::Call(name, args) = ast else {
            return None;
        };

        // Check if all arguments are constant numbers
        let const_args: Option<Vec<f64>> = args
            .iter()
            .map(|a| match a {
                Ast::Num(n) => Some(*n),
                _ => None,
            })
            .collect();

        let const_args = const_args?;

        // Evaluate the function
        let result = evaluate_scalar_function(name, &const_args)?;

        Some(Ast::Num(result))
    }
}

/// Evaluates a scalar function with constant arguments.
/// Returns None if the function is unknown or has wrong arity.
fn evaluate_scalar_function(name: &str, args: &[f64]) -> Option<f64> {
    match (name, args.len()) {
        // Constants (0-arg)
        ("pi", 0) => Some(std::f64::consts::PI),
        ("e", 0) => Some(std::f64::consts::E),
        ("tau", 0) => Some(std::f64::consts::TAU),

        // Trigonometric (1-arg)
        ("sin", 1) => Some(args[0].sin()),
        ("cos", 1) => Some(args[0].cos()),
        ("tan", 1) => Some(args[0].tan()),
        ("asin", 1) => Some(args[0].asin()),
        ("acos", 1) => Some(args[0].acos()),
        ("atan", 1) => Some(args[0].atan()),
        ("sinh", 1) => Some(args[0].sinh()),
        ("cosh", 1) => Some(args[0].cosh()),
        ("tanh", 1) => Some(args[0].tanh()),

        // Exponential/Logarithmic (1-arg)
        ("exp", 1) => Some(args[0].exp()),
        ("exp2", 1) => Some(args[0].exp2()),
        ("log", 1) => Some(args[0].ln()),
        ("ln", 1) => Some(args[0].ln()),
        ("log2", 1) => Some(args[0].log2()),
        ("log10", 1) => Some(args[0].log10()),
        ("sqrt", 1) => Some(args[0].sqrt()),
        ("inversesqrt", 1) => Some(1.0 / args[0].sqrt()),

        // Common math (1-arg)
        ("abs", 1) => Some(args[0].abs()),
        ("sign", 1) => Some(if args[0] > 0.0 {
            1.0
        } else if args[0] < 0.0 {
            -1.0
        } else {
            0.0
        }),
        ("floor", 1) => Some(args[0].floor()),
        ("ceil", 1) => Some(args[0].ceil()),
        ("round", 1) => Some(args[0].round()),
        ("trunc", 1) => Some(args[0].trunc()),
        ("fract", 1) => Some(args[0].fract()),
        ("saturate", 1) => Some(args[0].clamp(0.0, 1.0)),

        // Two-argument
        ("atan2", 2) => Some(args[0].atan2(args[1])),
        ("pow", 2) => Some(args[0].powf(args[1])),
        ("min", 2) => Some(args[0].min(args[1])),
        ("max", 2) => Some(args[0].max(args[1])),
        ("step", 2) => Some(if args[1] < args[0] { 0.0 } else { 1.0 }),

        // Three-argument
        ("clamp", 3) => Some(args[0].clamp(args[1], args[2])),
        ("lerp", 3) | ("mix", 3) => {
            let (a, b, t) = (args[0], args[1], args[2]);
            Some(a + (b - a) * t)
        }
        ("smoothstep", 3) => {
            let (edge0, edge1, x) = (args[0], args[1], args[2]);
            let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
            Some(t * t * (3.0 - 2.0 * t))
        }
        ("inverse_lerp", 3) => {
            let (a, b, v) = (args[0], args[1], args[2]);
            Some((v - a) / (b - a))
        }

        // Five-argument
        ("remap", 5) => {
            let (x, in_lo, in_hi, out_lo, out_hi) = (args[0], args[1], args[2], args[3], args[4]);
            let t = (x - in_lo) / (in_hi - in_lo);
            Some(out_lo + (out_hi - out_lo) * t)
        }

        // Unknown function or wrong arity
        _ => None,
    }
}

/// Returns all scalar optimization passes.
///
/// Currently includes:
/// - [`ScalarConstantFolding`]
pub fn scalar_passes() -> Vec<&'static dyn Pass> {
    vec![&ScalarConstantFolding]
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
        passes.push(&ScalarConstantFolding);
        let result = optimize(expr.ast().clone(), &passes);
        result.to_string()
    }

    fn optimized_value(input: &str) -> f64 {
        let expr = Expr::parse(input).unwrap();
        let mut passes: Vec<&dyn Pass> = standard_passes();
        passes.push(&ScalarConstantFolding);
        let result = optimize(expr.ast().clone(), &passes);
        match result {
            Ast::Num(n) => n,
            other => panic!("expected Num, got {other:?}"),
        }
    }

    // Constants
    #[test]
    fn test_pi() {
        let v = optimized_value("pi()");
        assert!((v - std::f64::consts::PI).abs() < 0.0001);
    }

    #[test]
    fn test_e() {
        let v = optimized_value("e()");
        assert!((v - std::f64::consts::E).abs() < 0.0001);
    }

    #[test]
    fn test_tau() {
        let v = optimized_value("tau()");
        assert!((v - std::f64::consts::TAU).abs() < 0.0001);
    }

    // Trigonometric
    #[test]
    fn test_sin_zero() {
        let v = optimized_value("sin(0)");
        assert!(v.abs() < 0.0001);
    }

    #[test]
    fn test_cos_zero() {
        let v = optimized_value("cos(0)");
        assert!((v - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_tan_zero() {
        let v = optimized_value("tan(0)");
        assert!(v.abs() < 0.0001);
    }

    // Exponential/Logarithmic
    #[test]
    fn test_exp_zero() {
        let v = optimized_value("exp(0)");
        assert!((v - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_ln_one() {
        let v = optimized_value("ln(1)");
        assert!(v.abs() < 0.0001);
    }

    #[test]
    fn test_sqrt() {
        let v = optimized_value("sqrt(16)");
        assert!((v - 4.0).abs() < 0.0001);
    }

    #[test]
    fn test_log2() {
        let v = optimized_value("log2(8)");
        assert!((v - 3.0).abs() < 0.0001);
    }

    // Common math
    #[test]
    fn test_abs() {
        assert_eq!(optimized("abs(-5)"), "5");
    }

    #[test]
    fn test_floor() {
        assert_eq!(optimized("floor(3.7)"), "3");
    }

    #[test]
    fn test_ceil() {
        assert_eq!(optimized("ceil(3.2)"), "4");
    }

    #[test]
    fn test_round() {
        assert_eq!(optimized("round(3.5)"), "4");
    }

    #[test]
    fn test_sign() {
        assert_eq!(optimized("sign(-5)"), "-1");
        assert_eq!(optimized("sign(5)"), "1");
        assert_eq!(optimized("sign(0)"), "0");
    }

    #[test]
    fn test_saturate() {
        assert_eq!(optimized("saturate(1.5)"), "1");
        assert_eq!(optimized("saturate(-0.5)"), "0");
    }

    // Two-argument
    #[test]
    fn test_min_max() {
        assert_eq!(optimized("min(3, 7)"), "3");
        assert_eq!(optimized("max(3, 7)"), "7");
    }

    #[test]
    fn test_pow() {
        let v = optimized_value("pow(2, 3)");
        assert!((v - 8.0).abs() < 0.0001);
    }

    #[test]
    fn test_step() {
        assert_eq!(optimized("step(0.5, 0.3)"), "0");
        assert_eq!(optimized("step(0.5, 0.7)"), "1");
    }

    // Three-argument
    #[test]
    fn test_clamp() {
        assert_eq!(optimized("clamp(5, 0, 3)"), "3");
        assert_eq!(optimized("clamp(-1, 0, 3)"), "0");
    }

    #[test]
    fn test_lerp() {
        assert_eq!(optimized("lerp(0, 10, 0.5)"), "5");
        assert_eq!(optimized("mix(0, 10, 0.5)"), "5");
    }

    #[test]
    fn test_smoothstep() {
        let v = optimized_value("smoothstep(0, 1, 0.5)");
        assert!((v - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_inverse_lerp() {
        let v = optimized_value("inverse_lerp(0, 10, 5)");
        assert!((v - 0.5).abs() < 0.0001);
    }

    // Five-argument
    #[test]
    fn test_remap() {
        let v = optimized_value("remap(5, 0, 10, 0, 100)");
        assert!((v - 50.0).abs() < 0.0001);
    }

    // Combined with core passes
    #[test]
    fn test_combined_sin_cos() {
        // sin(0) + cos(0) → 0 + 1 → 1
        assert_eq!(optimized("sin(0) + cos(0)"), "1");
    }

    #[test]
    fn test_combined_with_arithmetic() {
        // sqrt(4) * 3 + 1 → 2 * 3 + 1 → 6 + 1 → 7
        assert_eq!(optimized("sqrt(4) * 3 + 1"), "7");
    }

    #[test]
    fn test_nested_calls() {
        // sqrt(sqrt(16)) → sqrt(4) → 2
        assert_eq!(optimized("sqrt(sqrt(16))"), "2");
    }

    #[test]
    fn test_partial_constant() {
        // sqrt(x) should not be folded (x is variable)
        assert_eq!(optimized("sqrt(x)"), "sqrt(x)");
    }

    #[test]
    fn test_partial_mixed() {
        // sqrt(4) + x → 2 + x
        assert_eq!(optimized("sqrt(4) + x"), "(2 + x)");
    }
}
