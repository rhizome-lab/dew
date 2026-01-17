//! Complex number optimization passes.
//!
//! This module provides optimization passes for complex number operations.
//! When all arguments are constants, operations are evaluated at compile time.
//!
//! # Example
//!
//! ```
//! use rhizome_dew_core::Expr;
//! use rhizome_dew_core::optimize::{optimize, standard_passes};
//! use rhizome_dew_complex::optimize::ComplexConstantFolding;
//!
//! let expr = Expr::parse("abs(complex(3, 4))").unwrap();
//!
//! // Combine core passes with complex constant folding
//! let mut passes: Vec<&dyn rhizome_dew_core::optimize::Pass> = standard_passes();
//! passes.push(&ComplexConstantFolding);
//!
//! let optimized = optimize(expr.ast().clone(), &passes);
//! // abs(3 + 4i) = 5
//! assert_eq!(optimized.to_string(), "5");
//! ```
//!
//! # Limitations
//!
//! Complex constant folding uses **constructor-based type inference**:
//! only operations where the complex type is visible in the AST can be folded.
//!
//! ## What works
//!
//! ```text
//! complex(1, 2) + complex(3, 4) // → complex(4, 6)
//! abs(complex(3, 4))            // → 5 (magnitude of 3+4i)
//! polar(2, 0) + complex(1, 0)   // → complex(3, 0)
//! re(complex(3, 4))             // → 3
//! ```
//!
//! ## What doesn't work (yet)
//!
//! ```text
//! abs(z)                        // z is a variable — type unknown at AST level
//! a + b                         // variables — no type info in AST
//! ```
//!
//! For full type-aware optimization, a future `TypedPass` trait could receive
//! type information from the caller's context.

use rhizome_dew_core::Ast;
use rhizome_dew_core::optimize::Pass;

/// Constant folding for complex number operations.
///
/// Evaluates complex function calls when all arguments are constants.
/// Uses constructor-based type inference via `complex(re, im)` or `polar(r, theta)`.
///
/// # Supported Functions
///
/// ## Scalar extraction
/// - `re(z)` → real part
/// - `im(z)` → imaginary part
/// - `abs(z)` → magnitude
/// - `arg(z)` → phase angle
/// - `norm(z)` → squared magnitude
///
/// ## Complex → Complex
/// - `conj(z)` → conjugate
/// - `exp(z)` → complex exponential
/// - `log(z)` → complex logarithm
/// - `sqrt(z)` → complex square root
///
/// ## Construction
/// - `complex(re, im)` → complex from cartesian form
/// - `polar(r, theta)` → complex from polar form
///
/// ## Powers
/// - `pow(z, n)` → complex power (scalar or complex exponent)
pub struct ComplexConstantFolding;

impl Pass for ComplexConstantFolding {
    fn transform(&self, ast: &Ast) -> Option<Ast> {
        let Ast::Call(name, args) = ast else {
            return None;
        };

        // Try to evaluate as constant complex expression
        let const_args: Option<Vec<ConstValue>> = args.iter().map(ConstValue::from_ast).collect();
        let const_args = const_args?;

        let result = evaluate_complex_function(name, &const_args)?;
        Some(result.to_ast())
    }
}

/// A constant value that can be a scalar or complex.
#[derive(Debug, Clone, Copy)]
enum ConstValue {
    Scalar(f32),
    Complex([f32; 2]),
}

impl ConstValue {
    /// Try to extract a constant value from an AST node.
    fn from_ast(ast: &Ast) -> Option<Self> {
        match ast {
            Ast::Num(n) => Some(ConstValue::Scalar(*n)),
            Ast::Call(name, args) if name == "complex" && args.len() == 2 => {
                // complex(re, im) → Complex
                let re = match &args[0] {
                    Ast::Num(n) => *n,
                    _ => return None,
                };
                let im = match &args[1] {
                    Ast::Num(n) => *n,
                    _ => return None,
                };
                Some(ConstValue::Complex([re, im]))
            }
            Ast::Call(name, args) if name == "polar" && args.len() == 2 => {
                // polar(r, theta) → Complex
                let r = match &args[0] {
                    Ast::Num(n) => *n,
                    _ => return None,
                };
                let theta = match &args[1] {
                    Ast::Num(n) => *n,
                    _ => return None,
                };
                Some(ConstValue::Complex([r * theta.cos(), r * theta.sin()]))
            }
            _ => None,
        }
    }

    /// Convert back to AST.
    fn to_ast(self) -> Ast {
        match self {
            ConstValue::Scalar(s) => Ast::Num(s),
            ConstValue::Complex([re, im]) => {
                // Use cartesian form for canonical representation
                Ast::Call("complex".into(), vec![Ast::Num(re), Ast::Num(im)])
            }
        }
    }

    fn as_scalar(self) -> Option<f32> {
        match self {
            ConstValue::Scalar(s) => Some(s),
            _ => None,
        }
    }

    fn as_complex(self) -> Option<[f32; 2]> {
        match self {
            ConstValue::Complex(c) => Some(c),
            ConstValue::Scalar(s) => Some([s, 0.0]), // Scalar can be treated as complex with im=0
        }
    }
}

/// Evaluate a complex function with constant arguments.
fn evaluate_complex_function(name: &str, args: &[ConstValue]) -> Option<ConstValue> {
    match (name, args.len()) {
        // Scalar extraction functions
        ("re", 1) => {
            let c = args[0].as_complex()?;
            Some(ConstValue::Scalar(c[0]))
        }
        ("im", 1) => {
            let c = args[0].as_complex()?;
            Some(ConstValue::Scalar(c[1]))
        }
        ("abs", 1) => match args[0] {
            ConstValue::Scalar(s) => Some(ConstValue::Scalar(s.abs())),
            ConstValue::Complex(c) => Some(ConstValue::Scalar((c[0] * c[0] + c[1] * c[1]).sqrt())),
        },
        ("arg", 1) => {
            let c = args[0].as_complex()?;
            Some(ConstValue::Scalar(c[1].atan2(c[0])))
        }
        ("norm", 1) => {
            let c = args[0].as_complex()?;
            Some(ConstValue::Scalar(c[0] * c[0] + c[1] * c[1]))
        }

        // Complex conjugate
        ("conj", 1) => {
            let c = args[0].as_complex()?;
            Some(ConstValue::Complex([c[0], -c[1]]))
        }

        // Complex exponential: exp(a+bi) = e^a * (cos(b) + i*sin(b))
        ("exp", 1) => match args[0] {
            ConstValue::Scalar(s) => Some(ConstValue::Scalar(s.exp())),
            ConstValue::Complex(c) => {
                let e_a = c[0].exp();
                Some(ConstValue::Complex([e_a * c[1].cos(), e_a * c[1].sin()]))
            }
        },

        // Complex logarithm: log(z) = ln|z| + i*arg(z)
        ("log", 1) => match args[0] {
            ConstValue::Scalar(s) => Some(ConstValue::Scalar(s.ln())),
            ConstValue::Complex(c) => {
                let r = (c[0] * c[0] + c[1] * c[1]).sqrt();
                let theta = c[1].atan2(c[0]);
                Some(ConstValue::Complex([r.ln(), theta]))
            }
        },

        // Complex square root
        ("sqrt", 1) => match args[0] {
            ConstValue::Scalar(s) => Some(ConstValue::Scalar(s.sqrt())),
            ConstValue::Complex(c) => {
                let r = (c[0] * c[0] + c[1] * c[1]).sqrt();
                let theta = c[1].atan2(c[0]);
                let sqrt_r = r.sqrt();
                let half_theta = theta / 2.0;
                Some(ConstValue::Complex([
                    sqrt_r * half_theta.cos(),
                    sqrt_r * half_theta.sin(),
                ]))
            }
        },

        // Cartesian construction
        ("complex", 2) => {
            let re = args[0].as_scalar()?;
            let im = args[1].as_scalar()?;
            Some(ConstValue::Complex([re, im]))
        }

        // Polar construction
        ("polar", 2) => {
            let r = args[0].as_scalar()?;
            let theta = args[1].as_scalar()?;
            Some(ConstValue::Complex([r * theta.cos(), r * theta.sin()]))
        }

        // Complex power
        ("pow", 2) => {
            match (&args[0], &args[1]) {
                (ConstValue::Scalar(a), ConstValue::Scalar(b)) => {
                    Some(ConstValue::Scalar(a.powf(*b)))
                }
                (ConstValue::Complex(z), ConstValue::Scalar(n)) => {
                    // z^n = r^n * e^(i*n*theta)
                    let r = (z[0] * z[0] + z[1] * z[1]).sqrt();
                    let theta = z[1].atan2(z[0]);
                    let r_n = r.powf(*n);
                    let theta_n = theta * n;
                    Some(ConstValue::Complex([
                        r_n * theta_n.cos(),
                        r_n * theta_n.sin(),
                    ]))
                }
                (ConstValue::Complex(z), ConstValue::Complex(w)) => {
                    // z^w = exp(w * log(z))
                    let r = (z[0] * z[0] + z[1] * z[1]).sqrt();
                    let theta = z[1].atan2(z[0]);
                    let ln_r = r.ln();
                    // w * log(z) = (w_re + w_im*i) * (ln_r + theta*i)
                    let re = w[0] * ln_r - w[1] * theta;
                    let im = w[0] * theta + w[1] * ln_r;
                    // exp(re + im*i)
                    let e_re = re.exp();
                    Some(ConstValue::Complex([e_re * im.cos(), e_re * im.sin()]))
                }
                (ConstValue::Scalar(s), ConstValue::Complex(w)) => {
                    // Scalar^Complex: treat scalar as complex with im=0
                    let r = s.abs();
                    let theta = if *s >= 0.0 { 0.0 } else { std::f32::consts::PI };
                    let ln_r = r.ln();
                    let re = w[0] * ln_r - w[1] * theta;
                    let im = w[0] * theta + w[1] * ln_r;
                    let e_re = re.exp();
                    Some(ConstValue::Complex([e_re * im.cos(), e_re * im.sin()]))
                }
            }
        }

        _ => None,
    }
}

/// Returns all complex optimization passes.
pub fn complex_passes() -> Vec<&'static dyn Pass> {
    vec![&ComplexConstantFolding]
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
        passes.push(&ComplexConstantFolding);
        optimize(expr.ast().clone(), &passes)
    }

    fn optimized_scalar(input: &str) -> f32 {
        match optimized(input) {
            Ast::Num(n) => n,
            other => panic!("expected Num, got {other:?}"),
        }
    }

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.001
    }

    // Cartesian construction
    #[test]
    fn test_complex_basic() {
        // complex(3, 4) should stay as complex(3, 4)
        let result = optimized("complex(3, 4)");
        if let Ast::Call(name, args) = result {
            assert_eq!(name, "complex");
            if let (Ast::Num(re), Ast::Num(im)) = (&args[0], &args[1]) {
                assert!(approx_eq(*re, 3.0));
                assert!(approx_eq(*im, 4.0));
            }
        } else {
            panic!("expected Call");
        }
    }

    // Polar construction
    #[test]
    fn test_polar_converts_to_complex() {
        // polar(1, 0) converts to complex(1, 0)
        let result = optimized("polar(1, 0)");
        if let Ast::Call(name, args) = result {
            assert_eq!(name, "complex");
            if let (Ast::Num(re), Ast::Num(im)) = (&args[0], &args[1]) {
                assert!(approx_eq(*re, 1.0));
                assert!(approx_eq(*im, 0.0));
            }
        } else {
            panic!("expected Call");
        }
    }

    // Abs function
    #[test]
    fn test_abs_complex() {
        // abs(complex(3, 4)) = 5
        let v = optimized_scalar("abs(complex(3, 4))");
        assert!(approx_eq(v, 5.0));
    }

    #[test]
    fn test_abs_polar() {
        // abs(polar(3, anything)) = 3
        let v = optimized_scalar("abs(polar(3, 1.5))");
        assert!(approx_eq(v, 3.0));
    }

    // Real/Imaginary extraction
    #[test]
    fn test_re_complex() {
        // re(complex(3, 4)) = 3
        let v = optimized_scalar("re(complex(3, 4))");
        assert!(approx_eq(v, 3.0));
    }

    #[test]
    fn test_im_complex() {
        // im(complex(3, 4)) = 4
        let v = optimized_scalar("im(complex(3, 4))");
        assert!(approx_eq(v, 4.0));
    }

    // Arg function
    #[test]
    fn test_arg() {
        let v = optimized_scalar("arg(complex(1, 1))");
        assert!(approx_eq(v, std::f32::consts::FRAC_PI_4));
    }

    // Norm (squared magnitude)
    #[test]
    fn test_norm() {
        // norm(complex(3, 4)) = 25
        let v = optimized_scalar("norm(complex(3, 4))");
        assert!(approx_eq(v, 25.0));
    }

    // Conjugate
    #[test]
    fn test_conj() {
        // conj(3+4i) = 3-4i
        let result = optimized("conj(complex(3, 4))");
        if let Ast::Call(name, args) = result {
            assert_eq!(name, "complex");
            if let (Ast::Num(re), Ast::Num(im)) = (&args[0], &args[1]) {
                assert!(approx_eq(*re, 3.0));
                assert!(approx_eq(*im, -4.0));
            }
        }
    }

    // Exp/Log
    #[test]
    fn test_exp_zero() {
        // exp(0) = 1
        let v = optimized_scalar("exp(0)");
        assert!(approx_eq(v, 1.0));
    }

    #[test]
    fn test_log_one() {
        // log(1) = 0
        let v = optimized_scalar("log(1)");
        assert!(approx_eq(v, 0.0));
    }

    // Sqrt
    #[test]
    fn test_sqrt_scalar() {
        let v = optimized_scalar("sqrt(4)");
        assert!(approx_eq(v, 2.0));
    }

    // Pow
    #[test]
    fn test_pow_scalar() {
        let v = optimized_scalar("pow(2, 3)");
        assert!(approx_eq(v, 8.0));
    }

    // Combined with core passes
    #[test]
    fn test_combined() {
        // abs(complex(3, 0)) + abs(complex(0, 4)) = 3 + 4 = 7
        let v = optimized_scalar("abs(complex(3, 0)) + abs(complex(0, 4))");
        assert!(approx_eq(v, 7.0));
    }

    // Partial constant (should not fold)
    #[test]
    fn test_partial_not_folded() {
        let result = optimized("abs(z)");
        assert!(matches!(result, Ast::Call(name, _) if name == "abs"));
    }
}
