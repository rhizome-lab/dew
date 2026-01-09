//! Standard function library for sap expressions.
//!
//! Provides common math functions (sin, cos, sqrt, etc.) and constants (pi, e)
//! that can be registered with an expression `FunctionRegistry`.
//!
//! # Usage
//!
//! ```
//! use rhizome_sap_core::{Expr, FunctionRegistry};
//! use rhizome_sap_std::register_std;
//! use std::collections::HashMap;
//!
//! let mut registry = FunctionRegistry::new();
//! register_std(&mut registry);
//!
//! let expr = Expr::parse("sin(x) + pi()").unwrap();
//! let vars: HashMap<String, f32> = [("x".to_string(), 0.0)].into();
//! let value = expr.eval(&vars, &registry).unwrap();
//! ```
//!
//! # Backend-specific implementations
//!
//! Enable feature flags to get backend-specific function registries:
//!
//! - `wgsl` - WGSL code generation functions
//! - `lua` - Lua code generation functions
//! - `cranelift` - Cranelift JIT functions (limited to non-transcendental)

use rhizome_sap_core::{Ast, BinOp, ExprFn, FunctionRegistry};

#[cfg(feature = "wgsl")]
pub mod wgsl;

#[cfg(feature = "lua")]
pub mod lua;

#[cfg(feature = "cranelift")]
pub mod cranelift;

// Re-export convenience functions
#[cfg(feature = "wgsl")]
pub use wgsl::{register_wgsl, wgsl_std_registry};

#[cfg(feature = "lua")]
pub use lua::{lua_std_registry, register_lua};

#[cfg(feature = "cranelift")]
pub use cranelift::{cranelift_std_registry, register_cranelift};

#[cfg(all(test, feature = "all"))]
mod parity_tests;

#[cfg(all(test, feature = "all"))]
mod exhaustive_tests;

// ============================================================================
// Macro for simple functions
// ============================================================================

macro_rules! define_fn {
    ($name:ident, $str_name:literal, $args:literal, |$($arg:ident),*| $body:expr) => {
        pub struct $name;

        impl ExprFn for $name {
            fn name(&self) -> &str { $str_name }
            fn arg_count(&self) -> usize { $args }
            fn call(&self, args: &[f32]) -> f32 {
                let [$($arg),*] = args else { return 0.0 };
                $body
            }
        }
    };
}

// ============================================================================
// Constants (0-arg functions)
// ============================================================================

/// Pi constant: pi() = 3.14159...
pub struct Pi;
impl ExprFn for Pi {
    fn name(&self) -> &str {
        "pi"
    }
    fn arg_count(&self) -> usize {
        0
    }
    fn call(&self, _args: &[f32]) -> f32 {
        std::f32::consts::PI
    }
}

/// Euler's number: e() = 2.71828...
pub struct E;
impl ExprFn for E {
    fn name(&self) -> &str {
        "e"
    }
    fn arg_count(&self) -> usize {
        0
    }
    fn call(&self, _args: &[f32]) -> f32 {
        std::f32::consts::E
    }
}

/// Tau constant: tau() = 2*pi = 6.28318...
pub struct Tau;
impl ExprFn for Tau {
    fn name(&self) -> &str {
        "tau"
    }
    fn arg_count(&self) -> usize {
        0
    }
    fn call(&self, _args: &[f32]) -> f32 {
        std::f32::consts::TAU
    }
}

// ============================================================================
// Trigonometric functions
// ============================================================================

define_fn!(Sin, "sin", 1, |a| a.sin());
define_fn!(Cos, "cos", 1, |a| a.cos());
define_fn!(Tan, "tan", 1, |a| a.tan());
define_fn!(Asin, "asin", 1, |a| a.asin());
define_fn!(Acos, "acos", 1, |a| a.acos());
define_fn!(Atan, "atan", 1, |a| a.atan());
define_fn!(Atan2, "atan2", 2, |y, x| y.atan2(*x));
define_fn!(Sinh, "sinh", 1, |a| a.sinh());
define_fn!(Cosh, "cosh", 1, |a| a.cosh());
define_fn!(Tanh, "tanh", 1, |a| a.tanh());

// ============================================================================
// Exponential / logarithmic
// ============================================================================

define_fn!(Exp, "exp", 1, |a| a.exp());
define_fn!(Exp2, "exp2", 1, |a| a.exp2());
define_fn!(Log, "log", 1, |a| a.ln());
define_fn!(Ln, "ln", 1, |a| a.ln());
define_fn!(Log2, "log2", 1, |a| a.log2());
define_fn!(Log10, "log10", 1, |a| a.log10());
define_fn!(Pow, "pow", 2, |a, b| a.powf(*b));
define_fn!(Sqrt, "sqrt", 1, |a| a.sqrt());
define_fn!(InverseSqrt, "inversesqrt", 1, |a| 1.0 / a.sqrt());

// ============================================================================
// Common math functions
// ============================================================================

define_fn!(Abs, "abs", 1, |a| a.abs());
/// sign: -1 for negative, 0 for zero, 1 for positive (matches WGSL/GLSL)
pub struct Sign;
impl ExprFn for Sign {
    fn name(&self) -> &str {
        "sign"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[f32]) -> f32 {
        let x = args[0];
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
}
define_fn!(Floor, "floor", 1, |a| a.floor());
define_fn!(Ceil, "ceil", 1, |a| a.ceil());
define_fn!(Round, "round", 1, |a| a.round());
define_fn!(Trunc, "trunc", 1, |a| a.trunc());
define_fn!(Fract, "fract", 1, |a| a.fract());
define_fn!(Min, "min", 2, |a, b| a.min(*b));
define_fn!(Max, "max", 2, |a, b| a.max(*b));
define_fn!(Clamp, "clamp", 3, |x, lo, hi| x.clamp(*lo, *hi));
define_fn!(Saturate, "saturate", 1, |a| a.clamp(0.0, 1.0));

// ============================================================================
// Interpolation
// ============================================================================

/// Linear interpolation: lerp(a, b, t) = a + (b - a) * t
pub struct Lerp;
impl ExprFn for Lerp {
    fn name(&self) -> &str {
        "lerp"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[f32]) -> f32 {
        let [a, b, t] = args else { return 0.0 };
        a + (b - a) * t
    }
}

/// Alias for lerp (GLSL naming)
pub struct Mix;
impl ExprFn for Mix {
    fn name(&self) -> &str {
        "mix"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[f32]) -> f32 {
        let [a, b, t] = args else { return 0.0 };
        a + (b - a) * t
    }
}

/// Step function: step(edge, x) = x < edge ? 0.0 : 1.0
pub struct Step;
impl ExprFn for Step {
    fn name(&self) -> &str {
        "step"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[f32]) -> f32 {
        let [edge, x] = args else { return 0.0 };
        if x < edge { 0.0 } else { 1.0 }
    }
}

/// Smooth Hermite interpolation
pub struct Smoothstep;
impl ExprFn for Smoothstep {
    fn name(&self) -> &str {
        "smoothstep"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[f32]) -> f32 {
        let [edge0, edge1, x] = args else { return 0.0 };
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}

// ============================================================================
// Decomposition-based functions (work on all backends automatically)
// ============================================================================

/// Inverse lerp: inverse_lerp(a, b, v) = (v - a) / (b - a)
pub struct InverseLerp;
impl ExprFn for InverseLerp {
    fn name(&self) -> &str {
        "inverse_lerp"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[f32]) -> f32 {
        let [a, b, v] = args else { return 0.0 };
        (v - a) / (b - a)
    }
    fn decompose(&self, args: &[Ast]) -> Option<Ast> {
        if args.len() != 3 {
            return None;
        }
        let a = &args[0];
        let b = &args[1];
        let v = &args[2];
        // (v - a) / (b - a)
        Some(Ast::BinOp(
            BinOp::Div,
            Box::new(Ast::BinOp(
                BinOp::Sub,
                Box::new(v.clone()),
                Box::new(a.clone()),
            )),
            Box::new(Ast::BinOp(
                BinOp::Sub,
                Box::new(b.clone()),
                Box::new(a.clone()),
            )),
        ))
    }
}

/// Remap: remap(x, in_lo, in_hi, out_lo, out_hi)
pub struct Remap;
impl ExprFn for Remap {
    fn name(&self) -> &str {
        "remap"
    }
    fn arg_count(&self) -> usize {
        5
    }
    fn call(&self, args: &[f32]) -> f32 {
        let [x, in_lo, in_hi, out_lo, out_hi] = args else {
            return 0.0;
        };
        let t = (x - in_lo) / (in_hi - in_lo);
        out_lo + (out_hi - out_lo) * t
    }
}

// ============================================================================
// Registry
// ============================================================================

/// Registers all standard functions into the given registry.
pub fn register_std(registry: &mut FunctionRegistry) {
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

/// Creates a new registry with all standard functions.
pub fn std_registry() -> FunctionRegistry {
    let mut registry = FunctionRegistry::new();
    register_std(&mut registry);
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_sap_core::Expr;
    use std::collections::HashMap;

    fn eval(expr: &str, vars: &[(&str, f32)]) -> f32 {
        let registry = std_registry();
        let expr = Expr::parse(expr).unwrap();
        let var_map: HashMap<String, f32> = vars.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        expr.eval(&var_map, &registry).unwrap()
    }

    #[test]
    fn test_constants() {
        assert!((eval("pi()", &[]) - std::f32::consts::PI).abs() < 0.001);
        assert!((eval("e()", &[]) - std::f32::consts::E).abs() < 0.001);
        assert!((eval("tau()", &[]) - std::f32::consts::TAU).abs() < 0.001);
    }

    #[test]
    fn test_trig() {
        assert!(eval("sin(0)", &[]).abs() < 0.001);
        assert!((eval("cos(0)", &[]) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_exp_log() {
        assert!((eval("exp(0)", &[]) - 1.0).abs() < 0.001);
        assert!((eval("ln(1)", &[]) - 0.0).abs() < 0.001);
        assert!((eval("sqrt(16)", &[]) - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_common() {
        assert_eq!(eval("abs(-5)", &[]), 5.0);
        assert_eq!(eval("floor(3.7)", &[]), 3.0);
        assert_eq!(eval("ceil(3.2)", &[]), 4.0);
        assert_eq!(eval("min(3, 7)", &[]), 3.0);
        assert_eq!(eval("max(3, 7)", &[]), 7.0);
        assert_eq!(eval("clamp(5, 0, 3)", &[]), 3.0);
        assert_eq!(eval("saturate(1.5)", &[]), 1.0);
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(eval("lerp(0, 10, 0.5)", &[]), 5.0);
        assert_eq!(eval("mix(0, 10, 0.5)", &[]), 5.0);
        assert_eq!(eval("step(0.5, 0.3)", &[]), 0.0);
        assert_eq!(eval("step(0.5, 0.7)", &[]), 1.0);
        assert!((eval("smoothstep(0, 1, 0.5)", &[]) - 0.5).abs() < 0.1);
        assert_eq!(eval("inverse_lerp(0, 10, 5)", &[]), 0.5);
    }

    #[test]
    fn test_remap() {
        assert_eq!(eval("remap(5, 0, 10, 0, 100)", &[]), 50.0);
    }

    #[test]
    fn test_with_variables() {
        // sin(x * pi()) where x = 0.5 should be sin(pi/2) = 1
        let v = eval("sin(x * pi())", &[("x", 0.5)]);
        assert!((v - 1.0).abs() < 0.001);
    }
}
