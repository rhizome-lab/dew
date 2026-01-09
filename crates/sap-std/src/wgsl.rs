//! WGSL backend implementations for standard library functions.

use sap_wgsl::{WgslFn, WgslRegistry};

// ============================================================================
// Macro for simple WGSL functions
// ============================================================================

macro_rules! wgsl_fn {
    ($name:ident, $str_name:literal, $wgsl_name:literal) => {
        pub struct $name;

        impl WgslFn for $name {
            fn name(&self) -> &str {
                $str_name
            }
            fn emit(&self, args: &[String]) -> String {
                format!("{}({})", $wgsl_name, args.join(", "))
            }
        }
    };
}

macro_rules! wgsl_const {
    ($name:ident, $str_name:literal, $value:expr) => {
        pub struct $name;

        impl WgslFn for $name {
            fn name(&self) -> &str {
                $str_name
            }
            fn emit(&self, _args: &[String]) -> String {
                format!("{:.10}", $value)
            }
        }
    };
}

// ============================================================================
// Constants
// ============================================================================

wgsl_const!(Pi, "pi", std::f32::consts::PI);
wgsl_const!(E, "e", std::f32::consts::E);
wgsl_const!(Tau, "tau", std::f32::consts::TAU);

// ============================================================================
// Trigonometric functions
// ============================================================================

wgsl_fn!(Sin, "sin", "sin");
wgsl_fn!(Cos, "cos", "cos");
wgsl_fn!(Tan, "tan", "tan");
wgsl_fn!(Asin, "asin", "asin");
wgsl_fn!(Acos, "acos", "acos");
wgsl_fn!(Atan, "atan", "atan");
wgsl_fn!(Atan2, "atan2", "atan2");
wgsl_fn!(Sinh, "sinh", "sinh");
wgsl_fn!(Cosh, "cosh", "cosh");
wgsl_fn!(Tanh, "tanh", "tanh");

// ============================================================================
// Exponential / logarithmic
// ============================================================================

wgsl_fn!(Exp, "exp", "exp");
wgsl_fn!(Exp2, "exp2", "exp2");
wgsl_fn!(Log, "log", "log");
wgsl_fn!(Ln, "ln", "log"); // WGSL uses log for natural log
wgsl_fn!(Log2, "log2", "log2");

/// log10 in WGSL: log(x) / log(10)
pub struct Log10;
impl WgslFn for Log10 {
    fn name(&self) -> &str {
        "log10"
    }
    fn emit(&self, args: &[String]) -> String {
        format!(
            "(log({}) / log(10.0))",
            args.first().map(|s| s.as_str()).unwrap_or("0.0")
        )
    }
}

wgsl_fn!(Pow, "pow", "pow");
wgsl_fn!(Sqrt, "sqrt", "sqrt");
wgsl_fn!(InverseSqrt, "inversesqrt", "inverseSqrt");

// ============================================================================
// Common math functions
// ============================================================================

wgsl_fn!(Abs, "abs", "abs");
wgsl_fn!(Sign, "sign", "sign");
wgsl_fn!(Floor, "floor", "floor");
wgsl_fn!(Ceil, "ceil", "ceil");
wgsl_fn!(Round, "round", "round");
wgsl_fn!(Trunc, "trunc", "trunc");
wgsl_fn!(Fract, "fract", "fract");
wgsl_fn!(Min, "min", "min");
wgsl_fn!(Max, "max", "max");
wgsl_fn!(Clamp, "clamp", "clamp");
wgsl_fn!(Saturate, "saturate", "saturate");

// ============================================================================
// Interpolation
// ============================================================================

wgsl_fn!(Lerp, "lerp", "mix"); // WGSL uses mix
wgsl_fn!(Mix, "mix", "mix");
wgsl_fn!(Step, "step", "step");
wgsl_fn!(Smoothstep, "smoothstep", "smoothstep");

/// inverse_lerp: (v - a) / (b - a)
pub struct InverseLerp;
impl WgslFn for InverseLerp {
    fn name(&self) -> &str {
        "inverse_lerp"
    }
    fn emit(&self, args: &[String]) -> String {
        let a = args.first().map(|s| s.as_str()).unwrap_or("0.0");
        let b = args.get(1).map(|s| s.as_str()).unwrap_or("1.0");
        let v = args.get(2).map(|s| s.as_str()).unwrap_or("0.0");
        format!("(({v} - {a}) / ({b} - {a}))")
    }
}

/// remap: out_lo + (out_hi - out_lo) * ((x - in_lo) / (in_hi - in_lo))
pub struct Remap;
impl WgslFn for Remap {
    fn name(&self) -> &str {
        "remap"
    }
    fn emit(&self, args: &[String]) -> String {
        let x = args.first().map(|s| s.as_str()).unwrap_or("0.0");
        let in_lo = args.get(1).map(|s| s.as_str()).unwrap_or("0.0");
        let in_hi = args.get(2).map(|s| s.as_str()).unwrap_or("1.0");
        let out_lo = args.get(3).map(|s| s.as_str()).unwrap_or("0.0");
        let out_hi = args.get(4).map(|s| s.as_str()).unwrap_or("1.0");
        format!("({out_lo} + ({out_hi} - {out_lo}) * (({x} - {in_lo}) / ({in_hi} - {in_lo})))")
    }
}

// ============================================================================
// Registry
// ============================================================================

/// Registers all standard functions for WGSL backend.
pub fn register_wgsl(registry: &mut WgslRegistry) {
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

/// Creates a new registry with all standard WGSL functions.
pub fn wgsl_std_registry() -> WgslRegistry {
    let mut registry = WgslRegistry::new();
    register_wgsl(&mut registry);
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use sap_core::Expr;
    use sap_wgsl::to_wgsl;

    fn compile(input: &str) -> String {
        let expr = Expr::parse(input).unwrap();
        let registry = wgsl_std_registry();
        to_wgsl(expr.ast(), &registry)
    }

    #[test]
    fn test_constants() {
        assert!(compile("pi()").contains("3.14159"));
        assert!(compile("e()").contains("2.71828"));
        assert!(compile("tau()").contains("6.28318"));
    }

    #[test]
    fn test_trig() {
        assert_eq!(compile("sin(x)"), "sin(x)");
        assert_eq!(compile("cos(x)"), "cos(x)");
        assert_eq!(compile("atan2(y, x)"), "atan2(y, x)");
    }

    #[test]
    fn test_exp_log() {
        assert_eq!(compile("exp(x)"), "exp(x)");
        assert_eq!(compile("ln(x)"), "log(x)");
        assert_eq!(compile("log10(x)"), "(log(x) / log(10.0))");
        assert_eq!(compile("pow(x, 2)"), "pow(x, 2.0)");
        assert_eq!(compile("sqrt(x)"), "sqrt(x)");
        assert_eq!(compile("inversesqrt(x)"), "inverseSqrt(x)");
    }

    #[test]
    fn test_common() {
        assert_eq!(compile("abs(x)"), "abs(x)");
        assert_eq!(compile("floor(x)"), "floor(x)");
        assert_eq!(compile("clamp(x, 0, 1)"), "clamp(x, 0.0, 1.0)");
        assert_eq!(compile("saturate(x)"), "saturate(x)");
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(compile("lerp(a, b, t)"), "mix(a, b, t)");
        assert_eq!(compile("mix(a, b, t)"), "mix(a, b, t)");
        assert_eq!(compile("step(e, x)"), "step(e, x)");
        assert_eq!(compile("smoothstep(0, 1, x)"), "smoothstep(0.0, 1.0, x)");
    }

    #[test]
    fn test_inverse_lerp() {
        assert_eq!(
            compile("inverse_lerp(0, 10, x)"),
            "((x - 0.0) / (10.0 - 0.0))"
        );
    }

    #[test]
    fn test_remap() {
        let result = compile("remap(x, 0, 10, 0, 100)");
        assert!(result.contains("0.0 + (100.0 - 0.0)"));
    }
}
