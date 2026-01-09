//! Cranelift backend implementations for standard library functions.
//!
//! This module provides all standard library functions for the Cranelift backend:
//! - Constants (pi, e, tau)
//! - Transcendental functions (sin, cos, exp, log, sqrt, etc.) via Rust callbacks
//! - Basic comparisons (min, max, clamp, etc.)
//! - Interpolation (lerp, mix, smoothstep, etc.)

use sap_cranelift::{
    CraneliftFn, CraneliftRegistry, FloatCC, FunctionBuilder, InstBuilder, MathFuncs, Value,
};

// ============================================================================
// Macro helpers
// ============================================================================

macro_rules! cranelift_const {
    ($name:ident, $str_name:literal, $value:expr) => {
        pub struct $name;

        impl CraneliftFn for $name {
            fn name(&self) -> &str {
                $str_name
            }
            fn emit(
                &self,
                builder: &mut FunctionBuilder,
                _args: &[Value],
                _math: &MathFuncs,
            ) -> Value {
                builder.ins().f32const($value)
            }
        }
    };
}

/// Macro for transcendental functions that call into Rust
macro_rules! cranelift_math1 {
    ($name:ident, $str_name:literal, $sym:literal) => {
        pub struct $name;

        impl CraneliftFn for $name {
            fn name(&self) -> &str {
                $str_name
            }
            fn emit(
                &self,
                builder: &mut FunctionBuilder,
                args: &[Value],
                math: &MathFuncs,
            ) -> Value {
                let func_ref = math.get($sym).expect(concat!($sym, " not available"));
                let call = builder.ins().call(func_ref, &[args[0]]);
                builder.inst_results(call)[0]
            }
        }
    };
}

macro_rules! cranelift_math2 {
    ($name:ident, $str_name:literal, $sym:literal) => {
        pub struct $name;

        impl CraneliftFn for $name {
            fn name(&self) -> &str {
                $str_name
            }
            fn emit(
                &self,
                builder: &mut FunctionBuilder,
                args: &[Value],
                math: &MathFuncs,
            ) -> Value {
                let func_ref = math.get($sym).expect(concat!($sym, " not available"));
                let call = builder.ins().call(func_ref, &[args[0], args[1]]);
                builder.inst_results(call)[0]
            }
        }
    };
}

// ============================================================================
// Constants
// ============================================================================

cranelift_const!(Pi, "pi", std::f32::consts::PI);
cranelift_const!(E, "e", std::f32::consts::E);
cranelift_const!(Tau, "tau", std::f32::consts::TAU);

// ============================================================================
// Trigonometric functions (via Rust callbacks)
// ============================================================================

cranelift_math1!(Sin, "sin", "sap_sin");
cranelift_math1!(Cos, "cos", "sap_cos");
cranelift_math1!(Tan, "tan", "sap_tan");
cranelift_math1!(Asin, "asin", "sap_asin");
cranelift_math1!(Acos, "acos", "sap_acos");
cranelift_math1!(Atan, "atan", "sap_atan");
cranelift_math2!(Atan2, "atan2", "sap_atan2");
cranelift_math1!(Sinh, "sinh", "sap_sinh");
cranelift_math1!(Cosh, "cosh", "sap_cosh");
cranelift_math1!(Tanh, "tanh", "sap_tanh");

// ============================================================================
// Exponential / logarithmic (via Rust callbacks)
// ============================================================================

cranelift_math1!(Exp, "exp", "sap_exp");
cranelift_math1!(Exp2, "exp2", "sap_exp2");
cranelift_math1!(Log, "log", "sap_ln");
cranelift_math1!(Ln, "ln", "sap_ln");
cranelift_math1!(Log2, "log2", "sap_log2");
cranelift_math1!(Log10, "log10", "sap_log10");
cranelift_math2!(Pow, "pow", "sap_pow");
cranelift_math1!(Sqrt, "sqrt", "sap_sqrt");
cranelift_math1!(InverseSqrt, "inversesqrt", "sap_inversesqrt");

// ============================================================================
// Common math functions (native Cranelift IR)
// ============================================================================

/// abs: use fneg + select based on sign
pub struct Abs;
impl CraneliftFn for Abs {
    fn name(&self) -> &str {
        "abs"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        let x = args[0];
        builder.ins().fabs(x)
    }
}

/// sign: -1, 0, or 1 based on value
pub struct Sign;
impl CraneliftFn for Sign {
    fn name(&self) -> &str {
        "sign"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        let x = args[0];
        let zero = builder.ins().f32const(0.0);
        let one = builder.ins().f32const(1.0);
        let neg_one = builder.ins().f32const(-1.0);

        // x > 0 ? 1 : (x < 0 ? -1 : 0)
        let gt_zero = builder.ins().fcmp(FloatCC::GreaterThan, x, zero);
        let lt_zero = builder.ins().fcmp(FloatCC::LessThan, x, zero);

        let neg_or_zero = builder.ins().select(lt_zero, neg_one, zero);
        builder.ins().select(gt_zero, one, neg_or_zero)
    }
}

/// floor
pub struct Floor;
impl CraneliftFn for Floor {
    fn name(&self) -> &str {
        "floor"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        builder.ins().floor(args[0])
    }
}

/// ceil
pub struct Ceil;
impl CraneliftFn for Ceil {
    fn name(&self) -> &str {
        "ceil"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        builder.ins().ceil(args[0])
    }
}

/// round (nearest)
pub struct Round;
impl CraneliftFn for Round {
    fn name(&self) -> &str {
        "round"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        builder.ins().nearest(args[0])
    }
}

/// trunc
pub struct Trunc;
impl CraneliftFn for Trunc {
    fn name(&self) -> &str {
        "trunc"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        builder.ins().trunc(args[0])
    }
}

/// fract: x - floor(x)
pub struct Fract;
impl CraneliftFn for Fract {
    fn name(&self) -> &str {
        "fract"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        let x = args[0];
        let floor_x = builder.ins().floor(x);
        builder.ins().fsub(x, floor_x)
    }
}

/// min
pub struct Min;
impl CraneliftFn for Min {
    fn name(&self) -> &str {
        "min"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        builder.ins().fmin(args[0], args[1])
    }
}

/// max
pub struct Max;
impl CraneliftFn for Max {
    fn name(&self) -> &str {
        "max"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        builder.ins().fmax(args[0], args[1])
    }
}

/// clamp(x, lo, hi): max(lo, min(hi, x))
pub struct Clamp;
impl CraneliftFn for Clamp {
    fn name(&self) -> &str {
        "clamp"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        let x = args[0];
        let lo = args[1];
        let hi = args[2];
        let min_val = builder.ins().fmin(hi, x);
        builder.ins().fmax(lo, min_val)
    }
}

/// saturate: clamp(x, 0, 1)
pub struct Saturate;
impl CraneliftFn for Saturate {
    fn name(&self) -> &str {
        "saturate"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        let x = args[0];
        let zero = builder.ins().f32const(0.0);
        let one = builder.ins().f32const(1.0);
        let min_val = builder.ins().fmin(one, x);
        builder.ins().fmax(zero, min_val)
    }
}

// ============================================================================
// Interpolation (implementable without transcendentals)
// ============================================================================

/// lerp(a, b, t): a + (b - a) * t
pub struct Lerp;
impl CraneliftFn for Lerp {
    fn name(&self) -> &str {
        "lerp"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        let a = args[0];
        let b = args[1];
        let t = args[2];
        let diff = builder.ins().fsub(b, a);
        let scaled = builder.ins().fmul(diff, t);
        builder.ins().fadd(a, scaled)
    }
}

/// mix: alias for lerp
pub struct Mix;
impl CraneliftFn for Mix {
    fn name(&self) -> &str {
        "mix"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], math: &MathFuncs) -> Value {
        Lerp.emit(builder, args, math)
    }
}

/// step(edge, x): x < edge ? 0 : 1
pub struct Step;
impl CraneliftFn for Step {
    fn name(&self) -> &str {
        "step"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        let edge = args[0];
        let x = args[1];
        let zero = builder.ins().f32const(0.0);
        let one = builder.ins().f32const(1.0);
        let cmp = builder.ins().fcmp(FloatCC::LessThan, x, edge);
        builder.ins().select(cmp, zero, one)
    }
}

/// smoothstep(edge0, edge1, x): Hermite interpolation
pub struct Smoothstep;
impl CraneliftFn for Smoothstep {
    fn name(&self) -> &str {
        "smoothstep"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        let edge0 = args[0];
        let edge1 = args[1];
        let x = args[2];

        // t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
        let zero = builder.ins().f32const(0.0);
        let one = builder.ins().f32const(1.0);
        let two = builder.ins().f32const(2.0);
        let three = builder.ins().f32const(3.0);

        let numer = builder.ins().fsub(x, edge0);
        let denom = builder.ins().fsub(edge1, edge0);
        let t_raw = builder.ins().fdiv(numer, denom);
        let t_min = builder.ins().fmin(one, t_raw);
        let t = builder.ins().fmax(zero, t_min);

        // t * t * (3 - 2 * t)
        let t2 = builder.ins().fmul(t, t);
        let two_t = builder.ins().fmul(two, t);
        let three_minus = builder.ins().fsub(three, two_t);
        builder.ins().fmul(t2, three_minus)
    }
}

/// inverse_lerp(a, b, v): (v - a) / (b - a)
pub struct InverseLerp;
impl CraneliftFn for InverseLerp {
    fn name(&self) -> &str {
        "inverse_lerp"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        let a = args[0];
        let b = args[1];
        let v = args[2];
        let numer = builder.ins().fsub(v, a);
        let denom = builder.ins().fsub(b, a);
        builder.ins().fdiv(numer, denom)
    }
}

/// remap(x, in_lo, in_hi, out_lo, out_hi)
pub struct Remap;
impl CraneliftFn for Remap {
    fn name(&self) -> &str {
        "remap"
    }
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], _math: &MathFuncs) -> Value {
        let x = args[0];
        let in_lo = args[1];
        let in_hi = args[2];
        let out_lo = args[3];
        let out_hi = args[4];

        // t = (x - in_lo) / (in_hi - in_lo)
        let numer = builder.ins().fsub(x, in_lo);
        let denom = builder.ins().fsub(in_hi, in_lo);
        let t = builder.ins().fdiv(numer, denom);

        // out_lo + (out_hi - out_lo) * t
        let out_range = builder.ins().fsub(out_hi, out_lo);
        let scaled = builder.ins().fmul(out_range, t);
        builder.ins().fadd(out_lo, scaled)
    }
}

// ============================================================================
// Registry
// ============================================================================

/// Registers all standard functions for the Cranelift backend.
///
/// Transcendental functions (sin, cos, exp, log, pow, sqrt) are implemented
/// via callbacks to Rust's standard library math functions.
pub fn register_cranelift(registry: &mut CraneliftRegistry) {
    // Constants
    registry.register(Pi);
    registry.register(E);
    registry.register(Tau);

    // Trigonometric (via Rust callbacks)
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

    // Exponential / logarithmic (via Rust callbacks)
    registry.register(Exp);
    registry.register(Exp2);
    registry.register(Log);
    registry.register(Ln);
    registry.register(Log2);
    registry.register(Log10);
    registry.register(Pow);
    registry.register(Sqrt);
    registry.register(InverseSqrt);

    // Common math (native IR)
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

/// Creates a new registry with all standard Cranelift functions.
pub fn cranelift_std_registry() -> CraneliftRegistry {
    let mut registry = CraneliftRegistry::new();
    register_cranelift(&mut registry);
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use sap_core::Expr;
    use sap_cranelift::JitCompiler;

    fn eval(input: &str, params: &[&str], args: &[f32]) -> f32 {
        let expr = Expr::parse(input).unwrap();
        let registry = cranelift_std_registry();
        let jit = JitCompiler::new().unwrap();
        let func = jit.compile(expr.ast(), params, &registry).unwrap();
        func.call(args)
    }

    #[test]
    fn test_constants() {
        assert!((eval("pi()", &[], &[]) - std::f32::consts::PI).abs() < 0.001);
        assert!((eval("e()", &[], &[]) - std::f32::consts::E).abs() < 0.001);
        assert!((eval("tau()", &[], &[]) - std::f32::consts::TAU).abs() < 0.001);
    }

    #[test]
    fn test_abs() {
        assert_eq!(eval("abs(x)", &["x"], &[-5.0]), 5.0);
        assert_eq!(eval("abs(x)", &["x"], &[5.0]), 5.0);
    }

    #[test]
    fn test_sign() {
        assert_eq!(eval("sign(x)", &["x"], &[5.0]), 1.0);
        assert_eq!(eval("sign(x)", &["x"], &[-5.0]), -1.0);
        assert_eq!(eval("sign(x)", &["x"], &[0.0]), 0.0);
    }

    #[test]
    fn test_rounding() {
        assert_eq!(eval("floor(x)", &["x"], &[3.7]), 3.0);
        assert_eq!(eval("ceil(x)", &["x"], &[3.2]), 4.0);
        assert_eq!(eval("round(x)", &["x"], &[3.5]), 4.0);
        assert_eq!(eval("trunc(x)", &["x"], &[3.7]), 3.0);
        assert_eq!(eval("trunc(x)", &["x"], &[-3.7]), -3.0);
    }

    #[test]
    fn test_fract() {
        assert!((eval("fract(x)", &["x"], &[3.7]) - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_min_max() {
        assert_eq!(eval("min(x, y)", &["x", "y"], &[3.0, 7.0]), 3.0);
        assert_eq!(eval("max(x, y)", &["x", "y"], &[3.0, 7.0]), 7.0);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(
            eval("clamp(x, lo, hi)", &["x", "lo", "hi"], &[5.0, 0.0, 3.0]),
            3.0
        );
        assert_eq!(
            eval("clamp(x, lo, hi)", &["x", "lo", "hi"], &[-1.0, 0.0, 3.0]),
            0.0
        );
        assert_eq!(
            eval("clamp(x, lo, hi)", &["x", "lo", "hi"], &[1.5, 0.0, 3.0]),
            1.5
        );
    }

    #[test]
    fn test_saturate() {
        assert_eq!(eval("saturate(x)", &["x"], &[1.5]), 1.0);
        assert_eq!(eval("saturate(x)", &["x"], &[-0.5]), 0.0);
        assert_eq!(eval("saturate(x)", &["x"], &[0.5]), 0.5);
    }

    #[test]
    fn test_lerp() {
        assert_eq!(
            eval("lerp(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 0.5]),
            5.0
        );
        assert_eq!(
            eval("mix(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 0.5]),
            5.0
        );
    }

    #[test]
    fn test_step() {
        assert_eq!(eval("step(edge, x)", &["edge", "x"], &[0.5, 0.3]), 0.0);
        assert_eq!(eval("step(edge, x)", &["edge", "x"], &[0.5, 0.7]), 1.0);
    }

    #[test]
    fn test_smoothstep() {
        let result = eval(
            "smoothstep(e0, e1, x)",
            &["e0", "e1", "x"],
            &[0.0, 1.0, 0.5],
        );
        assert!((result - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_inverse_lerp() {
        assert_eq!(
            eval("inverse_lerp(a, b, v)", &["a", "b", "v"], &[0.0, 10.0, 5.0]),
            0.5
        );
    }

    #[test]
    fn test_remap() {
        let result = eval(
            "remap(x, in_lo, in_hi, out_lo, out_hi)",
            &["x", "in_lo", "in_hi", "out_lo", "out_hi"],
            &[5.0, 0.0, 10.0, 0.0, 100.0],
        );
        assert_eq!(result, 50.0);
    }

    #[test]
    fn test_trig() {
        assert!(eval("sin(0)", &[], &[]).abs() < 0.001);
        assert!((eval("cos(0)", &[], &[]) - 1.0).abs() < 0.001);
        assert!(eval("tan(0)", &[], &[]).abs() < 0.001);
        assert!((eval("atan2(1, 1)", &[], &[]) - std::f32::consts::FRAC_PI_4).abs() < 0.001);
    }

    #[test]
    fn test_hyperbolic() {
        assert!(eval("sinh(0)", &[], &[]).abs() < 0.001);
        assert!((eval("cosh(0)", &[], &[]) - 1.0).abs() < 0.001);
        assert!(eval("tanh(0)", &[], &[]).abs() < 0.001);
    }

    #[test]
    fn test_exp_log() {
        assert!((eval("exp(0)", &[], &[]) - 1.0).abs() < 0.001);
        assert!((eval("exp2(3)", &[], &[]) - 8.0).abs() < 0.001);
        assert!(eval("ln(1)", &[], &[]).abs() < 0.001);
        assert!((eval("log2(8)", &[], &[]) - 3.0).abs() < 0.001);
        assert!((eval("log10(100)", &[], &[]) - 2.0).abs() < 0.001);
        assert!((eval("pow(2, 3)", &[], &[]) - 8.0).abs() < 0.001);
        assert!((eval("sqrt(16)", &[], &[]) - 4.0).abs() < 0.001);
        assert!((eval("inversesqrt(4)", &[], &[]) - 0.5).abs() < 0.001);
    }
}
