//! Exhaustive tests for all functions across all backends.
//!
//! This module systematically tests every function and operation
//! in the standard library across interpreter, Lua, and Cranelift backends.

use crate::{cranelift_std_registry, lua_std_registry, std_registry};
use sap_core::Expr;
use sap_cranelift::JitCompiler;
use sap_lua::eval_with_registry;
use std::collections::HashMap;

const EPSILON: f32 = 0.0001;

fn assert_close(actual: f32, expected: f32, context: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < EPSILON || (expected.abs() > 1.0 && diff / expected.abs() < EPSILON),
        "{}: expected {}, got {} (diff: {})",
        context,
        expected,
        actual,
        diff
    );
}

// ============================================================================
// Backend evaluators
// ============================================================================

fn eval_interp(expr_str: &str, params: &[&str], args: &[f32]) -> f32 {
    let expr = Expr::parse(expr_str).unwrap();
    let vars: HashMap<String, f32> = params
        .iter()
        .zip(args.iter())
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    let registry = std_registry();
    expr.eval(&vars, &registry).unwrap()
}

fn eval_lua(expr_str: &str, params: &[&str], args: &[f32]) -> f32 {
    let expr = Expr::parse(expr_str).unwrap();
    let vars: HashMap<String, f32> = params
        .iter()
        .zip(args.iter())
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    let registry = lua_std_registry();
    eval_with_registry(expr.ast(), &vars, &registry).unwrap()
}

fn eval_cranelift(expr_str: &str, params: &[&str], args: &[f32]) -> f32 {
    let expr = Expr::parse(expr_str).unwrap();
    let registry = cranelift_std_registry();
    let jit = JitCompiler::new().unwrap();
    let func = jit.compile(expr.ast(), params, &registry).unwrap();
    func.call(args)
}

// ============================================================================
// Test case definition
// ============================================================================

struct TestCase {
    expr: &'static str,
    params: &'static [&'static str],
    args: &'static [f32],
    expected: f32,
}

impl TestCase {
    const fn new(
        expr: &'static str,
        params: &'static [&'static str],
        args: &'static [f32],
        expected: f32,
    ) -> Self {
        Self {
            expr,
            params,
            args,
            expected,
        }
    }

    fn run_all(&self) {
        let ctx = format!("expr='{}', args={:?}", self.expr, self.args);

        let interp = eval_interp(self.expr, self.params, self.args);
        assert_close(interp, self.expected, &format!("{} [interp]", ctx));

        let lua = eval_lua(self.expr, self.params, self.args);
        assert_close(lua, self.expected, &format!("{} [lua]", ctx));

        let cranelift = eval_cranelift(self.expr, self.params, self.args);
        assert_close(cranelift, self.expected, &format!("{} [cranelift]", ctx));

        // Also verify parity
        assert_close(interp, lua, &format!("{} [interp vs lua]", ctx));
        assert_close(interp, cranelift, &format!("{} [interp vs cranelift]", ctx));
    }
}

// ============================================================================
// Operations
// ============================================================================

#[test]
fn test_operations_add() {
    let cases = [
        TestCase::new("x + y", &["x", "y"], &[3.0, 4.0], 7.0),
        TestCase::new("x + y", &["x", "y"], &[-3.0, 4.0], 1.0),
        TestCase::new("x + y", &["x", "y"], &[0.0, 0.0], 0.0),
        TestCase::new("x + y", &["x", "y"], &[1.5, 2.5], 4.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_operations_sub() {
    let cases = [
        TestCase::new("x - y", &["x", "y"], &[10.0, 3.0], 7.0),
        TestCase::new("x - y", &["x", "y"], &[3.0, 10.0], -7.0),
        TestCase::new("x - y", &["x", "y"], &[5.0, 5.0], 0.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_operations_mul() {
    let cases = [
        TestCase::new("x * y", &["x", "y"], &[3.0, 4.0], 12.0),
        TestCase::new("x * y", &["x", "y"], &[-3.0, 4.0], -12.0),
        TestCase::new("x * y", &["x", "y"], &[0.0, 100.0], 0.0),
        TestCase::new("x * y", &["x", "y"], &[2.5, 4.0], 10.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_operations_div() {
    let cases = [
        TestCase::new("x / y", &["x", "y"], &[12.0, 3.0], 4.0),
        TestCase::new("x / y", &["x", "y"], &[10.0, 4.0], 2.5),
        TestCase::new("x / y", &["x", "y"], &[-12.0, 3.0], -4.0),
        TestCase::new("x / y", &["x", "y"], &[1.0, 2.0], 0.5),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_operations_pow() {
    let cases = [
        TestCase::new("x ^ y", &["x", "y"], &[2.0, 3.0], 8.0),
        TestCase::new("x ^ y", &["x", "y"], &[3.0, 2.0], 9.0),
        TestCase::new("x ^ y", &["x", "y"], &[4.0, 0.5], 2.0),
        TestCase::new("x ^ y", &["x", "y"], &[2.0, 0.0], 1.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_operations_neg() {
    let cases = [
        TestCase::new("-x", &["x"], &[5.0], -5.0),
        TestCase::new("-x", &["x"], &[-5.0], 5.0),
        TestCase::new("-x", &["x"], &[0.0], 0.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_operations_precedence() {
    let cases = [
        TestCase::new("x + y * z", &["x", "y", "z"], &[1.0, 2.0, 3.0], 7.0),
        TestCase::new("(x + y) * z", &["x", "y", "z"], &[1.0, 2.0, 3.0], 9.0),
        TestCase::new("x * y + z", &["x", "y", "z"], &[2.0, 3.0, 4.0], 10.0),
        TestCase::new("x - y - z", &["x", "y", "z"], &[10.0, 3.0, 2.0], 5.0),
        TestCase::new("x / y / z", &["x", "y", "z"], &[24.0, 4.0, 2.0], 3.0),
        TestCase::new("x ^ y ^ z", &["x", "y", "z"], &[2.0, 2.0, 2.0], 16.0), // right-associative: 2^(2^2)=2^4=16
    ];
    for case in &cases {
        case.run_all();
    }
}

// ============================================================================
// Constants
// ============================================================================

#[test]
fn test_constants() {
    let cases = [
        TestCase::new("pi()", &[], &[], std::f32::consts::PI),
        TestCase::new("e()", &[], &[], std::f32::consts::E),
        TestCase::new("tau()", &[], &[], std::f32::consts::TAU),
    ];
    for case in &cases {
        case.run_all();
    }
}

// ============================================================================
// Trigonometric functions
// ============================================================================

#[test]
fn test_sin() {
    let cases = [
        TestCase::new("sin(x)", &["x"], &[0.0], 0.0),
        TestCase::new("sin(x)", &["x"], &[std::f32::consts::FRAC_PI_2], 1.0),
        TestCase::new("sin(x)", &["x"], &[std::f32::consts::PI], 0.0),
        TestCase::new("sin(x)", &["x"], &[std::f32::consts::FRAC_PI_6], 0.5),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_cos() {
    let cases = [
        TestCase::new("cos(x)", &["x"], &[0.0], 1.0),
        TestCase::new("cos(x)", &["x"], &[std::f32::consts::FRAC_PI_2], 0.0),
        TestCase::new("cos(x)", &["x"], &[std::f32::consts::PI], -1.0),
        TestCase::new("cos(x)", &["x"], &[std::f32::consts::FRAC_PI_3], 0.5),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_tan() {
    let cases = [
        TestCase::new("tan(x)", &["x"], &[0.0], 0.0),
        TestCase::new("tan(x)", &["x"], &[std::f32::consts::FRAC_PI_4], 1.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_asin() {
    let cases = [
        TestCase::new("asin(x)", &["x"], &[0.0], 0.0),
        TestCase::new("asin(x)", &["x"], &[1.0], std::f32::consts::FRAC_PI_2),
        TestCase::new("asin(x)", &["x"], &[0.5], std::f32::consts::FRAC_PI_6),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_acos() {
    let cases = [
        TestCase::new("acos(x)", &["x"], &[1.0], 0.0),
        TestCase::new("acos(x)", &["x"], &[0.0], std::f32::consts::FRAC_PI_2),
        TestCase::new("acos(x)", &["x"], &[0.5], std::f32::consts::FRAC_PI_3),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_atan() {
    let cases = [
        TestCase::new("atan(x)", &["x"], &[0.0], 0.0),
        TestCase::new("atan(x)", &["x"], &[1.0], std::f32::consts::FRAC_PI_4),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_atan2() {
    let cases = [
        TestCase::new("atan2(y, x)", &["y", "x"], &[0.0, 1.0], 0.0),
        TestCase::new(
            "atan2(y, x)",
            &["y", "x"],
            &[1.0, 1.0],
            std::f32::consts::FRAC_PI_4,
        ),
        TestCase::new(
            "atan2(y, x)",
            &["y", "x"],
            &[1.0, 0.0],
            std::f32::consts::FRAC_PI_2,
        ),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_sinh() {
    let cases = [
        TestCase::new("sinh(x)", &["x"], &[0.0], 0.0),
        TestCase::new("sinh(x)", &["x"], &[1.0], 1.0_f32.sinh()),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_cosh() {
    let cases = [
        TestCase::new("cosh(x)", &["x"], &[0.0], 1.0),
        TestCase::new("cosh(x)", &["x"], &[1.0], 1.0_f32.cosh()),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_tanh() {
    let cases = [
        TestCase::new("tanh(x)", &["x"], &[0.0], 0.0),
        TestCase::new("tanh(x)", &["x"], &[1.0], 1.0_f32.tanh()),
    ];
    for case in &cases {
        case.run_all();
    }
}

// ============================================================================
// Exponential / Logarithmic
// ============================================================================

#[test]
fn test_exp() {
    let cases = [
        TestCase::new("exp(x)", &["x"], &[0.0], 1.0),
        TestCase::new("exp(x)", &["x"], &[1.0], std::f32::consts::E),
        TestCase::new(
            "exp(x)",
            &["x"],
            &[2.0],
            std::f32::consts::E * std::f32::consts::E,
        ),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_exp2() {
    let cases = [
        TestCase::new("exp2(x)", &["x"], &[0.0], 1.0),
        TestCase::new("exp2(x)", &["x"], &[1.0], 2.0),
        TestCase::new("exp2(x)", &["x"], &[3.0], 8.0),
        TestCase::new("exp2(x)", &["x"], &[0.5], std::f32::consts::SQRT_2),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_log_ln() {
    let cases = [
        TestCase::new("log(x)", &["x"], &[1.0], 0.0),
        TestCase::new("log(x)", &["x"], &[std::f32::consts::E], 1.0),
        TestCase::new("ln(x)", &["x"], &[1.0], 0.0),
        TestCase::new("ln(x)", &["x"], &[std::f32::consts::E], 1.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_log2() {
    let cases = [
        TestCase::new("log2(x)", &["x"], &[1.0], 0.0),
        TestCase::new("log2(x)", &["x"], &[2.0], 1.0),
        TestCase::new("log2(x)", &["x"], &[8.0], 3.0),
        TestCase::new("log2(x)", &["x"], &[16.0], 4.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_log10() {
    let cases = [
        TestCase::new("log10(x)", &["x"], &[1.0], 0.0),
        TestCase::new("log10(x)", &["x"], &[10.0], 1.0),
        TestCase::new("log10(x)", &["x"], &[100.0], 2.0),
        TestCase::new("log10(x)", &["x"], &[1000.0], 3.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_pow_fn() {
    let cases = [
        TestCase::new("pow(x, y)", &["x", "y"], &[2.0, 3.0], 8.0),
        TestCase::new("pow(x, y)", &["x", "y"], &[3.0, 2.0], 9.0),
        TestCase::new("pow(x, y)", &["x", "y"], &[4.0, 0.5], 2.0),
        TestCase::new("pow(x, y)", &["x", "y"], &[2.0, 0.0], 1.0),
        TestCase::new("pow(x, y)", &["x", "y"], &[10.0, 1.0], 10.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_sqrt() {
    let cases = [
        TestCase::new("sqrt(x)", &["x"], &[0.0], 0.0),
        TestCase::new("sqrt(x)", &["x"], &[1.0], 1.0),
        TestCase::new("sqrt(x)", &["x"], &[4.0], 2.0),
        TestCase::new("sqrt(x)", &["x"], &[9.0], 3.0),
        TestCase::new("sqrt(x)", &["x"], &[16.0], 4.0),
        TestCase::new("sqrt(x)", &["x"], &[2.0], std::f32::consts::SQRT_2),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_inversesqrt() {
    let cases = [
        TestCase::new("inversesqrt(x)", &["x"], &[1.0], 1.0),
        TestCase::new("inversesqrt(x)", &["x"], &[4.0], 0.5),
        TestCase::new("inversesqrt(x)", &["x"], &[16.0], 0.25),
    ];
    for case in &cases {
        case.run_all();
    }
}

// ============================================================================
// Common math functions
// ============================================================================

#[test]
fn test_abs() {
    let cases = [
        TestCase::new("abs(x)", &["x"], &[5.0], 5.0),
        TestCase::new("abs(x)", &["x"], &[-5.0], 5.0),
        TestCase::new("abs(x)", &["x"], &[0.0], 0.0),
        TestCase::new("abs(x)", &["x"], &[-0.0], 0.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_sign() {
    let cases = [
        TestCase::new("sign(x)", &["x"], &[5.0], 1.0),
        TestCase::new("sign(x)", &["x"], &[-5.0], -1.0),
        TestCase::new("sign(x)", &["x"], &[0.0], 0.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_floor() {
    let cases = [
        TestCase::new("floor(x)", &["x"], &[3.7], 3.0),
        TestCase::new("floor(x)", &["x"], &[3.0], 3.0),
        TestCase::new("floor(x)", &["x"], &[-3.7], -4.0),
        TestCase::new("floor(x)", &["x"], &[-3.0], -3.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_ceil() {
    let cases = [
        TestCase::new("ceil(x)", &["x"], &[3.2], 4.0),
        TestCase::new("ceil(x)", &["x"], &[3.0], 3.0),
        TestCase::new("ceil(x)", &["x"], &[-3.2], -3.0),
        TestCase::new("ceil(x)", &["x"], &[-3.0], -3.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_round() {
    let cases = [
        TestCase::new("round(x)", &["x"], &[3.4], 3.0),
        TestCase::new("round(x)", &["x"], &[3.5], 4.0),
        TestCase::new("round(x)", &["x"], &[3.6], 4.0),
        TestCase::new("round(x)", &["x"], &[-3.4], -3.0),
        TestCase::new("round(x)", &["x"], &[-3.5], -4.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_trunc() {
    let cases = [
        TestCase::new("trunc(x)", &["x"], &[3.7], 3.0),
        TestCase::new("trunc(x)", &["x"], &[3.0], 3.0),
        TestCase::new("trunc(x)", &["x"], &[-3.7], -3.0),
        TestCase::new("trunc(x)", &["x"], &[-3.0], -3.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_fract() {
    let cases = [
        TestCase::new("fract(x)", &["x"], &[3.7], 0.7),
        TestCase::new("fract(x)", &["x"], &[3.0], 0.0),
        TestCase::new("fract(x)", &["x"], &[0.5], 0.5),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_min() {
    let cases = [
        TestCase::new("min(x, y)", &["x", "y"], &[3.0, 7.0], 3.0),
        TestCase::new("min(x, y)", &["x", "y"], &[7.0, 3.0], 3.0),
        TestCase::new("min(x, y)", &["x", "y"], &[5.0, 5.0], 5.0),
        TestCase::new("min(x, y)", &["x", "y"], &[-3.0, 3.0], -3.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_max() {
    let cases = [
        TestCase::new("max(x, y)", &["x", "y"], &[3.0, 7.0], 7.0),
        TestCase::new("max(x, y)", &["x", "y"], &[7.0, 3.0], 7.0),
        TestCase::new("max(x, y)", &["x", "y"], &[5.0, 5.0], 5.0),
        TestCase::new("max(x, y)", &["x", "y"], &[-3.0, 3.0], 3.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_clamp() {
    let cases = [
        TestCase::new(
            "clamp(x, lo, hi)",
            &["x", "lo", "hi"],
            &[5.0, 0.0, 10.0],
            5.0,
        ),
        TestCase::new(
            "clamp(x, lo, hi)",
            &["x", "lo", "hi"],
            &[-5.0, 0.0, 10.0],
            0.0,
        ),
        TestCase::new(
            "clamp(x, lo, hi)",
            &["x", "lo", "hi"],
            &[15.0, 0.0, 10.0],
            10.0,
        ),
        TestCase::new(
            "clamp(x, lo, hi)",
            &["x", "lo", "hi"],
            &[0.0, 0.0, 10.0],
            0.0,
        ),
        TestCase::new(
            "clamp(x, lo, hi)",
            &["x", "lo", "hi"],
            &[10.0, 0.0, 10.0],
            10.0,
        ),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_saturate() {
    let cases = [
        TestCase::new("saturate(x)", &["x"], &[0.5], 0.5),
        TestCase::new("saturate(x)", &["x"], &[0.0], 0.0),
        TestCase::new("saturate(x)", &["x"], &[1.0], 1.0),
        TestCase::new("saturate(x)", &["x"], &[-0.5], 0.0),
        TestCase::new("saturate(x)", &["x"], &[1.5], 1.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

// ============================================================================
// Interpolation
// ============================================================================

#[test]
fn test_lerp() {
    let cases = [
        TestCase::new("lerp(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 0.0], 0.0),
        TestCase::new("lerp(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 0.5], 5.0),
        TestCase::new("lerp(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 1.0], 10.0),
        TestCase::new("lerp(a, b, t)", &["a", "b", "t"], &[10.0, 20.0, 0.25], 12.5),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_mix() {
    let cases = [
        TestCase::new("mix(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 0.0], 0.0),
        TestCase::new("mix(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 0.5], 5.0),
        TestCase::new("mix(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 1.0], 10.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_step() {
    let cases = [
        TestCase::new("step(edge, x)", &["edge", "x"], &[0.5, 0.3], 0.0),
        TestCase::new("step(edge, x)", &["edge", "x"], &[0.5, 0.5], 1.0),
        TestCase::new("step(edge, x)", &["edge", "x"], &[0.5, 0.7], 1.0),
        TestCase::new("step(edge, x)", &["edge", "x"], &[0.0, 0.0], 1.0),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_smoothstep() {
    let cases = [
        TestCase::new(
            "smoothstep(e0, e1, x)",
            &["e0", "e1", "x"],
            &[0.0, 1.0, 0.0],
            0.0,
        ),
        TestCase::new(
            "smoothstep(e0, e1, x)",
            &["e0", "e1", "x"],
            &[0.0, 1.0, 1.0],
            1.0,
        ),
        TestCase::new(
            "smoothstep(e0, e1, x)",
            &["e0", "e1", "x"],
            &[0.0, 1.0, 0.5],
            0.5,
        ),
        TestCase::new(
            "smoothstep(e0, e1, x)",
            &["e0", "e1", "x"],
            &[0.0, 1.0, -1.0],
            0.0,
        ),
        TestCase::new(
            "smoothstep(e0, e1, x)",
            &["e0", "e1", "x"],
            &[0.0, 1.0, 2.0],
            1.0,
        ),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_inverse_lerp() {
    let cases = [
        TestCase::new(
            "inverse_lerp(a, b, v)",
            &["a", "b", "v"],
            &[0.0, 10.0, 0.0],
            0.0,
        ),
        TestCase::new(
            "inverse_lerp(a, b, v)",
            &["a", "b", "v"],
            &[0.0, 10.0, 5.0],
            0.5,
        ),
        TestCase::new(
            "inverse_lerp(a, b, v)",
            &["a", "b", "v"],
            &[0.0, 10.0, 10.0],
            1.0,
        ),
        TestCase::new(
            "inverse_lerp(a, b, v)",
            &["a", "b", "v"],
            &[10.0, 20.0, 12.5],
            0.25,
        ),
    ];
    for case in &cases {
        case.run_all();
    }
}

#[test]
fn test_remap() {
    let cases = [
        TestCase::new(
            "remap(x, in_lo, in_hi, out_lo, out_hi)",
            &["x", "in_lo", "in_hi", "out_lo", "out_hi"],
            &[5.0, 0.0, 10.0, 0.0, 100.0],
            50.0,
        ),
        TestCase::new(
            "remap(x, in_lo, in_hi, out_lo, out_hi)",
            &["x", "in_lo", "in_hi", "out_lo", "out_hi"],
            &[0.0, 0.0, 10.0, 100.0, 200.0],
            100.0,
        ),
        TestCase::new(
            "remap(x, in_lo, in_hi, out_lo, out_hi)",
            &["x", "in_lo", "in_hi", "out_lo", "out_hi"],
            &[10.0, 0.0, 10.0, 100.0, 200.0],
            200.0,
        ),
    ];
    for case in &cases {
        case.run_all();
    }
}

// ============================================================================
// Complex expressions (combined operations)
// ============================================================================

#[test]
fn test_complex_expressions() {
    let cases = [
        TestCase::new("sin(x) + cos(x)", &["x"], &[0.0], 1.0),
        TestCase::new("sqrt(x * x + y * y)", &["x", "y"], &[3.0, 4.0], 5.0),
        TestCase::new(
            "lerp(a, b, saturate(t))",
            &["a", "b", "t"],
            &[0.0, 100.0, 1.5],
            100.0,
        ),
        TestCase::new("abs(sin(x))", &["x"], &[std::f32::consts::PI], 0.0),
        TestCase::new("floor(x) + fract(x)", &["x"], &[3.7], 3.7),
        TestCase::new(
            "min(max(x, lo), hi)",
            &["x", "lo", "hi"],
            &[5.0, 0.0, 3.0],
            3.0,
        ),
        TestCase::new("exp(ln(x))", &["x"], &[5.0], 5.0),
        TestCase::new("log2(exp2(x))", &["x"], &[3.0], 3.0),
    ];
    for case in &cases {
        case.run_all();
    }
}
