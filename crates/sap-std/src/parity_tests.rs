//! Parity tests to ensure all backends produce consistent results.
//!
//! These tests verify that the interpreter, Lua, and Cranelift backends
//! all produce the same results for the same expressions and inputs.

use crate::{cranelift_std_registry, lua_std_registry, std_registry};
use sap_core::Expr;
use sap_cranelift::JitCompiler;
use sap_lua::eval_with_registry;
use std::collections::HashMap;

/// Maximum allowed difference between backend results
const EPSILON: f32 = 0.0001;

fn assert_close(a: f32, b: f32, context: &str) {
    let diff = (a - b).abs();
    assert!(
        diff < EPSILON,
        "{}: values differ by {}: {} vs {}",
        context,
        diff,
        a,
        b
    );
}

/// Evaluates an expression across all backends and verifies parity.
fn check_parity(expr_str: &str, params: &[&str], args: &[f32]) {
    let expr = Expr::parse(expr_str).expect("failed to parse expression");

    // Build variable map for interpreter and Lua
    let vars: HashMap<String, f32> = params
        .iter()
        .zip(args.iter())
        .map(|(k, v)| (k.to_string(), *v))
        .collect();

    // Interpreter result
    let interp_registry = std_registry();
    let interp_result = expr
        .eval(&vars, &interp_registry)
        .expect("interpreter eval failed");

    // Lua result
    let lua_registry = lua_std_registry();
    let lua_result = eval_with_registry(expr.ast(), &vars, &lua_registry).expect("lua eval failed");

    // Cranelift result
    let cranelift_registry = cranelift_std_registry();
    let jit = JitCompiler::new().expect("JIT creation failed");
    let compiled = jit
        .compile(expr.ast(), params, &cranelift_registry)
        .expect("cranelift compile failed");
    let cranelift_result = compiled.call(args);

    // Compare all results
    let context = format!("expr='{}', args={:?}", expr_str, args);
    assert_close(
        interp_result,
        lua_result,
        &format!("{} (interp vs lua)", context),
    );
    assert_close(
        interp_result,
        cranelift_result,
        &format!("{} (interp vs cranelift)", context),
    );
    assert_close(
        lua_result,
        cranelift_result,
        &format!("{} (lua vs cranelift)", context),
    );
}

#[test]
fn test_parity_constants() {
    check_parity("pi()", &[], &[]);
    check_parity("e()", &[], &[]);
    check_parity("tau()", &[], &[]);
}

#[test]
fn test_parity_trig() {
    check_parity("sin(x)", &["x"], &[0.0]);
    check_parity("sin(x)", &["x"], &[1.0]);
    check_parity("cos(x)", &["x"], &[0.0]);
    check_parity("cos(x)", &["x"], &[1.0]);
    check_parity("tan(x)", &["x"], &[0.0]);
    check_parity("atan2(y, x)", &["y", "x"], &[1.0, 1.0]);
}

#[test]
fn test_parity_hyperbolic() {
    check_parity("sinh(x)", &["x"], &[0.0]);
    check_parity("sinh(x)", &["x"], &[1.0]);
    check_parity("cosh(x)", &["x"], &[0.0]);
    check_parity("cosh(x)", &["x"], &[1.0]);
    check_parity("tanh(x)", &["x"], &[0.0]);
    check_parity("tanh(x)", &["x"], &[1.0]);
}

#[test]
fn test_parity_exp_log() {
    check_parity("exp(x)", &["x"], &[0.0]);
    check_parity("exp(x)", &["x"], &[1.0]);
    check_parity("exp2(x)", &["x"], &[3.0]);
    check_parity("ln(x)", &["x"], &[1.0]);
    check_parity("ln(x)", &["x"], &[std::f32::consts::E]);
    check_parity("log2(x)", &["x"], &[8.0]);
    check_parity("log10(x)", &["x"], &[100.0]);
    check_parity("pow(x, y)", &["x", "y"], &[2.0, 3.0]);
    check_parity("sqrt(x)", &["x"], &[16.0]);
    check_parity("inversesqrt(x)", &["x"], &[4.0]);
}

#[test]
fn test_parity_basic_math() {
    check_parity("abs(x)", &["x"], &[-5.0]);
    check_parity("abs(x)", &["x"], &[5.0]);
    check_parity("sign(x)", &["x"], &[-5.0]);
    check_parity("sign(x)", &["x"], &[5.0]);
    check_parity("sign(x)", &["x"], &[0.0]);
}

#[test]
fn test_parity_rounding() {
    check_parity("floor(x)", &["x"], &[3.7]);
    check_parity("floor(x)", &["x"], &[-3.7]);
    check_parity("ceil(x)", &["x"], &[3.2]);
    check_parity("ceil(x)", &["x"], &[-3.2]);
    check_parity("round(x)", &["x"], &[3.5]);
    check_parity("round(x)", &["x"], &[3.4]);
    check_parity("trunc(x)", &["x"], &[3.7]);
    check_parity("trunc(x)", &["x"], &[-3.7]);
    check_parity("fract(x)", &["x"], &[3.7]);
}

#[test]
fn test_parity_min_max() {
    check_parity("min(x, y)", &["x", "y"], &[3.0, 7.0]);
    check_parity("min(x, y)", &["x", "y"], &[7.0, 3.0]);
    check_parity("max(x, y)", &["x", "y"], &[3.0, 7.0]);
    check_parity("max(x, y)", &["x", "y"], &[7.0, 3.0]);
}

#[test]
fn test_parity_clamp() {
    check_parity("clamp(x, lo, hi)", &["x", "lo", "hi"], &[5.0, 0.0, 3.0]);
    check_parity("clamp(x, lo, hi)", &["x", "lo", "hi"], &[-1.0, 0.0, 3.0]);
    check_parity("clamp(x, lo, hi)", &["x", "lo", "hi"], &[1.5, 0.0, 3.0]);
    check_parity("saturate(x)", &["x"], &[1.5]);
    check_parity("saturate(x)", &["x"], &[-0.5]);
    check_parity("saturate(x)", &["x"], &[0.5]);
}

#[test]
fn test_parity_lerp() {
    check_parity("lerp(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 0.0]);
    check_parity("lerp(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 0.5]);
    check_parity("lerp(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 1.0]);
    check_parity("mix(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 0.25]);
}

#[test]
fn test_parity_step() {
    check_parity("step(edge, x)", &["edge", "x"], &[0.5, 0.3]);
    check_parity("step(edge, x)", &["edge", "x"], &[0.5, 0.7]);
    check_parity("step(edge, x)", &["edge", "x"], &[0.5, 0.5]);
}

#[test]
fn test_parity_smoothstep() {
    check_parity(
        "smoothstep(e0, e1, x)",
        &["e0", "e1", "x"],
        &[0.0, 1.0, 0.0],
    );
    check_parity(
        "smoothstep(e0, e1, x)",
        &["e0", "e1", "x"],
        &[0.0, 1.0, 0.5],
    );
    check_parity(
        "smoothstep(e0, e1, x)",
        &["e0", "e1", "x"],
        &[0.0, 1.0, 1.0],
    );
    check_parity(
        "smoothstep(e0, e1, x)",
        &["e0", "e1", "x"],
        &[0.0, 1.0, 0.25],
    );
}

#[test]
fn test_parity_inverse_lerp() {
    check_parity("inverse_lerp(a, b, v)", &["a", "b", "v"], &[0.0, 10.0, 0.0]);
    check_parity("inverse_lerp(a, b, v)", &["a", "b", "v"], &[0.0, 10.0, 5.0]);
    check_parity(
        "inverse_lerp(a, b, v)",
        &["a", "b", "v"],
        &[0.0, 10.0, 10.0],
    );
}

#[test]
fn test_parity_remap() {
    check_parity(
        "remap(x, in_lo, in_hi, out_lo, out_hi)",
        &["x", "in_lo", "in_hi", "out_lo", "out_hi"],
        &[5.0, 0.0, 10.0, 0.0, 100.0],
    );
    check_parity(
        "remap(x, in_lo, in_hi, out_lo, out_hi)",
        &["x", "in_lo", "in_hi", "out_lo", "out_hi"],
        &[0.0, 0.0, 10.0, 100.0, 200.0],
    );
}

#[test]
fn test_parity_complex_expressions() {
    // Combined operations
    check_parity("abs(x) + sign(y)", &["x", "y"], &[-3.0, -5.0]);
    check_parity("clamp(x * 2, lo, hi)", &["x", "lo", "hi"], &[0.8, 0.0, 1.0]);
    check_parity(
        "lerp(a, b, saturate(t))",
        &["a", "b", "t"],
        &[0.0, 100.0, 1.5],
    );
    check_parity("min(max(x, lo), hi)", &["x", "lo", "hi"], &[5.0, 0.0, 3.0]);
}

#[test]
fn test_parity_arithmetic() {
    check_parity("x + y", &["x", "y"], &[3.0, 4.0]);
    check_parity("x - y", &["x", "y"], &[10.0, 3.0]);
    check_parity("x * y", &["x", "y"], &[3.0, 4.0]);
    check_parity("x / y", &["x", "y"], &[10.0, 4.0]);
    check_parity("-x", &["x"], &[5.0]);
    check_parity("x + y * z", &["x", "y", "z"], &[1.0, 2.0, 3.0]);
    check_parity("(x + y) * z", &["x", "y", "z"], &[1.0, 2.0, 3.0]);
}
