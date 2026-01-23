//! Rust code generation for scalar expressions.
//!
//! Compiles expression ASTs to Rust source code for AOT compilation.

use rhizome_dew_cond::rust as cond;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};

// ============================================================================
// Errors
// ============================================================================

/// Rust emission error.
#[derive(Debug, Clone, PartialEq)]
pub enum RustError {
    /// Unknown function.
    UnknownFunction(String),
    /// Feature not supported in Rust codegen.
    UnsupportedFeature(String),
}

impl std::fmt::Display for RustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RustError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            RustError::UnsupportedFeature(feat) => {
                write!(f, "unsupported feature in Rust codegen: {feat}")
            }
        }
    }
}

impl std::error::Error for RustError {}

// ============================================================================
// Rust expression output
// ============================================================================

/// Result of emitting Rust code.
#[derive(Debug, Clone)]
pub struct RustExpr {
    /// The Rust expression string.
    pub code: String,
}

impl RustExpr {
    pub fn new(code: impl Into<String>) -> Self {
        Self { code: code.into() }
    }
}

// ============================================================================
// Function name mapping
// ============================================================================

/// Returns the Rust equivalent for a scalar function name.
fn rust_func_name(name: &str) -> Option<RustFunc> {
    Some(match name {
        // Constants
        "pi" => RustFunc::Const("std::f32::consts::PI"),
        "e" => RustFunc::Const("std::f32::consts::E"),
        "tau" => RustFunc::Const("std::f32::consts::TAU"),

        // Trig - method syntax
        "sin" => RustFunc::Method("sin"),
        "cos" => RustFunc::Method("cos"),
        "tan" => RustFunc::Method("tan"),
        "asin" => RustFunc::Method("asin"),
        "acos" => RustFunc::Method("acos"),
        "atan" => RustFunc::Method("atan"),
        "atan2" => RustFunc::Atan2,
        "sinh" => RustFunc::Method("sinh"),
        "cosh" => RustFunc::Method("cosh"),
        "tanh" => RustFunc::Method("tanh"),

        // Exp/log
        "exp" => RustFunc::Method("exp"),
        "exp2" => RustFunc::Method("exp2"),
        "log2" => RustFunc::Method("log2"),
        "pow" => RustFunc::Powf,
        "sqrt" => RustFunc::Method("sqrt"),
        "log" | "ln" => RustFunc::Method("ln"),
        "log10" => RustFunc::Method("log10"),
        "inversesqrt" => RustFunc::InverseSqrt,

        // Common math
        "abs" => RustFunc::Method("abs"),
        "sign" => RustFunc::Method("signum"),
        "floor" => RustFunc::Method("floor"),
        "ceil" => RustFunc::Method("ceil"),
        "round" => RustFunc::Method("round"),
        "trunc" => RustFunc::Method("trunc"),
        "fract" => RustFunc::Method("fract"),
        "min" => RustFunc::Method2("min"),
        "max" => RustFunc::Method2("max"),
        "clamp" => RustFunc::Clamp,
        "saturate" => RustFunc::Saturate,

        // Interpolation
        "lerp" | "mix" => RustFunc::Lerp,
        "step" => RustFunc::Step,
        "smoothstep" => RustFunc::Smoothstep,
        "inverse_lerp" => RustFunc::InverseLerp,
        "remap" => RustFunc::Remap,

        _ => return None,
    })
}

enum RustFunc {
    /// Constant path (e.g., std::f32::consts::PI)
    Const(&'static str),
    /// Method call on first argument: arg0.method()
    Method(&'static str),
    /// Method call with second argument: arg0.method(arg1)
    Method2(&'static str),
    /// atan2(y, x) -> y.atan2(x)
    Atan2,
    /// pow(x, y) -> x.powf(y)
    Powf,
    /// inversesqrt(x) -> 1.0 / x.sqrt()
    InverseSqrt,
    /// clamp(x, lo, hi) -> x.clamp(lo, hi)
    Clamp,
    /// saturate(x) -> x.clamp(0.0, 1.0)
    Saturate,
    /// lerp(a, b, t) -> a + (b - a) * t
    Lerp,
    /// step(edge, x) -> if x < edge { 0.0 } else { 1.0 }
    Step,
    /// smoothstep(e0, e1, x)
    Smoothstep,
    /// inverse_lerp(a, b, v) -> (v - a) / (b - a)
    InverseLerp,
    /// remap(x, in_lo, in_hi, out_lo, out_hi)
    Remap,
}

// ============================================================================
// Code generation
// ============================================================================

/// Result of emitting code: accumulated statements + final expression.
struct Emission {
    statements: Vec<String>,
    expr: String,
}

impl Emission {
    fn expr_only(expr: String) -> Self {
        Self {
            statements: vec![],
            expr,
        }
    }

    fn with_statements(statements: Vec<String>, expr: String) -> Self {
        Self { statements, expr }
    }
}

/// Emit Rust code for an AST.
pub fn emit_rust(ast: &Ast) -> Result<RustExpr, RustError> {
    let emission = emit_full(ast)?;
    Ok(RustExpr::new(emission.expr))
}

/// Generate a complete Rust function.
pub fn emit_rust_fn(name: &str, ast: &Ast, params: &[&str]) -> Result<String, RustError> {
    let param_list: String = params
        .iter()
        .map(|p| format!("{}: f32", p))
        .collect::<Vec<_>>()
        .join(", ");

    let emission = emit_full(ast)?;

    let mut body = String::new();
    for stmt in &emission.statements {
        body.push_str("    ");
        body.push_str(stmt);
        body.push('\n');
    }
    body.push_str("    ");
    body.push_str(&emission.expr);

    Ok(format!(
        "fn {}({}) -> f32 {{\n{}\n}}",
        name, param_list, body
    ))
}

/// Emit with full statement support.
fn emit_full(ast: &Ast) -> Result<Emission, RustError> {
    match ast {
        Ast::Let { name, value, body } => {
            let value_emission = emit_full(value)?;
            let mut body_emission = emit_full(body)?;

            let mut statements = value_emission.statements;
            statements.push(format!("let {} = {};", name, value_emission.expr));
            statements.append(&mut body_emission.statements);

            Ok(Emission::with_statements(statements, body_emission.expr))
        }
        _ => Ok(Emission::expr_only(emit(ast)?)),
    }
}

/// Simple emit for expression-only nodes.
fn emit(ast: &Ast) -> Result<String, RustError> {
    match ast {
        Ast::Num(n) => Ok(format_float(*n)),
        Ast::Var(name) => Ok(name.clone()),
        Ast::BinOp(op, left, right) => {
            let l = emit_with_parens(left, Some(*op), true)?;
            let r = emit_with_parens(right, Some(*op), false)?;
            match op {
                BinOp::Add => Ok(format!("{} + {}", l, r)),
                BinOp::Sub => Ok(format!("{} - {}", l, r)),
                BinOp::Mul => Ok(format!("{} * {}", l, r)),
                BinOp::Div => Ok(format!("{} / {}", l, r)),
                BinOp::Pow => Ok(format!("{}.powf({})", emit(left)?, emit(right)?)),
                BinOp::Rem => Ok(format!("{} % {}", l, r)),
                BinOp::BitAnd => Ok(format!("{} & {}", l, r)),
                BinOp::BitOr => Ok(format!("{} | {}", l, r)),
                BinOp::Shl => Ok(format!("{} << {}", l, r)),
                BinOp::Shr => Ok(format!("{} >> {}", l, r)),
            }
        }
        Ast::UnaryOp(op, inner) => {
            let inner_str = emit_with_parens(inner, None, false)?;
            match op {
                UnaryOp::Neg => Ok(format!("-{}", inner_str)),
                UnaryOp::Not => {
                    // not(x) returns 1.0 if x == 0.0, else 0.0
                    let bool_expr = cond::scalar_to_bool(&inner_str);
                    Ok(cond::bool_to_scalar(&cond::emit_not(&bool_expr)))
                }
                UnaryOp::BitNot => Ok(format!("!{}", inner_str)),
            }
        }
        Ast::Compare(op, left, right) => {
            let l = emit(left)?;
            let r = emit(right)?;
            let bool_expr = cond::emit_compare(*op, &l, &r);
            Ok(cond::bool_to_scalar(&bool_expr))
        }
        Ast::And(left, right) => {
            let l = emit(left)?;
            let r = emit(right)?;
            let l_bool = cond::scalar_to_bool(&l);
            let r_bool = cond::scalar_to_bool(&r);
            let bool_expr = cond::emit_and(&l_bool, &r_bool);
            Ok(cond::bool_to_scalar(&bool_expr))
        }
        Ast::Or(left, right) => {
            let l = emit(left)?;
            let r = emit(right)?;
            let l_bool = cond::scalar_to_bool(&l);
            let r_bool = cond::scalar_to_bool(&r);
            let bool_expr = cond::emit_or(&l_bool, &r_bool);
            Ok(cond::bool_to_scalar(&bool_expr))
        }
        Ast::If(cond_ast, then_ast, else_ast) => {
            let c = emit(cond_ast)?;
            let then_expr = emit(then_ast)?;
            let else_expr = emit(else_ast)?;
            let cond_bool = cond::scalar_to_bool(&c);
            Ok(cond::emit_if(&cond_bool, &then_expr, &else_expr))
        }
        Ast::Call(name, args) => {
            let func =
                rust_func_name(name).ok_or_else(|| RustError::UnknownFunction(name.clone()))?;

            let args_str: Vec<String> = args.iter().map(emit).collect::<Result<_, _>>()?;

            Ok(match func {
                RustFunc::Const(path) => path.to_string(),
                RustFunc::Method(method) => {
                    let arg = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    format!("{}.{}()", arg, method)
                }
                RustFunc::Method2(method) => {
                    let arg0 = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    let arg1 = args_str.get(1).map(|s| s.as_str()).unwrap_or("0.0");
                    format!("{}.{}({})", arg0, method, arg1)
                }
                RustFunc::Atan2 => {
                    let y = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    let x = args_str.get(1).map(|s| s.as_str()).unwrap_or("1.0");
                    format!("{}.atan2({})", y, x)
                }
                RustFunc::Powf => {
                    let base = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    let exp = args_str.get(1).map(|s| s.as_str()).unwrap_or("1.0");
                    format!("{}.powf({})", base, exp)
                }
                RustFunc::InverseSqrt => {
                    let arg = args_str.first().map(|s| s.as_str()).unwrap_or("1.0");
                    format!("(1.0 / {}.sqrt())", arg)
                }
                RustFunc::Clamp => {
                    let x = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    let lo = args_str.get(1).map(|s| s.as_str()).unwrap_or("0.0");
                    let hi = args_str.get(2).map(|s| s.as_str()).unwrap_or("1.0");
                    format!("{}.clamp({}, {})", x, lo, hi)
                }
                RustFunc::Saturate => {
                    let arg = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    format!("{}.clamp(0.0, 1.0)", arg)
                }
                RustFunc::Lerp => {
                    let a = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    let b = args_str.get(1).map(|s| s.as_str()).unwrap_or("1.0");
                    let t = args_str.get(2).map(|s| s.as_str()).unwrap_or("0.0");
                    format!("({a} + ({b} - {a}) * {t})")
                }
                RustFunc::Step => {
                    let edge = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    let x = args_str.get(1).map(|s| s.as_str()).unwrap_or("0.0");
                    format!("(if {x} < {edge} {{ 0.0 }} else {{ 1.0 }})")
                }
                RustFunc::Smoothstep => {
                    let e0 = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    let e1 = args_str.get(1).map(|s| s.as_str()).unwrap_or("1.0");
                    let x = args_str.get(2).map(|s| s.as_str()).unwrap_or("0.0");
                    // t = clamp((x - e0) / (e1 - e0), 0.0, 1.0); t * t * (3.0 - 2.0 * t)
                    format!(
                        "({{ let t = (({x} - {e0}) / ({e1} - {e0})).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) }})"
                    )
                }
                RustFunc::InverseLerp => {
                    let a = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    let b = args_str.get(1).map(|s| s.as_str()).unwrap_or("1.0");
                    let v = args_str.get(2).map(|s| s.as_str()).unwrap_or("0.0");
                    format!("(({v} - {a}) / ({b} - {a}))")
                }
                RustFunc::Remap => {
                    let x = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    let in_lo = args_str.get(1).map(|s| s.as_str()).unwrap_or("0.0");
                    let in_hi = args_str.get(2).map(|s| s.as_str()).unwrap_or("1.0");
                    let out_lo = args_str.get(3).map(|s| s.as_str()).unwrap_or("0.0");
                    let out_hi = args_str.get(4).map(|s| s.as_str()).unwrap_or("1.0");
                    format!(
                        "({out_lo} + ({out_hi} - {out_lo}) * (({x} - {in_lo}) / ({in_hi} - {in_lo})))"
                    )
                }
            })
        }
        Ast::Let { .. } => {
            // Let in expression context: delegate to emit_full
            let emission = emit_full(ast)?;
            if emission.statements.is_empty() {
                Ok(emission.expr)
            } else {
                // Can't inline statements into expression position
                Err(RustError::UnsupportedFeature(
                    "let in expression position (use emit_rust_fn for full support)".to_string(),
                ))
            }
        }
    }
}

fn emit_with_parens(
    ast: &Ast,
    parent_op: Option<BinOp>,
    is_left: bool,
) -> Result<String, RustError> {
    let inner = emit(ast)?;

    let needs_parens = match ast {
        Ast::BinOp(child_op, _, _) => {
            if let Some(parent) = parent_op {
                let parent_prec = precedence(parent);
                let child_prec = precedence(*child_op);
                if child_prec < parent_prec {
                    true
                } else if child_prec == parent_prec && !is_left {
                    matches!(parent, BinOp::Sub | BinOp::Div)
                } else {
                    false
                }
            } else {
                false
            }
        }
        _ => false,
    };

    if needs_parens {
        Ok(format!("({})", inner))
    } else {
        Ok(inner)
    }
}

fn precedence(op: BinOp) -> u8 {
    match op {
        BinOp::BitOr => 0,
        BinOp::BitAnd => 0,
        BinOp::Shl | BinOp::Shr => 0,
        BinOp::Add | BinOp::Sub => 1,
        BinOp::Mul | BinOp::Div | BinOp::Rem => 2,
        BinOp::Pow => 3,
    }
}

fn format_float(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{:.1}", n)
    } else {
        format!("{}", n)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn compile(input: &str) -> String {
        let expr = Expr::parse(input).unwrap();
        emit_rust(expr.ast()).unwrap().code
    }

    #[test]
    fn test_constants() {
        assert_eq!(compile("pi()"), "std::f32::consts::PI");
        assert_eq!(compile("e()"), "std::f32::consts::E");
        assert_eq!(compile("tau()"), "std::f32::consts::TAU");
    }

    #[test]
    fn test_trig() {
        assert_eq!(compile("sin(x)"), "x.sin()");
        assert_eq!(compile("cos(x)"), "x.cos()");
        assert_eq!(compile("atan2(y, x)"), "y.atan2(x)");
    }

    #[test]
    fn test_exp_log() {
        assert_eq!(compile("exp(x)"), "x.exp()");
        assert_eq!(compile("ln(x)"), "x.ln()");
        assert_eq!(compile("log10(x)"), "x.log10()");
        assert_eq!(compile("pow(x, 2)"), "x.powf(2.0)");
        assert_eq!(compile("sqrt(x)"), "x.sqrt()");
        assert_eq!(compile("inversesqrt(x)"), "(1.0 / x.sqrt())");
    }

    #[test]
    fn test_common() {
        assert_eq!(compile("abs(x)"), "x.abs()");
        assert_eq!(compile("floor(x)"), "x.floor()");
        assert_eq!(compile("clamp(x, 0, 1)"), "x.clamp(0.0, 1.0)");
        assert_eq!(compile("saturate(x)"), "x.clamp(0.0, 1.0)");
        assert_eq!(compile("sign(x)"), "x.signum()");
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(compile("lerp(a, b, t)"), "(a + (b - a) * t)");
        assert_eq!(compile("mix(a, b, t)"), "(a + (b - a) * t)");
        assert_eq!(compile("step(e, x)"), "(if x < e { 0.0 } else { 1.0 })");
    }

    #[test]
    fn test_inverse_lerp() {
        assert_eq!(
            compile("inverse_lerp(0, 10, x)"),
            "((x - 0.0) / (10.0 - 0.0))"
        );
    }

    #[test]
    fn test_operators() {
        assert_eq!(compile("x + y"), "x + y");
        assert_eq!(compile("x * y + z"), "x * y + z");
        assert_eq!(compile("(x + y) * z"), "(x + y) * z");
        assert_eq!(compile("-x"), "-x");
        assert_eq!(compile("x ^ 2"), "x.powf(2.0)");
    }

    #[test]
    fn test_rust_fn() {
        let expr = Expr::parse("x + y").unwrap();
        let code = emit_rust_fn("add", expr.ast(), &["x", "y"]).unwrap();
        assert!(code.contains("fn add(x: f32, y: f32) -> f32"));
        assert!(code.contains("x + y"));
    }

    #[test]
    fn test_compare() {
        let code = compile("x < y");
        assert!(code.contains("if"));
        assert!(code.contains("(x < y)"));
    }

    #[test]
    fn test_if_then_else() {
        let code = compile("if x > 0 then 1 else 0");
        assert!(code.contains("if"));
        assert!(code.contains("else"));
    }

    #[test]
    fn test_and_or() {
        let and_code = compile("x > 0 and y > 0");
        assert!(and_code.contains("&&"));

        let or_code = compile("x < 0 or y < 0");
        assert!(or_code.contains("||"));
    }

    #[test]
    fn test_not() {
        let code = compile("not x");
        assert!(code.contains("!"));
    }

    #[test]
    fn test_let_in_fn() {
        let expr = Expr::parse("let t = x * 2; t + t").unwrap();
        let code = emit_rust_fn("double_add", expr.ast(), &["x"]).unwrap();
        assert!(code.contains("let t = x * 2.0;"));
        assert!(code.contains("t + t"));
    }

    #[test]
    fn test_nested_let() {
        let expr = Expr::parse("let a = x; let b = a * 2; b + 1").unwrap();
        let code = emit_rust_fn("nested", expr.ast(), &["x"]).unwrap();
        assert!(code.contains("let a = x;"));
        assert!(code.contains("let b = a * 2.0;"));
        assert!(code.contains("b + 1.0"));
    }

    #[test]
    fn test_min_max() {
        assert_eq!(compile("min(a, b)"), "a.min(b)");
        assert_eq!(compile("max(a, b)"), "a.max(b)");
    }
}
