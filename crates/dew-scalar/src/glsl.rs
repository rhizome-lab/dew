//! GLSL code generation for scalar expressions.
//!
//! Compiles expression ASTs to GLSL shader code (WebGL/OpenGL compatible).

use rhizome_dew_cond::glsl as cond;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};

// ============================================================================
// Errors
// ============================================================================

/// GLSL emission error.
#[derive(Debug, Clone, PartialEq)]
pub enum GlslError {
    /// Unknown function.
    UnknownFunction(String),
    /// Feature not supported in GLSL codegen.
    UnsupportedFeature(String),
}

impl std::fmt::Display for GlslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlslError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            GlslError::UnsupportedFeature(feat) => {
                write!(f, "unsupported feature in GLSL codegen: {feat}")
            }
        }
    }
}

impl std::error::Error for GlslError {}

// ============================================================================
// GLSL expression output
// ============================================================================

/// Result of emitting GLSL code.
#[derive(Debug, Clone)]
pub struct GlslExpr {
    /// The GLSL expression string.
    pub code: String,
}

impl GlslExpr {
    pub fn new(code: impl Into<String>) -> Self {
        Self { code: code.into() }
    }
}

// ============================================================================
// Function name mapping
// ============================================================================

/// Returns the GLSL equivalent for a scalar function name.
fn glsl_func_name(name: &str) -> Option<GlslFunc> {
    Some(match name {
        // Constants
        "pi" => GlslFunc::Const(std::f64::consts::PI),
        "e" => GlslFunc::Const(std::f64::consts::E),
        "tau" => GlslFunc::Const(std::f64::consts::TAU),

        // Trig - direct mapping
        "sin" => GlslFunc::Direct("sin"),
        "cos" => GlslFunc::Direct("cos"),
        "tan" => GlslFunc::Direct("tan"),
        "asin" => GlslFunc::Direct("asin"),
        "acos" => GlslFunc::Direct("acos"),
        "atan" => GlslFunc::Direct("atan"),
        "atan2" => GlslFunc::Atan2, // GLSL uses atan(y, x) not atan2
        "sinh" => GlslFunc::Direct("sinh"),
        "cosh" => GlslFunc::Direct("cosh"),
        "tanh" => GlslFunc::Direct("tanh"),

        // Exp/log
        "exp" => GlslFunc::Direct("exp"),
        "exp2" => GlslFunc::Direct("exp2"),
        "log2" => GlslFunc::Direct("log2"),
        "pow" => GlslFunc::Direct("pow"),
        "sqrt" => GlslFunc::Direct("sqrt"),
        "log" | "ln" => GlslFunc::Direct("log"),
        "log10" => GlslFunc::Log10,
        "inversesqrt" => GlslFunc::Direct("inversesqrt"),

        // Common math
        "abs" => GlslFunc::Direct("abs"),
        "sign" => GlslFunc::Direct("sign"),
        "floor" => GlslFunc::Direct("floor"),
        "ceil" => GlslFunc::Direct("ceil"),
        "round" => GlslFunc::Direct("round"),
        "trunc" => GlslFunc::Direct("trunc"),
        "fract" => GlslFunc::Direct("fract"),
        "min" => GlslFunc::Direct("min"),
        "max" => GlslFunc::Direct("max"),
        "clamp" => GlslFunc::Direct("clamp"),
        "saturate" => GlslFunc::Saturate, // GLSL doesn't have saturate

        // Interpolation
        "lerp" | "mix" => GlslFunc::Direct("mix"),
        "step" => GlslFunc::Direct("step"),
        "smoothstep" => GlslFunc::Direct("smoothstep"),
        "inverse_lerp" => GlslFunc::InverseLerp,
        "remap" => GlslFunc::Remap,

        _ => return None,
    })
}

enum GlslFunc {
    /// Constant value
    Const(f64),
    /// Direct function name mapping
    Direct(&'static str),
    /// atan2(y, x) -> atan(y, x) in GLSL
    Atan2,
    /// log10(x) = log(x) / log(10)
    Log10,
    /// saturate(x) = clamp(x, 0.0, 1.0)
    Saturate,
    /// inverse_lerp(a, b, v) = (v - a) / (b - a)
    InverseLerp,
    /// remap(x, in_lo, in_hi, out_lo, out_hi)
    Remap,
}

// ============================================================================
// Code generation
// ============================================================================

/// Emit GLSL code for an AST.
pub fn emit_glsl(ast: &Ast) -> Result<GlslExpr, GlslError> {
    Ok(GlslExpr::new(emit(ast)?))
}

/// Generate a complete GLSL function.
pub fn emit_glsl_fn(name: &str, ast: &Ast, params: &[&str]) -> Result<String, GlslError> {
    let param_list: String = params
        .iter()
        .map(|p| format!("float {}", p))
        .collect::<Vec<_>>()
        .join(", ");

    let body = emit(ast)?;
    Ok(format!(
        "float {}({}) {{\n    return {};\n}}",
        name, param_list, body
    ))
}

fn emit(ast: &Ast) -> Result<String, GlslError> {
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
                BinOp::Pow => Ok(format!("pow({}, {})", emit(left)?, emit(right)?)),
                BinOp::Rem => Ok(format!("mod({}, {})", emit(left)?, emit(right)?)),
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
                UnaryOp::BitNot => Ok(format!("~{}", inner_str)),
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
                glsl_func_name(name).ok_or_else(|| GlslError::UnknownFunction(name.clone()))?;

            let args_str: Vec<String> = args.iter().map(emit).collect::<Result<_, _>>()?;

            Ok(match func {
                GlslFunc::Const(v) => format!("{:.10}", v),
                GlslFunc::Direct(glsl_name) => format!("{}({})", glsl_name, args_str.join(", ")),
                GlslFunc::Atan2 => {
                    // GLSL uses atan(y, x) instead of atan2(y, x)
                    format!(
                        "atan({}, {})",
                        args_str.first().unwrap_or(&"0.0".to_string()),
                        args_str.get(1).unwrap_or(&"1.0".to_string())
                    )
                }
                GlslFunc::Log10 => {
                    format!(
                        "(log({}) / log(10.0))",
                        args_str.first().unwrap_or(&"0.0".to_string())
                    )
                }
                GlslFunc::Saturate => {
                    format!(
                        "clamp({}, 0.0, 1.0)",
                        args_str.first().unwrap_or(&"0.0".to_string())
                    )
                }
                GlslFunc::InverseLerp => {
                    let a = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    let b = args_str.get(1).map(|s| s.as_str()).unwrap_or("1.0");
                    let v = args_str.get(2).map(|s| s.as_str()).unwrap_or("0.0");
                    format!("(({v} - {a}) / ({b} - {a}))")
                }
                GlslFunc::Remap => {
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
            // GLSL uses statement-based variable declarations, not expression-based.
            // Use the optimizer to inline let bindings before GLSL codegen.
            Err(GlslError::UnsupportedFeature(
                "let expressions (use optimizer to inline)".to_string(),
            ))
        }
    }
}

fn emit_with_parens(
    ast: &Ast,
    parent_op: Option<BinOp>,
    is_left: bool,
) -> Result<String, GlslError> {
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
        emit_glsl(expr.ast()).unwrap().code
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
        assert_eq!(compile("atan2(y, x)"), "atan(y, x)");
    }

    #[test]
    fn test_exp_log() {
        assert_eq!(compile("exp(x)"), "exp(x)");
        assert_eq!(compile("ln(x)"), "log(x)");
        assert_eq!(compile("log10(x)"), "(log(x) / log(10.0))");
        assert_eq!(compile("pow(x, 2)"), "pow(x, 2.0)");
        assert_eq!(compile("sqrt(x)"), "sqrt(x)");
        assert_eq!(compile("inversesqrt(x)"), "inversesqrt(x)");
    }

    #[test]
    fn test_common() {
        assert_eq!(compile("abs(x)"), "abs(x)");
        assert_eq!(compile("floor(x)"), "floor(x)");
        assert_eq!(compile("clamp(x, 0, 1)"), "clamp(x, 0.0, 1.0)");
        assert_eq!(compile("saturate(x)"), "clamp(x, 0.0, 1.0)");
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
    fn test_operators() {
        assert_eq!(compile("x + y"), "x + y");
        assert_eq!(compile("x * y + z"), "x * y + z");
        assert_eq!(compile("(x + y) * z"), "(x + y) * z");
        assert_eq!(compile("-x"), "-x");
        assert_eq!(compile("x ^ 2"), "pow(x, 2.0)");
    }

    #[test]
    fn test_glsl_fn() {
        let expr = Expr::parse("x + y").unwrap();
        let code = emit_glsl_fn("add", expr.ast(), &["x", "y"]).unwrap();
        assert!(code.contains("float add(float x, float y)"));
        assert!(code.contains("return x + y"));
    }

    #[test]
    fn test_compare() {
        // Comparisons use ternary operator
        assert!(compile("x < y").contains("?"));
        assert!(compile("x < y").contains("(x < y)"));
    }

    #[test]
    fn test_if_then_else() {
        let code = compile("if x > 0 then 1 else 0");
        // Should use ternary operator
        assert!(code.contains("?"));
        assert!(code.contains(":"));
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
}
