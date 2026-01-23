//! Rust code generation for quaternion expressions.
//!
//! Quaternions use glam::Quat, vectors use glam::Vec3.

use crate::Type;
use rhizome_dew_cond::rust as cond;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

/// Error during Rust code generation.
#[derive(Debug, Clone, PartialEq)]
pub enum RustError {
    UnknownVariable(String),
    UnknownFunction(String),
    TypeMismatch {
        op: &'static str,
        left: Type,
        right: Type,
    },
    /// Conditionals require scalar types.
    UnsupportedTypeForConditional(Type),
    /// Operation not supported for this type.
    UnsupportedOperation(&'static str),
    /// Feature not supported in expression-only codegen.
    UnsupportedFeature(String),
}

impl std::fmt::Display for RustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RustError::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            RustError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            RustError::TypeMismatch { op, left, right } => {
                write!(f, "type mismatch for {op}: {left} vs {right}")
            }
            RustError::UnsupportedTypeForConditional(t) => {
                write!(f, "conditionals require scalar type, got {t}")
            }
            RustError::UnsupportedOperation(op) => {
                write!(f, "unsupported operation for quaternion: {op}")
            }
            RustError::UnsupportedFeature(feat) => {
                write!(f, "unsupported feature in expression codegen: {feat}")
            }
        }
    }
}

impl std::error::Error for RustError {}

/// Convert a Type to its Rust/glam representation.
pub fn type_to_rust(t: Type) -> &'static str {
    match t {
        Type::Scalar => "f32",
        Type::Vec3 => "Vec3",
        Type::Quaternion => "Quat",
    }
}

/// Result of Rust emission: code string and its type.
pub struct RustExpr {
    pub code: String,
    pub typ: Type,
}

/// Result of full Rust emission with accumulated statements.
struct Emission {
    statements: Vec<String>,
    expr: String,
    typ: Type,
}

/// Format a numeric literal for Rust.
fn format_literal(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{:.1}", n)
    } else {
        format!("{}", n)
    }
}

/// Emit Rust code for an AST with type propagation.
pub fn emit_rust(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<RustExpr, RustError> {
    match ast {
        Ast::Num(n) => Ok(RustExpr {
            code: format_literal(*n),
            typ: Type::Scalar,
        }),

        Ast::Var(name) => {
            let typ = var_types
                .get(name)
                .copied()
                .ok_or_else(|| RustError::UnknownVariable(name.clone()))?;
            Ok(RustExpr {
                code: name.clone(),
                typ,
            })
        }

        Ast::BinOp(op, left, right) => {
            let left_expr = emit_rust(left, var_types)?;
            let right_expr = emit_rust(right, var_types)?;
            emit_binop(*op, left_expr, right_expr)
        }

        Ast::UnaryOp(op, inner) => {
            let inner_expr = emit_rust(inner, var_types)?;
            emit_unaryop(*op, inner_expr)
        }

        Ast::Call(name, args) => {
            let arg_exprs: Vec<RustExpr> = args
                .iter()
                .map(|a| emit_rust(a, var_types))
                .collect::<Result<_, _>>()?;
            emit_function_call(name, arg_exprs)
        }

        Ast::Compare(op, left, right) => {
            let left_expr = emit_rust(left, var_types)?;
            let right_expr = emit_rust(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(RustError::UnsupportedTypeForConditional(left_expr.typ));
            }
            let bool_expr = cond::emit_compare(*op, &left_expr.code, &right_expr.code);
            Ok(RustExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::And(left, right) => {
            let left_expr = emit_rust(left, var_types)?;
            let right_expr = emit_rust(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(RustError::UnsupportedTypeForConditional(left_expr.typ));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_and(&l_bool, &r_bool);
            Ok(RustExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::Or(left, right) => {
            let left_expr = emit_rust(left, var_types)?;
            let right_expr = emit_rust(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(RustError::UnsupportedTypeForConditional(left_expr.typ));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_or(&l_bool, &r_bool);
            Ok(RustExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::If(cond_ast, then_ast, else_ast) => {
            let cond_expr = emit_rust(cond_ast, var_types)?;
            let then_expr = emit_rust(then_ast, var_types)?;
            let else_expr = emit_rust(else_ast, var_types)?;
            if cond_expr.typ != Type::Scalar {
                return Err(RustError::UnsupportedTypeForConditional(cond_expr.typ));
            }
            if then_expr.typ != else_expr.typ {
                return Err(RustError::TypeMismatch {
                    op: "if/else",
                    left: then_expr.typ,
                    right: else_expr.typ,
                });
            }
            let cond_bool = cond::scalar_to_bool(&cond_expr.code);
            Ok(RustExpr {
                code: cond::emit_if(&cond_bool, &then_expr.code, &else_expr.code),
                typ: then_expr.typ,
            })
        }

        Ast::Let { .. } => {
            let emission = emit_full(ast, var_types)?;
            if !emission.statements.is_empty() {
                return Err(RustError::UnsupportedFeature(
                    "let bindings in expression context (use emit_rust_fn)".to_string(),
                ));
            }
            Ok(RustExpr {
                code: emission.expr,
                typ: emission.typ,
            })
        }
    }
}

/// Emit Rust with full statement support.
fn emit_full(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<Emission, RustError> {
    match ast {
        Ast::Let { name, value, body } => {
            let value_emission = emit_full(value, var_types)?;
            let mut new_var_types = var_types.clone();
            new_var_types.insert(name.clone(), value_emission.typ);
            let body_emission = emit_full(body, &new_var_types)?;

            let mut statements = value_emission.statements;
            statements.push(format!(
                "let {}: {} = {};",
                name,
                type_to_rust(value_emission.typ),
                value_emission.expr
            ));
            statements.extend(body_emission.statements);

            Ok(Emission {
                statements,
                expr: body_emission.expr,
                typ: body_emission.typ,
            })
        }
        _ => {
            let result = emit_rust(ast, var_types)?;
            Ok(Emission {
                statements: vec![],
                expr: result.code,
                typ: result.typ,
            })
        }
    }
}

/// Emit a complete Rust function with let statement support.
pub fn emit_rust_fn(
    name: &str,
    ast: &Ast,
    params: &[(&str, Type)],
    return_type: Type,
) -> Result<String, RustError> {
    let var_types: HashMap<String, Type> =
        params.iter().map(|(n, t)| (n.to_string(), *t)).collect();
    let emission = emit_full(ast, &var_types)?;

    let params_str = params
        .iter()
        .map(|(n, t)| format!("{}: {}", n, type_to_rust(*t)))
        .collect::<Vec<_>>()
        .join(", ");

    let mut body = String::new();
    for stmt in &emission.statements {
        body.push_str("    ");
        body.push_str(stmt);
        body.push('\n');
    }
    body.push_str("    ");
    body.push_str(&emission.expr);

    Ok(format!(
        "fn {}({}) -> {} {{\n{}\n}}",
        name,
        params_str,
        type_to_rust(return_type),
        body
    ))
}

fn emit_binop(op: BinOp, left: RustExpr, right: RustExpr) -> Result<RustExpr, RustError> {
    match op {
        BinOp::Add => emit_add(left, right),
        BinOp::Sub => emit_sub(left, right),
        BinOp::Mul => emit_mul(left, right),
        BinOp::Div => emit_div(left, right),
        BinOp::Pow => emit_pow(left, right),
        BinOp::Rem => Err(RustError::UnsupportedOperation("%")),
        BinOp::BitAnd => Err(RustError::UnsupportedOperation("&")),
        BinOp::BitOr => Err(RustError::UnsupportedOperation("|")),
        BinOp::Shl => Err(RustError::UnsupportedOperation("<<")),
        BinOp::Shr => Err(RustError::UnsupportedOperation(">>")),
    }
}

fn emit_add(left: RustExpr, right: RustExpr) -> Result<RustExpr, RustError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar)
        | (Type::Vec3, Type::Vec3)
        | (Type::Quaternion, Type::Quaternion) => Ok(RustExpr {
            code: format!("({} + {})", left.code, right.code),
            typ: left.typ,
        }),
        _ => Err(RustError::TypeMismatch {
            op: "+",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_sub(left: RustExpr, right: RustExpr) -> Result<RustExpr, RustError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar)
        | (Type::Vec3, Type::Vec3)
        | (Type::Quaternion, Type::Quaternion) => Ok(RustExpr {
            code: format!("({} - {})", left.code, right.code),
            typ: left.typ,
        }),
        _ => Err(RustError::TypeMismatch {
            op: "-",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_mul(left: RustExpr, right: RustExpr) -> Result<RustExpr, RustError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(RustExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Scalar,
        }),

        (Type::Scalar, Type::Vec3) | (Type::Vec3, Type::Scalar) => Ok(RustExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Vec3,
        }),

        (Type::Scalar, Type::Quaternion) | (Type::Quaternion, Type::Scalar) => Ok(RustExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Quaternion,
        }),

        // Quaternion * Quaternion (Hamilton product)
        (Type::Quaternion, Type::Quaternion) => Ok(RustExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Quaternion,
        }),

        // Quaternion * Vec3 (rotate vector)
        (Type::Quaternion, Type::Vec3) => {
            let q = &left.code;
            let v = &right.code;
            Ok(RustExpr {
                code: format!("{q}.mul_vec3({v})"),
                typ: Type::Vec3,
            })
        }

        _ => Err(RustError::TypeMismatch {
            op: "*",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_div(left: RustExpr, right: RustExpr) -> Result<RustExpr, RustError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(RustExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        (Type::Vec3, Type::Scalar) => Ok(RustExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Vec3,
        }),
        (Type::Quaternion, Type::Scalar) => Ok(RustExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Quaternion,
        }),
        _ => Err(RustError::TypeMismatch {
            op: "/",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_pow(base: RustExpr, exp: RustExpr) -> Result<RustExpr, RustError> {
    match (base.typ, exp.typ) {
        (Type::Scalar, Type::Scalar) => Ok(RustExpr {
            code: format!("{}.powf({})", base.code, exp.code),
            typ: Type::Scalar,
        }),
        _ => Err(RustError::TypeMismatch {
            op: "^",
            left: base.typ,
            right: exp.typ,
        }),
    }
}

fn emit_unaryop(op: UnaryOp, inner: RustExpr) -> Result<RustExpr, RustError> {
    match op {
        UnaryOp::Neg => Ok(RustExpr {
            code: format!("(-{})", inner.code),
            typ: inner.typ,
        }),
        UnaryOp::Not => {
            if inner.typ != Type::Scalar {
                return Err(RustError::UnsupportedTypeForConditional(inner.typ));
            }
            let bool_expr = cond::scalar_to_bool(&inner.code);
            Ok(RustExpr {
                code: cond::bool_to_scalar(&cond::emit_not(&bool_expr)),
                typ: Type::Scalar,
            })
        }
        UnaryOp::BitNot => Err(RustError::UnsupportedOperation("~")),
    }
}

fn emit_function_call(name: &str, args: Vec<RustExpr>) -> Result<RustExpr, RustError> {
    match name {
        "conj" => {
            if args.len() != 1 || args[0].typ != Type::Quaternion {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.conjugate()", args[0].code),
                typ: Type::Quaternion,
            })
        }

        "length" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.length()", args[0].code),
                typ: Type::Scalar,
            })
        }

        "normalize" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.normalize()", args[0].code),
                typ: args[0].typ,
            })
        }

        "inverse" => {
            if args.len() != 1 || args[0].typ != Type::Quaternion {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.inverse()", args[0].code),
                typ: Type::Quaternion,
            })
        }

        "dot" => {
            if args.len() != 2 || args[0].typ != args[1].typ {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.dot({})", args[0].code, args[1].code),
                typ: Type::Scalar,
            })
        }

        "lerp" => {
            if args.len() != 3 || args[2].typ != Type::Scalar {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.lerp({}, {})", args[0].code, args[1].code, args[2].code),
                typ: args[0].typ,
            })
        }

        "slerp" => {
            if args.len() != 3
                || args[0].typ != Type::Quaternion
                || args[1].typ != Type::Quaternion
                || args[2].typ != Type::Scalar
            {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.slerp({}, {})", args[0].code, args[1].code, args[2].code),
                typ: Type::Quaternion,
            })
        }

        "axis_angle" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Scalar {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("Quat::from_axis_angle({}, {})", args[0].code, args[1].code),
                typ: Type::Quaternion,
            })
        }

        "rotate" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Quaternion {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.mul_vec3({})", args[1].code, args[0].code),
                typ: Type::Vec3,
            })
        }

        "cross" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Vec3 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.cross({})", args[0].code, args[1].code),
                typ: Type::Vec3,
            })
        }

        "vec3" => {
            if args.len() != 3
                || args[0].typ != Type::Scalar
                || args[1].typ != Type::Scalar
                || args[2].typ != Type::Scalar
            {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!(
                    "Vec3::new({}, {}, {})",
                    args[0].code, args[1].code, args[2].code
                ),
                typ: Type::Vec3,
            })
        }

        "quat" => {
            if args.len() != 4 || args.iter().any(|a| a.typ != Type::Scalar) {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!(
                    "Quat::from_xyzw({}, {}, {}, {})",
                    args[0].code, args[1].code, args[2].code, args[3].code
                ),
                typ: Type::Quaternion,
            })
        }

        _ => Err(RustError::UnknownFunction(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn emit(expr: &str, var_types: &[(&str, Type)]) -> Result<RustExpr, RustError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_rust(expr.ast(), &types)
    }

    #[test]
    fn test_quaternion_add() {
        let result = emit("a + b", &[("a", Type::Quaternion), ("b", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.contains("+"));
    }

    #[test]
    fn test_quaternion_mul() {
        let result = emit("a * b", &[("a", Type::Quaternion), ("b", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.contains("*"));
    }

    #[test]
    fn test_quaternion_rotate_vec() {
        let result = emit("q * v", &[("q", Type::Quaternion), ("v", Type::Vec3)]).unwrap();
        assert_eq!(result.typ, Type::Vec3);
        assert!(result.code.contains(".mul_vec3("));
    }

    #[test]
    fn test_normalize() {
        let result = emit("normalize(q)", &[("q", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.contains(".normalize()"));
    }

    #[test]
    fn test_conj() {
        let result = emit("conj(q)", &[("q", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.contains(".conjugate()"));
    }

    #[test]
    fn test_dot() {
        let result = emit(
            "dot(a, b)",
            &[("a", Type::Quaternion), ("b", Type::Quaternion)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains(".dot("));
    }

    #[test]
    fn test_axis_angle() {
        let result = emit(
            "axis_angle(v, a)",
            &[("v", Type::Vec3), ("a", Type::Scalar)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.contains("Quat::from_axis_angle"));
    }

    #[test]
    fn test_rotate() {
        let result = emit(
            "rotate(v, q)",
            &[("v", Type::Vec3), ("q", Type::Quaternion)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Vec3);
        assert!(result.code.contains(".mul_vec3("));
    }

    #[test]
    fn test_slerp() {
        let result = emit(
            "slerp(a, b, t)",
            &[
                ("a", Type::Quaternion),
                ("b", Type::Quaternion),
                ("t", Type::Scalar),
            ],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.contains(".slerp("));
    }

    #[test]
    fn test_emit_rust_fn_simple() {
        let expr = Expr::parse("normalize(q)").unwrap();
        let code = emit_rust_fn(
            "norm_quat",
            expr.ast(),
            &[("q", Type::Quaternion)],
            Type::Quaternion,
        )
        .unwrap();
        assert!(code.contains("fn norm_quat(q: Quat) -> Quat"));
        assert!(code.contains(".normalize()"));
    }

    #[test]
    fn test_emit_rust_fn_with_let() {
        let expr = Expr::parse("let sq = q * q; sq + q").unwrap();
        let code = emit_rust_fn(
            "square_add",
            expr.ast(),
            &[("q", Type::Quaternion)],
            Type::Quaternion,
        )
        .unwrap();
        assert!(code.contains("let sq: Quat ="));
        assert!(code.contains("(sq + q)"));
    }
}
