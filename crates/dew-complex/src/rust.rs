//! Rust code generation for complex expressions.
//!
//! Complex numbers are represented using num_complex::Complex32 (Complex<f32>).

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
                write!(f, "unsupported operation for complex: {op}")
            }
            RustError::UnsupportedFeature(feat) => {
                write!(f, "unsupported feature in expression codegen: {feat}")
            }
        }
    }
}

impl std::error::Error for RustError {}

/// Convert a Type to its Rust representation.
pub fn type_to_rust(t: Type) -> &'static str {
    match t {
        Type::Scalar => "f32",
        Type::Complex => "Complex32",
    }
}

/// Result of Rust emission: code string and its type.
pub struct RustExpr {
    pub code: String,
    pub typ: Type,
}

/// Result of emitting code with statement support.
struct Emission {
    statements: Vec<String>,
    expr: String,
    typ: Type,
}

impl Emission {
    fn expr_only(expr: String, typ: Type) -> Self {
        Self {
            statements: vec![],
            expr,
            typ,
        }
    }

    fn with_statements(statements: Vec<String>, expr: String, typ: Type) -> Self {
        Self {
            statements,
            expr,
            typ,
        }
    }
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

        Ast::Compare(op, left, right) => {
            let left_expr = emit_rust(left, var_types)?;
            let right_expr = emit_rust(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(RustError::UnsupportedTypeForConditional(Type::Complex));
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
                return Err(RustError::UnsupportedTypeForConditional(Type::Complex));
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
                return Err(RustError::UnsupportedTypeForConditional(Type::Complex));
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

        Ast::Call(name, args) => {
            let arg_exprs: Vec<RustExpr> = args
                .iter()
                .map(|a| emit_rust(a, var_types))
                .collect::<Result<_, _>>()?;
            emit_function_call(name, arg_exprs)
        }

        Ast::Let { .. } => {
            let emission = emit_full(ast, var_types)?;
            if emission.statements.is_empty() {
                Ok(RustExpr {
                    code: emission.expr,
                    typ: emission.typ,
                })
            } else {
                Err(RustError::UnsupportedFeature(
                    "let in expression position (use emit_rust_fn for full support)".to_string(),
                ))
            }
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

    let param_list: Vec<String> = params
        .iter()
        .map(|(n, t)| format!("{}: {}", n, type_to_rust(*t)))
        .collect();

    let mut body = String::new();
    for stmt in emission.statements {
        body.push_str("    ");
        body.push_str(&stmt);
        body.push('\n');
    }
    body.push_str("    ");
    body.push_str(&emission.expr);

    Ok(format!(
        "fn {}({}) -> {} {{\n{}\n}}",
        name,
        param_list.join(", "),
        type_to_rust(return_type),
        body
    ))
}

/// Emit with full statement support for let bindings.
fn emit_full(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<Emission, RustError> {
    match ast {
        Ast::Let { name, value, body } => {
            let value_emission = emit_full(value, var_types)?;
            let mut new_var_types = var_types.clone();
            new_var_types.insert(name.clone(), value_emission.typ);
            let mut body_emission = emit_full(body, &new_var_types)?;

            let mut statements = value_emission.statements;
            statements.push(format!(
                "let {}: {} = {};",
                name,
                type_to_rust(value_emission.typ),
                value_emission.expr
            ));
            statements.append(&mut body_emission.statements);

            Ok(Emission::with_statements(
                statements,
                body_emission.expr,
                body_emission.typ,
            ))
        }
        _ => {
            let result = emit_rust(ast, var_types)?;
            Ok(Emission::expr_only(result.code, result.typ))
        }
    }
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
        (Type::Scalar, Type::Scalar) | (Type::Complex, Type::Complex) => Ok(RustExpr {
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
        (Type::Scalar, Type::Scalar) | (Type::Complex, Type::Complex) => Ok(RustExpr {
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
        (Type::Scalar, Type::Complex) | (Type::Complex, Type::Scalar) => Ok(RustExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Complex,
        }),
        (Type::Complex, Type::Complex) => Ok(RustExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Complex,
        }),
    }
}

fn emit_div(left: RustExpr, right: RustExpr) -> Result<RustExpr, RustError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(RustExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        (Type::Complex, Type::Scalar) => Ok(RustExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Complex,
        }),
        (Type::Complex, Type::Complex) => Ok(RustExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Complex,
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
        (Type::Complex, Type::Scalar) => Ok(RustExpr {
            code: format!("{}.powf({})", base.code, exp.code),
            typ: Type::Complex,
        }),
        (Type::Complex, Type::Complex) => Ok(RustExpr {
            code: format!("{}.powc({})", base.code, exp.code),
            typ: Type::Complex,
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
        "re" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.re", args[0].code),
                typ: Type::Scalar,
            })
        }

        "im" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.im", args[0].code),
                typ: Type::Scalar,
            })
        }

        "conj" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.conj()", args[0].code),
                typ: Type::Complex,
            })
        }

        "abs" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(RustExpr {
                    code: format!("{}.abs()", args[0].code),
                    typ: Type::Scalar,
                }),
                Type::Complex => Ok(RustExpr {
                    code: format!("{}.norm()", args[0].code),
                    typ: Type::Scalar,
                }),
            }
        }

        "arg" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.arg()", args[0].code),
                typ: Type::Scalar,
            })
        }

        "norm" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.norm_sqr()", args[0].code),
                typ: Type::Scalar,
            })
        }

        "exp" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(RustExpr {
                    code: format!("{}.exp()", args[0].code),
                    typ: Type::Scalar,
                }),
                Type::Complex => Ok(RustExpr {
                    code: format!("{}.exp()", args[0].code),
                    typ: Type::Complex,
                }),
            }
        }

        "log" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(RustExpr {
                    code: format!("{}.ln()", args[0].code),
                    typ: Type::Scalar,
                }),
                Type::Complex => Ok(RustExpr {
                    code: format!("{}.ln()", args[0].code),
                    typ: Type::Complex,
                }),
            }
        }

        "sqrt" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(RustExpr {
                    code: format!("{}.sqrt()", args[0].code),
                    typ: Type::Scalar,
                }),
                Type::Complex => Ok(RustExpr {
                    code: format!("{}.sqrt()", args[0].code),
                    typ: Type::Complex,
                }),
            }
        }

        "polar" => {
            if args.len() != 2 || args[0].typ != Type::Scalar || args[1].typ != Type::Scalar {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("Complex32::from_polar({}, {})", args[0].code, args[1].code),
                typ: Type::Complex,
            })
        }

        "complex" => {
            if args.len() != 2 || args[0].typ != Type::Scalar || args[1].typ != Type::Scalar {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("Complex32::new({}, {})", args[0].code, args[1].code),
                typ: Type::Complex,
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
    fn test_complex_add() {
        let result = emit("a + b", &[("a", Type::Complex), ("b", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.contains("+"));
    }

    #[test]
    fn test_complex_mul() {
        let result = emit("a * b", &[("a", Type::Complex), ("b", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.contains("*"));
    }

    #[test]
    fn test_re() {
        let result = emit("re(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains(".re"));
    }

    #[test]
    fn test_im() {
        let result = emit("im(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains(".im"));
    }

    #[test]
    fn test_abs() {
        let result = emit("abs(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains(".norm()"));
    }

    #[test]
    fn test_conj() {
        let result = emit("conj(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.contains(".conj()"));
    }

    #[test]
    fn test_exp() {
        let result = emit("exp(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.contains(".exp()"));
    }

    #[test]
    fn test_let_in_fn() {
        let expr = Expr::parse("let w = z * z; w + z").unwrap();
        let code = emit_rust_fn(
            "square_add",
            expr.ast(),
            &[("z", Type::Complex)],
            Type::Complex,
        )
        .unwrap();
        assert!(code.contains("let w: Complex32"));
        assert!(code.contains("(w + z)"));
    }

    #[test]
    fn test_polar() {
        let result = emit(
            "polar(r, theta)",
            &[("r", Type::Scalar), ("theta", Type::Scalar)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.contains("Complex32::from_polar"));
    }
}
