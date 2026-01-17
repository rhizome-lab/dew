//! GLSL code generation for complex expressions.
//!
//! Complex numbers are represented as vec2 where x=real, y=imag.

use crate::Type;
use rhizome_dew_cond::glsl as cond;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

/// Error during GLSL code generation.
#[derive(Debug, Clone, PartialEq)]
pub enum GlslError {
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
}

impl std::fmt::Display for GlslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlslError::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            GlslError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            GlslError::TypeMismatch { op, left, right } => {
                write!(f, "type mismatch for {op}: {left} vs {right}")
            }
            GlslError::UnsupportedTypeForConditional(t) => {
                write!(f, "conditionals require scalar type, got {t}")
            }
            GlslError::UnsupportedOperation(op) => {
                write!(f, "unsupported operation for complex: {op}")
            }
        }
    }
}

impl std::error::Error for GlslError {}

/// Convert a Type to its GLSL representation.
pub fn type_to_glsl(t: Type) -> &'static str {
    match t {
        Type::Scalar => "float",
        Type::Complex => "vec2",
    }
}

/// Result of GLSL emission: code string and its type.
pub struct GlslExpr {
    pub code: String,
    pub typ: Type,
}

/// Emit GLSL code for an AST with type propagation.
pub fn emit_glsl(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<GlslExpr, GlslError> {
    match ast {
        Ast::Num(n) => Ok(GlslExpr {
            code: format!("{n:.10}"),
            typ: Type::Scalar,
        }),

        Ast::Var(name) => {
            let typ = var_types
                .get(name)
                .copied()
                .ok_or_else(|| GlslError::UnknownVariable(name.clone()))?;
            Ok(GlslExpr {
                code: name.clone(),
                typ,
            })
        }

        Ast::BinOp(op, left, right) => {
            let left_expr = emit_glsl(left, var_types)?;
            let right_expr = emit_glsl(right, var_types)?;
            emit_binop(*op, left_expr, right_expr)
        }

        Ast::UnaryOp(op, inner) => {
            let inner_expr = emit_glsl(inner, var_types)?;
            emit_unaryop(*op, inner_expr)
        }

        Ast::Compare(op, left, right) => {
            let left_expr = emit_glsl(left, var_types)?;
            let right_expr = emit_glsl(right, var_types)?;
            // Comparisons only supported for scalars
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(GlslError::UnsupportedTypeForConditional(Type::Complex));
            }
            let bool_expr = cond::emit_compare(*op, &left_expr.code, &right_expr.code);
            Ok(GlslExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::And(left, right) => {
            let left_expr = emit_glsl(left, var_types)?;
            let right_expr = emit_glsl(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(GlslError::UnsupportedTypeForConditional(Type::Complex));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_and(&l_bool, &r_bool);
            Ok(GlslExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::Or(left, right) => {
            let left_expr = emit_glsl(left, var_types)?;
            let right_expr = emit_glsl(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(GlslError::UnsupportedTypeForConditional(Type::Complex));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_or(&l_bool, &r_bool);
            Ok(GlslExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::If(cond_ast, then_ast, else_ast) => {
            let cond_expr = emit_glsl(cond_ast, var_types)?;
            let then_expr = emit_glsl(then_ast, var_types)?;
            let else_expr = emit_glsl(else_ast, var_types)?;
            if cond_expr.typ != Type::Scalar {
                return Err(GlslError::UnsupportedTypeForConditional(cond_expr.typ));
            }
            // then and else can be any matching types
            if then_expr.typ != else_expr.typ {
                return Err(GlslError::TypeMismatch {
                    op: "if/else",
                    left: then_expr.typ,
                    right: else_expr.typ,
                });
            }
            let cond_bool = cond::scalar_to_bool(&cond_expr.code);
            Ok(GlslExpr {
                code: cond::emit_if(&cond_bool, &then_expr.code, &else_expr.code),
                typ: then_expr.typ,
            })
        }

        Ast::Call(name, args) => {
            let arg_exprs: Vec<GlslExpr> = args
                .iter()
                .map(|a| emit_glsl(a, var_types))
                .collect::<Result<_, _>>()?;
            emit_function_call(name, arg_exprs)
        }
    }
}

fn emit_binop(op: BinOp, left: GlslExpr, right: GlslExpr) -> Result<GlslExpr, GlslError> {
    match op {
        BinOp::Add => emit_add(left, right),
        BinOp::Sub => emit_sub(left, right),
        BinOp::Mul => emit_mul(left, right),
        BinOp::Div => emit_div(left, right),
        BinOp::Pow => emit_pow(left, right),
        // Bitwise ops not supported for complex numbers
        BinOp::Rem => Err(GlslError::UnsupportedOperation("%")),
        BinOp::BitAnd => Err(GlslError::UnsupportedOperation("&")),
        BinOp::BitOr => Err(GlslError::UnsupportedOperation("|")),
        BinOp::Shl => Err(GlslError::UnsupportedOperation("<<")),
        BinOp::Shr => Err(GlslError::UnsupportedOperation(">>")),
    }
}

fn emit_add(left: GlslExpr, right: GlslExpr) -> Result<GlslExpr, GlslError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) | (Type::Complex, Type::Complex) => Ok(GlslExpr {
            code: format!("({} + {})", left.code, right.code),
            typ: left.typ,
        }),
        _ => Err(GlslError::TypeMismatch {
            op: "+",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_sub(left: GlslExpr, right: GlslExpr) -> Result<GlslExpr, GlslError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) | (Type::Complex, Type::Complex) => Ok(GlslExpr {
            code: format!("({} - {})", left.code, right.code),
            typ: left.typ,
        }),
        _ => Err(GlslError::TypeMismatch {
            op: "-",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_mul(left: GlslExpr, right: GlslExpr) -> Result<GlslExpr, GlslError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(GlslExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        // Scalar * Complex
        (Type::Scalar, Type::Complex) => Ok(GlslExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Complex,
        }),
        (Type::Complex, Type::Scalar) => Ok(GlslExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Complex,
        }),
        // Complex * Complex: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        (Type::Complex, Type::Complex) => Ok(GlslExpr {
            code: format!(
                "vec2({l}.x * {r}.x - {l}.y * {r}.y, {l}.x * {r}.y + {l}.y * {r}.x)",
                l = left.code,
                r = right.code
            ),
            typ: Type::Complex,
        }),
    }
}

fn emit_div(left: GlslExpr, right: GlslExpr) -> Result<GlslExpr, GlslError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(GlslExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        // Complex / Scalar
        (Type::Complex, Type::Scalar) => Ok(GlslExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Complex,
        }),
        // Complex / Complex: multiply by conjugate
        // (a+bi)/(c+di) = (a+bi)(c-di) / (c²+d²)
        (Type::Complex, Type::Complex) => Ok(GlslExpr {
            code: format!(
                "(vec2({l}.x * {r}.x + {l}.y * {r}.y, {l}.y * {r}.x - {l}.x * {r}.y) / ({r}.x * {r}.x + {r}.y * {r}.y))",
                l = left.code,
                r = right.code
            ),
            typ: Type::Complex,
        }),
        _ => Err(GlslError::TypeMismatch {
            op: "/",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_pow(base: GlslExpr, exp: GlslExpr) -> Result<GlslExpr, GlslError> {
    match (base.typ, exp.typ) {
        (Type::Scalar, Type::Scalar) => Ok(GlslExpr {
            code: format!("pow({}, {})", base.code, exp.code),
            typ: Type::Scalar,
        }),
        // Complex^Scalar using polar form: r^n * (cos(n*theta) + i*sin(n*theta))
        (Type::Complex, Type::Scalar) => Ok(GlslExpr {
            code: format!(
                "(pow(length({z}), {n}) * vec2(cos(atan({z}.y, {z}.x) * {n}), sin(atan({z}.y, {z}.x) * {n})))",
                z = base.code,
                n = exp.code
            ),
            typ: Type::Complex,
        }),
        _ => Err(GlslError::TypeMismatch {
            op: "^",
            left: base.typ,
            right: exp.typ,
        }),
    }
}

fn emit_unaryop(op: UnaryOp, inner: GlslExpr) -> Result<GlslExpr, GlslError> {
    match op {
        UnaryOp::Neg => Ok(GlslExpr {
            code: format!("(-{})", inner.code),
            typ: inner.typ,
        }),
        UnaryOp::Not => {
            if inner.typ != Type::Scalar {
                return Err(GlslError::UnsupportedTypeForConditional(inner.typ));
            }
            let bool_expr = cond::scalar_to_bool(&inner.code);
            Ok(GlslExpr {
                code: cond::bool_to_scalar(&cond::emit_not(&bool_expr)),
                typ: Type::Scalar,
            })
        }
        UnaryOp::BitNot => Err(GlslError::UnsupportedOperation("~")),
    }
}

fn emit_function_call(name: &str, args: Vec<GlslExpr>) -> Result<GlslExpr, GlslError> {
    match name {
        "re" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("{}.x", args[0].code),
                typ: Type::Scalar,
            })
        }

        "im" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("{}.y", args[0].code),
                typ: Type::Scalar,
            })
        }

        "conj" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("vec2({}.x, -{}.y)", args[0].code, args[0].code),
                typ: Type::Complex,
            })
        }

        "abs" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(GlslExpr {
                    code: format!("abs({})", args[0].code),
                    typ: Type::Scalar,
                }),
                Type::Complex => Ok(GlslExpr {
                    code: format!("length({})", args[0].code),
                    typ: Type::Scalar,
                }),
            }
        }

        "arg" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("atan({}.y, {}.x)", args[0].code, args[0].code),
                typ: Type::Scalar,
            })
        }

        "norm" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("dot({z}, {z})", z = args[0].code),
                typ: Type::Scalar,
            })
        }

        "exp" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(GlslExpr {
                    code: format!("exp({})", args[0].code),
                    typ: Type::Scalar,
                }),
                // exp(a+bi) = e^a * (cos(b) + i*sin(b))
                Type::Complex => Ok(GlslExpr {
                    code: format!(
                        "(exp({z}.x) * vec2(cos({z}.y), sin({z}.y)))",
                        z = args[0].code
                    ),
                    typ: Type::Complex,
                }),
            }
        }

        "log" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(GlslExpr {
                    code: format!("log({})", args[0].code),
                    typ: Type::Scalar,
                }),
                // log(z) = log|z| + i*arg(z)
                Type::Complex => Ok(GlslExpr {
                    code: format!(
                        "vec2(log(length({z})), atan({z}.y, {z}.x))",
                        z = args[0].code
                    ),
                    typ: Type::Complex,
                }),
            }
        }

        "sqrt" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(GlslExpr {
                    code: format!("sqrt({})", args[0].code),
                    typ: Type::Scalar,
                }),
                // sqrt(z) = sqrt(|z|) * (cos(arg/2) + i*sin(arg/2))
                Type::Complex => Ok(GlslExpr {
                    code: format!(
                        "(sqrt(length({z})) * vec2(cos(atan({z}.y, {z}.x) * 0.5), sin(atan({z}.y, {z}.x) * 0.5)))",
                        z = args[0].code
                    ),
                    typ: Type::Complex,
                }),
            }
        }

        "polar" => {
            if args.len() != 2 || args[0].typ != Type::Scalar || args[1].typ != Type::Scalar {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!(
                    "({r} * vec2(cos({t}), sin({t})))",
                    r = args[0].code,
                    t = args[1].code
                ),
                typ: Type::Complex,
            })
        }

        _ => Err(GlslError::UnknownFunction(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn emit(expr: &str, var_types: &[(&str, Type)]) -> Result<GlslExpr, GlslError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_glsl(expr.ast(), &types)
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
        // Should contain the complex multiplication formula
        assert!(result.code.contains(".x") && result.code.contains(".y"));
    }

    #[test]
    fn test_re() {
        let result = emit("re(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains(".x"));
    }

    #[test]
    fn test_abs() {
        let result = emit("abs(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("length"));
    }

    #[test]
    fn test_conj() {
        let result = emit("conj(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        // GLSL uses vec2, not vec2<f32>
        assert!(result.code.contains("vec2("));
        assert!(!result.code.contains("<f32>"));
    }

    #[test]
    fn test_exp() {
        let result = emit("exp(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.contains("cos") && result.code.contains("sin"));
    }
}
