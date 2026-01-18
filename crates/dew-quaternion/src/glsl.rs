//! GLSL code generation for quaternion expressions.
//!
//! Quaternions are represented as vec4 where xyz=imaginary, w=real (xyzw order).
//! Vectors are vec3.

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
    /// Feature not supported in expression-only codegen.
    UnsupportedFeature(String),
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
                write!(f, "unsupported operation for quaternion: {op}")
            }
            GlslError::UnsupportedFeature(feat) => {
                write!(f, "unsupported feature in expression codegen: {feat}")
            }
        }
    }
}

impl std::error::Error for GlslError {}

/// Convert a Type to its GLSL representation.
pub fn type_to_glsl(t: Type) -> &'static str {
    match t {
        Type::Scalar => "float",
        Type::Vec3 => "vec3",
        Type::Quaternion => "vec4",
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

        Ast::Call(name, args) => {
            let arg_exprs: Vec<GlslExpr> = args
                .iter()
                .map(|a| emit_glsl(a, var_types))
                .collect::<Result<_, _>>()?;
            emit_function_call(name, arg_exprs)
        }

        Ast::Compare(op, left, right) => {
            let left_expr = emit_glsl(left, var_types)?;
            let right_expr = emit_glsl(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(GlslError::UnsupportedTypeForConditional(left_expr.typ));
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
                return Err(GlslError::UnsupportedTypeForConditional(left_expr.typ));
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
                return Err(GlslError::UnsupportedTypeForConditional(left_expr.typ));
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

        Ast::Let { .. } => {
            // Let bindings require statement-based codegen.
            // Use the LetInlining optimization pass before emitting.
            Err(GlslError::UnsupportedFeature(
                "let bindings (use LetInlining pass first)".to_string(),
            ))
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
        BinOp::Rem => Err(GlslError::UnsupportedOperation("%")),
        BinOp::BitAnd => Err(GlslError::UnsupportedOperation("&")),
        BinOp::BitOr => Err(GlslError::UnsupportedOperation("|")),
        BinOp::Shl => Err(GlslError::UnsupportedOperation("<<")),
        BinOp::Shr => Err(GlslError::UnsupportedOperation(">>")),
    }
}

fn emit_add(left: GlslExpr, right: GlslExpr) -> Result<GlslExpr, GlslError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar)
        | (Type::Vec3, Type::Vec3)
        | (Type::Quaternion, Type::Quaternion) => Ok(GlslExpr {
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
        (Type::Scalar, Type::Scalar)
        | (Type::Vec3, Type::Vec3)
        | (Type::Quaternion, Type::Quaternion) => Ok(GlslExpr {
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

        // Scalar * Vec3 / Vec3 * Scalar
        (Type::Scalar, Type::Vec3) | (Type::Vec3, Type::Scalar) => Ok(GlslExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Vec3,
        }),

        // Scalar * Quaternion / Quaternion * Scalar
        (Type::Scalar, Type::Quaternion) | (Type::Quaternion, Type::Scalar) => Ok(GlslExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Quaternion,
        }),

        // Quaternion * Quaternion (Hamilton product)
        // q1 = [x1, y1, z1, w1], q2 = [x2, y2, z2, w2]
        (Type::Quaternion, Type::Quaternion) => {
            let q1 = &left.code;
            let q2 = &right.code;
            Ok(GlslExpr {
                code: format!(
                    "vec4(\
                        {q1}.w * {q2}.x + {q1}.x * {q2}.w + {q1}.y * {q2}.z - {q1}.z * {q2}.y, \
                        {q1}.w * {q2}.y - {q1}.x * {q2}.z + {q1}.y * {q2}.w + {q1}.z * {q2}.x, \
                        {q1}.w * {q2}.z + {q1}.x * {q2}.y - {q1}.y * {q2}.x + {q1}.z * {q2}.w, \
                        {q1}.w * {q2}.w - {q1}.x * {q2}.x - {q1}.y * {q2}.y - {q1}.z * {q2}.z)"
                ),
                typ: Type::Quaternion,
            })
        }

        // Quaternion * Vec3 (rotate vector)
        // Using optimized formula: v' = v + 2w(q×v) + 2(q×(q×v))
        (Type::Quaternion, Type::Vec3) => {
            let q = &left.code;
            let v = &right.code;
            // In GLSL we use inline expression with let-style variables via temp computation
            // Using the formula directly inline
            Ok(GlslExpr {
                code: format!(
                    "({v} + {q}.w * (2.0 * cross({q}.xyz, {v})) + cross({q}.xyz, 2.0 * cross({q}.xyz, {v})))"
                ),
                typ: Type::Vec3,
            })
        }

        _ => Err(GlslError::TypeMismatch {
            op: "*",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_div(left: GlslExpr, right: GlslExpr) -> Result<GlslExpr, GlslError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(GlslExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        (Type::Vec3, Type::Scalar) => Ok(GlslExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Vec3,
        }),
        (Type::Quaternion, Type::Scalar) => Ok(GlslExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Quaternion,
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
        "conj" => {
            if args.len() != 1 || args[0].typ != Type::Quaternion {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("vec4(-{q}.xyz, {q}.w)", q = args[0].code),
                typ: Type::Quaternion,
            })
        }

        "length" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("length({})", args[0].code),
                typ: Type::Scalar,
            })
        }

        "normalize" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("normalize({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "inverse" => {
            if args.len() != 1 || args[0].typ != Type::Quaternion {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            // inverse(q) = conj(q) / |q|²
            let q = &args[0].code;
            Ok(GlslExpr {
                code: format!("(vec4(-{q}.xyz, {q}.w) / dot({q}, {q}))"),
                typ: Type::Quaternion,
            })
        }

        "dot" => {
            if args.len() != 2 || args[0].typ != args[1].typ {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("dot({}, {})", args[0].code, args[1].code),
                typ: Type::Scalar,
            })
        }

        "lerp" => {
            if args.len() != 3 || args[2].typ != Type::Scalar {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("mix({}, {}, {})", args[0].code, args[1].code, args[2].code),
                typ: args[0].typ,
            })
        }

        "slerp" => {
            if args.len() != 3
                || args[0].typ != Type::Quaternion
                || args[1].typ != Type::Quaternion
                || args[2].typ != Type::Scalar
            {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            let q1 = &args[0].code;
            let q2 = &args[1].code;
            let t = &args[2].code;
            // GLSL doesn't have IIFEs, so we use a ternary-based approach
            // For complex slerp, we'd need helper functions in practice
            // Here we do a simplified inline version
            Ok(GlslExpr {
                code: format!(
                    "(dot({q1}, {q2}) > 0.9995 \
                        ? normalize(mix({q1}, {q2}, {t})) \
                        : (({q1} * sin((1.0 - {t}) * acos(abs(dot({q1}, {q2})))) + \
                           (dot({q1}, {q2}) < 0.0 ? -{q2} : {q2}) * sin({t} * acos(abs(dot({q1}, {q2}))))) \
                           / sin(acos(abs(dot({q1}, {q2}))))))"
                ),
                typ: Type::Quaternion,
            })
        }

        "axis_angle" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Scalar {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            let axis = &args[0].code;
            let angle = &args[1].code;
            Ok(GlslExpr {
                code: format!("vec4(normalize({axis}) * sin({angle} * 0.5), cos({angle} * 0.5))"),
                typ: Type::Quaternion,
            })
        }

        "rotate" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Quaternion {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let q = &args[1].code;
            // Using optimized formula inline
            Ok(GlslExpr {
                code: format!(
                    "({v} + {q}.w * (2.0 * cross({q}.xyz, {v})) + cross({q}.xyz, 2.0 * cross({q}.xyz, {v})))"
                ),
                typ: Type::Vec3,
            })
        }

        "cross" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Vec3 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("cross({}, {})", args[0].code, args[1].code),
                typ: Type::Vec3,
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
    fn test_quaternion_add() {
        let result = emit("a + b", &[("a", Type::Quaternion), ("b", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.contains("+"));
    }

    #[test]
    fn test_quaternion_mul() {
        let result = emit("a * b", &[("a", Type::Quaternion), ("b", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        // Should contain Hamilton product
        assert!(result.code.contains(".w") && result.code.contains(".x"));
    }

    #[test]
    fn test_quaternion_rotate_vec() {
        let result = emit("q * v", &[("q", Type::Quaternion), ("v", Type::Vec3)]).unwrap();
        assert_eq!(result.typ, Type::Vec3);
        assert!(result.code.contains("cross"));
    }

    #[test]
    fn test_normalize() {
        let result = emit("normalize(q)", &[("q", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.contains("normalize"));
    }

    #[test]
    fn test_conj() {
        let result = emit("conj(q)", &[("q", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.contains("vec4(-"));
    }

    #[test]
    fn test_dot() {
        let result = emit(
            "dot(a, b)",
            &[("a", Type::Quaternion), ("b", Type::Quaternion)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("dot("));
    }

    #[test]
    fn test_axis_angle() {
        let result = emit(
            "axis_angle(v, a)",
            &[("v", Type::Vec3), ("a", Type::Scalar)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.contains("sin") && result.code.contains("cos"));
    }

    #[test]
    fn test_rotate() {
        let result = emit(
            "rotate(v, q)",
            &[("v", Type::Vec3), ("q", Type::Quaternion)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Vec3);
        assert!(result.code.contains("cross"));
    }

    #[test]
    fn test_lerp() {
        let result = emit(
            "lerp(a, b, t)",
            &[
                ("a", Type::Quaternion),
                ("b", Type::Quaternion),
                ("t", Type::Scalar),
            ],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.contains("mix("));
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
    }
}
