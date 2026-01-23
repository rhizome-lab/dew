//! TokenStream code generation for quaternion expressions.
//!
//! Generates `proc_macro2::TokenStream` for use in proc-macro derives.
//! Quaternions use glam::Quat, vectors use glam::Vec3.

use crate::Type;
use proc_macro2::TokenStream;
use quote::quote;
use rhizome_dew_cond::tokenstream as cond;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

/// Error during TokenStream code generation.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenStreamError {
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
    /// Feature not supported in TokenStream codegen.
    UnsupportedFeature(String),
}

impl std::fmt::Display for TokenStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenStreamError::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            TokenStreamError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            TokenStreamError::TypeMismatch { op, left, right } => {
                write!(f, "type mismatch for {op}: {left} vs {right}")
            }
            TokenStreamError::UnsupportedTypeForConditional(t) => {
                write!(f, "conditionals require scalar type, got {t}")
            }
            TokenStreamError::UnsupportedOperation(op) => {
                write!(f, "unsupported operation for quaternion: {op}")
            }
            TokenStreamError::UnsupportedFeature(feat) => {
                write!(f, "unsupported feature in TokenStream codegen: {feat}")
            }
        }
    }
}

impl std::error::Error for TokenStreamError {}

/// Result of TokenStream emission: code and its type.
pub struct TokenStreamExpr {
    pub code: TokenStream,
    pub typ: Type,
}

/// Create a syn-compatible identifier from a string.
fn syn_ident(name: &str) -> proc_macro2::Ident {
    proc_macro2::Ident::new(name, proc_macro2::Span::call_site())
}

/// Emit TokenStream for an AST with type propagation.
pub fn emit_tokenstream(
    ast: &Ast,
    var_types: &HashMap<String, Type>,
) -> Result<TokenStreamExpr, TokenStreamError> {
    match ast {
        Ast::Num(n) => {
            let lit = *n as f32;
            Ok(TokenStreamExpr {
                code: quote! { #lit },
                typ: Type::Scalar,
            })
        }

        Ast::Var(name) => {
            let typ = var_types
                .get(name)
                .copied()
                .ok_or_else(|| TokenStreamError::UnknownVariable(name.clone()))?;
            let ident = syn_ident(name);
            Ok(TokenStreamExpr {
                code: quote! { #ident },
                typ,
            })
        }

        Ast::BinOp(op, left, right) => {
            let left_expr = emit_tokenstream(left, var_types)?;
            let right_expr = emit_tokenstream(right, var_types)?;
            emit_binop(*op, left_expr, right_expr)
        }

        Ast::UnaryOp(op, inner) => {
            let inner_expr = emit_tokenstream(inner, var_types)?;
            emit_unaryop(*op, inner_expr)
        }

        Ast::Call(name, args) => {
            let arg_exprs: Vec<TokenStreamExpr> = args
                .iter()
                .map(|a| emit_tokenstream(a, var_types))
                .collect::<Result<_, _>>()?;
            emit_function_call(name, arg_exprs)
        }

        Ast::Compare(op, left, right) => {
            let left_expr = emit_tokenstream(left, var_types)?;
            let right_expr = emit_tokenstream(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(TokenStreamError::UnsupportedTypeForConditional(
                    left_expr.typ,
                ));
            }
            let bool_expr = cond::emit_compare(*op, &left_expr.code, &right_expr.code);
            Ok(TokenStreamExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::And(left, right) => {
            let left_expr = emit_tokenstream(left, var_types)?;
            let right_expr = emit_tokenstream(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(TokenStreamError::UnsupportedTypeForConditional(
                    left_expr.typ,
                ));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_and(&l_bool, &r_bool);
            Ok(TokenStreamExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::Or(left, right) => {
            let left_expr = emit_tokenstream(left, var_types)?;
            let right_expr = emit_tokenstream(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(TokenStreamError::UnsupportedTypeForConditional(
                    left_expr.typ,
                ));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_or(&l_bool, &r_bool);
            Ok(TokenStreamExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::If(cond_ast, then_ast, else_ast) => {
            let cond_expr = emit_tokenstream(cond_ast, var_types)?;
            let then_expr = emit_tokenstream(then_ast, var_types)?;
            let else_expr = emit_tokenstream(else_ast, var_types)?;
            if cond_expr.typ != Type::Scalar {
                return Err(TokenStreamError::UnsupportedTypeForConditional(
                    cond_expr.typ,
                ));
            }
            if then_expr.typ != else_expr.typ {
                return Err(TokenStreamError::TypeMismatch {
                    op: "if/else",
                    left: then_expr.typ,
                    right: else_expr.typ,
                });
            }
            let cond_bool = cond::scalar_to_bool(&cond_expr.code);
            Ok(TokenStreamExpr {
                code: cond::emit_if(&cond_bool, &then_expr.code, &else_expr.code),
                typ: then_expr.typ,
            })
        }

        Ast::Let { name, value, body } => {
            let value_expr = emit_tokenstream(value, var_types)?;

            let mut new_var_types = var_types.clone();
            new_var_types.insert(name.clone(), value_expr.typ);

            let body_expr = emit_tokenstream(body, &new_var_types)?;

            let name_ident = syn_ident(name);
            let value_code = value_expr.code;
            let body_code = body_expr.code;

            let type_annotation = match value_expr.typ {
                Type::Scalar => quote! { f32 },
                Type::Vec3 => quote! { Vec3 },
                Type::Quaternion => quote! { Quat },
            };

            Ok(TokenStreamExpr {
                code: quote! {
                    {
                        let #name_ident: #type_annotation = #value_code;
                        #body_code
                    }
                },
                typ: body_expr.typ,
            })
        }
    }
}

fn emit_binop(
    op: BinOp,
    left: TokenStreamExpr,
    right: TokenStreamExpr,
) -> Result<TokenStreamExpr, TokenStreamError> {
    match op {
        BinOp::Add => emit_add(left, right),
        BinOp::Sub => emit_sub(left, right),
        BinOp::Mul => emit_mul(left, right),
        BinOp::Div => emit_div(left, right),
        BinOp::Pow => emit_pow(left, right),
        BinOp::Rem => Err(TokenStreamError::UnsupportedOperation("%")),
        BinOp::BitAnd => Err(TokenStreamError::UnsupportedOperation("&")),
        BinOp::BitOr => Err(TokenStreamError::UnsupportedOperation("|")),
        BinOp::Shl => Err(TokenStreamError::UnsupportedOperation("<<")),
        BinOp::Shr => Err(TokenStreamError::UnsupportedOperation(">>")),
    }
}

fn emit_add(
    left: TokenStreamExpr,
    right: TokenStreamExpr,
) -> Result<TokenStreamExpr, TokenStreamError> {
    let l = &left.code;
    let r = &right.code;
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar)
        | (Type::Vec3, Type::Vec3)
        | (Type::Quaternion, Type::Quaternion) => Ok(TokenStreamExpr {
            code: quote! { (#l + #r) },
            typ: left.typ,
        }),
        _ => Err(TokenStreamError::TypeMismatch {
            op: "+",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_sub(
    left: TokenStreamExpr,
    right: TokenStreamExpr,
) -> Result<TokenStreamExpr, TokenStreamError> {
    let l = &left.code;
    let r = &right.code;
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar)
        | (Type::Vec3, Type::Vec3)
        | (Type::Quaternion, Type::Quaternion) => Ok(TokenStreamExpr {
            code: quote! { (#l - #r) },
            typ: left.typ,
        }),
        _ => Err(TokenStreamError::TypeMismatch {
            op: "-",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_mul(
    left: TokenStreamExpr,
    right: TokenStreamExpr,
) -> Result<TokenStreamExpr, TokenStreamError> {
    let l = &left.code;
    let r = &right.code;
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(TokenStreamExpr {
            code: quote! { (#l * #r) },
            typ: Type::Scalar,
        }),

        (Type::Scalar, Type::Vec3) | (Type::Vec3, Type::Scalar) => Ok(TokenStreamExpr {
            code: quote! { (#l * #r) },
            typ: Type::Vec3,
        }),

        (Type::Scalar, Type::Quaternion) | (Type::Quaternion, Type::Scalar) => {
            Ok(TokenStreamExpr {
                code: quote! { (#l * #r) },
                typ: Type::Quaternion,
            })
        }

        // Quaternion * Quaternion (Hamilton product)
        (Type::Quaternion, Type::Quaternion) => Ok(TokenStreamExpr {
            code: quote! { (#l * #r) },
            typ: Type::Quaternion,
        }),

        // Quaternion * Vec3 (rotate vector)
        (Type::Quaternion, Type::Vec3) => Ok(TokenStreamExpr {
            code: quote! { #l.mul_vec3(#r) },
            typ: Type::Vec3,
        }),

        _ => Err(TokenStreamError::TypeMismatch {
            op: "*",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_div(
    left: TokenStreamExpr,
    right: TokenStreamExpr,
) -> Result<TokenStreamExpr, TokenStreamError> {
    let l = &left.code;
    let r = &right.code;
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(TokenStreamExpr {
            code: quote! { (#l / #r) },
            typ: Type::Scalar,
        }),
        (Type::Vec3, Type::Scalar) => Ok(TokenStreamExpr {
            code: quote! { (#l / #r) },
            typ: Type::Vec3,
        }),
        (Type::Quaternion, Type::Scalar) => Ok(TokenStreamExpr {
            code: quote! { (#l / #r) },
            typ: Type::Quaternion,
        }),
        _ => Err(TokenStreamError::TypeMismatch {
            op: "/",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_pow(
    base: TokenStreamExpr,
    exp: TokenStreamExpr,
) -> Result<TokenStreamExpr, TokenStreamError> {
    let b = &base.code;
    let e = &exp.code;
    match (base.typ, exp.typ) {
        (Type::Scalar, Type::Scalar) => Ok(TokenStreamExpr {
            code: quote! { #b.powf(#e) },
            typ: Type::Scalar,
        }),
        _ => Err(TokenStreamError::TypeMismatch {
            op: "^",
            left: base.typ,
            right: exp.typ,
        }),
    }
}

fn emit_unaryop(op: UnaryOp, inner: TokenStreamExpr) -> Result<TokenStreamExpr, TokenStreamError> {
    let inner_code = &inner.code;

    match op {
        UnaryOp::Neg => Ok(TokenStreamExpr {
            code: quote! { (-#inner_code) },
            typ: inner.typ,
        }),
        UnaryOp::Not => {
            if inner.typ != Type::Scalar {
                return Err(TokenStreamError::UnsupportedTypeForConditional(inner.typ));
            }
            let bool_expr = cond::scalar_to_bool(inner_code);
            Ok(TokenStreamExpr {
                code: cond::bool_to_scalar(&cond::emit_not(&bool_expr)),
                typ: Type::Scalar,
            })
        }
        UnaryOp::BitNot => Err(TokenStreamError::UnsupportedOperation("~")),
    }
}

fn emit_function_call(
    name: &str,
    args: Vec<TokenStreamExpr>,
) -> Result<TokenStreamExpr, TokenStreamError> {
    match name {
        "conj" => {
            if args.len() != 1 || args[0].typ != Type::Quaternion {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let q = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #q.conjugate() },
                typ: Type::Quaternion,
            })
        }

        "length" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #v.length() },
                typ: Type::Scalar,
            })
        }

        "normalize" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #v.normalize() },
                typ: args[0].typ,
            })
        }

        "inverse" => {
            if args.len() != 1 || args[0].typ != Type::Quaternion {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let q = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #q.inverse() },
                typ: Type::Quaternion,
            })
        }

        "dot" => {
            if args.len() != 2 || args[0].typ != args[1].typ {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { #a.dot(#b) },
                typ: Type::Scalar,
            })
        }

        "lerp" => {
            if args.len() != 3 || args[2].typ != Type::Scalar {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            let t = &args[2].code;
            Ok(TokenStreamExpr {
                code: quote! { #a.lerp(#b, #t) },
                typ: args[0].typ,
            })
        }

        "slerp" => {
            if args.len() != 3
                || args[0].typ != Type::Quaternion
                || args[1].typ != Type::Quaternion
                || args[2].typ != Type::Scalar
            {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            let t = &args[2].code;
            Ok(TokenStreamExpr {
                code: quote! { #a.slerp(#b, #t) },
                typ: Type::Quaternion,
            })
        }

        "axis_angle" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Scalar {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let axis = &args[0].code;
            let angle = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { Quat::from_axis_angle(#axis, #angle) },
                typ: Type::Quaternion,
            })
        }

        "rotate" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Quaternion {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let q = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { #q.mul_vec3(#v) },
                typ: Type::Vec3,
            })
        }

        "cross" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Vec3 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { #a.cross(#b) },
                typ: Type::Vec3,
            })
        }

        "vec3" => {
            if args.len() != 3
                || args[0].typ != Type::Scalar
                || args[1].typ != Type::Scalar
                || args[2].typ != Type::Scalar
            {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let x = &args[0].code;
            let y = &args[1].code;
            let z = &args[2].code;
            Ok(TokenStreamExpr {
                code: quote! { Vec3::new(#x, #y, #z) },
                typ: Type::Vec3,
            })
        }

        "quat" => {
            if args.len() != 4 || args.iter().any(|a| a.typ != Type::Scalar) {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let x = &args[0].code;
            let y = &args[1].code;
            let z = &args[2].code;
            let w = &args[3].code;
            Ok(TokenStreamExpr {
                code: quote! { Quat::from_xyzw(#x, #y, #z, #w) },
                typ: Type::Quaternion,
            })
        }

        _ => Err(TokenStreamError::UnknownFunction(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn emit(expr: &str, var_types: &[(&str, Type)]) -> Result<TokenStreamExpr, TokenStreamError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_tokenstream(expr.ast(), &types)
    }

    #[test]
    fn test_quaternion_add() {
        let result = emit("a + b", &[("a", Type::Quaternion), ("b", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.to_string().contains("+"));
    }

    #[test]
    fn test_quaternion_mul() {
        let result = emit("a * b", &[("a", Type::Quaternion), ("b", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.to_string().contains("*"));
    }

    #[test]
    fn test_quaternion_rotate_vec() {
        let result = emit("q * v", &[("q", Type::Quaternion), ("v", Type::Vec3)]).unwrap();
        assert_eq!(result.typ, Type::Vec3);
        assert!(result.code.to_string().contains(". mul_vec3"));
    }

    #[test]
    fn test_normalize() {
        let result = emit("normalize(q)", &[("q", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.to_string().contains(". normalize"));
    }

    #[test]
    fn test_conj() {
        let result = emit("conj(q)", &[("q", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.to_string().contains(". conjugate"));
    }

    #[test]
    fn test_dot() {
        let result = emit(
            "dot(a, b)",
            &[("a", Type::Quaternion), ("b", Type::Quaternion)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.to_string().contains(". dot"));
    }

    #[test]
    fn test_axis_angle() {
        let result = emit(
            "axis_angle(v, a)",
            &[("v", Type::Vec3), ("a", Type::Scalar)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        assert!(result.code.to_string().contains("from_axis_angle"));
    }

    #[test]
    fn test_rotate() {
        let result = emit(
            "rotate(v, q)",
            &[("v", Type::Vec3), ("q", Type::Quaternion)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Vec3);
        assert!(result.code.to_string().contains(". mul_vec3"));
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
        assert!(result.code.to_string().contains(". slerp"));
    }

    #[test]
    fn test_let() {
        let result = emit("let sq = q * q; sq + q", &[("q", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        let code = result.code.to_string();
        assert!(code.contains("let sq"));
        assert!(code.contains("Quat"));
    }
}
