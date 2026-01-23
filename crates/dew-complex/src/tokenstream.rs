//! TokenStream code generation for complex expressions.
//!
//! Generates `proc_macro2::TokenStream` for use in proc-macro derives.
//! Complex numbers are represented using num_complex::Complex32 (Complex<f32>).

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
                write!(f, "unsupported operation for complex: {op}")
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

        Ast::Compare(op, left, right) => {
            let left_expr = emit_tokenstream(left, var_types)?;
            let right_expr = emit_tokenstream(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(TokenStreamError::UnsupportedTypeForConditional(
                    Type::Complex,
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
                    Type::Complex,
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
                    Type::Complex,
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

        Ast::Call(name, args) => {
            let arg_exprs: Vec<TokenStreamExpr> = args
                .iter()
                .map(|a| emit_tokenstream(a, var_types))
                .collect::<Result<_, _>>()?;
            emit_function_call(name, arg_exprs)
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
                Type::Complex => quote! { Complex32 },
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
        (Type::Scalar, Type::Scalar) | (Type::Complex, Type::Complex) => Ok(TokenStreamExpr {
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
        (Type::Scalar, Type::Scalar) | (Type::Complex, Type::Complex) => Ok(TokenStreamExpr {
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
        (Type::Scalar, Type::Complex)
        | (Type::Complex, Type::Scalar)
        | (Type::Complex, Type::Complex) => Ok(TokenStreamExpr {
            code: quote! { (#l * #r) },
            typ: Type::Complex,
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
        (Type::Complex, Type::Scalar) | (Type::Complex, Type::Complex) => Ok(TokenStreamExpr {
            code: quote! { (#l / #r) },
            typ: Type::Complex,
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
        (Type::Complex, Type::Scalar) => Ok(TokenStreamExpr {
            code: quote! { #b.powf(#e) },
            typ: Type::Complex,
        }),
        (Type::Complex, Type::Complex) => Ok(TokenStreamExpr {
            code: quote! { #b.powc(#e) },
            typ: Type::Complex,
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
        "re" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let z = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #z.re },
                typ: Type::Scalar,
            })
        }

        "im" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let z = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #z.im },
                typ: Type::Scalar,
            })
        }

        "conj" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let z = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #z.conj() },
                typ: Type::Complex,
            })
        }

        "abs" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            match args[0].typ {
                Type::Scalar => Ok(TokenStreamExpr {
                    code: quote! { #v.abs() },
                    typ: Type::Scalar,
                }),
                Type::Complex => Ok(TokenStreamExpr {
                    code: quote! { #v.norm() },
                    typ: Type::Scalar,
                }),
            }
        }

        "arg" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let z = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #z.arg() },
                typ: Type::Scalar,
            })
        }

        "norm" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let z = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #z.norm_sqr() },
                typ: Type::Scalar,
            })
        }

        "exp" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #v.exp() },
                typ: args[0].typ,
            })
        }

        "log" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #v.ln() },
                typ: args[0].typ,
            })
        }

        "sqrt" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #v.sqrt() },
                typ: args[0].typ,
            })
        }

        "polar" => {
            if args.len() != 2 || args[0].typ != Type::Scalar || args[1].typ != Type::Scalar {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let r = &args[0].code;
            let theta = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { Complex32::from_polar(#r, #theta) },
                typ: Type::Complex,
            })
        }

        "complex" => {
            if args.len() != 2 || args[0].typ != Type::Scalar || args[1].typ != Type::Scalar {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let re = &args[0].code;
            let im = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { Complex32::new(#re, #im) },
                typ: Type::Complex,
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
    fn test_complex_add() {
        let result = emit("a + b", &[("a", Type::Complex), ("b", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.to_string().contains("+"));
    }

    #[test]
    fn test_complex_mul() {
        let result = emit("a * b", &[("a", Type::Complex), ("b", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.to_string().contains("*"));
    }

    #[test]
    fn test_re() {
        let result = emit("re(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.to_string().contains(". re"));
    }

    #[test]
    fn test_im() {
        let result = emit("im(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.to_string().contains(". im"));
    }

    #[test]
    fn test_abs() {
        let result = emit("abs(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.to_string().contains(". norm"));
    }

    #[test]
    fn test_conj() {
        let result = emit("conj(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.to_string().contains(". conj"));
    }

    #[test]
    fn test_exp() {
        let result = emit("exp(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.to_string().contains(". exp"));
    }

    #[test]
    fn test_let() {
        let result = emit("let w = z * z; w + z", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        let code = result.code.to_string();
        assert!(code.contains("let w"));
        assert!(code.contains("Complex32"));
    }

    #[test]
    fn test_polar() {
        let result = emit(
            "polar(r, theta)",
            &[("r", Type::Scalar), ("theta", Type::Scalar)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.to_string().contains("from_polar"));
    }
}
