//! TokenStream code generation for linalg expressions.
//!
//! Generates `proc_macro2::TokenStream` for use in proc-macro derives.
//! Emits code compatible with glam types (Vec2, Vec3, Vec4, Mat2, Mat3, Mat4).

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
    UnsupportedType(Type),
    /// Conditionals require scalar types.
    UnsupportedTypeForConditional(Type),
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
            TokenStreamError::UnsupportedType(t) => write!(f, "unsupported type: {t}"),
            TokenStreamError::UnsupportedTypeForConditional(t) => {
                write!(f, "conditionals require scalar type, got {t}")
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

            // Generate type annotation based on the inferred type
            let type_annotation = match value_expr.typ {
                Type::Scalar => quote! { f32 },
                Type::Vec2 => quote! { Vec2 },
                #[cfg(feature = "3d")]
                Type::Vec3 => quote! { Vec3 },
                #[cfg(feature = "4d")]
                Type::Vec4 => quote! { Vec4 },
                Type::Mat2 => quote! { Mat2 },
                #[cfg(feature = "3d")]
                Type::Mat3 => quote! { Mat3 },
                #[cfg(feature = "4d")]
                Type::Mat4 => quote! { Mat4 },
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
    let l = &left.code;
    let r = &right.code;

    let result_type = infer_binop_type(op, left.typ, right.typ)?;

    let code = match op {
        BinOp::Add => quote! { (#l + #r) },
        BinOp::Sub => quote! { (#l - #r) },
        BinOp::Mul => quote! { (#l * #r) },
        BinOp::Div => quote! { (#l / #r) },
        BinOp::Pow => quote! { #l.powf(#r) },
        BinOp::Rem => quote! { (#l % #r) },
        BinOp::BitAnd => quote! { (#l & #r) },
        BinOp::BitOr => quote! { (#l | #r) },
        BinOp::Shl => quote! { (#l << #r) },
        BinOp::Shr => quote! { (#l >> #r) },
    };

    Ok(TokenStreamExpr {
        code,
        typ: result_type,
    })
}

fn infer_binop_type(op: BinOp, left: Type, right: Type) -> Result<Type, TokenStreamError> {
    match op {
        BinOp::Add | BinOp::Sub => {
            if left == right {
                Ok(left)
            } else {
                Err(TokenStreamError::TypeMismatch {
                    op: if op == BinOp::Add { "+" } else { "-" },
                    left,
                    right,
                })
            }
        }
        BinOp::Mul => infer_mul_type(left, right),
        BinOp::Div => match (left, right) {
            (Type::Scalar, Type::Scalar) => Ok(Type::Scalar),
            (Type::Vec2, Type::Scalar) => Ok(Type::Vec2),
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Scalar) => Ok(Type::Vec3),
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Scalar) => Ok(Type::Vec4),
            _ => Err(TokenStreamError::TypeMismatch {
                op: "/",
                left,
                right,
            }),
        },
        BinOp::Pow => {
            if left == Type::Scalar && right == Type::Scalar {
                Ok(Type::Scalar)
            } else {
                Err(TokenStreamError::TypeMismatch {
                    op: "^",
                    left,
                    right,
                })
            }
        }
        BinOp::Rem | BinOp::BitAnd | BinOp::BitOr | BinOp::Shl | BinOp::Shr => {
            if left == Type::Scalar && right == Type::Scalar {
                Ok(Type::Scalar)
            } else {
                let op_str = match op {
                    BinOp::Rem => "%",
                    BinOp::BitAnd => "&",
                    BinOp::BitOr => "|",
                    BinOp::Shl => "<<",
                    BinOp::Shr => ">>",
                    _ => unreachable!(),
                };
                Err(TokenStreamError::TypeMismatch {
                    op: op_str,
                    left,
                    right,
                })
            }
        }
    }
}

fn infer_mul_type(left: Type, right: Type) -> Result<Type, TokenStreamError> {
    match (left, right) {
        (Type::Scalar, Type::Scalar) => Ok(Type::Scalar),

        (Type::Vec2, Type::Scalar) | (Type::Scalar, Type::Vec2) => Ok(Type::Vec2),
        #[cfg(feature = "3d")]
        (Type::Vec3, Type::Scalar) | (Type::Scalar, Type::Vec3) => Ok(Type::Vec3),
        #[cfg(feature = "4d")]
        (Type::Vec4, Type::Scalar) | (Type::Scalar, Type::Vec4) => Ok(Type::Vec4),

        (Type::Mat2, Type::Scalar) | (Type::Scalar, Type::Mat2) => Ok(Type::Mat2),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Scalar) | (Type::Scalar, Type::Mat3) => Ok(Type::Mat3),
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Scalar) | (Type::Scalar, Type::Mat4) => Ok(Type::Mat4),

        (Type::Mat2, Type::Vec2) => Ok(Type::Vec2),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Vec3) => Ok(Type::Vec3),
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Vec4) => Ok(Type::Vec4),

        (Type::Vec2, Type::Mat2) => Ok(Type::Vec2),
        #[cfg(feature = "3d")]
        (Type::Vec3, Type::Mat3) => Ok(Type::Vec3),
        #[cfg(feature = "4d")]
        (Type::Vec4, Type::Mat4) => Ok(Type::Vec4),

        (Type::Mat2, Type::Mat2) => Ok(Type::Mat2),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Mat3) => Ok(Type::Mat3),
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Mat4) => Ok(Type::Mat4),

        _ => Err(TokenStreamError::TypeMismatch {
            op: "*",
            left,
            right,
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
        UnaryOp::BitNot => {
            if inner.typ != Type::Scalar {
                return Err(TokenStreamError::UnsupportedType(inner.typ));
            }
            Ok(TokenStreamExpr {
                code: quote! { (!#inner_code) },
                typ: Type::Scalar,
            })
        }
    }
}

fn emit_function_call(
    name: &str,
    args: Vec<TokenStreamExpr>,
) -> Result<TokenStreamExpr, TokenStreamError> {
    match name {
        // ====================================================================
        // Vector functions
        // ====================================================================
        "dot" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { #a.dot(#b) },
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "3d")]
        "cross" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { #a.cross(#b) },
                typ: Type::Vec3,
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

        "distance" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { #a.distance(#b) },
                typ: Type::Scalar,
            })
        }

        "reflect" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            // glam doesn't have reflect, implement manually: v - 2.0 * dot(v, n) * n
            let v = &args[0].code;
            let n = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { (#v - 2.0 * #v.dot(#n) * #n) },
                typ: args[0].typ,
            })
        }

        "hadamard" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { (#a * #b) },
                typ: args[0].typ,
            })
        }

        "lerp" | "mix" => {
            if args.len() != 3 {
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

        // ====================================================================
        // Constructors
        // ====================================================================
        "vec2" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let x = &args[0].code;
            let y = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { Vec2::new(#x, #y) },
                typ: Type::Vec2,
            })
        }

        #[cfg(feature = "3d")]
        "vec3" => {
            if args.len() != 3 {
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

        #[cfg(feature = "4d")]
        "vec4" => {
            if args.len() != 4 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let x = &args[0].code;
            let y = &args[1].code;
            let z = &args[2].code;
            let w = &args[3].code;
            Ok(TokenStreamExpr {
                code: quote! { Vec4::new(#x, #y, #z, #w) },
                typ: Type::Vec4,
            })
        }

        // ====================================================================
        // Component extraction
        // ====================================================================
        "x" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #v.x },
                typ: Type::Scalar,
            })
        }

        "y" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #v.y },
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "3d")]
        "z" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #v.z },
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "4d")]
        "w" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #v.w },
                typ: Type::Scalar,
            })
        }

        // ====================================================================
        // Vectorized math functions (method syntax)
        // ====================================================================
        "sin" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            if args[0].typ == Type::Scalar {
                let v = &args[0].code;
                Ok(TokenStreamExpr {
                    code: quote! { #v.sin() },
                    typ: Type::Scalar,
                })
            } else {
                Err(TokenStreamError::UnsupportedType(args[0].typ))
            }
        }

        "cos" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            if args[0].typ == Type::Scalar {
                let v = &args[0].code;
                Ok(TokenStreamExpr {
                    code: quote! { #v.cos() },
                    typ: Type::Scalar,
                })
            } else {
                Err(TokenStreamError::UnsupportedType(args[0].typ))
            }
        }

        "abs" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #v.abs() },
                typ: args[0].typ,
            })
        }

        "floor" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #v.floor() },
                typ: args[0].typ,
            })
        }

        "fract" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            Ok(TokenStreamExpr {
                code: quote! { #v.fract() },
                typ: args[0].typ,
            })
        }

        "sqrt" => {
            if args.len() != 1 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            if args[0].typ == Type::Scalar {
                let v = &args[0].code;
                Ok(TokenStreamExpr {
                    code: quote! { #v.sqrt() },
                    typ: Type::Scalar,
                })
            } else {
                Err(TokenStreamError::UnsupportedType(args[0].typ))
            }
        }

        // ====================================================================
        // Vectorized comparison functions
        // ====================================================================
        "min" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { #a.min(#b) },
                typ: args[0].typ,
            })
        }

        "max" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { #a.max(#b) },
                typ: args[0].typ,
            })
        }

        "clamp" => {
            if args.len() != 3 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let x = &args[0].code;
            let lo = &args[1].code;
            let hi = &args[2].code;
            Ok(TokenStreamExpr {
                code: quote! { #x.clamp(#lo, #hi) },
                typ: args[0].typ,
            })
        }

        // ====================================================================
        // Interpolation functions
        // ====================================================================
        "step" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let edge = &args[0].code;
            let x = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! { (if #x < #edge { 0.0_f32 } else { 1.0_f32 }) },
                typ: Type::Scalar,
            })
        }

        "smoothstep" => {
            if args.len() != 3 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let e0 = &args[0].code;
            let e1 = &args[1].code;
            let x = &args[2].code;
            Ok(TokenStreamExpr {
                code: quote! {
                    {
                        let t = ((#x - #e0) / (#e1 - #e0)).clamp(0.0_f32, 1.0_f32);
                        t * t * (3.0_f32 - 2.0_f32 * t)
                    }
                },
                typ: Type::Scalar,
            })
        }

        // ====================================================================
        // Transform functions
        // ====================================================================
        "rotate2d" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! {
                    Vec2::new(
                        #v.x * #angle.cos() - #v.y * #angle.sin(),
                        #v.x * #angle.sin() + #v.y * #angle.cos()
                    )
                },
                typ: Type::Vec2,
            })
        }

        #[cfg(feature = "3d")]
        "rotate_x" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! {
                    Vec3::new(
                        #v.x,
                        #v.y * #angle.cos() - #v.z * #angle.sin(),
                        #v.y * #angle.sin() + #v.z * #angle.cos()
                    )
                },
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "3d")]
        "rotate_y" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! {
                    Vec3::new(
                        #v.x * #angle.cos() + #v.z * #angle.sin(),
                        #v.y,
                        -#v.x * #angle.sin() + #v.z * #angle.cos()
                    )
                },
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "3d")]
        "rotate_z" => {
            if args.len() != 2 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(TokenStreamExpr {
                code: quote! {
                    Vec3::new(
                        #v.x * #angle.cos() - #v.y * #angle.sin(),
                        #v.x * #angle.sin() + #v.y * #angle.cos(),
                        #v.z
                    )
                },
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "3d")]
        "rotate3d" => {
            if args.len() != 3 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            // Rodrigues' rotation formula
            let v = &args[0].code;
            let k = &args[1].code;
            let angle = &args[2].code;
            Ok(TokenStreamExpr {
                code: quote! {
                    (#v * #angle.cos() + #k.cross(#v) * #angle.sin() + #k * #k.dot(#v) * (1.0 - #angle.cos()))
                },
                typ: Type::Vec3,
            })
        }

        // ====================================================================
        // Matrix constructors
        // ====================================================================
        "mat2" => {
            if args.len() != 4 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            let c = &args[2].code;
            let d = &args[3].code;
            Ok(TokenStreamExpr {
                code: quote! { Mat2::from_cols(Vec2::new(#a, #b), Vec2::new(#c, #d)) },
                typ: Type::Mat2,
            })
        }

        #[cfg(feature = "3d")]
        "mat3" => {
            if args.len() != 9 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let (a0, a1, a2) = (&args[0].code, &args[1].code, &args[2].code);
            let (a3, a4, a5) = (&args[3].code, &args[4].code, &args[5].code);
            let (a6, a7, a8) = (&args[6].code, &args[7].code, &args[8].code);
            Ok(TokenStreamExpr {
                code: quote! {
                    Mat3::from_cols(
                        Vec3::new(#a0, #a1, #a2),
                        Vec3::new(#a3, #a4, #a5),
                        Vec3::new(#a6, #a7, #a8)
                    )
                },
                typ: Type::Mat3,
            })
        }

        #[cfg(feature = "4d")]
        "mat4" => {
            if args.len() != 16 {
                return Err(TokenStreamError::UnknownFunction(name.to_string()));
            }
            let (a0, a1, a2, a3) = (&args[0].code, &args[1].code, &args[2].code, &args[3].code);
            let (a4, a5, a6, a7) = (&args[4].code, &args[5].code, &args[6].code, &args[7].code);
            let (a8, a9, a10, a11) = (&args[8].code, &args[9].code, &args[10].code, &args[11].code);
            let (a12, a13, a14, a15) = (
                &args[12].code,
                &args[13].code,
                &args[14].code,
                &args[15].code,
            );
            Ok(TokenStreamExpr {
                code: quote! {
                    Mat4::from_cols(
                        Vec4::new(#a0, #a1, #a2, #a3),
                        Vec4::new(#a4, #a5, #a6, #a7),
                        Vec4::new(#a8, #a9, #a10, #a11),
                        Vec4::new(#a12, #a13, #a14, #a15)
                    )
                },
                typ: Type::Mat4,
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
    fn test_scalar_add() {
        let result = emit("a + b", &[("a", Type::Scalar), ("b", Type::Scalar)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.to_string().contains("+"));
    }

    #[test]
    fn test_vec2_add() {
        let result = emit("a + b", &[("a", Type::Vec2), ("b", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
    }

    #[test]
    fn test_mat_vec_mul() {
        let result = emit("m * v", &[("m", Type::Mat2), ("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
    }

    #[test]
    fn test_dot() {
        let result = emit("dot(a, b)", &[("a", Type::Vec2), ("b", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.to_string().contains(". dot"));
    }

    #[test]
    fn test_length() {
        let result = emit("length(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.to_string().contains(". length"));
    }

    #[test]
    fn test_normalize() {
        let result = emit("normalize(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
        assert!(result.code.to_string().contains(". normalize"));
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_cross() {
        let result = emit("cross(a, b)", &[("a", Type::Vec3), ("b", Type::Vec3)]).unwrap();
        assert_eq!(result.typ, Type::Vec3);
        assert!(result.code.to_string().contains(". cross"));
    }

    #[test]
    fn test_lerp() {
        let result = emit(
            "lerp(a, b, t)",
            &[("a", Type::Vec2), ("b", Type::Vec2), ("t", Type::Scalar)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Vec2);
        assert!(result.code.to_string().contains(". lerp"));
    }

    #[test]
    fn test_vec2_constructor() {
        let result = emit("vec2(x, y)", &[("x", Type::Scalar), ("y", Type::Scalar)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
        assert!(result.code.to_string().contains("Vec2"));
    }

    #[test]
    fn test_type_mismatch() {
        let result = emit("a + b", &[("a", Type::Scalar), ("b", Type::Vec2)]);
        assert!(matches!(result, Err(TokenStreamError::TypeMismatch { .. })));
    }

    #[test]
    fn test_component_extraction() {
        let result = emit("x(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.to_string().contains(". x"));
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_let() {
        let result = emit(
            "let hsl = v; vec3(x(hsl) + 0.1, y(hsl) * 1.2, z(hsl))",
            &[("v", Type::Vec3)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Vec3);
        let code = result.code.to_string();
        assert!(code.contains("let hsl"));
        assert!(code.contains("Vec3"));
    }
}
