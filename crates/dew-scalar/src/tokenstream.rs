//! TokenStream code generation for scalar expressions.
//!
//! Generates `proc_macro2::TokenStream` for use in proc-macro derives.

use proc_macro2::TokenStream;
use quote::quote;
use rhizome_dew_cond::tokenstream as cond;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};

/// TokenStream emission error.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenStreamError {
    /// Unknown function.
    UnknownFunction(String),
    /// Feature not supported.
    UnsupportedFeature(String),
}

impl std::fmt::Display for TokenStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenStreamError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            TokenStreamError::UnsupportedFeature(feat) => {
                write!(f, "unsupported feature in TokenStream codegen: {feat}")
            }
        }
    }
}

impl std::error::Error for TokenStreamError {}

/// Emit TokenStream for an AST.
pub fn emit_tokenstream(ast: &Ast) -> Result<TokenStream, TokenStreamError> {
    emit(ast)
}

/// Simple emit for expression nodes.
fn emit(ast: &Ast) -> Result<TokenStream, TokenStreamError> {
    match ast {
        Ast::Num(n) => {
            let lit = *n as f32;
            Ok(quote! { #lit })
        }
        Ast::Var(name) => {
            let ident = syn_ident(name);
            Ok(quote! { #ident })
        }
        Ast::BinOp(op, left, right) => {
            let l = emit(left)?;
            let r = emit(right)?;
            Ok(match op {
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
            })
        }
        Ast::UnaryOp(op, inner) => {
            let inner_ts = emit(inner)?;
            Ok(match op {
                UnaryOp::Neg => quote! { (-#inner_ts) },
                UnaryOp::Not => {
                    let bool_expr = cond::scalar_to_bool(&inner_ts);
                    cond::bool_to_scalar(&cond::emit_not(&bool_expr))
                }
                UnaryOp::BitNot => quote! { (!#inner_ts) },
            })
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
            let args_ts: Vec<TokenStream> = args.iter().map(emit).collect::<Result<_, _>>()?;
            emit_call(name, &args_ts)
        }
        Ast::Let { name, value, body } => {
            let name_ident = syn_ident(name);
            let value_ts = emit(value)?;
            let body_ts = emit(body)?;
            Ok(quote! {
                {
                    let #name_ident: f32 = #value_ts;
                    #body_ts
                }
            })
        }
    }
}

fn emit_call(name: &str, args: &[TokenStream]) -> Result<TokenStream, TokenStreamError> {
    Ok(match name {
        // Constants
        "pi" => quote! { ::std::f32::consts::PI },
        "e" => quote! { ::std::f32::consts::E },
        "tau" => quote! { ::std::f32::consts::TAU },

        // Trig - method syntax
        "sin" => {
            let arg = &args[0];
            quote! { #arg.sin() }
        }
        "cos" => {
            let arg = &args[0];
            quote! { #arg.cos() }
        }
        "tan" => {
            let arg = &args[0];
            quote! { #arg.tan() }
        }
        "asin" => {
            let arg = &args[0];
            quote! { #arg.asin() }
        }
        "acos" => {
            let arg = &args[0];
            quote! { #arg.acos() }
        }
        "atan" => {
            let arg = &args[0];
            quote! { #arg.atan() }
        }
        "atan2" => {
            let y = &args[0];
            let x = &args[1];
            quote! { #y.atan2(#x) }
        }
        "sinh" => {
            let arg = &args[0];
            quote! { #arg.sinh() }
        }
        "cosh" => {
            let arg = &args[0];
            quote! { #arg.cosh() }
        }
        "tanh" => {
            let arg = &args[0];
            quote! { #arg.tanh() }
        }

        // Exp/log
        "exp" => {
            let arg = &args[0];
            quote! { #arg.exp() }
        }
        "exp2" => {
            let arg = &args[0];
            quote! { #arg.exp2() }
        }
        "log2" => {
            let arg = &args[0];
            quote! { #arg.log2() }
        }
        "pow" => {
            let base = &args[0];
            let exp = &args[1];
            quote! { #base.powf(#exp) }
        }
        "sqrt" => {
            let arg = &args[0];
            quote! { #arg.sqrt() }
        }
        "log" | "ln" => {
            let arg = &args[0];
            quote! { #arg.ln() }
        }
        "log10" => {
            let arg = &args[0];
            quote! { #arg.log10() }
        }
        "inversesqrt" => {
            let arg = &args[0];
            quote! { (1.0_f32 / #arg.sqrt()) }
        }

        // Common math
        "abs" => {
            let arg = &args[0];
            quote! { #arg.abs() }
        }
        "sign" => {
            let arg = &args[0];
            quote! { #arg.signum() }
        }
        "floor" => {
            let arg = &args[0];
            quote! { #arg.floor() }
        }
        "ceil" => {
            let arg = &args[0];
            quote! { #arg.ceil() }
        }
        "round" => {
            let arg = &args[0];
            quote! { #arg.round() }
        }
        "trunc" => {
            let arg = &args[0];
            quote! { #arg.trunc() }
        }
        "fract" => {
            let arg = &args[0];
            quote! { #arg.fract() }
        }
        "min" => {
            let a = &args[0];
            let b = &args[1];
            quote! { #a.min(#b) }
        }
        "max" => {
            let a = &args[0];
            let b = &args[1];
            quote! { #a.max(#b) }
        }
        "clamp" => {
            let x = &args[0];
            let lo = &args[1];
            let hi = &args[2];
            quote! { #x.clamp(#lo, #hi) }
        }
        "saturate" => {
            let arg = &args[0];
            quote! { #arg.clamp(0.0_f32, 1.0_f32) }
        }

        // Interpolation
        "lerp" | "mix" => {
            let a = &args[0];
            let b = &args[1];
            let t = &args[2];
            quote! { (#a + (#b - #a) * #t) }
        }
        "step" => {
            let edge = &args[0];
            let x = &args[1];
            quote! { (if #x < #edge { 0.0_f32 } else { 1.0_f32 }) }
        }
        "smoothstep" => {
            let e0 = &args[0];
            let e1 = &args[1];
            let x = &args[2];
            quote! {
                {
                    let t = ((#x - #e0) / (#e1 - #e0)).clamp(0.0_f32, 1.0_f32);
                    t * t * (3.0_f32 - 2.0_f32 * t)
                }
            }
        }
        "inverse_lerp" => {
            let a = &args[0];
            let b = &args[1];
            let v = &args[2];
            quote! { ((#v - #a) / (#b - #a)) }
        }
        "remap" => {
            let x = &args[0];
            let in_lo = &args[1];
            let in_hi = &args[2];
            let out_lo = &args[3];
            let out_hi = &args[4];
            quote! { (#out_lo + (#out_hi - #out_lo) * ((#x - #in_lo) / (#in_hi - #in_lo))) }
        }

        _ => return Err(TokenStreamError::UnknownFunction(name.to_string())),
    })
}

/// Create a syn-compatible identifier from a string.
fn syn_ident(name: &str) -> proc_macro2::Ident {
    proc_macro2::Ident::new(name, proc_macro2::Span::call_site())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn compile(input: &str) -> String {
        let expr = Expr::parse(input).unwrap();
        emit_tokenstream(expr.ast()).unwrap().to_string()
    }

    #[test]
    fn test_constants() {
        assert!(compile("pi()").contains("PI"));
        assert!(compile("e()").contains("E"));
        assert!(compile("tau()").contains("TAU"));
    }

    #[test]
    fn test_trig() {
        // quote! produces spaced tokens like "x . sin ()"
        assert!(compile("sin(x)").contains(". sin"));
        assert!(compile("cos(x)").contains(". cos"));
        assert!(compile("atan2(y, x)").contains(". atan2"));
    }

    #[test]
    fn test_exp_log() {
        assert!(compile("exp(x)").contains(". exp"));
        assert!(compile("ln(x)").contains(". ln"));
        assert!(compile("sqrt(x)").contains(". sqrt"));
        assert!(compile("pow(x, 2)").contains(". powf"));
    }

    #[test]
    fn test_common() {
        assert!(compile("abs(x)").contains(". abs"));
        assert!(compile("floor(x)").contains(". floor"));
        assert!(compile("clamp(x, 0, 1)").contains(". clamp"));
        assert!(compile("sign(x)").contains(". signum"));
    }

    #[test]
    fn test_interpolation() {
        let lerp = compile("lerp(a, b, t)");
        assert!(lerp.contains("a") && lerp.contains("b") && lerp.contains("t"));

        let step = compile("step(e, x)");
        assert!(step.contains("if"));
    }

    #[test]
    fn test_operators() {
        let add = compile("x + y");
        assert!(add.contains("+"));

        let pow = compile("x ^ 2");
        assert!(pow.contains(". powf"));
    }

    #[test]
    fn test_let() {
        let code = compile("let t = x * 2; t + t");
        assert!(code.contains("let t"));
    }

    #[test]
    fn test_compare() {
        let code = compile("x < y");
        assert!(code.contains("<"));
    }

    #[test]
    fn test_if_then_else() {
        let code = compile("if x > 0 then 1 else 0");
        assert!(code.contains("if"));
        assert!(code.contains("else"));
    }
}
