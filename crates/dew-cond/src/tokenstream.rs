//! TokenStream code generation helpers for conditionals.
//!
//! Generates `proc_macro2::TokenStream` for use in proc-macro derives.

use proc_macro2::TokenStream;
use quote::quote;
use rhizome_dew_core::CompareOp;

/// Emit TokenStream for a comparison operation.
/// Returns boolean expression as TokenStream.
pub fn emit_compare(op: CompareOp, left: &TokenStream, right: &TokenStream) -> TokenStream {
    match op {
        CompareOp::Lt => quote! { (#left < #right) },
        CompareOp::Le => quote! { (#left <= #right) },
        CompareOp::Gt => quote! { (#left > #right) },
        CompareOp::Ge => quote! { (#left >= #right) },
        CompareOp::Eq => quote! { (#left == #right) },
        CompareOp::Ne => quote! { (#left != #right) },
    }
}

/// Emit TokenStream for logical AND.
/// Inputs are boolean expressions.
pub fn emit_and(left: &TokenStream, right: &TokenStream) -> TokenStream {
    quote! { (#left && #right) }
}

/// Emit TokenStream for logical OR.
/// Inputs are boolean expressions.
pub fn emit_or(left: &TokenStream, right: &TokenStream) -> TokenStream {
    quote! { (#left || #right) }
}

/// Emit TokenStream for logical NOT.
/// Input is a boolean expression.
pub fn emit_not(expr: &TokenStream) -> TokenStream {
    quote! { (!#expr) }
}

/// Emit TokenStream for a conditional (if/then/else).
/// Uses Rust's if-expression syntax.
/// `cond` should be a boolean expression.
pub fn emit_if(
    cond: &TokenStream,
    then_expr: &TokenStream,
    else_expr: &TokenStream,
) -> TokenStream {
    quote! { (if #cond { #then_expr } else { #else_expr }) }
}

/// Convert a scalar (float) expression to boolean for use in conditions.
/// Non-zero is true, zero is false.
pub fn scalar_to_bool(expr: &TokenStream) -> TokenStream {
    quote! { (#expr != 0.0) }
}

/// Convert a boolean expression to scalar (float).
/// true -> 1.0, false -> 0.0
pub fn bool_to_scalar(expr: &TokenStream) -> TokenStream {
    quote! { (if #expr { 1.0_f32 } else { 0.0_f32 }) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_compare() {
        let a = quote! { a };
        let b = quote! { b };
        let result = emit_compare(CompareOp::Lt, &a, &b);
        assert_eq!(result.to_string(), "(a < b)");
    }

    #[test]
    fn test_emit_logic() {
        let a = quote! { a };
        let b = quote! { b };
        assert_eq!(emit_and(&a, &b).to_string(), "(a && b)");
        assert_eq!(emit_or(&a, &b).to_string(), "(a || b)");
        assert_eq!(emit_not(&a).to_string(), "(! a)");
    }

    #[test]
    fn test_emit_if() {
        let cond = quote! { cond };
        let then_val = quote! { then_val };
        let else_val = quote! { else_val };
        let result = emit_if(&cond, &then_val, &else_val);
        assert!(result.to_string().contains("if cond"));
    }

    #[test]
    fn test_conversions() {
        let x = quote! { x };
        let cond = quote! { cond };
        assert!(scalar_to_bool(&x).to_string().contains("!= 0.0"));
        assert!(bool_to_scalar(&cond).to_string().contains("1.0_f32"));
    }
}
