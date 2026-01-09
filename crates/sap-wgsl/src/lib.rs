//! WGSL code generation backend for sap expressions.
//!
//! Compiles expression ASTs to WGSL shader code.
//!
//! # Example
//!
//! ```
//! use rhizome_sap_core::Expr;
//! use rhizome_sap_wgsl::{to_wgsl, WgslRegistry};
//!
//! let expr = Expr::parse("sin(x) + y * 2").unwrap();
//! let registry = WgslRegistry::new(); // use sap_std::register_wgsl() to add std functions
//! let wgsl = to_wgsl(expr.ast(), &registry);
//! ```

use rhizome_sap_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

// ============================================================================
// WGSL Function Registry
// ============================================================================

/// A function that can be emitted as WGSL code.
pub trait WgslFn: Send + Sync {
    /// Function name in the expression language.
    fn name(&self) -> &str;

    /// Emit WGSL code for this function call.
    fn emit(&self, args: &[String]) -> String;
}

/// Registry of WGSL function implementations.
#[derive(Default)]
pub struct WgslRegistry {
    funcs: HashMap<String, Box<dyn WgslFn>>,
}

impl WgslRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a function.
    pub fn register<F: WgslFn + 'static>(&mut self, func: F) {
        self.funcs.insert(func.name().to_string(), Box::new(func));
    }

    /// Gets a function by name.
    pub fn get(&self, name: &str) -> Option<&dyn WgslFn> {
        self.funcs.get(name).map(|f| f.as_ref())
    }
}

// ============================================================================
// Code generation
// ============================================================================

/// Compiles an AST to a WGSL expression string.
pub fn to_wgsl(ast: &Ast, registry: &WgslRegistry) -> String {
    emit(ast, registry)
}

/// Generates a complete WGSL function from an expression.
pub fn to_wgsl_fn(name: &str, ast: &Ast, params: &[&str], registry: &WgslRegistry) -> String {
    let param_list: String = params
        .iter()
        .map(|p| format!("{}: f32", p))
        .collect::<Vec<_>>()
        .join(", ");

    let body = emit(ast, registry);
    format!(
        "fn {}({}) -> f32 {{\n    return {};\n}}",
        name, param_list, body
    )
}

fn emit(ast: &Ast, registry: &WgslRegistry) -> String {
    match ast {
        Ast::Num(n) => format_float(*n),
        Ast::Var(name) => name.clone(),
        Ast::BinOp(op, left, right) => {
            let l = emit_with_parens(left, Some(*op), true, registry);
            let r = emit_with_parens(right, Some(*op), false, registry);
            let op_str = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Pow => {
                    return format!("pow({}, {})", emit(left, registry), emit(right, registry));
                }
            };
            format!("{} {} {}", l, op_str, r)
        }
        Ast::UnaryOp(op, inner) => {
            let inner_str = emit_with_parens(inner, None, false, registry);
            match op {
                UnaryOp::Neg => format!("-{}", inner_str),
            }
        }
        Ast::Call(name, args) => {
            let args_str: Vec<String> = args.iter().map(|a| emit(a, registry)).collect();
            if let Some(func) = registry.get(name) {
                func.emit(&args_str)
            } else {
                // Unknown function - emit as-is (assume it's a WGSL builtin)
                format!("{}({})", name, args_str.join(", "))
            }
        }
    }
}

fn emit_with_parens(
    ast: &Ast,
    parent_op: Option<BinOp>,
    is_left: bool,
    registry: &WgslRegistry,
) -> String {
    let inner = emit(ast, registry);

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
        format!("({})", inner)
    } else {
        inner
    }
}

fn precedence(op: BinOp) -> u8 {
    match op {
        BinOp::Add | BinOp::Sub => 1,
        BinOp::Mul | BinOp::Div => 2,
        BinOp::Pow => 3,
    }
}

fn format_float(n: f32) -> String {
    if n.fract() == 0.0 {
        format!("{:.1}", n)
    } else {
        format!("{}", n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_sap_core::Expr;

    fn compile(input: &str) -> String {
        let expr = Expr::parse(input).unwrap();
        let registry = WgslRegistry::new();
        to_wgsl(expr.ast(), &registry)
    }

    #[test]
    fn test_number() {
        assert_eq!(compile("42"), "42.0");
    }

    #[test]
    fn test_binary_ops() {
        assert_eq!(compile("1 + 2"), "1.0 + 2.0");
        assert_eq!(compile("2 ^ 3"), "pow(2.0, 3.0)");
    }

    #[test]
    fn test_precedence() {
        assert_eq!(compile("1 + 2 * 3"), "1.0 + 2.0 * 3.0");
        assert_eq!(compile("(1 + 2) * 3"), "(1.0 + 2.0) * 3.0");
    }

    #[test]
    fn test_function_call() {
        // Without registry, functions emit as-is
        assert_eq!(compile("sin(x)"), "sin(x)");
    }

    #[test]
    fn test_wgsl_fn() {
        let expr = Expr::parse("x + y").unwrap();
        let registry = WgslRegistry::new();
        let code = to_wgsl_fn("add", expr.ast(), &["x", "y"], &registry);
        assert!(code.contains("fn add(x: f32, y: f32) -> f32"));
    }
}
