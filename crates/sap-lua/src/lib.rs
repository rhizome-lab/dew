//! Lua code generation backend for sap expressions.
//!
//! Compiles expression ASTs to Lua code and optionally executes via mlua.
//!
//! # Example
//!
//! ```
//! use rhizome_sap_core::Expr;
//! use rhizome_sap_lua::{to_lua, LuaRegistry};
//!
//! let expr = Expr::parse("sin(x) + y * 2").unwrap();
//! let registry = LuaRegistry::new();
//! let lua_code = to_lua(expr.ast(), &registry);
//! ```

use rhizome_sap_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

// ============================================================================
// Lua Function Registry
// ============================================================================

/// A function that can be emitted as Lua code.
pub trait LuaFn: Send + Sync {
    /// Function name in the expression language.
    fn name(&self) -> &str;

    /// Emit Lua code for this function call.
    fn emit(&self, args: &[String]) -> String;
}

/// Registry of Lua function implementations.
#[derive(Default)]
pub struct LuaRegistry {
    funcs: HashMap<String, Box<dyn LuaFn>>,
}

impl LuaRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a function.
    pub fn register<F: LuaFn + 'static>(&mut self, func: F) {
        self.funcs.insert(func.name().to_string(), Box::new(func));
    }

    /// Gets a function by name.
    pub fn get(&self, name: &str) -> Option<&dyn LuaFn> {
        self.funcs.get(name).map(|f| f.as_ref())
    }
}

// ============================================================================
// Code generation
// ============================================================================

/// Compiles an AST to a Lua expression string.
pub fn to_lua(ast: &Ast, registry: &LuaRegistry) -> String {
    emit(ast, registry)
}

/// Generates a complete Lua function from an expression.
pub fn to_lua_fn(name: &str, ast: &Ast, params: &[&str], registry: &LuaRegistry) -> String {
    let param_list = params.join(", ");
    let body = emit(ast, registry);
    format!(
        "function {}({})\n    return {}\nend",
        name, param_list, body
    )
}

fn emit(ast: &Ast, registry: &LuaRegistry) -> String {
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
                BinOp::Pow => "^",
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
                // Unknown function - emit as math.name() for Lua stdlib
                format!("math.{}({})", name, args_str.join(", "))
            }
        }
    }
}

fn emit_with_parens(
    ast: &Ast,
    parent_op: Option<BinOp>,
    is_left: bool,
    registry: &LuaRegistry,
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
    if n.fract() == 0.0 && n.abs() < 1e10 {
        format!("{:.1}", n)
    } else {
        format!("{}", n)
    }
}

// ============================================================================
// Evaluation via mlua
// ============================================================================

/// Compiles and evaluates an expression with mlua.
///
/// Uses Lua's `math` library for unknown functions (sin, cos, etc.).
///
/// # Example
///
/// ```
/// use rhizome_sap_core::Expr;
/// use rhizome_sap_lua::eval;
/// use std::collections::HashMap;
///
/// let expr = Expr::parse("sin(x) + 1").unwrap();
/// let vars: HashMap<String, f32> = [("x".to_string(), 0.0)].into();
/// let result = eval(expr.ast(), &vars).unwrap();
/// assert!((result - 1.0).abs() < 0.001);
/// ```
pub fn eval(ast: &Ast, vars: &HashMap<String, f32>) -> Result<f32, mlua::Error> {
    eval_with_registry(ast, vars, &LuaRegistry::new())
}

/// Compiles and evaluates an expression with a custom function registry.
pub fn eval_with_registry(
    ast: &Ast,
    vars: &HashMap<String, f32>,
    registry: &LuaRegistry,
) -> Result<f32, mlua::Error> {
    let lua = mlua::Lua::new();
    let globals = lua.globals();

    // Set variables
    for (name, value) in vars {
        globals.set(name.as_str(), *value)?;
    }

    let code = emit(ast, registry);
    lua.load(format!("return {}", code)).eval()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_sap_core::Expr;

    fn compile(input: &str) -> String {
        let expr = Expr::parse(input).unwrap();
        let registry = LuaRegistry::new();
        to_lua(expr.ast(), &registry)
    }

    #[test]
    fn test_number() {
        assert_eq!(compile("42"), "42.0");
    }

    #[test]
    fn test_binary_ops() {
        assert_eq!(compile("1 + 2"), "1.0 + 2.0");
        assert_eq!(compile("2 ^ 3"), "2.0 ^ 3.0"); // Lua has native ^
    }

    #[test]
    fn test_precedence() {
        assert_eq!(compile("1 + 2 * 3"), "1.0 + 2.0 * 3.0");
        assert_eq!(compile("(1 + 2) * 3"), "(1.0 + 2.0) * 3.0");
    }

    #[test]
    fn test_function_call() {
        // Without registry, functions become math.name()
        assert_eq!(compile("sin(x)"), "math.sin(x)");
    }

    #[test]
    fn test_lua_fn() {
        let expr = Expr::parse("x + y").unwrap();
        let registry = LuaRegistry::new();
        let code = to_lua_fn("add", expr.ast(), &["x", "y"], &registry);
        assert!(code.contains("function add(x, y)"));
        assert!(code.contains("return x + y"));
    }

    #[test]
    fn test_eval() {
        let expr = Expr::parse("x * 2 + 1").unwrap();
        let vars: HashMap<String, f32> = [("x".to_string(), 3.0)].into();
        let result = eval(expr.ast(), &vars).unwrap();
        assert_eq!(result, 7.0);
    }

    #[test]
    fn test_eval_math() {
        let expr = Expr::parse("sin(0) + cos(0)").unwrap();
        let vars = HashMap::new();
        let result = eval(expr.ast(), &vars).unwrap();
        assert!((result - 1.0).abs() < 0.001); // sin(0) + cos(0) = 0 + 1 = 1
    }
}
