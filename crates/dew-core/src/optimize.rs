//! Expression optimization passes.
//!
//! This module provides infrastructure for transforming ASTs to reduce runtime computation.
//! Optimizations are applied before evaluation or code generation.
//!
//! # Example
//!
//! ```
//! use rhizome_dew_core::{Expr, Ast, BinOp};
//! use rhizome_dew_core::optimize::{optimize, standard_passes};
//!
//! let expr = Expr::parse("1 + 2 + x").unwrap();
//! let optimized = optimize(expr.ast().clone(), &standard_passes());
//!
//! // 1 + 2 was folded to 3
//! match &optimized {
//!     Ast::BinOp(BinOp::Add, left, right) => {
//!         assert!(matches!(left.as_ref(), Ast::Num(n) if (*n - 3.0).abs() < 0.001));
//!         assert!(matches!(right.as_ref(), Ast::Var(name) if name == "x"));
//!     }
//!     _ => panic!("unexpected structure"),
//! }
//! ```
//!
//! # Available Passes
//!
//! - [`ConstantFolding`]: Evaluates numeric operations at compile time
//! - [`AlgebraicIdentities`]: Eliminates identity operations (x*1, x+0, etc.)
//! - [`PowerReduction`]: Converts small powers to multiplication (x^2 → x*x)
//! - [`FunctionDecomposition`]: Expands functions using `ExprFn::decompose()`

use crate::{Ast, BinOp, UnaryOp};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[cfg(feature = "cond")]
use crate::CompareOp;

#[cfg(feature = "func")]
use crate::FunctionRegistry;

// ============================================================================
// Pass trait
// ============================================================================

/// A transformation pass that can simplify AST nodes.
///
/// Passes are applied bottom-up: children are transformed before parents.
/// Return `Some(new_ast)` to replace a node, `None` to keep the original.
///
/// # Example
///
/// ```
/// use rhizome_dew_core::{Ast, BinOp};
/// use rhizome_dew_core::optimize::Pass;
///
/// struct DoubleToAdd;
///
/// impl Pass for DoubleToAdd {
///     fn transform(&self, ast: &Ast) -> Option<Ast> {
///         // x * 2 → x + x
///         match ast {
///             Ast::BinOp(BinOp::Mul, left, right) => {
///                 if matches!(right.as_ref(), Ast::Num(n) if (*n - 2.0).abs() < 0.001) {
///                     Some(Ast::BinOp(BinOp::Add, left.clone(), left.clone()))
///                 } else {
///                     None
///                 }
///             }
///             _ => None,
///         }
///     }
/// }
/// ```
pub trait Pass: Send + Sync {
    /// Transform an AST node.
    ///
    /// Returns `Some(new_ast)` if the node should be replaced, `None` to keep original.
    fn transform(&self, ast: &Ast) -> Option<Ast>;
}

// ============================================================================
// Core optimization function
// ============================================================================

/// Applies optimization passes to an AST until no more changes occur.
///
/// Passes are applied bottom-up (children first, then parents) and the process
/// repeats until a fixed point is reached.
///
/// # Example
///
/// ```
/// use rhizome_dew_core::{Expr, Ast};
/// use rhizome_dew_core::optimize::{optimize, standard_passes};
///
/// let expr = Expr::parse("(1 + 2) * x + 0").unwrap();
/// let optimized = optimize(expr.ast().clone(), &standard_passes());
///
/// // Result: 3 * x (constant folded, identity eliminated)
/// assert_eq!(optimized.to_string(), "(3 * x)");
/// ```
pub fn optimize(ast: Ast, passes: &[&dyn Pass]) -> Ast {
    let mut current = ast;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 100; // Safety limit

    loop {
        let transformed = apply_passes_once(&current, passes);
        if transformed == current || iterations >= MAX_ITERATIONS {
            break;
        }
        current = transformed;
        iterations += 1;
    }

    current
}

/// Applies passes once, bottom-up, to the entire AST.
fn apply_passes_once(ast: &Ast, passes: &[&dyn Pass]) -> Ast {
    // First, recursively transform children
    let with_children = transform_children(ast, passes);

    // Then try each pass on the current node
    for pass in passes {
        if let Some(result) = pass.transform(&with_children) {
            return result;
        }
    }

    with_children
}

/// Recursively transforms children of an AST node.
fn transform_children(ast: &Ast, passes: &[&dyn Pass]) -> Ast {
    match ast {
        Ast::Num(n) => Ast::Num(*n),
        Ast::Var(name) => Ast::Var(name.clone()),
        Ast::BinOp(op, left, right) => Ast::BinOp(
            *op,
            Box::new(apply_passes_once(left, passes)),
            Box::new(apply_passes_once(right, passes)),
        ),
        Ast::UnaryOp(op, inner) => Ast::UnaryOp(*op, Box::new(apply_passes_once(inner, passes))),
        #[cfg(feature = "func")]
        Ast::Call(name, args) => Ast::Call(
            name.clone(),
            args.iter().map(|a| apply_passes_once(a, passes)).collect(),
        ),
        #[cfg(feature = "cond")]
        Ast::Compare(op, left, right) => Ast::Compare(
            *op,
            Box::new(apply_passes_once(left, passes)),
            Box::new(apply_passes_once(right, passes)),
        ),
        #[cfg(feature = "cond")]
        Ast::And(left, right) => Ast::And(
            Box::new(apply_passes_once(left, passes)),
            Box::new(apply_passes_once(right, passes)),
        ),
        #[cfg(feature = "cond")]
        Ast::Or(left, right) => Ast::Or(
            Box::new(apply_passes_once(left, passes)),
            Box::new(apply_passes_once(right, passes)),
        ),
        #[cfg(feature = "cond")]
        Ast::If(cond, then_expr, else_expr) => Ast::If(
            Box::new(apply_passes_once(cond, passes)),
            Box::new(apply_passes_once(then_expr, passes)),
            Box::new(apply_passes_once(else_expr, passes)),
        ),
        Ast::Let { name, value, body } => Ast::Let {
            name: name.clone(),
            value: Box::new(apply_passes_once(value, passes)),
            body: Box::new(apply_passes_once(body, passes)),
        },
    }
}

// ============================================================================
// Standard passes
// ============================================================================

/// Returns the standard set of optimization passes.
///
/// Includes:
/// - [`ConstantFolding`]
/// - [`AlgebraicIdentities`]
/// - [`PowerReduction`]
/// - [`LetInlining`]
///
/// Note: [`FunctionDecomposition`] is not included because it requires a
/// [`FunctionRegistry`]. Use [`standard_passes_with_funcs`] if you need it.
pub fn standard_passes() -> Vec<&'static dyn Pass> {
    vec![
        &ConstantFolding,
        &AlgebraicIdentities,
        &PowerReduction,
        &LetInlining,
    ]
}

/// Returns standard passes plus function decomposition.
///
/// # Example
///
/// ```ignore
/// use rhizome_dew_core::optimize::{optimize, standard_passes_with_funcs};
///
/// let registry = /* your function registry */;
/// let passes = standard_passes_with_funcs(&registry);
/// let optimized = optimize(ast, &passes);
/// ```
#[cfg(feature = "func")]
pub fn standard_passes_with_funcs(registry: &FunctionRegistry) -> Vec<Box<dyn Pass + '_>> {
    vec![
        Box::new(ConstantFolding),
        Box::new(AlgebraicIdentities),
        Box::new(PowerReduction),
        Box::new(LetInlining),
        Box::new(FunctionDecomposition::new(registry)),
    ]
}

// ============================================================================
// ConstantFolding pass
// ============================================================================

/// Evaluates operations on numeric literals at compile time.
///
/// # Examples
///
/// | Before | After |
/// |--------|-------|
/// | `1 + 2` | `3` |
/// | `2 * 3 * 4` | `24` |
/// | `-(-5)` | `5` |
/// | `2 ^ 3` | `8` |
pub struct ConstantFolding;

impl Pass for ConstantFolding {
    fn transform(&self, ast: &Ast) -> Option<Ast> {
        match ast {
            // Binary operations on two numbers
            Ast::BinOp(op, left, right) => {
                if let (Ast::Num(l), Ast::Num(r)) = (left.as_ref(), right.as_ref()) {
                    let result = match op {
                        BinOp::Add => l + r,
                        BinOp::Sub => l - r,
                        BinOp::Mul => l * r,
                        BinOp::Div => l / r,
                        BinOp::Pow => l.powf(*r),
                        BinOp::Rem => l % r,
                        // Bitwise ops don't fold for floats - they'll be handled
                        // in integer-specific evaluation
                        BinOp::BitAnd | BinOp::BitOr | BinOp::Shl | BinOp::Shr => {
                            return None;
                        }
                    };
                    Some(Ast::Num(result))
                } else {
                    None
                }
            }

            // Negation of a number
            Ast::UnaryOp(UnaryOp::Neg, inner) => {
                if let Ast::Num(n) = inner.as_ref() {
                    Some(Ast::Num(-n))
                } else {
                    None
                }
            }

            // Logical NOT of a number (cond feature)
            #[cfg(feature = "cond")]
            Ast::UnaryOp(UnaryOp::Not, inner) => {
                if let Ast::Num(n) = inner.as_ref() {
                    Some(Ast::Num(if *n == 0.0 { 1.0 } else { 0.0 }))
                } else {
                    None
                }
            }

            // Comparisons of two numbers (cond feature)
            #[cfg(feature = "cond")]
            Ast::Compare(op, left, right) => {
                if let (Ast::Num(l), Ast::Num(r)) = (left.as_ref(), right.as_ref()) {
                    let result = match op {
                        CompareOp::Lt => l < r,
                        CompareOp::Le => l <= r,
                        CompareOp::Gt => l > r,
                        CompareOp::Ge => l >= r,
                        CompareOp::Eq => l == r,
                        CompareOp::Ne => l != r,
                    };
                    Some(Ast::Num(if result { 1.0 } else { 0.0 }))
                } else {
                    None
                }
            }

            // And of two numbers (cond feature)
            #[cfg(feature = "cond")]
            Ast::And(left, right) => {
                if let (Ast::Num(l), Ast::Num(r)) = (left.as_ref(), right.as_ref()) {
                    let result = *l != 0.0 && *r != 0.0;
                    Some(Ast::Num(if result { 1.0 } else { 0.0 }))
                } else {
                    None
                }
            }

            // Or of two numbers (cond feature)
            #[cfg(feature = "cond")]
            Ast::Or(left, right) => {
                if let (Ast::Num(l), Ast::Num(r)) = (left.as_ref(), right.as_ref()) {
                    let result = *l != 0.0 || *r != 0.0;
                    Some(Ast::Num(if result { 1.0 } else { 0.0 }))
                } else {
                    None
                }
            }

            // If with constant condition (cond feature)
            #[cfg(feature = "cond")]
            Ast::If(cond, then_expr, else_expr) => {
                if let Ast::Num(c) = cond.as_ref() {
                    if *c != 0.0 {
                        Some(then_expr.as_ref().clone())
                    } else {
                        Some(else_expr.as_ref().clone())
                    }
                } else {
                    None
                }
            }

            _ => None,
        }
    }
}

// ============================================================================
// AlgebraicIdentities pass
// ============================================================================

/// Eliminates identity operations and absorbing elements.
///
/// # Rules Applied
///
/// | Before | After | Rule |
/// |--------|-------|------|
/// | `x * 1` | `x` | multiplicative identity |
/// | `x * 0` | `0` | zero absorbs |
/// | `x + 0` | `x` | additive identity |
/// | `x - 0` | `x` | subtractive identity |
/// | `x / 1` | `x` | divisive identity |
/// | `x ^ 1` | `x` | power identity |
/// | `x ^ 0` | `1` | zero power |
/// | `x - x` | `0` | self-subtraction |
/// | `x / x` | `1` | self-division |
/// | `--x` | `x` | double negation |
pub struct AlgebraicIdentities;

impl AlgebraicIdentities {
    fn is_zero(ast: &Ast) -> bool {
        matches!(ast, Ast::Num(n) if *n == 0.0)
    }

    fn is_one(ast: &Ast) -> bool {
        matches!(ast, Ast::Num(n) if (*n - 1.0).abs() < f64::EPSILON)
    }
}

impl Pass for AlgebraicIdentities {
    fn transform(&self, ast: &Ast) -> Option<Ast> {
        match ast {
            // Multiplication identities
            Ast::BinOp(BinOp::Mul, left, right) => {
                // x * 1 = x, 1 * x = x
                if Self::is_one(left) {
                    return Some(right.as_ref().clone());
                }
                if Self::is_one(right) {
                    return Some(left.as_ref().clone());
                }
                // x * 0 = 0, 0 * x = 0
                if Self::is_zero(left) || Self::is_zero(right) {
                    return Some(Ast::Num(0.0));
                }
                None
            }

            // Addition identities
            Ast::BinOp(BinOp::Add, left, right) => {
                // x + 0 = x, 0 + x = x
                if Self::is_zero(left) {
                    return Some(right.as_ref().clone());
                }
                if Self::is_zero(right) {
                    return Some(left.as_ref().clone());
                }
                None
            }

            // Subtraction identities
            Ast::BinOp(BinOp::Sub, left, right) => {
                // x - 0 = x
                if Self::is_zero(right) {
                    return Some(left.as_ref().clone());
                }
                // x - x = 0 (structural equality)
                if left == right {
                    return Some(Ast::Num(0.0));
                }
                None
            }

            // Division identities
            Ast::BinOp(BinOp::Div, left, right) => {
                // x / 1 = x
                if Self::is_one(right) {
                    return Some(left.as_ref().clone());
                }
                // x / x = 1 (structural equality, assumes x != 0)
                if left == right && !Self::is_zero(left) {
                    return Some(Ast::Num(1.0));
                }
                // 0 / x = 0 (assumes x != 0)
                if Self::is_zero(left) {
                    return Some(Ast::Num(0.0));
                }
                None
            }

            // Power identities
            Ast::BinOp(BinOp::Pow, left, right) => {
                // x ^ 1 = x
                if Self::is_one(right) {
                    return Some(left.as_ref().clone());
                }
                // x ^ 0 = 1 (assumes x != 0)
                if Self::is_zero(right) {
                    return Some(Ast::Num(1.0));
                }
                // 1 ^ x = 1
                if Self::is_one(left) {
                    return Some(Ast::Num(1.0));
                }
                // 0 ^ x = 0 (for x > 0)
                if Self::is_zero(left) {
                    if let Ast::Num(n) = right.as_ref() {
                        if *n > 0.0 {
                            return Some(Ast::Num(0.0));
                        }
                    }
                }
                None
            }

            // Double negation: --x = x
            Ast::UnaryOp(UnaryOp::Neg, inner) => {
                if let Ast::UnaryOp(UnaryOp::Neg, inner_inner) = inner.as_ref() {
                    return Some(inner_inner.as_ref().clone());
                }
                None
            }

            // Double NOT: not not x = x (cond feature)
            #[cfg(feature = "cond")]
            Ast::UnaryOp(UnaryOp::Not, inner) => {
                if let Ast::UnaryOp(UnaryOp::Not, inner_inner) = inner.as_ref() {
                    return Some(inner_inner.as_ref().clone());
                }
                None
            }

            // Short-circuit and (cond feature)
            #[cfg(feature = "cond")]
            Ast::And(left, right) => {
                // 0 and x = 0
                if Self::is_zero(left) {
                    return Some(Ast::Num(0.0));
                }
                // 1 and x = x (truthy, result depends on x)
                if let Ast::Num(n) = left.as_ref() {
                    if *n != 0.0 {
                        return Some(right.as_ref().clone());
                    }
                }
                None
            }

            // Short-circuit or (cond feature)
            #[cfg(feature = "cond")]
            Ast::Or(left, right) => {
                // 1 or x = 1 (truthy)
                if let Ast::Num(n) = left.as_ref() {
                    if *n != 0.0 {
                        return Some(Ast::Num(1.0));
                    }
                }
                // 0 or x = x
                if Self::is_zero(left) {
                    return Some(right.as_ref().clone());
                }
                None
            }

            _ => None,
        }
    }
}

// ============================================================================
// PowerReduction pass
// ============================================================================

/// Converts small integer powers to repeated multiplication.
///
/// | Before | After |
/// |--------|-------|
/// | `x ^ 2` | `x * x` |
/// | `x ^ 3` | `x * x * x` |
/// | `x ^ 4` | `x * x * x * x` |
///
/// Only reduces powers 2, 3, and 4. Higher powers remain as `powf` calls.
pub struct PowerReduction;

impl Pass for PowerReduction {
    fn transform(&self, ast: &Ast) -> Option<Ast> {
        if let Ast::BinOp(BinOp::Pow, base, exp) = ast {
            if let Ast::Num(n) = exp.as_ref() {
                let n_int = *n as i32;
                // Only reduce small integer powers
                if (*n - n_int as f64).abs() < f64::EPSILON {
                    match n_int {
                        2 => {
                            return Some(Ast::BinOp(BinOp::Mul, base.clone(), base.clone()));
                        }
                        3 => {
                            return Some(Ast::BinOp(
                                BinOp::Mul,
                                Box::new(Ast::BinOp(BinOp::Mul, base.clone(), base.clone())),
                                base.clone(),
                            ));
                        }
                        4 => {
                            let squared = Ast::BinOp(BinOp::Mul, base.clone(), base.clone());
                            return Some(Ast::BinOp(
                                BinOp::Mul,
                                Box::new(squared.clone()),
                                Box::new(squared),
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
}

// ============================================================================
// LetInlining pass
// ============================================================================

/// Inlines let bindings when safe and beneficial.
///
/// | Before | After | Condition |
/// |--------|-------|-----------|
/// | `let a = 3; a * 2` | `3 * 2` | Value is literal |
/// | `let a = x; a + y` | `x + y` | Value is variable |
/// | `let a = expr; a` | `expr` | Single use in body |
/// | `let a = expr; a + a` | kept | Multiple uses, non-trivial value |
///
/// This pass enables WGSL/GLSL codegen by eliminating let expressions.
pub struct LetInlining;

impl LetInlining {
    /// Count occurrences of a variable name in an AST.
    fn count_uses(ast: &Ast, name: &str) -> usize {
        match ast {
            Ast::Num(_) => 0,
            Ast::Var(v) => {
                if v == name {
                    1
                } else {
                    0
                }
            }
            Ast::BinOp(_, left, right) => {
                Self::count_uses(left, name) + Self::count_uses(right, name)
            }
            Ast::UnaryOp(_, inner) => Self::count_uses(inner, name),
            #[cfg(feature = "func")]
            Ast::Call(_, args) => args.iter().map(|a| Self::count_uses(a, name)).sum(),
            #[cfg(feature = "cond")]
            Ast::Compare(_, left, right) => {
                Self::count_uses(left, name) + Self::count_uses(right, name)
            }
            #[cfg(feature = "cond")]
            Ast::And(left, right) => Self::count_uses(left, name) + Self::count_uses(right, name),
            #[cfg(feature = "cond")]
            Ast::Or(left, right) => Self::count_uses(left, name) + Self::count_uses(right, name),
            #[cfg(feature = "cond")]
            Ast::If(cond, then_expr, else_expr) => {
                Self::count_uses(cond, name)
                    + Self::count_uses(then_expr, name)
                    + Self::count_uses(else_expr, name)
            }
            Ast::Let {
                name: bound,
                value,
                body,
            } => {
                let in_value = Self::count_uses(value, name);
                // If this let shadows the name, don't count uses in body
                if bound == name {
                    in_value
                } else {
                    in_value + Self::count_uses(body, name)
                }
            }
        }
    }

    /// Substitute all occurrences of a variable with a replacement AST.
    fn substitute(ast: &Ast, name: &str, replacement: &Ast) -> Ast {
        match ast {
            Ast::Num(n) => Ast::Num(*n),
            Ast::Var(v) => {
                if v == name {
                    replacement.clone()
                } else {
                    Ast::Var(v.clone())
                }
            }
            Ast::BinOp(op, left, right) => Ast::BinOp(
                *op,
                Box::new(Self::substitute(left, name, replacement)),
                Box::new(Self::substitute(right, name, replacement)),
            ),
            Ast::UnaryOp(op, inner) => {
                Ast::UnaryOp(*op, Box::new(Self::substitute(inner, name, replacement)))
            }
            #[cfg(feature = "func")]
            Ast::Call(func_name, args) => Ast::Call(
                func_name.clone(),
                args.iter()
                    .map(|a| Self::substitute(a, name, replacement))
                    .collect(),
            ),
            #[cfg(feature = "cond")]
            Ast::Compare(op, left, right) => Ast::Compare(
                *op,
                Box::new(Self::substitute(left, name, replacement)),
                Box::new(Self::substitute(right, name, replacement)),
            ),
            #[cfg(feature = "cond")]
            Ast::And(left, right) => Ast::And(
                Box::new(Self::substitute(left, name, replacement)),
                Box::new(Self::substitute(right, name, replacement)),
            ),
            #[cfg(feature = "cond")]
            Ast::Or(left, right) => Ast::Or(
                Box::new(Self::substitute(left, name, replacement)),
                Box::new(Self::substitute(right, name, replacement)),
            ),
            #[cfg(feature = "cond")]
            Ast::If(cond, then_expr, else_expr) => Ast::If(
                Box::new(Self::substitute(cond, name, replacement)),
                Box::new(Self::substitute(then_expr, name, replacement)),
                Box::new(Self::substitute(else_expr, name, replacement)),
            ),
            Ast::Let {
                name: bound,
                value,
                body,
            } => {
                let new_value = Self::substitute(value, name, replacement);
                // If this let shadows the name, don't substitute in body
                if bound == name {
                    Ast::Let {
                        name: bound.clone(),
                        value: Box::new(new_value),
                        body: body.clone(),
                    }
                } else {
                    Ast::Let {
                        name: bound.clone(),
                        value: Box::new(new_value),
                        body: Box::new(Self::substitute(body, name, replacement)),
                    }
                }
            }
        }
    }
}

impl Pass for LetInlining {
    fn transform(&self, ast: &Ast) -> Option<Ast> {
        if let Ast::Let { name, value, body } = ast {
            let uses = Self::count_uses(body, name);

            // Always inline if unused
            if uses == 0 {
                return Some(body.as_ref().clone());
            }

            // Inline trivial values (cheap to duplicate)
            if matches!(value.as_ref(), Ast::Num(_) | Ast::Var(_)) {
                return Some(Self::substitute(body, name, value));
            }

            // Inline if used exactly once (no duplication of work)
            if uses == 1 {
                return Some(Self::substitute(body, name, value));
            }

            // Keep the let: multiple uses of non-trivial expression
            // WGSL/GLSL codegen will fail for these - that's intentional
        }
        None
    }
}

// ============================================================================
// FunctionDecomposition pass (requires func feature)
// ============================================================================

/// Expands functions using their `decompose()` method.
///
/// This pass uses [`ExprFn::decompose`] to replace function calls with
/// equivalent expressions using simpler operations.
///
/// # Example
///
/// A `log10` function might decompose to `log(x) / log(10)`:
///
/// ```ignore
/// impl ExprFn for Log10 {
///     fn decompose(&self, args: &[Ast]) -> Option<Ast> {
///         Some(Ast::BinOp(
///             BinOp::Div,
///             Box::new(Ast::Call("log".into(), args.to_vec())),
///             Box::new(Ast::Call("log".into(), vec![Ast::Num(10.0)])),
///         ))
///     }
/// }
/// ```
#[cfg(feature = "func")]
pub struct FunctionDecomposition<'a> {
    registry: &'a FunctionRegistry,
}

#[cfg(feature = "func")]
impl<'a> FunctionDecomposition<'a> {
    /// Creates a new function decomposition pass with the given registry.
    pub fn new(registry: &'a FunctionRegistry) -> Self {
        Self { registry }
    }
}

#[cfg(feature = "func")]
impl Pass for FunctionDecomposition<'_> {
    fn transform(&self, ast: &Ast) -> Option<Ast> {
        if let Ast::Call(name, args) = ast {
            if let Some(func) = self.registry.get(name) {
                return func.decompose(args);
            }
        }
        None
    }
}

// ============================================================================
// AstHasher for CSE support
// ============================================================================

/// Utility for hashing AST nodes.
///
/// Used by backends to implement Common Subexpression Elimination (CSE).
/// Handles f32 by converting to bits, ensuring consistent hashing.
///
/// # Example
///
/// ```
/// use rhizome_dew_core::{Ast, BinOp};
/// use rhizome_dew_core::optimize::AstHasher;
///
/// let ast1 = Ast::BinOp(BinOp::Add, Box::new(Ast::Var("x".into())), Box::new(Ast::Num(1.0)));
/// let ast2 = Ast::BinOp(BinOp::Add, Box::new(Ast::Var("x".into())), Box::new(Ast::Num(1.0)));
/// let ast3 = Ast::BinOp(BinOp::Add, Box::new(Ast::Var("x".into())), Box::new(Ast::Num(2.0)));
///
/// assert_eq!(AstHasher::hash(&ast1), AstHasher::hash(&ast2));
/// assert_ne!(AstHasher::hash(&ast1), AstHasher::hash(&ast3));
/// ```
pub struct AstHasher;

impl AstHasher {
    /// Computes a hash for an AST node.
    ///
    /// Structurally identical ASTs produce the same hash.
    /// f32 values are converted to bits for hashing.
    pub fn hash(ast: &Ast) -> u64 {
        let mut hasher = DefaultHasher::new();
        Self::hash_into(ast, &mut hasher);
        hasher.finish()
    }

    fn hash_into(ast: &Ast, hasher: &mut DefaultHasher) {
        // Discriminant first for type differentiation
        std::mem::discriminant(ast).hash(hasher);

        match ast {
            Ast::Num(n) => {
                // Convert f32 to bits for consistent hashing
                n.to_bits().hash(hasher);
            }
            Ast::Var(name) => {
                name.hash(hasher);
            }
            Ast::BinOp(op, left, right) => {
                op.hash(hasher);
                Self::hash_into(left, hasher);
                Self::hash_into(right, hasher);
            }
            Ast::UnaryOp(op, inner) => {
                op.hash(hasher);
                Self::hash_into(inner, hasher);
            }
            #[cfg(feature = "func")]
            Ast::Call(name, args) => {
                name.hash(hasher);
                args.len().hash(hasher);
                for arg in args {
                    Self::hash_into(arg, hasher);
                }
            }
            #[cfg(feature = "cond")]
            Ast::Compare(op, left, right) => {
                op.hash(hasher);
                Self::hash_into(left, hasher);
                Self::hash_into(right, hasher);
            }
            #[cfg(feature = "cond")]
            Ast::And(left, right) => {
                Self::hash_into(left, hasher);
                Self::hash_into(right, hasher);
            }
            #[cfg(feature = "cond")]
            Ast::Or(left, right) => {
                Self::hash_into(left, hasher);
                Self::hash_into(right, hasher);
            }
            #[cfg(feature = "cond")]
            Ast::If(cond, then_expr, else_expr) => {
                Self::hash_into(cond, hasher);
                Self::hash_into(then_expr, hasher);
                Self::hash_into(else_expr, hasher);
            }
            Ast::Let { name, value, body } => {
                name.hash(hasher);
                Self::hash_into(value, hasher);
                Self::hash_into(body, hasher);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Expr;

    fn optimized(input: &str) -> String {
        let expr = Expr::parse(input).unwrap();
        let result = optimize(expr.ast().clone(), &standard_passes());
        result.to_string()
    }

    // Constant folding tests
    #[test]
    fn test_constant_fold_add() {
        assert_eq!(optimized("1 + 2"), "3");
    }

    #[test]
    fn test_constant_fold_mul() {
        assert_eq!(optimized("3 * 4"), "12");
    }

    #[test]
    fn test_constant_fold_sub() {
        assert_eq!(optimized("10 - 3"), "7");
    }

    #[test]
    fn test_constant_fold_div() {
        assert_eq!(optimized("8 / 2"), "4");
    }

    #[test]
    fn test_constant_fold_pow() {
        assert_eq!(optimized("2 ^ 3"), "8");
    }

    #[test]
    fn test_constant_fold_neg() {
        assert_eq!(optimized("-5"), "-5");
        assert_eq!(optimized("--5"), "5");
    }

    #[test]
    fn test_constant_fold_chain() {
        assert_eq!(optimized("1 + 2 + 3"), "6");
        assert_eq!(optimized("2 * 3 * 4"), "24");
    }

    // Algebraic identity tests
    #[test]
    fn test_identity_mul_one() {
        assert_eq!(optimized("x * 1"), "x");
        assert_eq!(optimized("1 * x"), "x");
    }

    #[test]
    fn test_identity_mul_zero() {
        assert_eq!(optimized("x * 0"), "0");
        assert_eq!(optimized("0 * x"), "0");
    }

    #[test]
    fn test_identity_add_zero() {
        assert_eq!(optimized("x + 0"), "x");
        assert_eq!(optimized("0 + x"), "x");
    }

    #[test]
    fn test_identity_sub_zero() {
        assert_eq!(optimized("x - 0"), "x");
    }

    #[test]
    fn test_identity_div_one() {
        assert_eq!(optimized("x / 1"), "x");
    }

    #[test]
    fn test_identity_pow_one() {
        assert_eq!(optimized("x ^ 1"), "x");
    }

    #[test]
    fn test_identity_pow_zero() {
        assert_eq!(optimized("x ^ 0"), "1");
    }

    #[test]
    fn test_identity_self_sub() {
        assert_eq!(optimized("x - x"), "0");
    }

    #[test]
    fn test_identity_self_div() {
        assert_eq!(optimized("x / x"), "1");
    }

    #[test]
    fn test_double_neg() {
        assert_eq!(optimized("--x"), "x");
    }

    // Power reduction tests
    #[test]
    fn test_power_reduce_2() {
        assert_eq!(optimized("x ^ 2"), "(x * x)");
    }

    #[test]
    fn test_power_reduce_3() {
        assert_eq!(optimized("x ^ 3"), "((x * x) * x)");
    }

    #[test]
    fn test_power_reduce_4() {
        assert_eq!(optimized("x ^ 4"), "((x * x) * (x * x))");
    }

    #[test]
    fn test_power_no_reduce_5() {
        // Power 5 should not be reduced
        assert_eq!(optimized("x ^ 5"), "(x ^ 5)");
    }

    // Combined optimizations
    #[test]
    fn test_combined() {
        // (1 + 2) * x + 0 → 3 * x
        assert_eq!(optimized("(1 + 2) * x + 0"), "(3 * x)");
    }

    #[test]
    fn test_combined_complex() {
        // x * 0 + y * 1 → y
        assert_eq!(optimized("x * 0 + y * 1"), "y");
    }

    #[test]
    fn test_nested_identity() {
        // (x + 0) * 1 → x
        assert_eq!(optimized("(x + 0) * 1"), "x");
    }

    // Let inlining tests
    #[test]
    fn test_let_inline_literal() {
        // let a = 3; a * 2 → 3 * 2 → 6
        assert_eq!(optimized("let a = 3; a * 2"), "6");
    }

    #[test]
    fn test_let_inline_variable() {
        // let a = x; a + y → x + y
        assert_eq!(optimized("let a = x; a + y"), "(x + y)");
    }

    #[test]
    fn test_let_inline_unused() {
        // let a = 42; x → x (unused binding eliminated)
        assert_eq!(optimized("let a = 42; x"), "x");
    }

    #[test]
    fn test_let_inline_chained() {
        // let a = 1; let b = 2; a + b → 1 + 2 → 3
        assert_eq!(optimized("let a = 1; let b = 2; a + b"), "3");
    }

    #[test]
    fn test_let_inline_nested_fold() {
        // let a = 1 + 2; a * 2 → let a = 3; a * 2 → 3 * 2 → 6
        assert_eq!(optimized("let a = 1 + 2; a * 2"), "6");
    }

    #[test]
    fn test_let_inline_multi_use_trivial() {
        // let a = x; a + a → x + x (trivial value, safe to duplicate)
        assert_eq!(optimized("let a = x; a + a"), "(x + x)");
    }

    #[test]
    fn test_let_keep_multi_use_complex() {
        // let a = x + y; a + a → kept (non-trivial, would duplicate work)
        assert_eq!(
            optimized("let a = x + y; a * a"),
            "(let a = (x + y); (a * a))"
        );
    }

    // AstHasher tests
    #[test]
    fn test_hash_equal() {
        let ast1 = Ast::BinOp(
            BinOp::Add,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(1.0)),
        );
        let ast2 = Ast::BinOp(
            BinOp::Add,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(1.0)),
        );
        assert_eq!(AstHasher::hash(&ast1), AstHasher::hash(&ast2));
    }

    #[test]
    fn test_hash_different() {
        let ast1 = Ast::BinOp(
            BinOp::Add,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(1.0)),
        );
        let ast2 = Ast::BinOp(
            BinOp::Add,
            Box::new(Ast::Var("x".into())),
            Box::new(Ast::Num(2.0)),
        );
        assert_ne!(AstHasher::hash(&ast1), AstHasher::hash(&ast2));
    }

    #[test]
    fn test_hash_op_matters() {
        let ast1 = Ast::BinOp(BinOp::Add, Box::new(Ast::Num(1.0)), Box::new(Ast::Num(2.0)));
        let ast2 = Ast::BinOp(BinOp::Mul, Box::new(Ast::Num(1.0)), Box::new(Ast::Num(2.0)));
        assert_ne!(AstHasher::hash(&ast1), AstHasher::hash(&ast2));
    }

    // Conditional feature tests
    #[cfg(feature = "cond")]
    mod cond_tests {
        use super::*;

        #[test]
        fn test_constant_fold_compare() {
            assert_eq!(optimized("1 < 2"), "1");
            assert_eq!(optimized("2 < 1"), "0");
            assert_eq!(optimized("1 == 1"), "1");
            assert_eq!(optimized("1 != 1"), "0");
        }

        #[test]
        fn test_constant_fold_and() {
            assert_eq!(optimized("1 and 1"), "1");
            assert_eq!(optimized("1 and 0"), "0");
            assert_eq!(optimized("0 and 1"), "0");
        }

        #[test]
        fn test_constant_fold_or() {
            assert_eq!(optimized("1 or 0"), "1");
            assert_eq!(optimized("0 or 1"), "1");
            assert_eq!(optimized("0 or 0"), "0");
        }

        #[test]
        fn test_constant_fold_not() {
            assert_eq!(optimized("not 0"), "1");
            assert_eq!(optimized("not 1"), "0");
        }

        #[test]
        fn test_constant_fold_if() {
            assert_eq!(optimized("if 1 then 10 else 20"), "10");
            assert_eq!(optimized("if 0 then 10 else 20"), "20");
        }

        #[test]
        fn test_identity_and() {
            // 0 and x = 0
            assert_eq!(optimized("0 and x"), "0");
            // 1 and x = x
            assert_eq!(optimized("1 and x"), "x");
        }

        #[test]
        fn test_identity_or() {
            // 1 or x = 1
            assert_eq!(optimized("1 or x"), "1");
            // 0 or x = x
            assert_eq!(optimized("0 or x"), "x");
        }

        #[test]
        fn test_double_not() {
            assert_eq!(optimized("not not x"), "x");
        }
    }

    // Function feature tests
    #[cfg(feature = "func")]
    mod func_tests {
        use super::*;
        use crate::{ExprFn, FunctionRegistry};

        struct Log10;
        impl ExprFn for Log10 {
            fn name(&self) -> &str {
                "log10"
            }
            fn arg_count(&self) -> usize {
                1
            }
            fn call(&self, args: &[f32]) -> f32 {
                args[0].log10()
            }
            fn decompose(&self, args: &[Ast]) -> Option<Ast> {
                // log10(x) = log(x) / log(10)
                Some(Ast::BinOp(
                    BinOp::Div,
                    Box::new(Ast::Call("log".into(), args.to_vec())),
                    Box::new(Ast::Call("log".into(), vec![Ast::Num(10.0)])),
                ))
            }
        }

        #[test]
        fn test_function_decomposition() {
            let mut registry = FunctionRegistry::new();
            registry.register(Log10);

            let expr = Expr::parse("log10(x)").unwrap();
            let passes = standard_passes_with_funcs(&registry);
            let pass_refs: Vec<&dyn Pass> = passes.iter().map(|b| b.as_ref()).collect();
            let result = optimize(expr.ast().clone(), &pass_refs);

            // Should decompose to log(x) / log(10)
            assert_eq!(result.to_string(), "(log(x) / log(10))");
        }
    }
}
