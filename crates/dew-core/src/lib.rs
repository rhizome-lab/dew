//! # dew-core
//!
//! Minimal expression language, multiple backends.
//!
//! This crate provides a simple expression parser that compiles string expressions
//! into evaluable ASTs. Variables and functions are provided by the caller—nothing
//! is hardcoded—making it suitable for user-facing expression inputs, shader
//! parameter systems, and dynamic computation pipelines.
//!
//! ## Design Philosophy
//!
//! - **Minimal by default**: Core supports only arithmetic and variables
//! - **Opt-in complexity**: Enable `cond` for conditionals, `func` for function calls
//! - **No runtime dependencies**: Pure Rust, no allocations during evaluation
//! - **Backend-agnostic**: AST can be compiled to WGSL, Lua, Cranelift, or evaluated directly
//!
//! ## Features
//!
//! | Feature      | Description |
//! |--------------|-------------|
//! | `introspect` | AST introspection (`free_vars`, etc.) - **enabled by default** |
//! | `cond`       | Conditionals (`if`/`then`/`else`), comparisons (`<`, `<=`, etc.), boolean logic (`and`, `or`, `not`) |
//! | `func`       | Function calls via [`ExprFn`] trait and [`FunctionRegistry`] |
//!
//! ## Syntax Reference
//!
//! ### Operators (by precedence, low to high)
//!
//! | Precedence | Operators | Description |
//! |------------|-----------|-------------|
//! | 1 | `if c then a else b` | Conditional (requires `cond`) |
//! | 2 | `a or b` | Logical OR, short-circuit (requires `cond`) |
//! | 3 | `a and b` | Logical AND, short-circuit (requires `cond`) |
//! | 4 | `<` `<=` `>` `>=` `==` `!=` | Comparison (requires `cond`) |
//! | 5 | `a + b`, `a - b` | Addition, subtraction |
//! | 6 | `a * b`, `a / b` | Multiplication, division |
//! | 7 | `a ^ b` | Exponentiation (right-associative) |
//! | 8 | `-a`, `not a` | Negation, logical NOT (`not` requires `cond`) |
//! | 9 | `(a)`, `f(a, b)` | Grouping, function calls (calls require `func`) |
//!
//! ### Literals and Identifiers
//!
//! - **Numbers**: `42`, `3.14`, `.5`, `1.0`
//! - **Variables**: Any identifier (`x`, `time`, `my_var`)
//! - **Functions**: Identifier followed by parentheses (`sin(x)`, `clamp(x, 0, 1)`)
//!
//! ### Boolean Semantics (with `cond` feature)
//!
//! - `0.0` is false, any non-zero value is true
//! - Comparisons and boolean operators return `1.0` (true) or `0.0` (false)
//! - `and`/`or` use short-circuit evaluation
//!
//! ## Examples
//!
//! ### Basic Arithmetic
//!
//! ```
//! use rhizome_dew_core::Expr;
//! use std::collections::HashMap;
//!
//! let expr = Expr::parse("x * 2 + y").unwrap();
//!
//! let mut vars = HashMap::new();
//! vars.insert("x".to_string(), 3.0);
//! vars.insert("y".to_string(), 1.0);
//!
//! # #[cfg(not(feature = "func"))]
//! let value = expr.eval(&vars).unwrap();
//! # #[cfg(feature = "func")]
//! # let value = expr.eval(&vars, &rhizome_dew_core::FunctionRegistry::new()).unwrap();
//! assert_eq!(value, 7.0);  // 3 * 2 + 1 = 7
//! ```
//!
//! ### Working with the AST
//!
//! ```
//! use rhizome_dew_core::{Expr, Ast, BinOp};
//!
//! let expr = Expr::parse("a + b * c").unwrap();
//!
//! // Inspect the AST structure
//! match expr.ast() {
//!     Ast::BinOp(BinOp::Add, left, right) => {
//!         assert!(matches!(left.as_ref(), Ast::Var(name) if name == "a"));
//!         assert!(matches!(right.as_ref(), Ast::BinOp(BinOp::Mul, _, _)));
//!     }
//!     _ => panic!("unexpected AST structure"),
//! }
//! ```
//!
//! ### Custom Functions (with `func` feature)
//!
#![cfg_attr(feature = "func", doc = "```")]
#![cfg_attr(not(feature = "func"), doc = "```ignore")]
//! use rhizome_dew_core::{Expr, ExprFn, FunctionRegistry, Ast};
//! use std::collections::HashMap;
//!
//! struct Clamp;
//! impl ExprFn for Clamp {
//!     fn name(&self) -> &str { "clamp" }
//!     fn arg_count(&self) -> usize { 3 }
//!     fn call(&self, args: &[f32]) -> f32 {
//!         args[0].clamp(args[1], args[2])
//!     }
//! }
//!
//! let mut registry = FunctionRegistry::new();
//! registry.register(Clamp);
//!
//! let expr = Expr::parse("clamp(x, 0, 1)").unwrap();
//! let mut vars = HashMap::new();
//! vars.insert("x".to_string(), 1.5);
//!
//! let value = expr.eval(&vars, &registry).unwrap();
//! assert_eq!(value, 1.0);  // clamped to [0, 1]
//! ```
//!
//! ### Conditionals (with `cond` feature)
//!
#![cfg_attr(feature = "cond", doc = "```")]
#![cfg_attr(not(feature = "cond"), doc = "```ignore")]
//! use rhizome_dew_core::Expr;
//! use std::collections::HashMap;
//!
//! let expr = Expr::parse("if x > 0 then x else -x").unwrap();  // absolute value
//!
//! let mut vars = HashMap::new();
//! vars.insert("x".to_string(), -5.0);
//!
//! # #[cfg(not(feature = "func"))]
//! let value = expr.eval(&vars).unwrap();
//! # #[cfg(feature = "func")]
//! # let value = expr.eval(&vars, &rhizome_dew_core::FunctionRegistry::new()).unwrap();
//! assert_eq!(value, 5.0);
//! ```

use std::collections::HashMap;
#[cfg(feature = "introspect")]
use std::collections::HashSet;
#[cfg(feature = "func")]
use std::sync::Arc;

use num_traits::{Num, NumCast, One, Zero};
use std::ops::Neg;

#[cfg(feature = "optimize")]
pub mod optimize;

// ============================================================================
// Numeric Trait
// ============================================================================

/// Trait for types that can be used as numeric values in expressions.
///
/// This is a marker trait that combines the necessary bounds for basic
/// arithmetic operations. Both float and integer types implement this.
pub trait Numeric:
    Num
    + NumCast
    + Copy
    + PartialOrd
    + Zero
    + One
    + Neg<Output = Self>
    + std::fmt::Debug
    + Send
    + Sync
    + 'static
{
    /// Whether this type supports bitwise operations.
    fn supports_bitwise() -> bool;

    /// Whether this type is a floating-point type.
    fn is_float() -> bool;

    /// Compute self raised to the power of exp.
    /// For floats, uses powf. For integers, uses repeated multiplication
    /// (returns None for negative exponents).
    fn numeric_pow(self, exp: Self) -> Option<Self>;
}

impl Numeric for f32 {
    fn supports_bitwise() -> bool {
        false
    }
    fn is_float() -> bool {
        true
    }
    fn numeric_pow(self, exp: Self) -> Option<Self> {
        Some(self.powf(exp))
    }
}

impl Numeric for f64 {
    fn supports_bitwise() -> bool {
        false
    }
    fn is_float() -> bool {
        true
    }
    fn numeric_pow(self, exp: Self) -> Option<Self> {
        Some(self.powf(exp))
    }
}

impl Numeric for i32 {
    fn supports_bitwise() -> bool {
        true
    }
    fn is_float() -> bool {
        false
    }
    fn numeric_pow(self, exp: Self) -> Option<Self> {
        if exp < 0 {
            return None;
        }
        Some(self.pow(exp as u32))
    }
}

impl Numeric for i64 {
    fn supports_bitwise() -> bool {
        true
    }
    fn is_float() -> bool {
        false
    }
    fn numeric_pow(self, exp: Self) -> Option<Self> {
        if exp < 0 {
            return None;
        }
        Some(self.pow(exp as u32))
    }
}

// Note: u32/u64 don't implement Neg, so they're not included in Numeric.
// Use i32/i64 for integer vectors/matrices.

// ============================================================================
// ExprFn trait and registry (func feature)
// ============================================================================

/// A function that can be called from expressions.
///
/// Implement this trait to add custom functions.
/// Constants (like `pi`) can be 0-arg functions.
#[cfg(feature = "func")]
pub trait ExprFn: Send + Sync {
    /// Function name (e.g., "sin", "pi").
    fn name(&self) -> &str;

    /// Number of arguments this function expects.
    fn arg_count(&self) -> usize;

    /// Evaluate the function with the given arguments.
    fn call(&self, args: &[f32]) -> f32;

    /// Express as simpler expressions (enables automatic backend support).
    /// If this returns Some, backends can compile without knowing about this function.
    fn decompose(&self, _args: &[Ast]) -> Option<Ast> {
        None
    }
}

/// Registry of expression functions.
#[cfg(feature = "func")]
#[derive(Clone, Default)]
pub struct FunctionRegistry {
    funcs: HashMap<String, Arc<dyn ExprFn>>,
}

#[cfg(feature = "func")]
impl FunctionRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a function.
    pub fn register<F: ExprFn + 'static>(&mut self, func: F) {
        self.funcs.insert(func.name().to_string(), Arc::new(func));
    }

    /// Gets a function by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn ExprFn>> {
        self.funcs.get(name)
    }

    /// Returns an iterator over all registered function names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.funcs.keys().map(|s| s.as_str())
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Expression parse error.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedChar(char),
    UnexpectedEnd,
    UnexpectedToken(String),
    InvalidNumber(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedChar(c) => write!(f, "unexpected character: '{}'", c),
            ParseError::UnexpectedEnd => write!(f, "unexpected end of expression"),
            ParseError::UnexpectedToken(t) => write!(f, "unexpected token: '{}'", t),
            ParseError::InvalidNumber(s) => write!(f, "invalid number: '{}'", s),
        }
    }
}

impl std::error::Error for ParseError {}

/// Expression evaluation error.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    UnknownVariable(String),
    #[cfg(feature = "func")]
    UnknownFunction(String),
    #[cfg(feature = "func")]
    WrongArgCount {
        func: String,
        expected: usize,
        got: usize,
    },
    /// Operation not supported for this numeric type (e.g., bitwise ops on floats).
    UnsupportedOperation(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::UnknownVariable(name) => write!(f, "unknown variable: '{}'", name),
            #[cfg(feature = "func")]
            EvalError::UnknownFunction(name) => write!(f, "unknown function: '{}'", name),
            #[cfg(feature = "func")]
            EvalError::WrongArgCount {
                func,
                expected,
                got,
            } => {
                write!(
                    f,
                    "function '{}' expects {} args, got {}",
                    func, expected, got
                )
            }
            EvalError::UnsupportedOperation(op) => {
                write!(f, "unsupported operation for this numeric type: '{}'", op)
            }
        }
    }
}

impl std::error::Error for EvalError {}

// ============================================================================
// Lexer
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Percent,
    Ampersand,
    Pipe,
    Tilde,
    Shl,
    Shr,
    LParen,
    RParen,
    #[cfg(feature = "func")]
    Comma,
    // Comparison operators
    #[cfg(feature = "cond")]
    Lt,
    #[cfg(feature = "cond")]
    Le,
    #[cfg(feature = "cond")]
    Gt,
    #[cfg(feature = "cond")]
    Ge,
    #[cfg(feature = "cond")]
    Eq,
    #[cfg(feature = "cond")]
    Ne,
    // Boolean operators (keywords)
    #[cfg(feature = "cond")]
    And,
    #[cfg(feature = "cond")]
    Or,
    #[cfg(feature = "cond")]
    Not,
    // Conditional
    #[cfg(feature = "cond")]
    If,
    #[cfg(feature = "cond")]
    Then,
    #[cfg(feature = "cond")]
    Else,
    Eof,
}

struct Lexer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn next_char(&mut self) -> Option<char> {
        let c = self.peek_char()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.next_char();
            } else {
                break;
            }
        }
    }

    fn read_number(&mut self) -> Result<f64, ParseError> {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() || c == '.' {
                self.next_char();
            } else {
                break;
            }
        }
        let s = &self.input[start..self.pos];
        s.parse()
            .map_err(|_| ParseError::InvalidNumber(s.to_string()))
    }

    fn read_ident(&mut self) -> String {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '_' {
                self.next_char();
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace();

        let Some(c) = self.peek_char() else {
            return Ok(Token::Eof);
        };

        match c {
            '+' => {
                self.next_char();
                Ok(Token::Plus)
            }
            '-' => {
                self.next_char();
                Ok(Token::Minus)
            }
            '*' => {
                self.next_char();
                Ok(Token::Star)
            }
            '/' => {
                self.next_char();
                Ok(Token::Slash)
            }
            '^' => {
                self.next_char();
                Ok(Token::Caret)
            }
            '%' => {
                self.next_char();
                Ok(Token::Percent)
            }
            '&' => {
                self.next_char();
                Ok(Token::Ampersand)
            }
            '|' => {
                self.next_char();
                Ok(Token::Pipe)
            }
            '~' => {
                self.next_char();
                Ok(Token::Tilde)
            }
            '(' => {
                self.next_char();
                Ok(Token::LParen)
            }
            ')' => {
                self.next_char();
                Ok(Token::RParen)
            }
            #[cfg(feature = "func")]
            ',' => {
                self.next_char();
                Ok(Token::Comma)
            }
            '<' => {
                self.next_char();
                if self.peek_char() == Some('<') {
                    self.next_char();
                    Ok(Token::Shl)
                } else {
                    #[cfg(feature = "cond")]
                    {
                        if self.peek_char() == Some('=') {
                            self.next_char();
                            Ok(Token::Le)
                        } else {
                            Ok(Token::Lt)
                        }
                    }
                    #[cfg(not(feature = "cond"))]
                    Err(ParseError::UnexpectedChar('<'))
                }
            }
            '>' => {
                self.next_char();
                if self.peek_char() == Some('>') {
                    self.next_char();
                    Ok(Token::Shr)
                } else {
                    #[cfg(feature = "cond")]
                    {
                        if self.peek_char() == Some('=') {
                            self.next_char();
                            Ok(Token::Ge)
                        } else {
                            Ok(Token::Gt)
                        }
                    }
                    #[cfg(not(feature = "cond"))]
                    Err(ParseError::UnexpectedChar('>'))
                }
            }
            #[cfg(feature = "cond")]
            '=' => {
                self.next_char();
                if self.peek_char() == Some('=') {
                    self.next_char();
                    Ok(Token::Eq)
                } else {
                    Err(ParseError::UnexpectedChar('='))
                }
            }
            #[cfg(feature = "cond")]
            '!' => {
                self.next_char();
                if self.peek_char() == Some('=') {
                    self.next_char();
                    Ok(Token::Ne)
                } else {
                    Err(ParseError::UnexpectedChar('!'))
                }
            }
            '0'..='9' | '.' => Ok(Token::Number(self.read_number()?)),
            'a'..='z' | 'A'..='Z' | '_' => {
                let ident = self.read_ident();
                // Check for keywords
                #[cfg(feature = "cond")]
                match ident.as_str() {
                    "and" => return Ok(Token::And),
                    "or" => return Ok(Token::Or),
                    "not" => return Ok(Token::Not),
                    "if" => return Ok(Token::If),
                    "then" => return Ok(Token::Then),
                    "else" => return Ok(Token::Else),
                    _ => {}
                }
                Ok(Token::Ident(ident))
            }
            _ => Err(ParseError::UnexpectedChar(c)),
        }
    }
}

// ============================================================================
// AST
// ============================================================================

/// Abstract syntax tree node for expressions.
///
/// The AST represents the structure of a parsed expression. Use [`Expr::ast()`]
/// to access the AST after parsing.
///
/// # Variants
///
/// The available variants depend on enabled features:
/// - **Always**: `Num`, `Var`, `BinOp`, `UnaryOp`
/// - **With `func`**: `Call`
/// - **With `cond`**: `Compare`, `And`, `Or`, `If`
///
/// # Example
///
/// ```
/// use rhizome_dew_core::{Expr, Ast, BinOp};
///
/// let expr = Expr::parse("2 + 3").unwrap();
/// match expr.ast() {
///     Ast::BinOp(BinOp::Add, left, right) => {
///         assert!(matches!(left.as_ref(), Ast::Num(2.0)));
///         assert!(matches!(right.as_ref(), Ast::Num(3.0)));
///     }
///     _ => panic!("expected addition"),
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Ast {
    /// Numeric literal (e.g., `42`, `3.14`).
    Num(f64),
    /// Variable reference, resolved at evaluation time.
    Var(String),
    /// Binary operation: `left op right`.
    BinOp(BinOp, Box<Ast>, Box<Ast>),
    /// Unary operation: `op operand`.
    UnaryOp(UnaryOp, Box<Ast>),
    /// Function call: `name(arg1, arg2, ...)`.
    #[cfg(feature = "func")]
    Call(String, Vec<Ast>),
    /// Comparison: `left op right`, evaluates to `0.0` or `1.0`.
    #[cfg(feature = "cond")]
    Compare(CompareOp, Box<Ast>, Box<Ast>),
    /// Logical AND with short-circuit evaluation.
    #[cfg(feature = "cond")]
    And(Box<Ast>, Box<Ast>),
    /// Logical OR with short-circuit evaluation.
    #[cfg(feature = "cond")]
    Or(Box<Ast>, Box<Ast>),
    /// Conditional: `if condition then then_expr else else_expr`.
    #[cfg(feature = "cond")]
    If(Box<Ast>, Box<Ast>, Box<Ast>),
}

/// Binary operators for arithmetic and bitwise operations.
///
/// Used in [`Ast::BinOp`] to specify the operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Division (`/`).
    Div,
    /// Exponentiation (`^`), right-associative.
    Pow,
    /// Remainder/modulo (`%`).
    Rem,
    /// Bitwise AND (`&`).
    BitAnd,
    /// Bitwise OR (`|`).
    BitOr,
    /// Left shift (`<<`).
    Shl,
    /// Right shift (`>>`).
    Shr,
}

/// Comparison operators (requires `cond` feature).
///
/// Used in [`Ast::Compare`] to specify the comparison.
/// All comparisons evaluate to `1.0` (true) or `0.0` (false).
#[cfg(feature = "cond")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    /// Less than (`<`).
    Lt,
    /// Less than or equal (`<=`).
    Le,
    /// Greater than (`>`).
    Gt,
    /// Greater than or equal (`>=`).
    Ge,
    /// Equal (`==`).
    Eq,
    /// Not equal (`!=`).
    Ne,
}

/// Unary operators.
///
/// Used in [`Ast::UnaryOp`] to specify the operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Numeric negation (`-x`).
    Neg,
    /// Logical NOT (`not x`), requires `cond` feature.
    /// Returns `1.0` if operand is `0.0`, otherwise `0.0`.
    #[cfg(feature = "cond")]
    Not,
    /// Bitwise NOT (`~x`).
    BitNot,
}

// ============================================================================
// AST Display (produces parseable expressions)
// ============================================================================

impl std::fmt::Display for Ast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ast::Num(n) => {
                if n.is_nan() {
                    write!(f, "(0.0 / 0.0)") // NaN
                } else if n.is_infinite() {
                    if *n > 0.0 {
                        write!(f, "(1.0 / 0.0)") // +Inf
                    } else {
                        write!(f, "(-1.0 / 0.0)") // -Inf
                    }
                } else {
                    write!(f, "{}", n)
                }
            }
            Ast::Var(name) => write!(f, "{}", name),
            Ast::BinOp(op, left, right) => {
                write!(f, "({} {} {})", left, op, right)
            }
            Ast::UnaryOp(op, inner) => {
                write!(f, "({}{})", op, inner)
            }
            #[cfg(feature = "func")]
            Ast::Call(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            #[cfg(feature = "cond")]
            Ast::Compare(op, left, right) => {
                write!(f, "({} {} {})", left, op, right)
            }
            #[cfg(feature = "cond")]
            Ast::And(left, right) => {
                write!(f, "({} and {})", left, right)
            }
            #[cfg(feature = "cond")]
            Ast::Or(left, right) => {
                write!(f, "({} or {})", left, right)
            }
            #[cfg(feature = "cond")]
            Ast::If(cond, then_expr, else_expr) => {
                write!(f, "(if {} then {} else {})", cond, then_expr, else_expr)
            }
        }
    }
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Pow => write!(f, "^"),
            BinOp::Rem => write!(f, "%"),
            BinOp::BitAnd => write!(f, "&"),
            BinOp::BitOr => write!(f, "|"),
            BinOp::Shl => write!(f, "<<"),
            BinOp::Shr => write!(f, ">>"),
        }
    }
}

impl std::fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            #[cfg(feature = "cond")]
            UnaryOp::Not => write!(f, "not "),
            UnaryOp::BitNot => write!(f, "~"),
        }
    }
}

#[cfg(feature = "cond")]
impl std::fmt::Display for CompareOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompareOp::Lt => write!(f, "<"),
            CompareOp::Le => write!(f, "<="),
            CompareOp::Gt => write!(f, ">"),
            CompareOp::Ge => write!(f, ">="),
            CompareOp::Eq => write!(f, "=="),
            CompareOp::Ne => write!(f, "!="),
        }
    }
}

// ============================================================================
// AST Introspection (introspect feature)
// ============================================================================

#[cfg(feature = "introspect")]
impl Ast {
    /// Returns the set of free variables referenced in this AST node.
    ///
    /// Traverses the entire AST and collects all variable names.
    ///
    /// # Example
    ///
    /// ```
    /// use rhizome_dew_core::{Expr, Ast};
    ///
    /// let expr = Expr::parse("sin(x) + y * z").unwrap();
    /// let vars = expr.ast().free_vars();
    /// assert!(vars.contains("x"));
    /// assert!(vars.contains("y"));
    /// assert!(vars.contains("z"));
    /// ```
    pub fn free_vars(&self) -> HashSet<&str> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars<'a>(&'a self, vars: &mut HashSet<&'a str>) {
        match self {
            Ast::Num(_) => {}
            Ast::Var(name) => {
                vars.insert(name.as_str());
            }
            Ast::BinOp(_, left, right) => {
                left.collect_vars(vars);
                right.collect_vars(vars);
            }
            Ast::UnaryOp(_, inner) => {
                inner.collect_vars(vars);
            }
            #[cfg(feature = "func")]
            Ast::Call(_, args) => {
                for arg in args {
                    arg.collect_vars(vars);
                }
            }
            #[cfg(feature = "cond")]
            Ast::Compare(_, left, right) => {
                left.collect_vars(vars);
                right.collect_vars(vars);
            }
            #[cfg(feature = "cond")]
            Ast::And(left, right) => {
                left.collect_vars(vars);
                right.collect_vars(vars);
            }
            #[cfg(feature = "cond")]
            Ast::Or(left, right) => {
                left.collect_vars(vars);
                right.collect_vars(vars);
            }
            #[cfg(feature = "cond")]
            Ast::If(cond, then_expr, else_expr) => {
                cond.collect_vars(vars);
                then_expr.collect_vars(vars);
                else_expr.collect_vars(vars);
            }
        }
    }
}

// ============================================================================
// Parser
// ============================================================================

struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token()?;
        Ok(Self { lexer, current })
    }

    fn advance(&mut self) -> Result<(), ParseError> {
        self.current = self.lexer.next_token()?;
        Ok(())
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        if self.current == expected {
            self.advance()
        } else {
            Err(ParseError::UnexpectedToken(format!("{:?}", self.current)))
        }
    }

    // Precedence (low to high):
    // 1. if/then/else (cond feature)
    // 2. or (cond feature, keyword)
    // 3. and (cond feature, keyword)
    // 4. bit_or (|)
    // 5. bit_and (&)
    // 6. comparison (<, <=, >, >=, ==, !=) (cond feature)
    // 7. shift (<<, >>)
    // 8. add/sub (+, -)
    // 9. mul/div/rem (*, /, %)
    // 10. power (^)
    // 11. unary (-, ~, not)
    // 12. primary

    fn parse_expr(&mut self) -> Result<Ast, ParseError> {
        #[cfg(feature = "cond")]
        {
            self.parse_if()
        }
        #[cfg(not(feature = "cond"))]
        {
            self.parse_bit_or()
        }
    }

    #[cfg(feature = "cond")]
    fn parse_if(&mut self) -> Result<Ast, ParseError> {
        if self.current == Token::If {
            self.advance()?;
            let cond = self.parse_or()?;
            self.expect(Token::Then)?;
            let then_expr = self.parse_if()?; // Allow nested if in then branch
            self.expect(Token::Else)?;
            let else_expr = self.parse_if()?; // Right associative for chained if/else
            Ok(Ast::If(
                Box::new(cond),
                Box::new(then_expr),
                Box::new(else_expr),
            ))
        } else {
            self.parse_or()
        }
    }

    #[cfg(feature = "cond")]
    fn parse_or(&mut self) -> Result<Ast, ParseError> {
        let mut left = self.parse_and()?;

        while self.current == Token::Or {
            self.advance()?;
            let right = self.parse_and()?;
            left = Ast::Or(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    #[cfg(feature = "cond")]
    fn parse_and(&mut self) -> Result<Ast, ParseError> {
        let mut left = self.parse_bit_or()?;

        while self.current == Token::And {
            self.advance()?;
            let right = self.parse_bit_or()?;
            left = Ast::And(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_bit_or(&mut self) -> Result<Ast, ParseError> {
        let mut left = self.parse_bit_and()?;

        while self.current == Token::Pipe {
            self.advance()?;
            let right = self.parse_bit_and()?;
            left = Ast::BinOp(BinOp::BitOr, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_bit_and(&mut self) -> Result<Ast, ParseError> {
        #[cfg(feature = "cond")]
        let mut left = self.parse_compare()?;
        #[cfg(not(feature = "cond"))]
        let mut left = self.parse_shift()?;

        while self.current == Token::Ampersand {
            self.advance()?;
            #[cfg(feature = "cond")]
            let right = self.parse_compare()?;
            #[cfg(not(feature = "cond"))]
            let right = self.parse_shift()?;
            left = Ast::BinOp(BinOp::BitAnd, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    #[cfg(feature = "cond")]
    fn parse_compare(&mut self) -> Result<Ast, ParseError> {
        let left = self.parse_shift()?;

        let op = match &self.current {
            Token::Lt => Some(CompareOp::Lt),
            Token::Le => Some(CompareOp::Le),
            Token::Gt => Some(CompareOp::Gt),
            Token::Ge => Some(CompareOp::Ge),
            Token::Eq => Some(CompareOp::Eq),
            Token::Ne => Some(CompareOp::Ne),
            _ => None,
        };

        if let Some(op) = op {
            self.advance()?;
            let right = self.parse_shift()?;
            Ok(Ast::Compare(op, Box::new(left), Box::new(right)))
        } else {
            Ok(left)
        }
    }

    fn parse_shift(&mut self) -> Result<Ast, ParseError> {
        let mut left = self.parse_add_sub()?;

        loop {
            match &self.current {
                Token::Shl => {
                    self.advance()?;
                    let right = self.parse_add_sub()?;
                    left = Ast::BinOp(BinOp::Shl, Box::new(left), Box::new(right));
                }
                Token::Shr => {
                    self.advance()?;
                    let right = self.parse_add_sub()?;
                    left = Ast::BinOp(BinOp::Shr, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_add_sub(&mut self) -> Result<Ast, ParseError> {
        let mut left = self.parse_mul_div()?;

        loop {
            match &self.current {
                Token::Plus => {
                    self.advance()?;
                    let right = self.parse_mul_div()?;
                    left = Ast::BinOp(BinOp::Add, Box::new(left), Box::new(right));
                }
                Token::Minus => {
                    self.advance()?;
                    let right = self.parse_mul_div()?;
                    left = Ast::BinOp(BinOp::Sub, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_mul_div(&mut self) -> Result<Ast, ParseError> {
        let mut left = self.parse_power()?;

        loop {
            match &self.current {
                Token::Star => {
                    self.advance()?;
                    let right = self.parse_power()?;
                    left = Ast::BinOp(BinOp::Mul, Box::new(left), Box::new(right));
                }
                Token::Slash => {
                    self.advance()?;
                    let right = self.parse_power()?;
                    left = Ast::BinOp(BinOp::Div, Box::new(left), Box::new(right));
                }
                Token::Percent => {
                    self.advance()?;
                    let right = self.parse_power()?;
                    left = Ast::BinOp(BinOp::Rem, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_power(&mut self) -> Result<Ast, ParseError> {
        let base = self.parse_unary()?;

        if self.current == Token::Caret {
            self.advance()?;
            let exp = self.parse_power()?; // Right associative
            Ok(Ast::BinOp(BinOp::Pow, Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    fn parse_unary(&mut self) -> Result<Ast, ParseError> {
        match &self.current {
            Token::Minus => {
                self.advance()?;
                let inner = self.parse_unary()?;
                Ok(Ast::UnaryOp(UnaryOp::Neg, Box::new(inner)))
            }
            Token::Tilde => {
                self.advance()?;
                let inner = self.parse_unary()?;
                Ok(Ast::UnaryOp(UnaryOp::BitNot, Box::new(inner)))
            }
            #[cfg(feature = "cond")]
            Token::Not => {
                self.advance()?;
                let inner = self.parse_unary()?;
                Ok(Ast::UnaryOp(UnaryOp::Not, Box::new(inner)))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Ast, ParseError> {
        match &self.current {
            Token::Number(n) => {
                let n = *n;
                self.advance()?;
                Ok(Ast::Num(n))
            }
            Token::Ident(name) => {
                let name = name.clone();
                self.advance()?;

                // Check if it's a function call (func feature)
                #[cfg(feature = "func")]
                if self.current == Token::LParen {
                    self.advance()?;
                    let mut args = Vec::new();
                    if self.current != Token::RParen {
                        args.push(self.parse_expr()?);
                        while self.current == Token::Comma {
                            self.advance()?;
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(Token::RParen)?;
                    return Ok(Ast::Call(name, args));
                }

                // It's a variable
                Ok(Ast::Var(name))
            }
            Token::LParen => {
                self.advance()?;
                let inner = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(inner)
            }
            Token::Eof => Err(ParseError::UnexpectedEnd),
            _ => Err(ParseError::UnexpectedToken(format!("{:?}", self.current))),
        }
    }
}

// ============================================================================
// Expression
// ============================================================================

/// A parsed expression that can be evaluated or inspected.
///
/// `Expr` is the main entry point for the expression language. Parse a string
/// with [`Expr::parse()`], then either evaluate it with [`Expr::eval()`] or
/// inspect the AST with [`Expr::ast()`].
///
/// # Example
///
/// ```
/// use rhizome_dew_core::Expr;
/// use std::collections::HashMap;
///
/// // Parse an expression
/// let expr = Expr::parse("x^2 + 2*x + 1").unwrap();
///
/// // Evaluate with different variable values
/// let mut vars = HashMap::new();
/// vars.insert("x".to_string(), 3.0);
/// # #[cfg(not(feature = "func"))]
/// assert_eq!(expr.eval(&vars).unwrap(), 16.0);  // 9 + 6 + 1
/// # #[cfg(feature = "func")]
/// # assert_eq!(expr.eval(&vars, &rhizome_dew_core::FunctionRegistry::new()).unwrap(), 16.0);
///
/// vars.insert("x".to_string(), 0.0);
/// # #[cfg(not(feature = "func"))]
/// assert_eq!(expr.eval(&vars).unwrap(), 1.0);   // 0 + 0 + 1
/// # #[cfg(feature = "func")]
/// # assert_eq!(expr.eval(&vars, &rhizome_dew_core::FunctionRegistry::new()).unwrap(), 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct Expr {
    ast: Ast,
}

impl Expr {
    /// Parses an expression from a string.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError`] if the input is not a valid expression:
    /// - [`ParseError::UnexpectedChar`] for invalid characters
    /// - [`ParseError::UnexpectedEnd`] for incomplete expressions
    /// - [`ParseError::UnexpectedToken`] for syntax errors
    /// - [`ParseError::InvalidNumber`] for malformed numeric literals
    ///
    /// # Example
    ///
    /// ```
    /// use rhizome_dew_core::{Expr, ParseError};
    ///
    /// // Valid expression
    /// assert!(Expr::parse("1 + 2").is_ok());
    ///
    /// // Invalid: unexpected character
    /// assert!(matches!(Expr::parse("1 @ 2"), Err(ParseError::UnexpectedChar('@'))));
    ///
    /// // Invalid: incomplete expression
    /// assert!(matches!(Expr::parse("1 +"), Err(ParseError::UnexpectedEnd)));
    /// ```
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let mut parser = Parser::new(input)?;
        let ast = parser.parse_expr()?;
        if parser.current != Token::Eof {
            return Err(ParseError::UnexpectedToken(format!("{:?}", parser.current)));
        }
        Ok(Self { ast })
    }

    /// Returns a reference to the parsed AST.
    ///
    /// Use this to inspect the expression structure or to compile it to
    /// a different target (WGSL, Lua, etc.).
    pub fn ast(&self) -> &Ast {
        &self.ast
    }

    /// Returns the set of free variables referenced in the expression.
    ///
    /// This is useful for determining which variables need to be provided
    /// at evaluation time, or for building dependency graphs.
    ///
    /// # Example
    ///
    /// ```
    /// use rhizome_dew_core::Expr;
    ///
    /// let expr = Expr::parse("x * 2 + y").unwrap();
    /// let vars = expr.free_vars();
    /// assert!(vars.contains("x"));
    /// assert!(vars.contains("y"));
    /// assert_eq!(vars.len(), 2);
    /// ```
    #[cfg(feature = "introspect")]
    pub fn free_vars(&self) -> HashSet<&str> {
        self.ast.free_vars()
    }

    /// Evaluates the expression with the given variables and function registry.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError`] if evaluation fails:
    /// - [`EvalError::UnknownVariable`] if a variable is not in `vars`
    /// - [`EvalError::UnknownFunction`] if a function is not in `funcs`
    /// - [`EvalError::WrongArgCount`] if a function is called with wrong arity
    #[cfg(feature = "func")]
    pub fn eval(
        &self,
        vars: &HashMap<String, f32>,
        funcs: &FunctionRegistry,
    ) -> Result<f32, EvalError> {
        eval_ast(&self.ast, vars, funcs)
    }

    /// Evaluates the expression with the given variables.
    ///
    /// This version is available when the `func` feature is disabled.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::UnknownVariable`] if a variable is not in `vars`.
    #[cfg(not(feature = "func"))]
    pub fn eval(&self, vars: &HashMap<String, f32>) -> Result<f32, EvalError> {
        eval_ast(&self.ast, vars)
    }
}

#[cfg(feature = "func")]
fn eval_ast(
    ast: &Ast,
    vars: &HashMap<String, f32>,
    funcs: &FunctionRegistry,
) -> Result<f32, EvalError> {
    match ast {
        Ast::Num(n) => Ok(*n as f32),
        Ast::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| EvalError::UnknownVariable(name.clone())),
        Ast::BinOp(op, l, r) => {
            let l = eval_ast(l, vars, funcs)?;
            let r = eval_ast(r, vars, funcs)?;
            match op {
                BinOp::Add => Ok(l + r),
                BinOp::Sub => Ok(l - r),
                BinOp::Mul => Ok(l * r),
                BinOp::Div => Ok(l / r),
                BinOp::Pow => Ok(l.powf(r)),
                BinOp::Rem => Ok(l % r),
                BinOp::BitAnd => Err(EvalError::UnsupportedOperation("&".to_string())),
                BinOp::BitOr => Err(EvalError::UnsupportedOperation("|".to_string())),
                BinOp::Shl => Err(EvalError::UnsupportedOperation("<<".to_string())),
                BinOp::Shr => Err(EvalError::UnsupportedOperation(">>".to_string())),
            }
        }
        Ast::UnaryOp(op, inner) => {
            let v = eval_ast(inner, vars, funcs)?;
            match op {
                UnaryOp::Neg => Ok(-v),
                UnaryOp::BitNot => Err(EvalError::UnsupportedOperation("~".to_string())),
                #[cfg(feature = "cond")]
                UnaryOp::Not => {
                    if v == 0.0 {
                        Ok(1.0)
                    } else {
                        Ok(0.0)
                    }
                }
            }
        }
        #[cfg(feature = "cond")]
        Ast::Compare(op, l, r) => {
            let l = eval_ast(l, vars, funcs)?;
            let r = eval_ast(r, vars, funcs)?;
            let result = match op {
                CompareOp::Lt => l < r,
                CompareOp::Le => l <= r,
                CompareOp::Gt => l > r,
                CompareOp::Ge => l >= r,
                CompareOp::Eq => l == r,
                CompareOp::Ne => l != r,
            };
            Ok(if result { 1.0 } else { 0.0 })
        }
        #[cfg(feature = "cond")]
        Ast::And(l, r) => {
            let l = eval_ast(l, vars, funcs)?;
            if l == 0.0 {
                Ok(0.0) // Short-circuit
            } else {
                let r = eval_ast(r, vars, funcs)?;
                Ok(if r != 0.0 { 1.0 } else { 0.0 })
            }
        }
        #[cfg(feature = "cond")]
        Ast::Or(l, r) => {
            let l = eval_ast(l, vars, funcs)?;
            if l != 0.0 {
                Ok(1.0) // Short-circuit
            } else {
                let r = eval_ast(r, vars, funcs)?;
                Ok(if r != 0.0 { 1.0 } else { 0.0 })
            }
        }
        #[cfg(feature = "cond")]
        Ast::If(cond, then_expr, else_expr) => {
            let cond = eval_ast(cond, vars, funcs)?;
            if cond != 0.0 {
                eval_ast(then_expr, vars, funcs)
            } else {
                eval_ast(else_expr, vars, funcs)
            }
        }
        Ast::Call(name, args) => {
            let func = funcs
                .get(name)
                .ok_or_else(|| EvalError::UnknownFunction(name.clone()))?;

            if args.len() != func.arg_count() {
                return Err(EvalError::WrongArgCount {
                    func: name.clone(),
                    expected: func.arg_count(),
                    got: args.len(),
                });
            }

            let arg_values: Vec<f32> = args
                .iter()
                .map(|a| eval_ast(a, vars, funcs))
                .collect::<Result<_, _>>()?;

            Ok(func.call(&arg_values))
        }
    }
}

#[cfg(not(feature = "func"))]
fn eval_ast(ast: &Ast, vars: &HashMap<String, f32>) -> Result<f32, EvalError> {
    match ast {
        Ast::Num(n) => Ok(*n as f32),
        Ast::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| EvalError::UnknownVariable(name.clone())),
        Ast::BinOp(op, l, r) => {
            let l = eval_ast(l, vars)?;
            let r = eval_ast(r, vars)?;
            match op {
                BinOp::Add => Ok(l + r),
                BinOp::Sub => Ok(l - r),
                BinOp::Mul => Ok(l * r),
                BinOp::Div => Ok(l / r),
                BinOp::Pow => Ok(l.powf(r)),
                BinOp::Rem => Ok(l % r),
                BinOp::BitAnd => Err(EvalError::UnsupportedOperation("&".to_string())),
                BinOp::BitOr => Err(EvalError::UnsupportedOperation("|".to_string())),
                BinOp::Shl => Err(EvalError::UnsupportedOperation("<<".to_string())),
                BinOp::Shr => Err(EvalError::UnsupportedOperation(">>".to_string())),
            }
        }
        Ast::UnaryOp(op, inner) => {
            let v = eval_ast(inner, vars)?;
            match op {
                UnaryOp::Neg => Ok(-v),
                UnaryOp::BitNot => Err(EvalError::UnsupportedOperation("~".to_string())),
                #[cfg(feature = "cond")]
                UnaryOp::Not => {
                    if v == 0.0 {
                        Ok(1.0)
                    } else {
                        Ok(0.0)
                    }
                }
            }
        }
        #[cfg(feature = "cond")]
        Ast::Compare(op, l, r) => {
            let l = eval_ast(l, vars)?;
            let r = eval_ast(r, vars)?;
            let result = match op {
                CompareOp::Lt => l < r,
                CompareOp::Le => l <= r,
                CompareOp::Gt => l > r,
                CompareOp::Ge => l >= r,
                CompareOp::Eq => l == r,
                CompareOp::Ne => l != r,
            };
            Ok(if result { 1.0 } else { 0.0 })
        }
        #[cfg(feature = "cond")]
        Ast::And(l, r) => {
            let l = eval_ast(l, vars)?;
            if l == 0.0 {
                Ok(0.0) // Short-circuit
            } else {
                let r = eval_ast(r, vars)?;
                Ok(if r != 0.0 { 1.0 } else { 0.0 })
            }
        }
        #[cfg(feature = "cond")]
        Ast::Or(l, r) => {
            let l = eval_ast(l, vars)?;
            if l != 0.0 {
                Ok(1.0) // Short-circuit
            } else {
                let r = eval_ast(r, vars)?;
                Ok(if r != 0.0 { 1.0 } else { 0.0 })
            }
        }
        #[cfg(feature = "cond")]
        Ast::If(cond, then_expr, else_expr) => {
            let cond = eval_ast(cond, vars)?;
            if cond != 0.0 {
                eval_ast(then_expr, vars)
            } else {
                eval_ast(else_expr, vars)
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

    #[cfg(feature = "func")]
    fn eval(expr_str: &str, vars: &[(&str, f32)]) -> f32 {
        let registry = FunctionRegistry::new();
        let expr = Expr::parse(expr_str).unwrap();
        let var_map: HashMap<String, f32> = vars.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        expr.eval(&var_map, &registry).unwrap()
    }

    #[cfg(not(feature = "func"))]
    fn eval(expr_str: &str, vars: &[(&str, f32)]) -> f32 {
        let expr = Expr::parse(expr_str).unwrap();
        let var_map: HashMap<String, f32> = vars.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        expr.eval(&var_map).unwrap()
    }

    #[test]
    fn test_parse_number() {
        assert_eq!(eval("42", &[]), 42.0);
    }

    #[test]
    fn test_parse_float() {
        assert!((eval("1.234", &[]) - 1.234).abs() < 0.001);
    }

    #[test]
    fn test_parse_variable() {
        assert_eq!(eval("x", &[("x", 5.0)]), 5.0);
        assert_eq!(eval("foo", &[("foo", 3.0)]), 3.0);
    }

    #[test]
    fn test_parse_add() {
        assert_eq!(eval("1 + 2", &[]), 3.0);
    }

    #[test]
    fn test_parse_mul() {
        assert_eq!(eval("3 * 4", &[]), 12.0);
    }

    #[test]
    fn test_precedence() {
        assert_eq!(eval("2 + 3 * 4", &[]), 14.0);
    }

    #[test]
    fn test_parentheses() {
        assert_eq!(eval("(2 + 3) * 4", &[]), 20.0);
    }

    #[test]
    fn test_negation() {
        assert_eq!(eval("-5", &[]), -5.0);
    }

    #[test]
    fn test_power() {
        assert_eq!(eval("2 ^ 3", &[]), 8.0);
    }

    #[cfg(feature = "func")]
    #[test]
    fn test_unknown_variable() {
        let registry = FunctionRegistry::new();
        let expr = Expr::parse("unknown").unwrap();
        let vars = HashMap::new();
        let result = expr.eval(&vars, &registry);
        assert!(matches!(result, Err(EvalError::UnknownVariable(_))));
    }

    #[cfg(not(feature = "func"))]
    #[test]
    fn test_unknown_variable() {
        let expr = Expr::parse("unknown").unwrap();
        let vars = HashMap::new();
        let result = expr.eval(&vars);
        assert!(matches!(result, Err(EvalError::UnknownVariable(_))));
    }

    #[cfg(feature = "func")]
    #[test]
    fn test_unknown_function() {
        let registry = FunctionRegistry::new();
        let expr = Expr::parse("unknown(1)").unwrap();
        let vars = HashMap::new();
        let result = expr.eval(&vars, &registry);
        assert!(matches!(result, Err(EvalError::UnknownFunction(_))));
    }

    #[cfg(feature = "func")]
    #[test]
    fn test_custom_function() {
        struct Double;
        impl ExprFn for Double {
            fn name(&self) -> &str {
                "double"
            }
            fn arg_count(&self) -> usize {
                1
            }
            fn call(&self, args: &[f32]) -> f32 {
                args[0] * 2.0
            }
        }

        let mut registry = FunctionRegistry::new();
        registry.register(Double);

        let expr = Expr::parse("double(5)").unwrap();
        let vars = HashMap::new();
        assert_eq!(expr.eval(&vars, &registry).unwrap(), 10.0);
    }

    #[cfg(feature = "func")]
    #[test]
    fn test_zero_arg_function() {
        struct Pi;
        impl ExprFn for Pi {
            fn name(&self) -> &str {
                "pi"
            }
            fn arg_count(&self) -> usize {
                0
            }
            fn call(&self, _args: &[f32]) -> f32 {
                std::f32::consts::PI
            }
        }

        let mut registry = FunctionRegistry::new();
        registry.register(Pi);

        let expr = Expr::parse("pi()").unwrap();
        let vars = HashMap::new();
        assert!((expr.eval(&vars, &registry).unwrap() - std::f32::consts::PI).abs() < 0.001);
    }

    #[cfg(feature = "func")]
    #[test]
    fn test_wrong_arg_count() {
        struct OneArg;
        impl ExprFn for OneArg {
            fn name(&self) -> &str {
                "one_arg"
            }
            fn arg_count(&self) -> usize {
                1
            }
            fn call(&self, args: &[f32]) -> f32 {
                args[0]
            }
        }

        let mut registry = FunctionRegistry::new();
        registry.register(OneArg);

        let expr = Expr::parse("one_arg(1, 2)").unwrap();
        let vars = HashMap::new();
        let result = expr.eval(&vars, &registry);
        assert!(matches!(result, Err(EvalError::WrongArgCount { .. })));
    }

    #[cfg(feature = "func")]
    #[test]
    fn test_complex_expression() {
        struct Add;
        impl ExprFn for Add {
            fn name(&self) -> &str {
                "add"
            }
            fn arg_count(&self) -> usize {
                2
            }
            fn call(&self, args: &[f32]) -> f32 {
                args[0] + args[1]
            }
        }

        let mut registry = FunctionRegistry::new();
        registry.register(Add);

        let expr = Expr::parse("add(x * 2, y + 1)").unwrap();
        let vars: HashMap<String, f32> = [("x".to_string(), 3.0), ("y".to_string(), 4.0)].into();
        assert_eq!(expr.eval(&vars, &registry).unwrap(), 11.0); // (3*2) + (4+1) = 11
    }

    // Comparison tests (cond feature)
    #[cfg(feature = "cond")]
    #[test]
    fn test_compare_lt() {
        assert_eq!(eval("1 < 2", &[]), 1.0);
        assert_eq!(eval("2 < 1", &[]), 0.0);
        assert_eq!(eval("1 < 1", &[]), 0.0);
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_compare_le() {
        assert_eq!(eval("1 <= 2", &[]), 1.0);
        assert_eq!(eval("2 <= 1", &[]), 0.0);
        assert_eq!(eval("1 <= 1", &[]), 1.0);
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_compare_gt() {
        assert_eq!(eval("2 > 1", &[]), 1.0);
        assert_eq!(eval("1 > 2", &[]), 0.0);
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_compare_ge() {
        assert_eq!(eval("2 >= 1", &[]), 1.0);
        assert_eq!(eval("1 >= 1", &[]), 1.0);
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_compare_eq() {
        assert_eq!(eval("1 == 1", &[]), 1.0);
        assert_eq!(eval("1 == 2", &[]), 0.0);
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_compare_ne() {
        assert_eq!(eval("1 != 2", &[]), 1.0);
        assert_eq!(eval("1 != 1", &[]), 0.0);
    }

    // Boolean logic tests (cond feature)
    #[cfg(feature = "cond")]
    #[test]
    fn test_and() {
        assert_eq!(eval("1 and 1", &[]), 1.0);
        assert_eq!(eval("1 and 0", &[]), 0.0);
        assert_eq!(eval("0 and 1", &[]), 0.0);
        assert_eq!(eval("0 and 0", &[]), 0.0);
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_or() {
        assert_eq!(eval("1 or 1", &[]), 1.0);
        assert_eq!(eval("1 or 0", &[]), 1.0);
        assert_eq!(eval("0 or 1", &[]), 1.0);
        assert_eq!(eval("0 or 0", &[]), 0.0);
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_not() {
        assert_eq!(eval("not 0", &[]), 1.0);
        assert_eq!(eval("not 1", &[]), 0.0);
        assert_eq!(eval("not 5", &[]), 0.0); // any non-zero is truthy
    }

    // Conditional tests (cond feature)
    #[cfg(feature = "cond")]
    #[test]
    fn test_if_then_else() {
        assert_eq!(eval("if 1 then 10 else 20", &[]), 10.0);
        assert_eq!(eval("if 0 then 10 else 20", &[]), 20.0);
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_if_with_comparison() {
        assert_eq!(eval("if x > 5 then 1 else 0", &[("x", 10.0)]), 1.0);
        assert_eq!(eval("if x > 5 then 1 else 0", &[("x", 3.0)]), 0.0);
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_nested_if() {
        // if x > 0 then (if x > 10 then 2 else 1) else 0
        assert_eq!(
            eval(
                "if x > 0 then if x > 10 then 2 else 1 else 0",
                &[("x", 15.0)]
            ),
            2.0
        );
        assert_eq!(
            eval(
                "if x > 0 then if x > 10 then 2 else 1 else 0",
                &[("x", 5.0)]
            ),
            1.0
        );
        assert_eq!(
            eval(
                "if x > 0 then if x > 10 then 2 else 1 else 0",
                &[("x", -1.0)]
            ),
            0.0
        );
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_compound_boolean() {
        assert_eq!(eval("x > 0 and x < 10", &[("x", 5.0)]), 1.0);
        assert_eq!(eval("x > 0 and x < 10", &[("x", 15.0)]), 0.0);
        assert_eq!(eval("x < 0 or x > 10", &[("x", 5.0)]), 0.0);
        assert_eq!(eval("x < 0 or x > 10", &[("x", 15.0)]), 1.0);
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_precedence_compare_vs_arithmetic() {
        // 1 + 2 < 4 should be (1 + 2) < 4, not 1 + (2 < 4)
        assert_eq!(eval("1 + 2 < 4", &[]), 1.0);
        assert_eq!(eval("1 + 2 < 3", &[]), 0.0);
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_precedence_and_vs_or() {
        // a or b and c should be a or (b and c)
        assert_eq!(eval("1 or 0 and 0", &[]), 1.0); // 1 or (0 and 0) = 1 or 0 = 1
        assert_eq!(eval("0 or 1 and 1", &[]), 1.0); // 0 or (1 and 1) = 0 or 1 = 1
        assert_eq!(eval("0 or 0 and 1", &[]), 0.0); // 0 or (0 and 1) = 0 or 0 = 0
    }

    // Free variables tests (introspect feature)
    #[cfg(feature = "introspect")]
    #[test]
    fn test_free_vars_simple() {
        let expr = Expr::parse("x + y").unwrap();
        let vars = expr.free_vars();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[cfg(feature = "introspect")]
    #[test]
    fn test_free_vars_no_vars() {
        let expr = Expr::parse("1 + 2 * 3").unwrap();
        let vars = expr.free_vars();
        assert!(vars.is_empty());
    }

    #[cfg(feature = "introspect")]
    #[test]
    fn test_free_vars_duplicates() {
        let expr = Expr::parse("x + x * x").unwrap();
        let vars = expr.free_vars();
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[cfg(all(feature = "introspect", feature = "func"))]
    #[test]
    fn test_free_vars_in_call() {
        let expr = Expr::parse("sin(x) + cos(y)").unwrap();
        let vars = expr.free_vars();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[cfg(all(feature = "introspect", feature = "cond"))]
    #[test]
    fn test_free_vars_in_conditional() {
        let expr = Expr::parse("if a > b then x else y").unwrap();
        let vars = expr.free_vars();
        assert_eq!(vars.len(), 4);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    // AST Display / roundtrip tests
    #[test]
    fn test_ast_display_simple() {
        let expr = Expr::parse("1 + 2").unwrap();
        let s = expr.ast().to_string();
        assert_eq!(s, "(1 + 2)");
    }

    #[test]
    fn test_ast_display_nested() {
        let expr = Expr::parse("1 + 2 * 3").unwrap();
        let s = expr.ast().to_string();
        // Should be fully parenthesized
        assert_eq!(s, "(1 + (2 * 3))");
    }

    #[test]
    fn test_ast_roundtrip() {
        let cases = [
            "1 + 2",
            "x * y",
            "1 + 2 * 3",
            "(1 + 2) * 3",
            "-x",
            "x ^ 2",
            "2 ^ 3 ^ 4", // right-associative
        ];
        for case in cases {
            let expr1 = Expr::parse(case).unwrap();
            let stringified = expr1.ast().to_string();
            let expr2 = Expr::parse(&stringified).unwrap();
            let stringified2 = expr2.ast().to_string();
            assert_eq!(stringified, stringified2, "Roundtrip failed for: {}", case);
        }
    }

    #[cfg(feature = "func")]
    #[test]
    fn test_ast_roundtrip_func() {
        let cases = ["sin(x)", "foo(a, b, c)", "f()"];
        for case in cases {
            let expr1 = Expr::parse(case).unwrap();
            let stringified = expr1.ast().to_string();
            let expr2 = Expr::parse(&stringified).unwrap();
            let stringified2 = expr2.ast().to_string();
            assert_eq!(stringified, stringified2, "Roundtrip failed for: {}", case);
        }
    }

    #[cfg(feature = "cond")]
    #[test]
    fn test_ast_roundtrip_cond() {
        let cases = [
            "x < y",
            "x and y",
            "x or y",
            "not x",
            "if x then y else z",
            "if a > b then x else y",
        ];
        for case in cases {
            let expr1 = Expr::parse(case).unwrap();
            let stringified = expr1.ast().to_string();
            let expr2 = Expr::parse(&stringified).unwrap();
            let stringified2 = expr2.ast().to_string();
            assert_eq!(stringified, stringified2, "Roundtrip failed for: {}", case);
        }
    }
}

// ============================================================================
// Property-based tests (proptest)
// ============================================================================

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating valid expression strings
    fn expr_strategy() -> impl Strategy<Value = String> {
        // Generate simple arithmetic expressions
        let num = prop::num::f32::NORMAL.prop_map(|n| format!("{:.6}", n));
        let var = prop::sample::select(vec!["x", "y", "z", "a", "b"]).prop_map(String::from);

        // Operators
        let binop = prop::sample::select(vec!["+", "-", "*", "/"]);

        // Combine into expressions
        prop::strategy::Union::new(vec![
            num.clone().boxed(),
            var.clone().boxed(),
            (num.clone(), binop.clone(), num.clone())
                .prop_map(|(l, op, r)| format!("({} {} {})", l, op, r))
                .boxed(),
            (var.clone(), binop.clone(), num.clone())
                .prop_map(|(l, op, r)| format!("({} {} {})", l, op, r))
                .boxed(),
            (num.clone(), binop, var.clone())
                .prop_map(|(l, op, r)| format!("({} {} {})", l, op, r))
                .boxed(),
        ])
    }

    proptest! {
        /// Parser should not panic on arbitrary input
        #[test]
        fn parse_never_panics(s in ".*") {
            // Just check that parsing doesn't panic, result can be Ok or Err
            let _ = Expr::parse(&s);
        }

        /// Valid expressions should parse successfully
        #[test]
        fn valid_expr_parses(expr in expr_strategy()) {
            let result = Expr::parse(&expr);
            prop_assert!(result.is_ok(), "Failed to parse: {}", expr);
        }

        /// Numbers round-trip correctly
        #[test]
        fn number_roundtrip(n in prop::num::f64::NORMAL) {
            let expr_str = format!("{:.6}", n);
            if let Ok(expr) = Expr::parse(&expr_str) {
                // The parsed number should be close to the original
                if let Ast::Num(parsed) = expr.ast() {
                    let diff = (parsed - n).abs();
                    prop_assert!(diff < 0.001 || diff / n.abs() < 0.001,
                        "Number mismatch: {} vs {}", n, parsed);
                }
            }
        }

        /// Evaluation with valid variables shouldn't panic
        #[test]
        #[cfg(not(feature = "func"))]
        fn eval_with_vars_no_panic(
            x in prop::num::f32::NORMAL,
            y in prop::num::f32::NORMAL,
            expr in expr_strategy()
        ) {
            if let Ok(parsed) = Expr::parse(&expr) {
                let vars: HashMap<String, f32> = [
                    ("x".into(), x),
                    ("y".into(), y),
                    ("z".into(), 1.0),
                    ("a".into(), 2.0),
                    ("b".into(), 3.0),
                ].into();
                // Just check it doesn't panic, result can be anything (including NaN/Inf)
                let _ = parsed.eval(&vars);
            }
        }

        /// Evaluation with valid variables shouldn't panic (func feature)
        #[test]
        #[cfg(feature = "func")]
        fn eval_with_vars_no_panic_func(
            x in prop::num::f32::NORMAL,
            y in prop::num::f32::NORMAL,
            expr in expr_strategy()
        ) {
            if let Ok(parsed) = Expr::parse(&expr) {
                let vars: HashMap<String, f32> = [
                    ("x".into(), x),
                    ("y".into(), y),
                    ("z".into(), 1.0),
                    ("a".into(), 2.0),
                    ("b".into(), 3.0),
                ].into();
                let registry = FunctionRegistry::new();
                let _ = parsed.eval(&vars, &registry);
            }
        }

        /// Negation is its own inverse
        #[test]
        #[cfg(not(feature = "func"))]
        fn negation_inverse(n in prop::num::f32::NORMAL) {
            let expr = Expr::parse(&format!("--{:.6}", n)).unwrap();
            let vars = HashMap::new();
            let result = expr.eval(&vars).unwrap();
            prop_assert!((result - n).abs() < 0.01,
                "Double negation failed: --{} = {}", n, result);
        }

        /// Negation is its own inverse (func feature)
        #[test]
        #[cfg(feature = "func")]
        fn negation_inverse_func(n in prop::num::f32::NORMAL) {
            let expr = Expr::parse(&format!("--{:.6}", n)).unwrap();
            let vars = HashMap::new();
            let registry = FunctionRegistry::new();
            let result = expr.eval(&vars, &registry).unwrap();
            prop_assert!((result - n).abs() < 0.01,
                "Double negation failed: --{} = {}", n, result);
        }
    }
}
