//! Backend helpers for conditionals, comparisons, and boolean logic.
//!
//! This crate provides code generation utilities that domain crates use to emit
//! conditional expressions to various backends (WGSL, Lua, Cranelift). Rather than
//! each domain crate reimplementing backend-specific conditional logic, they can
//! delegate to these shared helpers.
//!
//! # Architecture
//!
//! ```text
//! dew-core (AST: Compare, And, Or, If)
//!     │
//!     ▼
//! dew-cond (backend helpers)
//!     │
//!     ├── wgsl::emit_compare, emit_if, ...
//!     ├── glsl::emit_compare, emit_if, ...
//!     ├── rust::emit_compare, emit_if, ...
//!     ├── lua::emit_compare, emit_if, ...
//!     └── cranelift::emit_compare, emit_if, ...
//!     │
//!     ▼
//! domain crates (dew-scalar, dew-linalg, dew-complex, dew-quaternion)
//!     use dew-cond helpers in their backend modules
//! ```
//!
//! # Features
//!
//! | Feature       | Description                           |
//! |---------------|---------------------------------------|
//! | `wgsl`        | WGSL code generation helpers          |
//! | `lua-codegen` | Lua code generation (pure Rust)       |
//! | `lua`         | Alias for lua-codegen                 |
//! | `cranelift`   | Cranelift JIT compilation helpers     |
//!
//! # Backend Modules
//!
//! Each backend module provides the same set of functions with backend-specific implementations:
//!
//! | Function         | Description                                         |
//! |------------------|-----------------------------------------------------|
//! | `emit_compare`   | Comparison operators (`<`, `<=`, `>`, `>=`, `==`, `!=`) |
//! | `emit_and`       | Logical AND                                         |
//! | `emit_or`        | Logical OR                                          |
//! | `emit_not`       | Logical NOT                                         |
//! | `emit_if`        | Conditional expression (if/then/else)               |
//! | `scalar_to_bool` | Convert numeric value to boolean                    |
//! | `bool_to_scalar` | Convert boolean to numeric (1.0/0.0)                |
//!
//! # Examples
//!
//! ## WGSL Backend
//!
//! ```ignore
//! use rhizome_dew_cond::{wgsl, CompareOp};
//!
//! // Comparison
//! let code = wgsl::emit_compare(CompareOp::Lt, "a", "b");
//! assert_eq!(code, "(a < b)");
//!
//! // Conditional (uses WGSL's select function)
//! let code = wgsl::emit_if("cond", "then_val", "else_val");
//! assert_eq!(code, "select(else_val, then_val, cond)");
//!
//! // Boolean logic
//! let code = wgsl::emit_and("x > 0.0", "y > 0.0");
//! assert_eq!(code, "(x > 0.0 && y > 0.0)");
//! ```
//!
//! ## Lua Backend
//!
//! ```ignore
//! use rhizome_dew_cond::{lua, CompareOp};
//!
//! // Comparison (note: Lua uses ~= for not-equal)
//! let code = lua::emit_compare(CompareOp::Ne, "a", "b");
//! assert_eq!(code, "(a ~= b)");
//!
//! // Conditional (uses Lua's and/or idiom)
//! let code = lua::emit_if("cond", "then_val", "else_val");
//! assert_eq!(code, "(cond and then_val or else_val)");
//!
//! // Boolean logic
//! let code = lua::emit_not("flag");
//! assert_eq!(code, "(not flag)");
//! ```
//!
//! # Boolean Representation
//!
//! Dew expressions use scalars for boolean values (0.0 = false, non-zero = true).
//! The `scalar_to_bool` and `bool_to_scalar` functions handle conversions between
//! the numeric representation and backend-native booleans.
//!
//! ```ignore
//! use rhizome_dew_cond::wgsl;
//!
//! // Convert scalar to boolean for conditions
//! let bool_expr = wgsl::scalar_to_bool("x");
//! assert_eq!(bool_expr, "(x != 0.0)");
//!
//! // Convert boolean back to scalar
//! let scalar_expr = wgsl::bool_to_scalar("flag");
//! assert_eq!(scalar_expr, "select(0.0, 1.0, flag)");
//! ```

pub use rhizome_dew_core::CompareOp;

#[cfg(feature = "wgsl")]
pub mod wgsl;

#[cfg(feature = "glsl")]
pub mod glsl;

#[cfg(feature = "rust")]
pub mod rust;

#[cfg(any(feature = "lua", feature = "lua-codegen"))]
pub mod lua;

#[cfg(feature = "cranelift")]
pub mod cranelift;
