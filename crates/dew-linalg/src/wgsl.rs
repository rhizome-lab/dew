//! WGSL code generation for linalg expressions.
//!
//! Emits WGSL code with proper type handling for vectors and matrices.

use crate::Type;
use rhizome_dew_cond::wgsl as cond;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

/// Error during WGSL code generation.
#[derive(Debug, Clone, PartialEq)]
pub enum WgslError {
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
    /// Feature not supported in expression-only codegen.
    UnsupportedFeature(String),
}

impl std::fmt::Display for WgslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslError::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            WgslError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            WgslError::TypeMismatch { op, left, right } => {
                write!(f, "type mismatch for {op}: {left} vs {right}")
            }
            WgslError::UnsupportedType(t) => write!(f, "unsupported type: {t}"),
            WgslError::UnsupportedTypeForConditional(t) => {
                write!(f, "conditionals require scalar type, got {t}")
            }
            WgslError::UnsupportedFeature(feat) => {
                write!(f, "unsupported feature in expression codegen: {feat}")
            }
        }
    }
}

impl std::error::Error for WgslError {}

/// Numeric type for code generation (scalar element type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumericType {
    #[default]
    F32,
    I32,
    U32,
}

impl NumericType {
    /// WGSL scalar type name.
    pub fn wgsl_scalar(&self) -> &'static str {
        match self {
            NumericType::F32 => "f32",
            NumericType::I32 => "i32",
            NumericType::U32 => "u32",
        }
    }
}

/// Convert a Type to its WGSL representation with the given numeric type.
pub fn type_to_wgsl_with(t: Type, numeric: NumericType) -> String {
    let scalar = numeric.wgsl_scalar();
    match t {
        Type::Scalar => scalar.to_string(),
        Type::Vec2 => format!("vec2<{scalar}>"),
        #[cfg(feature = "3d")]
        Type::Vec3 => format!("vec3<{scalar}>"),
        #[cfg(feature = "4d")]
        Type::Vec4 => format!("vec4<{scalar}>"),
        Type::Mat2 => format!("mat2x2<{scalar}>"),
        #[cfg(feature = "3d")]
        Type::Mat3 => format!("mat3x3<{scalar}>"),
        #[cfg(feature = "4d")]
        Type::Mat4 => format!("mat4x4<{scalar}>"),
    }
}

/// Convert a Type to its WGSL representation (defaults to f32).
pub fn type_to_wgsl(t: Type) -> &'static str {
    match t {
        Type::Scalar => "f32",
        Type::Vec2 => "vec2<f32>",
        #[cfg(feature = "3d")]
        Type::Vec3 => "vec3<f32>",
        #[cfg(feature = "4d")]
        Type::Vec4 => "vec4<f32>",
        Type::Mat2 => "mat2x2<f32>",
        #[cfg(feature = "3d")]
        Type::Mat3 => "mat3x3<f32>",
        #[cfg(feature = "4d")]
        Type::Mat4 => "mat4x4<f32>",
    }
}

/// Result of WGSL emission: code string and its type.
pub struct WgslExpr {
    pub code: String,
    pub typ: Type,
}

/// Result of emitting code with statement support.
struct Emission {
    statements: Vec<String>,
    expr: String,
    typ: Type,
}

impl Emission {
    fn expr_only(expr: String, typ: Type) -> Self {
        Self {
            statements: vec![],
            expr,
            typ,
        }
    }

    fn with_statements(statements: Vec<String>, expr: String, typ: Type) -> Self {
        Self {
            statements,
            expr,
            typ,
        }
    }
}

/// Format a numeric literal for WGSL.
fn format_literal(n: f64, numeric: NumericType) -> String {
    match numeric {
        NumericType::F32 => format!("{n:.10}"),
        NumericType::I32 => format!("{}", n as i32),
        NumericType::U32 => format!("{}u", n as u32),
    }
}

/// Emit WGSL code for an AST with type propagation (defaults to f32).
pub fn emit_wgsl(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<WgslExpr, WgslError> {
    emit_wgsl_with(ast, var_types, NumericType::F32)
}

/// Emit WGSL code for an AST with specified numeric type.
pub fn emit_wgsl_with(
    ast: &Ast,
    var_types: &HashMap<String, Type>,
    numeric: NumericType,
) -> Result<WgslExpr, WgslError> {
    match ast {
        Ast::Num(n) => Ok(WgslExpr {
            code: format_literal(*n, numeric),
            typ: Type::Scalar,
        }),

        Ast::Var(name) => {
            let typ = var_types
                .get(name)
                .copied()
                .ok_or_else(|| WgslError::UnknownVariable(name.clone()))?;
            Ok(WgslExpr {
                code: name.clone(),
                typ,
            })
        }

        Ast::BinOp(op, left, right) => {
            let left_expr = emit_wgsl_with(left, var_types, numeric)?;
            let right_expr = emit_wgsl_with(right, var_types, numeric)?;
            emit_binop(*op, left_expr, right_expr, numeric)
        }

        Ast::UnaryOp(op, inner) => {
            let inner_expr = emit_wgsl_with(inner, var_types, numeric)?;
            emit_unaryop(*op, inner_expr)
        }

        Ast::Call(name, args) => {
            let arg_exprs: Vec<WgslExpr> = args
                .iter()
                .map(|a| emit_wgsl_with(a, var_types, numeric))
                .collect::<Result<_, _>>()?;
            emit_function_call(name, arg_exprs, numeric)
        }

        Ast::Compare(op, left, right) => {
            let left_expr = emit_wgsl_with(left, var_types, numeric)?;
            let right_expr = emit_wgsl_with(right, var_types, numeric)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(WgslError::UnsupportedTypeForConditional(left_expr.typ));
            }
            let bool_expr = cond::emit_compare(*op, &left_expr.code, &right_expr.code);
            Ok(WgslExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::And(left, right) => {
            let left_expr = emit_wgsl_with(left, var_types, numeric)?;
            let right_expr = emit_wgsl_with(right, var_types, numeric)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(WgslError::UnsupportedTypeForConditional(left_expr.typ));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_and(&l_bool, &r_bool);
            Ok(WgslExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::Or(left, right) => {
            let left_expr = emit_wgsl_with(left, var_types, numeric)?;
            let right_expr = emit_wgsl_with(right, var_types, numeric)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(WgslError::UnsupportedTypeForConditional(left_expr.typ));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_or(&l_bool, &r_bool);
            Ok(WgslExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::If(cond_ast, then_ast, else_ast) => {
            let cond_expr = emit_wgsl_with(cond_ast, var_types, numeric)?;
            let then_expr = emit_wgsl_with(then_ast, var_types, numeric)?;
            let else_expr = emit_wgsl_with(else_ast, var_types, numeric)?;
            if cond_expr.typ != Type::Scalar {
                return Err(WgslError::UnsupportedTypeForConditional(cond_expr.typ));
            }
            if then_expr.typ != else_expr.typ {
                return Err(WgslError::TypeMismatch {
                    op: "if/else",
                    left: then_expr.typ,
                    right: else_expr.typ,
                });
            }
            let cond_bool = cond::scalar_to_bool(&cond_expr.code);
            Ok(WgslExpr {
                code: cond::emit_if(&cond_bool, &then_expr.code, &else_expr.code),
                typ: then_expr.typ,
            })
        }

        Ast::Let { .. } => {
            // Let in expression context: delegate to emit_full
            let emission = emit_full(ast, var_types, numeric)?;
            if emission.statements.is_empty() {
                Ok(WgslExpr {
                    code: emission.expr,
                    typ: emission.typ,
                })
            } else {
                // Can't inline statements into expression position
                Err(WgslError::UnsupportedFeature(
                    "let in expression position (use emit_wgsl_fn for full support)".to_string(),
                ))
            }
        }
    }
}

/// Emit a complete WGSL function with let statement support.
///
/// This generates a function definition that can contain let statements,
/// unlike `emit_wgsl` which only produces expressions.
///
/// # Example output
/// ```wgsl
/// fn my_func(pixel: vec3<f32>) -> vec3<f32> {
///     let hsl = rgb_to_hsl(pixel);
///     let shifted = vec3<f32>(hsl.x + 0.1, hsl.y * 1.2, hsl.z);
///     return hsl_to_rgb(shifted);
/// }
/// ```
pub fn emit_wgsl_fn(
    name: &str,
    ast: &Ast,
    params: &[(&str, Type)],
    return_type: Type,
) -> Result<String, WgslError> {
    emit_wgsl_fn_with(name, ast, params, return_type, NumericType::F32)
}

/// Emit a complete WGSL function with specified numeric type.
pub fn emit_wgsl_fn_with(
    name: &str,
    ast: &Ast,
    params: &[(&str, Type)],
    return_type: Type,
    numeric: NumericType,
) -> Result<String, WgslError> {
    let var_types: HashMap<String, Type> =
        params.iter().map(|(n, t)| (n.to_string(), *t)).collect();

    let emission = emit_full(ast, &var_types, numeric)?;

    // Build parameter list
    let param_list: Vec<String> = params
        .iter()
        .map(|(n, t)| format!("{}: {}", n, type_to_wgsl_with(*t, numeric)))
        .collect();

    // Build function body
    let mut body = String::new();
    for stmt in emission.statements {
        body.push_str("    ");
        body.push_str(&stmt);
        body.push('\n');
    }
    body.push_str("    return ");
    body.push_str(&emission.expr);
    body.push(';');

    Ok(format!(
        "fn {}({}) -> {} {{\n{}\n}}",
        name,
        param_list.join(", "),
        type_to_wgsl_with(return_type, numeric),
        body
    ))
}

/// Emit with full statement support for let bindings.
fn emit_full(
    ast: &Ast,
    var_types: &HashMap<String, Type>,
    numeric: NumericType,
) -> Result<Emission, WgslError> {
    match ast {
        Ast::Let { name, value, body } => {
            // Emit value expression
            let value_emission = emit_full(value, var_types, numeric)?;

            // Extend var_types with the new binding
            let mut new_var_types = var_types.clone();
            new_var_types.insert(name.clone(), value_emission.typ);

            // Emit body with extended environment
            let mut body_emission = emit_full(body, &new_var_types, numeric)?;

            // Combine: value statements + let statement + body statements
            let mut statements = value_emission.statements;
            let type_str = type_to_wgsl_with(value_emission.typ, numeric);
            statements.push(format!(
                "let {}: {} = {};",
                name, type_str, value_emission.expr
            ));
            statements.append(&mut body_emission.statements);

            Ok(Emission::with_statements(
                statements,
                body_emission.expr,
                body_emission.typ,
            ))
        }

        // All other nodes: delegate to emit_wgsl_with and wrap result
        _ => {
            let result = emit_wgsl_with(ast, var_types, numeric)?;
            Ok(Emission::expr_only(result.code, result.typ))
        }
    }
}

fn emit_binop(
    op: BinOp,
    left: WgslExpr,
    right: WgslExpr,
    numeric: NumericType,
) -> Result<WgslExpr, WgslError> {
    let op_str = match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Pow => return emit_pow(left, right, numeric),
        BinOp::Rem => "%",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
    };

    let result_type = infer_binop_type(op, left.typ, right.typ)?;

    Ok(WgslExpr {
        code: format!("({} {} {})", left.code, op_str, right.code),
        typ: result_type,
    })
}

fn emit_pow(base: WgslExpr, exp: WgslExpr, numeric: NumericType) -> Result<WgslExpr, WgslError> {
    // WGSL pow() only works on scalars
    if base.typ != Type::Scalar || exp.typ != Type::Scalar {
        return Err(WgslError::TypeMismatch {
            op: "^",
            left: base.typ,
            right: exp.typ,
        });
    }
    // For floats, use native pow(). For integers, WGSL doesn't have native
    // integer pow, so we cast through f32 (precision loss for large values).
    let code = match numeric {
        NumericType::F32 => format!("pow({}, {})", base.code, exp.code),
        NumericType::I32 => format!("i32(pow(f32({}), f32({})))", base.code, exp.code),
        NumericType::U32 => format!("u32(pow(f32({}), f32({})))", base.code, exp.code),
    };
    Ok(WgslExpr {
        code,
        typ: Type::Scalar,
    })
}

fn infer_binop_type(op: BinOp, left: Type, right: Type) -> Result<Type, WgslError> {
    match op {
        BinOp::Add | BinOp::Sub => {
            // Same types only
            if left == right {
                Ok(left)
            } else {
                Err(WgslError::TypeMismatch {
                    op: if op == BinOp::Add { "+" } else { "-" },
                    left,
                    right,
                })
            }
        }
        BinOp::Mul => infer_mul_type(left, right),
        BinOp::Div => {
            // vec / scalar or scalar / scalar
            match (left, right) {
                (Type::Scalar, Type::Scalar) => Ok(Type::Scalar),
                (Type::Vec2, Type::Scalar) => Ok(Type::Vec2),
                #[cfg(feature = "3d")]
                (Type::Vec3, Type::Scalar) => Ok(Type::Vec3),
                #[cfg(feature = "4d")]
                (Type::Vec4, Type::Scalar) => Ok(Type::Vec4),
                _ => Err(WgslError::TypeMismatch {
                    op: "/",
                    left,
                    right,
                }),
            }
        }
        BinOp::Pow => {
            if left == Type::Scalar && right == Type::Scalar {
                Ok(Type::Scalar)
            } else {
                Err(WgslError::TypeMismatch {
                    op: "^",
                    left,
                    right,
                })
            }
        }
        // Bitwise and modulo ops: scalar only
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
                Err(WgslError::TypeMismatch {
                    op: op_str,
                    left,
                    right,
                })
            }
        }
    }
}

fn infer_mul_type(left: Type, right: Type) -> Result<Type, WgslError> {
    match (left, right) {
        // Scalar * Scalar
        (Type::Scalar, Type::Scalar) => Ok(Type::Scalar),

        // Vec * Scalar or Scalar * Vec
        (Type::Vec2, Type::Scalar) | (Type::Scalar, Type::Vec2) => Ok(Type::Vec2),
        #[cfg(feature = "3d")]
        (Type::Vec3, Type::Scalar) | (Type::Scalar, Type::Vec3) => Ok(Type::Vec3),
        #[cfg(feature = "4d")]
        (Type::Vec4, Type::Scalar) | (Type::Scalar, Type::Vec4) => Ok(Type::Vec4),

        // Mat * Scalar or Scalar * Mat
        (Type::Mat2, Type::Scalar) | (Type::Scalar, Type::Mat2) => Ok(Type::Mat2),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Scalar) | (Type::Scalar, Type::Mat3) => Ok(Type::Mat3),
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Scalar) | (Type::Scalar, Type::Mat4) => Ok(Type::Mat4),

        // Mat * Vec
        (Type::Mat2, Type::Vec2) => Ok(Type::Vec2),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Vec3) => Ok(Type::Vec3),
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Vec4) => Ok(Type::Vec4),

        // Vec * Mat
        (Type::Vec2, Type::Mat2) => Ok(Type::Vec2),
        #[cfg(feature = "3d")]
        (Type::Vec3, Type::Mat3) => Ok(Type::Vec3),
        #[cfg(feature = "4d")]
        (Type::Vec4, Type::Mat4) => Ok(Type::Vec4),

        // Mat * Mat
        (Type::Mat2, Type::Mat2) => Ok(Type::Mat2),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Mat3) => Ok(Type::Mat3),
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Mat4) => Ok(Type::Mat4),

        _ => Err(WgslError::TypeMismatch {
            op: "*",
            left,
            right,
        }),
    }
}

fn emit_unaryop(op: UnaryOp, inner: WgslExpr) -> Result<WgslExpr, WgslError> {
    match op {
        UnaryOp::Neg => Ok(WgslExpr {
            code: format!("(-{})", inner.code),
            typ: inner.typ,
        }),
        UnaryOp::Not => {
            if inner.typ != Type::Scalar {
                return Err(WgslError::UnsupportedTypeForConditional(inner.typ));
            }
            let bool_expr = cond::scalar_to_bool(&inner.code);
            Ok(WgslExpr {
                code: cond::bool_to_scalar(&cond::emit_not(&bool_expr)),
                typ: Type::Scalar,
            })
        }
        UnaryOp::BitNot => {
            if inner.typ != Type::Scalar {
                return Err(WgslError::UnsupportedType(inner.typ));
            }
            Ok(WgslExpr {
                code: format!("(~{})", inner.code),
                typ: Type::Scalar,
            })
        }
    }
}

fn emit_function_call(
    name: &str,
    args: Vec<WgslExpr>,
    numeric: NumericType,
) -> Result<WgslExpr, WgslError> {
    let scalar = numeric.wgsl_scalar();
    match name {
        // Vector functions that map directly to WGSL builtins
        "dot" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("dot({}, {})", args[0].code, args[1].code),
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "3d")]
        "cross" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("cross({}, {})", args[0].code, args[1].code),
                typ: Type::Vec3,
            })
        }

        "length" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("length({})", args[0].code),
                typ: Type::Scalar,
            })
        }

        "normalize" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("normalize({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "distance" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("distance({}, {})", args[0].code, args[1].code),
                typ: Type::Scalar,
            })
        }

        "reflect" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("reflect({}, {})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "hadamard" => {
            // WGSL doesn't have hadamard, use element-wise multiply
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("({} * {})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "lerp" | "mix" => {
            if args.len() != 3 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("mix({}, {}, {})", args[0].code, args[1].code, args[2].code),
                typ: args[0].typ,
            })
        }

        // ====================================================================
        // Constructors
        // ====================================================================
        "vec2" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("vec2<{scalar}>({}, {})", args[0].code, args[1].code),
                typ: Type::Vec2,
            })
        }

        #[cfg(feature = "3d")]
        "vec3" => {
            if args.len() != 3 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!(
                    "vec3<{scalar}>({}, {}, {})",
                    args[0].code, args[1].code, args[2].code
                ),
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "4d")]
        "vec4" => {
            if args.len() != 4 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!(
                    "vec4<{scalar}>({}, {}, {}, {})",
                    args[0].code, args[1].code, args[2].code, args[3].code
                ),
                typ: Type::Vec4,
            })
        }

        // ====================================================================
        // Component extraction
        // ====================================================================
        "x" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("{}.x", args[0].code),
                typ: Type::Scalar,
            })
        }

        "y" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("{}.y", args[0].code),
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "3d")]
        "z" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("{}.z", args[0].code),
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "4d")]
        "w" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("{}.w", args[0].code),
                typ: Type::Scalar,
            })
        }

        // ====================================================================
        // Vectorized math functions
        // ====================================================================
        "sin" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("sin({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "cos" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("cos({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "abs" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("abs({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "floor" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("floor({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "fract" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("fract({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "sqrt" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("sqrt({})", args[0].code),
                typ: args[0].typ,
            })
        }

        // ====================================================================
        // Vectorized comparison functions
        // ====================================================================
        "min" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("min({}, {})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "max" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("max({}, {})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "clamp" => {
            if args.len() != 3 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!(
                    "clamp({}, {}, {})",
                    args[0].code, args[1].code, args[2].code
                ),
                typ: args[0].typ,
            })
        }

        // ====================================================================
        // Interpolation functions
        // ====================================================================
        "step" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("step({}, {})", args[0].code, args[1].code),
                typ: args[1].typ,
            })
        }

        "smoothstep" => {
            if args.len() != 3 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!(
                    "smoothstep({}, {}, {})",
                    args[0].code, args[1].code, args[2].code
                ),
                typ: args[2].typ,
            })
        }

        // ====================================================================
        // Transform functions
        // ====================================================================
        "rotate2d" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            // rotate2d(v, angle) = vec2(v.x*cos(angle) - v.y*sin(angle), v.x*sin(angle) + v.y*cos(angle))
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(WgslExpr {
                code: format!(
                    "vec2<{scalar}>({v}.x*cos({angle}) - {v}.y*sin({angle}), {v}.x*sin({angle}) + {v}.y*cos({angle}))"
                ),
                typ: Type::Vec2,
            })
        }

        #[cfg(feature = "3d")]
        "rotate_x" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(WgslExpr {
                code: format!(
                    "vec3<{scalar}>({v}.x, {v}.y*cos({angle}) - {v}.z*sin({angle}), {v}.y*sin({angle}) + {v}.z*cos({angle}))"
                ),
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "3d")]
        "rotate_y" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(WgslExpr {
                code: format!(
                    "vec3<{scalar}>({v}.x*cos({angle}) + {v}.z*sin({angle}), {v}.y, -{v}.x*sin({angle}) + {v}.z*cos({angle}))"
                ),
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "3d")]
        "rotate_z" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(WgslExpr {
                code: format!(
                    "vec3<{scalar}>({v}.x*cos({angle}) - {v}.y*sin({angle}), {v}.x*sin({angle}) + {v}.y*cos({angle}), {v}.z)"
                ),
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "3d")]
        "rotate3d" => {
            if args.len() != 3 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            // Rodrigues' rotation formula
            let v = &args[0].code;
            let k = &args[1].code;
            let angle = &args[2].code;
            // v' = v*cos(θ) + cross(k, v)*sin(θ) + k*dot(k, v)*(1-cos(θ))
            Ok(WgslExpr {
                code: format!(
                    "({v}*cos({angle}) + cross({k}, {v})*sin({angle}) + {k}*dot({k}, {v})*(1.0 - cos({angle})))"
                ),
                typ: Type::Vec3,
            })
        }

        // ====================================================================
        // Matrix constructors
        // ====================================================================
        "mat2" => {
            if args.len() != 4 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!(
                    "mat2x2<{scalar}>({}, {}, {}, {})",
                    args[0].code, args[1].code, args[2].code, args[3].code
                ),
                typ: Type::Mat2,
            })
        }

        #[cfg(feature = "3d")]
        "mat3" => {
            if args.len() != 9 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!(
                    "mat3x3<{scalar}>({}, {}, {}, {}, {}, {}, {}, {}, {})",
                    args[0].code,
                    args[1].code,
                    args[2].code,
                    args[3].code,
                    args[4].code,
                    args[5].code,
                    args[6].code,
                    args[7].code,
                    args[8].code
                ),
                typ: Type::Mat3,
            })
        }

        #[cfg(feature = "4d")]
        "mat4" => {
            if args.len() != 16 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!(
                    "mat4x4<{scalar}>({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
                    args[0].code,
                    args[1].code,
                    args[2].code,
                    args[3].code,
                    args[4].code,
                    args[5].code,
                    args[6].code,
                    args[7].code,
                    args[8].code,
                    args[9].code,
                    args[10].code,
                    args[11].code,
                    args[12].code,
                    args[13].code,
                    args[14].code,
                    args[15].code
                ),
                typ: Type::Mat4,
            })
        }

        _ => Err(WgslError::UnknownFunction(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn emit(expr: &str, var_types: &[(&str, Type)]) -> Result<WgslExpr, WgslError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_wgsl(expr.ast(), &types)
    }

    #[test]
    fn test_scalar_add() {
        let result = emit("a + b", &[("a", Type::Scalar), ("b", Type::Scalar)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("+"));
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
    fn test_vec_mat_mul() {
        let result = emit("v * m", &[("v", Type::Vec2), ("m", Type::Mat2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
    }

    #[test]
    fn test_dot() {
        let result = emit("dot(a, b)", &[("a", Type::Vec2), ("b", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("dot"));
    }

    #[test]
    fn test_length() {
        let result = emit("length(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
    }

    #[test]
    fn test_normalize() {
        let result = emit("normalize(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_cross() {
        let result = emit("cross(a, b)", &[("a", Type::Vec3), ("b", Type::Vec3)]).unwrap();
        assert_eq!(result.typ, Type::Vec3);
    }

    #[test]
    fn test_lerp() {
        let result = emit(
            "lerp(a, b, t)",
            &[("a", Type::Vec2), ("b", Type::Vec2), ("t", Type::Scalar)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Vec2);
        assert!(result.code.contains("mix")); // WGSL uses mix
    }

    #[test]
    fn test_type_mismatch() {
        let result = emit("a + b", &[("a", Type::Scalar), ("b", Type::Vec2)]);
        assert!(matches!(result, Err(WgslError::TypeMismatch { .. })));
    }

    // Integer emission tests
    fn emit_int(expr: &str, var_types: &[(&str, Type)]) -> Result<WgslExpr, WgslError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_wgsl_with(expr.ast(), &types, NumericType::I32)
    }

    #[test]
    fn test_i32_literal() {
        let result = emit_int("42", &[]).unwrap();
        assert_eq!(result.code, "42");
        assert_eq!(result.typ, Type::Scalar);
    }

    #[test]
    fn test_i32_vec2_constructor() {
        let result = emit_int("vec2(1, 2)", &[]).unwrap();
        assert!(result.code.contains("vec2<i32>"));
        assert_eq!(result.typ, Type::Vec2);
    }

    #[test]
    fn test_i32_modulo() {
        let result = emit_int("a % b", &[("a", Type::Scalar), ("b", Type::Scalar)]).unwrap();
        assert!(result.code.contains("%"));
        assert_eq!(result.typ, Type::Scalar);
    }

    #[test]
    fn test_i32_bitwise() {
        let and = emit_int("a & b", &[("a", Type::Scalar), ("b", Type::Scalar)]).unwrap();
        assert!(and.code.contains("&"));

        let or = emit_int("a | b", &[("a", Type::Scalar), ("b", Type::Scalar)]).unwrap();
        assert!(or.code.contains("|"));

        let shl = emit_int("a << b", &[("a", Type::Scalar), ("b", Type::Scalar)]).unwrap();
        assert!(shl.code.contains("<<"));

        let shr = emit_int("a >> b", &[("a", Type::Scalar), ("b", Type::Scalar)]).unwrap();
        assert!(shr.code.contains(">>"));
    }

    #[test]
    fn test_i32_pow() {
        let result = emit_int("a ^ b", &[("a", Type::Scalar), ("b", Type::Scalar)]).unwrap();
        // Integer pow casts through f32
        assert!(result.code.contains("i32(pow(f32("));
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_let_in_fn() {
        let expr = Expr::parse("let hsl = v; vec3(x(hsl) + 0.1, y(hsl) * 1.2, z(hsl))").unwrap();
        let code = emit_wgsl_fn("shift_hue", expr.ast(), &[("v", Type::Vec3)], Type::Vec3).unwrap();
        assert!(code.contains("let hsl: vec3<f32> = v;"));
        assert!(code.contains("return vec3<f32>"));
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_nested_let() {
        let expr = Expr::parse("let a = v; let b = a * 2; b + a").unwrap();
        let code = emit_wgsl_fn("nested", expr.ast(), &[("v", Type::Vec3)], Type::Vec3).unwrap();
        assert!(code.contains("let a: vec3<f32> = v;"));
        assert!(code.contains("let b: vec3<f32> = (a * 2"));
        assert!(code.contains("return (b + a)"));
    }
}
