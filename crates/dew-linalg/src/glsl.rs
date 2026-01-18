//! GLSL code generation for linalg expressions.
//!
//! Emits GLSL code with proper type handling for vectors and matrices.

use crate::Type;
use rhizome_dew_cond::glsl as cond;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

/// Error during GLSL code generation.
#[derive(Debug, Clone, PartialEq)]
pub enum GlslError {
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

impl std::fmt::Display for GlslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlslError::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            GlslError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            GlslError::TypeMismatch { op, left, right } => {
                write!(f, "type mismatch for {op}: {left} vs {right}")
            }
            GlslError::UnsupportedType(t) => write!(f, "unsupported type: {t}"),
            GlslError::UnsupportedTypeForConditional(t) => {
                write!(f, "conditionals require scalar type, got {t}")
            }
            GlslError::UnsupportedFeature(feat) => {
                write!(f, "unsupported feature in expression codegen: {feat}")
            }
        }
    }
}

impl std::error::Error for GlslError {}

/// Numeric type for code generation (scalar element type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumericType {
    #[default]
    Float,
    Int,
    Uint,
}

impl NumericType {
    /// GLSL scalar type name.
    pub fn glsl_scalar(&self) -> &'static str {
        match self {
            NumericType::Float => "float",
            NumericType::Int => "int",
            NumericType::Uint => "uint",
        }
    }
}

/// Convert a Type to its GLSL representation with the given numeric type.
pub fn type_to_glsl_with(t: Type, numeric: NumericType) -> &'static str {
    match (t, numeric) {
        (Type::Scalar, NumericType::Float) => "float",
        (Type::Scalar, NumericType::Int) => "int",
        (Type::Scalar, NumericType::Uint) => "uint",
        (Type::Vec2, NumericType::Float) => "vec2",
        (Type::Vec2, NumericType::Int) => "ivec2",
        (Type::Vec2, NumericType::Uint) => "uvec2",
        #[cfg(feature = "3d")]
        (Type::Vec3, NumericType::Float) => "vec3",
        #[cfg(feature = "3d")]
        (Type::Vec3, NumericType::Int) => "ivec3",
        #[cfg(feature = "3d")]
        (Type::Vec3, NumericType::Uint) => "uvec3",
        #[cfg(feature = "4d")]
        (Type::Vec4, NumericType::Float) => "vec4",
        #[cfg(feature = "4d")]
        (Type::Vec4, NumericType::Int) => "ivec4",
        #[cfg(feature = "4d")]
        (Type::Vec4, NumericType::Uint) => "uvec4",
        (Type::Mat2, _) => "mat2", // Matrices are always float in GLSL
        #[cfg(feature = "3d")]
        (Type::Mat3, _) => "mat3",
        #[cfg(feature = "4d")]
        (Type::Mat4, _) => "mat4",
    }
}

/// Convert a Type to its GLSL representation (defaults to float).
pub fn type_to_glsl(t: Type) -> &'static str {
    type_to_glsl_with(t, NumericType::Float)
}

/// Result of GLSL emission: code string and its type.
pub struct GlslExpr {
    pub code: String,
    pub typ: Type,
}

/// Format a numeric literal for GLSL.
fn format_literal(n: f64, numeric: NumericType) -> String {
    match numeric {
        NumericType::Float => format!("{n:.10}"),
        NumericType::Int => format!("{}", n as i32),
        NumericType::Uint => format!("{}u", n as u32),
    }
}

/// Emit GLSL code for an AST with type propagation (defaults to float).
pub fn emit_glsl(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<GlslExpr, GlslError> {
    emit_glsl_with(ast, var_types, NumericType::Float)
}

/// Emit GLSL code for an AST with specified numeric type.
pub fn emit_glsl_with(
    ast: &Ast,
    var_types: &HashMap<String, Type>,
    numeric: NumericType,
) -> Result<GlslExpr, GlslError> {
    match ast {
        Ast::Num(n) => Ok(GlslExpr {
            code: format_literal(*n, numeric),
            typ: Type::Scalar,
        }),

        Ast::Var(name) => {
            let typ = var_types
                .get(name)
                .copied()
                .ok_or_else(|| GlslError::UnknownVariable(name.clone()))?;
            Ok(GlslExpr {
                code: name.clone(),
                typ,
            })
        }

        Ast::BinOp(op, left, right) => {
            let left_expr = emit_glsl_with(left, var_types, numeric)?;
            let right_expr = emit_glsl_with(right, var_types, numeric)?;
            emit_binop(*op, left_expr, right_expr, numeric)
        }

        Ast::UnaryOp(op, inner) => {
            let inner_expr = emit_glsl_with(inner, var_types, numeric)?;
            emit_unaryop(*op, inner_expr)
        }

        Ast::Call(name, args) => {
            let arg_exprs: Vec<GlslExpr> = args
                .iter()
                .map(|a| emit_glsl_with(a, var_types, numeric))
                .collect::<Result<_, _>>()?;
            emit_function_call(name, arg_exprs, numeric)
        }

        Ast::Compare(op, left, right) => {
            let left_expr = emit_glsl_with(left, var_types, numeric)?;
            let right_expr = emit_glsl_with(right, var_types, numeric)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(GlslError::UnsupportedTypeForConditional(left_expr.typ));
            }
            let bool_expr = cond::emit_compare(*op, &left_expr.code, &right_expr.code);
            Ok(GlslExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::And(left, right) => {
            let left_expr = emit_glsl_with(left, var_types, numeric)?;
            let right_expr = emit_glsl_with(right, var_types, numeric)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(GlslError::UnsupportedTypeForConditional(left_expr.typ));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_and(&l_bool, &r_bool);
            Ok(GlslExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::Or(left, right) => {
            let left_expr = emit_glsl_with(left, var_types, numeric)?;
            let right_expr = emit_glsl_with(right, var_types, numeric)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(GlslError::UnsupportedTypeForConditional(left_expr.typ));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_or(&l_bool, &r_bool);
            Ok(GlslExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::If(cond_ast, then_ast, else_ast) => {
            let cond_expr = emit_glsl_with(cond_ast, var_types, numeric)?;
            let then_expr = emit_glsl_with(then_ast, var_types, numeric)?;
            let else_expr = emit_glsl_with(else_ast, var_types, numeric)?;
            if cond_expr.typ != Type::Scalar {
                return Err(GlslError::UnsupportedTypeForConditional(cond_expr.typ));
            }
            if then_expr.typ != else_expr.typ {
                return Err(GlslError::TypeMismatch {
                    op: "if/else",
                    left: then_expr.typ,
                    right: else_expr.typ,
                });
            }
            let cond_bool = cond::scalar_to_bool(&cond_expr.code);
            Ok(GlslExpr {
                code: cond::emit_if(&cond_bool, &then_expr.code, &else_expr.code),
                typ: then_expr.typ,
            })
        }

        Ast::Let { .. } => {
            // Let bindings require statement-based codegen.
            // Use the LetInlining optimization pass before emitting.
            Err(GlslError::UnsupportedFeature(
                "let bindings (use LetInlining pass first)".to_string(),
            ))
        }
    }
}

fn emit_binop(
    op: BinOp,
    left: GlslExpr,
    right: GlslExpr,
    numeric: NumericType,
) -> Result<GlslExpr, GlslError> {
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

    Ok(GlslExpr {
        code: format!("({} {} {})", left.code, op_str, right.code),
        typ: result_type,
    })
}

fn emit_pow(base: GlslExpr, exp: GlslExpr, numeric: NumericType) -> Result<GlslExpr, GlslError> {
    // GLSL pow() only works on scalars
    if base.typ != Type::Scalar || exp.typ != Type::Scalar {
        return Err(GlslError::TypeMismatch {
            op: "^",
            left: base.typ,
            right: exp.typ,
        });
    }
    // For floats, use native pow(). For integers, GLSL doesn't have native
    // integer pow, so we cast through float (precision loss for large values).
    let code = match numeric {
        NumericType::Float => format!("pow({}, {})", base.code, exp.code),
        NumericType::Int => format!("int(pow(float({}), float({})))", base.code, exp.code),
        NumericType::Uint => format!("uint(pow(float({}), float({})))", base.code, exp.code),
    };
    Ok(GlslExpr {
        code,
        typ: Type::Scalar,
    })
}

fn infer_binop_type(op: BinOp, left: Type, right: Type) -> Result<Type, GlslError> {
    match op {
        BinOp::Add | BinOp::Sub => {
            // Same types only
            if left == right {
                Ok(left)
            } else {
                Err(GlslError::TypeMismatch {
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
                _ => Err(GlslError::TypeMismatch {
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
                Err(GlslError::TypeMismatch {
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
                Err(GlslError::TypeMismatch {
                    op: op_str,
                    left,
                    right,
                })
            }
        }
    }
}

fn infer_mul_type(left: Type, right: Type) -> Result<Type, GlslError> {
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

        _ => Err(GlslError::TypeMismatch {
            op: "*",
            left,
            right,
        }),
    }
}

fn emit_unaryop(op: UnaryOp, inner: GlslExpr) -> Result<GlslExpr, GlslError> {
    match op {
        UnaryOp::Neg => Ok(GlslExpr {
            code: format!("(-{})", inner.code),
            typ: inner.typ,
        }),
        UnaryOp::Not => {
            if inner.typ != Type::Scalar {
                return Err(GlslError::UnsupportedTypeForConditional(inner.typ));
            }
            let bool_expr = cond::scalar_to_bool(&inner.code);
            Ok(GlslExpr {
                code: cond::bool_to_scalar(&cond::emit_not(&bool_expr)),
                typ: Type::Scalar,
            })
        }
        UnaryOp::BitNot => {
            if inner.typ != Type::Scalar {
                return Err(GlslError::UnsupportedType(inner.typ));
            }
            Ok(GlslExpr {
                code: format!("(~{})", inner.code),
                typ: Type::Scalar,
            })
        }
    }
}

fn emit_function_call(
    name: &str,
    args: Vec<GlslExpr>,
    numeric: NumericType,
) -> Result<GlslExpr, GlslError> {
    let vec2_type = type_to_glsl_with(Type::Vec2, numeric);
    #[cfg(feature = "3d")]
    let vec3_type = type_to_glsl_with(Type::Vec3, numeric);
    #[cfg(feature = "4d")]
    let vec4_type = type_to_glsl_with(Type::Vec4, numeric);

    match name {
        // Vector functions that map directly to GLSL builtins
        "dot" => {
            if args.len() != 2 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("dot({}, {})", args[0].code, args[1].code),
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "3d")]
        "cross" => {
            if args.len() != 2 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("cross({}, {})", args[0].code, args[1].code),
                typ: Type::Vec3,
            })
        }

        "length" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("length({})", args[0].code),
                typ: Type::Scalar,
            })
        }

        "normalize" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("normalize({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "distance" => {
            if args.len() != 2 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("distance({}, {})", args[0].code, args[1].code),
                typ: Type::Scalar,
            })
        }

        "reflect" => {
            if args.len() != 2 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("reflect({}, {})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "hadamard" => {
            // GLSL doesn't have hadamard, use element-wise multiply
            if args.len() != 2 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("({} * {})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "lerp" | "mix" => {
            if args.len() != 3 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("mix({}, {}, {})", args[0].code, args[1].code, args[2].code),
                typ: args[0].typ,
            })
        }

        // ====================================================================
        // Constructors
        // ====================================================================
        "vec2" => {
            if args.len() != 2 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("{vec2_type}({}, {})", args[0].code, args[1].code),
                typ: Type::Vec2,
            })
        }

        #[cfg(feature = "3d")]
        "vec3" => {
            if args.len() != 3 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!(
                    "{vec3_type}({}, {}, {})",
                    args[0].code, args[1].code, args[2].code
                ),
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "4d")]
        "vec4" => {
            if args.len() != 4 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!(
                    "{vec4_type}({}, {}, {}, {})",
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
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("{}.x", args[0].code),
                typ: Type::Scalar,
            })
        }

        "y" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("{}.y", args[0].code),
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "3d")]
        "z" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("{}.z", args[0].code),
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "4d")]
        "w" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("{}.w", args[0].code),
                typ: Type::Scalar,
            })
        }

        // ====================================================================
        // Vectorized math functions
        // ====================================================================
        "sin" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("sin({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "cos" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("cos({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "abs" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("abs({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "floor" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("floor({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "fract" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("fract({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "sqrt" => {
            if args.len() != 1 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("sqrt({})", args[0].code),
                typ: args[0].typ,
            })
        }

        // ====================================================================
        // Vectorized comparison functions
        // ====================================================================
        "min" => {
            if args.len() != 2 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("min({}, {})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "max" => {
            if args.len() != 2 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("max({}, {})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "clamp" => {
            if args.len() != 3 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
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
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!("step({}, {})", args[0].code, args[1].code),
                typ: args[1].typ,
            })
        }

        "smoothstep" => {
            if args.len() != 3 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
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
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            // rotate2d(v, angle) = vec2(v.x*cos(angle) - v.y*sin(angle), v.x*sin(angle) + v.y*cos(angle))
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(GlslExpr {
                code: format!(
                    "{vec2_type}({v}.x*cos({angle}) - {v}.y*sin({angle}), {v}.x*sin({angle}) + {v}.y*cos({angle}))"
                ),
                typ: Type::Vec2,
            })
        }

        #[cfg(feature = "3d")]
        "rotate_x" => {
            if args.len() != 2 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(GlslExpr {
                code: format!(
                    "{vec3_type}({v}.x, {v}.y*cos({angle}) - {v}.z*sin({angle}), {v}.y*sin({angle}) + {v}.z*cos({angle}))"
                ),
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "3d")]
        "rotate_y" => {
            if args.len() != 2 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(GlslExpr {
                code: format!(
                    "{vec3_type}({v}.x*cos({angle}) + {v}.z*sin({angle}), {v}.y, -{v}.x*sin({angle}) + {v}.z*cos({angle}))"
                ),
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "3d")]
        "rotate_z" => {
            if args.len() != 2 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(GlslExpr {
                code: format!(
                    "{vec3_type}({v}.x*cos({angle}) - {v}.y*sin({angle}), {v}.x*sin({angle}) + {v}.y*cos({angle}), {v}.z)"
                ),
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "3d")]
        "rotate3d" => {
            if args.len() != 3 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            // Rodrigues' rotation formula
            let v = &args[0].code;
            let k = &args[1].code;
            let angle = &args[2].code;
            // v' = v*cos(θ) + cross(k, v)*sin(θ) + k*dot(k, v)*(1-cos(θ))
            Ok(GlslExpr {
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
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!(
                    "mat2({}, {}, {}, {})",
                    args[0].code, args[1].code, args[2].code, args[3].code
                ),
                typ: Type::Mat2,
            })
        }

        #[cfg(feature = "3d")]
        "mat3" => {
            if args.len() != 9 {
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!(
                    "mat3({}, {}, {}, {}, {}, {}, {}, {}, {})",
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
                return Err(GlslError::UnknownFunction(name.to_string()));
            }
            Ok(GlslExpr {
                code: format!(
                    "mat4({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
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

        _ => Err(GlslError::UnknownFunction(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn emit(expr: &str, var_types: &[(&str, Type)]) -> Result<GlslExpr, GlslError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_glsl(expr.ast(), &types)
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
        assert!(result.code.contains("mix")); // GLSL uses mix
    }

    #[test]
    fn test_vec2_constructor() {
        let result = emit("vec2(x, y)", &[("x", Type::Scalar), ("y", Type::Scalar)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
        assert!(result.code.contains("vec2("));
        // GLSL doesn't have <f32>
        assert!(!result.code.contains("<f32>"));
    }

    #[test]
    fn test_type_mismatch() {
        let result = emit("a + b", &[("a", Type::Scalar), ("b", Type::Vec2)]);
        assert!(matches!(result, Err(GlslError::TypeMismatch { .. })));
    }

    // Integer emission tests
    fn emit_int(expr: &str, var_types: &[(&str, Type)]) -> Result<GlslExpr, GlslError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_glsl_with(expr.ast(), &types, NumericType::Int)
    }

    #[test]
    fn test_int_literal() {
        let result = emit_int("42", &[]).unwrap();
        assert_eq!(result.code, "42");
        assert_eq!(result.typ, Type::Scalar);
    }

    #[test]
    fn test_ivec2_constructor() {
        let result = emit_int("vec2(1, 2)", &[]).unwrap();
        assert!(result.code.contains("ivec2("));
        assert_eq!(result.typ, Type::Vec2);
    }

    #[test]
    fn test_int_modulo() {
        let result = emit_int("a % b", &[("a", Type::Scalar), ("b", Type::Scalar)]).unwrap();
        assert!(result.code.contains("%"));
        assert_eq!(result.typ, Type::Scalar);
    }

    #[test]
    fn test_int_bitwise() {
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
    fn test_int_pow() {
        let result = emit_int("a ^ b", &[("a", Type::Scalar), ("b", Type::Scalar)]).unwrap();
        // Integer pow casts through float
        assert!(result.code.contains("int(pow(float("));
    }
}
