//! Rust code generation for linalg expressions.
//!
//! Emits Rust code with glam-compatible types (Vec2, Vec3, Vec4, Mat2, Mat3, Mat4).

use crate::Type;
use rhizome_dew_cond::rust as cond;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

/// Error during Rust code generation.
#[derive(Debug, Clone, PartialEq)]
pub enum RustError {
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

impl std::fmt::Display for RustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RustError::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            RustError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            RustError::TypeMismatch { op, left, right } => {
                write!(f, "type mismatch for {op}: {left} vs {right}")
            }
            RustError::UnsupportedType(t) => write!(f, "unsupported type: {t}"),
            RustError::UnsupportedTypeForConditional(t) => {
                write!(f, "conditionals require scalar type, got {t}")
            }
            RustError::UnsupportedFeature(feat) => {
                write!(f, "unsupported feature in expression codegen: {feat}")
            }
        }
    }
}

impl std::error::Error for RustError {}

/// Convert a Type to its Rust/glam representation.
pub fn type_to_rust(t: Type) -> &'static str {
    match t {
        Type::Scalar => "f32",
        Type::Vec2 => "Vec2",
        #[cfg(feature = "3d")]
        Type::Vec3 => "Vec3",
        #[cfg(feature = "4d")]
        Type::Vec4 => "Vec4",
        Type::Mat2 => "Mat2",
        #[cfg(feature = "3d")]
        Type::Mat3 => "Mat3",
        #[cfg(feature = "4d")]
        Type::Mat4 => "Mat4",
    }
}

/// Result of Rust emission: code string and its type.
pub struct RustExpr {
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

/// Format a numeric literal for Rust.
fn format_literal(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{:.1}", n)
    } else {
        format!("{}", n)
    }
}

/// Emit Rust code for an AST with type propagation.
pub fn emit_rust(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<RustExpr, RustError> {
    match ast {
        Ast::Num(n) => Ok(RustExpr {
            code: format_literal(*n),
            typ: Type::Scalar,
        }),

        Ast::Var(name) => {
            let typ = var_types
                .get(name)
                .copied()
                .ok_or_else(|| RustError::UnknownVariable(name.clone()))?;
            Ok(RustExpr {
                code: name.clone(),
                typ,
            })
        }

        Ast::BinOp(op, left, right) => {
            let left_expr = emit_rust(left, var_types)?;
            let right_expr = emit_rust(right, var_types)?;
            emit_binop(*op, left_expr, right_expr)
        }

        Ast::UnaryOp(op, inner) => {
            let inner_expr = emit_rust(inner, var_types)?;
            emit_unaryop(*op, inner_expr)
        }

        Ast::Call(name, args) => {
            let arg_exprs: Vec<RustExpr> = args
                .iter()
                .map(|a| emit_rust(a, var_types))
                .collect::<Result<_, _>>()?;
            emit_function_call(name, arg_exprs)
        }

        Ast::Compare(op, left, right) => {
            let left_expr = emit_rust(left, var_types)?;
            let right_expr = emit_rust(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(RustError::UnsupportedTypeForConditional(left_expr.typ));
            }
            let bool_expr = cond::emit_compare(*op, &left_expr.code, &right_expr.code);
            Ok(RustExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::And(left, right) => {
            let left_expr = emit_rust(left, var_types)?;
            let right_expr = emit_rust(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(RustError::UnsupportedTypeForConditional(left_expr.typ));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_and(&l_bool, &r_bool);
            Ok(RustExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::Or(left, right) => {
            let left_expr = emit_rust(left, var_types)?;
            let right_expr = emit_rust(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(RustError::UnsupportedTypeForConditional(left_expr.typ));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_or(&l_bool, &r_bool);
            Ok(RustExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::If(cond_ast, then_ast, else_ast) => {
            let cond_expr = emit_rust(cond_ast, var_types)?;
            let then_expr = emit_rust(then_ast, var_types)?;
            let else_expr = emit_rust(else_ast, var_types)?;
            if cond_expr.typ != Type::Scalar {
                return Err(RustError::UnsupportedTypeForConditional(cond_expr.typ));
            }
            if then_expr.typ != else_expr.typ {
                return Err(RustError::TypeMismatch {
                    op: "if/else",
                    left: then_expr.typ,
                    right: else_expr.typ,
                });
            }
            let cond_bool = cond::scalar_to_bool(&cond_expr.code);
            Ok(RustExpr {
                code: cond::emit_if(&cond_bool, &then_expr.code, &else_expr.code),
                typ: then_expr.typ,
            })
        }

        Ast::Let { .. } => {
            let emission = emit_full(ast, var_types)?;
            if emission.statements.is_empty() {
                Ok(RustExpr {
                    code: emission.expr,
                    typ: emission.typ,
                })
            } else {
                Err(RustError::UnsupportedFeature(
                    "let in expression position (use emit_rust_fn for full support)".to_string(),
                ))
            }
        }
    }
}

/// Emit a complete Rust function with let statement support.
pub fn emit_rust_fn(
    name: &str,
    ast: &Ast,
    params: &[(&str, Type)],
    return_type: Type,
) -> Result<String, RustError> {
    let var_types: HashMap<String, Type> =
        params.iter().map(|(n, t)| (n.to_string(), *t)).collect();

    let emission = emit_full(ast, &var_types)?;

    let param_list: Vec<String> = params
        .iter()
        .map(|(n, t)| format!("{}: {}", n, type_to_rust(*t)))
        .collect();

    let mut body = String::new();
    for stmt in emission.statements {
        body.push_str("    ");
        body.push_str(&stmt);
        body.push('\n');
    }
    body.push_str("    ");
    body.push_str(&emission.expr);

    Ok(format!(
        "fn {}({}) -> {} {{\n{}\n}}",
        name,
        param_list.join(", "),
        type_to_rust(return_type),
        body
    ))
}

/// Emit with full statement support for let bindings.
fn emit_full(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<Emission, RustError> {
    match ast {
        Ast::Let { name, value, body } => {
            let value_emission = emit_full(value, var_types)?;

            let mut new_var_types = var_types.clone();
            new_var_types.insert(name.clone(), value_emission.typ);

            let mut body_emission = emit_full(body, &new_var_types)?;

            let mut statements = value_emission.statements;
            let type_str = type_to_rust(value_emission.typ);
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

        _ => {
            let result = emit_rust(ast, var_types)?;
            Ok(Emission::expr_only(result.code, result.typ))
        }
    }
}

fn emit_binop(op: BinOp, left: RustExpr, right: RustExpr) -> Result<RustExpr, RustError> {
    let op_str = match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Pow => return emit_pow(left, right),
        BinOp::Rem => "%",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
    };

    let result_type = infer_binop_type(op, left.typ, right.typ)?;

    Ok(RustExpr {
        code: format!("({} {} {})", left.code, op_str, right.code),
        typ: result_type,
    })
}

fn emit_pow(base: RustExpr, exp: RustExpr) -> Result<RustExpr, RustError> {
    if base.typ != Type::Scalar || exp.typ != Type::Scalar {
        return Err(RustError::TypeMismatch {
            op: "^",
            left: base.typ,
            right: exp.typ,
        });
    }
    Ok(RustExpr {
        code: format!("{}.powf({})", base.code, exp.code),
        typ: Type::Scalar,
    })
}

fn infer_binop_type(op: BinOp, left: Type, right: Type) -> Result<Type, RustError> {
    match op {
        BinOp::Add | BinOp::Sub => {
            if left == right {
                Ok(left)
            } else {
                Err(RustError::TypeMismatch {
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
            _ => Err(RustError::TypeMismatch {
                op: "/",
                left,
                right,
            }),
        },
        BinOp::Pow => {
            if left == Type::Scalar && right == Type::Scalar {
                Ok(Type::Scalar)
            } else {
                Err(RustError::TypeMismatch {
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
                Err(RustError::TypeMismatch {
                    op: op_str,
                    left,
                    right,
                })
            }
        }
    }
}

fn infer_mul_type(left: Type, right: Type) -> Result<Type, RustError> {
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

        _ => Err(RustError::TypeMismatch {
            op: "*",
            left,
            right,
        }),
    }
}

fn emit_unaryop(op: UnaryOp, inner: RustExpr) -> Result<RustExpr, RustError> {
    match op {
        UnaryOp::Neg => Ok(RustExpr {
            code: format!("(-{})", inner.code),
            typ: inner.typ,
        }),
        UnaryOp::Not => {
            if inner.typ != Type::Scalar {
                return Err(RustError::UnsupportedTypeForConditional(inner.typ));
            }
            let bool_expr = cond::scalar_to_bool(&inner.code);
            Ok(RustExpr {
                code: cond::bool_to_scalar(&cond::emit_not(&bool_expr)),
                typ: Type::Scalar,
            })
        }
        UnaryOp::BitNot => {
            if inner.typ != Type::Scalar {
                return Err(RustError::UnsupportedType(inner.typ));
            }
            Ok(RustExpr {
                code: format!("(!{})", inner.code),
                typ: Type::Scalar,
            })
        }
    }
}

fn emit_function_call(name: &str, args: Vec<RustExpr>) -> Result<RustExpr, RustError> {
    match name {
        // ====================================================================
        // Vector functions
        // ====================================================================
        "dot" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.dot({})", args[0].code, args[1].code),
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "3d")]
        "cross" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.cross({})", args[0].code, args[1].code),
                typ: Type::Vec3,
            })
        }

        "length" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.length()", args[0].code),
                typ: Type::Scalar,
            })
        }

        "normalize" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.normalize()", args[0].code),
                typ: args[0].typ,
            })
        }

        "distance" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.distance({})", args[0].code, args[1].code),
                typ: Type::Scalar,
            })
        }

        "reflect" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            // glam doesn't have reflect, implement manually: v - 2.0 * dot(v, n) * n
            let v = &args[0].code;
            let n = &args[1].code;
            Ok(RustExpr {
                code: format!("({v} - 2.0 * {v}.dot({n}) * {n})"),
                typ: args[0].typ,
            })
        }

        "hadamard" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("({} * {})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "lerp" | "mix" => {
            if args.len() != 3 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.lerp({}, {})", args[0].code, args[1].code, args[2].code),
                typ: args[0].typ,
            })
        }

        // ====================================================================
        // Constructors
        // ====================================================================
        "vec2" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("Vec2::new({}, {})", args[0].code, args[1].code),
                typ: Type::Vec2,
            })
        }

        #[cfg(feature = "3d")]
        "vec3" => {
            if args.len() != 3 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!(
                    "Vec3::new({}, {}, {})",
                    args[0].code, args[1].code, args[2].code
                ),
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "4d")]
        "vec4" => {
            if args.len() != 4 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!(
                    "Vec4::new({}, {}, {}, {})",
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
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.x", args[0].code),
                typ: Type::Scalar,
            })
        }

        "y" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.y", args[0].code),
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "3d")]
        "z" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.z", args[0].code),
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "4d")]
        "w" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.w", args[0].code),
                typ: Type::Scalar,
            })
        }

        // ====================================================================
        // Vectorized math functions (method syntax)
        // ====================================================================
        "sin" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            if args[0].typ == Type::Scalar {
                Ok(RustExpr {
                    code: format!("{}.sin()", args[0].code),
                    typ: Type::Scalar,
                })
            } else {
                // glam vectors don't have sin method, need to use map or per-component
                Err(RustError::UnsupportedType(args[0].typ))
            }
        }

        "cos" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            if args[0].typ == Type::Scalar {
                Ok(RustExpr {
                    code: format!("{}.cos()", args[0].code),
                    typ: Type::Scalar,
                })
            } else {
                Err(RustError::UnsupportedType(args[0].typ))
            }
        }

        "abs" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.abs()", args[0].code),
                typ: args[0].typ,
            })
        }

        "floor" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.floor()", args[0].code),
                typ: args[0].typ,
            })
        }

        "fract" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.fract()", args[0].code),
                typ: args[0].typ,
            })
        }

        "sqrt" => {
            if args.len() != 1 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            if args[0].typ == Type::Scalar {
                Ok(RustExpr {
                    code: format!("{}.sqrt()", args[0].code),
                    typ: Type::Scalar,
                })
            } else {
                Err(RustError::UnsupportedType(args[0].typ))
            }
        }

        // ====================================================================
        // Vectorized comparison functions
        // ====================================================================
        "min" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.min({})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "max" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.max({})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "clamp" => {
            if args.len() != 3 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!("{}.clamp({}, {})", args[0].code, args[1].code, args[2].code),
                typ: args[0].typ,
            })
        }

        // ====================================================================
        // Interpolation functions
        // ====================================================================
        "step" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            let edge = &args[0].code;
            let x = &args[1].code;
            Ok(RustExpr {
                code: format!("(if {x} < {edge} {{ 0.0 }} else {{ 1.0 }})"),
                typ: Type::Scalar,
            })
        }

        "smoothstep" => {
            if args.len() != 3 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            let e0 = &args[0].code;
            let e1 = &args[1].code;
            let x = &args[2].code;
            Ok(RustExpr {
                code: format!(
                    "({{ let t = (({x} - {e0}) / ({e1} - {e0})).clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) }})"
                ),
                typ: Type::Scalar,
            })
        }

        // ====================================================================
        // Transform functions
        // ====================================================================
        "rotate2d" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(RustExpr {
                code: format!(
                    "Vec2::new({v}.x * {angle}.cos() - {v}.y * {angle}.sin(), {v}.x * {angle}.sin() + {v}.y * {angle}.cos())"
                ),
                typ: Type::Vec2,
            })
        }

        #[cfg(feature = "3d")]
        "rotate_x" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(RustExpr {
                code: format!(
                    "Vec3::new({v}.x, {v}.y * {angle}.cos() - {v}.z * {angle}.sin(), {v}.y * {angle}.sin() + {v}.z * {angle}.cos())"
                ),
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "3d")]
        "rotate_y" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(RustExpr {
                code: format!(
                    "Vec3::new({v}.x * {angle}.cos() + {v}.z * {angle}.sin(), {v}.y, -{v}.x * {angle}.sin() + {v}.z * {angle}.cos())"
                ),
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "3d")]
        "rotate_z" => {
            if args.len() != 2 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let angle = &args[1].code;
            Ok(RustExpr {
                code: format!(
                    "Vec3::new({v}.x * {angle}.cos() - {v}.y * {angle}.sin(), {v}.x * {angle}.sin() + {v}.y * {angle}.cos(), {v}.z)"
                ),
                typ: Type::Vec3,
            })
        }

        #[cfg(feature = "3d")]
        "rotate3d" => {
            if args.len() != 3 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            // Rodrigues' rotation formula
            let v = &args[0].code;
            let k = &args[1].code;
            let angle = &args[2].code;
            Ok(RustExpr {
                code: format!(
                    "({v} * {angle}.cos() + {k}.cross({v}) * {angle}.sin() + {k} * {k}.dot({v}) * (1.0 - {angle}.cos()))"
                ),
                typ: Type::Vec3,
            })
        }

        // ====================================================================
        // Matrix constructors
        // ====================================================================
        "mat2" => {
            if args.len() != 4 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!(
                    "Mat2::from_cols(Vec2::new({}, {}), Vec2::new({}, {}))",
                    args[0].code, args[1].code, args[2].code, args[3].code
                ),
                typ: Type::Mat2,
            })
        }

        #[cfg(feature = "3d")]
        "mat3" => {
            if args.len() != 9 {
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!(
                    "Mat3::from_cols(Vec3::new({}, {}, {}), Vec3::new({}, {}, {}), Vec3::new({}, {}, {}))",
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
                return Err(RustError::UnknownFunction(name.to_string()));
            }
            Ok(RustExpr {
                code: format!(
                    "Mat4::from_cols(Vec4::new({}, {}, {}, {}), Vec4::new({}, {}, {}, {}), Vec4::new({}, {}, {}, {}), Vec4::new({}, {}, {}, {}))",
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

        _ => Err(RustError::UnknownFunction(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn emit(expr: &str, var_types: &[(&str, Type)]) -> Result<RustExpr, RustError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_rust(expr.ast(), &types)
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
    fn test_dot() {
        let result = emit("dot(a, b)", &[("a", Type::Vec2), ("b", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains(".dot("));
    }

    #[test]
    fn test_length() {
        let result = emit("length(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains(".length()"));
    }

    #[test]
    fn test_normalize() {
        let result = emit("normalize(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
        assert!(result.code.contains(".normalize()"));
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_cross() {
        let result = emit("cross(a, b)", &[("a", Type::Vec3), ("b", Type::Vec3)]).unwrap();
        assert_eq!(result.typ, Type::Vec3);
        assert!(result.code.contains(".cross("));
    }

    #[test]
    fn test_lerp() {
        let result = emit(
            "lerp(a, b, t)",
            &[("a", Type::Vec2), ("b", Type::Vec2), ("t", Type::Scalar)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Vec2);
        assert!(result.code.contains(".lerp("));
    }

    #[test]
    fn test_vec2_constructor() {
        let result = emit("vec2(x, y)", &[("x", Type::Scalar), ("y", Type::Scalar)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
        assert!(result.code.contains("Vec2::new("));
    }

    #[test]
    fn test_type_mismatch() {
        let result = emit("a + b", &[("a", Type::Scalar), ("b", Type::Vec2)]);
        assert!(matches!(result, Err(RustError::TypeMismatch { .. })));
    }

    #[test]
    fn test_component_extraction() {
        let result = emit("x(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains(".x"));
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_let_in_fn() {
        let expr = Expr::parse("let hsl = v; vec3(x(hsl) + 0.1, y(hsl) * 1.2, z(hsl))").unwrap();
        let code = emit_rust_fn("shift_hue", expr.ast(), &[("v", Type::Vec3)], Type::Vec3).unwrap();
        assert!(code.contains("let hsl: Vec3 = v;"));
        assert!(code.contains("Vec3::new("));
    }
}
