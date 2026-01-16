//! Cranelift JIT compilation for linalg expressions.
//!
//! Compiles typed expressions to native code via Cranelift.
//!
//! # Vector Representation
//!
//! Vectors are passed as consecutive f32 parameters:
//! - Vec2: 2 f32 values
//! - Vec3: 3 f32 values
//! - Vec4: 4 f32 values
//!
//! This module currently supports expressions that evaluate to scalars
//! (using dot, length, distance, etc. on vectors).

/// Dispatch a JIT function call based on parameter count.
/// Centralizes the unsafe transmute logic for all arities 0-16.
macro_rules! jit_call {
    ($func_ptr:expr, $args:expr, $ret:ty, []) => {{
        let f: extern "C" fn() -> $ret = std::mem::transmute($func_ptr);
        f()
    }};
    ($func_ptr:expr, $args:expr, $ret:ty, [$($idx:tt),+]) => {{
        let f: extern "C" fn($(jit_call!(@ty $idx)),+) -> $ret = std::mem::transmute($func_ptr);
        f($($args[$idx]),+)
    }};
    (@ty $idx:tt) => { f32 };
}

/// Dispatch a JIT function call with an output pointer parameter.
/// The function signature is `fn(args..., *mut f32) -> ()`.
macro_rules! jit_call_outptr {
    ($func_ptr:expr, $args:expr, $out_ptr:expr, []) => {{
        let f: extern "C" fn(*mut f32) = std::mem::transmute($func_ptr);
        f($out_ptr)
    }};
    ($func_ptr:expr, $args:expr, $out_ptr:expr, [$($idx:tt),+]) => {{
        let f: extern "C" fn($(jit_call_outptr!(@ty $idx),)+ *mut f32) = std::mem::transmute($func_ptr);
        f($($args[$idx],)+ $out_ptr)
    }};
    (@ty $idx:tt) => { f32 };
}

use crate::Type;
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::ir::{
    AbiParam, FuncRef, InstBuilder, MemFlags, Value as CraneliftValue, types,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

// ============================================================================
// Errors
// ============================================================================

/// Error during Cranelift compilation.
#[derive(Debug, Clone)]
pub enum CraneliftError {
    UnknownVariable(String),
    UnknownFunction(String),
    TypeMismatch {
        op: &'static str,
        left: Type,
        right: Type,
    },
    UnsupportedReturnType(Type),
    JitError(String),
    /// Conditionals not supported in cranelift backend.
    UnsupportedConditional(&'static str),
}

impl std::fmt::Display for CraneliftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CraneliftError::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            CraneliftError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            CraneliftError::TypeMismatch { op, left, right } => {
                write!(f, "type mismatch for {op}: {left} vs {right}")
            }
            CraneliftError::UnsupportedReturnType(t) => {
                write!(f, "unsupported return type: {t} (only scalar supported)")
            }
            CraneliftError::JitError(msg) => write!(f, "JIT error: {msg}"),
            CraneliftError::UnsupportedConditional(what) => {
                write!(
                    f,
                    "conditionals not supported in linalg cranelift backend: {what}"
                )
            }
        }
    }
}

impl std::error::Error for CraneliftError {}

// ============================================================================
// Math function wrappers
// ============================================================================

extern "C" fn math_sqrt(x: f32) -> f32 {
    x.sqrt()
}
extern "C" fn math_pow(base: f32, exp: f32) -> f32 {
    base.powf(exp)
}

struct MathSymbol {
    name: &'static str,
    ptr: *const u8,
}

fn math_symbols() -> Vec<MathSymbol> {
    vec![
        MathSymbol {
            name: "linalg_sqrt",
            ptr: math_sqrt as *const u8,
        },
        MathSymbol {
            name: "linalg_pow",
            ptr: math_pow as *const u8,
        },
    ]
}

// ============================================================================
// Typed values during compilation
// ============================================================================

/// A typed value during compilation.
/// Scalars are single CraneliftValue, vectors are multiple CraneliftValues.
#[derive(Clone)]
pub enum TypedValue {
    Scalar(CraneliftValue),
    Vec2([CraneliftValue; 2]),
    #[cfg(feature = "3d")]
    Vec3([CraneliftValue; 3]),
    #[cfg(feature = "4d")]
    Vec4([CraneliftValue; 4]),
    /// Mat2 stored as [c0r0, c0r1, c1r0, c1r1] (column-major)
    Mat2([CraneliftValue; 4]),
    /// Mat3 stored as 9 values (column-major)
    #[cfg(feature = "3d")]
    Mat3([CraneliftValue; 9]),
}

impl TypedValue {
    fn typ(&self) -> Type {
        match self {
            TypedValue::Scalar(_) => Type::Scalar,
            TypedValue::Vec2(_) => Type::Vec2,
            #[cfg(feature = "3d")]
            TypedValue::Vec3(_) => Type::Vec3,
            #[cfg(feature = "4d")]
            TypedValue::Vec4(_) => Type::Vec4,
            TypedValue::Mat2(_) => Type::Mat2,
            #[cfg(feature = "3d")]
            TypedValue::Mat3(_) => Type::Mat3,
        }
    }

    fn as_scalar(&self) -> Option<CraneliftValue> {
        match self {
            TypedValue::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    fn as_vec2(&self) -> Option<[CraneliftValue; 2]> {
        match self {
            TypedValue::Vec2(v) => Some(*v),
            _ => None,
        }
    }

    #[cfg(feature = "3d")]
    fn as_vec3(&self) -> Option<[CraneliftValue; 3]> {
        match self {
            TypedValue::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    #[cfg(feature = "4d")]
    #[allow(dead_code)]
    fn as_vec4(&self) -> Option<[CraneliftValue; 4]> {
        match self {
            TypedValue::Vec4(v) => Some(*v),
            _ => None,
        }
    }

    fn as_mat2(&self) -> Option<[CraneliftValue; 4]> {
        match self {
            TypedValue::Mat2(m) => Some(*m),
            _ => None,
        }
    }

    #[cfg(feature = "3d")]
    fn as_mat3(&self) -> Option<[CraneliftValue; 9]> {
        match self {
            TypedValue::Mat3(m) => Some(*m),
            _ => None,
        }
    }
}

// ============================================================================
// Variable specification
// ============================================================================

/// Specification of a variable with its type.
#[derive(Debug, Clone)]
pub struct VarSpec {
    pub name: String,
    pub typ: Type,
}

impl VarSpec {
    pub fn new(name: impl Into<String>, typ: Type) -> Self {
        Self {
            name: name.into(),
            typ,
        }
    }

    /// Number of f32 parameters this variable needs.
    pub fn param_count(&self) -> usize {
        match self.typ {
            Type::Scalar => 1,
            Type::Vec2 => 2,
            #[cfg(feature = "3d")]
            Type::Vec3 => 3,
            #[cfg(feature = "4d")]
            Type::Vec4 => 4,
            Type::Mat2 => 4,
            #[cfg(feature = "3d")]
            Type::Mat3 => 9,
            #[cfg(feature = "4d")]
            Type::Mat4 => 16,
        }
    }
}

// ============================================================================
// Compiled Function
// ============================================================================

/// A compiled linalg function.
pub struct CompiledLinalgFn {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

unsafe impl Send for CompiledLinalgFn {}
unsafe impl Sync for CompiledLinalgFn {}

impl CompiledLinalgFn {
    /// Calls the compiled function.
    /// All vector components are flattened into the args array.
    pub fn call(&self, args: &[f32]) -> f32 {
        assert_eq!(args.len(), self.param_count, "wrong number of arguments");

        unsafe {
            match self.param_count {
                0 => jit_call!(self.func_ptr, args, f32, []),
                1 => jit_call!(self.func_ptr, args, f32, [0]),
                2 => jit_call!(self.func_ptr, args, f32, [0, 1]),
                3 => jit_call!(self.func_ptr, args, f32, [0, 1, 2]),
                4 => jit_call!(self.func_ptr, args, f32, [0, 1, 2, 3]),
                5 => jit_call!(self.func_ptr, args, f32, [0, 1, 2, 3, 4]),
                6 => jit_call!(self.func_ptr, args, f32, [0, 1, 2, 3, 4, 5]),
                7 => jit_call!(self.func_ptr, args, f32, [0, 1, 2, 3, 4, 5, 6]),
                8 => jit_call!(self.func_ptr, args, f32, [0, 1, 2, 3, 4, 5, 6, 7]),
                9 => jit_call!(self.func_ptr, args, f32, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
                10 => jit_call!(self.func_ptr, args, f32, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                11 => jit_call!(self.func_ptr, args, f32, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                12 => jit_call!(
                    self.func_ptr,
                    args,
                    f32,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                ),
                13 => jit_call!(
                    self.func_ptr,
                    args,
                    f32,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                ),
                14 => jit_call!(
                    self.func_ptr,
                    args,
                    f32,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                ),
                15 => jit_call!(
                    self.func_ptr,
                    args,
                    f32,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                ),
                16 => jit_call!(
                    self.func_ptr,
                    args,
                    f32,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                ),
                _ => panic!("too many parameters (max 16)"),
            }
        }
    }
}

/// A compiled linalg function that returns a Vec2 (two f32s).
/// Uses output pointer approach for reliable ABI handling.
pub struct CompiledVec2Fn {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

unsafe impl Send for CompiledVec2Fn {}
unsafe impl Sync for CompiledVec2Fn {}

impl CompiledVec2Fn {
    /// Calls the compiled function, returning a Vec2 as [x, y].
    pub fn call(&self, args: &[f32]) -> [f32; 2] {
        assert_eq!(args.len(), self.param_count, "wrong number of arguments");

        let mut output = [0.0f32; 2];
        let out_ptr = output.as_mut_ptr();

        unsafe {
            match self.param_count {
                0 => jit_call_outptr!(self.func_ptr, args, out_ptr, []),
                1 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0]),
                2 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1]),
                3 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2]),
                4 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3]),
                5 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4]),
                6 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5]),
                7 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6]),
                8 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6, 7]),
                9 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
                10 => {
                    jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                }
                11 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                ),
                12 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                ),
                13 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                ),
                14 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                ),
                15 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                ),
                16 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                ),
                _ => panic!("too many parameters (max 16)"),
            };
        }
        output
    }
}

/// A compiled linalg function that returns a Vec3 (three f32s).
/// Uses output pointer approach for reliable ABI handling.
pub struct CompiledVec3Fn {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

unsafe impl Send for CompiledVec3Fn {}
unsafe impl Sync for CompiledVec3Fn {}

impl CompiledVec3Fn {
    /// Calls the compiled function, returning a Vec3 as [x, y, z].
    pub fn call(&self, args: &[f32]) -> [f32; 3] {
        assert_eq!(args.len(), self.param_count, "wrong number of arguments");

        let mut output = [0.0f32; 3];
        let out_ptr = output.as_mut_ptr();

        unsafe {
            match self.param_count {
                0 => jit_call_outptr!(self.func_ptr, args, out_ptr, []),
                1 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0]),
                2 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1]),
                3 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2]),
                4 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3]),
                5 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4]),
                6 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5]),
                7 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6]),
                8 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6, 7]),
                9 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
                10 => {
                    jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                }
                11 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                ),
                12 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                ),
                13 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                ),
                14 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                ),
                15 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                ),
                16 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                ),
                _ => panic!("too many parameters (max 16)"),
            };
        }
        output
    }
}

/// A compiled linalg function that returns a Mat2 (four f32s).
/// Uses output pointer approach for reliable ABI handling.
pub struct CompiledMat2Fn {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

unsafe impl Send for CompiledMat2Fn {}
unsafe impl Sync for CompiledMat2Fn {}

impl CompiledMat2Fn {
    /// Calls the compiled function, returning a Mat2 as [c0r0, c0r1, c1r0, c1r1].
    pub fn call(&self, args: &[f32]) -> [f32; 4] {
        assert_eq!(args.len(), self.param_count, "wrong number of arguments");

        let mut output = [0.0f32; 4];
        let out_ptr = output.as_mut_ptr();

        unsafe {
            match self.param_count {
                0 => jit_call_outptr!(self.func_ptr, args, out_ptr, []),
                1 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0]),
                2 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1]),
                3 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2]),
                4 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3]),
                5 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4]),
                6 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5]),
                7 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6]),
                8 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6, 7]),
                9 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
                10 => {
                    jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                }
                11 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                ),
                12 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                ),
                13 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                ),
                14 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                ),
                15 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                ),
                16 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                ),
                _ => panic!("too many parameters (max 16)"),
            };
        }
        output
    }
}

/// A compiled linalg function that returns a Mat3 (nine f32s).
/// Uses output pointer approach for reliable ABI handling.
#[cfg(feature = "3d")]
pub struct CompiledMat3Fn {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

#[cfg(feature = "3d")]
unsafe impl Send for CompiledMat3Fn {}
#[cfg(feature = "3d")]
unsafe impl Sync for CompiledMat3Fn {}

#[cfg(feature = "3d")]
impl CompiledMat3Fn {
    /// Calls the compiled function, returning a Mat3 as 9 f32s (column-major).
    pub fn call(&self, args: &[f32]) -> [f32; 9] {
        assert_eq!(args.len(), self.param_count, "wrong number of arguments");

        let mut output = [0.0f32; 9];
        let out_ptr = output.as_mut_ptr();

        unsafe {
            match self.param_count {
                0 => jit_call_outptr!(self.func_ptr, args, out_ptr, []),
                1 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0]),
                2 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1]),
                3 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2]),
                4 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3]),
                5 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4]),
                6 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5]),
                7 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6]),
                8 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6, 7]),
                9 => jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
                10 => {
                    jit_call_outptr!(self.func_ptr, args, out_ptr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                }
                11 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                ),
                12 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                ),
                13 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                ),
                14 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                ),
                15 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                ),
                16 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                ),
                17 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                ),
                18 => jit_call_outptr!(
                    self.func_ptr,
                    args,
                    out_ptr,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
                ),
                _ => panic!("too many parameters (max 18)"),
            };
        }
        output
    }
}

// ============================================================================
// JIT Compiler
// ============================================================================

/// JIT compiler for linalg expressions.
pub struct LinalgJit {
    builder: JITBuilder,
}

impl LinalgJit {
    /// Creates a new JIT compiler.
    pub fn new() -> Result<Self, CraneliftError> {
        let mut builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        for sym in math_symbols() {
            builder.symbol(sym.name, sym.ptr);
        }

        Ok(Self { builder })
    }

    /// Compiles an expression that returns a scalar.
    pub fn compile_scalar(
        self,
        ast: &Ast,
        vars: &[VarSpec],
    ) -> Result<CompiledLinalgFn, CraneliftError> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        // Declare math functions
        let sqrt_sig = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };
        let pow_sig = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };

        let sqrt_id = module
            .declare_function("linalg_sqrt", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("linalg_pow", Linkage::Import, &pow_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        // Build function signature: all params as f32, returns f32
        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = module
            .declare_function("linalg_expr", Linkage::Export, &sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        ctx.func.signature = sig;

        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Import math functions
            let sqrt_ref = module.declare_func_in_func(sqrt_id, builder.func);
            let pow_ref = module.declare_func_in_func(pow_id, builder.func);

            // Map variables to typed values
            let block_params = builder.block_params(entry_block).to_vec();
            let mut var_map: HashMap<String, TypedValue> = HashMap::new();
            let mut param_idx = 0;

            for var in vars {
                let typed_val = match var.typ {
                    Type::Scalar => {
                        let v = TypedValue::Scalar(block_params[param_idx]);
                        param_idx += 1;
                        v
                    }
                    Type::Vec2 => {
                        let v = TypedValue::Vec2([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                        ]);
                        param_idx += 2;
                        v
                    }
                    #[cfg(feature = "3d")]
                    Type::Vec3 => {
                        let v = TypedValue::Vec3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                        ]);
                        param_idx += 3;
                        v
                    }
                    #[cfg(feature = "4d")]
                    Type::Vec4 => {
                        let v = TypedValue::Vec4([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                    _ => return Err(CraneliftError::UnsupportedReturnType(var.typ)),
                };
                var_map.insert(var.name.clone(), typed_val);
            }

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
            };
            let result = compile_ast(ast, &mut builder, &var_map, &math_funcs)?;

            let scalar = result
                .as_scalar()
                .ok_or(CraneliftError::UnsupportedReturnType(result.typ()))?;
            builder.ins().return_(&[scalar]);
            builder.finalize();
        }

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        let func_ptr = module.get_finalized_function(func_id);

        Ok(CompiledLinalgFn {
            _module: module,
            func_ptr,
            param_count: total_params,
        })
    }

    /// Compiles an expression that returns a Vec2.
    /// The compiled function takes an output pointer as the last argument.
    pub fn compile_vec2(
        self,
        ast: &Ast,
        vars: &[VarSpec],
    ) -> Result<CompiledVec2Fn, CraneliftError> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        let sqrt_sig = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };
        let pow_sig = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };

        let sqrt_id = module
            .declare_function("linalg_sqrt", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("linalg_pow", Linkage::Import, &pow_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let ptr_type = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.params.push(AbiParam::new(ptr_type));

        let func_id = module
            .declare_function("linalg_expr", Linkage::Export, &sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        ctx.func.signature = sig;

        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let sqrt_ref = module.declare_func_in_func(sqrt_id, builder.func);
            let pow_ref = module.declare_func_in_func(pow_id, builder.func);

            let block_params = builder.block_params(entry_block).to_vec();
            let out_ptr = block_params[total_params];
            let mut var_map: HashMap<String, TypedValue> = HashMap::new();
            let mut param_idx = 0;

            for var in vars {
                let typed_val = match var.typ {
                    Type::Scalar => {
                        let v = TypedValue::Scalar(block_params[param_idx]);
                        param_idx += 1;
                        v
                    }
                    Type::Vec2 => {
                        let v = TypedValue::Vec2([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                        ]);
                        param_idx += 2;
                        v
                    }
                    #[cfg(feature = "3d")]
                    Type::Vec3 => {
                        let v = TypedValue::Vec3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                        ]);
                        param_idx += 3;
                        v
                    }
                    #[cfg(feature = "4d")]
                    Type::Vec4 => {
                        let v = TypedValue::Vec4([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                    Type::Mat2 => {
                        let v = TypedValue::Mat2([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                    #[cfg(feature = "3d")]
                    Type::Mat3 => {
                        let v = TypedValue::Mat3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                            block_params[param_idx + 4],
                            block_params[param_idx + 5],
                            block_params[param_idx + 6],
                            block_params[param_idx + 7],
                            block_params[param_idx + 8],
                        ]);
                        param_idx += 9;
                        v
                    }
                };
                var_map.insert(var.name.clone(), typed_val);
            }

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
            };
            let result = compile_ast(ast, &mut builder, &var_map, &math_funcs)?;

            let [x, y] = result
                .as_vec2()
                .ok_or(CraneliftError::UnsupportedReturnType(result.typ()))?;

            builder
                .ins()
                .store(MemFlags::new(), x, out_ptr, Offset32::new(0));
            builder
                .ins()
                .store(MemFlags::new(), y, out_ptr, Offset32::new(4));
            builder.ins().return_(&[]);
            builder.finalize();
        }

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        let func_ptr = module.get_finalized_function(func_id);

        Ok(CompiledVec2Fn {
            _module: module,
            func_ptr,
            param_count: total_params,
        })
    }

    /// Compiles an expression that returns a Vec3.
    /// The compiled function takes an output pointer as the last argument.
    #[cfg(feature = "3d")]
    pub fn compile_vec3(
        self,
        ast: &Ast,
        vars: &[VarSpec],
    ) -> Result<CompiledVec3Fn, CraneliftError> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        let sqrt_sig = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };
        let pow_sig = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };

        let sqrt_id = module
            .declare_function("linalg_sqrt", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("linalg_pow", Linkage::Import, &pow_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let ptr_type = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.params.push(AbiParam::new(ptr_type));

        let func_id = module
            .declare_function("linalg_expr", Linkage::Export, &sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        ctx.func.signature = sig;

        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let sqrt_ref = module.declare_func_in_func(sqrt_id, builder.func);
            let pow_ref = module.declare_func_in_func(pow_id, builder.func);

            let block_params = builder.block_params(entry_block).to_vec();
            let out_ptr = block_params[total_params];
            let mut var_map: HashMap<String, TypedValue> = HashMap::new();
            let mut param_idx = 0;

            for var in vars {
                let typed_val = match var.typ {
                    Type::Scalar => {
                        let v = TypedValue::Scalar(block_params[param_idx]);
                        param_idx += 1;
                        v
                    }
                    Type::Vec2 => {
                        let v = TypedValue::Vec2([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                        ]);
                        param_idx += 2;
                        v
                    }
                    Type::Vec3 => {
                        let v = TypedValue::Vec3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                        ]);
                        param_idx += 3;
                        v
                    }
                    #[cfg(feature = "4d")]
                    Type::Vec4 => {
                        let v = TypedValue::Vec4([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                    Type::Mat2 => {
                        let v = TypedValue::Mat2([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                    Type::Mat3 => {
                        let v = TypedValue::Mat3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                            block_params[param_idx + 4],
                            block_params[param_idx + 5],
                            block_params[param_idx + 6],
                            block_params[param_idx + 7],
                            block_params[param_idx + 8],
                        ]);
                        param_idx += 9;
                        v
                    }
                };
                var_map.insert(var.name.clone(), typed_val);
            }

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
            };
            let result = compile_ast(ast, &mut builder, &var_map, &math_funcs)?;

            let [x, y, z] = result
                .as_vec3()
                .ok_or(CraneliftError::UnsupportedReturnType(result.typ()))?;

            builder
                .ins()
                .store(MemFlags::new(), x, out_ptr, Offset32::new(0));
            builder
                .ins()
                .store(MemFlags::new(), y, out_ptr, Offset32::new(4));
            builder
                .ins()
                .store(MemFlags::new(), z, out_ptr, Offset32::new(8));
            builder.ins().return_(&[]);
            builder.finalize();
        }

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        let func_ptr = module.get_finalized_function(func_id);

        Ok(CompiledVec3Fn {
            _module: module,
            func_ptr,
            param_count: total_params,
        })
    }

    /// Compiles an expression that returns a Mat2.
    /// The compiled function takes an output pointer as the last argument.
    pub fn compile_mat2(
        self,
        ast: &Ast,
        vars: &[VarSpec],
    ) -> Result<CompiledMat2Fn, CraneliftError> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        let sqrt_sig = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };
        let pow_sig = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };

        let sqrt_id = module
            .declare_function("linalg_sqrt", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("linalg_pow", Linkage::Import, &pow_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let ptr_type = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.params.push(AbiParam::new(ptr_type));

        let func_id = module
            .declare_function("linalg_expr", Linkage::Export, &sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        ctx.func.signature = sig;

        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let sqrt_ref = module.declare_func_in_func(sqrt_id, builder.func);
            let pow_ref = module.declare_func_in_func(pow_id, builder.func);

            let block_params = builder.block_params(entry_block).to_vec();
            let out_ptr = block_params[total_params];
            let mut var_map: HashMap<String, TypedValue> = HashMap::new();
            let mut param_idx = 0;

            for var in vars {
                let typed_val = match var.typ {
                    Type::Scalar => {
                        let v = TypedValue::Scalar(block_params[param_idx]);
                        param_idx += 1;
                        v
                    }
                    Type::Vec2 => {
                        let v = TypedValue::Vec2([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                        ]);
                        param_idx += 2;
                        v
                    }
                    #[cfg(feature = "3d")]
                    Type::Vec3 => {
                        let v = TypedValue::Vec3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                        ]);
                        param_idx += 3;
                        v
                    }
                    #[cfg(feature = "4d")]
                    Type::Vec4 => {
                        let v = TypedValue::Vec4([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                    Type::Mat2 => {
                        let v = TypedValue::Mat2([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                    #[cfg(feature = "3d")]
                    Type::Mat3 => {
                        let v = TypedValue::Mat3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                            block_params[param_idx + 4],
                            block_params[param_idx + 5],
                            block_params[param_idx + 6],
                            block_params[param_idx + 7],
                            block_params[param_idx + 8],
                        ]);
                        param_idx += 9;
                        v
                    }
                };
                var_map.insert(var.name.clone(), typed_val);
            }

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
            };
            let result = compile_ast(ast, &mut builder, &var_map, &math_funcs)?;

            let m = result
                .as_mat2()
                .ok_or(CraneliftError::UnsupportedReturnType(result.typ()))?;

            for i in 0..4 {
                builder.ins().store(
                    MemFlags::new(),
                    m[i],
                    out_ptr,
                    Offset32::new((i * 4) as i32),
                );
            }
            builder.ins().return_(&[]);
            builder.finalize();
        }

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        let func_ptr = module.get_finalized_function(func_id);

        Ok(CompiledMat2Fn {
            _module: module,
            func_ptr,
            param_count: total_params,
        })
    }

    /// Compiles an expression that returns a Mat3.
    /// The compiled function takes an output pointer as the last argument.
    #[cfg(feature = "3d")]
    pub fn compile_mat3(
        self,
        ast: &Ast,
        vars: &[VarSpec],
    ) -> Result<CompiledMat3Fn, CraneliftError> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        let sqrt_sig = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };
        let pow_sig = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };

        let sqrt_id = module
            .declare_function("linalg_sqrt", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("linalg_pow", Linkage::Import, &pow_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let ptr_type = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.params.push(AbiParam::new(ptr_type));

        let func_id = module
            .declare_function("linalg_expr", Linkage::Export, &sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        ctx.func.signature = sig;

        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let sqrt_ref = module.declare_func_in_func(sqrt_id, builder.func);
            let pow_ref = module.declare_func_in_func(pow_id, builder.func);

            let block_params = builder.block_params(entry_block).to_vec();
            let out_ptr = block_params[total_params];
            let mut var_map: HashMap<String, TypedValue> = HashMap::new();
            let mut param_idx = 0;

            for var in vars {
                let typed_val = match var.typ {
                    Type::Scalar => {
                        let v = TypedValue::Scalar(block_params[param_idx]);
                        param_idx += 1;
                        v
                    }
                    Type::Vec2 => {
                        let v = TypedValue::Vec2([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                        ]);
                        param_idx += 2;
                        v
                    }
                    Type::Vec3 => {
                        let v = TypedValue::Vec3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                        ]);
                        param_idx += 3;
                        v
                    }
                    #[cfg(feature = "4d")]
                    Type::Vec4 => {
                        let v = TypedValue::Vec4([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                    Type::Mat2 => {
                        let v = TypedValue::Mat2([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                    Type::Mat3 => {
                        let v = TypedValue::Mat3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                            block_params[param_idx + 4],
                            block_params[param_idx + 5],
                            block_params[param_idx + 6],
                            block_params[param_idx + 7],
                            block_params[param_idx + 8],
                        ]);
                        param_idx += 9;
                        v
                    }
                };
                var_map.insert(var.name.clone(), typed_val);
            }

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
            };
            let result = compile_ast(ast, &mut builder, &var_map, &math_funcs)?;

            let m = result
                .as_mat3()
                .ok_or(CraneliftError::UnsupportedReturnType(result.typ()))?;

            for i in 0..9 {
                builder.ins().store(
                    MemFlags::new(),
                    m[i],
                    out_ptr,
                    Offset32::new((i * 4) as i32),
                );
            }
            builder.ins().return_(&[]);
            builder.finalize();
        }

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        let func_ptr = module.get_finalized_function(func_id);

        Ok(CompiledMat3Fn {
            _module: module,
            func_ptr,
            param_count: total_params,
        })
    }
}

struct MathFuncs {
    sqrt: FuncRef,
    pow: FuncRef,
}

fn compile_ast(
    ast: &Ast,
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, TypedValue>,
    math: &MathFuncs,
) -> Result<TypedValue, CraneliftError> {
    match ast {
        Ast::Num(n) => Ok(TypedValue::Scalar(builder.ins().f32const(*n))),

        Ast::Var(name) => vars
            .get(name)
            .cloned()
            .ok_or_else(|| CraneliftError::UnknownVariable(name.clone())),

        Ast::BinOp(op, left, right) => {
            let l = compile_ast(left, builder, vars, math)?;
            let r = compile_ast(right, builder, vars, math)?;
            compile_binop(*op, l, r, builder, math)
        }

        Ast::UnaryOp(op, inner) => {
            let v = compile_ast(inner, builder, vars, math)?;
            compile_unaryop(*op, v, builder)
        }

        Ast::Call(name, args) => {
            let arg_vals: Vec<TypedValue> = args
                .iter()
                .map(|a| compile_ast(a, builder, vars, math))
                .collect::<Result<_, _>>()?;
            compile_call(name, arg_vals, builder, math)
        }

        Ast::Compare(_, _, _) => Err(CraneliftError::UnsupportedConditional("Compare")),

        Ast::And(_, _) => Err(CraneliftError::UnsupportedConditional("And")),

        Ast::Or(_, _) => Err(CraneliftError::UnsupportedConditional("Or")),

        Ast::If(_, _, _) => Err(CraneliftError::UnsupportedConditional("If")),
    }
}

fn compile_binop(
    op: BinOp,
    left: TypedValue,
    right: TypedValue,
    builder: &mut FunctionBuilder,
    math: &MathFuncs,
) -> Result<TypedValue, CraneliftError> {
    match (op, &left, &right) {
        // Scalar operations
        (BinOp::Add, TypedValue::Scalar(l), TypedValue::Scalar(r)) => {
            Ok(TypedValue::Scalar(builder.ins().fadd(*l, *r)))
        }
        (BinOp::Sub, TypedValue::Scalar(l), TypedValue::Scalar(r)) => {
            Ok(TypedValue::Scalar(builder.ins().fsub(*l, *r)))
        }
        (BinOp::Mul, TypedValue::Scalar(l), TypedValue::Scalar(r)) => {
            Ok(TypedValue::Scalar(builder.ins().fmul(*l, *r)))
        }
        (BinOp::Div, TypedValue::Scalar(l), TypedValue::Scalar(r)) => {
            Ok(TypedValue::Scalar(builder.ins().fdiv(*l, *r)))
        }
        (BinOp::Pow, TypedValue::Scalar(l), TypedValue::Scalar(r)) => {
            let call = builder.ins().call(math.pow, &[*l, *r]);
            Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
        }

        // Vec2 + Vec2
        (BinOp::Add, TypedValue::Vec2(l), TypedValue::Vec2(r)) => Ok(TypedValue::Vec2([
            builder.ins().fadd(l[0], r[0]),
            builder.ins().fadd(l[1], r[1]),
        ])),
        (BinOp::Sub, TypedValue::Vec2(l), TypedValue::Vec2(r)) => Ok(TypedValue::Vec2([
            builder.ins().fsub(l[0], r[0]),
            builder.ins().fsub(l[1], r[1]),
        ])),

        // Vec2 * Scalar
        (BinOp::Mul, TypedValue::Vec2(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec2([
            builder.ins().fmul(v[0], *s),
            builder.ins().fmul(v[1], *s),
        ])),
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Vec2(v)) => Ok(TypedValue::Vec2([
            builder.ins().fmul(*s, v[0]),
            builder.ins().fmul(*s, v[1]),
        ])),

        // Vec2 / Scalar
        (BinOp::Div, TypedValue::Vec2(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec2([
            builder.ins().fdiv(v[0], *s),
            builder.ins().fdiv(v[1], *s),
        ])),

        #[cfg(feature = "3d")]
        (BinOp::Add, TypedValue::Vec3(l), TypedValue::Vec3(r)) => Ok(TypedValue::Vec3([
            builder.ins().fadd(l[0], r[0]),
            builder.ins().fadd(l[1], r[1]),
            builder.ins().fadd(l[2], r[2]),
        ])),
        #[cfg(feature = "3d")]
        (BinOp::Sub, TypedValue::Vec3(l), TypedValue::Vec3(r)) => Ok(TypedValue::Vec3([
            builder.ins().fsub(l[0], r[0]),
            builder.ins().fsub(l[1], r[1]),
            builder.ins().fsub(l[2], r[2]),
        ])),
        #[cfg(feature = "3d")]
        (BinOp::Mul, TypedValue::Vec3(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec3([
            builder.ins().fmul(v[0], *s),
            builder.ins().fmul(v[1], *s),
            builder.ins().fmul(v[2], *s),
        ])),
        #[cfg(feature = "3d")]
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Vec3(v)) => Ok(TypedValue::Vec3([
            builder.ins().fmul(*s, v[0]),
            builder.ins().fmul(*s, v[1]),
            builder.ins().fmul(*s, v[2]),
        ])),
        #[cfg(feature = "3d")]
        (BinOp::Div, TypedValue::Vec3(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec3([
            builder.ins().fdiv(v[0], *s),
            builder.ins().fdiv(v[1], *s),
            builder.ins().fdiv(v[2], *s),
        ])),

        #[cfg(feature = "4d")]
        (BinOp::Add, TypedValue::Vec4(l), TypedValue::Vec4(r)) => Ok(TypedValue::Vec4([
            builder.ins().fadd(l[0], r[0]),
            builder.ins().fadd(l[1], r[1]),
            builder.ins().fadd(l[2], r[2]),
            builder.ins().fadd(l[3], r[3]),
        ])),
        #[cfg(feature = "4d")]
        (BinOp::Sub, TypedValue::Vec4(l), TypedValue::Vec4(r)) => Ok(TypedValue::Vec4([
            builder.ins().fsub(l[0], r[0]),
            builder.ins().fsub(l[1], r[1]),
            builder.ins().fsub(l[2], r[2]),
            builder.ins().fsub(l[3], r[3]),
        ])),
        #[cfg(feature = "4d")]
        (BinOp::Mul, TypedValue::Vec4(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec4([
            builder.ins().fmul(v[0], *s),
            builder.ins().fmul(v[1], *s),
            builder.ins().fmul(v[2], *s),
            builder.ins().fmul(v[3], *s),
        ])),
        #[cfg(feature = "4d")]
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Vec4(v)) => Ok(TypedValue::Vec4([
            builder.ins().fmul(*s, v[0]),
            builder.ins().fmul(*s, v[1]),
            builder.ins().fmul(*s, v[2]),
            builder.ins().fmul(*s, v[3]),
        ])),
        #[cfg(feature = "4d")]
        (BinOp::Div, TypedValue::Vec4(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec4([
            builder.ins().fdiv(v[0], *s),
            builder.ins().fdiv(v[1], *s),
            builder.ins().fdiv(v[2], *s),
            builder.ins().fdiv(v[3], *s),
        ])),

        // Mat2 + Mat2, Mat2 - Mat2
        (BinOp::Add, TypedValue::Mat2(a), TypedValue::Mat2(b)) => Ok(TypedValue::Mat2([
            builder.ins().fadd(a[0], b[0]),
            builder.ins().fadd(a[1], b[1]),
            builder.ins().fadd(a[2], b[2]),
            builder.ins().fadd(a[3], b[3]),
        ])),
        (BinOp::Sub, TypedValue::Mat2(a), TypedValue::Mat2(b)) => Ok(TypedValue::Mat2([
            builder.ins().fsub(a[0], b[0]),
            builder.ins().fsub(a[1], b[1]),
            builder.ins().fsub(a[2], b[2]),
            builder.ins().fsub(a[3], b[3]),
        ])),

        // Mat2 * Scalar, Scalar * Mat2
        (BinOp::Mul, TypedValue::Mat2(m), TypedValue::Scalar(s)) => Ok(TypedValue::Mat2([
            builder.ins().fmul(m[0], *s),
            builder.ins().fmul(m[1], *s),
            builder.ins().fmul(m[2], *s),
            builder.ins().fmul(m[3], *s),
        ])),
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Mat2(m)) => Ok(TypedValue::Mat2([
            builder.ins().fmul(*s, m[0]),
            builder.ins().fmul(*s, m[1]),
            builder.ins().fmul(*s, m[2]),
            builder.ins().fmul(*s, m[3]),
        ])),

        // Mat2 * Mat2 (column-major)
        // result[i] = a[i%2] * b[(i/2)*2] + a[i%2 + 2] * b[(i/2)*2 + 1]
        (BinOp::Mul, TypedValue::Mat2(a), TypedValue::Mat2(b)) => {
            // r[0] = a[0]*b[0] + a[2]*b[1]
            let r0_t1 = builder.ins().fmul(a[0], b[0]);
            let r0_t2 = builder.ins().fmul(a[2], b[1]);
            let r0 = builder.ins().fadd(r0_t1, r0_t2);
            // r[1] = a[1]*b[0] + a[3]*b[1]
            let r1_t1 = builder.ins().fmul(a[1], b[0]);
            let r1_t2 = builder.ins().fmul(a[3], b[1]);
            let r1 = builder.ins().fadd(r1_t1, r1_t2);
            // r[2] = a[0]*b[2] + a[2]*b[3]
            let r2_t1 = builder.ins().fmul(a[0], b[2]);
            let r2_t2 = builder.ins().fmul(a[2], b[3]);
            let r2 = builder.ins().fadd(r2_t1, r2_t2);
            // r[3] = a[1]*b[2] + a[3]*b[3]
            let r3_t1 = builder.ins().fmul(a[1], b[2]);
            let r3_t2 = builder.ins().fmul(a[3], b[3]);
            let r3 = builder.ins().fadd(r3_t1, r3_t2);
            Ok(TypedValue::Mat2([r0, r1, r2, r3]))
        }

        // Mat2 * Vec2 (column vector, column-major storage)
        // result[i] = m[i] * v[0] + m[i+2] * v[1]
        (BinOp::Mul, TypedValue::Mat2(m), TypedValue::Vec2(v)) => {
            let x_t1 = builder.ins().fmul(m[0], v[0]);
            let x_t2 = builder.ins().fmul(m[2], v[1]);
            let x = builder.ins().fadd(x_t1, x_t2);
            let y_t1 = builder.ins().fmul(m[1], v[0]);
            let y_t2 = builder.ins().fmul(m[3], v[1]);
            let y = builder.ins().fadd(y_t1, y_t2);
            Ok(TypedValue::Vec2([x, y]))
        }

        // Vec2 * Mat2 (row vector, column-major storage)
        // result[j] = v[0] * m[j*2] + v[1] * m[j*2+1]
        (BinOp::Mul, TypedValue::Vec2(v), TypedValue::Mat2(m)) => {
            let x_t1 = builder.ins().fmul(v[0], m[0]);
            let x_t2 = builder.ins().fmul(v[1], m[1]);
            let x = builder.ins().fadd(x_t1, x_t2);
            let y_t1 = builder.ins().fmul(v[0], m[2]);
            let y_t2 = builder.ins().fmul(v[1], m[3]);
            let y = builder.ins().fadd(y_t1, y_t2);
            Ok(TypedValue::Vec2([x, y]))
        }

        // Mat3 + Mat3, Mat3 - Mat3
        #[cfg(feature = "3d")]
        (BinOp::Add, TypedValue::Mat3(a), TypedValue::Mat3(b)) => Ok(TypedValue::Mat3([
            builder.ins().fadd(a[0], b[0]),
            builder.ins().fadd(a[1], b[1]),
            builder.ins().fadd(a[2], b[2]),
            builder.ins().fadd(a[3], b[3]),
            builder.ins().fadd(a[4], b[4]),
            builder.ins().fadd(a[5], b[5]),
            builder.ins().fadd(a[6], b[6]),
            builder.ins().fadd(a[7], b[7]),
            builder.ins().fadd(a[8], b[8]),
        ])),
        #[cfg(feature = "3d")]
        (BinOp::Sub, TypedValue::Mat3(a), TypedValue::Mat3(b)) => Ok(TypedValue::Mat3([
            builder.ins().fsub(a[0], b[0]),
            builder.ins().fsub(a[1], b[1]),
            builder.ins().fsub(a[2], b[2]),
            builder.ins().fsub(a[3], b[3]),
            builder.ins().fsub(a[4], b[4]),
            builder.ins().fsub(a[5], b[5]),
            builder.ins().fsub(a[6], b[6]),
            builder.ins().fsub(a[7], b[7]),
            builder.ins().fsub(a[8], b[8]),
        ])),

        // Mat3 * Scalar, Scalar * Mat3
        #[cfg(feature = "3d")]
        (BinOp::Mul, TypedValue::Mat3(m), TypedValue::Scalar(s)) => Ok(TypedValue::Mat3([
            builder.ins().fmul(m[0], *s),
            builder.ins().fmul(m[1], *s),
            builder.ins().fmul(m[2], *s),
            builder.ins().fmul(m[3], *s),
            builder.ins().fmul(m[4], *s),
            builder.ins().fmul(m[5], *s),
            builder.ins().fmul(m[6], *s),
            builder.ins().fmul(m[7], *s),
            builder.ins().fmul(m[8], *s),
        ])),
        #[cfg(feature = "3d")]
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Mat3(m)) => Ok(TypedValue::Mat3([
            builder.ins().fmul(*s, m[0]),
            builder.ins().fmul(*s, m[1]),
            builder.ins().fmul(*s, m[2]),
            builder.ins().fmul(*s, m[3]),
            builder.ins().fmul(*s, m[4]),
            builder.ins().fmul(*s, m[5]),
            builder.ins().fmul(*s, m[6]),
            builder.ins().fmul(*s, m[7]),
            builder.ins().fmul(*s, m[8]),
        ])),

        // Mat3 * Mat3 (column-major)
        // result[col*3 + row] = a[row] * b[col*3] + a[3+row] * b[col*3+1] + a[6+row] * b[col*3+2]
        #[cfg(feature = "3d")]
        (BinOp::Mul, TypedValue::Mat3(a), TypedValue::Mat3(b)) => {
            let mut result = [a[0]; 9]; // placeholder
            for col in 0..3 {
                for row in 0..3 {
                    let t1 = builder.ins().fmul(a[row], b[col * 3]);
                    let t2 = builder.ins().fmul(a[3 + row], b[col * 3 + 1]);
                    let t3 = builder.ins().fmul(a[6 + row], b[col * 3 + 2]);
                    let sum12 = builder.ins().fadd(t1, t2);
                    result[col * 3 + row] = builder.ins().fadd(sum12, t3);
                }
            }
            Ok(TypedValue::Mat3(result))
        }

        // Mat3 * Vec3 (column vector, column-major storage)
        #[cfg(feature = "3d")]
        (BinOp::Mul, TypedValue::Mat3(m), TypedValue::Vec3(v)) => {
            let x_t1 = builder.ins().fmul(m[0], v[0]);
            let x_t2 = builder.ins().fmul(m[3], v[1]);
            let x_t3 = builder.ins().fmul(m[6], v[2]);
            let x_12 = builder.ins().fadd(x_t1, x_t2);
            let x = builder.ins().fadd(x_12, x_t3);
            let y_t1 = builder.ins().fmul(m[1], v[0]);
            let y_t2 = builder.ins().fmul(m[4], v[1]);
            let y_t3 = builder.ins().fmul(m[7], v[2]);
            let y_12 = builder.ins().fadd(y_t1, y_t2);
            let y = builder.ins().fadd(y_12, y_t3);
            let z_t1 = builder.ins().fmul(m[2], v[0]);
            let z_t2 = builder.ins().fmul(m[5], v[1]);
            let z_t3 = builder.ins().fmul(m[8], v[2]);
            let z_12 = builder.ins().fadd(z_t1, z_t2);
            let z = builder.ins().fadd(z_12, z_t3);
            Ok(TypedValue::Vec3([x, y, z]))
        }

        // Vec3 * Mat3 (row vector, column-major storage)
        #[cfg(feature = "3d")]
        (BinOp::Mul, TypedValue::Vec3(v), TypedValue::Mat3(m)) => {
            let x_t1 = builder.ins().fmul(v[0], m[0]);
            let x_t2 = builder.ins().fmul(v[1], m[1]);
            let x_t3 = builder.ins().fmul(v[2], m[2]);
            let x_12 = builder.ins().fadd(x_t1, x_t2);
            let x = builder.ins().fadd(x_12, x_t3);
            let y_t1 = builder.ins().fmul(v[0], m[3]);
            let y_t2 = builder.ins().fmul(v[1], m[4]);
            let y_t3 = builder.ins().fmul(v[2], m[5]);
            let y_12 = builder.ins().fadd(y_t1, y_t2);
            let y = builder.ins().fadd(y_12, y_t3);
            let z_t1 = builder.ins().fmul(v[0], m[6]);
            let z_t2 = builder.ins().fmul(v[1], m[7]);
            let z_t3 = builder.ins().fmul(v[2], m[8]);
            let z_12 = builder.ins().fadd(z_t1, z_t2);
            let z = builder.ins().fadd(z_12, z_t3);
            Ok(TypedValue::Vec3([x, y, z]))
        }

        _ => Err(CraneliftError::TypeMismatch {
            op: match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Pow => "^",
            },
            left: left.typ(),
            right: right.typ(),
        }),
    }
}

fn compile_unaryop(
    op: UnaryOp,
    val: TypedValue,
    builder: &mut FunctionBuilder,
) -> Result<TypedValue, CraneliftError> {
    match op {
        UnaryOp::Neg => match val {
            TypedValue::Scalar(v) => Ok(TypedValue::Scalar(builder.ins().fneg(v))),
            TypedValue::Vec2(v) => Ok(TypedValue::Vec2([
                builder.ins().fneg(v[0]),
                builder.ins().fneg(v[1]),
            ])),
            #[cfg(feature = "3d")]
            TypedValue::Vec3(v) => Ok(TypedValue::Vec3([
                builder.ins().fneg(v[0]),
                builder.ins().fneg(v[1]),
                builder.ins().fneg(v[2]),
            ])),
            #[cfg(feature = "4d")]
            TypedValue::Vec4(v) => Ok(TypedValue::Vec4([
                builder.ins().fneg(v[0]),
                builder.ins().fneg(v[1]),
                builder.ins().fneg(v[2]),
                builder.ins().fneg(v[3]),
            ])),
            TypedValue::Mat2(m) => Ok(TypedValue::Mat2([
                builder.ins().fneg(m[0]),
                builder.ins().fneg(m[1]),
                builder.ins().fneg(m[2]),
                builder.ins().fneg(m[3]),
            ])),
            #[cfg(feature = "3d")]
            TypedValue::Mat3(m) => Ok(TypedValue::Mat3([
                builder.ins().fneg(m[0]),
                builder.ins().fneg(m[1]),
                builder.ins().fneg(m[2]),
                builder.ins().fneg(m[3]),
                builder.ins().fneg(m[4]),
                builder.ins().fneg(m[5]),
                builder.ins().fneg(m[6]),
                builder.ins().fneg(m[7]),
                builder.ins().fneg(m[8]),
            ])),
        },
        UnaryOp::Not => Err(CraneliftError::UnsupportedConditional("Not")),
    }
}

fn compile_call(
    name: &str,
    args: Vec<TypedValue>,
    builder: &mut FunctionBuilder,
    math: &MathFuncs,
) -> Result<TypedValue, CraneliftError> {
    match name {
        "dot" => {
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1]) {
                (TypedValue::Vec2(a), TypedValue::Vec2(b)) => {
                    let x = builder.ins().fmul(a[0], b[0]);
                    let y = builder.ins().fmul(a[1], b[1]);
                    Ok(TypedValue::Scalar(builder.ins().fadd(x, y)))
                }
                #[cfg(feature = "3d")]
                (TypedValue::Vec3(a), TypedValue::Vec3(b)) => {
                    let x = builder.ins().fmul(a[0], b[0]);
                    let y = builder.ins().fmul(a[1], b[1]);
                    let z = builder.ins().fmul(a[2], b[2]);
                    let xy = builder.ins().fadd(x, y);
                    Ok(TypedValue::Scalar(builder.ins().fadd(xy, z)))
                }
                #[cfg(feature = "4d")]
                (TypedValue::Vec4(a), TypedValue::Vec4(b)) => {
                    let x = builder.ins().fmul(a[0], b[0]);
                    let y = builder.ins().fmul(a[1], b[1]);
                    let z = builder.ins().fmul(a[2], b[2]);
                    let w = builder.ins().fmul(a[3], b[3]);
                    let xy = builder.ins().fadd(x, y);
                    let zw = builder.ins().fadd(z, w);
                    Ok(TypedValue::Scalar(builder.ins().fadd(xy, zw)))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "length" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Vec2(v) => {
                    let x2 = builder.ins().fmul(v[0], v[0]);
                    let y2 = builder.ins().fmul(v[1], v[1]);
                    let sum = builder.ins().fadd(x2, y2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                #[cfg(feature = "3d")]
                TypedValue::Vec3(v) => {
                    let x2 = builder.ins().fmul(v[0], v[0]);
                    let y2 = builder.ins().fmul(v[1], v[1]);
                    let z2 = builder.ins().fmul(v[2], v[2]);
                    let xy = builder.ins().fadd(x2, y2);
                    let sum = builder.ins().fadd(xy, z2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                #[cfg(feature = "4d")]
                TypedValue::Vec4(v) => {
                    let x2 = builder.ins().fmul(v[0], v[0]);
                    let y2 = builder.ins().fmul(v[1], v[1]);
                    let z2 = builder.ins().fmul(v[2], v[2]);
                    let w2 = builder.ins().fmul(v[3], v[3]);
                    let xy = builder.ins().fadd(x2, y2);
                    let zw = builder.ins().fadd(z2, w2);
                    let sum = builder.ins().fadd(xy, zw);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "distance" => {
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1]) {
                (TypedValue::Vec2(a), TypedValue::Vec2(b)) => {
                    let dx = builder.ins().fsub(a[0], b[0]);
                    let dy = builder.ins().fsub(a[1], b[1]);
                    let dx2 = builder.ins().fmul(dx, dx);
                    let dy2 = builder.ins().fmul(dy, dy);
                    let sum = builder.ins().fadd(dx2, dy2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                #[cfg(feature = "3d")]
                (TypedValue::Vec3(a), TypedValue::Vec3(b)) => {
                    let dx = builder.ins().fsub(a[0], b[0]);
                    let dy = builder.ins().fsub(a[1], b[1]);
                    let dz = builder.ins().fsub(a[2], b[2]);
                    let dx2 = builder.ins().fmul(dx, dx);
                    let dy2 = builder.ins().fmul(dy, dy);
                    let dz2 = builder.ins().fmul(dz, dz);
                    let dxy = builder.ins().fadd(dx2, dy2);
                    let sum = builder.ins().fadd(dxy, dz2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                #[cfg(feature = "4d")]
                (TypedValue::Vec4(a), TypedValue::Vec4(b)) => {
                    let dx = builder.ins().fsub(a[0], b[0]);
                    let dy = builder.ins().fsub(a[1], b[1]);
                    let dz = builder.ins().fsub(a[2], b[2]);
                    let dw = builder.ins().fsub(a[3], b[3]);
                    let dx2 = builder.ins().fmul(dx, dx);
                    let dy2 = builder.ins().fmul(dy, dy);
                    let dz2 = builder.ins().fmul(dz, dz);
                    let dw2 = builder.ins().fmul(dw, dw);
                    let dxy = builder.ins().fadd(dx2, dy2);
                    let dzw = builder.ins().fadd(dz2, dw2);
                    let sum = builder.ins().fadd(dxy, dzw);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        _ => Err(CraneliftError::UnknownFunction(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    #[test]
    fn test_scalar_add() {
        let expr = Expr::parse("a + b").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[
                    VarSpec::new("a", Type::Scalar),
                    VarSpec::new("b", Type::Scalar),
                ],
            )
            .unwrap();
        assert_eq!(func.call(&[3.0, 4.0]), 7.0);
    }

    #[test]
    fn test_dot_vec2() {
        let expr = Expr::parse("dot(a, b)").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[VarSpec::new("a", Type::Vec2), VarSpec::new("b", Type::Vec2)],
            )
            .unwrap();
        // dot([1, 2], [3, 4]) = 1*3 + 2*4 = 11
        assert_eq!(func.call(&[1.0, 2.0, 3.0, 4.0]), 11.0);
    }

    #[test]
    fn test_length_vec2() {
        let expr = Expr::parse("length(v)").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(expr.ast(), &[VarSpec::new("v", Type::Vec2)])
            .unwrap();
        // length([3, 4]) = 5
        assert_eq!(func.call(&[3.0, 4.0]), 5.0);
    }

    #[test]
    fn test_distance_vec2() {
        let expr = Expr::parse("distance(a, b)").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[VarSpec::new("a", Type::Vec2), VarSpec::new("b", Type::Vec2)],
            )
            .unwrap();
        // distance([0, 0], [3, 4]) = 5
        assert_eq!(func.call(&[0.0, 0.0, 3.0, 4.0]), 5.0);
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_dot_vec3() {
        let expr = Expr::parse("dot(a, b)").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[VarSpec::new("a", Type::Vec3), VarSpec::new("b", Type::Vec3)],
            )
            .unwrap();
        // dot([1, 2, 3], [4, 5, 6]) = 1*4 + 2*5 + 3*6 = 32
        assert_eq!(func.call(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn test_complex_expression() {
        // length(a - b) should equal distance(a, b)
        let expr = Expr::parse("length(a - b)").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[VarSpec::new("a", Type::Vec2), VarSpec::new("b", Type::Vec2)],
            )
            .unwrap();
        assert_eq!(func.call(&[0.0, 0.0, 3.0, 4.0]), 5.0);
    }

    #[test]
    fn test_vec_scalar_mul() {
        let expr = Expr::parse("length(v * 2)").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(expr.ast(), &[VarSpec::new("v", Type::Vec2)])
            .unwrap();
        // length([3, 4] * 2) = length([6, 8]) = 10
        assert_eq!(func.call(&[3.0, 4.0]), 10.0);
    }

    #[test]
    fn test_compile_vec2_add() {
        // [1, 2] + [3, 4] = [4, 6]
        let expr = Expr::parse("a + b").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_vec2(
                expr.ast(),
                &[VarSpec::new("a", Type::Vec2), VarSpec::new("b", Type::Vec2)],
            )
            .unwrap();
        let [x, y] = func.call(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(x, 4.0);
        assert_eq!(y, 6.0);
    }

    #[test]
    fn test_compile_vec2_scalar_mul() {
        // [1, 2] * 3 = [3, 6]
        let expr = Expr::parse("v * 3").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_vec2(expr.ast(), &[VarSpec::new("v", Type::Vec2)])
            .unwrap();
        let [x, y] = func.call(&[1.0, 2.0]);
        assert_eq!(x, 3.0);
        assert_eq!(y, 6.0);
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_compile_vec3_add() {
        // [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
        let expr = Expr::parse("a + b").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_vec3(
                expr.ast(),
                &[VarSpec::new("a", Type::Vec3), VarSpec::new("b", Type::Vec3)],
            )
            .unwrap();
        let [x, y, z] = func.call(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(x, 5.0);
        assert_eq!(y, 7.0);
        assert_eq!(z, 9.0);
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_compile_vec3_scalar_mul() {
        // [1, 2, 3] * 2 = [2, 4, 6]
        let expr = Expr::parse("v * 2").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_vec3(expr.ast(), &[VarSpec::new("v", Type::Vec3)])
            .unwrap();
        let [x, y, z] = func.call(&[1.0, 2.0, 3.0]);
        assert_eq!(x, 2.0);
        assert_eq!(y, 4.0);
        assert_eq!(z, 6.0);
    }

    // ========================================================================
    // Matrix tests
    // ========================================================================

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn test_mat2_mul_vec2() {
        // Identity matrix times vector
        // m = [1, 0, 0, 1] (column-major: col0=[1,0], col1=[0,1])
        let expr = Expr::parse("m * v").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_vec2(
                expr.ast(),
                &[VarSpec::new("m", Type::Mat2), VarSpec::new("v", Type::Vec2)],
            )
            .unwrap();
        let [x, y] = func.call(&[1.0, 0.0, 0.0, 1.0, 3.0, 4.0]);
        assert!(approx_eq(x, 3.0));
        assert!(approx_eq(y, 4.0));
    }

    #[test]
    fn test_mat2_mul_vec2_rotation() {
        // 90° rotation matrix (column-major): [0, 1, -1, 0]
        // rotates [1, 0] to [0, 1]
        let expr = Expr::parse("m * v").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_vec2(
                expr.ast(),
                &[VarSpec::new("m", Type::Mat2), VarSpec::new("v", Type::Vec2)],
            )
            .unwrap();
        let [x, y] = func.call(&[0.0, 1.0, -1.0, 0.0, 1.0, 0.0]);
        assert!(approx_eq(x, 0.0));
        assert!(approx_eq(y, 1.0));
    }

    #[test]
    fn test_vec2_mul_mat2() {
        // Row vector times matrix
        let expr = Expr::parse("v * m").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_vec2(
                expr.ast(),
                &[VarSpec::new("v", Type::Vec2), VarSpec::new("m", Type::Mat2)],
            )
            .unwrap();
        // v = [1, 2], m = [1, 2, 3, 4] (column-major)
        // result = [v[0]*m[0] + v[1]*m[1], v[0]*m[2] + v[1]*m[3]]
        //        = [1*1 + 2*2, 1*3 + 2*4] = [5, 11]
        let [x, y] = func.call(&[1.0, 2.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(approx_eq(x, 5.0));
        assert!(approx_eq(y, 11.0));
    }

    #[test]
    fn test_mat2_mul_mat2() {
        // Identity * A = A
        let expr = Expr::parse("a * b").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_mat2(
                expr.ast(),
                &[VarSpec::new("a", Type::Mat2), VarSpec::new("b", Type::Mat2)],
            )
            .unwrap();
        // identity = [1, 0, 0, 1], a = [1, 2, 3, 4]
        let result = func.call(&[1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(approx_eq(result[0], 1.0));
        assert!(approx_eq(result[1], 2.0));
        assert!(approx_eq(result[2], 3.0));
        assert!(approx_eq(result[3], 4.0));
    }

    #[test]
    fn test_mat2_scalar_mul() {
        let expr = Expr::parse("m * 2").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_mat2(expr.ast(), &[VarSpec::new("m", Type::Mat2)])
            .unwrap();
        let result = func.call(&[1.0, 2.0, 3.0, 4.0]);
        assert!(approx_eq(result[0], 2.0));
        assert!(approx_eq(result[1], 4.0));
        assert!(approx_eq(result[2], 6.0));
        assert!(approx_eq(result[3], 8.0));
    }

    #[test]
    fn test_mat2_add() {
        let expr = Expr::parse("a + b").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_mat2(
                expr.ast(),
                &[VarSpec::new("a", Type::Mat2), VarSpec::new("b", Type::Mat2)],
            )
            .unwrap();
        let result = func.call(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(approx_eq(result[0], 6.0));
        assert!(approx_eq(result[1], 8.0));
        assert!(approx_eq(result[2], 10.0));
        assert!(approx_eq(result[3], 12.0));
    }

    #[test]
    fn test_mat2_neg() {
        let expr = Expr::parse("-m").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_mat2(expr.ast(), &[VarSpec::new("m", Type::Mat2)])
            .unwrap();
        let result = func.call(&[1.0, 2.0, 3.0, 4.0]);
        assert!(approx_eq(result[0], -1.0));
        assert!(approx_eq(result[1], -2.0));
        assert!(approx_eq(result[2], -3.0));
        assert!(approx_eq(result[3], -4.0));
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_mat3_mul_vec3() {
        // Identity matrix
        let expr = Expr::parse("m * v").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_vec3(
                expr.ast(),
                &[VarSpec::new("m", Type::Mat3), VarSpec::new("v", Type::Vec3)],
            )
            .unwrap();
        // identity = [1,0,0, 0,1,0, 0,0,1]
        let [x, y, z] = func.call(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(approx_eq(x, 2.0));
        assert!(approx_eq(y, 3.0));
        assert!(approx_eq(z, 4.0));
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_mat3_scalar_mul() {
        let expr = Expr::parse("m * 2").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_mat3(expr.ast(), &[VarSpec::new("m", Type::Mat3)])
            .unwrap();
        let result = func.call(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        for i in 0..9 {
            assert!(approx_eq(result[i], (i as f32 + 1.0) * 2.0));
        }
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_mat3_mul_mat3_identity() {
        let expr = Expr::parse("a * b").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_mat3(
                expr.ast(),
                &[VarSpec::new("a", Type::Mat3), VarSpec::new("b", Type::Mat3)],
            )
            .unwrap();
        // identity * identity = identity
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut args = [0.0f32; 18];
        args[..9].copy_from_slice(&identity);
        args[9..].copy_from_slice(&identity);
        let result = func.call(&args);
        for i in 0..9 {
            assert!(approx_eq(result[i], identity[i]));
        }
    }
}
