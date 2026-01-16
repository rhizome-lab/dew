//! Cranelift JIT compilation for quaternion expressions.
//!
//! Compiles typed expressions to native code via Cranelift.
//!
//! # Representation
//!
//! - Scalar: single f32
//! - Vec3: three f32 values (x, y, z)
//! - Quaternion: four f32 values (x, y, z, w)

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
    AbiParam, FuncRef, InstBuilder, MemFlags, Value as CraneliftValue, condcodes::FloatCC, types,
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
                write!(f, "unsupported return type: {t}")
            }
            CraneliftError::JitError(msg) => write!(f, "JIT error: {msg}"),
            CraneliftError::UnsupportedConditional(what) => {
                write!(
                    f,
                    "conditionals not supported in quaternion cranelift backend: {what}"
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
extern "C" fn math_acos(x: f32) -> f32 {
    x.acos()
}
extern "C" fn math_sin(x: f32) -> f32 {
    x.sin()
}
extern "C" fn math_cos(x: f32) -> f32 {
    x.cos()
}

struct MathSymbol {
    name: &'static str,
    ptr: *const u8,
}

fn math_symbols() -> Vec<MathSymbol> {
    vec![
        MathSymbol {
            name: "quat_sqrt",
            ptr: math_sqrt as *const u8,
        },
        MathSymbol {
            name: "quat_pow",
            ptr: math_pow as *const u8,
        },
        MathSymbol {
            name: "quat_acos",
            ptr: math_acos as *const u8,
        },
        MathSymbol {
            name: "quat_sin",
            ptr: math_sin as *const u8,
        },
        MathSymbol {
            name: "quat_cos",
            ptr: math_cos as *const u8,
        },
    ]
}

// ============================================================================
// Typed values during compilation
// ============================================================================

/// A typed value during compilation.
#[derive(Clone)]
pub enum TypedValue {
    Scalar(CraneliftValue),
    Vec3([CraneliftValue; 3]),
    Quaternion([CraneliftValue; 4]),
}

impl TypedValue {
    fn typ(&self) -> Type {
        match self {
            TypedValue::Scalar(_) => Type::Scalar,
            TypedValue::Vec3(_) => Type::Vec3,
            TypedValue::Quaternion(_) => Type::Quaternion,
        }
    }

    fn as_scalar(&self) -> Option<CraneliftValue> {
        match self {
            TypedValue::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    fn as_vec3(&self) -> Option<[CraneliftValue; 3]> {
        match self {
            TypedValue::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    fn as_quaternion(&self) -> Option<[CraneliftValue; 4]> {
        match self {
            TypedValue::Quaternion(v) => Some(*v),
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
            Type::Vec3 => 3,
            Type::Quaternion => 4,
        }
    }
}

// ============================================================================
// Compiled Function
// ============================================================================

/// A compiled quaternion function that returns a scalar.
pub struct CompiledQuaternionFn {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

unsafe impl Send for CompiledQuaternionFn {}
unsafe impl Sync for CompiledQuaternionFn {}

impl CompiledQuaternionFn {
    /// Calls the compiled function.
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

/// A compiled quaternion function that returns a Vec3 (three f32s).
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

/// A compiled quaternion function that returns a Quaternion (four f32s).
/// Uses output pointer approach for reliable ABI handling.
pub struct CompiledQuatFn {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

unsafe impl Send for CompiledQuatFn {}
unsafe impl Sync for CompiledQuatFn {}

impl CompiledQuatFn {
    /// Calls the compiled function, returning a Quaternion as [x, y, z, w].
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

// ============================================================================
// Math functions struct
// ============================================================================

#[allow(dead_code)]
struct MathFuncs {
    sqrt: FuncRef,
    pow: FuncRef,
    acos: FuncRef,
    sin: FuncRef,
    cos: FuncRef,
}

// ============================================================================
// JIT Compiler
// ============================================================================

/// JIT compiler for quaternion expressions.
pub struct QuaternionJit {
    builder: JITBuilder,
}

impl QuaternionJit {
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
    ) -> Result<CompiledQuaternionFn, CraneliftError> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        // Declare math functions
        let sig_f32_f32 = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };
        let sig_f32_f32_f32 = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };

        let sqrt_id = module
            .declare_function("quat_sqrt", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("quat_pow", Linkage::Import, &sig_f32_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let acos_id = module
            .declare_function("quat_acos", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let sin_id = module
            .declare_function("quat_sin", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("quat_cos", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        // Build function signature
        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = module
            .declare_function("quat_expr", Linkage::Export, &sig)
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
            let math_funcs = MathFuncs {
                sqrt: module.declare_func_in_func(sqrt_id, builder.func),
                pow: module.declare_func_in_func(pow_id, builder.func),
                acos: module.declare_func_in_func(acos_id, builder.func),
                sin: module.declare_func_in_func(sin_id, builder.func),
                cos: module.declare_func_in_func(cos_id, builder.func),
            };

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
                    Type::Vec3 => {
                        let v = TypedValue::Vec3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                        ]);
                        param_idx += 3;
                        v
                    }
                    Type::Quaternion => {
                        let v = TypedValue::Quaternion([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                };
                var_map.insert(var.name.clone(), typed_val);
            }

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

        Ok(CompiledQuaternionFn {
            _module: module,
            func_ptr,
            param_count: total_params,
        })
    }

    /// Compiles an expression that returns a Vec3.
    /// The compiled function takes an output pointer as the last argument.
    pub fn compile_vec3(
        self,
        ast: &Ast,
        vars: &[VarSpec],
    ) -> Result<CompiledVec3Fn, CraneliftError> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        // Declare math functions
        let sig_f32_f32 = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };
        let sig_f32_f32_f32 = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };

        let sqrt_id = module
            .declare_function("quat_sqrt", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("quat_pow", Linkage::Import, &sig_f32_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let acos_id = module
            .declare_function("quat_acos", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let sin_id = module
            .declare_function("quat_sin", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("quat_cos", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        // Build function signature - input params + output pointer, no return
        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let ptr_type = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.params.push(AbiParam::new(ptr_type)); // output pointer

        let func_id = module
            .declare_function("quat_expr", Linkage::Export, &sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        ctx.func.signature = sig;

        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let math_funcs = MathFuncs {
                sqrt: module.declare_func_in_func(sqrt_id, builder.func),
                pow: module.declare_func_in_func(pow_id, builder.func),
                acos: module.declare_func_in_func(acos_id, builder.func),
                sin: module.declare_func_in_func(sin_id, builder.func),
                cos: module.declare_func_in_func(cos_id, builder.func),
            };

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
                    Type::Vec3 => {
                        let v = TypedValue::Vec3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                        ]);
                        param_idx += 3;
                        v
                    }
                    Type::Quaternion => {
                        let v = TypedValue::Quaternion([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                };
                var_map.insert(var.name.clone(), typed_val);
            }

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

    /// Compiles an expression that returns a Quaternion.
    /// The compiled function takes an output pointer as the last argument.
    pub fn compile_quaternion(
        self,
        ast: &Ast,
        vars: &[VarSpec],
    ) -> Result<CompiledQuatFn, CraneliftError> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        // Declare math functions
        let sig_f32_f32 = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };
        let sig_f32_f32_f32 = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };

        let sqrt_id = module
            .declare_function("quat_sqrt", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("quat_pow", Linkage::Import, &sig_f32_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let acos_id = module
            .declare_function("quat_acos", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let sin_id = module
            .declare_function("quat_sin", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("quat_cos", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        // Build function signature - input params + output pointer, no return
        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let ptr_type = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.params.push(AbiParam::new(ptr_type)); // output pointer

        let func_id = module
            .declare_function("quat_expr", Linkage::Export, &sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        ctx.func.signature = sig;

        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let math_funcs = MathFuncs {
                sqrt: module.declare_func_in_func(sqrt_id, builder.func),
                pow: module.declare_func_in_func(pow_id, builder.func),
                acos: module.declare_func_in_func(acos_id, builder.func),
                sin: module.declare_func_in_func(sin_id, builder.func),
                cos: module.declare_func_in_func(cos_id, builder.func),
            };

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
                    Type::Vec3 => {
                        let v = TypedValue::Vec3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                        ]);
                        param_idx += 3;
                        v
                    }
                    Type::Quaternion => {
                        let v = TypedValue::Quaternion([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                };
                var_map.insert(var.name.clone(), typed_val);
            }

            let result = compile_ast(ast, &mut builder, &var_map, &math_funcs)?;

            let [x, y, z, w] = result
                .as_quaternion()
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
            builder
                .ins()
                .store(MemFlags::new(), w, out_ptr, Offset32::new(12));
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

        Ok(CompiledQuatFn {
            _module: module,
            func_ptr,
            param_count: total_params,
        })
    }
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

        // Vec3 + Vec3
        (BinOp::Add, TypedValue::Vec3(l), TypedValue::Vec3(r)) => Ok(TypedValue::Vec3([
            builder.ins().fadd(l[0], r[0]),
            builder.ins().fadd(l[1], r[1]),
            builder.ins().fadd(l[2], r[2]),
        ])),
        (BinOp::Sub, TypedValue::Vec3(l), TypedValue::Vec3(r)) => Ok(TypedValue::Vec3([
            builder.ins().fsub(l[0], r[0]),
            builder.ins().fsub(l[1], r[1]),
            builder.ins().fsub(l[2], r[2]),
        ])),

        // Vec3 * Scalar
        (BinOp::Mul, TypedValue::Vec3(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec3([
            builder.ins().fmul(v[0], *s),
            builder.ins().fmul(v[1], *s),
            builder.ins().fmul(v[2], *s),
        ])),
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Vec3(v)) => Ok(TypedValue::Vec3([
            builder.ins().fmul(*s, v[0]),
            builder.ins().fmul(*s, v[1]),
            builder.ins().fmul(*s, v[2]),
        ])),

        // Quaternion + Quaternion
        (BinOp::Add, TypedValue::Quaternion(l), TypedValue::Quaternion(r)) => {
            Ok(TypedValue::Quaternion([
                builder.ins().fadd(l[0], r[0]),
                builder.ins().fadd(l[1], r[1]),
                builder.ins().fadd(l[2], r[2]),
                builder.ins().fadd(l[3], r[3]),
            ]))
        }
        (BinOp::Sub, TypedValue::Quaternion(l), TypedValue::Quaternion(r)) => {
            Ok(TypedValue::Quaternion([
                builder.ins().fsub(l[0], r[0]),
                builder.ins().fsub(l[1], r[1]),
                builder.ins().fsub(l[2], r[2]),
                builder.ins().fsub(l[3], r[3]),
            ]))
        }

        // Quaternion * Scalar
        (BinOp::Mul, TypedValue::Quaternion(q), TypedValue::Scalar(s)) => {
            Ok(TypedValue::Quaternion([
                builder.ins().fmul(q[0], *s),
                builder.ins().fmul(q[1], *s),
                builder.ins().fmul(q[2], *s),
                builder.ins().fmul(q[3], *s),
            ]))
        }
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Quaternion(q)) => {
            Ok(TypedValue::Quaternion([
                builder.ins().fmul(*s, q[0]),
                builder.ins().fmul(*s, q[1]),
                builder.ins().fmul(*s, q[2]),
                builder.ins().fmul(*s, q[3]),
            ]))
        }

        // Quaternion * Quaternion (Hamilton product)
        (BinOp::Mul, TypedValue::Quaternion(a), TypedValue::Quaternion(b)) => {
            let (x1, y1, z1, w1) = (a[0], a[1], a[2], a[3]);
            let (x2, y2, z2, w2) = (b[0], b[1], b[2], b[3]);

            // x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            let wx2 = builder.ins().fmul(w1, x2);
            let xw2 = builder.ins().fmul(x1, w2);
            let yz2 = builder.ins().fmul(y1, z2);
            let zy2 = builder.ins().fmul(z1, y2);
            let x_part1 = builder.ins().fadd(wx2, xw2);
            let x_part2 = builder.ins().fsub(yz2, zy2);
            let new_x = builder.ins().fadd(x_part1, x_part2);

            // y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            let wy2 = builder.ins().fmul(w1, y2);
            let xz2 = builder.ins().fmul(x1, z2);
            let yw2 = builder.ins().fmul(y1, w2);
            let zx2 = builder.ins().fmul(z1, x2);
            let y_part1 = builder.ins().fsub(wy2, xz2);
            let y_part2 = builder.ins().fadd(yw2, zx2);
            let new_y = builder.ins().fadd(y_part1, y_part2);

            // z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            let wz2 = builder.ins().fmul(w1, z2);
            let xy2 = builder.ins().fmul(x1, y2);
            let yx2 = builder.ins().fmul(y1, x2);
            let zw2 = builder.ins().fmul(z1, w2);
            let z_part1 = builder.ins().fadd(wz2, xy2);
            let z_part2 = builder.ins().fsub(zw2, yx2);
            let new_z = builder.ins().fadd(z_part1, z_part2);

            // w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            let ww2 = builder.ins().fmul(w1, w2);
            let xx2 = builder.ins().fmul(x1, x2);
            let yy2 = builder.ins().fmul(y1, y2);
            let zz2 = builder.ins().fmul(z1, z2);
            let w_part1 = builder.ins().fsub(ww2, xx2);
            let w_part2 = builder.ins().fadd(yy2, zz2);
            let new_w = builder.ins().fsub(w_part1, w_part2);

            Ok(TypedValue::Quaternion([new_x, new_y, new_z, new_w]))
        }

        // Quaternion / Scalar
        (BinOp::Div, TypedValue::Quaternion(q), TypedValue::Scalar(s)) => {
            Ok(TypedValue::Quaternion([
                builder.ins().fdiv(q[0], *s),
                builder.ins().fdiv(q[1], *s),
                builder.ins().fdiv(q[2], *s),
                builder.ins().fdiv(q[3], *s),
            ]))
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
            TypedValue::Vec3(v) => Ok(TypedValue::Vec3([
                builder.ins().fneg(v[0]),
                builder.ins().fneg(v[1]),
                builder.ins().fneg(v[2]),
            ])),
            TypedValue::Quaternion(q) => Ok(TypedValue::Quaternion([
                builder.ins().fneg(q[0]),
                builder.ins().fneg(q[1]),
                builder.ins().fneg(q[2]),
                builder.ins().fneg(q[3]),
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
        "length" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Vec3(v) => {
                    let x2 = builder.ins().fmul(v[0], v[0]);
                    let y2 = builder.ins().fmul(v[1], v[1]);
                    let z2 = builder.ins().fmul(v[2], v[2]);
                    let xy = builder.ins().fadd(x2, y2);
                    let sum = builder.ins().fadd(xy, z2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                TypedValue::Quaternion(q) => {
                    let x2 = builder.ins().fmul(q[0], q[0]);
                    let y2 = builder.ins().fmul(q[1], q[1]);
                    let z2 = builder.ins().fmul(q[2], q[2]);
                    let w2 = builder.ins().fmul(q[3], q[3]);
                    let xy = builder.ins().fadd(x2, y2);
                    let zw = builder.ins().fadd(z2, w2);
                    let sum = builder.ins().fadd(xy, zw);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "dot" => {
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1]) {
                (TypedValue::Vec3(a), TypedValue::Vec3(b)) => {
                    let x = builder.ins().fmul(a[0], b[0]);
                    let y = builder.ins().fmul(a[1], b[1]);
                    let z = builder.ins().fmul(a[2], b[2]);
                    let xy = builder.ins().fadd(x, y);
                    Ok(TypedValue::Scalar(builder.ins().fadd(xy, z)))
                }
                (TypedValue::Quaternion(a), TypedValue::Quaternion(b)) => {
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

        "conj" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Quaternion(q) => Ok(TypedValue::Quaternion([
                    builder.ins().fneg(q[0]),
                    builder.ins().fneg(q[1]),
                    builder.ins().fneg(q[2]),
                    q[3],
                ])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "axis_angle" => {
            // axis_angle(axis, angle) -> quaternion
            // q = [normalize(axis) * sin(angle/2), cos(angle/2)]
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1]) {
                (TypedValue::Vec3(axis), TypedValue::Scalar(angle)) => {
                    // Normalize the axis
                    let ax2 = builder.ins().fmul(axis[0], axis[0]);
                    let ay2 = builder.ins().fmul(axis[1], axis[1]);
                    let az2 = builder.ins().fmul(axis[2], axis[2]);
                    let xy_sum = builder.ins().fadd(ax2, ay2);
                    let sum = builder.ins().fadd(xy_sum, az2);
                    let len_call = builder.ins().call(math.sqrt, &[sum]);
                    let len = builder.inst_results(len_call)[0];

                    let ax_norm = builder.ins().fdiv(axis[0], len);
                    let ay_norm = builder.ins().fdiv(axis[1], len);
                    let az_norm = builder.ins().fdiv(axis[2], len);

                    // half = angle / 2
                    let two = builder.ins().f32const(2.0);
                    let half = builder.ins().fdiv(*angle, two);

                    // sin(half) and cos(half)
                    let sin_call = builder.ins().call(math.sin, &[half]);
                    let sin_half = builder.inst_results(sin_call)[0];
                    let cos_call = builder.ins().call(math.cos, &[half]);
                    let cos_half = builder.inst_results(cos_call)[0];

                    // q = [axis_norm * sin_half, cos_half]
                    let qx = builder.ins().fmul(ax_norm, sin_half);
                    let qy = builder.ins().fmul(ay_norm, sin_half);
                    let qz = builder.ins().fmul(az_norm, sin_half);

                    Ok(TypedValue::Quaternion([qx, qy, qz, cos_half]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "slerp" => {
            // slerp(q1, q2, t) -> quaternion
            // Spherical linear interpolation with branchless selection
            if args.len() != 3 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1], &args[2]) {
                (TypedValue::Quaternion(q1), TypedValue::Quaternion(q2), TypedValue::Scalar(t)) => {
                    // dot = q1 · q2
                    let dot_x = builder.ins().fmul(q1[0], q2[0]);
                    let dot_y = builder.ins().fmul(q1[1], q2[1]);
                    let dot_z = builder.ins().fmul(q1[2], q2[2]);
                    let dot_w = builder.ins().fmul(q1[3], q2[3]);
                    let dot_xy = builder.ins().fadd(dot_x, dot_y);
                    let dot_zw = builder.ins().fadd(dot_z, dot_w);
                    let dot = builder.ins().fadd(dot_xy, dot_zw);

                    // If dot < 0, negate q2 to take shorter path
                    let zero = builder.ins().f32const(0.0);
                    let neg_dot = builder.ins().fcmp(FloatCC::LessThan, dot, zero);

                    // q2_adj = select(neg_dot, -q2, q2)
                    let q2x_neg = builder.ins().fneg(q2[0]);
                    let q2y_neg = builder.ins().fneg(q2[1]);
                    let q2z_neg = builder.ins().fneg(q2[2]);
                    let q2w_neg = builder.ins().fneg(q2[3]);
                    let q2x_adj = builder.ins().select(neg_dot, q2x_neg, q2[0]);
                    let q2y_adj = builder.ins().select(neg_dot, q2y_neg, q2[1]);
                    let q2z_adj = builder.ins().select(neg_dot, q2z_neg, q2[2]);
                    let q2w_adj = builder.ins().select(neg_dot, q2w_neg, q2[3]);

                    // abs_dot = select(neg_dot, -dot, dot)
                    let dot_neg = builder.ins().fneg(dot);
                    let abs_dot = builder.ins().select(neg_dot, dot_neg, dot);

                    // Check if nearly parallel: use_lerp = abs_dot > 0.9995
                    let threshold = builder.ins().f32const(0.9995);
                    let use_lerp = builder.ins().fcmp(FloatCC::GreaterThan, abs_dot, threshold);

                    // theta = acos(abs_dot)
                    let acos_call = builder.ins().call(math.acos, &[abs_dot]);
                    let theta = builder.inst_results(acos_call)[0];

                    // sin_theta = sin(theta)
                    let sin_theta_call = builder.ins().call(math.sin, &[theta]);
                    let sin_theta = builder.inst_results(sin_theta_call)[0];

                    // t1 = (1 - t) * theta
                    let one = builder.ins().f32const(1.0);
                    let one_minus_t = builder.ins().fsub(one, *t);
                    let t1_theta = builder.ins().fmul(one_minus_t, theta);
                    let sin_t1_call = builder.ins().call(math.sin, &[t1_theta]);
                    let sin_t1 = builder.inst_results(sin_t1_call)[0];

                    // t2 = t * theta
                    let t2_theta = builder.ins().fmul(*t, theta);
                    let sin_t2_call = builder.ins().call(math.sin, &[t2_theta]);
                    let sin_t2 = builder.inst_results(sin_t2_call)[0];

                    // slerp_result = (q1 * sin_t1 + q2_adj * sin_t2) / sin_theta
                    let q1x_sin = builder.ins().fmul(q1[0], sin_t1);
                    let q2x_sin = builder.ins().fmul(q2x_adj, sin_t2);
                    let s_x = builder.ins().fadd(q1x_sin, q2x_sin);
                    let q1y_sin = builder.ins().fmul(q1[1], sin_t1);
                    let q2y_sin = builder.ins().fmul(q2y_adj, sin_t2);
                    let s_y = builder.ins().fadd(q1y_sin, q2y_sin);
                    let q1z_sin = builder.ins().fmul(q1[2], sin_t1);
                    let q2z_sin = builder.ins().fmul(q2z_adj, sin_t2);
                    let s_z = builder.ins().fadd(q1z_sin, q2z_sin);
                    let q1w_sin = builder.ins().fmul(q1[3], sin_t1);
                    let q2w_sin = builder.ins().fmul(q2w_adj, sin_t2);
                    let s_w = builder.ins().fadd(q1w_sin, q2w_sin);
                    let slerp_x = builder.ins().fdiv(s_x, sin_theta);
                    let slerp_y = builder.ins().fdiv(s_y, sin_theta);
                    let slerp_z = builder.ins().fdiv(s_z, sin_theta);
                    let slerp_w = builder.ins().fdiv(s_w, sin_theta);

                    // lerp_result = normalize(q1 * (1-t) + q2_adj * t)
                    let q1x_lerp = builder.ins().fmul(q1[0], one_minus_t);
                    let q2x_lerp = builder.ins().fmul(q2x_adj, *t);
                    let l_x = builder.ins().fadd(q1x_lerp, q2x_lerp);
                    let q1y_lerp = builder.ins().fmul(q1[1], one_minus_t);
                    let q2y_lerp = builder.ins().fmul(q2y_adj, *t);
                    let l_y = builder.ins().fadd(q1y_lerp, q2y_lerp);
                    let q1z_lerp = builder.ins().fmul(q1[2], one_minus_t);
                    let q2z_lerp = builder.ins().fmul(q2z_adj, *t);
                    let l_z = builder.ins().fadd(q1z_lerp, q2z_lerp);
                    let q1w_lerp = builder.ins().fmul(q1[3], one_minus_t);
                    let q2w_lerp = builder.ins().fmul(q2w_adj, *t);
                    let l_w = builder.ins().fadd(q1w_lerp, q2w_lerp);
                    // Normalize lerp result
                    let lx2 = builder.ins().fmul(l_x, l_x);
                    let ly2 = builder.ins().fmul(l_y, l_y);
                    let lz2 = builder.ins().fmul(l_z, l_z);
                    let lw2 = builder.ins().fmul(l_w, l_w);
                    let lxy = builder.ins().fadd(lx2, ly2);
                    let lzw = builder.ins().fadd(lz2, lw2);
                    let lerp_len_sq = builder.ins().fadd(lxy, lzw);
                    let lerp_len_call = builder.ins().call(math.sqrt, &[lerp_len_sq]);
                    let lerp_len = builder.inst_results(lerp_len_call)[0];
                    let lerp_x = builder.ins().fdiv(l_x, lerp_len);
                    let lerp_y = builder.ins().fdiv(l_y, lerp_len);
                    let lerp_z = builder.ins().fdiv(l_z, lerp_len);
                    let lerp_w = builder.ins().fdiv(l_w, lerp_len);

                    // Select between lerp and slerp
                    let result_x = builder.ins().select(use_lerp, lerp_x, slerp_x);
                    let result_y = builder.ins().select(use_lerp, lerp_y, slerp_y);
                    let result_z = builder.ins().select(use_lerp, lerp_z, slerp_z);
                    let result_w = builder.ins().select(use_lerp, lerp_w, slerp_w);

                    Ok(TypedValue::Quaternion([
                        result_x, result_y, result_z, result_w,
                    ]))
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

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.0001
    }

    #[test]
    fn test_scalar_add() {
        let expr = Expr::parse("a + b").unwrap();
        let jit = QuaternionJit::new().unwrap();
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
    fn test_quaternion_length() {
        let expr = Expr::parse("length(q)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_scalar(expr.ast(), &[VarSpec::new("q", Type::Quaternion)])
            .unwrap();
        // length([0, 0, 0, 1]) = 1
        assert!(approx_eq(func.call(&[0.0, 0.0, 0.0, 1.0]), 1.0));
        // length([1, 2, 2, 0]) = 3
        assert!(approx_eq(func.call(&[1.0, 2.0, 2.0, 0.0]), 3.0));
    }

    #[test]
    fn test_vec3_length() {
        let expr = Expr::parse("length(v)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_scalar(expr.ast(), &[VarSpec::new("v", Type::Vec3)])
            .unwrap();
        // length([3, 4, 0]) = 5
        assert!(approx_eq(func.call(&[3.0, 4.0, 0.0]), 5.0));
    }

    #[test]
    fn test_quaternion_dot() {
        let expr = Expr::parse("dot(a, b)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[
                    VarSpec::new("a", Type::Quaternion),
                    VarSpec::new("b", Type::Quaternion),
                ],
            )
            .unwrap();
        // dot([1,0,0,0], [1,0,0,0]) = 1
        assert!(approx_eq(
            func.call(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            1.0
        ));
    }

    #[test]
    fn test_quaternion_mul_identity() {
        // q * identity, then take length (should preserve length)
        let expr = Expr::parse("length(q * identity)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[
                    VarSpec::new("q", Type::Quaternion),
                    VarSpec::new("identity", Type::Quaternion),
                ],
            )
            .unwrap();
        // q=[1,2,2,0], identity=[0,0,0,1], |q|=3
        assert!(approx_eq(
            func.call(&[1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            3.0
        ));
    }

    #[test]
    fn test_compile_vec3_add() {
        // [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
        let expr = Expr::parse("a + b").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_vec3(
                expr.ast(),
                &[VarSpec::new("a", Type::Vec3), VarSpec::new("b", Type::Vec3)],
            )
            .unwrap();
        let [x, y, z] = func.call(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(approx_eq(x, 5.0));
        assert!(approx_eq(y, 7.0));
        assert!(approx_eq(z, 9.0));
    }

    #[test]
    fn test_compile_vec3_scalar_mul() {
        // [1, 2, 3] * 2 = [2, 4, 6]
        let expr = Expr::parse("v * 2").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_vec3(expr.ast(), &[VarSpec::new("v", Type::Vec3)])
            .unwrap();
        let [x, y, z] = func.call(&[1.0, 2.0, 3.0]);
        assert!(approx_eq(x, 2.0));
        assert!(approx_eq(y, 4.0));
        assert!(approx_eq(z, 6.0));
    }

    #[test]
    fn test_compile_quaternion_add() {
        // [1, 0, 0, 0] + [0, 1, 0, 0] = [1, 1, 0, 0]
        let expr = Expr::parse("a + b").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_quaternion(
                expr.ast(),
                &[
                    VarSpec::new("a", Type::Quaternion),
                    VarSpec::new("b", Type::Quaternion),
                ],
            )
            .unwrap();
        let [x, y, z, w] = func.call(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert!(approx_eq(x, 1.0));
        assert!(approx_eq(y, 1.0));
        assert!(approx_eq(z, 0.0));
        assert!(approx_eq(w, 0.0));
    }

    #[test]
    fn test_compile_quaternion_conj() {
        // conj([1, 2, 3, 4]) = [-1, -2, -3, 4]
        let expr = Expr::parse("conj(q)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_quaternion(expr.ast(), &[VarSpec::new("q", Type::Quaternion)])
            .unwrap();
        let [x, y, z, w] = func.call(&[1.0, 2.0, 3.0, 4.0]);
        assert!(approx_eq(x, -1.0));
        assert!(approx_eq(y, -2.0));
        assert!(approx_eq(z, -3.0));
        assert!(approx_eq(w, 4.0));
    }

    #[test]
    fn test_axis_angle() {
        // 90° rotation around Z axis
        // half angle = 45°, sin(45°) ≈ 0.7071, cos(45°) ≈ 0.7071
        let expr = Expr::parse("axis_angle(axis, angle)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_quaternion(
                expr.ast(),
                &[
                    VarSpec::new("axis", Type::Vec3),
                    VarSpec::new("angle", Type::Scalar),
                ],
            )
            .unwrap();
        let angle = std::f32::consts::FRAC_PI_2; // 90°
        let [x, y, z, w] = func.call(&[0.0, 0.0, 1.0, angle]);
        assert!(approx_eq(x, 0.0));
        assert!(approx_eq(y, 0.0));
        assert!(approx_eq(z, std::f32::consts::FRAC_PI_4.sin())); // sin(45°)
        assert!(approx_eq(w, std::f32::consts::FRAC_PI_4.cos())); // cos(45°)
    }

    #[test]
    fn test_axis_angle_identity() {
        // 0° rotation should give identity quaternion [0, 0, 0, 1]
        let expr = Expr::parse("axis_angle(axis, angle)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_quaternion(
                expr.ast(),
                &[
                    VarSpec::new("axis", Type::Vec3),
                    VarSpec::new("angle", Type::Scalar),
                ],
            )
            .unwrap();
        let [x, y, z, w] = func.call(&[1.0, 0.0, 0.0, 0.0]); // angle = 0
        assert!(approx_eq(x, 0.0));
        assert!(approx_eq(y, 0.0));
        assert!(approx_eq(z, 0.0));
        assert!(approx_eq(w, 1.0));
    }

    #[test]
    fn test_slerp_endpoints() {
        // slerp at t=0 should return q1, at t=1 should return q2
        let expr = Expr::parse("slerp(q1, q2, t)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_quaternion(
                expr.ast(),
                &[
                    VarSpec::new("q1", Type::Quaternion),
                    VarSpec::new("q2", Type::Quaternion),
                    VarSpec::new("t", Type::Scalar),
                ],
            )
            .unwrap();

        // q1 = identity [0, 0, 0, 1], q2 = 90° around Z [0, 0, sin(45°), cos(45°)]
        let sin45 = std::f32::consts::FRAC_PI_4.sin();
        let cos45 = std::f32::consts::FRAC_PI_4.cos();

        // t = 0 → q1
        let [x, y, z, w] = func.call(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, sin45, cos45, 0.0]);
        assert!(approx_eq(x, 0.0));
        assert!(approx_eq(y, 0.0));
        assert!(approx_eq(z, 0.0));
        assert!(approx_eq(w, 1.0));

        // t = 1 → q2
        let [x, y, z, w] = func.call(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, sin45, cos45, 1.0]);
        assert!(approx_eq(x, 0.0));
        assert!(approx_eq(y, 0.0));
        assert!(approx_eq(z, sin45));
        assert!(approx_eq(w, cos45));
    }

    #[test]
    fn test_slerp_midpoint() {
        // slerp between identity and 180° rotation around Z should give 90° at t=0.5
        let expr = Expr::parse("slerp(q1, q2, t)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_quaternion(
                expr.ast(),
                &[
                    VarSpec::new("q1", Type::Quaternion),
                    VarSpec::new("q2", Type::Quaternion),
                    VarSpec::new("t", Type::Scalar),
                ],
            )
            .unwrap();

        // q1 = identity [0, 0, 0, 1]
        // q2 = 180° around Z = [0, 0, 1, 0] (sin(90°)=1, cos(90°)=0)
        let [x, y, z, w] = func.call(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5]);
        // Should be 90° rotation: [0, 0, sin(45°), cos(45°)]
        assert!(approx_eq(x, 0.0));
        assert!(approx_eq(y, 0.0));
        assert!(approx_eq(z, std::f32::consts::FRAC_PI_4.sin()));
        assert!(approx_eq(w, std::f32::consts::FRAC_PI_4.cos()));
    }
}
