//! LinalgJit compiler and helper functions.

#[cfg(feature = "3d")]
use super::compiled::CompiledMat3Fn;
use super::compiled::{CompiledLinalgFn, CompiledMat2Fn, CompiledVec2Fn, CompiledVec3Fn};
#[cfg(feature = "4d")]
use super::compiled::{CompiledMat4Fn, CompiledVec4Fn};
use super::error::CraneliftError;
use super::types::{MathFuncs, TypedValue, VarSpec, math_symbols};
use crate::Type;
use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value as CraneliftValue, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

// ============================================================================
// JIT Compiler
// ============================================================================

/// JIT compiler for linalg expressions.
pub struct LinalgJit {
    builder: JITBuilder,
}

impl LinalgJit {
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
        let sin_id = module
            .declare_function("linalg_sin", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("linalg_cos", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

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

            let sqrt_ref = module.declare_func_in_func(sqrt_id, builder.func);
            let pow_ref = module.declare_func_in_func(pow_id, builder.func);
            let sin_ref = module.declare_func_in_func(sin_id, builder.func);
            let cos_ref = module.declare_func_in_func(cos_id, builder.func);

            let block_params = builder.block_params(entry_block).to_vec();
            let var_map = build_var_map(vars, &block_params);

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
                sin: sin_ref,
                cos: cos_ref,
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
        let sin_id = module
            .declare_function("linalg_sin", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("linalg_cos", Linkage::Import, &sqrt_sig)
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
            let sin_ref = module.declare_func_in_func(sin_id, builder.func);
            let cos_ref = module.declare_func_in_func(cos_id, builder.func);

            let block_params = builder.block_params(entry_block).to_vec();
            let out_ptr = block_params[total_params];
            let var_map = build_var_map(vars, &block_params);

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
                sin: sin_ref,
                cos: cos_ref,
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
        let sin_id = module
            .declare_function("linalg_sin", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("linalg_cos", Linkage::Import, &sqrt_sig)
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
            let sin_ref = module.declare_func_in_func(sin_id, builder.func);
            let cos_ref = module.declare_func_in_func(cos_id, builder.func);

            let block_params = builder.block_params(entry_block).to_vec();
            let out_ptr = block_params[total_params];
            let var_map = build_var_map(vars, &block_params);

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
                sin: sin_ref,
                cos: cos_ref,
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

    /// Compiles an expression that returns a Vec4.
    #[cfg(feature = "4d")]
    pub fn compile_vec4(
        self,
        ast: &Ast,
        vars: &[VarSpec],
    ) -> Result<CompiledVec4Fn, CraneliftError> {
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
        let sin_id = module
            .declare_function("linalg_sin", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("linalg_cos", Linkage::Import, &sqrt_sig)
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
            let sin_ref = module.declare_func_in_func(sin_id, builder.func);
            let cos_ref = module.declare_func_in_func(cos_id, builder.func);

            let block_params = builder.block_params(entry_block).to_vec();
            let out_ptr = block_params[total_params];
            let var_map = build_var_map(vars, &block_params);

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
                sin: sin_ref,
                cos: cos_ref,
            };
            let result = compile_ast(ast, &mut builder, &var_map, &math_funcs)?;

            let v = result
                .as_vec4()
                .ok_or(CraneliftError::UnsupportedReturnType(result.typ()))?;

            for (i, val) in v.iter().enumerate() {
                builder.ins().store(
                    MemFlags::new(),
                    *val,
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

        Ok(CompiledVec4Fn {
            _module: module,
            func_ptr,
            param_count: total_params,
        })
    }

    /// Compiles an expression that returns a Mat2.
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
        let sin_id = module
            .declare_function("linalg_sin", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("linalg_cos", Linkage::Import, &sqrt_sig)
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
            let sin_ref = module.declare_func_in_func(sin_id, builder.func);
            let cos_ref = module.declare_func_in_func(cos_id, builder.func);

            let block_params = builder.block_params(entry_block).to_vec();
            let out_ptr = block_params[total_params];
            let var_map = build_var_map(vars, &block_params);

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
                sin: sin_ref,
                cos: cos_ref,
            };
            let result = compile_ast(ast, &mut builder, &var_map, &math_funcs)?;

            let m = result
                .as_mat2()
                .ok_or(CraneliftError::UnsupportedReturnType(result.typ()))?;

            for (i, val) in m.iter().enumerate() {
                builder.ins().store(
                    MemFlags::new(),
                    *val,
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
        let sin_id = module
            .declare_function("linalg_sin", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("linalg_cos", Linkage::Import, &sqrt_sig)
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
            let sin_ref = module.declare_func_in_func(sin_id, builder.func);
            let cos_ref = module.declare_func_in_func(cos_id, builder.func);

            let block_params = builder.block_params(entry_block).to_vec();
            let out_ptr = block_params[total_params];
            let var_map = build_var_map(vars, &block_params);

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
                sin: sin_ref,
                cos: cos_ref,
            };
            let result = compile_ast(ast, &mut builder, &var_map, &math_funcs)?;

            let m = result
                .as_mat3()
                .ok_or(CraneliftError::UnsupportedReturnType(result.typ()))?;

            for (i, val) in m.iter().enumerate() {
                builder.ins().store(
                    MemFlags::new(),
                    *val,
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

    /// Compiles an expression that returns a Mat4.
    #[cfg(feature = "4d")]
    pub fn compile_mat4(
        self,
        ast: &Ast,
        vars: &[VarSpec],
    ) -> Result<CompiledMat4Fn, CraneliftError> {
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
        let sin_id = module
            .declare_function("linalg_sin", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("linalg_cos", Linkage::Import, &sqrt_sig)
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
            let sin_ref = module.declare_func_in_func(sin_id, builder.func);
            let cos_ref = module.declare_func_in_func(cos_id, builder.func);

            let block_params = builder.block_params(entry_block).to_vec();
            let out_ptr = block_params[total_params];
            let var_map = build_var_map(vars, &block_params);

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
                sin: sin_ref,
                cos: cos_ref,
            };
            let result = compile_ast(ast, &mut builder, &var_map, &math_funcs)?;

            let m = result
                .as_mat4()
                .ok_or(CraneliftError::UnsupportedReturnType(result.typ()))?;

            for (i, val) in m.iter().enumerate() {
                builder.ins().store(
                    MemFlags::new(),
                    *val,
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

        Ok(CompiledMat4Fn {
            _module: module,
            func_ptr,
            param_count: total_params,
        })
    }
}

// ============================================================================
// Helper: build_var_map
// ============================================================================

fn build_var_map(vars: &[VarSpec], block_params: &[CraneliftValue]) -> HashMap<String, TypedValue> {
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
                let v = TypedValue::Vec2([block_params[param_idx], block_params[param_idx + 1]]);
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
            #[cfg(feature = "4d")]
            Type::Mat4 => {
                let v = TypedValue::Mat4([
                    block_params[param_idx],
                    block_params[param_idx + 1],
                    block_params[param_idx + 2],
                    block_params[param_idx + 3],
                    block_params[param_idx + 4],
                    block_params[param_idx + 5],
                    block_params[param_idx + 6],
                    block_params[param_idx + 7],
                    block_params[param_idx + 8],
                    block_params[param_idx + 9],
                    block_params[param_idx + 10],
                    block_params[param_idx + 11],
                    block_params[param_idx + 12],
                    block_params[param_idx + 13],
                    block_params[param_idx + 14],
                    block_params[param_idx + 15],
                ]);
                param_idx += 16;
                v
            }
        };
        var_map.insert(var.name.clone(), typed_val);
    }

    var_map
}

// ============================================================================
// AST Compilation
// ============================================================================

fn compile_ast(
    ast: &Ast,
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, TypedValue>,
    math: &MathFuncs,
) -> Result<TypedValue, CraneliftError> {
    match ast {
        Ast::Num(n) => Ok(TypedValue::Scalar(builder.ins().f32const(*n as f32))),

        Ast::Var(name) => vars
            .get(name)
            .cloned()
            .ok_or_else(|| CraneliftError::UnknownVariable(name.clone())),

        Ast::BinOp(op, left, right) => {
            let left_val = compile_ast(left, builder, vars, math)?;
            let right_val = compile_ast(right, builder, vars, math)?;
            compile_binop(*op, &left_val, &right_val, builder, math)
        }

        Ast::UnaryOp(op, inner) => {
            let val = compile_ast(inner, builder, vars, math)?;
            compile_unaryop(*op, val, builder)
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

        Ast::Let { name, value, body } => {
            // Compile the value expression
            let value_val = compile_ast(value, builder, vars, math)?;
            // Extend vars with the new binding
            let mut new_vars = vars.clone();
            new_vars.insert(name.clone(), value_val);
            // Compile the body with extended environment
            compile_ast(body, builder, &new_vars, math)
        }
    }
}

fn compile_binop(
    op: BinOp,
    left: &TypedValue,
    right: &TypedValue,
    builder: &mut FunctionBuilder,
    math: &MathFuncs,
) -> Result<TypedValue, CraneliftError> {
    match (op, left, right) {
        // Scalar ops
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

        // Vec2 ops
        (BinOp::Add, TypedValue::Vec2(l), TypedValue::Vec2(r)) => Ok(TypedValue::Vec2([
            builder.ins().fadd(l[0], r[0]),
            builder.ins().fadd(l[1], r[1]),
        ])),
        (BinOp::Sub, TypedValue::Vec2(l), TypedValue::Vec2(r)) => Ok(TypedValue::Vec2([
            builder.ins().fsub(l[0], r[0]),
            builder.ins().fsub(l[1], r[1]),
        ])),
        (BinOp::Mul, TypedValue::Vec2(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec2([
            builder.ins().fmul(v[0], *s),
            builder.ins().fmul(v[1], *s),
        ])),
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Vec2(v)) => Ok(TypedValue::Vec2([
            builder.ins().fmul(*s, v[0]),
            builder.ins().fmul(*s, v[1]),
        ])),
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
        (BinOp::Mul, TypedValue::Mat2(a), TypedValue::Mat2(b)) => {
            let r0_t1 = builder.ins().fmul(a[0], b[0]);
            let r0_t2 = builder.ins().fmul(a[2], b[1]);
            let r0 = builder.ins().fadd(r0_t1, r0_t2);
            let r1_t1 = builder.ins().fmul(a[1], b[0]);
            let r1_t2 = builder.ins().fmul(a[3], b[1]);
            let r1 = builder.ins().fadd(r1_t1, r1_t2);
            let r2_t1 = builder.ins().fmul(a[0], b[2]);
            let r2_t2 = builder.ins().fmul(a[2], b[3]);
            let r2 = builder.ins().fadd(r2_t1, r2_t2);
            let r3_t1 = builder.ins().fmul(a[1], b[2]);
            let r3_t2 = builder.ins().fmul(a[3], b[3]);
            let r3 = builder.ins().fadd(r3_t1, r3_t2);
            Ok(TypedValue::Mat2([r0, r1, r2, r3]))
        }

        // Mat2 * Vec2 (column vector, column-major storage)
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
        #[cfg(feature = "3d")]
        (BinOp::Mul, TypedValue::Mat3(a), TypedValue::Mat3(b)) => {
            let mut result = [a[0]; 9];
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

        // Mat4 + Mat4, Mat4 - Mat4
        #[cfg(feature = "4d")]
        (BinOp::Add, TypedValue::Mat4(a), TypedValue::Mat4(b)) => Ok(TypedValue::Mat4([
            builder.ins().fadd(a[0], b[0]),
            builder.ins().fadd(a[1], b[1]),
            builder.ins().fadd(a[2], b[2]),
            builder.ins().fadd(a[3], b[3]),
            builder.ins().fadd(a[4], b[4]),
            builder.ins().fadd(a[5], b[5]),
            builder.ins().fadd(a[6], b[6]),
            builder.ins().fadd(a[7], b[7]),
            builder.ins().fadd(a[8], b[8]),
            builder.ins().fadd(a[9], b[9]),
            builder.ins().fadd(a[10], b[10]),
            builder.ins().fadd(a[11], b[11]),
            builder.ins().fadd(a[12], b[12]),
            builder.ins().fadd(a[13], b[13]),
            builder.ins().fadd(a[14], b[14]),
            builder.ins().fadd(a[15], b[15]),
        ])),
        #[cfg(feature = "4d")]
        (BinOp::Sub, TypedValue::Mat4(a), TypedValue::Mat4(b)) => Ok(TypedValue::Mat4([
            builder.ins().fsub(a[0], b[0]),
            builder.ins().fsub(a[1], b[1]),
            builder.ins().fsub(a[2], b[2]),
            builder.ins().fsub(a[3], b[3]),
            builder.ins().fsub(a[4], b[4]),
            builder.ins().fsub(a[5], b[5]),
            builder.ins().fsub(a[6], b[6]),
            builder.ins().fsub(a[7], b[7]),
            builder.ins().fsub(a[8], b[8]),
            builder.ins().fsub(a[9], b[9]),
            builder.ins().fsub(a[10], b[10]),
            builder.ins().fsub(a[11], b[11]),
            builder.ins().fsub(a[12], b[12]),
            builder.ins().fsub(a[13], b[13]),
            builder.ins().fsub(a[14], b[14]),
            builder.ins().fsub(a[15], b[15]),
        ])),

        // Mat4 * Scalar, Scalar * Mat4
        #[cfg(feature = "4d")]
        (BinOp::Mul, TypedValue::Mat4(m), TypedValue::Scalar(s)) => Ok(TypedValue::Mat4([
            builder.ins().fmul(m[0], *s),
            builder.ins().fmul(m[1], *s),
            builder.ins().fmul(m[2], *s),
            builder.ins().fmul(m[3], *s),
            builder.ins().fmul(m[4], *s),
            builder.ins().fmul(m[5], *s),
            builder.ins().fmul(m[6], *s),
            builder.ins().fmul(m[7], *s),
            builder.ins().fmul(m[8], *s),
            builder.ins().fmul(m[9], *s),
            builder.ins().fmul(m[10], *s),
            builder.ins().fmul(m[11], *s),
            builder.ins().fmul(m[12], *s),
            builder.ins().fmul(m[13], *s),
            builder.ins().fmul(m[14], *s),
            builder.ins().fmul(m[15], *s),
        ])),
        #[cfg(feature = "4d")]
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Mat4(m)) => Ok(TypedValue::Mat4([
            builder.ins().fmul(*s, m[0]),
            builder.ins().fmul(*s, m[1]),
            builder.ins().fmul(*s, m[2]),
            builder.ins().fmul(*s, m[3]),
            builder.ins().fmul(*s, m[4]),
            builder.ins().fmul(*s, m[5]),
            builder.ins().fmul(*s, m[6]),
            builder.ins().fmul(*s, m[7]),
            builder.ins().fmul(*s, m[8]),
            builder.ins().fmul(*s, m[9]),
            builder.ins().fmul(*s, m[10]),
            builder.ins().fmul(*s, m[11]),
            builder.ins().fmul(*s, m[12]),
            builder.ins().fmul(*s, m[13]),
            builder.ins().fmul(*s, m[14]),
            builder.ins().fmul(*s, m[15]),
        ])),

        // Mat4 * Mat4 (column-major)
        #[cfg(feature = "4d")]
        (BinOp::Mul, TypedValue::Mat4(a), TypedValue::Mat4(b)) => {
            let mut result = [a[0]; 16];
            for col in 0..4 {
                for row in 0..4 {
                    let t1 = builder.ins().fmul(a[row], b[col * 4]);
                    let t2 = builder.ins().fmul(a[4 + row], b[col * 4 + 1]);
                    let t3 = builder.ins().fmul(a[8 + row], b[col * 4 + 2]);
                    let t4 = builder.ins().fmul(a[12 + row], b[col * 4 + 3]);
                    let sum12 = builder.ins().fadd(t1, t2);
                    let sum34 = builder.ins().fadd(t3, t4);
                    result[col * 4 + row] = builder.ins().fadd(sum12, sum34);
                }
            }
            Ok(TypedValue::Mat4(result))
        }

        // Mat4 * Vec4 (column vector, column-major storage)
        #[cfg(feature = "4d")]
        (BinOp::Mul, TypedValue::Mat4(m), TypedValue::Vec4(v)) => {
            let x_t1 = builder.ins().fmul(m[0], v[0]);
            let x_t2 = builder.ins().fmul(m[4], v[1]);
            let x_t3 = builder.ins().fmul(m[8], v[2]);
            let x_t4 = builder.ins().fmul(m[12], v[3]);
            let x_12 = builder.ins().fadd(x_t1, x_t2);
            let x_34 = builder.ins().fadd(x_t3, x_t4);
            let x = builder.ins().fadd(x_12, x_34);

            let y_t1 = builder.ins().fmul(m[1], v[0]);
            let y_t2 = builder.ins().fmul(m[5], v[1]);
            let y_t3 = builder.ins().fmul(m[9], v[2]);
            let y_t4 = builder.ins().fmul(m[13], v[3]);
            let y_12 = builder.ins().fadd(y_t1, y_t2);
            let y_34 = builder.ins().fadd(y_t3, y_t4);
            let y = builder.ins().fadd(y_12, y_34);

            let z_t1 = builder.ins().fmul(m[2], v[0]);
            let z_t2 = builder.ins().fmul(m[6], v[1]);
            let z_t3 = builder.ins().fmul(m[10], v[2]);
            let z_t4 = builder.ins().fmul(m[14], v[3]);
            let z_12 = builder.ins().fadd(z_t1, z_t2);
            let z_34 = builder.ins().fadd(z_t3, z_t4);
            let z = builder.ins().fadd(z_12, z_34);

            let w_t1 = builder.ins().fmul(m[3], v[0]);
            let w_t2 = builder.ins().fmul(m[7], v[1]);
            let w_t3 = builder.ins().fmul(m[11], v[2]);
            let w_t4 = builder.ins().fmul(m[15], v[3]);
            let w_12 = builder.ins().fadd(w_t1, w_t2);
            let w_34 = builder.ins().fadd(w_t3, w_t4);
            let w = builder.ins().fadd(w_12, w_34);

            Ok(TypedValue::Vec4([x, y, z, w]))
        }

        // Vec4 * Mat4 (row vector, column-major storage)
        #[cfg(feature = "4d")]
        (BinOp::Mul, TypedValue::Vec4(v), TypedValue::Mat4(m)) => {
            let x_t1 = builder.ins().fmul(v[0], m[0]);
            let x_t2 = builder.ins().fmul(v[1], m[1]);
            let x_t3 = builder.ins().fmul(v[2], m[2]);
            let x_t4 = builder.ins().fmul(v[3], m[3]);
            let x_12 = builder.ins().fadd(x_t1, x_t2);
            let x_34 = builder.ins().fadd(x_t3, x_t4);
            let x = builder.ins().fadd(x_12, x_34);

            let y_t1 = builder.ins().fmul(v[0], m[4]);
            let y_t2 = builder.ins().fmul(v[1], m[5]);
            let y_t3 = builder.ins().fmul(v[2], m[6]);
            let y_t4 = builder.ins().fmul(v[3], m[7]);
            let y_12 = builder.ins().fadd(y_t1, y_t2);
            let y_34 = builder.ins().fadd(y_t3, y_t4);
            let y = builder.ins().fadd(y_12, y_34);

            let z_t1 = builder.ins().fmul(v[0], m[8]);
            let z_t2 = builder.ins().fmul(v[1], m[9]);
            let z_t3 = builder.ins().fmul(v[2], m[10]);
            let z_t4 = builder.ins().fmul(v[3], m[11]);
            let z_12 = builder.ins().fadd(z_t1, z_t2);
            let z_34 = builder.ins().fadd(z_t3, z_t4);
            let z = builder.ins().fadd(z_12, z_34);

            let w_t1 = builder.ins().fmul(v[0], m[12]);
            let w_t2 = builder.ins().fmul(v[1], m[13]);
            let w_t3 = builder.ins().fmul(v[2], m[14]);
            let w_t4 = builder.ins().fmul(v[3], m[15]);
            let w_12 = builder.ins().fadd(w_t1, w_t2);
            let w_34 = builder.ins().fadd(w_t3, w_t4);
            let w = builder.ins().fadd(w_12, w_34);

            Ok(TypedValue::Vec4([x, y, z, w]))
        }

        _ => Err(CraneliftError::TypeMismatch {
            op: match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Pow => "^",
                BinOp::Rem => "%",
                BinOp::BitAnd => "&",
                BinOp::BitOr => "|",
                BinOp::Shl => "<<",
                BinOp::Shr => ">>",
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
            #[cfg(feature = "4d")]
            TypedValue::Mat4(m) => Ok(TypedValue::Mat4([
                builder.ins().fneg(m[0]),
                builder.ins().fneg(m[1]),
                builder.ins().fneg(m[2]),
                builder.ins().fneg(m[3]),
                builder.ins().fneg(m[4]),
                builder.ins().fneg(m[5]),
                builder.ins().fneg(m[6]),
                builder.ins().fneg(m[7]),
                builder.ins().fneg(m[8]),
                builder.ins().fneg(m[9]),
                builder.ins().fneg(m[10]),
                builder.ins().fneg(m[11]),
                builder.ins().fneg(m[12]),
                builder.ins().fneg(m[13]),
                builder.ins().fneg(m[14]),
                builder.ins().fneg(m[15]),
            ])),
        },
        UnaryOp::Not => Err(CraneliftError::UnsupportedConditional("Not")),
        UnaryOp::BitNot => Err(CraneliftError::UnsupportedConditional("BitNot")),
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

        #[cfg(feature = "3d")]
        "cross" => {
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1]) {
                (TypedValue::Vec3(a), TypedValue::Vec3(b)) => {
                    let ay_bz = builder.ins().fmul(a[1], b[2]);
                    let az_by = builder.ins().fmul(a[2], b[1]);
                    let x = builder.ins().fsub(ay_bz, az_by);
                    let az_bx = builder.ins().fmul(a[2], b[0]);
                    let ax_bz = builder.ins().fmul(a[0], b[2]);
                    let y = builder.ins().fsub(az_bx, ax_bz);
                    let ax_by = builder.ins().fmul(a[0], b[1]);
                    let ay_bx = builder.ins().fmul(a[1], b[0]);
                    let z = builder.ins().fsub(ax_by, ay_bx);
                    Ok(TypedValue::Vec3([x, y, z]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "normalize" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Vec2(v) => {
                    let x2 = builder.ins().fmul(v[0], v[0]);
                    let y2 = builder.ins().fmul(v[1], v[1]);
                    let sum = builder.ins().fadd(x2, y2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    let len = builder.inst_results(call)[0];
                    Ok(TypedValue::Vec2([
                        builder.ins().fdiv(v[0], len),
                        builder.ins().fdiv(v[1], len),
                    ]))
                }
                #[cfg(feature = "3d")]
                TypedValue::Vec3(v) => {
                    let x2 = builder.ins().fmul(v[0], v[0]);
                    let y2 = builder.ins().fmul(v[1], v[1]);
                    let z2 = builder.ins().fmul(v[2], v[2]);
                    let xy = builder.ins().fadd(x2, y2);
                    let sum = builder.ins().fadd(xy, z2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    let len = builder.inst_results(call)[0];
                    Ok(TypedValue::Vec3([
                        builder.ins().fdiv(v[0], len),
                        builder.ins().fdiv(v[1], len),
                        builder.ins().fdiv(v[2], len),
                    ]))
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
                    let len = builder.inst_results(call)[0];
                    Ok(TypedValue::Vec4([
                        builder.ins().fdiv(v[0], len),
                        builder.ins().fdiv(v[1], len),
                        builder.ins().fdiv(v[2], len),
                        builder.ins().fdiv(v[3], len),
                    ]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "reflect" => {
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1]) {
                (TypedValue::Vec2(i), TypedValue::Vec2(n)) => {
                    let nx_ix = builder.ins().fmul(n[0], i[0]);
                    let ny_iy = builder.ins().fmul(n[1], i[1]);
                    let dot = builder.ins().fadd(nx_ix, ny_iy);
                    let two = builder.ins().f32const(2.0);
                    let factor = builder.ins().fmul(two, dot);
                    let fn0 = builder.ins().fmul(factor, n[0]);
                    let fn1 = builder.ins().fmul(factor, n[1]);
                    let rx = builder.ins().fsub(i[0], fn0);
                    let ry = builder.ins().fsub(i[1], fn1);
                    Ok(TypedValue::Vec2([rx, ry]))
                }
                #[cfg(feature = "3d")]
                (TypedValue::Vec3(i), TypedValue::Vec3(n)) => {
                    let nx_ix = builder.ins().fmul(n[0], i[0]);
                    let ny_iy = builder.ins().fmul(n[1], i[1]);
                    let nz_iz = builder.ins().fmul(n[2], i[2]);
                    let dot_xy = builder.ins().fadd(nx_ix, ny_iy);
                    let dot = builder.ins().fadd(dot_xy, nz_iz);
                    let two = builder.ins().f32const(2.0);
                    let factor = builder.ins().fmul(two, dot);
                    let fn0 = builder.ins().fmul(factor, n[0]);
                    let fn1 = builder.ins().fmul(factor, n[1]);
                    let fn2 = builder.ins().fmul(factor, n[2]);
                    let rx = builder.ins().fsub(i[0], fn0);
                    let ry = builder.ins().fsub(i[1], fn1);
                    let rz = builder.ins().fsub(i[2], fn2);
                    Ok(TypedValue::Vec3([rx, ry, rz]))
                }
                #[cfg(feature = "4d")]
                (TypedValue::Vec4(i), TypedValue::Vec4(n)) => {
                    let nx_ix = builder.ins().fmul(n[0], i[0]);
                    let ny_iy = builder.ins().fmul(n[1], i[1]);
                    let nz_iz = builder.ins().fmul(n[2], i[2]);
                    let nw_iw = builder.ins().fmul(n[3], i[3]);
                    let dot_xy = builder.ins().fadd(nx_ix, ny_iy);
                    let dot_zw = builder.ins().fadd(nz_iz, nw_iw);
                    let dot = builder.ins().fadd(dot_xy, dot_zw);
                    let two = builder.ins().f32const(2.0);
                    let factor = builder.ins().fmul(two, dot);
                    let fn0 = builder.ins().fmul(factor, n[0]);
                    let fn1 = builder.ins().fmul(factor, n[1]);
                    let fn2 = builder.ins().fmul(factor, n[2]);
                    let fn3 = builder.ins().fmul(factor, n[3]);
                    let rx = builder.ins().fsub(i[0], fn0);
                    let ry = builder.ins().fsub(i[1], fn1);
                    let rz = builder.ins().fsub(i[2], fn2);
                    let rw = builder.ins().fsub(i[3], fn3);
                    Ok(TypedValue::Vec4([rx, ry, rz, rw]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "hadamard" => {
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1]) {
                (TypedValue::Vec2(a), TypedValue::Vec2(b)) => Ok(TypedValue::Vec2([
                    builder.ins().fmul(a[0], b[0]),
                    builder.ins().fmul(a[1], b[1]),
                ])),
                #[cfg(feature = "3d")]
                (TypedValue::Vec3(a), TypedValue::Vec3(b)) => Ok(TypedValue::Vec3([
                    builder.ins().fmul(a[0], b[0]),
                    builder.ins().fmul(a[1], b[1]),
                    builder.ins().fmul(a[2], b[2]),
                ])),
                #[cfg(feature = "4d")]
                (TypedValue::Vec4(a), TypedValue::Vec4(b)) => Ok(TypedValue::Vec4([
                    builder.ins().fmul(a[0], b[0]),
                    builder.ins().fmul(a[1], b[1]),
                    builder.ins().fmul(a[2], b[2]),
                    builder.ins().fmul(a[3], b[3]),
                ])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "lerp" | "mix" => {
            if args.len() != 3 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1], &args[2]) {
                (TypedValue::Vec2(a), TypedValue::Vec2(b), TypedValue::Scalar(t)) => {
                    let dx = builder.ins().fsub(b[0], a[0]);
                    let dy = builder.ins().fsub(b[1], a[1]);
                    let dx_t = builder.ins().fmul(dx, *t);
                    let dy_t = builder.ins().fmul(dy, *t);
                    let rx = builder.ins().fadd(a[0], dx_t);
                    let ry = builder.ins().fadd(a[1], dy_t);
                    Ok(TypedValue::Vec2([rx, ry]))
                }
                #[cfg(feature = "3d")]
                (TypedValue::Vec3(a), TypedValue::Vec3(b), TypedValue::Scalar(t)) => {
                    let dx = builder.ins().fsub(b[0], a[0]);
                    let dy = builder.ins().fsub(b[1], a[1]);
                    let dz = builder.ins().fsub(b[2], a[2]);
                    let dx_t = builder.ins().fmul(dx, *t);
                    let dy_t = builder.ins().fmul(dy, *t);
                    let dz_t = builder.ins().fmul(dz, *t);
                    let rx = builder.ins().fadd(a[0], dx_t);
                    let ry = builder.ins().fadd(a[1], dy_t);
                    let rz = builder.ins().fadd(a[2], dz_t);
                    Ok(TypedValue::Vec3([rx, ry, rz]))
                }
                #[cfg(feature = "4d")]
                (TypedValue::Vec4(a), TypedValue::Vec4(b), TypedValue::Scalar(t)) => {
                    let dx = builder.ins().fsub(b[0], a[0]);
                    let dy = builder.ins().fsub(b[1], a[1]);
                    let dz = builder.ins().fsub(b[2], a[2]);
                    let dw = builder.ins().fsub(b[3], a[3]);
                    let dx_t = builder.ins().fmul(dx, *t);
                    let dy_t = builder.ins().fmul(dy, *t);
                    let dz_t = builder.ins().fmul(dz, *t);
                    let dw_t = builder.ins().fmul(dw, *t);
                    let rx = builder.ins().fadd(a[0], dx_t);
                    let ry = builder.ins().fadd(a[1], dy_t);
                    let rz = builder.ins().fadd(a[2], dz_t);
                    let rw = builder.ins().fadd(a[3], dw_t);
                    Ok(TypedValue::Vec4([rx, ry, rz, rw]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "vec2" => {
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1]) {
                (TypedValue::Scalar(x), TypedValue::Scalar(y)) => Ok(TypedValue::Vec2([*x, *y])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        #[cfg(feature = "3d")]
        "vec3" => {
            if args.len() != 3 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1], &args[2]) {
                (TypedValue::Scalar(x), TypedValue::Scalar(y), TypedValue::Scalar(z)) => {
                    Ok(TypedValue::Vec3([*x, *y, *z]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        #[cfg(feature = "4d")]
        "vec4" => {
            if args.len() != 4 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1], &args[2], &args[3]) {
                (
                    TypedValue::Scalar(x),
                    TypedValue::Scalar(y),
                    TypedValue::Scalar(z),
                    TypedValue::Scalar(w),
                ) => Ok(TypedValue::Vec4([*x, *y, *z, *w])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "x" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Vec2(v) => Ok(TypedValue::Scalar(v[0])),
                #[cfg(feature = "3d")]
                TypedValue::Vec3(v) => Ok(TypedValue::Scalar(v[0])),
                #[cfg(feature = "4d")]
                TypedValue::Vec4(v) => Ok(TypedValue::Scalar(v[0])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "y" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Vec2(v) => Ok(TypedValue::Scalar(v[1])),
                #[cfg(feature = "3d")]
                TypedValue::Vec3(v) => Ok(TypedValue::Scalar(v[1])),
                #[cfg(feature = "4d")]
                TypedValue::Vec4(v) => Ok(TypedValue::Scalar(v[1])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        #[cfg(feature = "3d")]
        "z" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Vec3(v) => Ok(TypedValue::Scalar(v[2])),
                #[cfg(feature = "4d")]
                TypedValue::Vec4(v) => Ok(TypedValue::Scalar(v[2])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        #[cfg(feature = "4d")]
        "w" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Vec4(v) => Ok(TypedValue::Scalar(v[3])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "sin" => compile_vectorized_math(name, &args, builder, math.sin),
        "cos" => compile_vectorized_math(name, &args, builder, math.cos),
        "sqrt" => compile_vectorized_math(name, &args, builder, math.sqrt),

        "abs" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Vec2(v) => Ok(TypedValue::Vec2([
                    builder.ins().fabs(v[0]),
                    builder.ins().fabs(v[1]),
                ])),
                #[cfg(feature = "3d")]
                TypedValue::Vec3(v) => Ok(TypedValue::Vec3([
                    builder.ins().fabs(v[0]),
                    builder.ins().fabs(v[1]),
                    builder.ins().fabs(v[2]),
                ])),
                #[cfg(feature = "4d")]
                TypedValue::Vec4(v) => Ok(TypedValue::Vec4([
                    builder.ins().fabs(v[0]),
                    builder.ins().fabs(v[1]),
                    builder.ins().fabs(v[2]),
                    builder.ins().fabs(v[3]),
                ])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "floor" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Vec2(v) => Ok(TypedValue::Vec2([
                    builder.ins().floor(v[0]),
                    builder.ins().floor(v[1]),
                ])),
                #[cfg(feature = "3d")]
                TypedValue::Vec3(v) => Ok(TypedValue::Vec3([
                    builder.ins().floor(v[0]),
                    builder.ins().floor(v[1]),
                    builder.ins().floor(v[2]),
                ])),
                #[cfg(feature = "4d")]
                TypedValue::Vec4(v) => Ok(TypedValue::Vec4([
                    builder.ins().floor(v[0]),
                    builder.ins().floor(v[1]),
                    builder.ins().floor(v[2]),
                    builder.ins().floor(v[3]),
                ])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "fract" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Vec2(v) => {
                    let f0 = builder.ins().floor(v[0]);
                    let f1 = builder.ins().floor(v[1]);
                    Ok(TypedValue::Vec2([
                        builder.ins().fsub(v[0], f0),
                        builder.ins().fsub(v[1], f1),
                    ]))
                }
                #[cfg(feature = "3d")]
                TypedValue::Vec3(v) => {
                    let f0 = builder.ins().floor(v[0]);
                    let f1 = builder.ins().floor(v[1]);
                    let f2 = builder.ins().floor(v[2]);
                    Ok(TypedValue::Vec3([
                        builder.ins().fsub(v[0], f0),
                        builder.ins().fsub(v[1], f1),
                        builder.ins().fsub(v[2], f2),
                    ]))
                }
                #[cfg(feature = "4d")]
                TypedValue::Vec4(v) => {
                    let f0 = builder.ins().floor(v[0]);
                    let f1 = builder.ins().floor(v[1]);
                    let f2 = builder.ins().floor(v[2]);
                    let f3 = builder.ins().floor(v[3]);
                    Ok(TypedValue::Vec4([
                        builder.ins().fsub(v[0], f0),
                        builder.ins().fsub(v[1], f1),
                        builder.ins().fsub(v[2], f2),
                        builder.ins().fsub(v[3], f3),
                    ]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "min" => {
            compile_vectorized_binary_builtin(name, &args, builder, |b, a, c| b.ins().fmin(a, c))
        }
        "max" => {
            compile_vectorized_binary_builtin(name, &args, builder, |b, a, c| b.ins().fmax(a, c))
        }

        "clamp" => {
            if args.len() != 3 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1], &args[2]) {
                (TypedValue::Vec2(x), TypedValue::Vec2(lo), TypedValue::Vec2(hi)) => {
                    let m0 = builder.ins().fmax(x[0], lo[0]);
                    let m1 = builder.ins().fmax(x[1], lo[1]);
                    Ok(TypedValue::Vec2([
                        builder.ins().fmin(m0, hi[0]),
                        builder.ins().fmin(m1, hi[1]),
                    ]))
                }
                #[cfg(feature = "3d")]
                (TypedValue::Vec3(x), TypedValue::Vec3(lo), TypedValue::Vec3(hi)) => {
                    let m0 = builder.ins().fmax(x[0], lo[0]);
                    let m1 = builder.ins().fmax(x[1], lo[1]);
                    let m2 = builder.ins().fmax(x[2], lo[2]);
                    Ok(TypedValue::Vec3([
                        builder.ins().fmin(m0, hi[0]),
                        builder.ins().fmin(m1, hi[1]),
                        builder.ins().fmin(m2, hi[2]),
                    ]))
                }
                #[cfg(feature = "4d")]
                (TypedValue::Vec4(x), TypedValue::Vec4(lo), TypedValue::Vec4(hi)) => {
                    let m0 = builder.ins().fmax(x[0], lo[0]);
                    let m1 = builder.ins().fmax(x[1], lo[1]);
                    let m2 = builder.ins().fmax(x[2], lo[2]);
                    let m3 = builder.ins().fmax(x[3], lo[3]);
                    Ok(TypedValue::Vec4([
                        builder.ins().fmin(m0, hi[0]),
                        builder.ins().fmin(m1, hi[1]),
                        builder.ins().fmin(m2, hi[2]),
                        builder.ins().fmin(m3, hi[3]),
                    ]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "step" => {
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            let zero = builder.ins().f32const(0.0);
            let one = builder.ins().f32const(1.0);
            match (&args[0], &args[1]) {
                (TypedValue::Vec2(edge), TypedValue::Vec2(x)) => {
                    let c0 = builder.ins().fcmp(FloatCC::LessThan, x[0], edge[0]);
                    let c1 = builder.ins().fcmp(FloatCC::LessThan, x[1], edge[1]);
                    Ok(TypedValue::Vec2([
                        builder.ins().select(c0, zero, one),
                        builder.ins().select(c1, zero, one),
                    ]))
                }
                #[cfg(feature = "3d")]
                (TypedValue::Vec3(edge), TypedValue::Vec3(x)) => {
                    let c0 = builder.ins().fcmp(FloatCC::LessThan, x[0], edge[0]);
                    let c1 = builder.ins().fcmp(FloatCC::LessThan, x[1], edge[1]);
                    let c2 = builder.ins().fcmp(FloatCC::LessThan, x[2], edge[2]);
                    Ok(TypedValue::Vec3([
                        builder.ins().select(c0, zero, one),
                        builder.ins().select(c1, zero, one),
                        builder.ins().select(c2, zero, one),
                    ]))
                }
                #[cfg(feature = "4d")]
                (TypedValue::Vec4(edge), TypedValue::Vec4(x)) => {
                    let c0 = builder.ins().fcmp(FloatCC::LessThan, x[0], edge[0]);
                    let c1 = builder.ins().fcmp(FloatCC::LessThan, x[1], edge[1]);
                    let c2 = builder.ins().fcmp(FloatCC::LessThan, x[2], edge[2]);
                    let c3 = builder.ins().fcmp(FloatCC::LessThan, x[3], edge[3]);
                    Ok(TypedValue::Vec4([
                        builder.ins().select(c0, zero, one),
                        builder.ins().select(c1, zero, one),
                        builder.ins().select(c2, zero, one),
                        builder.ins().select(c3, zero, one),
                    ]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "smoothstep" => compile_smoothstep(&args, builder),

        "rotate2d" => {
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1]) {
                (TypedValue::Vec2(v), TypedValue::Scalar(angle)) => {
                    let cos_call = builder.ins().call(math.cos, &[*angle]);
                    let sin_call = builder.ins().call(math.sin, &[*angle]);
                    let c = builder.inst_results(cos_call)[0];
                    let s = builder.inst_results(sin_call)[0];
                    let vx_c = builder.ins().fmul(v[0], c);
                    let vy_s = builder.ins().fmul(v[1], s);
                    let vx_s = builder.ins().fmul(v[0], s);
                    let vy_c = builder.ins().fmul(v[1], c);
                    let rx = builder.ins().fsub(vx_c, vy_s);
                    let ry = builder.ins().fadd(vx_s, vy_c);
                    Ok(TypedValue::Vec2([rx, ry]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        #[cfg(feature = "3d")]
        "rotate_x" => compile_rotate_axis(&args, builder, math, 0),
        #[cfg(feature = "3d")]
        "rotate_y" => compile_rotate_axis(&args, builder, math, 1),
        #[cfg(feature = "3d")]
        "rotate_z" => compile_rotate_axis(&args, builder, math, 2),
        #[cfg(feature = "3d")]
        "rotate3d" => compile_rotate3d(&args, builder, math),

        "mat2" => {
            if args.len() != 4 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1], &args[2], &args[3]) {
                (
                    TypedValue::Scalar(a),
                    TypedValue::Scalar(b),
                    TypedValue::Scalar(c),
                    TypedValue::Scalar(d),
                ) => Ok(TypedValue::Mat2([*a, *b, *c, *d])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        #[cfg(feature = "3d")]
        "mat3" => {
            if args.len() != 9 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            let mut vals = [builder.ins().f32const(0.0); 9];
            for (i, arg) in args.iter().enumerate() {
                match arg {
                    TypedValue::Scalar(v) => vals[i] = *v,
                    _ => return Err(CraneliftError::UnknownFunction(name.to_string())),
                }
            }
            Ok(TypedValue::Mat3(vals))
        }

        #[cfg(feature = "4d")]
        "mat4" => {
            if args.len() != 16 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            let mut vals = [builder.ins().f32const(0.0); 16];
            for (i, arg) in args.iter().enumerate() {
                match arg {
                    TypedValue::Scalar(v) => vals[i] = *v,
                    _ => return Err(CraneliftError::UnknownFunction(name.to_string())),
                }
            }
            Ok(TypedValue::Mat4(vals))
        }

        _ => Err(CraneliftError::UnknownFunction(name.to_string())),
    }
}

fn compile_vectorized_math(
    name: &str,
    args: &[TypedValue],
    builder: &mut FunctionBuilder,
    func_ref: cranelift_codegen::ir::FuncRef,
) -> Result<TypedValue, CraneliftError> {
    if args.len() != 1 {
        return Err(CraneliftError::UnknownFunction(name.to_string()));
    }
    match &args[0] {
        TypedValue::Vec2(v) => {
            let c0 = builder.ins().call(func_ref, &[v[0]]);
            let c1 = builder.ins().call(func_ref, &[v[1]]);
            Ok(TypedValue::Vec2([
                builder.inst_results(c0)[0],
                builder.inst_results(c1)[0],
            ]))
        }
        #[cfg(feature = "3d")]
        TypedValue::Vec3(v) => {
            let c0 = builder.ins().call(func_ref, &[v[0]]);
            let c1 = builder.ins().call(func_ref, &[v[1]]);
            let c2 = builder.ins().call(func_ref, &[v[2]]);
            Ok(TypedValue::Vec3([
                builder.inst_results(c0)[0],
                builder.inst_results(c1)[0],
                builder.inst_results(c2)[0],
            ]))
        }
        #[cfg(feature = "4d")]
        TypedValue::Vec4(v) => {
            let c0 = builder.ins().call(func_ref, &[v[0]]);
            let c1 = builder.ins().call(func_ref, &[v[1]]);
            let c2 = builder.ins().call(func_ref, &[v[2]]);
            let c3 = builder.ins().call(func_ref, &[v[3]]);
            Ok(TypedValue::Vec4([
                builder.inst_results(c0)[0],
                builder.inst_results(c1)[0],
                builder.inst_results(c2)[0],
                builder.inst_results(c3)[0],
            ]))
        }
        _ => Err(CraneliftError::UnknownFunction(name.to_string())),
    }
}

fn compile_vectorized_binary_builtin<F>(
    name: &str,
    args: &[TypedValue],
    builder: &mut FunctionBuilder,
    op: F,
) -> Result<TypedValue, CraneliftError>
where
    F: Fn(&mut FunctionBuilder, CraneliftValue, CraneliftValue) -> CraneliftValue,
{
    if args.len() != 2 {
        return Err(CraneliftError::UnknownFunction(name.to_string()));
    }
    match (&args[0], &args[1]) {
        (TypedValue::Vec2(a), TypedValue::Vec2(b)) => Ok(TypedValue::Vec2([
            op(builder, a[0], b[0]),
            op(builder, a[1], b[1]),
        ])),
        #[cfg(feature = "3d")]
        (TypedValue::Vec3(a), TypedValue::Vec3(b)) => Ok(TypedValue::Vec3([
            op(builder, a[0], b[0]),
            op(builder, a[1], b[1]),
            op(builder, a[2], b[2]),
        ])),
        #[cfg(feature = "4d")]
        (TypedValue::Vec4(a), TypedValue::Vec4(b)) => Ok(TypedValue::Vec4([
            op(builder, a[0], b[0]),
            op(builder, a[1], b[1]),
            op(builder, a[2], b[2]),
            op(builder, a[3], b[3]),
        ])),
        _ => Err(CraneliftError::UnknownFunction(name.to_string())),
    }
}

fn compile_smoothstep(
    args: &[TypedValue],
    builder: &mut FunctionBuilder,
) -> Result<TypedValue, CraneliftError> {
    if args.len() != 3 {
        return Err(CraneliftError::UnknownFunction("smoothstep".to_string()));
    }
    let zero = builder.ins().f32const(0.0);
    let one = builder.ins().f32const(1.0);
    let two = builder.ins().f32const(2.0);
    let three = builder.ins().f32const(3.0);

    #[allow(clippy::too_many_arguments)]
    fn compute(
        builder: &mut FunctionBuilder,
        e0: CraneliftValue,
        e1: CraneliftValue,
        x: CraneliftValue,
        zero: CraneliftValue,
        one: CraneliftValue,
        two: CraneliftValue,
        three: CraneliftValue,
    ) -> CraneliftValue {
        let diff = builder.ins().fsub(x, e0);
        let range = builder.ins().fsub(e1, e0);
        let t_raw = builder.ins().fdiv(diff, range);
        let t_min = builder.ins().fmax(t_raw, zero);
        let t = builder.ins().fmin(t_min, one);
        let t2 = builder.ins().fmul(t, t);
        let two_t = builder.ins().fmul(two, t);
        let sub = builder.ins().fsub(three, two_t);
        builder.ins().fmul(t2, sub)
    }

    match (&args[0], &args[1], &args[2]) {
        (TypedValue::Vec2(e0), TypedValue::Vec2(e1), TypedValue::Vec2(x)) => {
            let r0 = compute(builder, e0[0], e1[0], x[0], zero, one, two, three);
            let r1 = compute(builder, e0[1], e1[1], x[1], zero, one, two, three);
            Ok(TypedValue::Vec2([r0, r1]))
        }
        #[cfg(feature = "3d")]
        (TypedValue::Vec3(e0), TypedValue::Vec3(e1), TypedValue::Vec3(x)) => {
            let r0 = compute(builder, e0[0], e1[0], x[0], zero, one, two, three);
            let r1 = compute(builder, e0[1], e1[1], x[1], zero, one, two, three);
            let r2 = compute(builder, e0[2], e1[2], x[2], zero, one, two, three);
            Ok(TypedValue::Vec3([r0, r1, r2]))
        }
        #[cfg(feature = "4d")]
        (TypedValue::Vec4(e0), TypedValue::Vec4(e1), TypedValue::Vec4(x)) => {
            let r0 = compute(builder, e0[0], e1[0], x[0], zero, one, two, three);
            let r1 = compute(builder, e0[1], e1[1], x[1], zero, one, two, three);
            let r2 = compute(builder, e0[2], e1[2], x[2], zero, one, two, three);
            let r3 = compute(builder, e0[3], e1[3], x[3], zero, one, two, three);
            Ok(TypedValue::Vec4([r0, r1, r2, r3]))
        }
        _ => Err(CraneliftError::UnknownFunction("smoothstep".to_string())),
    }
}

#[cfg(feature = "3d")]
fn compile_rotate_axis(
    args: &[TypedValue],
    builder: &mut FunctionBuilder,
    math: &MathFuncs,
    axis: usize,
) -> Result<TypedValue, CraneliftError> {
    if args.len() != 2 {
        return Err(CraneliftError::UnknownFunction("rotate".to_string()));
    }
    match (&args[0], &args[1]) {
        (TypedValue::Vec3(v), TypedValue::Scalar(angle)) => {
            let cos_call = builder.ins().call(math.cos, &[*angle]);
            let sin_call = builder.ins().call(math.sin, &[*angle]);
            let c = builder.inst_results(cos_call)[0];
            let s = builder.inst_results(sin_call)[0];

            let result = match axis {
                0 => {
                    // rotate_x: [x, y*c - z*s, y*s + z*c]
                    let vy_c = builder.ins().fmul(v[1], c);
                    let vz_s = builder.ins().fmul(v[2], s);
                    let vy_s = builder.ins().fmul(v[1], s);
                    let vz_c = builder.ins().fmul(v[2], c);
                    let ry = builder.ins().fsub(vy_c, vz_s);
                    let rz = builder.ins().fadd(vy_s, vz_c);
                    [v[0], ry, rz]
                }
                1 => {
                    // rotate_y: [x*c + z*s, y, -x*s + z*c]
                    let vx_c = builder.ins().fmul(v[0], c);
                    let vz_s = builder.ins().fmul(v[2], s);
                    let vx_s = builder.ins().fmul(v[0], s);
                    let vz_c = builder.ins().fmul(v[2], c);
                    let rx = builder.ins().fadd(vx_c, vz_s);
                    let neg_vx_s = builder.ins().fneg(vx_s);
                    let rz = builder.ins().fadd(neg_vx_s, vz_c);
                    [rx, v[1], rz]
                }
                _ => {
                    // rotate_z: [x*c - y*s, x*s + y*c, z]
                    let vx_c = builder.ins().fmul(v[0], c);
                    let vy_s = builder.ins().fmul(v[1], s);
                    let vx_s = builder.ins().fmul(v[0], s);
                    let vy_c = builder.ins().fmul(v[1], c);
                    let rx = builder.ins().fsub(vx_c, vy_s);
                    let ry = builder.ins().fadd(vx_s, vy_c);
                    [rx, ry, v[2]]
                }
            };
            Ok(TypedValue::Vec3(result))
        }
        _ => Err(CraneliftError::UnknownFunction("rotate".to_string())),
    }
}

#[cfg(feature = "3d")]
fn compile_rotate3d(
    args: &[TypedValue],
    builder: &mut FunctionBuilder,
    math: &MathFuncs,
) -> Result<TypedValue, CraneliftError> {
    if args.len() != 3 {
        return Err(CraneliftError::UnknownFunction("rotate3d".to_string()));
    }
    match (&args[0], &args[1], &args[2]) {
        (TypedValue::Vec3(v), TypedValue::Vec3(k), TypedValue::Scalar(angle)) => {
            let cos_call = builder.ins().call(math.cos, &[*angle]);
            let sin_call = builder.ins().call(math.sin, &[*angle]);
            let c = builder.inst_results(cos_call)[0];
            let s = builder.inst_results(sin_call)[0];

            // k . v
            let kx_vx = builder.ins().fmul(k[0], v[0]);
            let ky_vy = builder.ins().fmul(k[1], v[1]);
            let kz_vz = builder.ins().fmul(k[2], v[2]);
            let k_dot_v_xy = builder.ins().fadd(kx_vx, ky_vy);
            let k_dot_v = builder.ins().fadd(k_dot_v_xy, kz_vz);

            // k x v
            let ky_vz = builder.ins().fmul(k[1], v[2]);
            let kz_vy = builder.ins().fmul(k[2], v[1]);
            let cross_x = builder.ins().fsub(ky_vz, kz_vy);
            let kz_vx = builder.ins().fmul(k[2], v[0]);
            let kx_vz = builder.ins().fmul(k[0], v[2]);
            let cross_y = builder.ins().fsub(kz_vx, kx_vz);
            let kx_vy = builder.ins().fmul(k[0], v[1]);
            let ky_vx = builder.ins().fmul(k[1], v[0]);
            let cross_z = builder.ins().fsub(kx_vy, ky_vx);

            let one = builder.ins().f32const(1.0);
            let one_minus_c = builder.ins().fsub(one, c);

            let compute = |builder: &mut FunctionBuilder, vi, cross_i, ki| -> CraneliftValue {
                let vi_c = builder.ins().fmul(vi, c);
                let cross_s = builder.ins().fmul(cross_i, s);
                let ki_kdv = builder.ins().fmul(ki, k_dot_v);
                let ki_kdv_omc = builder.ins().fmul(ki_kdv, one_minus_c);
                let sum1 = builder.ins().fadd(vi_c, cross_s);
                builder.ins().fadd(sum1, ki_kdv_omc)
            };

            let rx = compute(builder, v[0], cross_x, k[0]);
            let ry = compute(builder, v[1], cross_y, k[1]);
            let rz = compute(builder, v[2], cross_z, k[2]);

            Ok(TypedValue::Vec3([rx, ry, rz]))
        }
        _ => Err(CraneliftError::UnknownFunction("rotate3d".to_string())),
    }
}
