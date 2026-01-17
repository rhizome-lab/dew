//! JIT compiler for complex expressions.

use super::compiled::{CompiledComplexFn, CompiledComplexPairFn};
use super::error::CraneliftError;
use super::types::{MathFuncs, TypedValue, VarSpec, math_symbols};
use crate::Type;
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

/// JIT compiler for complex expressions.
pub struct ComplexJit {
    builder: JITBuilder,
}

impl ComplexJit {
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
    ) -> Result<CompiledComplexFn, CraneliftError> {
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
            .declare_function("complex_sqrt", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("complex_pow", Linkage::Import, &sig_f32_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let atan2_id = module
            .declare_function("complex_atan2", Linkage::Import, &sig_f32_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let sin_id = module
            .declare_function("complex_sin", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("complex_cos", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let exp_id = module
            .declare_function("complex_exp", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let log_id = module
            .declare_function("complex_log", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        // Build function signature
        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = module
            .declare_function("complex_expr", Linkage::Export, &sig)
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
                atan2: module.declare_func_in_func(atan2_id, builder.func),
                sin: module.declare_func_in_func(sin_id, builder.func),
                cos: module.declare_func_in_func(cos_id, builder.func),
                exp: module.declare_func_in_func(exp_id, builder.func),
                log: module.declare_func_in_func(log_id, builder.func),
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
                    Type::Complex => {
                        let v = TypedValue::Complex([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                        ]);
                        param_idx += 2;
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

        Ok(CompiledComplexFn {
            _module: module,
            func_ptr,
            param_count: total_params,
        })
    }

    /// Compiles an expression that returns a complex value.
    /// The compiled function takes an output pointer as the last argument.
    pub fn compile_complex(
        self,
        ast: &Ast,
        vars: &[VarSpec],
    ) -> Result<CompiledComplexPairFn, CraneliftError> {
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
            .declare_function("complex_sqrt", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("complex_pow", Linkage::Import, &sig_f32_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let atan2_id = module
            .declare_function("complex_atan2", Linkage::Import, &sig_f32_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let sin_id = module
            .declare_function("complex_sin", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("complex_cos", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let exp_id = module
            .declare_function("complex_exp", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let log_id = module
            .declare_function("complex_log", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        // Build function signature - input params + output pointer, no return
        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let ptr_type = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.params.push(AbiParam::new(ptr_type)); // output pointer
        // No return value - writes to pointer

        let func_id = module
            .declare_function("complex_expr", Linkage::Export, &sig)
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
                atan2: module.declare_func_in_func(atan2_id, builder.func),
                sin: module.declare_func_in_func(sin_id, builder.func),
                cos: module.declare_func_in_func(cos_id, builder.func),
                exp: module.declare_func_in_func(exp_id, builder.func),
                log: module.declare_func_in_func(log_id, builder.func),
            };

            // Map variables to typed values (excluding output pointer)
            let block_params = builder.block_params(entry_block).to_vec();
            let out_ptr = block_params[total_params]; // Last param is output pointer
            let mut var_map: HashMap<String, TypedValue> = HashMap::new();
            let mut param_idx = 0;

            for var in vars {
                let typed_val = match var.typ {
                    Type::Scalar => {
                        let v = TypedValue::Scalar(block_params[param_idx]);
                        param_idx += 1;
                        v
                    }
                    Type::Complex => {
                        let v = TypedValue::Complex([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                        ]);
                        param_idx += 2;
                        v
                    }
                };
                var_map.insert(var.name.clone(), typed_val);
            }

            let result = compile_ast(ast, &mut builder, &var_map, &math_funcs)?;

            let [re, im] = result
                .as_complex()
                .ok_or(CraneliftError::UnsupportedReturnType(result.typ()))?;

            // Store results to output pointer
            builder
                .ins()
                .store(MemFlags::new(), re, out_ptr, Offset32::new(0));
            builder
                .ins()
                .store(MemFlags::new(), im, out_ptr, Offset32::new(4));
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

        Ok(CompiledComplexPairFn {
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
        Ast::Num(n) => Ok(TypedValue::Scalar(builder.ins().f32const(*n as f32))),

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

        // Complex + Complex
        (BinOp::Add, TypedValue::Complex(l), TypedValue::Complex(r)) => Ok(TypedValue::Complex([
            builder.ins().fadd(l[0], r[0]),
            builder.ins().fadd(l[1], r[1]),
        ])),
        (BinOp::Sub, TypedValue::Complex(l), TypedValue::Complex(r)) => Ok(TypedValue::Complex([
            builder.ins().fsub(l[0], r[0]),
            builder.ins().fsub(l[1], r[1]),
        ])),

        // Complex * Scalar
        (BinOp::Mul, TypedValue::Complex(c), TypedValue::Scalar(s)) => Ok(TypedValue::Complex([
            builder.ins().fmul(c[0], *s),
            builder.ins().fmul(c[1], *s),
        ])),
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Complex(c)) => Ok(TypedValue::Complex([
            builder.ins().fmul(*s, c[0]),
            builder.ins().fmul(*s, c[1]),
        ])),

        // Complex * Complex: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        (BinOp::Mul, TypedValue::Complex(l), TypedValue::Complex(r)) => {
            let ac = builder.ins().fmul(l[0], r[0]);
            let bd = builder.ins().fmul(l[1], r[1]);
            let ad = builder.ins().fmul(l[0], r[1]);
            let bc = builder.ins().fmul(l[1], r[0]);
            Ok(TypedValue::Complex([
                builder.ins().fsub(ac, bd),
                builder.ins().fadd(ad, bc),
            ]))
        }

        // Complex / Scalar
        (BinOp::Div, TypedValue::Complex(c), TypedValue::Scalar(s)) => Ok(TypedValue::Complex([
            builder.ins().fdiv(c[0], *s),
            builder.ins().fdiv(c[1], *s),
        ])),

        // Complex / Complex
        (BinOp::Div, TypedValue::Complex(l), TypedValue::Complex(r)) => {
            // (a+bi)/(c+di) = (ac+bd)/(c²+d²) + (bc-ad)/(c²+d²)i
            let c2 = builder.ins().fmul(r[0], r[0]);
            let d2 = builder.ins().fmul(r[1], r[1]);
            let denom = builder.ins().fadd(c2, d2);

            let ac = builder.ins().fmul(l[0], r[0]);
            let bd = builder.ins().fmul(l[1], r[1]);
            let bc = builder.ins().fmul(l[1], r[0]);
            let ad = builder.ins().fmul(l[0], r[1]);

            let real_num = builder.ins().fadd(ac, bd);
            let imag_num = builder.ins().fsub(bc, ad);

            Ok(TypedValue::Complex([
                builder.ins().fdiv(real_num, denom),
                builder.ins().fdiv(imag_num, denom),
            ]))
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
            TypedValue::Complex(c) => Ok(TypedValue::Complex([
                builder.ins().fneg(c[0]),
                builder.ins().fneg(c[1]),
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
        "re" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Complex(c) => Ok(TypedValue::Scalar(c[0])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "im" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Complex(c) => Ok(TypedValue::Scalar(c[1])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "abs" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Scalar(v) => {
                    // For scalar, just compute absolute value via sqrt(x*x)
                    let sq = builder.ins().fmul(*v, *v);
                    let call = builder.ins().call(math.sqrt, &[sq]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                TypedValue::Complex(c) => {
                    // |z| = sqrt(a² + b²)
                    let a2 = builder.ins().fmul(c[0], c[0]);
                    let b2 = builder.ins().fmul(c[1], c[1]);
                    let sum = builder.ins().fadd(a2, b2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
            }
        }

        "arg" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Complex(c) => {
                    let call = builder.ins().call(math.atan2, &[c[1], c[0]]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "norm" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Complex(c) => {
                    // norm(z) = a² + b²
                    let a2 = builder.ins().fmul(c[0], c[0]);
                    let b2 = builder.ins().fmul(c[1], c[1]);
                    Ok(TypedValue::Scalar(builder.ins().fadd(a2, b2)))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "conj" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Complex(c) => Ok(TypedValue::Complex([c[0], builder.ins().fneg(c[1])])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "exp" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Scalar(v) => {
                    let call = builder.ins().call(math.exp, &[*v]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                TypedValue::Complex(c) => {
                    // exp(a + bi) = e^a * (cos(b) + i*sin(b))
                    let e_a_call = builder.ins().call(math.exp, &[c[0]]);
                    let e_a = builder.inst_results(e_a_call)[0];
                    let cos_b_call = builder.ins().call(math.cos, &[c[1]]);
                    let cos_b = builder.inst_results(cos_b_call)[0];
                    let sin_b_call = builder.ins().call(math.sin, &[c[1]]);
                    let sin_b = builder.inst_results(sin_b_call)[0];
                    let re = builder.ins().fmul(e_a, cos_b);
                    let im = builder.ins().fmul(e_a, sin_b);
                    Ok(TypedValue::Complex([re, im]))
                }
            }
        }

        "log" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Scalar(v) => {
                    let call = builder.ins().call(math.log, &[*v]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                TypedValue::Complex(c) => {
                    // log(a + bi) = ln(|z|) + i*arg(z) = ln(sqrt(a² + b²)) + i*atan2(b, a)
                    let a2 = builder.ins().fmul(c[0], c[0]);
                    let b2 = builder.ins().fmul(c[1], c[1]);
                    let sum = builder.ins().fadd(a2, b2);
                    let r_call = builder.ins().call(math.sqrt, &[sum]);
                    let r = builder.inst_results(r_call)[0];
                    let ln_r_call = builder.ins().call(math.log, &[r]);
                    let ln_r = builder.inst_results(ln_r_call)[0];
                    let theta_call = builder.ins().call(math.atan2, &[c[1], c[0]]);
                    let theta = builder.inst_results(theta_call)[0];
                    Ok(TypedValue::Complex([ln_r, theta]))
                }
            }
        }

        "sqrt" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Scalar(v) => {
                    let call = builder.ins().call(math.sqrt, &[*v]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                TypedValue::Complex(c) => {
                    // sqrt(z) = sqrt(r) * e^(i*θ/2)
                    // where r = |z| = sqrt(a² + b²), θ = atan2(b, a)
                    let a2 = builder.ins().fmul(c[0], c[0]);
                    let b2 = builder.ins().fmul(c[1], c[1]);
                    let sum = builder.ins().fadd(a2, b2);
                    let r_call = builder.ins().call(math.sqrt, &[sum]);
                    let r = builder.inst_results(r_call)[0];
                    let sqrt_r_call = builder.ins().call(math.sqrt, &[r]);
                    let sqrt_r = builder.inst_results(sqrt_r_call)[0];
                    let theta_call = builder.ins().call(math.atan2, &[c[1], c[0]]);
                    let theta = builder.inst_results(theta_call)[0];
                    let half = builder.ins().f32const(0.5);
                    let half_theta = builder.ins().fmul(theta, half);
                    let cos_call = builder.ins().call(math.cos, &[half_theta]);
                    let cos_half = builder.inst_results(cos_call)[0];
                    let sin_call = builder.ins().call(math.sin, &[half_theta]);
                    let sin_half = builder.inst_results(sin_call)[0];
                    let re = builder.ins().fmul(sqrt_r, cos_half);
                    let im = builder.ins().fmul(sqrt_r, sin_half);
                    Ok(TypedValue::Complex([re, im]))
                }
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
        let jit = ComplexJit::new().unwrap();
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
    fn test_complex_abs() {
        let expr = Expr::parse("abs(z)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_scalar(expr.ast(), &[VarSpec::new("z", Type::Complex)])
            .unwrap();
        // abs(3+4i) = 5
        assert_eq!(func.call(&[3.0, 4.0]), 5.0);
    }

    #[test]
    fn test_complex_re_im() {
        let expr_re = Expr::parse("re(z)").unwrap();
        let expr_im = Expr::parse("im(z)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func_re = jit
            .compile_scalar(expr_re.ast(), &[VarSpec::new("z", Type::Complex)])
            .unwrap();
        let jit = ComplexJit::new().unwrap();
        let func_im = jit
            .compile_scalar(expr_im.ast(), &[VarSpec::new("z", Type::Complex)])
            .unwrap();

        assert_eq!(func_re.call(&[3.0, 4.0]), 3.0);
        assert_eq!(func_im.call(&[3.0, 4.0]), 4.0);
    }

    #[test]
    fn test_complex_arg() {
        let expr = Expr::parse("arg(z)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_scalar(expr.ast(), &[VarSpec::new("z", Type::Complex)])
            .unwrap();
        // arg(1+1i) = pi/4
        let result = func.call(&[1.0, 1.0]);
        assert!(approx_eq(result, std::f32::consts::FRAC_PI_4));
    }

    #[test]
    fn test_complex_mul() {
        // abs((1+2i) * (3+4i)) = abs(-5+10i) = sqrt(125)
        let expr = Expr::parse("abs(a * b)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[
                    VarSpec::new("a", Type::Complex),
                    VarSpec::new("b", Type::Complex),
                ],
            )
            .unwrap();
        let result = func.call(&[1.0, 2.0, 3.0, 4.0]);
        assert!(approx_eq(result, (125.0_f32).sqrt()));
    }

    #[test]
    fn test_complex_div() {
        // (4+2i) / (2+0i) = (2+1i), abs = sqrt(5)
        let expr = Expr::parse("abs(a / b)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[
                    VarSpec::new("a", Type::Complex),
                    VarSpec::new("b", Type::Complex),
                ],
            )
            .unwrap();
        let result = func.call(&[4.0, 2.0, 2.0, 0.0]);
        assert!(approx_eq(result, (5.0_f32).sqrt()));
    }

    #[test]
    fn test_compile_complex_add() {
        // (1+2i) + (3+4i) = (4+6i)
        let expr = Expr::parse("a + b").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_complex(
                expr.ast(),
                &[
                    VarSpec::new("a", Type::Complex),
                    VarSpec::new("b", Type::Complex),
                ],
            )
            .unwrap();
        let [re, im] = func.call(&[1.0, 2.0, 3.0, 4.0]);
        assert!(approx_eq(re, 4.0));
        assert!(approx_eq(im, 6.0));
    }

    #[test]
    fn test_compile_complex_mul() {
        // (1+2i) * (3+4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        let expr = Expr::parse("a * b").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_complex(
                expr.ast(),
                &[
                    VarSpec::new("a", Type::Complex),
                    VarSpec::new("b", Type::Complex),
                ],
            )
            .unwrap();
        let [re, im] = func.call(&[1.0, 2.0, 3.0, 4.0]);
        assert!(approx_eq(re, -5.0));
        assert!(approx_eq(im, 10.0));
    }

    #[test]
    fn test_compile_complex_conj() {
        // conj(3+4i) = 3-4i
        let expr = Expr::parse("conj(z)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_complex(expr.ast(), &[VarSpec::new("z", Type::Complex)])
            .unwrap();
        let [re, im] = func.call(&[3.0, 4.0]);
        assert!(approx_eq(re, 3.0));
        assert!(approx_eq(im, -4.0));
    }

    #[test]
    fn test_compile_complex_exp() {
        // exp(0 + πi) = -1 + 0i (Euler's identity)
        let expr = Expr::parse("exp(z)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_complex(expr.ast(), &[VarSpec::new("z", Type::Complex)])
            .unwrap();
        let [re, im] = func.call(&[0.0, std::f32::consts::PI]);
        assert!(approx_eq(re, -1.0));
        assert!(approx_eq(im, 0.0));
    }

    #[test]
    fn test_compile_complex_log() {
        // log(e + 0i) = 1 + 0i
        let expr = Expr::parse("log(z)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_complex(expr.ast(), &[VarSpec::new("z", Type::Complex)])
            .unwrap();
        let [re, im] = func.call(&[std::f32::consts::E, 0.0]);
        assert!(approx_eq(re, 1.0));
        assert!(approx_eq(im, 0.0));
    }

    #[test]
    fn test_compile_complex_sqrt() {
        // sqrt(0 + 4i) should give sqrt(2) + sqrt(2)i
        // Since sqrt(4i) = sqrt(4) * e^(i*π/4) = 2 * (cos(π/4) + i*sin(π/4)) = sqrt(2) + sqrt(2)i
        let expr = Expr::parse("sqrt(z)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_complex(expr.ast(), &[VarSpec::new("z", Type::Complex)])
            .unwrap();
        let [re, im] = func.call(&[0.0, 4.0]);
        let sqrt2 = std::f32::consts::SQRT_2;
        assert!(approx_eq(re, sqrt2));
        assert!(approx_eq(im, sqrt2));
    }
}
