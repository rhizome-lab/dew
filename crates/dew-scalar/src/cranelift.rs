//! Cranelift JIT compilation for scalar expressions.
//!
//! Compiles expression ASTs to native code via Cranelift.

use cranelift_codegen::ir::{AbiParam, types};
use cranelift_codegen::ir::{InstBuilder, Value as CraneliftValue};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use rhizome_dew_cond::cranelift as cond;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

/// Dispatch a JIT function call based on parameter count (f32 version).
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

/// Dispatch a JIT function call based on parameter count (i32 version).
macro_rules! jit_call_int {
    ($func_ptr:expr, $args:expr, $ret:ty, []) => {{
        let f: extern "C" fn() -> $ret = std::mem::transmute($func_ptr);
        f()
    }};
    ($func_ptr:expr, $args:expr, $ret:ty, [$($idx:tt),+]) => {{
        let f: extern "C" fn($(jit_call_int!(@ty $idx)),+) -> $ret = std::mem::transmute($func_ptr);
        f($($args[$idx]),+)
    }};
    (@ty $idx:tt) => { i32 };
}

// ============================================================================
// Math function wrappers (extern "C" for Cranelift to call)
// ============================================================================

extern "C" fn math_sin(x: f32) -> f32 {
    x.sin()
}
extern "C" fn math_cos(x: f32) -> f32 {
    x.cos()
}
extern "C" fn math_tan(x: f32) -> f32 {
    x.tan()
}
extern "C" fn math_asin(x: f32) -> f32 {
    x.asin()
}
extern "C" fn math_acos(x: f32) -> f32 {
    x.acos()
}
extern "C" fn math_atan(x: f32) -> f32 {
    x.atan()
}
extern "C" fn math_atan2(y: f32, x: f32) -> f32 {
    y.atan2(x)
}
extern "C" fn math_sinh(x: f32) -> f32 {
    x.sinh()
}
extern "C" fn math_cosh(x: f32) -> f32 {
    x.cosh()
}
extern "C" fn math_tanh(x: f32) -> f32 {
    x.tanh()
}
extern "C" fn math_exp(x: f32) -> f32 {
    x.exp()
}
extern "C" fn math_exp2(x: f32) -> f32 {
    x.exp2()
}
extern "C" fn math_ln(x: f32) -> f32 {
    x.ln()
}
extern "C" fn math_log2(x: f32) -> f32 {
    x.log2()
}
extern "C" fn math_log10(x: f32) -> f32 {
    x.log10()
}
extern "C" fn math_pow(base: f32, exp: f32) -> f32 {
    base.powf(exp)
}
extern "C" fn math_sqrt(x: f32) -> f32 {
    x.sqrt()
}
extern "C" fn math_inversesqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

// Integer math wrappers
extern "C" fn math_pow_int(base: i32, exp: i32) -> i32 {
    if exp < 0 {
        0 // Integer power with negative exponent returns 0
    } else {
        let mut result = 1i32;
        let mut b = base;
        let mut e = exp as u32;
        while e > 0 {
            if e & 1 == 1 {
                result = result.wrapping_mul(b);
            }
            b = b.wrapping_mul(b);
            e >>= 1;
        }
        result
    }
}

extern "C" fn math_abs_int(x: i32) -> i32 {
    x.abs()
}

extern "C" fn math_min_int(a: i32, b: i32) -> i32 {
    a.min(b)
}

extern "C" fn math_max_int(a: i32, b: i32) -> i32 {
    a.max(b)
}

extern "C" fn math_clamp_int(x: i32, lo: i32, hi: i32) -> i32 {
    x.max(lo).min(hi)
}

extern "C" fn math_sign_int(x: i32) -> i32 {
    x.signum()
}

struct MathSymbol {
    name: &'static str,
    ptr: *const u8,
    arity: usize,
}

fn math_symbols() -> Vec<MathSymbol> {
    vec![
        MathSymbol {
            name: "dew_sin",
            ptr: math_sin as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_cos",
            ptr: math_cos as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_tan",
            ptr: math_tan as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_asin",
            ptr: math_asin as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_acos",
            ptr: math_acos as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_atan",
            ptr: math_atan as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_atan2",
            ptr: math_atan2 as *const u8,
            arity: 2,
        },
        MathSymbol {
            name: "dew_sinh",
            ptr: math_sinh as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_cosh",
            ptr: math_cosh as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_tanh",
            ptr: math_tanh as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_exp",
            ptr: math_exp as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_exp2",
            ptr: math_exp2 as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_ln",
            ptr: math_ln as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_log2",
            ptr: math_log2 as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_log10",
            ptr: math_log10 as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_pow",
            ptr: math_pow as *const u8,
            arity: 2,
        },
        MathSymbol {
            name: "dew_sqrt",
            ptr: math_sqrt as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_inversesqrt",
            ptr: math_inversesqrt as *const u8,
            arity: 1,
        },
    ]
}

fn math_symbols_int() -> Vec<MathSymbol> {
    vec![
        MathSymbol {
            name: "dew_pow_int",
            ptr: math_pow_int as *const u8,
            arity: 2,
        },
        MathSymbol {
            name: "dew_abs_int",
            ptr: math_abs_int as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "dew_min_int",
            ptr: math_min_int as *const u8,
            arity: 2,
        },
        MathSymbol {
            name: "dew_max_int",
            ptr: math_max_int as *const u8,
            arity: 2,
        },
        MathSymbol {
            name: "dew_clamp_int",
            ptr: math_clamp_int as *const u8,
            arity: 3,
        },
        MathSymbol {
            name: "dew_sign_int",
            ptr: math_sign_int as *const u8,
            arity: 1,
        },
    ]
}

// ============================================================================
// Compiled function
// ============================================================================

/// A compiled function that can be called.
pub struct CompiledFn {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

// SAFETY: The compiled code is self-contained
unsafe impl Send for CompiledFn {}
unsafe impl Sync for CompiledFn {}

impl CompiledFn {
    /// Calls the compiled function with the given arguments.
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

/// A compiled integer function that can be called.
pub struct CompiledFnInt {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

// SAFETY: The compiled code is self-contained
unsafe impl Send for CompiledFnInt {}
unsafe impl Sync for CompiledFnInt {}

impl CompiledFnInt {
    /// Calls the compiled function with the given arguments.
    pub fn call(&self, args: &[i32]) -> i32 {
        assert_eq!(args.len(), self.param_count, "wrong number of arguments");

        unsafe {
            match self.param_count {
                0 => jit_call_int!(self.func_ptr, args, i32, []),
                1 => jit_call_int!(self.func_ptr, args, i32, [0]),
                2 => jit_call_int!(self.func_ptr, args, i32, [0, 1]),
                3 => jit_call_int!(self.func_ptr, args, i32, [0, 1, 2]),
                4 => jit_call_int!(self.func_ptr, args, i32, [0, 1, 2, 3]),
                5 => jit_call_int!(self.func_ptr, args, i32, [0, 1, 2, 3, 4]),
                6 => jit_call_int!(self.func_ptr, args, i32, [0, 1, 2, 3, 4, 5]),
                7 => jit_call_int!(self.func_ptr, args, i32, [0, 1, 2, 3, 4, 5, 6]),
                8 => jit_call_int!(self.func_ptr, args, i32, [0, 1, 2, 3, 4, 5, 6, 7]),
                9 => jit_call_int!(self.func_ptr, args, i32, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
                10 => jit_call_int!(self.func_ptr, args, i32, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                11 => jit_call_int!(self.func_ptr, args, i32, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                12 => jit_call_int!(
                    self.func_ptr,
                    args,
                    i32,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                ),
                13 => jit_call_int!(
                    self.func_ptr,
                    args,
                    i32,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                ),
                14 => jit_call_int!(
                    self.func_ptr,
                    args,
                    i32,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                ),
                15 => jit_call_int!(
                    self.func_ptr,
                    args,
                    i32,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                ),
                16 => jit_call_int!(
                    self.func_ptr,
                    args,
                    i32,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                ),
                _ => panic!("too many parameters (max 16)"),
            }
        }
    }
}

// ============================================================================
// JIT Compiler
// ============================================================================

/// JIT compiler for scalar expressions.
pub struct ScalarJit {
    builder: JITBuilder,
}

impl ScalarJit {
    /// Creates a new JIT compiler.
    pub fn new() -> Result<Self, String> {
        let mut builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|e| e.to_string())?;

        // Register math symbols
        for sym in math_symbols() {
            builder.symbol(sym.name, sym.ptr);
        }

        Ok(Self { builder })
    }

    /// Compiles an expression to a callable function.
    pub fn compile(self, ast: &Ast, params: &[&str]) -> Result<CompiledFn, String> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        // Declare math functions
        let math_ids = declare_math_funcs(&mut module)?;

        // Build function signature
        let mut sig = module.make_signature();
        for _ in params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = module
            .declare_function("expr", Linkage::Export, &sig)
            .map_err(|e| e.to_string())?;

        ctx.func.signature = sig;

        // Build function body
        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Import math functions
            let math_refs = import_math_funcs(&mut builder, &mut module, &math_ids);

            // Map params
            let mut var_map: HashMap<String, CraneliftValue> = HashMap::new();
            for (i, name) in params.iter().enumerate() {
                let val = builder.block_params(entry_block)[i];
                var_map.insert(name.to_string(), val);
            }

            // Compile
            let result = compile_ast(ast, &mut builder, &var_map, &math_refs)?;
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        // Compile to machine code
        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let func_ptr = module.get_finalized_function(func_id);

        Ok(CompiledFn {
            _module: module,
            func_ptr,
            param_count: params.len(),
        })
    }
}

/// JIT compiler for integer scalar expressions.
pub struct ScalarJitInt {
    builder: JITBuilder,
}

impl ScalarJitInt {
    /// Creates a new integer JIT compiler.
    pub fn new() -> Result<Self, String> {
        let mut builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|e| e.to_string())?;

        // Register integer math symbols
        for sym in math_symbols_int() {
            builder.symbol(sym.name, sym.ptr);
        }

        Ok(Self { builder })
    }

    /// Compiles an expression to a callable integer function.
    pub fn compile(self, ast: &Ast, params: &[&str]) -> Result<CompiledFnInt, String> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        // Declare math functions
        let math_ids = declare_math_funcs_int(&mut module)?;

        // Build function signature (i32 params and return)
        let mut sig = module.make_signature();
        for _ in params {
            sig.params.push(AbiParam::new(types::I32));
        }
        sig.returns.push(AbiParam::new(types::I32));

        let func_id = module
            .declare_function("expr_int", Linkage::Export, &sig)
            .map_err(|e| e.to_string())?;

        ctx.func.signature = sig;

        // Build function body
        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Import math functions
            let math_refs = import_math_funcs_int(&mut builder, &mut module, &math_ids);

            // Map params
            let mut var_map: HashMap<String, CraneliftValue> = HashMap::new();
            for (i, name) in params.iter().enumerate() {
                let val = builder.block_params(entry_block)[i];
                var_map.insert(name.to_string(), val);
            }

            // Compile
            let result = compile_ast_int(ast, &mut builder, &var_map, &math_refs)?;
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        // Compile to machine code
        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let func_ptr = module.get_finalized_function(func_id);

        Ok(CompiledFnInt {
            _module: module,
            func_ptr,
            param_count: params.len(),
        })
    }
}

// ============================================================================
// Math function registration
// ============================================================================

struct DeclaredMathFuncs {
    func_ids: HashMap<String, (FuncId, usize)>,
}

fn declare_math_funcs(module: &mut JITModule) -> Result<DeclaredMathFuncs, String> {
    let mut func_ids = HashMap::new();

    for sym in math_symbols() {
        let mut sig = module.make_signature();
        for _ in 0..sym.arity {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = module
            .declare_function(sym.name, Linkage::Import, &sig)
            .map_err(|e| e.to_string())?;

        func_ids.insert(sym.name.to_string(), (func_id, sym.arity));
    }

    Ok(DeclaredMathFuncs { func_ids })
}

struct MathRefs {
    funcs: HashMap<String, cranelift_codegen::ir::FuncRef>,
}

fn import_math_funcs(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    declared: &DeclaredMathFuncs,
) -> MathRefs {
    let mut funcs = HashMap::new();

    for (name, (func_id, _)) in &declared.func_ids {
        let func_ref = module.declare_func_in_func(*func_id, builder.func);
        funcs.insert(name.clone(), func_ref);
    }

    MathRefs { funcs }
}

// ============================================================================
// AST compilation
// ============================================================================

fn compile_ast(
    ast: &Ast,
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, CraneliftValue>,
    math: &MathRefs,
) -> Result<CraneliftValue, String> {
    match ast {
        Ast::Num(n) => Ok(builder.ins().f32const(*n as f32)),

        Ast::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| format!("unknown variable: {}", name)),

        Ast::BinOp(op, left, right) => {
            let l = compile_ast(left, builder, vars, math)?;
            let r = compile_ast(right, builder, vars, math)?;
            Ok(match op {
                BinOp::Add => builder.ins().fadd(l, r),
                BinOp::Sub => builder.ins().fsub(l, r),
                BinOp::Mul => builder.ins().fmul(l, r),
                BinOp::Div => builder.ins().fdiv(l, r),
                BinOp::Pow => {
                    let func_ref = math.funcs.get("dew_pow").ok_or("pow not available")?;
                    let call = builder.ins().call(*func_ref, &[l, r]);
                    builder.inst_results(call)[0]
                }
                // Bitwise ops not supported for floats - convert through i32
                BinOp::Rem | BinOp::BitAnd | BinOp::BitOr | BinOp::Shl | BinOp::Shr => {
                    return Err(format!(
                        "bitwise operator {:?} not supported for floats",
                        op
                    ));
                }
            })
        }

        Ast::UnaryOp(op, inner) => {
            let v = compile_ast(inner, builder, vars, math)?;
            Ok(match op {
                UnaryOp::Neg => builder.ins().fneg(v),
                UnaryOp::Not => {
                    // not(x) returns 1.0 if x == 0.0, else 0.0
                    let bool_val = cond::scalar_to_bool(builder, v);
                    let inverted = cond::emit_not(builder, bool_val);
                    cond::bool_to_scalar(builder, inverted)
                }
                UnaryOp::BitNot => return Err("bitwise NOT not supported for floats".to_string()),
            })
        }

        Ast::Compare(op, left, right) => {
            let l = compile_ast(left, builder, vars, math)?;
            let r = compile_ast(right, builder, vars, math)?;
            let bool_val = cond::emit_compare(builder, *op, l, r);
            Ok(cond::bool_to_scalar(builder, bool_val))
        }

        Ast::And(left, right) => {
            let l = compile_ast(left, builder, vars, math)?;
            let r = compile_ast(right, builder, vars, math)?;
            let l_bool = cond::scalar_to_bool(builder, l);
            let r_bool = cond::scalar_to_bool(builder, r);
            let result_bool = cond::emit_and(builder, l_bool, r_bool);
            Ok(cond::bool_to_scalar(builder, result_bool))
        }

        Ast::Or(left, right) => {
            let l = compile_ast(left, builder, vars, math)?;
            let r = compile_ast(right, builder, vars, math)?;
            let l_bool = cond::scalar_to_bool(builder, l);
            let r_bool = cond::scalar_to_bool(builder, r);
            let result_bool = cond::emit_or(builder, l_bool, r_bool);
            Ok(cond::bool_to_scalar(builder, result_bool))
        }

        Ast::If(cond_ast, then_ast, else_ast) => {
            let c = compile_ast(cond_ast, builder, vars, math)?;
            let then_val = compile_ast(then_ast, builder, vars, math)?;
            let else_val = compile_ast(else_ast, builder, vars, math)?;
            let cond_bool = cond::scalar_to_bool(builder, c);
            Ok(cond::emit_if(builder, cond_bool, then_val, else_val))
        }

        Ast::Call(name, args) => {
            let arg_vals: Vec<CraneliftValue> = args
                .iter()
                .map(|a| compile_ast(a, builder, vars, math))
                .collect::<Result<_, _>>()?;

            compile_function(name, &arg_vals, builder, math)
        }

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

fn compile_function(
    name: &str,
    args: &[CraneliftValue],
    builder: &mut FunctionBuilder,
    math: &MathRefs,
) -> Result<CraneliftValue, String> {
    use cranelift_codegen::ir::condcodes::FloatCC;

    Ok(match name {
        // Constants
        "pi" => builder.ins().f32const(std::f32::consts::PI),
        "e" => builder.ins().f32const(std::f32::consts::E),
        "tau" => builder.ins().f32const(std::f32::consts::TAU),

        // Transcendental functions via Rust callbacks
        "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | "exp"
        | "exp2" | "ln" | "log" | "log2" | "log10" | "sqrt" | "inversesqrt" => {
            let sym_name = if name == "log" || name == "ln" {
                "dew_ln".to_string()
            } else {
                format!("dew_{}", name)
            };
            let func_ref = math
                .funcs
                .get(&sym_name)
                .ok_or_else(|| format!("{} not available", name))?;
            let call = builder.ins().call(*func_ref, &[args[0]]);
            builder.inst_results(call)[0]
        }
        "atan2" => {
            let func_ref = math.funcs.get("dew_atan2").ok_or("atan2 not available")?;
            let call = builder.ins().call(*func_ref, &[args[0], args[1]]);
            builder.inst_results(call)[0]
        }
        "pow" => {
            let func_ref = math.funcs.get("dew_pow").ok_or("pow not available")?;
            let call = builder.ins().call(*func_ref, &[args[0], args[1]]);
            builder.inst_results(call)[0]
        }

        // Native IR functions
        "abs" => builder.ins().fabs(args[0]),
        "sign" => {
            let x = args[0];
            let zero = builder.ins().f32const(0.0);
            let one = builder.ins().f32const(1.0);
            let neg_one = builder.ins().f32const(-1.0);
            let gt_zero = builder.ins().fcmp(FloatCC::GreaterThan, x, zero);
            let lt_zero = builder.ins().fcmp(FloatCC::LessThan, x, zero);
            let neg_or_zero = builder.ins().select(lt_zero, neg_one, zero);
            builder.ins().select(gt_zero, one, neg_or_zero)
        }
        "floor" => builder.ins().floor(args[0]),
        "ceil" => builder.ins().ceil(args[0]),
        "round" => builder.ins().nearest(args[0]),
        "trunc" => builder.ins().trunc(args[0]),
        "fract" => {
            let x = args[0];
            let floor_x = builder.ins().floor(x);
            builder.ins().fsub(x, floor_x)
        }
        "min" => builder.ins().fmin(args[0], args[1]),
        "max" => builder.ins().fmax(args[0], args[1]),
        "clamp" => {
            let (x, lo, hi) = (args[0], args[1], args[2]);
            let min_val = builder.ins().fmin(hi, x);
            builder.ins().fmax(lo, min_val)
        }
        "saturate" => {
            let x = args[0];
            let zero = builder.ins().f32const(0.0);
            let one = builder.ins().f32const(1.0);
            let min_val = builder.ins().fmin(one, x);
            builder.ins().fmax(zero, min_val)
        }

        // Interpolation
        "lerp" | "mix" => {
            let (a, b, t) = (args[0], args[1], args[2]);
            let diff = builder.ins().fsub(b, a);
            let scaled = builder.ins().fmul(diff, t);
            builder.ins().fadd(a, scaled)
        }
        "step" => {
            let (edge, x) = (args[0], args[1]);
            let zero = builder.ins().f32const(0.0);
            let one = builder.ins().f32const(1.0);
            let cmp = builder.ins().fcmp(FloatCC::LessThan, x, edge);
            builder.ins().select(cmp, zero, one)
        }
        "smoothstep" => {
            let (edge0, edge1, x) = (args[0], args[1], args[2]);
            let zero = builder.ins().f32const(0.0);
            let one = builder.ins().f32const(1.0);
            let two = builder.ins().f32const(2.0);
            let three = builder.ins().f32const(3.0);
            let numer = builder.ins().fsub(x, edge0);
            let denom = builder.ins().fsub(edge1, edge0);
            let t_raw = builder.ins().fdiv(numer, denom);
            let t_min = builder.ins().fmin(one, t_raw);
            let t = builder.ins().fmax(zero, t_min);
            let t2 = builder.ins().fmul(t, t);
            let two_t = builder.ins().fmul(two, t);
            let three_minus = builder.ins().fsub(three, two_t);
            builder.ins().fmul(t2, three_minus)
        }
        "inverse_lerp" => {
            let (a, b, v) = (args[0], args[1], args[2]);
            let numer = builder.ins().fsub(v, a);
            let denom = builder.ins().fsub(b, a);
            builder.ins().fdiv(numer, denom)
        }
        "remap" => {
            let (x, in_lo, in_hi, out_lo, out_hi) = (args[0], args[1], args[2], args[3], args[4]);
            let numer = builder.ins().fsub(x, in_lo);
            let denom = builder.ins().fsub(in_hi, in_lo);
            let t = builder.ins().fdiv(numer, denom);
            let out_range = builder.ins().fsub(out_hi, out_lo);
            let scaled = builder.ins().fmul(out_range, t);
            builder.ins().fadd(out_lo, scaled)
        }

        _ => return Err(format!("unknown function: {}", name)),
    })
}

// ============================================================================
// Integer math function registration
// ============================================================================

fn declare_math_funcs_int(module: &mut JITModule) -> Result<DeclaredMathFuncs, String> {
    let mut func_ids = HashMap::new();

    for sym in math_symbols_int() {
        let mut sig = module.make_signature();
        for _ in 0..sym.arity {
            sig.params.push(AbiParam::new(types::I32));
        }
        sig.returns.push(AbiParam::new(types::I32));

        let func_id = module
            .declare_function(sym.name, Linkage::Import, &sig)
            .map_err(|e| e.to_string())?;

        func_ids.insert(sym.name.to_string(), (func_id, sym.arity));
    }

    Ok(DeclaredMathFuncs { func_ids })
}

fn import_math_funcs_int(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    declared: &DeclaredMathFuncs,
) -> MathRefs {
    let mut funcs = HashMap::new();

    for (name, (func_id, _)) in &declared.func_ids {
        let func_ref = module.declare_func_in_func(*func_id, builder.func);
        funcs.insert(name.clone(), func_ref);
    }

    MathRefs { funcs }
}

// ============================================================================
// Integer AST compilation
// ============================================================================

fn compile_ast_int(
    ast: &Ast,
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, CraneliftValue>,
    math: &MathRefs,
) -> Result<CraneliftValue, String> {
    match ast {
        Ast::Num(n) => {
            // Check for fractional part
            if n.fract() != 0.0 {
                return Err(format!("fractional literal {} not allowed for integers", n));
            }
            Ok(builder.ins().iconst(types::I32, *n as i64))
        }

        Ast::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| format!("unknown variable: {}", name)),

        Ast::BinOp(op, left, right) => {
            let l = compile_ast_int(left, builder, vars, math)?;
            let r = compile_ast_int(right, builder, vars, math)?;
            Ok(match op {
                BinOp::Add => builder.ins().iadd(l, r),
                BinOp::Sub => builder.ins().isub(l, r),
                BinOp::Mul => builder.ins().imul(l, r),
                BinOp::Div => builder.ins().sdiv(l, r),
                BinOp::Pow => {
                    let func_ref = math
                        .funcs
                        .get("dew_pow_int")
                        .ok_or("pow not available for integers")?;
                    let call = builder.ins().call(*func_ref, &[l, r]);
                    builder.inst_results(call)[0]
                }
                BinOp::Rem => builder.ins().srem(l, r),
                BinOp::BitAnd => builder.ins().band(l, r),
                BinOp::BitOr => builder.ins().bor(l, r),
                BinOp::Shl => builder.ins().ishl(l, r),
                BinOp::Shr => builder.ins().sshr(l, r),
            })
        }

        Ast::UnaryOp(op, inner) => {
            let v = compile_ast_int(inner, builder, vars, math)?;
            Ok(match op {
                UnaryOp::Neg => builder.ins().ineg(v),
                UnaryOp::Not => {
                    // not(x) returns 1 if x == 0, else 0
                    let zero = builder.ins().iconst(types::I32, 0);
                    let one = builder.ins().iconst(types::I32, 1);
                    let is_zero =
                        builder
                            .ins()
                            .icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, v, zero);
                    builder.ins().select(is_zero, one, zero)
                }
                UnaryOp::BitNot => builder.ins().bnot(v),
            })
        }

        Ast::Compare(op, left, right) => {
            use cranelift_codegen::ir::condcodes::IntCC;
            let l = compile_ast_int(left, builder, vars, math)?;
            let r = compile_ast_int(right, builder, vars, math)?;
            let cc = match op {
                rhizome_dew_core::CompareOp::Lt => IntCC::SignedLessThan,
                rhizome_dew_core::CompareOp::Le => IntCC::SignedLessThanOrEqual,
                rhizome_dew_core::CompareOp::Gt => IntCC::SignedGreaterThan,
                rhizome_dew_core::CompareOp::Ge => IntCC::SignedGreaterThanOrEqual,
                rhizome_dew_core::CompareOp::Eq => IntCC::Equal,
                rhizome_dew_core::CompareOp::Ne => IntCC::NotEqual,
            };
            let cmp = builder.ins().icmp(cc, l, r);
            let one = builder.ins().iconst(types::I32, 1);
            let zero = builder.ins().iconst(types::I32, 0);
            Ok(builder.ins().select(cmp, one, zero))
        }

        Ast::And(left, right) => {
            let l = compile_ast_int(left, builder, vars, math)?;
            let r = compile_ast_int(right, builder, vars, math)?;
            let zero = builder.ins().iconst(types::I32, 0);
            let one = builder.ins().iconst(types::I32, 1);
            let l_nonzero =
                builder
                    .ins()
                    .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, l, zero);
            let r_nonzero =
                builder
                    .ins()
                    .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, r, zero);
            let both = builder.ins().band(l_nonzero, r_nonzero);
            Ok(builder.ins().select(both, one, zero))
        }

        Ast::Or(left, right) => {
            let l = compile_ast_int(left, builder, vars, math)?;
            let r = compile_ast_int(right, builder, vars, math)?;
            let zero = builder.ins().iconst(types::I32, 0);
            let one = builder.ins().iconst(types::I32, 1);
            let l_nonzero =
                builder
                    .ins()
                    .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, l, zero);
            let r_nonzero =
                builder
                    .ins()
                    .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, r, zero);
            let either = builder.ins().bor(l_nonzero, r_nonzero);
            Ok(builder.ins().select(either, one, zero))
        }

        Ast::If(cond_ast, then_ast, else_ast) => {
            let c = compile_ast_int(cond_ast, builder, vars, math)?;
            let then_val = compile_ast_int(then_ast, builder, vars, math)?;
            let else_val = compile_ast_int(else_ast, builder, vars, math)?;
            let zero = builder.ins().iconst(types::I32, 0);
            let cond_nonzero =
                builder
                    .ins()
                    .icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, c, zero);
            Ok(builder.ins().select(cond_nonzero, then_val, else_val))
        }

        Ast::Call(name, args) => {
            let arg_vals: Vec<CraneliftValue> = args
                .iter()
                .map(|a| compile_ast_int(a, builder, vars, math))
                .collect::<Result<_, _>>()?;

            compile_function_int(name, &arg_vals, builder, math)
        }

        Ast::Let { name, value, body } => {
            // Compile the value expression
            let value_val = compile_ast_int(value, builder, vars, math)?;
            // Extend vars with the new binding
            let mut new_vars = vars.clone();
            new_vars.insert(name.clone(), value_val);
            // Compile the body with extended environment
            compile_ast_int(body, builder, &new_vars, math)
        }
    }
}

fn compile_function_int(
    name: &str,
    args: &[CraneliftValue],
    builder: &mut FunctionBuilder,
    math: &MathRefs,
) -> Result<CraneliftValue, String> {
    use cranelift_codegen::ir::condcodes::IntCC;

    Ok(match name {
        // Functions via extern calls
        "abs" => {
            let func_ref = math.funcs.get("dew_abs_int").ok_or("abs not available")?;
            let call = builder.ins().call(*func_ref, &[args[0]]);
            builder.inst_results(call)[0]
        }
        "sign" => {
            let func_ref = math.funcs.get("dew_sign_int").ok_or("sign not available")?;
            let call = builder.ins().call(*func_ref, &[args[0]]);
            builder.inst_results(call)[0]
        }
        "min" => {
            let func_ref = math.funcs.get("dew_min_int").ok_or("min not available")?;
            let call = builder.ins().call(*func_ref, &[args[0], args[1]]);
            builder.inst_results(call)[0]
        }
        "max" => {
            let func_ref = math.funcs.get("dew_max_int").ok_or("max not available")?;
            let call = builder.ins().call(*func_ref, &[args[0], args[1]]);
            builder.inst_results(call)[0]
        }
        "clamp" => {
            let func_ref = math
                .funcs
                .get("dew_clamp_int")
                .ok_or("clamp not available")?;
            let call = builder.ins().call(*func_ref, &[args[0], args[1], args[2]]);
            builder.inst_results(call)[0]
        }
        "pow" => {
            let func_ref = math.funcs.get("dew_pow_int").ok_or("pow not available")?;
            let call = builder.ins().call(*func_ref, &[args[0], args[1]]);
            builder.inst_results(call)[0]
        }

        // Interpolation (integer version)
        "lerp" | "mix" => {
            let (a, b, t) = (args[0], args[1], args[2]);
            let diff = builder.ins().isub(b, a);
            let scaled = builder.ins().imul(diff, t);
            builder.ins().iadd(a, scaled)
        }
        "step" => {
            let (edge, x) = (args[0], args[1]);
            let zero = builder.ins().iconst(types::I32, 0);
            let one = builder.ins().iconst(types::I32, 1);
            let cmp = builder.ins().icmp(IntCC::SignedLessThan, x, edge);
            builder.ins().select(cmp, zero, one)
        }

        // Trig/exp functions not available for integers
        "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2" | "sinh" | "cosh" | "tanh"
        | "exp" | "exp2" | "ln" | "log" | "log2" | "log10" | "sqrt" | "inversesqrt" | "floor"
        | "ceil" | "round" | "trunc" | "fract" | "smoothstep" | "saturate" | "inverse_lerp"
        | "remap" | "pi" | "e" | "tau" => {
            return Err(format!("function {} not available for integers", name));
        }

        _ => return Err(format!("unknown function: {}", name)),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn eval(input: &str, params: &[&str], args: &[f32]) -> f32 {
        let expr = Expr::parse(input).unwrap();
        let jit = ScalarJit::new().unwrap();
        let func = jit.compile(expr.ast(), params).unwrap();
        func.call(args)
    }

    #[test]
    fn test_constants() {
        assert!((eval("pi()", &[], &[]) - std::f32::consts::PI).abs() < 0.001);
        assert!((eval("e()", &[], &[]) - std::f32::consts::E).abs() < 0.001);
        assert!((eval("tau()", &[], &[]) - std::f32::consts::TAU).abs() < 0.001);
    }

    #[test]
    fn test_operators() {
        assert_eq!(eval("x + y", &["x", "y"], &[3.0, 4.0]), 7.0);
        assert_eq!(eval("x * y", &["x", "y"], &[3.0, 4.0]), 12.0);
        assert_eq!(eval("-x", &["x"], &[5.0]), -5.0);
        assert_eq!(eval("x ^ 2", &["x"], &[3.0]), 9.0);
    }

    #[test]
    fn test_trig() {
        assert!(eval("sin(0)", &[], &[]).abs() < 0.001);
        assert!((eval("cos(0)", &[], &[]) - 1.0).abs() < 0.001);
        assert!((eval("atan2(1, 1)", &[], &[]) - std::f32::consts::FRAC_PI_4).abs() < 0.001);
    }

    #[test]
    fn test_exp_log() {
        assert!((eval("exp(0)", &[], &[]) - 1.0).abs() < 0.001);
        assert!((eval("exp2(3)", &[], &[]) - 8.0).abs() < 0.001);
        assert!(eval("ln(1)", &[], &[]).abs() < 0.001);
        assert!((eval("log2(8)", &[], &[]) - 3.0).abs() < 0.001);
        assert!((eval("sqrt(16)", &[], &[]) - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_common() {
        assert_eq!(eval("abs(x)", &["x"], &[-5.0]), 5.0);
        assert_eq!(eval("floor(x)", &["x"], &[3.7]), 3.0);
        assert_eq!(eval("ceil(x)", &["x"], &[3.2]), 4.0);
        assert_eq!(eval("min(x, y)", &["x", "y"], &[3.0, 7.0]), 3.0);
        assert_eq!(eval("max(x, y)", &["x", "y"], &[3.0, 7.0]), 7.0);
        assert_eq!(
            eval("clamp(x, lo, hi)", &["x", "lo", "hi"], &[5.0, 0.0, 3.0]),
            3.0
        );
        assert_eq!(eval("saturate(x)", &["x"], &[1.5]), 1.0);
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(
            eval("lerp(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 0.5]),
            5.0
        );
        assert_eq!(eval("step(edge, x)", &["edge", "x"], &[0.5, 0.3]), 0.0);
        assert_eq!(eval("step(edge, x)", &["edge", "x"], &[0.5, 0.7]), 1.0);
        assert_eq!(
            eval("inverse_lerp(a, b, v)", &["a", "b", "v"], &[0.0, 10.0, 5.0]),
            0.5
        );
    }

    #[test]
    fn test_remap() {
        let result = eval(
            "remap(x, in_lo, in_hi, out_lo, out_hi)",
            &["x", "in_lo", "in_hi", "out_lo", "out_hi"],
            &[5.0, 0.0, 10.0, 0.0, 100.0],
        );
        assert_eq!(result, 50.0);
    }

    #[test]
    fn test_compare() {
        assert_eq!(eval("1 < 2", &[], &[]), 1.0);
        assert_eq!(eval("2 < 1", &[], &[]), 0.0);
        assert_eq!(eval("x < 5", &["x"], &[3.0]), 1.0);
        assert_eq!(eval("x >= 5", &["x"], &[5.0]), 1.0);
        assert_eq!(eval("x == 5", &["x"], &[5.0]), 1.0);
        assert_eq!(eval("x != 5", &["x"], &[5.0]), 0.0);
    }

    #[test]
    fn test_if_then_else() {
        assert_eq!(eval("if 1 then 10 else 20", &[], &[]), 10.0);
        assert_eq!(eval("if 0 then 10 else 20", &[], &[]), 20.0);
        assert_eq!(eval("if x > 5 then 1 else 0", &["x"], &[10.0]), 1.0);
        assert_eq!(eval("if x > 5 then 1 else 0", &["x"], &[3.0]), 0.0);
    }

    #[test]
    fn test_and_or() {
        assert_eq!(eval("1 and 1", &[], &[]), 1.0);
        assert_eq!(eval("1 and 0", &[], &[]), 0.0);
        assert_eq!(eval("0 or 1", &[], &[]), 1.0);
        assert_eq!(eval("0 or 0", &[], &[]), 0.0);
    }

    #[test]
    fn test_not() {
        assert_eq!(eval("not 0", &[], &[]), 1.0);
        assert_eq!(eval("not 1", &[], &[]), 0.0);
    }

    // Integer JIT tests
    fn eval_int(input: &str, params: &[&str], args: &[i32]) -> i32 {
        let expr = Expr::parse(input).unwrap();
        let jit = ScalarJitInt::new().unwrap();
        let func = jit.compile(expr.ast(), params).unwrap();
        func.call(args)
    }

    #[test]
    fn test_int_operators() {
        assert_eq!(eval_int("x + y", &["x", "y"], &[3, 4]), 7);
        assert_eq!(eval_int("x * y", &["x", "y"], &[3, 4]), 12);
        assert_eq!(eval_int("x - y", &["x", "y"], &[10, 3]), 7);
        assert_eq!(eval_int("x / y", &["x", "y"], &[10, 3]), 3);
        assert_eq!(eval_int("-x", &["x"], &[5]), -5);
        assert_eq!(eval_int("x ^ 3", &["x"], &[2]), 8);
    }

    #[test]
    fn test_int_modulo() {
        assert_eq!(eval_int("x % y", &["x", "y"], &[10, 3]), 1);
        assert_eq!(eval_int("8 % 3", &[], &[]), 2);
    }

    #[test]
    fn test_int_bitwise() {
        assert_eq!(eval_int("x & y", &["x", "y"], &[0b1010, 0b1100]), 0b1000);
        assert_eq!(eval_int("x | y", &["x", "y"], &[0b1010, 0b1100]), 0b1110);
        assert_eq!(eval_int("x << 2", &["x"], &[1]), 4);
        assert_eq!(eval_int("x >> 2", &["x"], &[16]), 4);
        assert_eq!(eval_int("~0", &[], &[]), -1);
    }

    #[test]
    fn test_int_compare() {
        assert_eq!(eval_int("1 < 2", &[], &[]), 1);
        assert_eq!(eval_int("2 < 1", &[], &[]), 0);
        assert_eq!(eval_int("x == 5", &["x"], &[5]), 1);
    }

    #[test]
    fn test_int_if_then_else() {
        assert_eq!(eval_int("if 1 then 10 else 20", &[], &[]), 10);
        assert_eq!(eval_int("if 0 then 10 else 20", &[], &[]), 20);
        assert_eq!(eval_int("if x > 5 then 1 else 0", &["x"], &[10]), 1);
    }

    #[test]
    fn test_int_functions() {
        assert_eq!(eval_int("abs(x)", &["x"], &[-5]), 5);
        assert_eq!(eval_int("min(x, y)", &["x", "y"], &[3, 7]), 3);
        assert_eq!(eval_int("max(x, y)", &["x", "y"], &[3, 7]), 7);
        assert_eq!(eval_int("clamp(x, 0, 10)", &["x"], &[15]), 10);
        assert_eq!(eval_int("sign(x)", &["x"], &[-5]), -1);
    }
}
