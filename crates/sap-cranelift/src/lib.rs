//! Cranelift JIT compilation backend for sap expressions.
//!
//! Compiles expression ASTs to native code via Cranelift.
//!
//! # Example
//!
//! ```ignore
//! use sap_core::Expr;
//! use sap_cranelift::{JitCompiler, CraneliftRegistry};
//!
//! let expr = Expr::parse("x * 2 + y").unwrap();
//! let registry = CraneliftRegistry::new();
//! let mut jit = JitCompiler::new().unwrap();
//! let func = jit.compile(expr.ast(), &["x", "y"], &registry).unwrap();
//!
//! // Call the compiled function
//! let result = func.call(&[3.0, 1.0]);
//! assert_eq!(result, 7.0);
//! ```

use cranelift::prelude::*;

// Re-export Cranelift types needed for implementing CraneliftFn
pub use cranelift::codegen::ir::FuncRef;
pub use cranelift::prelude::{FloatCC, FunctionBuilder, InstBuilder, Value};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use sap_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

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

/// Info about a math symbol: name and arity
struct MathSymbol {
    name: &'static str,
    ptr: *const u8,
    arity: usize,
}

fn math_symbols() -> Vec<MathSymbol> {
    vec![
        MathSymbol {
            name: "sap_sin",
            ptr: math_sin as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_cos",
            ptr: math_cos as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_tan",
            ptr: math_tan as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_asin",
            ptr: math_asin as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_acos",
            ptr: math_acos as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_atan",
            ptr: math_atan as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_atan2",
            ptr: math_atan2 as *const u8,
            arity: 2,
        },
        MathSymbol {
            name: "sap_sinh",
            ptr: math_sinh as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_cosh",
            ptr: math_cosh as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_tanh",
            ptr: math_tanh as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_exp",
            ptr: math_exp as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_exp2",
            ptr: math_exp2 as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_ln",
            ptr: math_ln as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_log2",
            ptr: math_log2 as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_log10",
            ptr: math_log10 as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_pow",
            ptr: math_pow as *const u8,
            arity: 2,
        },
        MathSymbol {
            name: "sap_sqrt",
            ptr: math_sqrt as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_inversesqrt",
            ptr: math_inversesqrt as *const u8,
            arity: 1,
        },
    ]
}

/// Available math function references for use during compilation.
#[derive(Default)]
pub struct MathFuncs {
    funcs: HashMap<String, FuncRef>,
}

impl MathFuncs {
    /// Get a FuncRef for a math function by symbol name (e.g., "sap_sin").
    pub fn get(&self, name: &str) -> Option<FuncRef> {
        self.funcs.get(name).copied()
    }
}

// ============================================================================
// Cranelift Function Registry
// ============================================================================

/// A function that can be JIT compiled.
pub trait CraneliftFn: Send + Sync {
    /// Function name in the expression language.
    fn name(&self) -> &str;

    /// Emit Cranelift IR for this function call.
    ///
    /// The `math` parameter provides access to pre-registered math functions
    /// (sin, cos, exp, etc.) that can be called from JIT code.
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value], math: &MathFuncs) -> Value;
}

/// Registry of Cranelift function implementations.
#[derive(Default)]
pub struct CraneliftRegistry {
    funcs: HashMap<String, Box<dyn CraneliftFn>>,
}

impl CraneliftRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a function.
    pub fn register<F: CraneliftFn + 'static>(&mut self, func: F) {
        self.funcs.insert(func.name().to_string(), Box::new(func));
    }

    /// Gets a function by name.
    pub fn get(&self, name: &str) -> Option<&dyn CraneliftFn> {
        self.funcs.get(name).map(|f| f.as_ref())
    }
}

// ============================================================================
// JIT Compiler
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
    ///
    /// # Panics
    ///
    /// Panics if the number of arguments doesn't match the parameter count.
    pub fn call(&self, args: &[f32]) -> f32 {
        assert_eq!(args.len(), self.param_count, "wrong number of arguments");

        // SAFETY: We generated this code and know its signature
        unsafe {
            match self.param_count {
                0 => {
                    let f: extern "C" fn() -> f32 = std::mem::transmute(self.func_ptr);
                    f()
                }
                1 => {
                    let f: extern "C" fn(f32) -> f32 = std::mem::transmute(self.func_ptr);
                    f(args[0])
                }
                2 => {
                    let f: extern "C" fn(f32, f32) -> f32 = std::mem::transmute(self.func_ptr);
                    f(args[0], args[1])
                }
                3 => {
                    let f: extern "C" fn(f32, f32, f32) -> f32 = std::mem::transmute(self.func_ptr);
                    f(args[0], args[1], args[2])
                }
                4 => {
                    let f: extern "C" fn(f32, f32, f32, f32) -> f32 =
                        std::mem::transmute(self.func_ptr);
                    f(args[0], args[1], args[2], args[3])
                }
                5 => {
                    let f: extern "C" fn(f32, f32, f32, f32, f32) -> f32 =
                        std::mem::transmute(self.func_ptr);
                    f(args[0], args[1], args[2], args[3], args[4])
                }
                6 => {
                    let f: extern "C" fn(f32, f32, f32, f32, f32, f32) -> f32 =
                        std::mem::transmute(self.func_ptr);
                    f(args[0], args[1], args[2], args[3], args[4], args[5])
                }
                _ => panic!("too many parameters (max 6)"),
            }
        }
    }
}

/// Declared math function IDs in the module
struct DeclaredMathFuncs {
    func_ids: HashMap<String, (FuncId, usize)>, // name -> (func_id, arity)
}

/// JIT compiler for expressions.
pub struct JitCompiler {
    builder: JITBuilder,
}

impl JitCompiler {
    /// Creates a new JIT compiler with math functions pre-registered.
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
    pub fn compile(
        self,
        ast: &Ast,
        params: &[&str],
        registry: &CraneliftRegistry,
    ) -> Result<CompiledFn, String> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        // Declare math functions in the module
        let declared = declare_math_funcs(&mut module)?;

        // Build function signature: (f32, f32, ...) -> f32
        let mut sig = module.make_signature();
        for _ in params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.returns.push(AbiParam::new(types::F32));

        // Declare the function
        let func_id = module
            .declare_function("expr", Linkage::Export, &sig)
            .map_err(|e| e.to_string())?;

        ctx.func.signature = sig;

        // Build the function body
        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Import math functions into this function
            let math = import_math_funcs(&mut builder, &mut module, &declared);

            // Map parameter names to values
            let mut var_map: HashMap<String, Value> = HashMap::new();
            for (i, name) in params.iter().enumerate() {
                let val = builder.block_params(entry_block)[i];
                var_map.insert(name.to_string(), val);
            }

            // Compile the expression
            let result = compile_ast(ast, &mut builder, &var_map, registry, &math)?;
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

fn import_math_funcs(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    declared: &DeclaredMathFuncs,
) -> MathFuncs {
    let mut funcs = HashMap::new();

    for (name, (func_id, _arity)) in &declared.func_ids {
        let func_ref = module.declare_func_in_func(*func_id, builder.func);
        funcs.insert(name.clone(), func_ref);
    }

    MathFuncs { funcs }
}

fn compile_ast(
    ast: &Ast,
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, Value>,
    registry: &CraneliftRegistry,
    math: &MathFuncs,
) -> Result<Value, String> {
    match ast {
        Ast::Num(n) => Ok(builder.ins().f32const(*n)),
        Ast::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| format!("unknown variable: {}", name)),
        Ast::BinOp(op, left, right) => {
            let l = compile_ast(left, builder, vars, registry, math)?;
            let r = compile_ast(right, builder, vars, registry, math)?;
            Ok(match op {
                BinOp::Add => builder.ins().fadd(l, r),
                BinOp::Sub => builder.ins().fsub(l, r),
                BinOp::Mul => builder.ins().fmul(l, r),
                BinOp::Div => builder.ins().fdiv(l, r),
                BinOp::Pow => {
                    // Call the registered pow function
                    let pow_ref = math.get("sap_pow").ok_or("pow function not available")?;
                    let call = builder.ins().call(pow_ref, &[l, r]);
                    builder.inst_results(call)[0]
                }
            })
        }
        Ast::UnaryOp(op, inner) => {
            let v = compile_ast(inner, builder, vars, registry, math)?;
            Ok(match op {
                UnaryOp::Neg => builder.ins().fneg(v),
            })
        }
        Ast::Call(name, args) => {
            let arg_vals: Vec<Value> = args
                .iter()
                .map(|a| compile_ast(a, builder, vars, registry, math))
                .collect::<Result<_, _>>()?;

            if let Some(func) = registry.get(name) {
                Ok(func.emit(builder, &arg_vals, math))
            } else {
                Err(format!("unknown function: {}", name))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sap_core::Expr;

    #[test]
    fn test_compile_number() {
        let expr = Expr::parse("42").unwrap();
        let registry = CraneliftRegistry::new();
        let jit = JitCompiler::new().unwrap();
        let func = jit.compile(expr.ast(), &[], &registry).unwrap();
        assert_eq!(func.call(&[]), 42.0);
    }

    #[test]
    fn test_compile_add() {
        let expr = Expr::parse("x + y").unwrap();
        let registry = CraneliftRegistry::new();
        let jit = JitCompiler::new().unwrap();
        let func = jit.compile(expr.ast(), &["x", "y"], &registry).unwrap();
        assert_eq!(func.call(&[3.0, 4.0]), 7.0);
    }

    #[test]
    fn test_compile_complex() {
        let expr = Expr::parse("x * 2 + y").unwrap();
        let registry = CraneliftRegistry::new();
        let jit = JitCompiler::new().unwrap();
        let func = jit.compile(expr.ast(), &["x", "y"], &registry).unwrap();
        assert_eq!(func.call(&[3.0, 1.0]), 7.0);
    }

    #[test]
    fn test_compile_negation() {
        let expr = Expr::parse("-x").unwrap();
        let registry = CraneliftRegistry::new();
        let jit = JitCompiler::new().unwrap();
        let func = jit.compile(expr.ast(), &["x"], &registry).unwrap();
        assert_eq!(func.call(&[5.0]), -5.0);
    }

    #[test]
    fn test_compile_pow() {
        let expr = Expr::parse("x ^ 2").unwrap();
        let registry = CraneliftRegistry::new();
        let jit = JitCompiler::new().unwrap();
        let func = jit.compile(expr.ast(), &["x"], &registry).unwrap();
        assert_eq!(func.call(&[3.0]), 9.0);
    }
}
