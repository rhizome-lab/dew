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
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use sap_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

// ============================================================================
// Cranelift Function Registry
// ============================================================================

/// A function that can be JIT compiled.
pub trait CraneliftFn: Send + Sync {
    /// Function name in the expression language.
    fn name(&self) -> &str;

    /// Emit Cranelift IR for this function call.
    fn emit(&self, builder: &mut FunctionBuilder, args: &[Value]) -> Value;
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
                _ => panic!("too many parameters (max 4)"),
            }
        }
    }
}

/// JIT compiler for expressions.
pub struct JitCompiler {
    builder: JITBuilder,
}

impl JitCompiler {
    /// Creates a new JIT compiler.
    pub fn new() -> Result<Self, String> {
        let builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|e| e.to_string())?;
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

            // Map parameter names to values
            let mut var_map: HashMap<String, Value> = HashMap::new();
            for (i, name) in params.iter().enumerate() {
                let val = builder.block_params(entry_block)[i];
                var_map.insert(name.to_string(), val);
            }

            // Compile the expression
            let result = compile_ast(ast, &mut builder, &var_map, registry)?;
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

fn compile_ast(
    ast: &Ast,
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, Value>,
    registry: &CraneliftRegistry,
) -> Result<Value, String> {
    match ast {
        Ast::Num(n) => Ok(builder.ins().f32const(*n)),
        Ast::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| format!("unknown variable: {}", name)),
        Ast::BinOp(op, left, right) => {
            let l = compile_ast(left, builder, vars, registry)?;
            let r = compile_ast(right, builder, vars, registry)?;
            Ok(match op {
                BinOp::Add => builder.ins().fadd(l, r),
                BinOp::Sub => builder.ins().fsub(l, r),
                BinOp::Mul => builder.ins().fmul(l, r),
                BinOp::Div => builder.ins().fdiv(l, r),
                BinOp::Pow => {
                    // No native pow - would need libm call
                    // For now, approximate with exp(r * ln(l)) for positive l
                    // This is a simplification; real impl would call libm
                    return Err("pow not yet implemented in cranelift backend".to_string());
                }
            })
        }
        Ast::UnaryOp(op, inner) => {
            let v = compile_ast(inner, builder, vars, registry)?;
            Ok(match op {
                UnaryOp::Neg => builder.ins().fneg(v),
            })
        }
        Ast::Call(name, args) => {
            let arg_vals: Vec<Value> = args
                .iter()
                .map(|a| compile_ast(a, builder, vars, registry))
                .collect::<Result<_, _>>()?;

            if let Some(func) = registry.get(name) {
                Ok(func.emit(builder, &arg_vals))
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
}
