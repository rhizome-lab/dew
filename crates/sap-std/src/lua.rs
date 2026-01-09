//! Lua backend implementations for standard library functions.

use sap_lua::{LuaFn, LuaRegistry};

// ============================================================================
// Macro for simple Lua functions
// ============================================================================

macro_rules! lua_fn {
    ($name:ident, $str_name:literal, $lua_name:literal) => {
        pub struct $name;

        impl LuaFn for $name {
            fn name(&self) -> &str {
                $str_name
            }
            fn emit(&self, args: &[String]) -> String {
                format!("{}({})", $lua_name, args.join(", "))
            }
        }
    };
}

macro_rules! lua_const {
    ($name:ident, $str_name:literal, $lua_expr:literal) => {
        pub struct $name;

        impl LuaFn for $name {
            fn name(&self) -> &str {
                $str_name
            }
            fn emit(&self, _args: &[String]) -> String {
                $lua_expr.to_string()
            }
        }
    };
}

// ============================================================================
// Constants
// ============================================================================

lua_const!(Pi, "pi", "math.pi");
lua_const!(E, "e", "math.exp(1)"); // Lua has no math.e, compute e^1
lua_const!(Tau, "tau", "(2 * math.pi)");

// ============================================================================
// Trigonometric functions
// ============================================================================

lua_fn!(Sin, "sin", "math.sin");
lua_fn!(Cos, "cos", "math.cos");
lua_fn!(Tan, "tan", "math.tan");
lua_fn!(Asin, "asin", "math.asin");
lua_fn!(Acos, "acos", "math.acos");
lua_fn!(Atan, "atan", "math.atan");

/// atan2(y, x) in Lua
pub struct Atan2;
impl LuaFn for Atan2 {
    fn name(&self) -> &str {
        "atan2"
    }
    fn emit(&self, args: &[String]) -> String {
        // Lua 5.3+ uses math.atan(y, x), older uses math.atan2
        // mlua uses Lua 5.4, so math.atan with two args works
        format!(
            "math.atan({}, {})",
            args.first().map(|s| s.as_str()).unwrap_or("0"),
            args.get(1).map(|s| s.as_str()).unwrap_or("1")
        )
    }
}

/// sinh in Lua (not a builtin, must compute)
pub struct Sinh;
impl LuaFn for Sinh {
    fn name(&self) -> &str {
        "sinh"
    }
    fn emit(&self, args: &[String]) -> String {
        let x = args.first().map(|s| s.as_str()).unwrap_or("0");
        format!("((math.exp({x}) - math.exp(-{x})) / 2)")
    }
}

/// cosh in Lua
pub struct Cosh;
impl LuaFn for Cosh {
    fn name(&self) -> &str {
        "cosh"
    }
    fn emit(&self, args: &[String]) -> String {
        let x = args.first().map(|s| s.as_str()).unwrap_or("0");
        format!("((math.exp({x}) + math.exp(-{x})) / 2)")
    }
}

/// tanh in Lua
pub struct Tanh;
impl LuaFn for Tanh {
    fn name(&self) -> &str {
        "tanh"
    }
    fn emit(&self, args: &[String]) -> String {
        let x = args.first().map(|s| s.as_str()).unwrap_or("0");
        format!("((math.exp({x}) - math.exp(-{x})) / (math.exp({x}) + math.exp(-{x})))")
    }
}

// ============================================================================
// Exponential / logarithmic
// ============================================================================

lua_fn!(Exp, "exp", "math.exp");

/// exp2 in Lua: 2^x
pub struct Exp2;
impl LuaFn for Exp2 {
    fn name(&self) -> &str {
        "exp2"
    }
    fn emit(&self, args: &[String]) -> String {
        format!("(2 ^ {})", args.first().map(|s| s.as_str()).unwrap_or("0"))
    }
}

lua_fn!(Log, "log", "math.log");
lua_fn!(Ln, "ln", "math.log");

/// log2 in Lua: log(x) / log(2)
pub struct Log2;
impl LuaFn for Log2 {
    fn name(&self) -> &str {
        "log2"
    }
    fn emit(&self, args: &[String]) -> String {
        // Lua 5.2+ supports math.log(x, base)
        format!(
            "math.log({}, 2)",
            args.first().map(|s| s.as_str()).unwrap_or("1")
        )
    }
}

/// log10 in Lua
pub struct Log10;
impl LuaFn for Log10 {
    fn name(&self) -> &str {
        "log10"
    }
    fn emit(&self, args: &[String]) -> String {
        format!(
            "math.log({}, 10)",
            args.first().map(|s| s.as_str()).unwrap_or("1")
        )
    }
}

/// pow in Lua: uses ^ operator
pub struct Pow;
impl LuaFn for Pow {
    fn name(&self) -> &str {
        "pow"
    }
    fn emit(&self, args: &[String]) -> String {
        format!(
            "({} ^ {})",
            args.first().map(|s| s.as_str()).unwrap_or("0"),
            args.get(1).map(|s| s.as_str()).unwrap_or("1")
        )
    }
}

lua_fn!(Sqrt, "sqrt", "math.sqrt");

/// inversesqrt: 1 / sqrt(x)
pub struct InverseSqrt;
impl LuaFn for InverseSqrt {
    fn name(&self) -> &str {
        "inversesqrt"
    }
    fn emit(&self, args: &[String]) -> String {
        format!(
            "(1 / math.sqrt({}))",
            args.first().map(|s| s.as_str()).unwrap_or("1")
        )
    }
}

// ============================================================================
// Common math functions
// ============================================================================

lua_fn!(Abs, "abs", "math.abs");

/// sign in Lua (not a builtin)
pub struct Sign;
impl LuaFn for Sign {
    fn name(&self) -> &str {
        "sign"
    }
    fn emit(&self, args: &[String]) -> String {
        let x = args.first().map(|s| s.as_str()).unwrap_or("0");
        format!("(({x} > 0 and 1) or ({x} < 0 and -1) or 0)")
    }
}

lua_fn!(Floor, "floor", "math.floor");
lua_fn!(Ceil, "ceil", "math.ceil");

/// round in Lua (round half away from zero, matching Rust)
pub struct Round;
impl LuaFn for Round {
    fn name(&self) -> &str {
        "round"
    }
    fn emit(&self, args: &[String]) -> String {
        let x = args.first().map(|s| s.as_str()).unwrap_or("0");
        // Round half away from zero: positive uses floor(x+0.5), negative uses ceil(x-0.5)
        format!("(({x} >= 0) and math.floor({x} + 0.5) or math.ceil({x} - 0.5))")
    }
}

/// trunc in Lua
pub struct Trunc;
impl LuaFn for Trunc {
    fn name(&self) -> &str {
        "trunc"
    }
    fn emit(&self, args: &[String]) -> String {
        let x = args.first().map(|s| s.as_str()).unwrap_or("0");
        format!("(({x} >= 0) and math.floor({x}) or math.ceil({x}))")
    }
}

/// fract in Lua: x - floor(x)
pub struct Fract;
impl LuaFn for Fract {
    fn name(&self) -> &str {
        "fract"
    }
    fn emit(&self, args: &[String]) -> String {
        let x = args.first().map(|s| s.as_str()).unwrap_or("0");
        format!("({x} - math.floor({x}))")
    }
}

lua_fn!(Min, "min", "math.min");
lua_fn!(Max, "max", "math.max");

/// clamp in Lua
pub struct Clamp;
impl LuaFn for Clamp {
    fn name(&self) -> &str {
        "clamp"
    }
    fn emit(&self, args: &[String]) -> String {
        let x = args.first().map(|s| s.as_str()).unwrap_or("0");
        let lo = args.get(1).map(|s| s.as_str()).unwrap_or("0");
        let hi = args.get(2).map(|s| s.as_str()).unwrap_or("1");
        format!("math.max({lo}, math.min({hi}, {x}))")
    }
}

/// saturate: clamp(x, 0, 1)
pub struct Saturate;
impl LuaFn for Saturate {
    fn name(&self) -> &str {
        "saturate"
    }
    fn emit(&self, args: &[String]) -> String {
        let x = args.first().map(|s| s.as_str()).unwrap_or("0");
        format!("math.max(0, math.min(1, {x}))")
    }
}

// ============================================================================
// Interpolation
// ============================================================================

/// lerp: a + (b - a) * t
pub struct Lerp;
impl LuaFn for Lerp {
    fn name(&self) -> &str {
        "lerp"
    }
    fn emit(&self, args: &[String]) -> String {
        let a = args.first().map(|s| s.as_str()).unwrap_or("0");
        let b = args.get(1).map(|s| s.as_str()).unwrap_or("1");
        let t = args.get(2).map(|s| s.as_str()).unwrap_or("0");
        format!("({a} + ({b} - {a}) * {t})")
    }
}

/// mix: alias for lerp
pub struct Mix;
impl LuaFn for Mix {
    fn name(&self) -> &str {
        "mix"
    }
    fn emit(&self, args: &[String]) -> String {
        let a = args.first().map(|s| s.as_str()).unwrap_or("0");
        let b = args.get(1).map(|s| s.as_str()).unwrap_or("1");
        let t = args.get(2).map(|s| s.as_str()).unwrap_or("0");
        format!("({a} + ({b} - {a}) * {t})")
    }
}

/// step(edge, x) = x < edge ? 0 : 1
pub struct Step;
impl LuaFn for Step {
    fn name(&self) -> &str {
        "step"
    }
    fn emit(&self, args: &[String]) -> String {
        let edge = args.first().map(|s| s.as_str()).unwrap_or("0");
        let x = args.get(1).map(|s| s.as_str()).unwrap_or("0");
        format!("(({x} < {edge}) and 0 or 1)")
    }
}

/// smoothstep(edge0, edge1, x)
pub struct Smoothstep;
impl LuaFn for Smoothstep {
    fn name(&self) -> &str {
        "smoothstep"
    }
    fn emit(&self, args: &[String]) -> String {
        let e0 = args.first().map(|s| s.as_str()).unwrap_or("0");
        let e1 = args.get(1).map(|s| s.as_str()).unwrap_or("1");
        let x = args.get(2).map(|s| s.as_str()).unwrap_or("0");
        // t = clamp((x - e0) / (e1 - e0), 0, 1); t * t * (3 - 2 * t)
        format!(
            "(function() local t = math.max(0, math.min(1, ({x} - {e0}) / ({e1} - {e0}))); return t * t * (3 - 2 * t) end)()"
        )
    }
}

/// inverse_lerp: (v - a) / (b - a)
pub struct InverseLerp;
impl LuaFn for InverseLerp {
    fn name(&self) -> &str {
        "inverse_lerp"
    }
    fn emit(&self, args: &[String]) -> String {
        let a = args.first().map(|s| s.as_str()).unwrap_or("0");
        let b = args.get(1).map(|s| s.as_str()).unwrap_or("1");
        let v = args.get(2).map(|s| s.as_str()).unwrap_or("0");
        format!("(({v} - {a}) / ({b} - {a}))")
    }
}

/// remap: out_lo + (out_hi - out_lo) * ((x - in_lo) / (in_hi - in_lo))
pub struct Remap;
impl LuaFn for Remap {
    fn name(&self) -> &str {
        "remap"
    }
    fn emit(&self, args: &[String]) -> String {
        let x = args.first().map(|s| s.as_str()).unwrap_or("0");
        let in_lo = args.get(1).map(|s| s.as_str()).unwrap_or("0");
        let in_hi = args.get(2).map(|s| s.as_str()).unwrap_or("1");
        let out_lo = args.get(3).map(|s| s.as_str()).unwrap_or("0");
        let out_hi = args.get(4).map(|s| s.as_str()).unwrap_or("1");
        format!("({out_lo} + ({out_hi} - {out_lo}) * (({x} - {in_lo}) / ({in_hi} - {in_lo})))")
    }
}

// ============================================================================
// Registry
// ============================================================================

/// Registers all standard functions for Lua backend.
pub fn register_lua(registry: &mut LuaRegistry) {
    // Constants
    registry.register(Pi);
    registry.register(E);
    registry.register(Tau);

    // Trigonometric
    registry.register(Sin);
    registry.register(Cos);
    registry.register(Tan);
    registry.register(Asin);
    registry.register(Acos);
    registry.register(Atan);
    registry.register(Atan2);
    registry.register(Sinh);
    registry.register(Cosh);
    registry.register(Tanh);

    // Exponential / logarithmic
    registry.register(Exp);
    registry.register(Exp2);
    registry.register(Log);
    registry.register(Ln);
    registry.register(Log2);
    registry.register(Log10);
    registry.register(Pow);
    registry.register(Sqrt);
    registry.register(InverseSqrt);

    // Common math
    registry.register(Abs);
    registry.register(Sign);
    registry.register(Floor);
    registry.register(Ceil);
    registry.register(Round);
    registry.register(Trunc);
    registry.register(Fract);
    registry.register(Min);
    registry.register(Max);
    registry.register(Clamp);
    registry.register(Saturate);

    // Interpolation
    registry.register(Lerp);
    registry.register(Mix);
    registry.register(Step);
    registry.register(Smoothstep);
    registry.register(InverseLerp);
    registry.register(Remap);
}

/// Creates a new registry with all standard Lua functions.
pub fn lua_std_registry() -> LuaRegistry {
    let mut registry = LuaRegistry::new();
    register_lua(&mut registry);
    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use sap_core::Expr;
    use sap_lua::{eval_with_registry, to_lua};
    use std::collections::HashMap;

    fn compile(input: &str) -> String {
        let expr = Expr::parse(input).unwrap();
        let registry = lua_std_registry();
        to_lua(expr.ast(), &registry)
    }

    fn eval(input: &str, vars: &[(&str, f32)]) -> f32 {
        let expr = Expr::parse(input).unwrap();
        let registry = lua_std_registry();
        let var_map: HashMap<String, f32> = vars.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        eval_with_registry(expr.ast(), &var_map, &registry).unwrap()
    }

    #[test]
    fn test_constants() {
        assert!((eval("pi()", &[]) - std::f32::consts::PI).abs() < 0.001);
        assert!((eval("e()", &[]) - std::f32::consts::E).abs() < 0.001);
        assert!((eval("tau()", &[]) - std::f32::consts::TAU).abs() < 0.001);
    }

    #[test]
    fn test_trig() {
        assert!(eval("sin(0)", &[]).abs() < 0.001);
        assert!((eval("cos(0)", &[]) - 1.0).abs() < 0.001);
        assert!((eval("atan2(1, 1)", &[]) - std::f32::consts::FRAC_PI_4).abs() < 0.001);
    }

    #[test]
    fn test_hyperbolic() {
        assert!(eval("sinh(0)", &[]).abs() < 0.001);
        assert!((eval("cosh(0)", &[]) - 1.0).abs() < 0.001);
        assert!(eval("tanh(0)", &[]).abs() < 0.001);
    }

    #[test]
    fn test_exp_log() {
        assert!((eval("exp(0)", &[]) - 1.0).abs() < 0.001);
        assert!((eval("exp2(3)", &[]) - 8.0).abs() < 0.001);
        assert!(eval("ln(1)", &[]).abs() < 0.001);
        assert!((eval("log2(8)", &[]) - 3.0).abs() < 0.001);
        assert!((eval("log10(100)", &[]) - 2.0).abs() < 0.001);
        assert!((eval("pow(2, 3)", &[]) - 8.0).abs() < 0.001);
        assert!((eval("sqrt(16)", &[]) - 4.0).abs() < 0.001);
        assert!((eval("inversesqrt(4)", &[]) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_common() {
        assert_eq!(eval("abs(-5)", &[]), 5.0);
        assert_eq!(eval("sign(-3)", &[]), -1.0);
        assert_eq!(eval("sign(3)", &[]), 1.0);
        assert_eq!(eval("sign(0)", &[]), 0.0);
        assert_eq!(eval("floor(3.7)", &[]), 3.0);
        assert_eq!(eval("ceil(3.2)", &[]), 4.0);
        assert_eq!(eval("round(3.5)", &[]), 4.0);
        assert_eq!(eval("trunc(3.7)", &[]), 3.0);
        assert_eq!(eval("trunc(-3.7)", &[]), -3.0);
        assert!((eval("fract(3.7)", &[]) - 0.7).abs() < 0.001);
        assert_eq!(eval("min(3, 7)", &[]), 3.0);
        assert_eq!(eval("max(3, 7)", &[]), 7.0);
        assert_eq!(eval("clamp(5, 0, 3)", &[]), 3.0);
        assert_eq!(eval("saturate(1.5)", &[]), 1.0);
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(eval("lerp(0, 10, 0.5)", &[]), 5.0);
        assert_eq!(eval("mix(0, 10, 0.5)", &[]), 5.0);
        assert_eq!(eval("step(0.5, 0.3)", &[]), 0.0);
        assert_eq!(eval("step(0.5, 0.7)", &[]), 1.0);
        assert!((eval("smoothstep(0, 1, 0.5)", &[]) - 0.5).abs() < 0.1);
        assert_eq!(eval("inverse_lerp(0, 10, 5)", &[]), 0.5);
    }

    #[test]
    fn test_remap() {
        assert_eq!(eval("remap(5, 0, 10, 0, 100)", &[]), 50.0);
    }

    #[test]
    fn test_code_generation() {
        assert_eq!(compile("sin(x)"), "math.sin(x)");
        assert!(compile("pi()").contains("math.pi"));
        assert!(compile("pow(x, 2)").contains("^"));
    }
}
