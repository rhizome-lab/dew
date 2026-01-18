# dew-core

The foundational crate providing the AST, parser, and base types for dew expressions.

## Installation

```toml
[dependencies]
rhizome-dew-core = "0.1"

# Enable optional features
rhizome-dew-core = { version = "0.1", features = ["cond", "func"] }
```

## Features

dew-core provides opt-in features to manage complexity:

| Feature | Description |
|---------|-------------|
| (none)  | Basic expressions: numbers, variables, arithmetic, power |
| `cond`  | Conditionals: `if`/`then`/`else`, comparisons, boolean logic |
| `func`  | Function calls: `name(args...)`, `FunctionRegistry` |
| `optimize` | Expression optimization passes (constant folding, algebraic simplification) |

## Basic Usage

Without any features, you get a minimal expression language:

```rust
use rhizome_dew_core::Expr;
use std::collections::HashMap;

// Parse an arithmetic expression
let expr = Expr::parse("x * 2 + y").unwrap();

// Evaluate with variables
let vars: HashMap<String, f32> = [
    ("x".into(), 3.0),
    ("y".into(), 1.0),
].into();
let result = expr.eval(&vars).unwrap();
assert_eq!(result, 7.0);
```

## Expression Syntax

### Numbers

Numeric literals with optional decimal and exponent:

```
42
3.14
1e-5
2.5e10
```

### Variables

Alphanumeric identifiers starting with a letter:

```
x
velocity
myVar_1
```

### Operators

| Operator | Precedence | Description |
|----------|------------|-------------|
| `^`      | 3 (highest)| Power (right-associative) |
| `*` `/` `%` | 2       | Multiplication, division, modulo |
| `+` `-`  | 1          | Addition, subtraction |
| `<<` `>>` | 0         | Bit shift left/right |
| `&`      | -1         | Bitwise AND |
| `\|`     | -2         | Bitwise OR |
| `-x`     | unary      | Negation |
| `~x`     | unary      | Bitwise NOT |

Note: Modulo (`%`) and bitwise operators work with integer types (i32, i64).

### Parentheses

Group expressions for explicit precedence:

```
(x + y) * z
-(a + b)
```

### Let Bindings

Local variable bindings for naming intermediate values:

```
let name = value; body
```

The bound variable is only visible in the body expression. Bindings can be chained:

```
let a = x * 2; let b = a + 1; b * b
```

Example:
```rust
use rhizome_dew_core::Expr;
use std::collections::HashMap;

let expr = Expr::parse("let t = x * 2 + 1; sin(t) * cos(t)").unwrap();
let vars: HashMap<String, f32> = [("x".into(), 1.0)].into();
// t is computed once and used twice
```

Let bindings:
- Evaluate the value expression first
- Bind the result to the name in scope
- Evaluate and return the body expression
- Support shadowing (inner bindings hide outer ones)

**Backend note:** WGSL and GLSL backends do not support `let` expressions directly. Use the optimizer to inline let bindings before codegen. Lua and Cranelift fully support `let`.

## Conditionals (feature = "cond")

Enable with `features = ["cond"]`:

```rust
use rhizome_dew_core::Expr;
use std::collections::HashMap;

let expr = Expr::parse("if x > 0 then x else -x").unwrap();
let vars: HashMap<String, f32> = [("x".into(), -5.0)].into();
let result = expr.eval(&vars).unwrap();
assert_eq!(result, 5.0); // abs(x)
```

### Comparison Operators

| Operator | Description |
|----------|-------------|
| `<`      | Less than |
| `<=`     | Less than or equal |
| `>`      | Greater than |
| `>=`     | Greater than or equal |
| `==`     | Equal |
| `!=`     | Not equal |

### Boolean Logic

| Operator | Description |
|----------|-------------|
| `and`    | Logical AND (short-circuit) |
| `or`     | Logical OR (short-circuit) |
| `not`    | Logical NOT |

### If-Then-Else

```
if condition then value_if_true else value_if_false
```

Boolean results are represented as scalars: 0.0 = false, 1.0 = true.

## Function Calls (feature = "func")

Enable with `features = ["func"]`:

```rust
use rhizome_dew_core::{Expr, ExprFn, FunctionRegistry};
use std::collections::HashMap;

// Define a custom function
struct Double;
impl ExprFn for Double {
    fn name(&self) -> &str { "double" }
    fn arg_count(&self) -> usize { 1 }
    fn call(&self, args: &[f32]) -> f32 { args[0] * 2.0 }
}

// Create registry and register function
let mut registry = FunctionRegistry::new();
registry.register(Double);

// Parse and evaluate
let expr = Expr::parse("double(x) + 1").unwrap();
let vars: HashMap<String, f32> = [("x".into(), 5.0)].into();
let result = expr.eval(&vars, &registry).unwrap();
assert_eq!(result, 11.0);
```

Function call syntax:
```
name()           # Zero arguments
name(arg)        # One argument
name(a, b, c)    # Multiple arguments
```

## AST Types

The core AST is defined by the `Ast` enum:

```rust
pub enum Ast {
    Num(f64),                           // Numeric literal
    Var(String),                        // Variable reference
    BinOp(BinOp, Box<Ast>, Box<Ast>),   // Binary operation
    UnaryOp(UnaryOp, Box<Ast>),         // Unary operation
    Let { name, value, body },          // Local binding

    // With "cond" feature:
    Compare(CompareOp, Box<Ast>, Box<Ast>), // Comparison
    And(Box<Ast>, Box<Ast>),                // Logical AND
    Or(Box<Ast>, Box<Ast>),                 // Logical OR
    If(Box<Ast>, Box<Ast>, Box<Ast>),       // Conditional

    // With "func" feature:
    Call(String, Vec<Ast>),                 // Function call
}
```

## Error Handling

Parse errors provide location and context:

```rust
let result = Expr::parse("x + + y");
// ParseError at position 4: unexpected token
```

Eval errors identify missing variables:

```rust
let result = expr.eval(&empty_vars);
// EvalError::UnknownVariable("x")
```

## Optimization (feature = "optimize")

Enable with `features = ["optimize"]`:

```rust
use rhizome_dew_core::{Expr, Ast};
use rhizome_dew_core::optimize::{optimize, standard_passes};

let expr = Expr::parse("(1 + 2) * x + 0").unwrap();
let optimized = optimize(expr.ast().clone(), &standard_passes());

// (1 + 2) folded to 3, + 0 eliminated
assert_eq!(optimized.to_string(), "(3 * x)");
```

Available passes:
- **ConstantFolding**: Evaluates `1 + 2` → `3`
- **AlgebraicIdentities**: Eliminates `x * 1`, `x + 0`, etc.
- **PowerReduction**: Converts `x ^ 2` → `x * x`
- **FunctionDecomposition**: Uses `ExprFn::decompose()` to expand functions

See the [Optimization Guide](/optimization) for details.

## Combining with Domain Crates

dew-core is syntax-only. Domain crates add semantics:

- **dew-scalar**: Scalar math functions (sin, cos, exp, etc.)
- **dew-linalg**: Linear algebra (Vec2, Vec3, Mat2, Mat3)
- **dew-complex**: Complex numbers
- **dew-quaternion**: Quaternions for 3D rotation

All domain crates require the `func` feature (they depend on function calls).
