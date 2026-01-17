# Expression Optimization

This document describes dew's expression optimization system: design decisions, available passes, and how to extend it.

## Overview

Dew provides optional expression optimization that transforms ASTs before evaluation or code generation. Optimizations reduce runtime computation by simplifying expressions at compile time.

```rust
use rhizome_dew_core::{Expr, optimize};

let expr = Expr::parse("x * 1 + 0").unwrap();
let optimized = optimize(expr.ast().clone());
// Result: Ast::Var("x") — the identity operations were eliminated
```

## Design Decisions

### Why Optimize in Core?

Originally (see TODO.md), optimization was planned for domain crates. We reconsidered because:

1. **Most passes are domain-agnostic**: Constant folding, algebraic identities, and power reduction work on any numeric AST regardless of whether you're doing scalar math or linear algebra.

2. **Shared infrastructure**: The traversal machinery and pass trait are useful for all domains.

3. **Composability**: If optimization lived only in domain crates, users combining multiple domains would need to run multiple optimizers or choose one.

**Decision**: Core provides the optimization infrastructure and numeric passes. Domain crates can register additional passes for domain-specific optimizations (e.g., `dot(v, v)` → `length_squared(v)`).

### Pass Architecture

We chose a **functional transformation** model over a visitor-mutation model:

```rust
pub trait Pass: Send + Sync {
    /// Transform an AST node. Returns Some(new) if changed, None to keep original.
    fn transform(&self, ast: &Ast) -> Option<Ast>;
}
```

Rationale:
- **Immutability**: Passes return new ASTs rather than mutating. Easier to reason about, enables caching.
- **Composability**: Passes are independent; order matters but passes don't share state.
- **Testability**: Each pass can be tested in isolation.

### Fixed-Point Iteration

Optimization runs passes repeatedly until the AST stops changing:

```rust
pub fn optimize(ast: Ast, passes: &[&dyn Pass]) -> Ast {
    let mut current = ast;
    loop {
        let transformed = apply_passes_once(&current, passes);
        if transformed == current {
            break;
        }
        current = transformed;
    }
    current
}
```

This handles cases where one optimization enables another. For example:
1. `(2 + 3) * x` → constant fold → `5 * x`
2. If later we add `5 * x` → `x + x + x + x + x` (hypothetically), we'd need another pass.

Fixed-point ensures all optimizations are fully applied.

### Bottom-Up Traversal

Passes are applied **bottom-up** (children first, then parents):

```
    *
   / \
  +   x
 / \
2   3
```

1. Visit `2` and `3` (leaves, no change)
2. Visit `+` with children `2`, `3` → constant fold → `5`
3. Visit `*` with children `5`, `x` → no change (can't fold)

Result: `5 * x`

Bottom-up ensures that when a pass sees a node, its children are already optimized.

### CSE: Backend Responsibility

Common Subexpression Elimination (CSE) is **not** an AST transformation pass. Here's why:

**Problem**: The AST has no `let` bindings. To express CSE at the AST level, we'd need:
```rust
// Would need new AST variant:
Let { name: String, value: Box<Ast>, body: Box<Ast> }
```

This changes the AST structure and affects all backends.

**Alternative**: Backends can implement CSE during code generation:
1. Hash each subexpression as it's generated
2. If a subexpression was already emitted, reuse the temporary variable
3. Generate: `let _t1 = sin(x); ... _t1 * _t1 ...` instead of `sin(x) * sin(x)`

**Decision**: Core provides `AstHasher` utility for hashing ASTs (handles f32 → bits conversion). Backends use this to implement CSE during emit. This keeps the AST simple and gives backends control over naming and scoping.

### Feature Flag

Optimization is behind the `optimize` feature:

```toml
[dependencies]
rhizome-dew-core = { version = "...", features = ["optimize"] }
```

Rationale:
- Not everyone needs optimization (interpreted use cases, simple expressions)
- Keeps compile time down for minimal builds
- Opt-in complexity (dew philosophy)

## Available Passes

### ConstantFolding

Evaluates operations on numeric literals at compile time.

| Before | After |
|--------|-------|
| `1 + 2` | `3` |
| `2 * 3 * 4` | `24` |
| `-(-5)` | `5` |
| `2 ^ 3` | `8` |

**Does not fold** function calls (would require function registry at optimization time).

### AlgebraicIdentities

Eliminates identity operations and absorbing elements.

| Before | After | Rule |
|--------|-------|------|
| `x * 1` | `x` | multiplicative identity |
| `1 * x` | `x` | multiplicative identity |
| `x * 0` | `0` | zero absorbs |
| `0 * x` | `0` | zero absorbs |
| `x + 0` | `x` | additive identity |
| `0 + x` | `x` | additive identity |
| `x - 0` | `x` | subtractive identity |
| `x / 1` | `x` | divisive identity |
| `x ^ 1` | `x` | power identity |
| `x ^ 0` | `1` | zero power (x ≠ 0 assumed) |
| `x - x` | `0` | self-subtraction |
| `x / x` | `1` | self-division (x ≠ 0 assumed) |
| `--x` | `x` | double negation |

**Note**: `x - x` and `x / x` only apply when both operands are structurally identical (same AST). `a - b` where a and b happen to have the same runtime value is not optimized (would require symbolic reasoning).

### PowerReduction

Converts small integer powers to repeated multiplication.

| Before | After | Benefit |
|--------|-------|---------|
| `x ^ 2` | `x * x` | Avoids `powf` call |
| `x ^ 3` | `x * x * x` | Avoids `powf` call |
| `x ^ 4` | `x * x * x * x` | Avoids `powf` call |

**Limit**: Only reduces powers ≤ 4. Higher powers stay as `powf` calls to avoid code bloat.

**Caveat**: This introduces multiple evaluations of `x`. If `x` is a complex subexpression, backends should apply CSE to avoid redundant computation.

### FunctionDecomposition

Uses `ExprFn::decompose()` to expand functions into simpler operations.

Example: A `log10` function might decompose to `log(x) / log(10)`:

```rust
impl ExprFn for Log10 {
    fn decompose(&self, args: &[Ast]) -> Option<Ast> {
        Some(Ast::BinOp(
            BinOp::Div,
            Box::new(Ast::Call("log".into(), args.to_vec())),
            Box::new(Ast::Call("log".into(), vec![Ast::Num(10.0)])),
        ))
    }
}
```

**Requires**: `func` feature enabled in dew-core.

**Use case**: Backends that don't support a function natively can still work if the function decomposes to supported operations.

## Domain Extensions

Domain crates provide additional optimization passes for their types:

### dew-scalar (feature = "optimize")

`ScalarConstantFolding` evaluates scalar functions at compile time:

```rust
use rhizome_dew_core::optimize::{optimize, standard_passes};
use rhizome_dew_scalar::optimize::ScalarConstantFolding;

let mut passes = standard_passes();
passes.push(&ScalarConstantFolding);

// sin(0) + cos(0) → 0 + 1 → 1
```

Supported: all standard math functions (sin, cos, sqrt, exp, log, etc.)

### dew-linalg (feature = "optimize")

`LinalgConstantFolding` evaluates vector operations at compile time:

```rust
use rhizome_dew_core::optimize::{optimize, standard_passes};
use rhizome_dew_linalg::optimize::LinalgConstantFolding;

let mut passes = standard_passes();
passes.push(&LinalgConstantFolding);

// vec2(1, 2) + vec2(3, 4) → vec2(4, 6)
// dot(vec2(1, 0), vec2(0, 1)) → 0
// length(vec2(3, 4)) → 5
```

**Current limitation**: Uses constructor-based type inference. Only folds operations
where vector types are visible in the AST (e.g., `vec2(...)` calls). Operations on
typed variables like `a + b` where `a, b: Vec3` are not yet optimized.

### dew-complex (feature = "optimize")

`ComplexConstantFolding` evaluates complex number operations at compile time:

```rust
use rhizome_dew_core::optimize::{optimize, standard_passes};
use rhizome_dew_complex::optimize::ComplexConstantFolding;

let mut passes = standard_passes();
passes.push(&ComplexConstantFolding);

// abs(complex(3, 4)) → 5
// complex(1, 2) + complex(3, 4) → complex(4, 6)
// re(complex(3, 4)) → 3
```

Uses `complex(re, im)` or `polar(r, theta)` as constructors for complex values.
Supports: re, im, conj, abs, arg, norm, exp, log, sqrt, pow.

### dew-quaternion (feature = "optimize")

`QuaternionConstantFolding` evaluates quaternion and vector operations at compile time:

```rust
use rhizome_dew_core::optimize::{optimize, standard_passes};
use rhizome_dew_quaternion::optimize::QuaternionConstantFolding;

let mut passes = standard_passes();
passes.push(&QuaternionConstantFolding);

// length(vec3(3, 4, 0)) → 5
// dot(vec3(1, 0, 0), vec3(0, 1, 0)) → 0
// normalize(quat(0, 0, 0, 2)) → quat(0, 0, 0, 1)
// conj(quat(1, 2, 3, 4)) → quat(-1, -2, -3, 4)
```

Uses `vec3(x, y, z)` and `quat(x, y, z, w)` as constructors. Supports: length, dot,
normalize, conj, inverse, lerp, slerp, axis_angle, rotate.

### Custom Passes

Implement the `Pass` trait for domain-specific optimizations:

```rust
use rhizome_dew_core::optimize::Pass;
use rhizome_dew_core::Ast;

pub struct MyDomainOptimizations;

impl Pass for MyDomainOptimizations {
    fn transform(&self, ast: &Ast) -> Option<Ast> {
        // Your optimization logic here
        None
    }
}
```

## CSE Utilities

For backends implementing CSE, core provides `AstHasher`:

```rust
use rhizome_dew_core::optimize::AstHasher;
use std::collections::HashMap;

struct EmitContext {
    seen: HashMap<u64, String>,  // hash → temp var name
    counter: usize,
}

impl EmitContext {
    fn emit_with_cse(&mut self, ast: &Ast) -> String {
        let hash = AstHasher::hash(ast);

        if let Some(var) = self.seen.get(&hash) {
            return var.clone();  // Reuse existing temp
        }

        let code = self.emit_inner(ast);

        // If subexpression is non-trivial, extract to temp
        if !matches!(ast, Ast::Num(_) | Ast::Var(_)) {
            let temp = format!("_t{}", self.counter);
            self.counter += 1;
            self.seen.insert(hash, temp.clone());
            format!("let {} = {};\n{}", temp, code, temp)
        } else {
            code
        }
    }
}
```

**Hashing f32**: `AstHasher` converts `f32` to bits (`u32`) for hashing. NaN values with the same bit pattern hash equally; different NaN bit patterns hash differently. This matches the behavior most users expect.

## Performance Considerations

### When to Optimize

- **JIT compilation**: Optimize before Cranelift codegen. Pays off across many evaluations.
- **Shader generation**: Optimize before WGSL emit. GPU drivers may optimize further, but reducing IR helps.
- **Interpreted eval**: Usually skip optimization. Overhead may exceed benefit for one-shot evaluation.

### Optimization Cost

Fixed-point iteration means worst-case O(n × p × d) where:
- n = AST node count
- p = pass count
- d = maximum depth of cascading optimizations

In practice, d is small (usually 1-2 iterations suffice).

### Caching

If the same expression is optimized repeatedly:
```rust
use std::collections::HashMap;

struct OptimizedCache {
    cache: HashMap<String, Ast>,  // source → optimized AST
}

impl OptimizedCache {
    fn get_or_optimize(&mut self, source: &str) -> &Ast {
        self.cache.entry(source.to_string()).or_insert_with(|| {
            let expr = Expr::parse(source).unwrap();
            optimize(expr.ast().clone(), &standard_passes())
        })
    }
}
```

## Future Work

Potential optimizations not yet implemented:

1. **Reassociation**: `(a + b) + c` → `a + (b + c)` when one form is more efficient
2. **Strength reduction**: `x * 2` → `x + x` (debatable benefit on modern CPUs)
3. **Dead code elimination**: Remove unreachable branches in conditionals
4. **Symbolic simplification**: `sin(x)^2 + cos(x)^2` → `1` (requires pattern matching library)

These are deferred until profiling shows they'd help real workloads.
