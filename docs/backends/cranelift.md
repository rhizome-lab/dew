# Cranelift Backend

JIT compile dew expressions to native code via Cranelift.

## Enable

```toml
rhizome-dew-scalar = { version = "0.1", features = ["cranelift"] }
rhizome-dew-linalg = { version = "0.1", features = ["cranelift"] }
```

## dew-scalar

### Compile and Execute

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::cranelift::ScalarJit;

// Create JIT compiler
let jit = ScalarJit::new().unwrap();

// Compile expression to native function
let expr = Expr::parse("sin(x) * cos(y) + z").unwrap();
let compiled = jit.compile(expr.ast(), &["x", "y", "z"]).unwrap();

// Call the compiled function
let result = compiled.call(&[1.0, 2.0, 3.0]);
println!("Result: {}", result);
```

### Reuse Compiled Functions

```rust
// Compile once
let compiled = jit.compile(expr.ast(), &["x"]).unwrap();

// Call many times (fast!)
for i in 0..1000 {
    let x = i as f32 * 0.001;
    let result = compiled.call(&[x]);
}
```

## dew-linalg

### Compile with Types

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_linalg::cranelift::LinalgJit;
use rhizome_dew_linalg::Type;

let jit = LinalgJit::new().unwrap();

let expr = Expr::parse("dot(a, b)").unwrap();
let var_types = vec![
    ("a", Type::Vec2),
    ("b", Type::Vec2),
];

let compiled = jit.compile(expr.ast(), &var_types).unwrap();

// Pass flattened arrays: [a.x, a.y, b.x, b.y]
let result = compiled.call(&[1.0, 0.0, 0.0, 1.0]);
// result = [0.0] (perpendicular vectors)
```

## How It Works

1. **Parse**: Expression string → AST
2. **Lower**: AST → Cranelift IR
3. **Compile**: IR → native machine code
4. **Execute**: Direct function call

Transcendental functions (sin, cos, exp, etc.) are implemented via callbacks to Rust's stdlib, while basic operations (add, mul, min, max) use native CPU instructions.

## Performance

- **First call**: ~1ms compile time
- **Subsequent calls**: Native speed
- **Best for**: Hot loops, real-time applications

## Use Cases

- Procedural generation in games
- Real-time audio/video processing
- Scientific computing
- Any hot path with dynamic expressions
