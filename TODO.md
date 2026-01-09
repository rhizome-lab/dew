# TODO

## Backlog

### Port from resin-expr

- [x] Port `resin-expr` core expression types and AST
- [x] Port `resin-expr-std` standard library functions (excluding noise)

### Core

- [x] Define expression AST (functions, numeric values)
- [ ] Implement type system for expressions
- [ ] Add expression validation/normalization

### Backends

- [x] WGSL code generation (sap-wgsl)
- [x] Cranelift JIT compilation (sap-cranelift)
- [x] Lua code generation (sap-lua, mlua)

#### Cranelift: pow support

Currently returns error. Options:
1. **Link libm** - call `powf` via Cranelift external function linkage
2. **Self-implement** - `pow(a,b) = exp(b * ln(a))` (still needs exp/ln from libm)
3. **Integer powers only** - repeated multiplication for small integer exponents
4. **Decompose in sap-core** - transform `a^b` to `pow(a,b)` call, let registry handle it

### Standard Library Backend Implementations

- [ ] Add WGSL std functions to sap-std (behind feature flag)
- [ ] Add Lua std functions to sap-std (behind feature flag)
- [ ] Add Cranelift std functions to sap-std (behind feature flag)

### Infrastructure

- [x] Set up CI tests for all backends
- [ ] Add integration tests
- [ ] Documentation examples
