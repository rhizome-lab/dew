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

### Standard Library Backend Implementations

- [ ] Add WGSL std functions to sap-std (behind feature flag)
- [ ] Add Lua std functions to sap-std (behind feature flag)
- [ ] Add Cranelift std functions to sap-std (behind feature flag)

### Infrastructure

- [x] Set up CI tests for all backends
- [ ] Add integration tests
- [ ] Documentation examples
