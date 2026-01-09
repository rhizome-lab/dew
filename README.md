# Sap

Expression language for procedural generation.

Part of the [Rhizome](https://rhizome-lab.github.io) ecosystem.

## Overview

Sap is a domain-specific expression language designed for procedural content generation. It provides a composable way to define generation rules that can be compiled to multiple backends.

## Crates

| Crate | Description |
|-------|-------------|
| `rhizome-sap-core` | Core types and AST |
| `rhizome-sap-std` | Standard library functions |
| `rhizome-sap-lua` | Lua backend |
| `rhizome-sap-cranelift` | Cranelift JIT backend |
| `rhizome-sap-wgsl` | WGSL shader backend |
| `rhizome-sap-linalg` | Linear algebra operations |

## License

MIT
