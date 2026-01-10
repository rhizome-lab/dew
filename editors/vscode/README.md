# Dew Language for VSCode

Syntax highlighting for the Dew expression language.

## Installation

### From Source

1. Copy this folder to `~/.vscode/extensions/dew-lang`
2. Restart VSCode

### Development

```bash
cd editors/vscode
code --extensionDevelopmentPath=$(pwd)
```

## Features

- Syntax highlighting for `.dew` files
- Bracket matching
- Comment toggling (`//`)

## Syntax

```dew
// Basic arithmetic
x * 2 + y

// Function calls
sin(x) + cos(y)

// Conditionals (with cond feature)
if x > 0 then sqrt(x) else 0

// Boolean logic
x > 0 and y < 10 or not z
```
