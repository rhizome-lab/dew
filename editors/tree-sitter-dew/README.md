# tree-sitter-dew

Tree-sitter grammar for the Dew expression language.

## Usage

### Generate Parser

```bash
npm install
npm run generate
```

### Test

```bash
npm test
```

### Editor Integration

#### Neovim

Add to your nvim-treesitter config:

```lua
local parser_config = require("nvim-treesitter.parsers").get_parser_configs()
parser_config.dew = {
  install_info = {
    url = "https://github.com/rhi-zone/dew",
    files = { "editors/tree-sitter-dew/src/parser.c" },
    location = "editors/tree-sitter-dew",
  },
  filetype = "dew",
}
```

Then copy `queries/highlights.scm` to `~/.config/nvim/queries/dew/highlights.scm`.

#### Helix

Add to `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "dew"
scope = "source.dew"
file-types = ["dew"]
roots = []
comment-token = "//"

[[grammar]]
name = "dew"
source = { git = "https://github.com/rhi-zone/dew", subpath = "editors/tree-sitter-dew" }
```

## Syntax

```dew
// Basic arithmetic
x * 2 + y

// Function calls
sin(x) + cos(y)

// Conditionals
if x > 0 then sqrt(x) else 0

// Boolean logic
x > 0 and y < 10 or not z
```
