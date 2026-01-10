; Keywords
"if" @keyword.conditional
"then" @keyword.conditional
"else" @keyword.conditional
"and" @keyword.operator
"or" @keyword.operator
"not" @keyword.operator

; Operators
[
  "+"
  "-"
  "*"
  "/"
  "%"
  "^"
] @operator

[
  "<"
  "<="
  ">"
  ">="
  "=="
  "!="
] @operator

; Punctuation
"(" @punctuation.bracket
")" @punctuation.bracket
"," @punctuation.delimiter

; Literals
(number) @number

; Functions
(call_expression
  function: (identifier) @function)

; Variables
(identifier) @variable

; Comments
(comment) @comment
