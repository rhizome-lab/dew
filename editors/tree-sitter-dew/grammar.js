/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

module.exports = grammar({
  name: 'dew',

  extras: $ => [
    /\s/,
    $.comment,
  ],

  precedences: $ => [
    [
      'unary',
      'power',
      'multiplicative',
      'additive',
      'comparison',
      'and',
      'or',
      'conditional',
    ],
  ],

  rules: {
    source_file: $ => $.expression,

    expression: $ => choice(
      $.conditional,
      $.binary_expression,
      $.unary_expression,
      $.call_expression,
      $.parenthesized_expression,
      $.number,
      $.identifier,
    ),

    conditional: $ => prec.right('conditional', seq(
      'if',
      field('condition', $.expression),
      'then',
      field('consequence', $.expression),
      'else',
      field('alternative', $.expression),
    )),

    binary_expression: $ => choice(
      prec.left('or', seq(
        field('left', $.expression),
        field('operator', 'or'),
        field('right', $.expression),
      )),
      prec.left('and', seq(
        field('left', $.expression),
        field('operator', 'and'),
        field('right', $.expression),
      )),
      prec.left('comparison', seq(
        field('left', $.expression),
        field('operator', choice('<', '<=', '>', '>=', '==', '!=')),
        field('right', $.expression),
      )),
      prec.left('additive', seq(
        field('left', $.expression),
        field('operator', choice('+', '-')),
        field('right', $.expression),
      )),
      prec.left('multiplicative', seq(
        field('left', $.expression),
        field('operator', choice('*', '/', '%')),
        field('right', $.expression),
      )),
      prec.right('power', seq(
        field('left', $.expression),
        field('operator', '^'),
        field('right', $.expression),
      )),
    ),

    unary_expression: $ => prec('unary', choice(
      seq('-', $.expression),
      seq('not', $.expression),
    )),

    call_expression: $ => seq(
      field('function', $.identifier),
      '(',
      optional(seq(
        $.expression,
        repeat(seq(',', $.expression)),
        optional(','),
      )),
      ')',
    ),

    parenthesized_expression: $ => seq('(', $.expression, ')'),

    number: $ => token(choice(
      /\d+\.\d*([eE][+-]?\d+)?/,
      /\d*\.\d+([eE][+-]?\d+)?/,
      /\d+[eE][+-]?\d+/,
      /\d+/,
    )),

    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    comment: $ => token(seq('//', /.*/)),
  },
});
