# Pygments lexer for MLIR.
# Authors: Karl F. A. Friebel (@KFAFSP), Clément Fournier (@oowekyala)
# Usage: pygmentize -x -l ./MLIRLexer.py:MLIRLexer file.mlir
#
#  MIT License
#
# Copyright (c) 2024 Clément Fournier, Karl F. A. Friebel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pygments.lexer import Lexer, RegexLexer, bygroups, include, default, combined
from pygments.token import Name, Keyword, Operator, Comment, Text, Punctuation, Literal, Whitespace

comment_rule = (r'//.*?\n', Comment)
ssa_value_rule = (r'%[^[ )]*]*', Name.Variable)
symbol_rule = (r'@[^({]*', Name.Function)
basic_block_rule = (r'\^[^(:\]]*', Name.Label)
operation_rule = (r'(=)( +)([a-z_]+)(\.)([a-z_]+)',
                  bygroups(Operator, Text, Name.Namespace, Text,
                           Keyword.Function))
opInRegion_rule = (r'(=)( +)([a-z_]+)',
                   bygroups(Operator, Text, Keyword.Function))
opNoResults_rule = (r'([a-z_]+)( +)(%[^[ )]*]*)', bygroups(Keyword.Function, Text, Name.Variable))
non_assign_operation_rule = (r'([a-z_]+)(\.)([a-z_]+)',
                             bygroups(Name.Namespace, Text, Keyword.Function))
type_rule = (r'(!)([a-z_]+)(\.)([a-z0-9_]+)(<([^>]*)>)?',
             bygroups(Operator, Name.Namespace, Text, Keyword.Type, Keyword.Type))
int_float_rule = (r'(i|f)([0-9]+)', bygroups(Text, Keyword.Type))
abbrev_type_tule = (r'(!)([a-z0-9]+)', bygroups(Operator, Keyword.Type))
first_attribute_rule = (r'([{\[])([a-z_A-Z]+)( = +)([@a-z0-9">=]+)',
                        bygroups(Text, Name.Attribute, Text, Name.Tag))
following_attribute_rule = (r'(, +)([a-z_]+)( = +)([a-z0-9">=@]+)',
                            bygroups(Text, Name.Attribute, Text, Name.Tag))
abbrev_following_attribute_rule = (r'(, +)([a-z_]+)( = +)',
                                   bygroups(Text, Name.Attribute, Text))

digit = r'[0-9]'
hex_digit = r'[0-9a-fA-F]'
letter = r'[a-zA-Z]'
id_punct = r'[$._\-]'

decimal_literal = rf'{digit}+'
hexadecimal_literal = rf'0x{hex_digit}+'
float_literal = r'[-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?'
string_literal = r'"[^"\n\f\v\r]*"'

suffix_id = rf'(?:{digit}+|(?:{letter}|{id_punct})(?:{letter}|{id_punct}|{digit})*)'
value_id = rf'%{suffix_id}'

symbol_ref_id = rf'@(?:{suffix_id}|{string_literal})'

bare_id = rf'(?:{letter}|_)(?:{letter}|{digit}|[_$.])*'
bare_id_without_ns = rf'(?:{letter}|_)(?:{letter}|{digit}|[_$])*'
bare_id_with_ns = rf'((?:{letter}|_)(?:{letter}|{digit}|[_$])*)(\.)((?:{letter}|{digit}|[_$.])+)'

integer_type = rf'[su]?i{digit}+'
float_type = r'(?:f(?:16|32|64|80|128)|bf16|tf32|f8E5M2|f8E4M3FN|f8E5M2FNUZ|f8E5M3FNUZ|f8E4M3B11FNUZ)'

op_result_1 = rf'({value_id})(:)({decimal_literal})'


class MlirLexer(RegexLexer):
    name = 'MLIR'
    aliases = ['mlir']
    filenames = ['*.mlir']

    tokens = {
        'comments': [
            (r'//.*?\n', Comment),
            (r'\.\.\.', Comment)  # pretend ellipsis is comment
        ],

        'literals': [
            (float_literal, Literal.Number),
            (hexadecimal_literal, Literal.Number),
            (decimal_literal, Literal.Number),
            (string_literal, Literal.String),
            (r'[^\S\r\n]+', Whitespace),
        ],
        'punctuation': [
            (r'[()\[\],*+?{}<>-\|:]|->|<=?|>=?|==?|::', Punctuation),
        ],


        'sigils': [
            (rf'\^{bare_id}', Name.Label),
            (rf'%{bare_id}', Name.Variable),
            (rf'%{decimal_literal}', Name.Variable),
            (rf'@{bare_id}', Name.Variable.Global),
            (rf'!{bare_id}|!{bare_id_with_ns}', Name.Type),
            (rf'#{bare_id}|#{bare_id_with_ns}', Name.Attribute),
            (fr'{bare_id_with_ns}', bygroups(Name.Namespace, Punctuation, Name.Function)),
            (rf'({integer_type}|{float_type}|index|tensor|memref)\b', Keyword.Type),
            (rf'(x)({decimal_literal})', bygroups(Punctuation, Literal.Number)),
            (rf'(x)({integer_type}|{float_type})\b', bygroups(Punctuation, Keyword.Type)),
            (fr'{bare_id}', Name.Identifier),
        ],

        'tfl': [
            (r'\b(fill_buf|empty_buf|transfer|kill_buffer|tile|scope|kernel|return|gather|yield|reduce|schedule)\b', Name.Function),
            (r'\b((hw)?parallel|vectorized|reduction|rankreduce|factor|attributes|ranks|tasklets|dpus|into|threaded|ins|outs|sdim|symbolic|dim|by|scheduler|variables|platform|par|red|to|with)\b', Keyword.Declaration),
        ],

        'root': [
            include('tfl'),
            include('sigils'),
            include('punctuation'),
            include('literals'),
            include('comments'),
        ]
    }


class MlirSuperLexer(Lexer):
    name = 'MLIR'
    aliases = ['mlir']
    filenames = ['*.mlir']

    def __init__(self, **options):
        super(MlirSuperLexer, self).__init__(**options)

    def get_tokens_unprocessed(self, text):
        pass


