# Shared constants and functions for compiler.py and interpreter.py

# Token types
TT_EOF = 0
TT_PLUS = 2
TT_MINUS = 3
TT_MULT = 4
TT_DIV = 5
TT_MOD = 6
TT_LPAREN = 7
TT_RPAREN = 8
TT_SEMI = 9
TT_ASSIGN = 10
TT_IDENT = 11
TT_IF = 12
TT_ELSE = 13
TT_END = 14
TT_PRINT = 15
TT_EQ = 16
TT_NE = 17
TT_GE = 18
TT_LE = 19
TT_AND = 20
TT_OR = 21
TT_NOT = 22
TT_BITAND = 23
TT_BITOR = 24
TT_BITNOT = 25
TT_XOR = 26
TT_DO = 27
TT_WHILE = 28
TT_LT = 29
TT_GT = 30
TT_BREAK = 31
TT_CONTINUE = 32
# Compound assignment operators
TT_PLUS_ASSIGN = 33
TT_MINUS_ASSIGN = 34
TT_MULT_ASSIGN = 35
TT_DIV_ASSIGN = 36
TT_MOD_ASSIGN = 37
TT_SHL = 38
TT_SHR = 39
# Variable declarations
TT_VAR = 40
TT_LET = 41
# Type system additions
TT_COLON = 42
TT_TYPE_ASSIGN = 43  # :=
TT_TYPE_INT = 44
TT_TYPE_FLOAT = 45
TT_INT_LITERAL = 46
TT_FLOAT_LITERAL = 47
TT_TYPE_UINT = 48
TT_TYPE_LONG = 49
TT_TYPE_ULONG = 50
TT_UINT_LITERAL = 51
TT_LONG_LITERAL = 52
TT_ULONG_LITERAL = 53
TT_STRING_LITERAL = 54
TT_TYPE_STRING = 55
TT_DEF = 56
TT_RETURN = 57
TT_COMMA = 58
TT_NEWLINE = 59
# Struct system additions
TT_STRUCT = 60      # 'struct' keyword
TT_DOT = 61         # '.' for member access
TT_NEW = 62         # 'new' keyword for heap allocation
TT_DEL = 63         # 'del' keyword for heap deallocation
TT_LBRACE = 64      # '{' for future use
TT_RBRACE = 65      # '}' for future use
TT_SELF = 66        # 'self' keyword for method self-reference

# AST Node types (C-style enums)
AST_NODE_BASE = 0
AST_NODE_NUMBER = 1
AST_NODE_VARIABLE = 2
AST_NODE_BINARY_OP = 3
AST_NODE_UNARY_OP = 4
AST_NODE_ASSIGN = 5
AST_NODE_COMPOUND_ASSIGN = 6
AST_NODE_PRINT = 7
AST_NODE_IF = 8
AST_NODE_WHILE = 9
AST_NODE_BREAK = 10
AST_NODE_CONTINUE = 11
AST_NODE_EXPR_STMT = 12
AST_NODE_VAR_DECL = 13
AST_NODE_COMPARE = 15
AST_NODE_LOGICAL = 16
AST_NODE_BITOP = 17
AST_NODE_STRING = 18
AST_NODE_FUNCTION_DECL = 19
AST_NODE_FUNCTION_CALL = 20
AST_NODE_RETURN = 21
AST_NODE_PARAM = 22
# Struct system additions
AST_NODE_STRUCT_DEF = 23    # struct definition
AST_NODE_STRUCT_INIT = 24   # struct initialization
AST_NODE_METHOD_DEF = 25    # method definition
AST_NODE_MEMBER_ACCESS = 26 # field/method access
AST_NODE_METHOD_CALL = 27   # method call
AST_NODE_NEW = 28           # heap allocation with 'new'
AST_NODE_DEL = 29           # heap deallocation with 'del'

# Variable types
TYPE_UNKNOWN = 0
TYPE_INT = 1
TYPE_FLOAT = 2
TYPE_UINT = 3
TYPE_LONG = 4
TYPE_ULONG = 5
TYPE_STRING = 6
TYPE_VOID = 7

# Type system constants
TYPE_STRUCT_BASE = 100      # Base for struct types
REF_TYPE_FLAG = 0x80000000  # Bit 32 set for reference types

# Order of type precedence (highest to lowest)
TYPE_PRECEDENCE = [TYPE_STRING, TYPE_FLOAT, TYPE_ULONG, TYPE_LONG, TYPE_UINT, TYPE_INT]

# Type promotion helper functions
def can_promote(src_type, dst_type):
    """
    Determine if a value of src_type can be promoted to dst_type
    Rules:
    1. Any type can be promoted to itself
    2. Any integer literal (TYPE_INT) can be promoted to any type during declaration
    3. For other cases, follow type precedence rules
    """
    # Same type - always compatible
    if src_type == dst_type:
        return True
    
    # Int literal special case - can promote to any other type 
    # This allows writing var x:uint = 42; without having to add the 'u' suffix
    if src_type == TYPE_INT:
        return dst_type != TYPE_STRING
        
    # For other types, only allow promotion to higher precedence types
    src_prec = TYPE_PRECEDENCE.index(src_type) if src_type in TYPE_PRECEDENCE else -1
    dst_prec = TYPE_PRECEDENCE.index(dst_type) if dst_type in TYPE_PRECEDENCE else -1
    return src_prec > dst_prec  # Lower index = higher precedence

# Mapping from type constants to their string representations
TYPE_TO_STRING_MAP = {
    TYPE_UNKNOWN: "unknown",
    TYPE_INT: "int",
    TYPE_FLOAT: "float",
    TYPE_UINT: "uint",
    TYPE_LONG: "long",
    TYPE_ULONG: "ulong",
    TYPE_STRING: "string",
    TYPE_VOID: "void"
}

# Mapping from token types to variable types
TOKEN_TO_TYPE_MAP = {
    TT_INT_LITERAL: TYPE_INT,
    TT_FLOAT_LITERAL: TYPE_FLOAT,
    TT_UINT_LITERAL: TYPE_UINT,
    TT_LONG_LITERAL: TYPE_LONG,
    TT_ULONG_LITERAL: TYPE_ULONG,
    TT_STRING_LITERAL: TYPE_STRING
}

# Mapping from type tokens to variable types
TYPE_TOKEN_MAP = {
    TT_TYPE_INT: TYPE_INT,
    TT_TYPE_FLOAT: TYPE_FLOAT,
    TT_TYPE_UINT: TYPE_UINT,
    TT_TYPE_LONG: TYPE_LONG,
    TT_TYPE_ULONG: TYPE_ULONG,
    TT_TYPE_STRING: TYPE_STRING
}

# Global hashtable for keywords
KEYWORDS = {
    'if': TT_IF,
    'else': TT_ELSE,
    'end': TT_END,
    'print': TT_PRINT,
    'and': TT_AND,
    'or': TT_OR,
    'do': TT_DO,
    'while': TT_WHILE,
    'break': TT_BREAK,
    'continue': TT_CONTINUE,
    'xor': TT_XOR,
    'bitnot': TT_BITNOT,
    'shl': TT_SHL,
    'shr': TT_SHR,
    'var': TT_VAR,  # Variable declaration
    'let': TT_LET,  # Constant declaration
    'int': TT_TYPE_INT,  # Int type
    'float': TT_TYPE_FLOAT,  # Float type
    'uint': TT_TYPE_UINT,  # Unsigned Int type
    'long': TT_TYPE_LONG,  # Long type
    'ulong': TT_TYPE_ULONG,  # Unsigned Long type
    'string': TT_TYPE_STRING,  # String type
    'def': TT_DEF,          # Function definition
    'return': TT_RETURN,    # Return statement
    # Struct system additions
    'struct': TT_STRUCT,
    'new': TT_NEW,
    'del': TT_DEL,
    'self': TT_SELF,
}

# Global precedence table for binary operators
BINARY_PRECEDENCE = {
    TT_ASSIGN: 10,   # lowest precedence (Python: assignments)
    TT_OR: 20,       # Python: Boolean OR
    TT_AND: 30,      # Python: Boolean AND
    TT_EQ: 40,       # Python: comparisons
    TT_NE: 40,
    TT_GE: 40,
    TT_GT: 40,
    TT_LE: 40,
    TT_LT: 40,
    TT_BITOR: 50,    # Python: bitwise OR
    TT_XOR: 60,      # Python: bitwise XOR
    TT_BITAND: 70,   # Python: bitwise AND
    TT_PLUS: 80,     # Python: addition/subtraction
    TT_MINUS: 80,
    TT_MULT: 90,     # Python: multiplication/division/modulus/shift
    TT_DIV: 90,
    TT_MOD: 90,
    TT_SHL: 90,      # Python: shift has same precedence as multiplication
    TT_SHR: 90,
    TT_LPAREN: 100,  # needed for function calls only
    # Member access has highest precedence
    TT_DOT: 110,     # Member access
}

# Unary operator precedence (higher than binary operators)
UNARY_PRECEDENCE = 120

# Type helpers for struct and reference types
def is_struct_type(type_):
    """Check if a type is a struct type"""
    return type_ >= TYPE_STRUCT_BASE and (type_ & REF_TYPE_FLAG) == 0

def is_ref_type(type_):
    """Check if a type is a reference type"""
    return (type_ & REF_TYPE_FLAG) != 0

def get_base_type(type_):
    """Get the base type of a reference type"""
    if is_ref_type(type_):
        return type_ & ~REF_TYPE_FLAG
    return type_

def make_ref_type(type_):
    """Convert a type to its reference equivalent"""
    if is_ref_type(type_):
        return type_  # Already a reference
    return type_ | REF_TYPE_FLAG

def var_type_to_string(var_type):
    """Convert a variable type constant to a string for error messages using the map"""
    if is_ref_type(var_type):
        base_type = get_base_type(var_type)
        base_type_name = var_type_to_string(base_type)
        return "ref to " + base_type_name
    elif is_struct_type(var_type):
        # Import here to avoid circular imports
        from type_registry import get_struct_name
        struct_name = get_struct_name(var_type)
        return struct_name if struct_name else "unknown struct"
    return TYPE_TO_STRING_MAP.get(var_type, "unknown")

def ast_node_type_to_string(node_type):
    """Convert AST node type to string for debugging"""
    type_names = {
        AST_NODE_BASE: "BASE",
        AST_NODE_NUMBER: "NUMBER",
        AST_NODE_VARIABLE: "VARIABLE",
        AST_NODE_BINARY_OP: "BINARY_OP",
        AST_NODE_UNARY_OP: "UNARY_OP",
        AST_NODE_ASSIGN: "ASSIGN",
        AST_NODE_COMPOUND_ASSIGN: "COMPOUND_ASSIGN",
        AST_NODE_PRINT: "PRINT",
        AST_NODE_IF: "IF",
        AST_NODE_WHILE: "WHILE",
        AST_NODE_BREAK: "BREAK",
        AST_NODE_CONTINUE: "CONTINUE",
        AST_NODE_EXPR_STMT: "EXPR_STMT",
        AST_NODE_VAR_DECL: "VAR_DECL",
        AST_NODE_COMPARE: "COMPARE",
        AST_NODE_LOGICAL: "LOGICAL",
        AST_NODE_BITOP: "BITOP",
        AST_NODE_STRING: "STRING",
        AST_NODE_FUNCTION_DECL: "FUNCTION_DECL",
        AST_NODE_FUNCTION_CALL: "FUNCTION_CALL",
        AST_NODE_RETURN: "RETURN",
        AST_NODE_PARAM: "PARAM",
        # Struct system additions
        AST_NODE_STRUCT_DEF: "STRUCT_DEF",
        AST_NODE_STRUCT_INIT: "STRUCT_INIT",
        AST_NODE_METHOD_DEF: "METHOD_DEF",
        AST_NODE_MEMBER_ACCESS: "MEMBER_ACCESS",
        AST_NODE_METHOD_CALL: "METHOD_CALL",
        AST_NODE_NEW: "NEW",
        AST_NODE_DEL: "DEL",
    }
    return type_names.get(node_type, "UNKNOWN")

# Helper function for promoting literal values
def promote_literal_if_needed(value, from_type, to_type):
    """Promote a literal value if needed between compatible types"""
    if from_type != to_type and can_promote(from_type, to_type):
        # Most numeric promotions don't require value changes in Python
        # But certain conversions might need special handling in the future
        return value
    return value

# Define token type names for debugging (pre-populated)
TOKEN_NAMES = {
    TT_EOF: "TT_EOF",
    TT_PLUS: "TT_PLUS",
    TT_MINUS: "TT_MINUS",
    TT_MULT: "TT_MULT",
    TT_DIV: "TT_DIV",
    TT_MOD: "TT_MOD",
    TT_LPAREN: "TT_LPAREN",
    TT_RPAREN: "TT_RPAREN",
    TT_SEMI: "TT_SEMI",
    TT_ASSIGN: "TT_ASSIGN",
    TT_IDENT: "TT_IDENT",
    TT_IF: "TT_IF",
    TT_ELSE: "TT_ELSE",
    TT_END: "TT_END",
    TT_PRINT: "TT_PRINT",
    TT_EQ: "TT_EQ",
    TT_NE: "TT_NE",
    TT_GE: "TT_GE",
    TT_LE: "TT_LE",
    TT_AND: "TT_AND",
    TT_OR: "TT_OR",
    TT_NOT: "TT_NOT",
    TT_BITAND: "TT_BITAND",
    TT_BITOR: "TT_BITOR",
    TT_BITNOT: "TT_BITNOT",
    TT_XOR: "TT_XOR",
    TT_DO: "TT_DO",
    TT_WHILE: "TT_WHILE",
    TT_LT: "TT_LT",
    TT_GT: "TT_GT",
    TT_BREAK: "TT_BREAK",
    TT_CONTINUE: "TT_CONTINUE",
    TT_PLUS_ASSIGN: "TT_PLUS_ASSIGN",
    TT_MINUS_ASSIGN: "TT_MINUS_ASSIGN",
    TT_MULT_ASSIGN: "TT_MULT_ASSIGN",
    TT_DIV_ASSIGN: "TT_DIV_ASSIGN",
    TT_MOD_ASSIGN: "TT_MOD_ASSIGN",
    TT_SHL: "TT_SHL",
    TT_SHR: "TT_SHR",
    TT_VAR: "TT_VAR",
    TT_LET: "TT_LET",
    TT_COLON: "TT_COLON",
    TT_TYPE_ASSIGN: "TT_TYPE_ASSIGN",
    TT_TYPE_INT: "TT_TYPE_INT",
    TT_TYPE_FLOAT: "TT_TYPE_FLOAT",
    TT_INT_LITERAL: "TT_INT_LITERAL",
    TT_FLOAT_LITERAL: "TT_FLOAT_LITERAL",
    TT_TYPE_UINT: "TT_TYPE_UINT",
    TT_TYPE_LONG: "TT_TYPE_LONG",
    TT_TYPE_ULONG: "TT_TYPE_ULONG",
    TT_UINT_LITERAL: "TT_UINT_LITERAL",
    TT_LONG_LITERAL: "TT_LONG_LITERAL",
    TT_ULONG_LITERAL: "TT_ULONG_LITERAL",
    TT_STRING_LITERAL: "TT_STRING_LITERAL",
    TT_TYPE_STRING: "TT_TYPE_STRING",
    TT_DEF: "TT_DEF",
    TT_RETURN: "TT_RETURN",
    TT_COMMA: "TT_COMMA",
    TT_NEWLINE: "TT_NEWLINE",
    # Struct system additions
    TT_STRUCT: "TT_STRUCT",
    TT_DOT: "TT_DOT",
    TT_NEW: "TT_NEW",
    TT_DEL: "TT_DEL",
    TT_LBRACE: "TT_LBRACE",
    TT_RBRACE: "TT_RBRACE",
    TT_SELF: "TT_SELF",
}

def token_name(token_type):
    """Convert a token type number to its name for better debugging"""
    return TOKEN_NAMES.get(token_type, str(token_type))

# Base class for all compiler-defined exceptions
class CompilerException(Exception):
    """Base class for all exceptions raised by the compiler"""
    pass
