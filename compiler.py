# Implementation of a Pratt parser in Python 2.7

from shared import *
from lexer import Token, Lexer
from type_registry import get_registry
from scope import EnvironmentStack

# import registry singleton
registry = get_registry()

# Constants for assignment context
ASSIGN_CTX_FUNCTION_RESULT = 1
ASSIGN_CTX_MEMBER_ACCESS = 2
ASSIGN_CTX_ARRAY_ELEMENT = 3
ASSIGN_CTX_VARIABLE = 4

# Base class for all AST nodes
class ASTNode(object):
    def __init__(self, node_type=AST_NODE_BASE, token=None):
        assert(token)
        self.token = token
        self.node_type = node_type
        self.expr_type = TYPE_UNKNOWN
        self.ref_kind = REF_KIND_NONE

    def __repr__(self):
        return "%s" % ast_node_type_to_string(self.node_type)

class NilNode(ASTNode):
    def __init__(self, token, target_type=TYPE_UNKNOWN):
        ASTNode.__init__(self, AST_NODE_NIL, token)
        self.ref_kind = REF_KIND_GENERIC
        self.expr_type = target_type
        self.value = 0

    def __repr__(self):
        return "Nil()"

class NumberNode(ASTNode):
    def __init__(self, token, value, expr_type):
        ASTNode.__init__(self, AST_NODE_NUMBER, token)
        self.value = value
        self.expr_type = expr_type  # TYPE_INT, TYPE_FLOAT, etc.

    def __repr__(self):
        return "Number(%s, %s)" % (self.value, registry.var_type_to_string(self.expr_type))

class StringNode(ASTNode):
    def __init__(self, token, value):
        ASTNode.__init__(self, AST_NODE_STRING, token)
        self.value = value
        self.expr_type = TYPE_STRING

    def __repr__(self):
        return "String(\"%s\")" % self.value

class VariableNode(ASTNode):
    def __init__(self, token, name, var_type, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_VARIABLE, token)
        self.name = name
        self.expr_type = var_type
        self.ref_kind = ref_kind

    def __repr__(self):
        return "Var(%s, %s)" % (self.name, registry.format_type_with_ref_kind(self.expr_type))

class BinaryOpNode(ASTNode):
    def __init__(self, token, operator, left, right, result_type, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_BINARY_OP, token)
        self.operator = operator
        self.left = left
        self.right = right
        self.expr_type = result_type
        self.ref_kind = ref_kind

    def __repr__(self):
        return "BinaryOp(%s, %s, %s) -> %s" % (
            self.operator, repr(self.left), repr(self.right),
            registry.format_type_with_ref_kind(self.expr_type)
        )

class UnaryOpNode(ASTNode):
    def __init__(self, token, operator, operand, result_type):
        ASTNode.__init__(self, AST_NODE_UNARY_OP, token)
        self.operator = operator
        self.operand = operand
        self.expr_type = result_type

    def __repr__(self):
        return "UnaryOp(%s, %s) -> %s" % (
            self.operator, repr(self.operand), registry.var_type_to_string(self.expr_type)
        )

class PrintNode(ASTNode):
    def __init__(self, token, expr):
        ASTNode.__init__(self, AST_NODE_PRINT, token)
        self.expr = expr

    def __repr__(self):
        return "Print(%s)" % repr(self.expr)

class IfNode(ASTNode):
    def __init__(self, token, condition, then_body, else_body=None):
        ASTNode.__init__(self, AST_NODE_IF, token)
        self.condition = condition
        self.then_body = then_body  # List of statement nodes
        self.else_body = else_body  # List of statement nodes or None

    def __repr__(self):
        if self.else_body:
            return "If(%s, [%s], [%s])" % (
                repr(self.condition),
                ", ".join(repr(stmt) for stmt in self.then_body),
                ", ".join(repr(stmt) for stmt in self.else_body),
            )
        else:
            return "If(%s, [%s])" % (
                repr(self.condition),
                ", ".join(repr(stmt) for stmt in self.then_body),
            )

class WhileNode(ASTNode):
    def __init__(self, token, condition, body):
        ASTNode.__init__(self, AST_NODE_WHILE, token)
        self.condition = condition
        self.body = body  # List of statement nodes

    def __repr__(self):
        return "While(%s, [%s])" % (
            repr(self.condition),
            ", ".join(repr(stmt) for stmt in self.body),
        )

class BreakNode(ASTNode):
    def __init__(self, token):
        ASTNode.__init__(self, AST_NODE_BREAK, token)

    def __repr__(self):
        return "Break()"

class ContinueNode(ASTNode):
    def __init__(self, token):
        ASTNode.__init__(self, AST_NODE_CONTINUE, token)

    def __repr__(self):
        return "Continue()"

class ExprStmtNode(ASTNode):
    def __init__(self, token, expr):
        ASTNode.__init__(self, AST_NODE_EXPR_STMT, token)
        self.expr = expr

    def __repr__(self):
        return "ExprStmt(%s)" % repr(self.expr)

class VarDeclNode(ASTNode):
    def __init__(self, token, decl_type, var_name, var_type, expr, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_VAR_DECL, token)
        self.decl_type = decl_type
        self.var_name = var_name
        self.var_type = var_type
        self.expr = expr
        self.ref_kind = ref_kind

    def __repr__(self):
        decl_type_str = "var" if self.decl_type == TT_VAR else "const"
        return "VarDecl(%s, %s, %s, %s)" % (
            decl_type_str, self.var_name, registry.var_type_to_string(self.var_type), repr(self.expr)
        )

class FunctionDeclNode(ASTNode):
    def __init__(self, token, name, params, return_type, body, parent_struct_id=-1, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_FUNCTION_DECL, token)
        self.name = name
        self.params = params  # List of (name, type) tuples
        self.return_type = return_type
        self.ref_kind = ref_kind
        self.body = body
        self.expr_type = TYPE_VOID # a function decl can't be used in expression context
        self.parent_struct_id = parent_struct_id  # -1 for global functions

    def __repr__(self):
        name = self.name if self.parent_struct_id == -1 else "%s.%s"%(registry.get_struct_name(self.parent_struct_id), self.name)
        func_or_method = "Function" if self.parent_struct_id == -1 else "Method"
        params_str = ", ".join([
            "%s%s:%s" % (
                "byref " if is_ref else "",
                pname,
                registry.var_type_to_string(ptype)
            )
            for pname, ptype, is_ref in self.params
        ])

        # Also include byref in return type if applicable
        return_type_str = registry.var_type_to_string(self.return_type)

        if self.ref_kind != REF_KIND_NONE:
            return_type_str = "byref " + return_type_str

        return "%s(%s(%s):%s, [%s])" % (
            func_or_method,
            name,
            params_str,
            return_type_str,
            ", ".join(repr(stmt) for stmt in self.body),
        )

class ReturnNode(ASTNode):
    def __init__(self, token, expr=None, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_RETURN, token)
        self.expr = expr  # Can be None for return with no value
        self.expr_type = TYPE_VOID if expr is None else (expr.expr_type if hasattr(expr, 'expr_type') else TYPE_UNKNOWN)
        self.ref_kind = ref_kind  # Whether this return is from a byref function

    def __repr__(self):
        if self.expr:
            byref_str = "byref " if self.ref_kind != REF_KIND_NONE else ""
            return "Return(%s) -> %s%s" %(repr(self.expr), byref_str, registry.var_type_to_string(self.expr_type))
        else:
            return "Return()"

class CompareNode(ASTNode):
    def __init__(self, token, operator, left, right):
        ASTNode.__init__(self, AST_NODE_COMPARE, token)
        self.operator = operator
        self.left = left
        self.right = right
        self.expr_type = TYPE_INT  # Comparisons always return int

    def __repr__(self):
        return "Compare(%s, %s, %s)" % (self.operator, repr(self.left), repr(self.right))

class LogicalNode(ASTNode):
    def __init__(self, token, operator, left, right):
        ASTNode.__init__(self, AST_NODE_LOGICAL, token)
        self.operator = operator
        self.left = left
        self.right = right
        self.expr_type = TYPE_INT  # Logical ops always return int

    def __repr__(self):
        return "Logical(%s, %s, %s)" % (self.operator, repr(self.left), repr(self.right))

class BitOpNode(ASTNode):
    def __init__(self, token, operator, left, right):
        ASTNode.__init__(self, AST_NODE_BITOP, token)
        self.operator = operator
        self.left = left
        self.right = right
        self.expr_type = TYPE_INT  # Bitwise ops always return int

    def __repr__(self):
        return "BitOp(%s, %s, %s)" % (self.operator, repr(self.left), repr(self.right))

class StructDefNode(ASTNode):
    def __init__(self, token, name, parent_name, fields, struct_id):
        ASTNode.__init__(self, AST_NODE_STRUCT_DEF, token)
        self.name = name
        self.parent_name = parent_name
        self.fields = fields  # [(name, type), ...]
        self.struct_id = struct_id
        self.expr_type = TYPE_VOID

    def __repr__(self):
        parent_str = "(%s)" % self.parent_name if self.parent_name else ""
        fields_str = ", ".join(["%s:%s" % (name, registry.var_type_to_string(type_)) for name, type_ in self.fields])
        return "StructDef(%s%s, [%s])" % (self.name, parent_str, fields_str)

class StructInitNode(ASTNode):
    def __init__(self, token, struct_name, struct_id, args=None):
        ASTNode.__init__(self, AST_NODE_STRUCT_INIT, token)
        self.struct_name = struct_name
        self.struct_id = struct_id
        self.args = args or []  # Args for constructor
        self.expr_type = struct_id

    def __repr__(self):
        args_str = ", ".join(repr(arg) for arg in self.args)
        return "StructInit(%s(%s))" % (self.struct_name, args_str)

class MemberAccessNode(ASTNode):
    def __init__(self, token, obj, member_name, member_type, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_MEMBER_ACCESS, token)
        self.obj = obj  # Object expression
        self.member_name = member_name
        self.expr_type = member_type
        self.ref_kind = ref_kind

    def __repr__(self):
        return "MemberAccess(%s.%s)" % (repr(self.obj), self.member_name)

class CallNode(ASTNode):
    def __init__(self, token, name, args, return_type, obj=None, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_CALL, token)
        self.name = name
        self.args = args
        self.expr_type = return_type
        self.obj = obj  # None for regular functions, object expr for methods
        self.ref_kind = ref_kind

    def is_method_call(self):
        return self.obj is not None

    def __repr__(self):
        args_str = ", ".join(repr(arg) for arg in self.args)
        if self.obj:
            return "Call(%s.%s(%s))" % (repr(self.obj), self.name, args_str)
        else:
            return "Call(%s(%s))" % (self.name, args_str)

class NewNode(ASTNode):
    def __init__(self, token, struct_init):
        ASTNode.__init__(self, AST_NODE_NEW, token)
        self.struct_init = struct_init
        self.expr_type = struct_init.expr_type
        self.ref_kind = REF_KIND_HEAP

    def __repr__(self):
        return "New(%s) -> heap_ref<%s>" % (repr(self.struct_init), registry.var_type_to_string(self.expr_type))

class DelNode(ASTNode):
    def __init__(self, token, expr):
        ASTNode.__init__(self, AST_NODE_DEL, token)
        self.expr = expr
        self.expr_type = TYPE_VOID

    def __repr__(self):
        return "Del(%s)" % repr(self.expr)

class GenericInitializerNode(ASTNode):
    def __init__(self, token, elements, subtype, target_type=TYPE_UNKNOWN):
        ASTNode.__init__(self, AST_NODE_GENERIC_INITIALIZER, token)
        self.elements = elements    # List of expressions or nested initializers
        self.subtype = subtype
        self.target_type = target_type  # The expected type (struct/array)
        self.expr_type = target_type    # Result type of this initializer

    def __repr__(self):
        subtype_str = ["TUPLE", "LINEAR", "NAMED"][self.subtype]
        elements_str = ", ".join(repr(e) for e in self.elements)
        return "Initializer(%s, %s, [%s])" % (
            subtype_str, registry.var_type_to_string(self.target_type), elements_str)

class ArrayAccessNode(ASTNode):
    def __init__(self, token, array, index, element_type):
        ASTNode.__init__(self, AST_NODE_ARRAY_ACCESS, token)
        self.array = array
        self.index = index
        self.expr_type = element_type
        # Arrays elements can be used as references when the array is a reference
        self.ref_kind = array.ref_kind

    def __repr__(self):
        return "ArrayAccess(%s[%s])" % (repr(self.array), repr(self.index))

def is_literal_node(node):
    """Check if a node represents a literal value (for global var init)"""
    if node.node_type in [AST_NODE_NUMBER, AST_NODE_STRING]:
        return True
    elif node.node_type == AST_NODE_GENERIC_INITIALIZER:
        # Allow initializers where all elements are literals
        return all(is_literal_node(elem) for elem in node.elements)
    return False

class Variable:
    def __init__(self, expr_type, is_const=False, ref_kind=REF_KIND_NONE):
        self.expr_type = expr_type
        self.is_const = is_const
        self.ref_kind = ref_kind

class EnumBuilder:
    """Helper class for building enum declarations"""
    def __init__(self, name, base_type):
        self.name = name
        self.base_type = base_type
        self.members = []  # List of (name, value) tuples
        self.next_value = 0  # For auto-incrementing

    def add_member(self, name, value=None):
        """Add a member with explicit value or auto-increment"""
        if value is None:
            value = self.next_value

        self.members.append((name, value))
        self.next_value = value + 1

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.token = self.lexer.next_token()
        self.prev_token = None
        self.env = EnvironmentStack()
        self.statements = None

        # Track if we've seen functions - used to enforce globals-before-functions rule
        self.seen_main_function = False
        self.current_function = -1  # Track current function for return checking

        # Track current struct for method definitions
        self.current_struct = None
        self.current_initializer_type = TYPE_UNKNOWN

    def create_default_value(self, type_id):
        """Create an AST node with the default/zero value for given type"""
        if registry.is_array_type(type_id):
            # For arrays with fixed size, create array of zeros
            element_type = registry.get_array_element_type(type_id)
            size = registry.get_array_size(type_id)

            if size is not None:
                # Create array with default-initialized elements
                elements = []
                for _ in range(size):
                    elements.append(self.create_default_value(element_type))
                return GenericInitializerNode(self.token, elements, INITIALIZER_SUBTYPE_LINEAR, type_id)
            else:
                # Dynamic arrays default to nil
                return NilNode(self.token)

        elif registry.is_struct_type(type_id):
            # Get all fields including inherited ones
            struct_name = registry.get_struct_name(type_id)
            fields = registry.get_all_fields(struct_name)

            # Create default initializer for each field
            elements = []
            for _, field_type in fields:
                elements.append(self.create_default_value(field_type))

            # Create struct initializer with default values
            init_node = GenericInitializerNode(self.token, elements, INITIALIZER_SUBTYPE_LINEAR, type_id)
            init_node.expr_type = type_id  # Ensure expression type is set
            return init_node

        else:
            # Primitives get appropriate zero values
            if type_id == TYPE_STRING:
                return StringNode(self.token, "")
            elif is_float_type(type_id):
                return NumberNode(self.token, 0.0, type_id)
            else:
                # For all integer types and unknown types
                return NumberNode(self.token, 0, type_id)

    def get_common_type(self, type1, type2):
        """
        Implement C's type promotion rules for binary operations
        Returns the resulting type
        """
        # Handle string concatenation special case
        if TYPE_STRING in (type1, type2):
            return TYPE_STRING if type1 == type2 else None

        # Handle struct types - they must be identical
        if registry.is_struct_type(type1) or registry.is_struct_type(type2):
            return type1 if type1 == type2 else None

        # If either operand is floating point, result is the highest precision float
        if is_float_type(type1) or is_float_type(type2):
            if TYPE_DOUBLE in (type1, type2):
                return TYPE_DOUBLE
            return TYPE_FLOAT

        # Handle integer promotions
        # For integer types, use the type promotion rules we defined in can_promote
        # Try promoting each type to the other, take the higher one.
        # This depends on the right order of the TYPE_XXX values in shared.py
        if can_promote(type1, type2):
            if can_promote(type2, type1):
                return max(type1, type2)
            return type2
        if can_promote(type2, type1):
            return type1

    def calculate_result_type(self, op, left_type, right_type):
        """Calculate result type for binary operation following C rules"""
        # String concatenation
        if op == '+' and TYPE_STRING in (left_type, right_type):
            if left_type != right_type:
                self.error("Cannot concatenate string with non-string type")
            return TYPE_STRING

        # Comparison operators always return int
        if op in ['==', '!=', '<', '<=', '>', '>=']:
            return TYPE_INT

        # Logical operators always return int
        if op in ['and', 'or']:
            return TYPE_INT

        # Bitwise operators require integer operands
        if op in ['&', '|', 'xor', 'shl', 'shr']:
            if is_float_type(left_type) or is_float_type(right_type) or registry.is_struct_type(left_type) or registry.is_struct_type(right_type):
                self.error("Bitwise operators require integer operands")

        # Arithmetic operators
        return self.get_common_type(left_type, right_type)

    def is_variable_declared(self, var_name):
        """Check if a variable is declared in any accessible scope"""
        return self.env.get(var_name) is not None

    def is_constant(self, var_name):
        """Check if a variable is a constant in any accessible scope"""
        var = self.env.get(var_name)
        return var is not None and var.is_const

    def get_variable_type(self, var_name):
        """Get a variable's type from the appropriate scope"""
        var = self.env.get(var_name)
        return var.expr_type

    def declare_variable(self, var_name, var_type, is_const=False, ref_kind=REF_KIND_NONE):
        """Declare a variable in the current scope"""
        var = self.env.get(var_name)
        # Check if already declared in current scope
        if var is not None:
            return False  # Already declared in this scope

        var = Variable(var_type, is_const=is_const, ref_kind=ref_kind)
        self.env.set(var_name, var)
        return True

    def get_variable_ref_kind(self, var_name):
        """Get a variable's reference kind from the appropriate scope"""
        var = self.env.get(var_name)
        return var.ref_kind

    def error(self, message):
        """Raise a compiler exception with current token information"""
        raise CompilerException(message, self.token)

    def type_mismatch_error(self, context, from_type, to_type):
        """Generate consistent type mismatch error messages"""
        self.error("%s: cannot convert %s to %s" %
                  (context, registry.var_type_to_string(from_type), registry.var_type_to_string(to_type)))

    def type_mismatch_assignment_error(self, var_name, expr_type, expr_ref_kind, var_type, var_ref_kind):
        """Generate type mismatch error for assignment operations"""
        self.error("Type mismatch: can't assign a value of type %s to %s (type %s)" % 
                  (registry.format_type_with_ref_kind(expr_type, expr_ref_kind),
                   var_name,
                   registry.format_type_with_ref_kind(var_type, var_ref_kind)))

    def already_declared_error(self, var_name):
        """Generate error for already declared variables"""
        self.error("Variable '%s' is already declared in this scope" % var_name)

    def advance(self):
        self.prev_token = self.token
        self.token = self.lexer.next_token()

    def consume(self, token_type, emessage=None):
        if self.token.type == token_type:
            self.advance()
        else:
            expected_type_name = token_name(token_type)
            actual_type_name = token_name(self.token.type)
            if not emessage: emessage = 'Expected %s but got %s'%(
                      expected_type_name, actual_type_name)
            self.error(emessage)


    def lbp(self, t):
        return BINARY_PRECEDENCE.get(t.type, 0)

    def expression(self, rbp=0):
        t = self.token
        self.advance()
        old_initializer_type = self.current_initializer_type
        left = self.nud(t)
        while rbp < self.lbp(self.token):
            t = self.token
            self.advance()
            left = self.led(t, left)
        self.current_initializer_type = old_initializer_type
        return left

    def check_type_compatibility(self, var_name, expr):
        """Check if the expression's type is compatible with the variable's type"""
        # Get variable type from the appropriate scope
        var_type = self.get_variable_type(var_name)

        # Retrieve the actual reference kind for the variable
        var_ref_kind = self.get_variable_ref_kind(var_name)

        if var_type != TYPE_UNKNOWN and expr.expr_type != TYPE_UNKNOWN:
            # Check type and reference kind compatibility
            if not self.can_promote_with_ref(expr.expr_type, expr.ref_kind, var_type, var_ref_kind):
                self.type_mismatch_assignment_error(var_name, expr.expr_type, expr.ref_kind, var_type, var_ref_kind)

    def can_promote_with_ref(self, from_type, from_ref_kind, to_type, to_ref_kind):
        """Check if types are compatible considering reference kinds"""
        # References can't be promoted to non-references
        if from_ref_kind != REF_KIND_NONE and to_ref_kind == REF_KIND_NONE:
            return False

        # For references, allow inheritance relationships
        if from_ref_kind != REF_KIND_NONE and to_ref_kind != REF_KIND_NONE:
            # If both are struct types, check inheritance relationship
            if registry.is_struct_type(from_type) and registry.is_struct_type(to_type):
                # Allow child class where parent class is expected (covariance)
                if registry.is_subtype_of(from_type, to_type):
                    return True

        # Now check the underlying types
        return can_promote(from_type, to_type)

    def check_argument_types(self, args, params, context_name):
        """Check if argument types match parameter types"""
        for i in range(len(params)):
            # This should never happen as we check arg count before this
            if i >= len(args): break

            param_name, param_type, is_byref = params[i]
            arg = args[i]

            # Special handling for byref parameters
            if is_byref:
                # All expressions should be allowed for byref parameters
                # The interpreter will handle creating temporary storage as needed

                # The referenced variable's type must be compatible with the parameter type
                # We use REF_KIND_HEAP here as it doesn't matter in this context - only that a ref type is used
                if not self.can_promote_with_ref(arg.expr_type, REF_KIND_HEAP, param_type, REF_KIND_HEAP):
                    self.type_mismatch_error(
                        "Type mismatch for byref argument '%s'" % param_name,
                        arg.expr_type, param_type
                    )

                # If we get here, the byref parameter check passed
                continue

            if not can_promote(arg.expr_type, param_type):
                self.type_mismatch_error("Type mismatch for argument %d of %s" % (i+1, context_name), 
                                         arg.expr_type, param_type)

    def check_arg_count(self, func_name, params, args, is_method=False):
        """Helper to check argument count against parameter count"""
        if len(args) != len(params):
            subtrahend = 1 if is_method else 0
            self.error("%s expects %d arguments, got %d" %
                      (func_name, len(params)-subtrahend, len(args)-subtrahend))

    def parse_args(self, args):
        """Helper to parse argument lists for function/method calls"""
        if self.token.type != TT_RPAREN:
            args.append(self.expression(0))
            while self.token.type == TT_COMMA:
                self.advance()  # Skip comma
                args.append(self.expression(0))
        return args

    def check_field_compatibility(self, from_type, to_type):
        """Helper to check field assignment type compatibility"""
        if not can_promote(from_type, to_type):
            self.error("Type mismatch: cannot assign %s to field of type %s" % 
                      (registry.var_type_to_string(from_type), registry.var_type_to_string(to_type)))

    def parse_member_access(self, obj_node):
        """
        Parse member access for an object (obj.member)
        Handles both field access and method calls
        Returns a MemberAccessNode or CallNode
        """
        # Get object type
        obj_type = obj_node.expr_type
        obj_ref_kind = obj_node.ref_kind  # Get object's reference kind
        # No need to unwrap reference since we track ref_kind separately
        base_type = obj_type

        if not registry.is_struct_type(base_type):
            self.error("Left side of '.' is not a struct type")

        struct_name = registry.get_struct_name(base_type)

        # Parse member name
        if self.token.type != TT_IDENT:
            self.error("Expected member name after '.'")

        member_name = self.token.value
        self.advance()

        # Check if it's a method call or field access
        if self.token.type == TT_LPAREN:
            # Method call
            self.advance()  # Skip '('

            # Get method details directly from registry
            method_id = registry.lookup_function(member_name, base_type)
            if method_id == -1:
                self.error("Method '%s' not found in struct '%s'" % (member_name, struct_name))

            func_obj = registry.get_func_from_id(method_id)

            # Parse arguments
            args = []
            self.parse_args(args)
            # For methods, insert the object as the first argument (for 'self')
            args.insert(0, obj_node)
            self.consume(TT_RPAREN)

            # Type check arguments
            self.check_arg_count("Method '%s'" % member_name, func_obj.params, args, is_method=True)
            self.check_argument_types(args, func_obj.params, "method '%s'" % member_name)
            ref_kind = REF_KIND_GENERIC if func_obj.is_ref_return else REF_KIND_NONE
            return CallNode(self.token, member_name, args, func_obj.return_type, obj_node, ref_kind=ref_kind)
        else:
            # Field access
            field_type = registry.get_field_type(struct_name, member_name)
            if field_type is None:
                self.error("Field '%s' not found in struct '%s'" % (member_name, struct_name))

            return MemberAccessNode(self.token, obj_node, member_name, field_type, obj_ref_kind)

    def function_declaration(self):
        """Parse a function declaration or method definition"""

        if self.current_function != -1:
            self.error("Nested function declarations are not allowed")

        self.advance()  # Skip 'def'

        # Parse function/method name
        if self.token.type != TT_IDENT:
            self.error("Expected name after 'def'")

        name = self.token.value
        struct_name = None

        self.advance()

        # Check if it's a method (has a dot after struct name)
        if self.token.type == TT_DOT:
            struct_name = name

            # Verify struct exists
            if not registry.struct_exists(struct_name):
                self.error("Struct '%s' is not defined" % struct_name)

            # Parse method name after the dot
            self.advance()  # Skip the dot
            if self.token.type != TT_IDENT:
                self.error("Expected method name after '.'")

            name = self.token.value
            self.advance()

        elif name == "main":
            self.seen_main_function = True

        # Parse parameters
        self.consume(TT_LPAREN)
        params = []

        while self.token.type != TT_RPAREN:
            tmp = self.parameter()

            # For methods, check if parameter name is 'self'
            if struct_name and tmp[0] == "self":
                self.error("Cannot use 'self' as a parameter name, it is reserved")

            for n, _, _ in params:
                if n == tmp[0]:
                    self.error("Parameter '%s' is already defined" % n)

            params.append(tmp)

            if self.token.type == TT_COMMA:
                self.advance()  # Skip comma
                if self.token.type == TT_RPAREN:  # Handle trailing comma
                    break

        self.consume(TT_RPAREN)

        # Parse return type (if specified)
        return_type = TYPE_VOID  # Default to void
        is_ref_return = False    # Default to non-reference return

        if self.token.type == TT_COLON:
            self.advance()
            # Check for byref return type
            if self.token.type == TT_BYREF:
                is_ref_return = True
                self.advance()  # Consume 'byref'
            return_type = self.parse_type_reference()

        struct_id = -1
        # Special checks for methods
        if struct_name:
            struct_id = registry.get_struct_id(struct_name)
            # Check constructor and destructor constraints
            if name == "init" and return_type != TYPE_VOID:
                self.error("Constructor 'init' must have void return type")
            elif name == "fini":
                if return_type != TYPE_VOID:
                    self.error("Destructor 'fini' must have void return type")
                if len(params) > 0:
                    self.error("Destructor 'fini' cannot have parameters")

        if registry.lookup_function(name, struct_id, check_parents=False) != -1:
                self.error("Function '%s' is already defined" % name if not struct_name else "%s.%s"%(struct_name, name))

        # Enter function/method scope
        self.env.enter_scope()
        prev_function = self.current_function
        ref_kind = REF_KIND_GENERIC if is_ref_return else REF_KIND_NONE
        self.current_function = registry.register_function(name, return_type, params, parent_struct_id=struct_id, is_ref_return=is_ref_return)
        # create a new node with empty body - we'll add it later
        # That's needed so type checking inside the body can find it
        node = FunctionDeclNode(self.token, name, params, return_type, body=None, parent_struct_id=struct_id, ref_kind=ref_kind)
        registry.set_function_ast_node(self.current_function, node)

        if struct_name:
            # Insert 'self' as first parameter (implicitly byref)
            params.insert(0, ("self", struct_id, True))

        # Add parameters to scope
        for param_name, param_type, param_is_byref in params:
            ref_kind = REF_KIND_GENERIC if param_is_byref else REF_KIND_NONE
            self.declare_variable(param_name, param_type, is_const=False, ref_kind=ref_kind)

        # Parse function/method body
        body = self.doblock()

        # Restore previous context
        self.env.leave_scope()
        self.current_function = prev_function

        # update the node body and return it
        node.body = body
        return node

    def parameter(self):
        """Parse a function parameter (name:type)"""
        is_byref = False

        # Check for byref keyword
        if self.token.type == TT_BYREF:
            is_byref = True
            self.advance()  # Consume 'byref'

        if self.token.type != TT_IDENT:
            self.error("Expected parameter name")

        name = self.token.value
        self.advance()

        # Parse type - REQUIRED
        if self.token.type != TT_COLON:
            self.error("Function parameters require explicit type annotation")
        self.advance() # Skip colon
        param_type = self.parse_type_reference()

        return (name, param_type, is_byref)

    def parse_type_reference(self):
        """Parse a type reference (tuple, primitive or struct)"""
        # Check for tuple types: {type1, type2, ...}
        if self.token.type == TT_LBRACE:
            return self.parse_tuple_type()

        # Regular types (primitive or struct)
        if self.token.type in TYPE_TOKEN_MAP:
            type_id = TYPE_TOKEN_MAP[self.token.type]
            self.advance()
            # Check for array syntax: Type[N] or Type[]
            if self.token.type == TT_LBRACKET:
                return self.parse_array_dimensions(type_id)
            return type_id
        elif self.token.type == TT_IDENT:
            type_name = self.token.value
            if registry.struct_exists(type_name):
                type_id = registry.get_struct_id(type_name)
                self.advance()
                # Check for array syntax: Type[N] or Type[]
                if self.token.type == TT_LBRACKET:
                    return self.parse_array_dimensions(type_id)
                return type_id
            else:
                self.error("Unknown type '%s'" % type_name)
        else:
            self.error("Expected a type")

    def parse_array_dimensions(self, element_type):
        """Parse array dimensions: [size] or []"""
        self.advance()  # Skip '['

        # Parse dimension if provided
        size = None  # Default to dynamic size
        if self.token.type == TT_INT_LITERAL:
            size = self.token.value
            if size <= 0:
                self.error("Array dimension must be positive")
            self.advance()

        # Consume closing bracket
        self.consume(TT_RBRACKET)

        # Register array type
        return registry.register_array_type(element_type, size)

    def struct_definition(self):
        """Parse a struct definition: struct Name [:ParentName] do ... end"""
        self.advance()  # Skip 'struct' keyword

        if self.token.type != TT_IDENT:
            self.error("Expected struct name after 'struct'")

        struct_name = self.token.value
        self.advance()

        # Check for parent struct (inheritance)
        parent_name = None
        if self.token.type == TT_COLON:
            self.advance()  # Skip ':'

            if self.token.type != TT_IDENT:
                self.error("Expected parent struct name after ':'")

            parent_name = self.token.value

            # Verify parent struct exists
            if not registry.struct_exists(parent_name):
                self.error("Parent struct '%s' is not defined" % parent_name)

            self.advance()

        # Register the struct
        struct_id = registry.register_struct(struct_name, parent_name, self.token)

        # Parse struct body
        self.skip_separators()
        self.consume(TT_DO)

        fields = []
        while self.token.type != TT_END:
            if self.token.type == TT_EOF:
                self.error("Unexpected end of file in struct definition")

            self.skip_separators()

            # Check for end of struct
            if self.token.type == TT_END:
                break

            # Parse field: name:type;
            if self.token.type != TT_IDENT:
                self.error("Expected field name in struct definition")

            field_name = self.token.value
            self.advance()

            self.consume(TT_COLON)

            # Parse field type
            field_type = self.parse_type_reference()

            # Register field
            registry.add_field(struct_name, field_name, field_type, self.token)
            fields.append((field_name, field_type))

            # If on the same line, require a semicolon
            if self.token.type != TT_NEWLINE and self.token.type != TT_EOF and self.token.type != TT_END:
                self.consume(TT_SEMI)

            self.skip_separators()

        self.advance()  # Skip 'end'

        return StructDefNode(self.token, struct_name, parent_name, fields, struct_id)

    def register_tuple_type(self, elements_or_types, is_type_annotation=False):
        """
        Register a tuple type with the given elements
        Works for both initializer expressions and tuple type annotations

        Args:
            elements_or_types: List of type IDs or expression nodes
            is_type_annotation: Whether this is a type annotation (True) or expression (False)

        Returns:
            The struct ID for the tuple type
        """
        if is_type_annotation:
            # Element types are already type IDs (from type annotations)
            type_list = elements_or_types
        else:
            # Elements are expression nodes, extract types
            type_list = []
            for elem in elements_or_types:
                if elem.node_type == AST_NODE_GENERIC_INITIALIZER:
                    type_list.append(elem.expr_type)
                else:
                    type_list.append(elem.expr_type)

        # Generate a unique struct type name for this tuple
        elements_str = "_".join(registry.var_type_to_string(t) for t in type_list)
        struct_name = "_tuple_%d_%s" % (len(type_list), elements_str)

        # Register the tuple as an anonymous struct if not already registered
        if not registry.struct_exists(struct_name):
            struct_id = registry.register_struct(struct_name, None, self.token)
            for i, element_type in enumerate(type_list):
                registry.add_field(struct_name, "_%d" % i, element_type)
        else:
            struct_id = registry.get_struct_id(struct_name)

        return struct_id

    def parse_tuple_type(self):
        """Parse a tuple type annotation: {type1, type2, ...}"""
        self.advance()  # Skip '{'

        element_types = []

        # Parse first element type
        element_types.append(self.parse_type_reference())

        # Parse remaining element types
        while self.token.type == TT_COMMA:
            self.advance()  # Skip comma
            element_types.append(self.parse_type_reference())

        self.consume(TT_RBRACE)

        # Register and return the tuple type ID
        return self.register_tuple_type(element_types, is_type_annotation=True)

    def parse_initializer_expression(self, target_type=TYPE_UNKNOWN, consume_token=True):
        """Parse an initializer expression: {expr1, expr2, ...} or {{...}, {...}}"""
        if consume_token: self.consume(TT_LBRACE)  # Skip '{'
        self.skip_newlines()
        if target_type == TYPE_UNKNOWN: target_type = self.current_initializer_type

        elements = []
        subtype = INITIALIZER_SUBTYPE_TUPLE  # Default

        # Determine subtype based on target_type
        if target_type != TYPE_UNKNOWN:
            if registry.is_array_type(target_type):
                subtype = INITIALIZER_SUBTYPE_LINEAR
                # Get element type for array elements
                element_type = registry.get_array_element_type(target_type)
            elif registry.is_struct_type(target_type):
                subtype = INITIALIZER_SUBTYPE_LINEAR
                # Check if struct has a constructor - if so, initializer is not allowed
                if registry.get_method(target_type, "init"):
                    struct_name = registry.get_struct_name(target_type)
                    self.error("Cannot use initializer for struct '%s' because it has a constructor" % struct_name)

        # Empty initializer not allowed (for now)
        if self.token.type == TT_RBRACE:
            self.error("Empty initializers are not supported")

        # Parse elements recursively
        while True:
            if self.token.type == TT_LBRACE:
                # Nested initializer - derive element type based on context
                element_type = TYPE_UNKNOWN
                if subtype == INITIALIZER_SUBTYPE_LINEAR:
                    if registry.is_array_type(target_type):
                        element_type = registry.get_array_element_type(target_type)
                    elif registry.is_struct_type(target_type):
                        # Get field type for the current index
                        field_index = len(elements)
                        struct_name = registry.get_struct_name(target_type)
                        fields = registry.get_all_fields(struct_name)
                        if field_index < len(fields):
                            _, element_type = fields[field_index]

                # Recursively parse nested initializer
                nested_init = self.parse_initializer_expression(element_type)
                elements.append(nested_init)
            else:
                # Regular expression element
                elements.append(self.expression(0))

            self.skip_newlines()
            # Check for end of initializer or comma
            if self.token.type != TT_COMMA:
                break

            self.advance()  # Skip comma
            self.skip_newlines()

        self.consume(TT_RBRACE)

        # Create initializer node
        init_node = GenericInitializerNode(self.token, elements, subtype, target_type)

        # Handle array dimension inference for arrays with inferred size
        if (subtype == INITIALIZER_SUBTYPE_LINEAR and
                registry.is_array_type(target_type) and
                registry.get_array_size(target_type) is None):
            # Get element type and inferred size
            element_type = registry.get_array_element_type(target_type)
            inferred_size = len(elements)
            # Create a sized array type
            sized_array_type = registry.register_array_type(element_type, inferred_size)
            # Update target and result type
            target_type = sized_array_type
            init_node.target_type = sized_array_type
            init_node.expr_type = sized_array_type

        # Handle type inference for tuple initializers
        if subtype == INITIALIZER_SUBTYPE_TUPLE:
            # Register anonymous struct type
            struct_id = self.register_tuple_type(elements, is_type_annotation=False)
            init_node.expr_type = struct_id

        # Type validation for LINEAR initializers
        if subtype == INITIALIZER_SUBTYPE_LINEAR and registry.is_struct_type(target_type):
            struct_name = registry.get_struct_name(target_type)
            fields = registry.get_all_fields(struct_name)

            # Validate field count - too many elements is an error
            if len(elements) > len(fields):
                self.error("Initializer for %s has %d elements, but struct has only %d fields" % 
                          (struct_name, len(elements), len(fields)))

            # Fill in missing fields with zero values
            if len(elements) < len(fields):
                for i in range(len(elements), len(fields)):
                    _, field_type = fields[i]
                    elements.append(self.create_default_value(field_type))

            # Validate field types
            for i, elem in enumerate(elements):
                _, field_type = fields[i]
                if not can_promote(elem.expr_type, field_type):
                    self.type_mismatch_error("Field %d in %s initializer" % (i+1, struct_name), 
                                            elem.expr_type, field_type)

        # Type validation for LINEAR initializers with array types
        elif subtype == INITIALIZER_SUBTYPE_LINEAR and registry.is_array_type(target_type):
            element_type = registry.get_array_element_type(target_type)
            array_size = registry.get_array_size(target_type)

            # Validate element count for fixed-size arrays
            if array_size is not None and len(elements) > array_size:
                self.error("Array initializer has %d elements, but array has size %d" % 
                          (len(elements), array_size))

            # Validate element types
            for i, elem in enumerate(elements):
                if not can_promote(elem.expr_type, element_type):
                    self.type_mismatch_error("Element %d in array initializer" % (i+1),
                                            elem.expr_type, element_type)

        return init_node

    def parse_constructor_call(self, struct_name, struct_id):
        """Parse a constructor call and return a StructInitNode with properly handled arguments."""
        # Verify struct parentheses
        if self.token.type != TT_LPAREN:
            self.error("Constructor invocation requires parentheses")

        # Parse constructor arguments
        self.advance()  # Skip '('
        args = []
        self.parse_args(args)
        self.consume(TT_RPAREN)

        # Check if init method exists for the struct - this MUST exist
        init_method = registry.get_method(struct_id, "init")
        # we allow StructName() without arguments as a convenient shortcut to get a zero-initialized struct.
        if not init_method and len(args) > 0:
            self.error("Cannot use constructor syntax for struct '%s' - no init method defined. Use initializer syntax {...} instead." % struct_name)

        if init_method:
            # Add self placeholder to arguments list
            args_with_self = [VariableNode(self.token, "", struct_id)]  # Placeholder for self
            args_with_self.extend(args)
            args = args_with_self

            # Perform type checking
            self.check_arg_count("Constructor for '%s'" % struct_name, init_method.params, args, is_method=True)
            self.check_argument_types(args, init_method.params, "Constructor for '%s'" % struct_name)

        # Create and return struct initialization node with self included in args
        return StructInitNode(self.token, struct_name, struct_id, args)

    def nud(self, t):
        # Handle number literals using the type mapping
        if t.type in TOKEN_TO_TYPE_MAP:
            return NumberNode(t, t.value, TOKEN_TO_TYPE_MAP[t.type])

        if t.type == TT_STRING_LITERAL:
            return StringNode(t, t.value)

        if t.type == TT_NIL:
            return NilNode(t)

        if t.type == TT_IDENT:
            var_name = t.value

            # Check if it's a struct type name (for initialization)
            if registry.struct_exists(var_name):
                struct_id = registry.get_struct_id(var_name)

                # Parse initializer: StructName() or StructName(arg1, arg2, ...)
                if self.token.type == TT_LPAREN:
                    return self.parse_constructor_call(var_name, struct_id)

            # For a variable in an expression context:
            # Could be a function name.
            func_id = registry.lookup_function(var_name)
            if func_id != -1:
                if self.token.type == TT_LPAREN:
                    return self.funccall(var_name)
                return_type = registry.get_func_from_id(func_id).return_type
                return VariableNode(t, var_name, return_type)

            # It's a variable name, see if it's declared
            if not self.is_variable_declared(var_name):
                self.error("Variable '%s' is not declared" % var_name)

            # Get the variable type from the appropriate scope
            var_type = self.get_variable_type(var_name)
            # Get the variable's reference kind
            ref_kind = self.get_variable_ref_kind(var_name)

            # All variables can be referenced, even if not declared with byref
            # This allows functions to return references to variables
            variable_node = VariableNode(t, var_name, var_type, ref_kind)

            # Mark variables as referenceable for byref return functions
            if self.current_function != -1:
                func_obj = registry.get_func_from_id(self.current_function)
                if func_obj.is_ref_return:
                    # Variables can be returned as references
                    variable_node.ref_kind = REF_KIND_GENERIC

            return variable_node

        if t.type in [TT_MINUS, TT_NOT, TT_BITNOT]:  # Unary operators
            expr = self.expression(UNARY_PRECEDENCE)
            return UnaryOpNode(t, t.value, expr, expr.expr_type)

        if t.type == TT_LBRACE:
            return self.parse_initializer_expression(consume_token=False)

        if t.type == TT_LPAREN:
            expr = self.expression(0)
            self.consume(TT_RPAREN)
            return expr

        if t.type == TT_NEW:
            # Parse the struct name after 'new'
            is_primitive_type = self.token.type in TYPE_TOKEN_MAP
            if self.token.type != TT_IDENT and not is_primitive_type:
                self.error("Expected type name after 'new'")

            type_name = self.token.value

            # Verify type exists
            if not is_primitive_type and not registry.struct_exists(type_name):
                self.error("Type '%s' is not defined" % type_name)

            # Get the struct type ID
            type_id = registry.get_struct_id(type_name)
            if is_primitive_type and type_id == -1: type_id = TYPE_TOKEN_MAP[self.token.type]

            self.advance()

            # Check if this is an array allocation: new Type[N]
            if self.token.type == TT_LBRACKET:
                # Parse array dimensions - this creates the array type
                array_type_id = self.parse_array_dimensions(type_id)

                # Get element type and array size
                element_type = registry.get_array_element_type(array_type_id)
                array_size = registry.get_array_size(array_type_id)

                if array_size is None:
                    self.error("Array size must be specified for new operator")

                # Create array elements
                elements = []

                constructor_node = None

                # Check if we need to call constructors (has parentheses)
                if self.token.type == TT_LPAREN:
                    # Parse the constructor call once
                    constructor_node = self.parse_constructor_call(type_name, element_type)

                # Initialize all elements
                for i in range(array_size):
                    if constructor_node:
                        # re-use the constructor node for all elements
                        # WARNING: this requires that the interpreter NOT modify it!
                        elements.append(constructor_node)
                    else:
                        # Default initialization (zero values)
                        elements.append(self.create_default_value(element_type))

                # Return the array with properly initialized elements
                return NewNode(t, GenericInitializerNode(t, elements, INITIALIZER_SUBTYPE_LINEAR, array_type_id))

            # Check if this is a constructor call
            elif self.token.type == TT_LPAREN:
                return NewNode(t, self.parse_constructor_call(type_name, type_id))

            # likely a primitive type
            return NewNode(t, self.create_default_value(type_id))

        raise CompilerException('Unexpected token %s' % token_name(t), t)

    def led(self, t, left):
        # Handle dot operator for member access
        if t.type == TT_DOT:
            return self.parse_member_access(left)

        # Handle array indexing
        if t.type == TT_LBRACKET:
            return self.parse_array_access(left, consume_lbracket=False)

        # Handle function call
        if t.type == TT_LPAREN and left.node_type == AST_NODE_VARIABLE:
            return self.funccall(left.name, consume_lparen=False)

        # Handle assignment as an operator
        if self.is_assignment_operator(t.type) and left.node_type == AST_NODE_VARIABLE:
            # Get variable name and info
            var_name = left.name
            var_type = self.get_variable_type(var_name)
            var_ref_kind = self.get_variable_ref_kind(var_name)

            # Create a proper variable node for handling
            var_node = VariableNode(t, var_name, var_type, var_ref_kind)

            # Use our centralized assignment handler
            return self.handle_assignment(var_node, var_type, ASSIGN_CTX_VARIABLE, t.type)

        # Handle member assignment (obj.field = value)
        elif self.is_assignment_operator(t.type) and left.node_type == AST_NODE_MEMBER_ACCESS:
            # Use our centralized assignment handler
            return self.handle_assignment(left, left.expr_type, ASSIGN_CTX_MEMBER_ACCESS, t.type)

        # Handle array element assignment (arr[idx] = value)
        elif self.is_assignment_operator(t.type) and left.node_type == AST_NODE_ARRAY_ACCESS:
            # Use our centralized assignment handler
            return self.handle_assignment(left, left.expr_type, ASSIGN_CTX_ARRAY_ELEMENT, t.type)

        # Handle function call result assignment
        elif self.is_assignment_operator(t.type) and left.node_type == AST_NODE_CALL:
            # Verify this is a reference function
            if left.ref_kind == REF_KIND_NONE:
                self.error("Cannot assign to a non-reference value")
            return self.handle_assignment(left, left.expr_type, ASSIGN_CTX_FUNCTION_RESULT, t.type)

        # Handle type-inference assignment (:=)
        if t.type == TT_TYPE_ASSIGN and left.node_type == AST_NODE_VARIABLE:
            var = left.name
            # Check if variable is already declared - if so, this is an error
            if self.is_variable_declared(var):
                self.error("Cannot use ':=' with already declared variable '%s'. Use '=' instead" % var)
            # Type inference doesn't make sense in expression context without a proper declaration
            # This is unusual, but we'll handle the error here to ensure consistent behavior
            self.error("Type inference assignment (:=) is only valid in variable declarations")

        if t.type in [TT_PLUS, TT_MINUS, TT_MULT, TT_DIV, TT_MOD, TT_SHL, TT_SHR]:
            right = self.expression(self.lbp(t))

            # Determine the result type based on operand types
            result_type = self.calculate_result_type(t.value, left.expr_type, right.expr_type)
            if result_type is None:
                self.type_mismatch_error("Type mismatch in binary operation", left.expr_type, right.expr_type)

            return BinaryOpNode(t, t.value, left, right, result_type)

        elif t.type in [TT_EQ, TT_NE, TT_GE, TT_LE, TT_LT, TT_GT]:
            right = self.expression(self.lbp(t))
            # Comparisons always return an integer (0/1 representing false/true)
            # But operands must be compatible types
            result_type = self.calculate_result_type(t.value, left.expr_type, right.expr_type)
            if result_type is None:
                self.type_mismatch_error("Type mismatch in comparison", left.expr_type, right.expr_type)
            return CompareNode(t, t.value, left, right)

        elif t.type in [TT_AND, TT_OR]:
            right = self.expression(self.lbp(t))
            # Logical operations always return an integer (0/1 representing false/true)
            return LogicalNode(t, t.value, left, right)

        elif t.type in [TT_XOR, TT_BITOR, TT_BITAND]:
            right = self.expression(self.lbp(t))
            # Both operands must be integer types
            if is_integer_type(left.expr_type) and is_integer_type(right.expr_type):
                result_type = self.calculate_result_type(t.value, left.expr_type, right.expr_type)
                if result_type is not None:
                    return BitOpNode(t, t.value, left, right)
            self.error("Bitwise operators require integer operands")

        raise CompilerException('Unexpected token type %d' % t.type, t)

    def parse_array_access(self, array_node, consume_lbracket=True):
        """Parse array access: array[index]"""
        # Skip '[' if needed
        if consume_lbracket:
            self.consume(TT_LBRACKET)

        # Check if type is an array
        if not registry.is_array_type(array_node.expr_type):
            self.error("Cannot index non-array type %s" %
                      registry.var_type_to_string(array_node.expr_type))

        # Parse index expression
        index_expr = self.expression(0)

        # Ensure index is an integer type
        if not is_integer_type(index_expr.expr_type):
            self.error("Array index must be an integer type")

        # Consume the closing bracket
        self.consume(TT_RBRACKET)

        # Get the element type
        element_type = registry.get_array_element_type(array_node.expr_type)

        # Create and return the array access node
        access_node = ArrayAccessNode(self.token, array_node, index_expr, element_type)

        # Handle reference kind propagation - array elements can be referenced for modification
        if array_node.ref_kind != REF_KIND_NONE:
            access_node.ref_kind = array_node.ref_kind

        return access_node

    def parse_type(self):
        """Parse a type annotation or return None if not present"""
        if self.token.type == TT_COLON:
            self.advance()  # Consume the colon

            return self.parse_type_reference()
        return TYPE_UNKNOWN

    def skip_separators(self):
        while self.token.type == TT_SEMI or self.token.type == TT_NEWLINE:
            self.advance() # Skip empty lines or semicolon (as statement separator)

    def skip_newlines(self):
        """Skip just newlines (used inside expressions that can span multiple lines)"""
        while self.token.type == TT_NEWLINE:
            self.advance()

    def doblock(self):
        self.skip_separators()
        self.consume(TT_DO)
        body = []
        while self.token.type != TT_END:
            if self.token.type == TT_EOF:
                self.error("Unexpected end of file while parsing a block (missing 'end')")
            if self.token.type == TT_ELSE:
                self.error("Missing 'end' before 'else'")
            self.skip_separators()  # Skip any separators before checking for END again
            if self.token.type == TT_END: break
            stmt = self.statement()
            body.append(stmt)
        self.advance()
        self.skip_separators()
        return body

    def if_statement(self):
            self.advance()
            condition = self.expression(0)
            then_body = self.doblock()
            if self.token.type != TT_ELSE:
                return IfNode(self.token, condition, then_body, None)

            self.advance()
            self.skip_separators()

            # Handle both "else if" and "else do" cases
            if self.token.type == TT_IF:
                # Parse the nested if as part of else body
                else_body = [self.if_statement()]
            elif self.token.type == TT_DO:
                else_body = self.doblock()
            else:
                self.error("Expected 'if' or 'do' after 'else'")
            return IfNode(self.token, condition, then_body, else_body)

    def funccall(self, func_name, consume_lparen=True):
        """Parse a function call and return a CallNode"""
        func_id = registry.lookup_function(func_name)
        if func_id == -1:
                self.error("'%s' is not a function" % func_name)

        if consume_lparen: self.consume(TT_LPAREN)

        # Parse arguments
        args = []
        self.parse_args(args)
        self.consume(TT_RPAREN)

        # Type checking for function call
        func_obj = registry.get_func_from_id(func_id)
        func_params = func_obj.params
        func_return_type = func_obj.return_type

        # Check number of arguments
        self.check_arg_count("Function '%s'" % func_name, func_params, args, is_method=False)
        self.check_argument_types(args, func_params, "function '%s'" % func_name)
        ref_kind = REF_KIND_GENERIC if func_obj.is_ref_return else REF_KIND_NONE
        return CallNode(self.token, func_name, args, func_return_type, ref_kind=ref_kind)

    def is_assignment_operator(self, token_type):
        """Check if a token is an assignment operator"""
        return token_type in [TT_ASSIGN, TT_PLUS_ASSIGN, TT_MINUS_ASSIGN,
                             TT_MULT_ASSIGN, TT_DIV_ASSIGN, TT_MOD_ASSIGN]

    def handle_assignment(self, lhs, target_type, context_type, op):
        """
        Handle an assignment operation based on context type.

        Args:
            lhs: The left-hand side node being assigned to
            target_type: The type to assign to
            context_type: The context in which this assignment occurs
            op: the assignment operator token, e.g. TT_DIV_ASSIGN

        Returns:
            A BinaryOpNode representing the assignment
        """
        ref_kind = lhs.ref_kind

        # Pre-checks based on context
        if context_type == ASSIGN_CTX_FUNCTION_RESULT:
            if lhs.ref_kind == REF_KIND_NONE:
                self.error("Cannot assign to a non-reference value")
        elif context_type == ASSIGN_CTX_VARIABLE:
            var_name = lhs.name
            if self.is_constant(var_name):
                self.error("Cannot reassign to constant '%s'"%var_name)
        elif context_type == ASSIGN_CTX_MEMBER_ACCESS:
            # Check if trying to assign to a field of a constant variable
            if lhs.obj.node_type == AST_NODE_VARIABLE:
                obj_name = lhs.obj.name
                if self.is_constant(obj_name):
                    self.error("Cannot modify field '%s' of constant '%s'" %
                               (lhs.member_name, obj_name))

        old_initializer_type = self.current_initializer_type
        self.current_initializer_type = target_type

        expr = self.expression(0)

        self.current_initializer_type = old_initializer_type

        # Post-checks based on context
        if context_type == ASSIGN_CTX_MEMBER_ACCESS:
            self.check_field_compatibility(expr.expr_type, target_type)
        elif context_type == ASSIGN_CTX_ARRAY_ELEMENT:
            if not can_promote(expr.expr_type, target_type):
                self.type_mismatch_error("Array element assignment", expr.expr_type, target_type)
        elif context_type == ASSIGN_CTX_VARIABLE:
            self.check_type_compatibility(lhs.name, expr)

        # Handle compound operators
        if op != TT_ASSIGN:
            # Get the operator for this compound assignment
            binary_op = get_operator_for_compound_assign(op)

            # Create a binary operation node for the operation
            binary_expr = BinaryOpNode(self.token, binary_op, lhs, expr, target_type)

            # Create and return assignment node with binary expression
            return BinaryOpNode(self.token, '=', lhs, binary_expr, target_type, ref_kind)
        else:
            # Regular assignment
            return BinaryOpNode(self.token, '=', lhs, expr, target_type, ref_kind)

    def typedef_declaration(self):
        """Parse typedef Name: Type"""
        # TODO: parse also function prototype typedefs looking like:
        # typedef callback:(x:int, y:int):long
        self.advance()  # Skip 'typedef'

        # Parse the alias name
        if self.token.type != TT_IDENT:
            self.error("Expected identifier after 'typedef'")

        alias_name = self.token.value
        current_token = self.token  # Save for error reporting
        self.advance()

        # Parse the colon and target type
        self.consume(TT_COLON)

        # Parse the target type (primitive or user-defined)
        target_type_id = self.parse_type_reference()

        # Register the typedef
        registry.register_typedef(alias_name, target_type_id, current_token)

        self.check_statement_end()
        # return a NOP node :)
        return ExprStmtNode(self.token, NumberNode(self.token, 0, TYPE_INT))

    def parse_var_declaration(self, var_name, decl_type):
        """Parse a variable declaration after the identifier"""
        # Process type annotation if present
        var_type = self.parse_type()  # This will consume the type if present

        # Check what kind of assignment we have
        is_type_inference = (self.token.type == TT_TYPE_ASSIGN)

        if is_type_inference:
            # Type inference assignment (:=)
            self.advance()  # Skip the := operator
        elif self.token.type == TT_ASSIGN:
            # Regular assignment with explicit type (=)
            if var_type == TYPE_UNKNOWN:
                self.error("Variable declaration with '=' requires explicit type annotation")
            self.advance()  # Skip the = sign
        else:
            # No initializer provided - check if we have a type annotation
            if var_type == TYPE_UNKNOWN:
                self.error("Variable declaration must include either a type annotation or initialization")

            # Create default initialization based on the type
            expr = self.create_default_value(var_type)

            # Set appropriate reference kind for dynamic arrays
            ref_kind = REF_KIND_GENERIC if (registry.is_array_type(var_type) and registry.get_array_size(var_type) is None) else REF_KIND_NONE

            if self.env.get(var_name, all_scopes=False):
                self.already_declared_error(var_name)
            self.declare_variable(var_name, var_type, decl_type == TT_CONST, ref_kind=ref_kind)

            # Skip to expression evaluation logic
            return VarDeclNode(self.token, decl_type, var_name, var_type, expr, ref_kind=ref_kind)

        # Use the current_initializer_type pattern
        old_initializer_type = self.current_initializer_type
        self.current_initializer_type = TYPE_UNKNOWN if is_type_inference else var_type
        expr = self.expression(0)
        self.current_initializer_type = old_initializer_type

        # In global scope, ensure only literal initializers
        if self.current_function == -1 and self.seen_main_function:
            self.error("Global variables must be declared before main function")

        if self.current_function == -1 and not is_literal_node(expr):
            self.error("Global variables can only be initialized with literals")

        # Handle type resolution based on assignment type
        if is_type_inference:
            # Check if expression is void
            if expr.expr_type == TYPE_VOID:
                self.error("Cannot assign void expression to variable '%s'" % var_name)
            # Infer the type from expression
            if expr.node_type == AST_NODE_VARIABLE:
                # Get type from referenced variable
                ref_var = expr.name
                var_type = self.get_variable_type(ref_var)
                if var_type == TYPE_UNKNOWN:
                    self.error("Cannot infer type from variable '%s' with unknown type" % ref_var)
            elif expr.node_type == AST_NODE_STRUCT_INIT:
                # Set var_type to struct type
                var_type = expr.struct_id
            else: # for everything else, infer the type from the expression
                var_type = expr.expr_type
        else:
            # For size-inferred arrays, update the variable's type with the inferred size
            if (registry.is_array_type(var_type) and
                registry.get_array_size(var_type) is None and
                registry.is_array_type(expr.expr_type)):
                # Get element types
                var_elem_type = registry.get_array_element_type(var_type)
                init_elem_type = registry.get_array_element_type(expr.expr_type)

                # Ensure element types are compatible
                if can_promote(init_elem_type, var_elem_type):
                    # Get the inferred size
                    inferred_size = registry.get_array_size(expr.expr_type)

                    # Create a sized array with the variable's element type
                    var_type = registry.register_array_type(var_elem_type, inferred_size)

            # Check type compatibility with expression type
            # Special cases for literals in declarations
            if expr.node_type == AST_NODE_NUMBER:
                if expr.expr_type == TYPE_INT:
                    # Allow int literals to initialize any numeric type, but not string nor structs
                    if var_type not in FLOAT_TYPES and var_type not in UNSIGNED_TYPES and var_type not in SIGNED_TYPES:
                        self.type_mismatch_error("Type mismatch in initialization", expr.expr_type, var_type)
                elif expr.expr_type == TYPE_DOUBLE and var_type == TYPE_FLOAT:
                    # Allow double literals to initialize float variables
                    pass
            elif expr.expr_type != TYPE_UNKNOWN and var_type != expr.expr_type and not can_promote(expr.expr_type, var_type):
                self.type_mismatch_error("Type mismatch in initialization", expr.expr_type, var_type)

        # Register the variable as defined in the current scope
        if self.env.get(var_name, all_scopes=False):
            self.already_declared_error(var_name)

        # If initializing with a reference (from new), keep the reference kind
        ref_kind = expr.ref_kind

        # Declare the variable in current scope
        self.declare_variable(var_name, var_type, decl_type == TT_CONST, ref_kind=ref_kind)

        return VarDeclNode(self.token, decl_type, var_name, var_type, expr, ref_kind=ref_kind)

    def parse_enum_declaration(self):
        """Parse an enum declaration and transform it into struct + const"""
        # Check if we're in global scope
        if self.current_function != -1:
            self.error("Enum definitions are not allowed inside functions")

        self.advance()  # Skip 'enum' token

        # Parse enum name
        if self.token.type != TT_IDENT:
            self.error("Expected enum name")

        enum_name = self.token.value
        self.advance()

        # Create hidden struct name for enum implementation
        hidden_struct_name = "__enum_%s" % enum_name

        # Check if a struct with this name already exists
        for n in [enum_name, hidden_struct_name]:
            if registry.struct_exists(enum_name):
                self.error("Type '%s' is already defined" % n)

        # Parse optional base type
        base_type = TYPE_INT  # Default
        if self.token.type == TT_COLON:
            self.advance()
            base_type = self.parse_type_reference()

        # Create builder to track state during parsing
        builder = EnumBuilder(enum_name, base_type)

        # Parse enum body
        self.consume(TT_DO)
        self.skip_separators()

        # Parse enum members
        while self.token.type not in [TT_END, TT_EOF]:
            if self.token.type != TT_IDENT:
                self.error("Expected enum member identifier")

            member_name = self.token.value
            self.advance()

            # Check for explicit value
            if self.token.type == TT_ASSIGN:
                self.advance()
                # Call expression parser to evaluate the value
                value_expr = self.expression(0)

                # For now, only support literal values
                if value_expr.node_type != AST_NODE_NUMBER:
                    self.error("Only numeric literals supported for enum values")

                value = value_expr.value
                builder.add_member(member_name, value)
            else:
                # Auto-increment value
                builder.add_member(member_name)

            # Skip separator (semicolon or newline)
            self.skip_separators()

        if self.token.type != TT_END:
            self.error("Expected 'end' to close enum declaration")
        self.advance()

        # Register struct type
        struct_id = registry.register_struct(hidden_struct_name, None, self.token)

        # Register a typedef from enum_name to the enum's base type
        registry.register_typedef(enum_name, base_type, self.token)

        # Transform to struct declaration and const initialization
        struct_members = []
        init_values = []

        for name, value in builder.members:
            struct_members.append(VarDeclNode(self.token, TT_CONST, name, base_type, None))
            init_values.append(NumberNode(self.token, value, base_type))
            # register the fields of the struct in the type_registry
            registry.add_field(hidden_struct_name, name, base_type, self.token)

        # Create struct declaration for the hidden implementation
        struct_decl = StructDefNode(self.token, hidden_struct_name, None, [(name, base_type) for name, _ in builder.members], struct_id)

        # Register the enum in the environment so it can be referenced
        self.declare_variable(enum_name, struct_id, is_const=True)

        # Create const declaration with initializer
        initializer = GenericInitializerNode(self.token, init_values, INITIALIZER_SUBTYPE_LINEAR, struct_id)
        const_decl = VarDeclNode(self.token, TT_CONST, enum_name, struct_id, initializer)

        # We need to directly add the struct declaration to program statements
        # Then return the const declaration
        self.statements.append(struct_decl)
        return const_decl

    def statement(self):
        self.skip_separators()

        # Handle struct definitions
        if self.token.type == TT_STRUCT:
            # Only allowed in global scope
            if self.current_function != -1:
                self.error("Struct definitions are not allowed inside functions")
            return self.struct_definition()

        # Handle typedef definitions
        if self.token.type == TT_TYPEDEF:
            return self.typedef_declaration()

        # Handle function/method definitions
        if self.token.type == TT_DEF:
            return self.function_declaration()

        # Handle enum definitions
        if self.token.type == TT_ENUM:
            return self.parse_enum_declaration()

        # Handle return statements
        if self.token.type == TT_RETURN:
            # Must be inside a function
            if self.current_function == -1:
                self.error("'return' statement outside function")

            self.advance()

            # Get function return type for context
            current_func_obj = registry.get_func_from_id(self.current_function)
            func_return_type = current_func_obj.return_type

            # Return with no value
            if self.token.type in [TT_SEMI, TT_NEWLINE, TT_EOF] or (self.prev_token and self.token.line > self.prev_token.line):
                # Check if empty return is allowed
                if func_return_type != TYPE_VOID:
                    self.error("Non-void function '%s' must return a value" % current_func_obj.name)

                self.check_statement_end()
                return ReturnNode(self.token, None)

            # Return with value
            # Set context for expression parsing
            old_initializer_type = self.current_initializer_type
            self.current_initializer_type = func_return_type
            expr = self.expression(0)
            self.current_initializer_type = old_initializer_type

            if func_return_type == TYPE_VOID:
                fn = registry.get_func_from_id(self.current_function).name
                self.error("Void function '%s' cannot return a value" % fn)

            # Special handling for return statement type checking
            # Check if function returns by reference
            if current_func_obj.is_ref_return and expr.ref_kind == REF_KIND_NONE:
                self.error("Function with 'byref' return type must return a reference")

            # Prevent returning references to local variables
            if current_func_obj.is_ref_return and expr.node_type == AST_NODE_VARIABLE:
                var_name = expr.name
                # Check if this variable was declared in the current function's scope
                # (not a parameter or global)
                is_param = any(p[0] == var_name for p in current_func_obj.ast_node.params)
                if not (is_param or self.env.is_global(var_name)):
                    self.error("Cannot return a reference to a local variable")

            # Disallow returning references to struct fields
            if current_func_obj.is_ref_return and expr.node_type == AST_NODE_MEMBER_ACCESS:
                self.error("Returning references to struct fields is not supported")

            # Check that return expression type matches function return type
            if not can_promote(expr.expr_type, func_return_type):
                self.type_mismatch_error("Type mismatch in return", expr.expr_type, func_return_type)

            self.check_statement_end()
            return ReturnNode(self.token, expr, REF_KIND_GENERIC if current_func_obj.is_ref_return else REF_KIND_NONE)

        # Handle del statement for heap deallocation
        if self.token.type == TT_DEL:
            self.advance()
            expr = self.expression(0)

            # Verify expr is a heap reference
            if expr.ref_kind != REF_KIND_HEAP:
                self.error("'del' can only be used with reference types (created with 'new')")

            self.check_statement_end()
            return DelNode(self.token, expr)

        # Handle variable declarations (var and const)
        if self.token.type in [TT_VAR, TT_CONST]:
            decl_type = self.token.type  # Save the declaration type (var or const)
            self.advance()

            # Expect an identifier after var/const
            if self.token.type != TT_IDENT:
                self.error("Expected identifier after '%s'" % ('var' if decl_type == TT_VAR else 'const'))

            var_name = self.token.value
            self.advance()

            node = self.parse_var_declaration(var_name, decl_type)
            self.check_statement_end()
            return node

        if self.token.type == TT_IF:
            return self.if_statement()

        # Handle while loop
        elif self.token.type == TT_WHILE:
            self.advance()
            condition = self.expression(0)
            body = self.doblock()
            return WhileNode(self.token, condition, body)
        elif self.token.type == TT_PRINT:
            self.advance()
            expr = self.expression(0)
            self.check_statement_end()
            return PrintNode(self.token, expr)
        elif self.token.type == TT_BREAK:
            self.advance()
            self.check_statement_end()
            return BreakNode(self.token)
        elif self.token.type == TT_CONTINUE:
            self.advance()
            self.check_statement_end()
            return ContinueNode(self.token)
        else:
            # First check if we're in global scope, where only declarations are allowed
            if self.current_function == -1:
                self.error("Only variable declarations and function declarations are allowed in global scope")
            # handle everything else in the pratt expression parser
            expr = self.expression(0)
            self.check_statement_end()
            return ExprStmtNode(self.token, expr)

    def check_statement_end(self, allow_also=None):
        """Check if a statement is properly terminated by semicolon, newline, or EOF"""
        # Allow specific token (e.g. "do" for assignments in if/while conditions)
        if allow_also and self.token.type == globals().get('TT_' + allow_also.upper(), 0):
            return

        # Consume semicolon or newline if present
        if self.token.type == TT_SEMI or self.token.type == TT_NEWLINE:
            while self.token.type == TT_SEMI or self.token.type == TT_NEWLINE:
                self.advance()
            return

        # Check if we're at the end of a line or file
        if self.token.type != TT_EOF and self.prev_token and self.token.line == self.prev_token.line:
            self.error("Expected semicolon between statements on the same line")

    def parse(self):
        self.statements = []
        while self.token.type != TT_EOF:
            self.statements.append(self.statement())
        return self.statements


### END OF COMPILER.PY
