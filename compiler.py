# Implementation of a Pratt parser in Python 2.7

from shared import *
from lexer import Token, Lexer
from type_registry import get_registry

# import registry singleton
registry = get_registry()

# Base class for all AST nodes
class ASTNode(object):
    def __init__(self, node_type=AST_NODE_BASE):
        self.node_type = node_type
        self.expr_type = TYPE_UNKNOWN
        self.ref_kind = REF_KIND_NONE

    def __repr__(self):
        return "%s" % ast_node_type_to_string(self.node_type)

class NumberNode(ASTNode):
    def __init__(self, value, expr_type):
        ASTNode.__init__(self, AST_NODE_NUMBER)
        self.value = value
        self.expr_type = expr_type  # TYPE_INT, TYPE_FLOAT, etc.

    def __repr__(self):
        return "Number(%s, %s)" % (self.value, registry.var_type_to_string(self.expr_type))

class StringNode(ASTNode):
    def __init__(self, value):
        ASTNode.__init__(self, AST_NODE_STRING)
        self.value = value
        self.expr_type = TYPE_STRING

    def __repr__(self):
        return "String(\"%s\")" % self.value

class VariableNode(ASTNode):
    def __init__(self, name, var_type, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_VARIABLE)
        self.name = name
        self.expr_type = var_type
        self.ref_kind = ref_kind

    def __repr__(self):
        return "Var(%s, %s)" % (self.name, registry.format_type_with_ref_kind(self.expr_type))

class BinaryOpNode(ASTNode):
    def __init__(self, operator, left, right, result_type, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_BINARY_OP)
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
    def __init__(self, operator, operand, result_type):
        ASTNode.__init__(self, AST_NODE_UNARY_OP)
        self.operator = operator
        self.operand = operand
        self.expr_type = result_type

    def __repr__(self):
        return "UnaryOp(%s, %s) -> %s" % (
            self.operator, repr(self.operand), registry.var_type_to_string(self.expr_type)
        )

class AssignNode(ASTNode):
    def __init__(self, var_name, expr, var_type, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_ASSIGN)
        self.var_name = var_name
        self.expr = expr
        self.expr_type = var_type
        self.ref_kind = ref_kind

    def __repr__(self):
        return "Assign(%s, %s) -> %s" % (
            self.var_name, repr(self.expr), registry.format_type_with_ref_kind(self.expr_type)
        )

class CompoundAssignNode(ASTNode):
    def __init__(self, op_type, var_name, expr, var_type, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_COMPOUND_ASSIGN)
        self.op_type = op_type
        self.var_name = var_name
        self.expr = expr
        self.expr_type = var_type
        self.ref_kind = ref_kind

    def __repr__(self):
        op_name = token_name(self.op_type)
        return "CompoundAssign(%s, %s, %s) -> %s" % (
            op_name, self.var_name, repr(self.expr), registry.format_type_with_ref_kind(self.expr_type)
        )

class PrintNode(ASTNode):
    def __init__(self, expr):
        ASTNode.__init__(self, AST_NODE_PRINT)
        self.expr = expr

    def __repr__(self):
        return "Print(%s)" % repr(self.expr)

class IfNode(ASTNode):
    def __init__(self, condition, then_body, else_body=None):
        ASTNode.__init__(self, AST_NODE_IF)
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
    def __init__(self, condition, body):
        ASTNode.__init__(self, AST_NODE_WHILE)
        self.condition = condition
        self.body = body  # List of statement nodes

    def __repr__(self):
        return "While(%s, [%s])" % (
            repr(self.condition),
            ", ".join(repr(stmt) for stmt in self.body),
        )

class BreakNode(ASTNode):
    def __init__(self):
        ASTNode.__init__(self, AST_NODE_BREAK)

    def __repr__(self):
        return "Break()"

class ContinueNode(ASTNode):
    def __init__(self):
        ASTNode.__init__(self, AST_NODE_CONTINUE)

    def __repr__(self):
        return "Continue()"

class ExprStmtNode(ASTNode):
    def __init__(self, expr):
        ASTNode.__init__(self, AST_NODE_EXPR_STMT)
        self.expr = expr

    def __repr__(self):
        return "ExprStmt(%s)" % repr(self.expr)

class VarDeclNode(ASTNode):
    def __init__(self, decl_type, var_name, var_type, expr, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_VAR_DECL)
        self.decl_type = decl_type
        self.var_name = var_name
        self.var_type = var_type
        self.expr = expr
        self.ref_kind = ref_kind

    def __repr__(self):
        decl_type_str = "var" if self.decl_type == TT_VAR else "let"
        return "VarDecl(%s, %s, %s, %s)" % (
            decl_type_str, self.var_name, registry.var_type_to_string(self.var_type), repr(self.expr)
        )

class FunctionDeclNode(ASTNode):
    def __init__(self, name, params, return_type, body, parent_struct_id=-1):
        ASTNode.__init__(self, AST_NODE_FUNCTION_DECL)
        self.name = name
        self.params = params  # List of (name, type) tuples
        self.return_type = return_type
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
        if hasattr(self, 'is_ref_return') and self.is_ref_return:
            return_type_str = "byref " + return_type_str

        return "%s(%s(%s):%s, [%s])" % (
            func_or_method,
            name,
            params_str,
            return_type_str,
            ", ".join(repr(stmt) for stmt in self.body),
        )

class ReturnNode(ASTNode):
    def __init__(self, expr=None):
        ASTNode.__init__(self, AST_NODE_RETURN)
        self.expr = expr  # Can be None for return with no value
        self.expr_type = TYPE_VOID if expr is None else (expr.expr_type if hasattr(expr, 'expr_type') else TYPE_UNKNOWN)

    def __repr__(self):
        if self.expr:
            return "Return(%s) -> %s" %(repr(self.expr), registry.var_type_to_string(self.expr_type))
        else:
            return "Return()"

class CompareNode(ASTNode):
    def __init__(self, operator, left, right):
        ASTNode.__init__(self, AST_NODE_COMPARE)
        self.operator = operator
        self.left = left
        self.right = right
        self.expr_type = TYPE_INT  # Comparisons always return int

    def __repr__(self):
        return "Compare(%s, %s, %s)" % (self.operator, repr(self.left), repr(self.right))

class LogicalNode(ASTNode):
    def __init__(self, operator, left, right):
        ASTNode.__init__(self, AST_NODE_LOGICAL)
        self.operator = operator
        self.left = left
        self.right = right
        self.expr_type = TYPE_INT  # Logical ops always return int

    def __repr__(self):
        return "Logical(%s, %s, %s)" % (self.operator, repr(self.left), repr(self.right))

class BitOpNode(ASTNode):
    def __init__(self, operator, left, right):
        ASTNode.__init__(self, AST_NODE_BITOP)
        self.operator = operator
        self.left = left
        self.right = right
        self.expr_type = TYPE_INT  # Bitwise ops always return int

    def __repr__(self):
        return "BitOp(%s, %s, %s)" % (self.operator, repr(self.left), repr(self.right))

class StructDefNode(ASTNode):
    def __init__(self, name, parent_name, fields, struct_id):
        ASTNode.__init__(self, AST_NODE_STRUCT_DEF)
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
    def __init__(self, struct_name, struct_id, args=None):
        ASTNode.__init__(self, AST_NODE_STRUCT_INIT)
        self.struct_name = struct_name
        self.struct_id = struct_id
        self.args = args or []  # Args for constructor
        self.expr_type = struct_id

    def __repr__(self):
        args_str = ", ".join(repr(arg) for arg in self.args)
        return "StructInit(%s(%s))" % (self.struct_name, args_str)

class MemberAccessNode(ASTNode):
    def __init__(self, obj, member_name, member_type, ref_kind=REF_KIND_NONE):
        ASTNode.__init__(self, AST_NODE_MEMBER_ACCESS)
        self.obj = obj  # Object expression
        self.member_name = member_name
        self.expr_type = member_type
        self.ref_kind = ref_kind

    def __repr__(self):
        return "MemberAccess(%s.%s)" % (repr(self.obj), self.member_name)

class CallNode(ASTNode):
    def __init__(self, name, args, return_type, obj=None):
        ASTNode.__init__(self, AST_NODE_CALL)
        self.name = name
        self.args = args
        self.expr_type = return_type
        self.obj = obj  # None for regular functions, object expr for methods

    def is_method_call(self):
        return self.obj is not None

    def __repr__(self):
        args_str = ", ".join(repr(arg) for arg in self.args)
        if self.obj:
            return "Call(%s.%s(%s))" % (repr(self.obj), self.name, args_str)
        else:
            return "Call(%s(%s))" % (self.name, args_str)

class NewNode(ASTNode):
    def __init__(self, struct_init):
        ASTNode.__init__(self, AST_NODE_NEW)
        self.struct_init = struct_init
        self.expr_type = struct_init.expr_type
        self.ref_kind = REF_KIND_HEAP

    def __repr__(self):
        return "New(%s) -> heap_ref<%s>" % (repr(self.struct_init), registry.var_type_to_string(self.expr_type))

class DelNode(ASTNode):
    def __init__(self, expr):
        ASTNode.__init__(self, AST_NODE_DEL)
        self.expr = expr
        self.expr_type = TYPE_VOID

    def __repr__(self):
        return "Del(%s)" % repr(self.expr)

class GenericInitializerNode(ASTNode):
    def __init__(self, elements, subtype, target_type=TYPE_UNKNOWN):
        ASTNode.__init__(self, AST_NODE_GENERIC_INITIALIZER)
        self.elements = elements    # List of expressions or nested initializers
        self.subtype = subtype
        self.target_type = target_type  # The expected type (struct/array)
        self.expr_type = target_type    # Result type of this initializer

    def __repr__(self):
        subtype_str = ["TUPLE", "LINEAR", "NAMED"][self.subtype]
        elements_str = ", ".join(repr(e) for e in self.elements)
        return "Initializer(%s, %s, [%s])" % (
            subtype_str, registry.var_type_to_string(self.target_type), elements_str)

def is_literal_node(node):
    """Check if a node represents a literal value (for global var init)"""
    return node.node_type in [AST_NODE_NUMBER, AST_NODE_STRING]

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.token = self.lexer.next_token()
        self.prev_token = None

        self.scopes = ["global"]  # Stack of scope names
        self.variables = {"global": set()}  # Track declared variables per scope
        self.constants = {"global": set()}  # Track constants (let declarations) per scope
        self.var_types = {"global": {}}     # Track variable types per scope
        self.var_ref_kinds = {"global": {}} # Track variable reference kinds per scope

        # Track if we've seen functions - used to enforce globals-before-functions rule
        self.seen_main_function = False
        self.current_function = -1  # Track current function for return checking

        # Track current struct for method definitions
        self.current_struct = None

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

    def enter_scope(self, scope_name):
        """Enter a new scope for variable tracking"""
        self.scopes.append(scope_name)
        self.variables[scope_name] = set()
        self.constants[scope_name] = set()
        self.var_types[scope_name] = {}
        self.var_ref_kinds[scope_name] = {}

    def leave_scope(self):
        """Leave the current scope"""
        if len(self.scopes) > 1:  # Don't leave global scope
            self.scopes.pop()

    def current_scope(self):
        """Get the current scope name"""
        return self.scopes[-1]

    def is_variable_declared(self, var_name):
        """Check if a variable is declared in any accessible scope"""
        # Check all scopes from current to global
        for scope in reversed(self.scopes):
            if var_name in self.variables[scope]:
                return True
        return False

    def is_constant(self, var_name):
        """Check if a variable is a constant in any accessible scope"""
        # Check all scopes from current to global
        for scope in reversed(self.scopes):
            if var_name in self.constants[scope]:
                return True
        return False

    def get_variable_type(self, var_name):
        """Get a variable's type from the appropriate scope"""
        # Check all scopes from current to global
        for scope in reversed(self.scopes):
            if var_name in self.var_types[scope]:
                return self.var_types[scope][var_name]
        return TYPE_UNKNOWN

    def declare_variable(self, var_name, var_type, is_const=False, ref_kind=REF_KIND_NONE):
        """Declare a variable in the current scope"""
        current = self.current_scope()
        # Check if already declared in current scope
        if var_name in self.variables[current]:
            return False  # Already declared in this scope

        self.variables[current].add(var_name)
        self.var_types[current][var_name] = var_type
        self.var_ref_kinds[current][var_name] = ref_kind

        if is_const:
            self.constants[current].add(var_name)
        return True

    def get_variable_ref_kind(self, var_name):
        """Get a variable's reference kind from the appropriate scope"""
        # Check all scopes from current to global
        for scope in reversed(self.scopes):
            if var_name in self.var_ref_kinds[scope]:
                return self.var_ref_kinds[scope][var_name]
        return REF_KIND_NONE

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
        left = self.nud(t)
        while rbp < self.lbp(self.token):
            t = self.token
            self.advance()
            left = self.led(t, left)
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

        # For references, any ref_kind can be assigned to any other ref_kind
        # (they're compatible from a typing perspective)
        # Memory management needs to check ref_kind explicitly for del/cleanup operations

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
                # For byref parameters, make sure the argument is addressable
                if arg.node_type != AST_NODE_VARIABLE:
                    # Only variables can be passed by reference
                    self.error("Cannot pass expression result to 'byref' parameter '%s' - must be a variable" % param_name)

                # The referenced variable's type must be compatible with the parameter type
                if not can_promote(arg.expr_type, param_type):
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
        expected = len(params) if not is_method else len(params)-1
        if len(args) != expected:
            self.error("%s expects %d arguments, got %d" %
                      (func_name, expected, len(args)))

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
            self.consume(TT_RPAREN)

            # Type check arguments
            self.check_arg_count("Method '%s'" % member_name, func_obj.params, args, is_method=True)
            self.check_argument_types(args, func_obj.params, "method '%s'" % member_name)

            return CallNode(member_name, args, func_obj.return_type, obj_node)
        else:
            # Field access
            field_type = registry.get_field_type(struct_name, member_name)
            if field_type is None:
                self.error("Field '%s' not found in struct '%s'" % (member_name, struct_name))

            return MemberAccessNode(obj_node, member_name, field_type, obj_ref_kind)

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
        if self.token.type == TT_COLON:
            self.advance()
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
        self.enter_scope(name)
        prev_function = self.current_function
        self.current_function = registry.register_function(name, return_type, params, parent_struct_id=struct_id)
        # create a new node with empty body - we'll add it later
        # That's needed so type checking inside the body can find it
        node = FunctionDeclNode(name, params, return_type, body=None, parent_struct_id=struct_id)
        registry.set_function_ast_node(self.current_function, node)

        if struct_name:
            # Insert 'self' as first parameter (implicitly byref)
            params.insert(0, ("self", struct_id, True))

        # Add parameters to scope
        for param_name, param_type, _ in params:
            self.declare_variable(param_name, param_type)

        # Parse function/method body
        body = self.doblock()

        # Restore previous context
        self.leave_scope()
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
            return type_id
        elif self.token.type == TT_IDENT:
            type_name = self.token.value
            if registry.struct_exists(type_name):
                type_id = registry.get_struct_id(type_name)
                self.advance()
                return type_id
            else:
                self.error("Unknown type '%s'" % type_name)
        else:
            self.error("Expected a type")

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

        return StructDefNode(struct_name, parent_name, fields, struct_id)

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

    def create_zero_value_node(self, field_type):
        """Create an AST node with the appropriate zero/default value for a given type"""
        if field_type == TYPE_STRING:
            return StringNode("")
        elif is_float_type(field_type):
            return NumberNode(0.0, field_type)
        elif registry.is_struct_type(field_type):
            # For struct fields, create a zero-filled initializer recursively
            struct_name = registry.get_struct_name(field_type)
            fields = registry.get_all_fields(struct_name)
            zero_elements = []

            for _, nested_field_type in fields:
                zero_elements.append(self.create_zero_value_node(nested_field_type))

            # Create initializer with zero elements
            init_node = GenericInitializerNode(zero_elements, INITIALIZER_SUBTYPE_LINEAR, field_type)
            init_node.expr_type = field_type
            return init_node
        else:
            # For all other types (int, etc.), use 0
            return NumberNode(0, field_type)

    def parse_initializer_expression(self, target_type=TYPE_UNKNOWN):
        """Parse an initializer expression: {expr1, expr2, ...} or {{...}, {...}}"""
        self.consume(TT_LBRACE)  # Skip '{'

        elements = []
        subtype = INITIALIZER_SUBTYPE_TUPLE  # Default

        # Determine subtype based on target_type
        if target_type != TYPE_UNKNOWN:
            if registry.is_struct_type(target_type):
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
                    if registry.is_struct_type(target_type):
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

            # Check for end of initializer or comma
            if self.token.type != TT_COMMA:
                break

            self.advance()  # Skip comma

        self.consume(TT_RBRACE)

        # Create initializer node
        init_node = GenericInitializerNode(elements, subtype, target_type)

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
                    elements.append(self.create_zero_value_node(field_type))

            # Validate field types
            for i, elem in enumerate(elements):
                _, field_type = fields[i]
                if not can_promote(elem.expr_type, field_type):
                    self.type_mismatch_error("Field %d in %s initializer" % (i+1, struct_name), 
                                            elem.expr_type, field_type)

        return init_node

    def nud(self, t):
        # Handle number literals using the type mapping
        if t.type in TOKEN_TO_TYPE_MAP:
            return NumberNode(t.value, TOKEN_TO_TYPE_MAP[t.type])

        if t.type == TT_STRING_LITERAL:
            return StringNode(t.value)

        if t.type == TT_IDENT:
            var_name = t.value

            # Check if it's a struct type name (for initialization)
            if registry.struct_exists(var_name):
                struct_id = registry.get_struct_id(var_name)

                # Parse initializer: StructName() or StructName(arg1, arg2, ...)
                if self.token.type == TT_LPAREN:
                    self.advance()  # Skip '('

                    # Parse constructor args
                    args = []
                    self.parse_args(args)
                    self.consume(TT_RPAREN)

                    # Create struct initialization
                    return StructInitNode(var_name, struct_id, args)

            # For a variable in an expression context:
            # Could be a function name.
            # FIXME: we actually need to check whether the var_name token is
            # a struct name or "self" followed by TT_DOT and used inside a
            # method, then derive
            # the full struct_name + function_name "tuple" for full function
            # lookup in type_registry
            func_id = registry.lookup_function(var_name)
            if func_id != -1:
                return_type = registry.get_func_from_id(func_id).return_type
                return VariableNode(var_name, return_type)

            # It's a variable name, see if it's declared
            if not self.is_variable_declared(var_name):
                self.error("Variable '%s' is not declared" % var_name)

            # Get the variable type from the appropriate scope
            var_type = self.get_variable_type(var_name)
            # Get the variable's reference kind
            ref_kind = self.get_variable_ref_kind(var_name)

            return VariableNode(var_name, var_type, ref_kind)

        if t.type in [TT_MINUS, TT_NOT, TT_BITNOT]:  # Unary operators
            expr = self.expression(UNARY_PRECEDENCE)
            return UnaryOpNode(t.value, expr, expr.expr_type)

        if t.type == TT_LBRACE:
            return self.parse_initializer_expression()

        if t.type == TT_LPAREN:
            expr = self.expression(0)
            self.consume(TT_RPAREN)
            return expr

        if t.type == TT_NEW:
            # Parse the struct name after 'new'
            if self.token.type != TT_IDENT:
                self.error("Expected struct name after 'new'")

            struct_name = self.token.value
            self.advance()

            # Verify struct exists
            if not registry.struct_exists(struct_name):
                self.error("Struct '%s' is not defined" % struct_name)

            # Get the struct type ID
            struct_id = registry.get_struct_id(struct_name)

            # Check for constructor call
            args = []

            if self.token.type != TT_LPAREN:
                self.error("constructor invocation requires parenthesis")

            # Parse constructor arguments
            self.advance()  # Skip '('
            self.parse_args(args)
            self.consume(TT_RPAREN)

            # Check if init method exists for the struct
            init_method = registry.get_method(struct_id, "init")
            if init_method:
                # Check argument count
                self.check_arg_count("Constructor for '%s'" % struct_name, init_method.params, args, is_method=True)
                self.check_argument_types(args, init_method.params, "constructor for '%s'" % struct_name)

            # Create heap allocated struct
            struct_init = StructInitNode(struct_name, struct_id, args)
            return NewNode(struct_init)

        raise CompilerException('Unexpected token %s' % token_name(t), t)

    def led(self, t, left):
        # Handle dot operator for member access
        if t.type == TT_DOT:
            return self.parse_member_access(left)

        # Handle function call
        if t.type == TT_LPAREN and left.node_type == AST_NODE_VARIABLE:
            return self.funccall(left.name, consume_lparen=False)

        # Handle assignment as an operator
        if t.type == TT_ASSIGN and left.node_type == AST_NODE_VARIABLE:
            # Get variable name from left side
            var_name = left.name
            var_type = self.get_variable_type(var_name)
            var_ref_kind = self.get_variable_ref_kind(var_name)

            # Check if variable is a constant (declared with 'let')
            if self.is_constant(var_name):
                self.error("Cannot reassign to constant '%s'" % var_name)

            # Parse the right side expression
            right = self.expression(0)

            # For assignments in conditions (e.g. while x = y do),
            # use the fully resolved types
            if right.node_type == AST_NODE_VARIABLE:
                right_var = right.name
                right_type = self.get_variable_type(right_var)

            # Check type compatibility
            self.check_type_compatibility(var_name, right)

            return AssignNode(var_name, right, var_type, var_ref_kind)

        # Handle member assignment (obj.field = value)
        elif t.type == TT_ASSIGN and left.node_type == AST_NODE_MEMBER_ACCESS:
            # Parse the right side expression
            right = self.expression(0)

            # Check type compatibility
            self.check_field_compatibility(right.expr_type, left.expr_type)

            # Create a special binary operation that models the assignment
            return BinaryOpNode('=', left, right, left.expr_type)

        if t.type in [TT_PLUS, TT_MINUS, TT_MULT, TT_DIV, TT_MOD, TT_SHL, TT_SHR]:
            right = self.expression(self.lbp(t))

            # Determine the result type based on operand types
            result_type = self.calculate_result_type(t.value, left.expr_type, right.expr_type)
            if result_type is None:
                self.type_mismatch_error("Type mismatch in binary operation", left.expr_type, right.expr_type)

            return BinaryOpNode(t.value, left, right, result_type)

        elif t.type in [TT_EQ, TT_NE, TT_GE, TT_LE, TT_LT, TT_GT]:
            right = self.expression(self.lbp(t))
            # Comparisons always return an integer (0/1 representing false/true)
            # But operands must be compatible types
            result_type = self.calculate_result_type(t.value, left.expr_type, right.expr_type)
            if result_type is None:
                self.type_mismatch_error("Type mismatch in comparison", left.expr_type, right.expr_type)
            return CompareNode(t.value, left, right)

        elif t.type in [TT_AND, TT_OR]:
            right = self.expression(self.lbp(t))
            # Logical operations always return an integer (0/1 representing false/true)
            return LogicalNode(t.value, left, right)

        elif t.type in [TT_XOR, TT_BITOR, TT_BITAND]:
            right = self.expression(self.lbp(t))
            # Both operands must be integer types
            if is_integer_type(left.expr_type) and is_integer_type(right.expr_type):
                result_type = self.calculate_result_type(t.value, left.expr_type, right.expr_type)
                if result_type is not None:
                    return BitOpNode(t.value, left, right)
            self.error("Bitwise operators require integer operands")

        raise CompilerException('Unexpected token type %d' % t.type, t)

    def parse_type(self):
        """Parse a type annotation or return None if not present"""
        if self.token.type == TT_COLON:
            self.advance()  # Consume the colon

            return self.parse_type_reference()
        return TYPE_UNKNOWN

    def skip_separators(self):
        while self.token.type == TT_SEMI or self.token.type == TT_NEWLINE:
            self.advance() # Skip empty lines or semicolon (as statement separator)

    def doblock(self):
        self.skip_separators()
        self.consume(TT_DO)
        body = []
        while self.token.type != TT_END:
            if self.token.type == TT_EOF:
                self.error("Unexpected end of file while parsing a block (missing 'end')")
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
                return IfNode(condition, then_body, None)

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
            return IfNode(condition, then_body, else_body)

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
        return CallNode(func_name, args, func_return_type)

    def statement(self):
        self.skip_separators()

        # Handle struct definitions
        if self.token.type == TT_STRUCT:
            # Only allowed in global scope
            if self.current_function != -1:
                self.error("Struct definitions are not allowed inside functions")
            return self.struct_definition()

        # Handle method definitions
        if self.token.type == TT_DEF:
            return self.function_declaration()

        # Handle return statements
        if self.token.type == TT_RETURN:
            # Must be inside a function
            if self.current_function == -1:
                self.error("'return' statement outside function")

            self.advance()

            # Return with no value
            if self.token.type in [TT_SEMI, TT_NEWLINE, TT_EOF] or (self.prev_token and self.token.line > self.prev_token.line):
                self.check_statement_end()
                return ReturnNode(None)

            # Return with value
            # Special handling for initializers in return statements
            if self.token.type == TT_LBRACE:
                # Get return type from current function
                func_return_type = registry.get_func_from_id(self.current_function).return_type
                expr = self.parse_initializer_expression(func_return_type)
            else:
                expr = self.expression(0)

            # Check if return type matches function return type
            current_func_obj = registry.get_func_from_id(self.current_function)
            func_return_type = current_func_obj.return_type

            if func_return_type == TYPE_VOID:
                fn = registry.get_func_from_id(self.current_function).name
                self.error("Void function '%s' cannot return a value" % fn)

            # Check that return expression type matches function return type
            if not can_promote(expr.expr_type, func_return_type):
                self.type_mismatch_error("Type mismatch in return", expr.expr_type, func_return_type)

            self.check_statement_end()
            return ReturnNode(expr)

        # Handle del statement for heap deallocation
        if self.token.type == TT_DEL:
            self.advance()
            expr = self.expression(0)

            # Verify expr is a heap reference
            if expr.ref_kind != REF_KIND_HEAP:
                self.error("'del' can only be used with reference types (created with 'new')")

            self.check_statement_end()
            return DelNode(expr)

        # Handle variable declarations (var and let)
        if self.token.type in [TT_VAR, TT_CONST]:
            decl_type = self.token.type  # Save the declaration type (var or let)
            self.advance()

            # Expect an identifier after var/let
            if self.token.type != TT_IDENT:
                self.error("Expected identifier after '%s'" % ('var' if decl_type == TT_VAR else 'let'))

            var_name = self.token.value
            self.advance()

            # Process type annotation if present
            var_type = self.parse_type()  # This will consume the type if present

            # Check for assignment operator
            if self.token.type == TT_TYPE_ASSIGN:
                # Type inference assignment (:=)
                self.advance()  # Skip the := operator

                # Parse the initializer expression
                if self.token.type == TT_LBRACE:
                    expr = self.parse_initializer_expression()
                else:
                    expr = self.expression(0)

                # In global scope, ensure only literal initializers
                if self.current_function == -1 and self.seen_main_function:
                    self.error("Global variables must be declared before main function")

                if self.current_function == -1 and not is_literal_node(expr):
                    self.error("Global variables can only be initialized with literals")

                # Infer the type from expression
                if expr.node_type == AST_NODE_NUMBER:
                    var_type = expr.expr_type
                elif expr.node_type == AST_NODE_VARIABLE:
                    # Get type from referenced variable
                    ref_var = expr.name
                    var_type = self.get_variable_type(ref_var)
                    if var_type == TYPE_UNKNOWN:
                        self.error("Cannot infer type from variable '%s' with unknown type" % ref_var)
                elif expr.node_type == AST_NODE_STRUCT_INIT:
                    # Set var_type to struct type
                    var_type = expr.struct_id
                elif expr.node_type == AST_NODE_NEW:
                    # Set var_type to reference type
                    var_type = expr.expr_type  # Already a reference type
                elif expr.node_type == AST_NODE_GENERIC_INITIALIZER:
                    # For initializers, use the inferred type
                    var_type = expr.expr_type
                elif hasattr(expr, 'expr_type'):
                    var_type = expr.expr_type
                else:
                    # Default to int for other cases
                    var_type = TYPE_INT

            elif self.token.type == TT_ASSIGN:
                # Regular assignment with explicit type (=)
                if var_type == TYPE_UNKNOWN:
                    self.error("Variable declaration with '=' requires explicit type annotation")

                self.advance()  # Skip the = sign

                # Check for initializer syntax
                if self.token.type == TT_LBRACE:
                    expr = self.parse_initializer_expression(var_type)
                else:
                    # Parse the initializer expression
                    expr = self.expression(0)

                # In global scope, ensure only literal initializers
                if self.current_function == -1 and self.seen_main_function:
                    self.error("Global variables must be declared before main function")

                if self.current_function == -1 and not is_literal_node(expr):
                    self.error("Global variables can only be initialized with literals")

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
            else:
                self.error("Variable declaration must include an initialization")
            # Register the variable as defined in the current scope
            if var_name in self.variables[self.current_scope()]:
                self.already_declared_error(var_name)

            # If initializing with a reference (from new), keep the reference kind
            ref_kind = expr.ref_kind

            # Declare the variable in current scope
            self.declare_variable(var_name, var_type, decl_type == TT_CONST, ref_kind=ref_kind)
            self.check_statement_end()
            return VarDeclNode(decl_type, var_name, var_type, expr, ref_kind=ref_kind)

        if self.token.type == TT_IF:
            return self.if_statement()

        # Handle while loop
        elif self.token.type == TT_WHILE:
            self.advance()
            condition = self.expression(0)
            body = self.doblock()
            return WhileNode(condition, body)
        elif self.token.type == TT_PRINT:
            self.advance()
            expr = self.expression(0)
            self.check_statement_end()
            return PrintNode(expr)
        elif self.token.type == TT_BREAK:
            self.advance()
            self.check_statement_end()
            return BreakNode()
        elif self.token.type == TT_CONTINUE:
            self.advance()
            self.check_statement_end()
            return ContinueNode()
        elif self.token.type == TT_IDENT:
            var = self.token.value
            self.advance()

            # Function call
            if self.token.type == TT_LPAREN:
                node = self.funccall(var)
                self.check_statement_end()
                return node

            # Member access (variable.field)
            elif self.token.type == TT_DOT:
                # Create variable node for the object
                if not self.is_variable_declared(var):
                    self.error("Variable '%s' is not declared" % var)

                var_type = self.get_variable_type(var)
                obj_node = VariableNode(var, var_type)

                # Parse member access
                self.advance() # skip dot
                member_node = self.parse_member_access(obj_node)

                # Handle possible assignment for field access
                if member_node.node_type == AST_NODE_MEMBER_ACCESS and self.token.type in [
                    TT_ASSIGN, TT_PLUS_ASSIGN, TT_MINUS_ASSIGN,
                    TT_MULT_ASSIGN, TT_DIV_ASSIGN, TT_MOD_ASSIGN
                ]:
                    # Save the operator type
                    op = self.token.type
                    field_type = member_node.expr_type

                    # Advance past the operator
                    self.advance()

                    # Parse the right-hand expression
                    expr = self.expression(0)

                    # Check type compatibility
                    self.check_field_compatibility(expr.expr_type, field_type)

                    # For compound operators, create a BinaryOpNode with the appropriate operation
                    if op != TT_ASSIGN:
                        # Create a binary op that combines the field access and the operation
                        node = BinaryOpNode('=', member_node,
                                          BinaryOpNode(get_operator_for_compound_assign(op), member_node, expr, field_type), field_type)
                    else:
                        # Regular assignment
                        node = BinaryOpNode('=', member_node, expr, member_node.expr_type)

                    self.check_statement_end()
                    return node

                # Handle method chaining - if we have a method call result followed by a dot
                if member_node.node_type == AST_NODE_CALL and self.token.type == TT_DOT:
                    # Continue parsing the chain
                    expr = member_node
                    while self.token.type == TT_DOT:
                        self.advance()  # Skip dot
                        # Parse next member in the chain
                        expr = self.parse_member_access(expr)

                    # Now we're at the end of the chain
                    self.check_statement_end()
                    return ExprStmtNode(expr)

                # Method call or field access as expression statement
                self.check_statement_end()
                return ExprStmtNode(member_node)

            # Variable reference
            else:
                # Check if variable has been declared
                if not self.is_variable_declared(var):
                    self.error("Variable '%s' is not declared" % var)

                # Explicitly check for type-inference assignment on already declared variable
                if self.token.type == TT_TYPE_ASSIGN:
                    self.error("Cannot use ':=' with already declared variable '%s'. Use '=' instead" % var)

                # Handle all assignment operators (regular and compound)
                if self.token.type in [TT_ASSIGN, TT_PLUS_ASSIGN, TT_MINUS_ASSIGN, 
                                       TT_MULT_ASSIGN, TT_DIV_ASSIGN, TT_MOD_ASSIGN]:
                    # Check if variable is a constant (declared with 'const')
                    if self.is_constant(var):
                        self.error("Cannot reassign to constant '%s'" % var)

                    op = self.token.type
                    var_type = self.get_variable_type(var)

                    # Advance past the operator
                    self.advance()

                    # Parse the expression
                    expr = self.expression(0)

                    # Check type compatibility for assignments
                    self.check_type_compatibility(var, expr)

                    # For compound operators, use CompoundAssignNode
                    if op != TT_ASSIGN:
                        var_ref_kind = self.get_variable_ref_kind(var)
                        self.check_statement_end()
                        return CompoundAssignNode(op, var, expr, var_type, var_ref_kind)

                    # Regular assignment
                    self.check_statement_end()
                    return AssignNode(var, expr, var_type)

                # Handle expression statements (e.g., an identifier by itself)
                # Get variable type for the identifier
                var_type = self.get_variable_type(var)

                # Create the variable node
                var_node = VariableNode(var, var_type)

                # Check if this is the start of a binary expression (identifier followed by operator)
                if self.token.type in [TT_PLUS, TT_MINUS, TT_MULT, TT_DIV, TT_MOD,
                                     TT_EQ, TT_NE, TT_GT, TT_LT, TT_GE, TT_LE]:
                    # This is a binary operation starting with a variable
                    op = self.token.value
                    self.advance()  # Consume the operator token

                    # Parse the right-hand side of the binary operation
                    right = self.expression(0)

                    # Create the binary operation node
                    expr = BinaryOpNode(op, var_node, right, var_type)
                else:
                    # Just a variable reference on its own
                    expr = var_node

                self.check_statement_end()
                return ExprStmtNode(expr)
        elif self.token.type in [TT_INT_LITERAL, TT_UINT_LITERAL, TT_LONG_LITERAL, TT_ULONG_LITERAL, 
                                TT_FLOAT_LITERAL, TT_STRING_LITERAL, TT_LPAREN, TT_MINUS, TT_NOT, TT_BITNOT]:
            # Also handle expressions that start with other tokens
            expr = self.expression(0)
            self.check_statement_end()
            return ExprStmtNode(expr)

        # If we're in global scope and not at a var/let/function declaration, error
        if self.current_function == -1:
            self.error("Only variable declarations and function declarations are allowed in global scope")

        token_type_name = token_name(self.token.type)
        self.error('Invalid statement starting with "%s" (%s)' %
                  (self.token.value, token_type_name))

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
        statements = []
        while self.token.type != TT_EOF:
            statements.append(self.statement())
        return statements


### END OF COMPILER.PY
