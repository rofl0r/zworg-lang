# Implementation of a Pratt parser in Python 2.7

from shared import *
from lexer import Token, Lexer
import type_registry

# Base class for all AST nodes
class ASTNode(object):
    def __init__(self, node_type=AST_NODE_BASE):
        self.node_type = node_type
        self.expr_type = TYPE_UNKNOWN

    def __repr__(self):
        return "%s" % ast_node_type_to_string(self.node_type)

class NumberNode(ASTNode):
    def __init__(self, value, expr_type):
        ASTNode.__init__(self, AST_NODE_NUMBER)
        self.value = value
        self.expr_type = expr_type  # TYPE_INT, TYPE_FLOAT, etc.

    def __repr__(self):
        return "Number(%s, %s)" % (self.value, var_type_to_string(self.expr_type))

class StringNode(ASTNode):
    def __init__(self, value):
        ASTNode.__init__(self, AST_NODE_STRING)
        self.value = value
        self.expr_type = TYPE_STRING

    def __repr__(self):
        return "String(\"%s\")" % self.value

class VariableNode(ASTNode):
    def __init__(self, name, var_type):
        ASTNode.__init__(self, AST_NODE_VARIABLE)
        self.name = name
        self.expr_type = var_type

    def __repr__(self):
        return "Var(%s, %s)" % (self.name, var_type_to_string(self.expr_type))

class BinaryOpNode(ASTNode):
    def __init__(self, operator, left, right, result_type):
        ASTNode.__init__(self, AST_NODE_BINARY_OP)
        self.operator = operator
        self.left = left
        self.right = right
        self.expr_type = result_type

    def __repr__(self):
        return "BinaryOp(%s, %s, %s) -> %s" % (
            self.operator, repr(self.left), repr(self.right),
            var_type_to_string(self.expr_type)
        )

class UnaryOpNode(ASTNode):
    def __init__(self, operator, operand, result_type):
        ASTNode.__init__(self, AST_NODE_UNARY_OP)
        self.operator = operator
        self.operand = operand
        self.expr_type = result_type

    def __repr__(self):
        return "UnaryOp(%s, %s) -> %s" % (
            self.operator, repr(self.operand), var_type_to_string(self.expr_type)
        )

class AssignNode(ASTNode):
    def __init__(self, var_name, expr, var_type):
        ASTNode.__init__(self, AST_NODE_ASSIGN)
        self.var_name = var_name
        self.expr = expr
        self.expr_type = var_type

    def __repr__(self):
        return "Assign(%s, %s) -> %s" % (
            self.var_name, repr(self.expr), var_type_to_string(self.expr_type)
        )

class CompoundAssignNode(ASTNode):
    def __init__(self, op_type, var_name, expr, var_type):
        ASTNode.__init__(self, AST_NODE_COMPOUND_ASSIGN)
        self.op_type = op_type
        self.var_name = var_name
        self.expr = expr
        self.expr_type = var_type

    def __repr__(self):
        op_name = token_name(self.op_type)
        return "CompoundAssign(%s, %s, %s) -> %s" % (
            op_name, self.var_name, repr(self.expr), var_type_to_string(self.expr_type)
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
    def __init__(self, decl_type, var_name, var_type, expr):
        ASTNode.__init__(self, AST_NODE_VAR_DECL)
        self.decl_type = decl_type
        self.var_name = var_name
        self.var_type = var_type
        self.expr = expr

    def __repr__(self):
        decl_type_str = "var" if self.decl_type == TT_VAR else "let"
        return "VarDecl(%s, %s, %s, %s)" % (
            decl_type_str, self.var_name, var_type_to_string(self.var_type), repr(self.expr)
        )

class FunctionDeclNode(ASTNode):
    def __init__(self, name, params, return_type, body):
        ASTNode.__init__(self, AST_NODE_FUNCTION_DECL)
        self.name = name
        self.params = params  # List of (name, type) tuples
        self.return_type = return_type
        self.body = body
        self.expr_type = return_type

    def __repr__(self):
        params_str = ", ".join(["%s:%s" % (name, var_type_to_string(ptype)) for name, ptype in self.params])
        return "Function(%s(%s):%s, [%s])" % (
            self.name, 
            params_str, 
            var_type_to_string(self.return_type), 
            ", ".join(repr(stmt) for stmt in self.body),
        )

class ReturnNode(ASTNode):
    def __init__(self, expr=None):
        ASTNode.__init__(self, AST_NODE_RETURN)
        self.expr = expr  # Can be None for return with no value
        self.expr_type = TYPE_VOID if expr is None else (expr.expr_type if hasattr(expr, 'expr_type') else TYPE_UNKNOWN)

    def __repr__(self):
        if self.expr:
            return "Return(%s)" % repr(self.expr)
        else:
            return "Return()"

class FunctionCallNode(ASTNode):
    def __init__(self, name, args, expr_type = TYPE_UNKNOWN):
        ASTNode.__init__(self, AST_NODE_FUNCTION_CALL)
        self.name = name
        self.args = args
        self.expr_type = expr_type  # Will be set during type checking

    def __repr__(self):
        args_str = ", ".join(repr(arg) for arg in self.args)
        return "Call(%s(%s))" % (self.name, args_str)

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
        fields_str = ", ".join(["%s:%s" % (name, var_type_to_string(type_)) for name, type_ in self.fields])
        return "StructDef(%s%s, [%s])" % (self.name, parent_str, fields_str)

class MethodDefNode(ASTNode):
    def __init__(self, struct_name, method_name, params, return_type, body):
        ASTNode.__init__(self, AST_NODE_METHOD_DEF)
        self.struct_name = struct_name
        self.method_name = method_name
        self.params = params  # [(name, type), ...]
        self.return_type = return_type
        self.body = body
        self.expr_type = TYPE_VOID

    def __repr__(self):
        params_str = ", ".join(["%s:%s" % (name, var_type_to_string(type_)) for name, type_ in self.params])
        body_str = ", ".join(repr(stmt) for stmt in self.body)
        return "MethodDef(%s.%s(%s):%s, [%s])" % (
            self.struct_name, self.method_name, params_str, 
            var_type_to_string(self.return_type), body_str
        )

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
    def __init__(self, obj, member_name, member_type):
        ASTNode.__init__(self, AST_NODE_MEMBER_ACCESS)
        self.obj = obj  # Object expression
        self.member_name = member_name
        self.expr_type = member_type

    def __repr__(self):
        return "MemberAccess(%s.%s)" % (repr(self.obj), self.member_name)

class MethodCallNode(ASTNode):
    def __init__(self, obj, method_name, args, return_type):
        ASTNode.__init__(self, AST_NODE_METHOD_CALL)
        self.obj = obj  # Object expression
        self.method_name = method_name
        self.args = args
        self.expr_type = return_type

    def __repr__(self):
        args_str = ", ".join(repr(arg) for arg in self.args)
        return "MethodCall(%s.%s(%s))" % (repr(self.obj), self.method_name, args_str)

class NewNode(ASTNode):
    def __init__(self, struct_init):
        ASTNode.__init__(self, AST_NODE_NEW)
        self.struct_init = struct_init
        self.expr_type = make_ref_type(struct_init.expr_type)

    def __repr__(self):
        return "New(%s)" % repr(self.struct_init)

class DelNode(ASTNode):
    def __init__(self, expr):
        ASTNode.__init__(self, AST_NODE_DEL)
        self.expr = expr
        self.expr_type = TYPE_VOID

    def __repr__(self):
        return "Del(%s)" % repr(self.expr)

class TupleNode(ASTNode):
    def __init__(self, elements):
        ASTNode.__init__(self, AST_NODE_TUPLE)
        self.elements = elements  # List of expression nodes
        self.element_types = [elem.expr_type for elem in elements]
        # Will be set later by the parser
        self.expr_type = TYPE_UNKNOWN

    def __repr__(self):
        elements_str = ", ".join(repr(e) for e in self.elements)
        return "Tuple(%s)" % elements_str

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

        # Track if we've seen functions - used to enforce globals-before-functions rule
        self.seen_main_function = False

        self.functions = {}     # Track function declarations (name -> (params, return_type))
        self.current_function = None  # Track current function for return checking

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
        if is_struct_type(type1) or is_struct_type(type2):
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
            if is_float_type(left_type) or is_float_type(right_type) or is_struct_type(left_type) or is_struct_type(right_type):
                self.error("Bitwise operators require integer operands")

        # Arithmetic operators
        return self.get_common_type(left_type, right_type)

    def enter_scope(self, scope_name):
        """Enter a new scope for variable tracking"""
        self.scopes.append(scope_name)
        self.variables[scope_name] = set()
        self.constants[scope_name] = set()
        self.var_types[scope_name] = {}

    def leave_scope(self):
        """Leave the current scope"""
        if len(self.scopes) > 1:  # Don't leave global scope
            self.scopes.pop()

    def current_scope(self):
        """Get the current scope name"""
        return self.scopes[-1]

    def is_function_declared(self, var_name):
        return var_name in self.functions

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

    def declare_variable(self, var_name, var_type, is_const=False):
        """Declare a variable in the current scope"""
        current = self.current_scope()
        # Check if already declared in current scope
        if var_name in self.variables[current]:
            return False  # Already declared in this scope

        self.variables[current].add(var_name)
        self.var_types[current][var_name] = var_type
        if is_const:
            self.constants[current].add(var_name)
        return True

    def error(self, message):
        """Raise a compiler exception with current token information"""
        raise CompilerException(message, self.token)

    def type_mismatch_error(self, context, from_type, to_type):
        """Generate consistent type mismatch error messages"""
        self.error("%s: cannot convert %s to %s" % 
                  (context, var_type_to_string(from_type), var_type_to_string(to_type)))

    def type_mismatch_assignment_error(self, var_name, expr_type, var_type):
        """Generate type mismatch error for assignment operations"""
        self.error("Type mismatch: can't assign a value of type %s to %s (type %s)" % 
                  (var_type_to_string(expr_type), var_name, var_type_to_string(var_type)))

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

    def token_type_to_var_type(self, token_type):
        """Convert token type to variable type"""
        # Use the token-to-type mapping or raise an error for unknown types
        if token_type not in TOKEN_TO_TYPE_MAP:
            self.error("Unknown token type for variable type conversion: %s" % str(token_type))
        return TOKEN_TO_TYPE_MAP[token_type]

    def check_type_compatibility(self, var_name, expr_type):
        """Check if the expression's type is compatible with the variable's type"""
        # Get variable type from the appropriate scope
        var_type = self.get_variable_type(var_name)

        if var_type is not None and var_type != TYPE_UNKNOWN and expr_type != TYPE_UNKNOWN and not can_promote(expr_type, var_type):
            self.type_mismatch_assignment_error(var_name, expr_type, var_type)

    def check_argument_types(self, args, params, context_name):
        """Check if argument types match parameter types"""
        for i, ((param_name, param_type), arg) in enumerate(zip(params, args)):
            if not can_promote(arg.expr_type, param_type):
                self.type_mismatch_error("Type mismatch for argument %d of %s" % (i+1, context_name), 
                                         arg.expr_type, param_type)

    def check_arg_count(self, func_name, params, args):
        """Helper to check argument count against parameter count"""
        if len(args) != len(params):
            self.error("%s expects %d arguments, got %d" %
                      (func_name, len(params), len(args)))

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
                      (var_type_to_string(from_type), var_type_to_string(to_type)))

    def parse_member_access(self, obj_node):
        """
        Parse member access for an object (obj.member)
        Handles both field access and method calls
        Returns a MemberAccessNode or MethodCallNode
        """
        # Get object type
        obj_type = obj_node.expr_type
        base_type = get_base_type(obj_type)  # Unwrap reference if needed

        if not is_struct_type(base_type):
            self.error("Left side of '.' is not a struct type")

        struct_name = type_registry.get_struct_name(base_type)

        # Parse member name
        if self.token.type != TT_IDENT:
            self.error("Expected member name after '.'")

        member_name = self.token.value
        self.advance()

        # Check if it's a method call or field access
        if self.token.type == TT_LPAREN:
            # Method call
            self.advance()  # Skip '('

            # Get method details
            method = type_registry.get_method(struct_name, member_name)
            if not method:
                self.error("Method '%s' not found in struct '%s'" % (member_name, struct_name))

            # Parse arguments
            args = []
            self.parse_args(args)
            self.consume(TT_RPAREN)

            # Type check arguments
            self.check_arg_count("Method '%s'" % member_name, method.params, args)
            self.check_argument_types(args, method.params, "method '%s'" % member_name)

            return MethodCallNode(obj_node, member_name, args, method.return_type)
        else:
            # Field access
            field_type = type_registry.get_field_type(struct_name, member_name)
            if field_type is None:
                self.error("Field '%s' not found in struct '%s'" % (member_name, struct_name))

            return MemberAccessNode(obj_node, member_name, field_type)

    def function_declaration(self):
        """Parse a function declaration or method definition"""
        self.advance()  # Skip 'def'

        # Parse function/method name
        if self.token.type != TT_IDENT:
            self.error("Expected name after 'def'")

        name = self.token.value
        is_method = False
        struct_name = None
        method_name = None

        self.advance()

        # Check if it's a method (has a dot after struct name)
        if self.token.type == TT_DOT:
            is_method = True
            struct_name = name

            # Verify struct exists
            if not type_registry.struct_exists(struct_name):
                self.error("Struct '%s' is not defined" % struct_name)

            # Parse method name after the dot
            self.advance()  # Skip the dot
            if self.token.type != TT_IDENT:
                self.error("Expected method name after '.'")

            method_name = self.token.value
            name = struct_name + "." + method_name  # Store full name for scope
            self.advance()
        elif self.current_function is not None and not is_method:
            self.error("Nested function declarations are not allowed")

        if name == "main":
            self.seen_main_function = True

        # Parse parameters
        self.consume(TT_LPAREN)
        params = []

        while self.token.type != TT_RPAREN:
            tmp = self.parameter()

            # For methods, check if parameter name is 'self'
            if is_method and tmp[0] == "self":
                self.error("Cannot use 'self' as a parameter name, it is reserved")

            for n, _ in params:
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

        # Special checks for methods
        if is_method:
            # Check constructor and destructor constraints
            if method_name == "init" and return_type != TYPE_VOID:
                self.error("Constructor 'init' must have void return type")
            elif method_name == "fini":
                if return_type != TYPE_VOID:
                    self.error("Destructor 'fini' must have void return type")
                if len(params) > 0:
                    self.error("Destructor 'fini' cannot have parameters")
        else:
            # Regular function - register it
            if name in self.functions:
                self.error("Function '%s' is already defined" % name)
            self.functions[name] = (params, return_type)

        # Enter function/method scope
        self.enter_scope(name)
        prev_function = self.current_function
        prev_struct = self.current_struct
        self.current_function = name

        if is_method:
            self.current_struct = struct_name
            # Add implicit 'self' parameter of struct type
            struct_id = type_registry.get_struct_id(struct_name)
            # Create and register the method node with empty body first
            # so type checking inside the body can find it
            temp_method = MethodDefNode(struct_name, method_name, params, return_type, [])
            type_registry.add_method(struct_name, method_name, temp_method)
            self.declare_variable("self", struct_id)

        # Add parameters to scope
        for param_name, param_type in params:
            self.declare_variable(param_name, param_type)

        # Parse function/method body
        body = self.doblock()

        # Restore previous context
        self.leave_scope()
        self.current_function = prev_function
        self.current_struct = prev_struct

        # Create and return the appropriate node
        if is_method:
            temp_method.body = body
            return temp_method
        else:
            return FunctionDeclNode(name, params, return_type, body)

    def parameter(self):
        """Parse a function parameter (name:type)"""
        if self.token.type != TT_IDENT:
            self.error("Expected parameter name")

        name = self.token.value
        self.advance()

        # Parse type - REQUIRED
        if self.token.type != TT_COLON:
            self.error("Function parameters require explicit type annotation")
        self.advance() # Skip colon
        param_type = self.parse_type_reference()

        return (name, param_type)

    def parse_type_reference(self):
        """Parse a type reference (tuple, primitive or struct)"""
        # Check for tuple types: (type1, type2, ...)
        if self.token.type == TT_LPAREN:
            return self.parse_tuple_type()

        # Regular types (primitive or struct)
        if self.token.type in TYPE_TOKEN_MAP:
            type_id = TYPE_TOKEN_MAP[self.token.type]
            self.advance()
            return type_id
        elif self.token.type == TT_IDENT:
            type_name = self.token.value
            if type_registry.struct_exists(type_name):
                type_id = type_registry.get_struct_id(type_name)
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
            if not type_registry.struct_exists(parent_name):
                self.error("Parent struct '%s' is not defined" % parent_name)

            self.advance()

        # Register the struct
        struct_id = type_registry.register_struct(struct_name, parent_name, self.token)

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
            type_registry.add_field(struct_name, field_name, field_type, self.token)
            fields.append((field_name, field_type))

            # If on the same line, require a semicolon
            if self.token.type != TT_NEWLINE and self.token.type != TT_EOF and self.token.type != TT_END:
                self.consume(TT_SEMI)

            self.skip_separators()

        self.advance()  # Skip 'end'

        return StructDefNode(struct_name, parent_name, fields, struct_id)

    def determine_result_type(self, left_type, right_type):
        """Determine the result type of a binary operation based on operand types"""
        if left_type != right_type:
            self.error("Type mismatch: cannot operate on values of different types")
        return left_type

    def is_tuple_expression(self):
        """
        Check if the current token sequence represents a tuple expression
        Uses a token-by-token approach that respects nested parentheses
        """
        if self.token.type != TT_LPAREN:
            return False

        # Save current position
        saved_token = self.token
        saved_prev_token = self.prev_token
        saved_pos = self.lexer.pos
        saved_line = self.lexer.line
        saved_col = self.lexer.column
        saved_curr_char = self.lexer.current_char

        self.advance()  # Skip '('

        # Empty tuple () is not supported
        if self.token.type == TT_RPAREN:
            # Restore state and return
            self.token = saved_token
            self.lexer.pos = saved_pos
            self.lexer.line = saved_line
            self.lexer.column = saved_col
            return False

        # Skip the first expression while respecting nested parentheses
        paren_level = 0
        result = False  # Default is not a tuple

        while self.token.type != TT_EOF:
            if self.token.type == TT_LPAREN:
                paren_level += 1
            elif self.token.type == TT_RPAREN:
                if paren_level == 0:
                    # We reached the end of the outer parentheses without finding a comma
                    break
                paren_level -= 1
            elif self.token.type == TT_COMMA and paren_level == 0:
                # Found a comma at the top level - it's a tuple!
                result = True
                break
            self.advance()

        # Restore lexer position
        self.token = saved_token
        self.prev_token = saved_prev_token
        self.lexer.pos = saved_pos
        self.lexer.line = saved_line
        self.lexer.column = saved_col
        self.lexer.current_char = saved_curr_char
        return result

    def generate_tuple_type_name(self, element_types):
        """Generate a unique name for a tuple type based on its element types"""
        elements_str = "_".join(var_type_to_string(t) for t in element_types)
        return "_tuple_%d_%s" % (len(element_types), elements_str)

    def register_tuple_type(self, element_types, is_type_annotation=False):
        """
        Register a tuple type with the given element types
        Works for both tuple expressions and tuple type annotations

        Args:
            element_types: List of type IDs for tuple fields

        Returns:
            The struct ID for the tuple type
        """
        # Generate a unique struct type name for this tuple
        if is_type_annotation:
            # Element types are already type IDs (from type annotations)
            type_list = element_types
            elements_str = "_".join(var_type_to_string(t) for t in element_types)
        else:
            # Element types are from expressions, need to extract the expr_type
            type_list = [e.expr_type for e in element_types]
            elements_str = "_".join(var_type_to_string(t) for t in type_list)

        struct_name = "_tuple_%d_%s" % (len(element_types), elements_str)

        # Register the tuple as an anonymous struct if not already registered
        if not type_registry.struct_exists(struct_name):
            struct_id = type_registry.register_struct(struct_name, None, self.token)
            for i, element_type in enumerate(type_list):
                type_registry.add_field(struct_name, "_%d" % i, element_type)
        else:
            struct_id = type_registry.get_struct_id(struct_name)

        return struct_id

    def parse_tuple_type(self):
        """Parse a tuple type annotation: (type1, type2, ...)"""
        self.advance()  # Skip '('

        element_types = []

        # Parse first element type
        element_types.append(self.parse_type_reference())

        # Parse remaining element types
        while self.token.type == TT_COMMA:
            self.advance()  # Skip comma
            element_types.append(self.parse_type_reference())

        self.consume(TT_RPAREN)

        # Register and return the tuple type ID
        return self.register_tuple_type(element_types, is_type_annotation=True)

    def parse_tuple_expression(self):
        """Parse a tuple expression: (expr1, expr2, ...)"""
        self.advance()  # Skip '('

        elements = []

        # Empty tuple not allowed
        if self.token.type == TT_RPAREN:
            self.error("Empty tuples are not supported")

        # Parse first element
        elements = [self.expression(0)]

        # Parse remaining elements
        while True:
            # Check if we've reached the end of the tuple
            if self.token.type != TT_COMMA:
                break

            self.advance()  # Skip comma

            # Parse the next element
            elements.append(self.expression(0))

        self.consume(TT_RPAREN)

        # Create tuple node
        tuple_node = TupleNode(elements)

        # Register and get the tuple type
        struct_id = self.register_tuple_type(elements, is_type_annotation=False)
        
        # Set the type of the tuple node
        tuple_node.expr_type = struct_id
        
        return tuple_node

    def nud(self, t):
        # Handle number literals using the type mapping
        if t.type in TOKEN_TO_TYPE_MAP:
            return NumberNode(t.value, TOKEN_TO_TYPE_MAP[t.type])

        if t.type == TT_STRING_LITERAL:
            return StringNode(t.value)

        if t.type == TT_IDENT:
            var_name = t.value

            # Check if it's a struct type name (for initialization)
            if type_registry.struct_exists(var_name):
                struct_id = type_registry.get_struct_id(var_name)

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
            if self.is_function_declared(var_name):
                _, return_type = self.functions[var_name]
                return VariableNode(var_name, return_type)

            # It's a variable name, see if it's declared
            if not self.is_variable_declared(var_name):
                self.error("Variable '%s' is not declared" % var_name)

            # Get the variable type from the appropriate scope
            var_type = self.get_variable_type(var_name)
            return VariableNode(var_name, var_type)

        if t.type in [TT_MINUS, TT_NOT, TT_BITNOT]:  # Unary operators
            expr = self.expression(UNARY_PRECEDENCE)
            return UnaryOpNode(t.value, expr, expr.expr_type)

        if t.type == TT_LPAREN:
            # Check if this is a tuple expression or just parenthesized expression
            if self.is_tuple_expression():
                expr = self.parse_tuple_expression()
            else:
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
            if not type_registry.struct_exists(struct_name):
                self.error("Struct '%s' is not defined" % struct_name)

            # Get the struct type ID
            struct_id = type_registry.get_struct_id(struct_name)

            # Check for constructor call
            args = []

            if self.token.type != TT_LPAREN:
                self.error("constructor invocation requires parenthesis")

            # Parse constructor arguments
            self.advance()  # Skip '('
            self.parse_args(args)
            self.consume(TT_RPAREN)

            # Check if init method exists for the struct
            init_method = type_registry.get_method(struct_name, "init")
            if init_method:
                # Check argument count
                self.check_arg_count("Constructor for '%s'" % struct_name, init_method.params, args)
                self.check_argument_types(args, init_method.params, "constructor for '%s'" % struct_name)

            # Create heap allocated struct
            struct_init = StructInitNode(struct_name, struct_id, args)
            return NewNode(struct_init)

        raise CompilerException('Unexpected token type %d' % t.type, t)

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
            self.check_type_compatibility(var_name, right.expr_type)

            return AssignNode(var_name, right, var_type)

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
        """Parse a function call and return a FunctionCallNode"""
        if not self.is_function_declared(func_name):
                self.error("'%s' is not a function" % func_name)

        if consume_lparen: self.consume(TT_LPAREN)

        # Parse arguments
        args = []
        self.parse_args(args)
        self.consume(TT_RPAREN)

        # Type checking for function call
        func_params, func_return_type = self.functions[func_name]

        # Check number of arguments
        self.check_arg_count("Function '%s'" % func_name, func_params, args)
        self.check_argument_types(args, func_params, "function '%s'" % func_name)
        return FunctionCallNode(func_name, args, func_return_type)

    def statement(self):
        self.skip_separators()

        # Handle struct definitions
        if self.token.type == TT_STRUCT:
            # Only allowed in global scope
            if self.current_function is not None:
                self.error("Struct definitions are not allowed inside functions")
            return self.struct_definition()

        # Handle method definitions
        if self.token.type == TT_DEF:
            return self.function_declaration()

        # Handle return statements
        if self.token.type == TT_RETURN:
            # Must be inside a function
            if self.current_function is None:
                self.error("'return' statement outside function")

            self.advance()

            # Return with no value
            if self.token.type in [TT_SEMI, TT_NEWLINE, TT_EOF] or (self.prev_token and self.token.line > self.prev_token.line):
                self.check_statement_end()
                return ReturnNode(None)

            # Return with value
            # Special handling for tuple expressions in return statements
            if self.token.type == TT_LPAREN and self.is_tuple_expression():
                expr = self.parse_tuple_expression()
            else:
                expr = self.expression(0)

            # Check if return type matches function return type
            func_return_type = None
            # Handle return in methods
            if "." in self.current_function:
                struct_name, method_name = self.current_function.split(".")
                method = type_registry.get_method(struct_name, method_name)
                if method:
                    func_return_type = method.return_type
            # Regular function return
            else:
                func_return_type = self.functions[self.current_function][1]

            if func_return_type == TYPE_VOID:
                self.error("Void function '%s' cannot return a value" % self.current_function)

            # If we're returning a tuple, make sure it matches the function's return type
            if expr.node_type == AST_NODE_TUPLE:
                if not is_struct_type(func_return_type):
                    self.type_mismatch_error("Type mismatch in return", expr.expr_type, func_return_type)
            else:
                # Check that return expression type matches function return type
                if not can_promote(expr.expr_type, func_return_type):
                    self.type_mismatch_error("Type mismatch in return", expr.expr_type, func_return_type)

            self.check_statement_end()
            return ReturnNode(expr)

        # Handle del statement for heap deallocation
        if self.token.type == TT_DEL:
            self.advance()
            expr = self.expression(0)

            # Verify expr is a reference type
            if not is_ref_type(expr.expr_type):
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
                expr = self.expression(0)

                # In global scope, ensure only literal initializers
                if self.current_function is None and self.seen_main_function:
                    self.error("Global variables must be declared before main function")

                if self.current_function is None and not is_literal_node(expr):
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

                # Parse the initializer expression
                expr = self.expression(0)

                # In global scope, ensure only literal initializers
                if self.current_function is None and self.seen_main_function:
                    self.error("Global variables must be declared before main function")

                if self.current_function is None and not is_literal_node(expr):
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

            # Declare the variable in current scope
            self.declare_variable(var_name, var_type, decl_type == TT_CONST)
            self.check_statement_end()
            return VarDeclNode(decl_type, var_name, var_type, expr)

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
                if member_node.node_type == AST_NODE_MEMBER_ACCESS and self.token.type == TT_ASSIGN:
                    self.advance()  # Skip =
                    expr = self.expression(0)

                    # Check type compatibility
                    self.check_field_compatibility(expr.expr_type, member_node.expr_type)

                    # Create binary op node for the assignment
                    node = BinaryOpNode('=', member_node, expr, member_node.expr_type)
                    self.check_statement_end()
                    return node

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
                    # Check if variable is a constant (declared with 'let')
                    if self.is_constant(var):
                        self.error("Cannot reassign to constant '%s'" % var)

                    op = self.token.type
                    var_type = self.get_variable_type(var)

                    # Advance past the operator
                    self.advance()

                    # Parse the expression
                    expr = self.expression(0)

                    # Check type compatibility for assignments
                    self.check_type_compatibility(var, expr.expr_type)

                    # For compound operators, use CompoundAssignNode
                    if op != TT_ASSIGN:
                        self.check_statement_end()
                        return CompoundAssignNode(op, var, expr, var_type)

                    # Regular assignment
                    self.check_statement_end()
                    return AssignNode(var, expr, var_type)

                # Handle expression statements (e.g., an identifier by itself)
                var_type = self.get_variable_type(var)
                expr = VariableNode(var, var_type)
                self.check_statement_end()
                return ExprStmtNode(expr)
        elif self.token.type in [TT_INT_LITERAL, TT_UINT_LITERAL, TT_LONG_LITERAL, TT_ULONG_LITERAL, 
                                TT_FLOAT_LITERAL, TT_STRING_LITERAL, TT_LPAREN, TT_MINUS, TT_NOT, TT_BITNOT]:
            # Also handle expressions that start with other tokens
            expr = self.expression(0)
            self.check_statement_end()
            return ExprStmtNode(expr)

        # If we're in global scope and not at a var/let/function declaration, error
        if self.current_function is None:
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
