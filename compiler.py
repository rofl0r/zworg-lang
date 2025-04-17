# Implementation of a Pratt parser in Python 2.7

from shared import *
from lexer import Token, Lexer

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
        self.expr_type = expr_type  # TYPE_INT, TYPE_FLOAT, TYPE_UINT, TYPE_LONG, TYPE_ULONG

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

def is_literal_node(node):
    """Check if a node represents a literal value (for global var init)"""
    return node.node_type in [AST_NODE_NUMBER, AST_NODE_STRING]

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.token = self.lexer.next_token() # Current token
        self.prev_token = None  # Previous token (for better error messages)

        # Per-scope tracking structures
        self.scopes = ["global"]  # Stack of scope names
        self.variables = {"global": set()}  # Track declared variables per scope
        self.constants = {"global": set()}  # Track constants (let declarations) per scope
        self.var_types = {"global": {}}     # Track variable types per scope

        # Track if we've seen functions - used to enforce globals-before-functions rule
        self.seen_main_function = False

        self.functions = {}     # Track function declarations (name -> (params, return_type))
        self.current_function = None  # Track current function for return checking

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
        token_type_name = token_name(self.token.type)
        raise CompilerException("%s at line %d, column %d. Token: %s (%s)" %
                       (message, self.token.line, self.token.column,
                        self.token.value, token_type_name))

    def advance(self):
        self.prev_token = self.token
        self.token = self.lexer.next_token()

    def consume(self, token_type):
        if self.token.type == token_type:
            self.advance()
        else:
            expected_type_name = token_name(token_type)
            actual_type_name = token_name(self.token.type)
            self.error('Expected %s but got %s' %
                       (expected_type_name, actual_type_name))

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

        # Check compatibility using can_promote function
        if var_type is not None and var_type != TYPE_UNKNOWN and expr_type != TYPE_UNKNOWN and not can_promote(expr_type, var_type):
            self.error("Type mismatch: can't assign a value of type %s to %s (type %s)" % 
                       (var_type_to_string(expr_type), var_name, var_type_to_string(var_type)))

    def function_declaration(self):
        """Parse a function declaration"""
        self.advance()  # Skip 'def'
        # Parse function name
        if self.token.type != TT_IDENT:
            self.error("Expected function name after 'def'")
        name = self.token.value
        if name == "main":# Mark that we've seen main- used to enforce globals-before-main rule
            self.seen_main_function = True
        self.advance()

        # Parse parameters
        self.consume(TT_LPAREN)
        params = []
        while self.token.type != TT_RPAREN:
            tmp = self.parameter()
            for n, _ in params:
                if n == tmp[0]:
                    self.error("Parameter '%s' is already defined" % n)
            params.append(tmp)
            if self.token.type == TT_COMMA:
                self.advance()  # Skip comma
        self.consume(TT_RPAREN)

        # Parse return type (if specified)
        return_type = TYPE_VOID  # Default to void (implicitly)
        if self.token.type == TT_COLON:
            self.advance()
            if self.token.type in TYPE_TOKEN_MAP:
                return_type = TYPE_TOKEN_MAP[self.token.type]
                self.advance()
            else:
                self.error("Expected type name after ':'")

        # Register function
        if name in self.functions:
            self.error("Function '%s' is already defined" % name)
        self.functions[name] = (params, return_type)

        # Enter function scope
        self.enter_scope(name)

        # Add parameters to function scope variables
        for param_name, param_type in params:
            # Add parameter to function scope
            self.variables[name].add(param_name)
            self.var_types[name][param_name] = param_type

        # Save and set current function for return checking
        prev_function = self.current_function
        self.current_function = name

        # Parse function body statements
        body = self.doblock()

        # Restore previous context
        self.current_function = prev_function
        self.leave_scope()

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
        if self.token.type not in TYPE_TOKEN_MAP:
            self.error("Expected type name after ':'")

        param_type = TYPE_TOKEN_MAP[self.token.type]
        self.advance()

        return (name, param_type)

    def determine_result_type(self, left_type, right_type):
        """Determine the result type of a binary operation based on operand types"""
        if left_type != right_type:
            self.error("Type mismatch: cannot operate on values of different types")
        return left_type

    def nud(self, t):
        # Handle number literals using the type mapping
        if t.type in TOKEN_TO_TYPE_MAP:
            return NumberNode(t.value, TOKEN_TO_TYPE_MAP[t.type])

        if t.type == TT_STRING_LITERAL:
            return StringNode(t.value)

        if t.type == TT_IDENT:
            var_name = t.value
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
            expr = self.expression(0)
            self.consume(TT_RPAREN)
            return expr

        raise CompilerException('Unexpected token type %d' % t.type)

    def led(self, t, left):
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

        if t.type in [TT_PLUS, TT_MINUS, TT_MULT, TT_DIV, TT_MOD, TT_SHL, TT_SHR]:
            right = self.expression(self.lbp(t))

            # If types don't match, we need to fail
            if can_promote(left.expr_type, right.expr_type):
                result_type = right.expr_type
            elif can_promote(right.expr_type, left.expr_type):
                result_type = left.expr_type
            else:
                self.error("Type mismatch in binary operation: %s and %s" %
                          (var_type_to_string(left.expr_type), var_type_to_string(right.expr_type)))

            # Special handling for string concatenation
            if t.type == TT_PLUS and (left.expr_type == TYPE_STRING or right.expr_type == TYPE_STRING):
                if left.expr_type != TYPE_STRING or right.expr_type != TYPE_STRING:
                    self.error("Cannot concatenate string with non-string type")
                result_type = TYPE_STRING
            # Normal case - determine result type based on operands
            elif left.expr_type != TYPE_UNKNOWN and left.expr_type == right.expr_type:
                result_type = left.expr_type
            else:
                # Use the type with the highest precedence from TYPE_PRECEDENCE list
                for tp in TYPE_PRECEDENCE:
                    if left.expr_type == tp or right.expr_type == tp:
                        result_type = tp
                        break

            return BinaryOpNode(t.value, left, right, result_type)

        elif t.type in [TT_EQ, TT_NE, TT_GE, TT_LE, TT_LT, TT_GT]:
            right = self.expression(self.lbp(t))
            # Comparisons always return an integer (0/1 representing false/true)
            return CompareNode(t.value, left, right)

        elif t.type in [TT_AND, TT_OR]:
            right = self.expression(self.lbp(t))
            # Logical operations always return an integer (0/1 representing false/true)
            return LogicalNode(t.value, left, right)

        elif t.type in [TT_XOR, TT_BITOR, TT_BITAND]:
            right = self.expression(self.lbp(t))
            # Bit operations are performed on integers and return integers
            return BitOpNode(t.value, left, right)

        raise CompilerException('Unexpected token type %d' % t.type)

    def parse_type(self):
        """Parse a type annotation or return None if not present"""
        if self.token.type == TT_COLON:
            self.advance()  # Consume the colon

            # Check if token is a valid type token
            if self.token.type in TYPE_TOKEN_MAP:
                var_type = TYPE_TOKEN_MAP[self.token.type]
                self.advance()  # Consume the type token
                return var_type
            else:
                self.error("Expected type name after ':'")
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
        if self.token.type != TT_RPAREN:
                args.append(self.expression(0))
                while self.token.type == TT_COMMA:
                        self.advance()  # Skip comma
                        args.append(self.expression(0))

        self.consume(TT_RPAREN)

        # Type checking for function call
        func_params, func_return_type = self.functions[func_name]

        # Check number of arguments
        if len(args) != len(func_params):
                self.error("Function '%s' expects %d arguments, got %d" %
                                  (func_name, len(func_params), len(args)))
        # Check argument types
        for i, ((param_name, param_type), arg) in enumerate(zip(func_params, args)):
                if arg.expr_type != param_type and not can_promote(arg.expr_type, param_type):
                        self.error("Type mismatch for argument %d of function '%s': expected %s, got %s" %
                                         (i+1, func_name, var_type_to_string(param_type), var_type_to_string(arg.expr_type)))
        return FunctionCallNode(func_name, args, func_return_type)

    def statement(self):
        self.skip_separators()

        # Handle function declarations
        if self.token.type == TT_DEF:
            # Only allowed in global scope
            if self.current_function is not None:
                self.error("Nested function declarations are not allowed")
            return self.function_declaration()

        # Handle return statements
        if self.token.type == TT_RETURN:
            # Must be inside a function
            if self.current_function is None:
                self.error("'return' statement outside function")

            self.advance()

            # Return with no value
            if self.token.type in [TT_SEMI, TT_EOF] or (self.prev_token and self.token.line > self.prev_token.line):
                self.check_statement_end()
                return ReturnNode(None)

            # Return with value
            expr = self.expression(0)

            # Check if return type matches function return type
            func_return_type = self.functions[self.current_function][1]
            if func_return_type == TYPE_VOID:
                self.error("Void function '%s' cannot return a value" % self.current_function)

            self.check_statement_end()
            return ReturnNode(expr)

        # Handle variable declarations (var and let)
        if self.token.type in [TT_VAR, TT_LET]:
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
                if expr.expr_type != TYPE_UNKNOWN and var_type != expr.expr_type and not can_promote(expr.expr_type, var_type):
                    self.error("Type mismatch in initialization: can't assign %s to %s (type %s)" % 
                              (var_type_to_string(expr.expr_type), var_name, var_type_to_string(var_type)))
            else:
                self.error("Variable declaration must include an initialization")
            # Register the variable as defined in the current scope
            if var_name in self.variables[self.current_scope()]:
                self.error("Variable '%s' is already declared in this scope" % var_name)

            # Declare the variable in current scope
            self.declare_variable(var_name, var_type, decl_type == TT_LET)
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

