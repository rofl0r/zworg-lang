# Implementation of interpreter for the AST with C-style type handling

from __future__ import division  # Use true division
from shared import *
from cops import add, subtract, multiply, divide, modulo, shift_left, shift_right, negate, logical_not, bitwise_not, compare_eq, compare_ne, compare_lt, compare_le, compare_gt, compare_ge, logical_and, logical_or, bitwise_and, bitwise_or, bitwise_xor
import type_registry

# Struct instance class for runtime
class StructInstance:
    """Object to store a struct's fields at runtime"""

    def __init__(self, struct_id, struct_name):
        self.struct_id = struct_id
        self.struct_name = struct_name
        self.fields = {}  # field_name -> value

    def __repr__(self):
        fields_str = ", ".join(["%s=%s" % (name, value) for name, value in self.fields.items()])
        return "%s(%s)" % (self.struct_name, fields_str)

class Interpreter(object):
    def __init__(self, environment=None):
        """Initialize the interpreter with an optional environment"""
        self.environment = environment if environment else EnvironmentStack()

        # Create node type to visitor method mapping
        self.visitor_map = {
            AST_NODE_NUMBER: self.visit_number,
            AST_NODE_STRING: self.visit_string,
            AST_NODE_VARIABLE: self.visit_variable,
            AST_NODE_BINARY_OP: self.visit_binary_op,
            AST_NODE_UNARY_OP: self.visit_unary_op,
            AST_NODE_ASSIGN: self.visit_assign,
            AST_NODE_COMPOUND_ASSIGN: self.visit_compound_assign,
            AST_NODE_PRINT: self.visit_print,
            AST_NODE_IF: self.visit_if,
            AST_NODE_WHILE: self.visit_while,
            AST_NODE_BREAK: self.visit_break,
            AST_NODE_CONTINUE: self.visit_continue,
            AST_NODE_EXPR_STMT: self.visit_expr_stmt,
            AST_NODE_VAR_DECL: self.visit_var_decl,
            AST_NODE_FUNCTION_DECL: self.visit_function_decl,
            AST_NODE_CALL: self.visit_call,
            AST_NODE_RETURN: self.visit_return,
            AST_NODE_COMPARE: self.visit_compare,
            AST_NODE_LOGICAL: self.visit_logical,
            AST_NODE_BITOP: self.visit_bitop,
            # Struct-related nodes
            AST_NODE_STRUCT_DEF: self.visit_struct_def,
            AST_NODE_STRUCT_INIT: self.visit_struct_init,
            AST_NODE_MEMBER_ACCESS: self.visit_member_access,
            AST_NODE_NEW: self.visit_new,
            AST_NODE_DEL: self.visit_del,
            AST_NODE_TUPLE: self.visit_tuple,
            AST_NODE_STRUCT_INITIALIZER: self.visit_struct_initializer,
        }

    def reset(self):
        type_registry.reset_registry()
        if self.environment: self.environment.reset()

    def run(self, text):
        from lexer import Lexer
        from compiler import Parser

        lexer = Lexer(text)
        parser = Parser(lexer)
        try:
            # Parse the program
            program = parser.parse()
            ast = program
            interpreter = self

            # First process struct definitions
            for node in program:
                if node.node_type == AST_NODE_STRUCT_DEF:
                    interpreter.evaluate(node)

            # Execute global variable declarations
            for node in program:
                if node.node_type == AST_NODE_VAR_DECL:
                    interpreter.evaluate(node)

            main_func_id = type_registry.lookup_function("main")

            if main_func_id == -1:
                return {'success': False, 'error': "No 'main' function defined", 'ast': ast}

            main_func = type_registry.get_func_from_id(main_func_id).ast_node

            # Make sure main has no parameters
            if len(main_func.params) > 0:
                return {'success': False, 'error': "Function 'main' cannot have parameters", 'ast': ast}

            # Create a new scope for main function
            interpreter.environment.enter_scope()

            try:
                # Execute main function
                for stmt in main_func.body:
                    interpreter.evaluate(stmt)

                # Return both global and main's environment
                return {
                    'success': True,
                    'global_env': interpreter.environment.stack[0],
                    'main_env': interpreter.environment.stack[1],
                    'ast': ast
                }
            except ReturnException as ret:
                # If main returns a value, include it in the result
                return {
                    'success': True,
                    'result': ret.value,
                    'global_env': interpreter.environment.stack[0],
                    'main_env': interpreter.environment.stack[1],
                    'ast': ast
                }
        except CompilerException as e:
            return {'success': False, 'error': str(e), 'ast': None}

    def evaluate(self, node):
        """Main entry point to evaluate an AST node"""
        if node is None:
            return None

        # If given a list of nodes (program), execute each one
        if isinstance(node, list):
            result = None
            for n in node:
                result = self.evaluate(n)
            return result

        # Dispatch to the appropriate visitor method
        if node.node_type in self.visitor_map:
            visitor = self.visitor_map[node.node_type]
            return visitor(node)

        raise CompilerException("No visit method defined for node type: %s" % ast_node_type_to_string(node.node_type))

    def check_and_set_params(self, method_name, method_params, args, arg_nodes):
        """Helper function to check parameter types and set them in current scope
        Args:
            method_name: Name of method/constructor for error messages
            method_params: List of (param_name, param_type) tuples
            args: List of evaluated argument values
            arg_nodes: List of argument AST nodes for type information
        """
        if len(args) != len(method_params):
            self.environment.leave_scope()
            raise CompilerException("%s expects %d arguments, got %d" % 
                                  (method_name, len(method_params), len(args)))

        for i in range(len(method_params)):
            param_name, param_type = method_params[i]
            arg_value = args[i]
            if not can_promote(arg_nodes[i].expr_type, param_type):
                self.environment.leave_scope()
                raise CompilerException("Type mismatch in %s argument: cannot convert %s to %s" %
                                      (method_name, var_type_to_string(arg_nodes[i].expr_type), 
                                       var_type_to_string(param_type)))
            self.environment.set(param_name, arg_value)

    def visit_number(self, node):
        """Evaluate a number node (integer or float)"""
        return node.value

    def visit_string(self, node):
        """Evaluate a string node"""
        return node.value

    def visit_variable(self, node):
        """Evaluate a variable reference node"""
        if not self.environment.has(node.name):
            raise CompilerException("Variable '%s' is not defined" % node.name)
        return self.environment.get(node.name)

    def visit_binary_op(self, node):
        """Evaluate a binary operation node"""
        left_val = self.evaluate(node.left)
        right_val = self.evaluate(node.right)

        if node.operator == '+':
            # Handle string concatenation
            if node.left.expr_type == TYPE_STRING and node.right.expr_type == TYPE_STRING:
                return left_val + right_val

            return add(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == '-':
            return subtract(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == '*':
            return multiply(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == '/':
            return divide(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == '%':
            return modulo(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == 'shl':
            return shift_left(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == 'shr':
            return shift_right(left_val, right_val, node.left.expr_type, node.right.expr_type)
        # Special case for assignment to struct field (obj.field = value)
        elif node.operator == '=' and node.left.node_type == AST_NODE_MEMBER_ACCESS:
            # First evaluate the object to get the struct instance
            obj = self.evaluate(node.left.obj)

            # Check if obj is a struct instance
            if not isinstance(obj, StructInstance):
                raise CompilerException("Cannot assign to field of non-struct value")

            # Set the field value
            obj.fields[node.left.member_name] = right_val
            return right_val

        raise CompilerException("Unknown binary operator: %s" % node.operator)

    def visit_unary_op(self, node):
        """Evaluate a unary operation node"""
        value = self.evaluate(node.operand)

        if node.operator == '-':
            return negate(value, node.operand.expr_type)
        elif node.operator == '!':
            return logical_not(value)
        elif node.operator == 'bitnot':
            return bitwise_not(value, node.operand.expr_type)

        raise CompilerException("Unknown unary operator: %s" % node.operator)

    def visit_assign(self, node):
        """Evaluate an assignment node"""
        value = self.evaluate(node.expr)

        # Check if type promotion is needed and allowed
        if node.expr_type != node.expr.expr_type:
            if not can_promote(node.expr.expr_type, node.expr_type):
                raise CompilerException("Cannot assign %s to %s" % 
                                      (var_type_to_string(node.expr.expr_type), var_type_to_string(node.expr_type)))

        # Handle number literal promotion
        if node.expr.node_type == AST_NODE_NUMBER:
            value = promote_literal_if_needed(value, node.expr.expr_type, node.expr_type)

        self.environment.set(node.var_name, value)
        return value

    def visit_compound_assign(self, node):
        """Evaluate a compound assignment node (+=, -=, etc.)"""
        current_value = self.environment.get(node.var_name)
        expr_value = self.evaluate(node.expr)

        if node.op_type == TT_PLUS_ASSIGN:
            # Handle string concatenation for += operator
            if node.expr_type == TYPE_STRING:
                if node.expr.expr_type != TYPE_STRING:
                    raise CompilerException("Cannot use += with string and %s" % var_type_to_string(node.expr.expr_type))
                result = current_value + expr_value
            else:
                result = add(current_value, expr_value, node.expr_type, node.expr.expr_type)
        elif node.op_type == TT_MINUS_ASSIGN:
            result = subtract(current_value, expr_value, node.expr_type, node.expr.expr_type)
        elif node.op_type == TT_MULT_ASSIGN:
            result = multiply(current_value, expr_value, node.expr_type, node.expr.expr_type)
        elif node.op_type == TT_DIV_ASSIGN:
            result = divide(current_value, expr_value, node.expr_type, node.expr.expr_type)
        elif node.op_type == TT_MOD_ASSIGN:
            result = modulo(current_value, expr_value, node.expr_type, node.expr.expr_type)
        else:
            raise CompilerException("Unknown compound assignment operator: %s" % token_name(node.op_type))

        self.environment.set(node.var_name, result)
        return result

    def visit_print(self, node):
        """Evaluate a print statement node"""
        value = self.evaluate(node.expr)
        print(value)
        return value

    def visit_if(self, node):
        """Evaluate an if statement node"""
        if self.evaluate(node.condition):
            for stmt in node.then_body:
                self.evaluate(stmt)
        elif node.else_body:
            for stmt in node.else_body:
                self.evaluate(stmt)
        return 0

    def visit_while(self, node):
        """Evaluate a while loop node"""
        while self.evaluate(node.condition):
            try:
                for stmt in node.body:
                    try:
                        self.evaluate(stmt)
                    except ContinueException:
                        break
                    except BreakException:
                        raise
            except BreakException:
                break
        return 0

    def visit_break(self, node):
        """Evaluate a break statement node"""
        raise BreakException()

    def visit_continue(self, node):
        """Evaluate a continue statement node"""
        raise ContinueException()

    def visit_expr_stmt(self, node):
        """Evaluate an expression statement node"""
        return self.evaluate(node.expr)

    def visit_var_decl(self, node):
        """Evaluate a variable declaration node"""
        value = self.evaluate(node.expr)
        # types are already checked in compiler
        self.environment.set(node.var_name, value)
        return value

    def visit_function_decl(self, node):
        """Evaluate a function declaration node"""
        # Functions are already registered in the type registry during parsing
        return 0

    def visit_return(self, node):
        """Evaluate a return statement node"""
        value = None if node.expr is None else self.evaluate(node.expr)
        # Throw a special exception to unwind the call stack
        raise ReturnException(value)

    def visit_compare(self, node):
        """Evaluate a comparison node"""
        left_val = self.evaluate(node.left)
        right_val = self.evaluate(node.right)

        # Handle string comparison operations
        if node.left.expr_type == TYPE_STRING and node.right.expr_type == TYPE_STRING:
            if node.operator == '==':
                return 1 if left_val == right_val else 0
            elif node.operator == '!=':
                return 1 if left_val != right_val else 0
            # Other comparison operators are not supported for strings
            elif node.operator in ['>', '>=', '<', '<=']:
                raise CompilerException("Operator %s not supported for strings" % node.operator)
            else:
                # Unknown operator
                raise CompilerException("Unknown comparison operator: %s" % node.operator)

        if node.operator == '==':
            return compare_eq(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == '!=':
            return compare_ne(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == '>=':
            return compare_ge(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == '>':
            return compare_gt(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == '<':
            return compare_lt(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == '<=':
            return compare_le(left_val, right_val, node.left.expr_type, node.right.expr_type)

        raise CompilerException("Unknown comparison operator: %s" % node.operator)

    def visit_logical(self, node):
        """Evaluate a logical operation node"""
        left_val = self.evaluate(node.left)

        if node.operator == 'and':
            return logical_and(left_val, self.evaluate(node.right))
        elif node.operator == 'or':
            return logical_or(left_val, self.evaluate(node.right))

        raise CompilerException("Unknown logical operator: %s" % node.operator)

    def visit_bitop(self, node):
        """Evaluate a bitwise operation node"""
        left_val = self.evaluate(node.left)
        right_val = self.evaluate(node.right)

        if node.operator == '&':
            return bitwise_and(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == '|':
            return bitwise_or(left_val, right_val, node.left.expr_type, node.right.expr_type)
        elif node.operator == 'xor':
            return bitwise_xor(left_val, right_val, node.left.expr_type, node.right.expr_type)

        raise CompilerException("Unknown bitwise operator: %s" % node.operator)

    # Struct-related visitor methods
    def visit_struct_def(self, node):
        """Visit a struct definition node - Nothing to do at runtime"""
        # Struct definitions are handled at parse time
        return None

    def visit_struct_init(self, node):
        """Visit a struct initialization node"""
        # Create a new struct instance
        instance = StructInstance(node.struct_id, node.struct_name)

        # Initialize fields with default values first
        all_fields = type_registry.get_all_fields(node.struct_name)
        for field_name, field_type in all_fields:
            # Set default value based on type
            if is_struct_type(field_type):
                # For now, we don't auto-initialize nested structs
                instance.fields[field_name] = None
            elif field_type == TYPE_STRING:
                instance.fields[field_name] = ""
            elif is_float_type(field_type):
                instance.fields[field_name] = 0.0
            else:
                instance.fields[field_name] = 0

        # Call constructor if it exists and there are args or it's "init"
        init_method = type_registry.get_method(node.struct_name, "init")
        if init_method:
            # Evaluate and pass constructor arguments
            # these need to be evaluated BEFORE constructing a new scope!
            # else self used in an expression ment for other methods calls
            # now points to the instance.
            args = [self.evaluate(arg) for arg in node.args]
            self.check_and_set_params("Constructor for '%s'" % node.struct_name, init_method.params, args, node.args)

            # Create a temporary scope for the constructor
            self.environment.enter_scope()
            self.environment.set("self", instance)

            # Execute constructor body
            try:
                for stmt in init_method.body:
                    self.evaluate(stmt)
            except ReturnException:
                # Ignore return value from constructor
                pass

            self.environment.leave_scope()

        return instance

    def visit_member_access(self, node):
        """Visit a member access node (obj.field)"""
        # Evaluate the object expression
        obj = self.evaluate(node.obj)

        # Handle nil reference
        if obj is None:
            raise CompilerException("Attempt to access field of a nil reference")

        # Make sure it's a struct instance
        if not isinstance(obj, StructInstance):
            raise CompilerException("Cannot access member '%s' on non-struct value" % node.member_name)

        # Get the field value
        if node.member_name not in obj.fields:
            raise CompilerException("Field '%s' not found in struct '%s'" % (node.member_name, obj.struct_name))

        return obj.fields[node.member_name]

    def visit_call(self, node):
        """Unified handler for function and method calls using only the type registry"""
        # Determine if this is a method call (has an object context)
        is_method_call = node.obj is not None

        if is_method_call:
            # Evaluate the object
            obj = self.evaluate(node.obj)

            # Handle nil reference
            if obj is None:
                raise CompilerException("Attempt to call method '%s' on a nil reference" % node.name)

            # Make sure it's a struct instance
            if not isinstance(obj, StructInstance):
                raise CompilerException("Cannot call method '%s' on non-struct value" % node.name)

            # Get the method ID from registry
            method_id = type_registry.lookup_function(node.name, obj.struct_id)
            if method_id == -1:
                raise CompilerException("Method '%s' not found in struct '%s'" % (node.name, obj.struct_name))

            # Create context description for error messages
            context_name = "Method '%s'" % node.name

            # Enter method scope and set 'self'
            self.environment.enter_scope()
            self.environment.set("self", obj)
        else:
            # Get function ID from registry
            method_id = type_registry.lookup_function(node.name)
            if method_id == -1:
                raise CompilerException("Function '%s' is not defined" % node.name)

            # Create context description for error messages
            context_name = "Function '%s'" % node.name

            # Enter function scope
            self.environment.enter_scope()

        # Get function details from registry
        func_obj = type_registry.get_func_from_id(method_id)

        # Evaluate arguments
        args = [self.evaluate(arg) for arg in node.args]

        # Bind parameters
        self.check_and_set_params(context_name, func_obj.params, args, node.args)

        # Default return value for void functions
        result = None

        try:
            # Execute body
            for stmt in func_obj.ast_node.body:
                self.evaluate(stmt)

            # If no return and non-void return type, that's an error
            if func_obj.return_type != TYPE_VOID:
                self.environment.leave_scope()  # Clean up before raising exception
                raise CompilerException("%s has non-void return type but reached end without returning" % context_name)

        except ReturnException as ret:
            # Check return value type against function's return type
            if func_obj.return_type == TYPE_VOID and ret.value is not None:
                self.environment.leave_scope()  # Clean up before raising exception
                raise CompilerException("Void %s returned a value" % 
                                      ("method" if is_method_call else "function"))

            result = ret.value

        # Clean up scope
        self.environment.leave_scope()
        return result

    def visit_new(self, node):
        """Visit a new expression for heap allocation"""
        # Evaluate the struct initialization
        instance = self.evaluate(node.struct_init)
        return instance  # The expression type is already set to reference type

    def visit_del(self, node):
        """Visit a del statement for heap deallocation"""
        # Evaluate the expression
        obj = self.evaluate(node.expr)

        # Handle nil reference
        if obj is None:
            raise CompilerException("Attempt to delete a nil reference")

        # Call destructor if it exists
        if isinstance(obj, StructInstance):
            fini_method = type_registry.get_method(obj.struct_name, "fini")
            if fini_method:
                # Create a temporary scope for the destructor
                self.environment.enter_scope()
                self.environment.set("self", obj)

                # Execute destructor body
                try:
                    for stmt in fini_method.body:
                        self.evaluate(stmt)
                except ReturnException:
                    # Ignore return from destructor
                    pass

                self.environment.leave_scope()

        # Mark object as deleted (set to None)
        # In a real implementation, this would free memory
        return None

    def visit_tuple(self, node):
        """Visit a tuple expression node"""
        # Since tuples are implemented as anonymous structs, we need to:
        # 1. Get the tuple's struct type name from its expr_type
        # 2. Create a struct instance
        # 3. Set each field (_0, _1, etc.) to its corresponding value
        
        # Get struct name from type registry
        struct_name = type_registry.get_struct_name(node.expr_type)
        if not struct_name:
            raise CompilerException("Unknown tuple type")
            
        # Create a struct instance for the tuple
        instance = StructInstance(node.expr_type, struct_name)
        
        # Evaluate each element and set the corresponding field
        for i, elem in enumerate(node.elements):
            field_name = "_%d" % i
            value = self.evaluate(elem)
            instance.fields[field_name] = value
            
        return instance

    def visit_struct_initializer(self, node):
        """Visit a struct initializer with named fields {.field1=value1, ...}"""
        # Create a new struct instance of the appropriate type
        if node.struct_type is None:
            raise CompilerException("Struct type not set for initializer")
            
        struct_name = type_registry.get_struct_name(node.struct_type)
        if not struct_name:
            raise CompilerException("Unknown struct type")
            
        # Create the struct instance
        instance = StructInstance(node.struct_type, struct_name)
        
        # Initialize all fields with default values first
        all_fields = type_registry.get_all_fields(struct_name)
        for field_name, field_type in all_fields:
            # Set default value based on type
            if is_struct_type(field_type):
                instance.fields[field_name] = None
            elif field_type == TYPE_STRING:
                instance.fields[field_name] = ""
            elif is_float_type(field_type):
                instance.fields[field_name] = 0.0
            else:
                instance.fields[field_name] = 0

        # Now set the specified field values
        for field_name, field_expr in node.field_initializers:
            # Evaluate the field value
            value = self.evaluate(field_expr)
            
            # Set the field value
            instance.fields[field_name] = value
            
        return instance

# Custom exceptions for control flow
class BreakException(Exception):
    """Raised when a break statement is encountered"""
    pass

class ReturnException(Exception):
    """Raised when a return statement is encountered"""
    def __init__(self, value=None):
        self.value = value

class ContinueException(Exception):
    """Raised when a continue statement is encountered"""
    pass

class EnvironmentStack:
    """Stack-based environment implementation with support for scopes"""
    def reset(self):
        self.stack = [{}]  # Start with global scope at index 0
        self.stackptr = 0

    def __init__(self):
        self.reset()

    def enter_scope(self):
        """Enter a new scope - reuse existing or create new one"""
        self.stackptr += 1
        if self.stackptr >= len(self.stack):
            self.stack.append({})
        else:
            # Reuse existing dict but clear it
            self.stack[self.stackptr].clear()

    def leave_scope(self):
        """Leave current scope and return to previous"""
        if self.stackptr > 0:
            self.stackptr -= 1

    def get(self, name):
        """Get a variable value looking through all accessible scopes"""
        # Search from current scope down to global
        for i in range(self.stackptr, -1, -1):
            if name in self.stack[i]:
                return self.stack[i][name]
        raise KeyError(name)

    def has(self, name):
        """Check if a variable exists in any accessible scope"""
        for i in range(self.stackptr, -1, -1):
            if name in self.stack[i]:
                return True
        return False

    def set(self, name, value):
        """Set a variable in the current scope"""
        self.stack[self.stackptr][name] = value

# runtime emulation of C promotions.
# TODO coverage says the first branch is always taken.
# investigate why that is so, and remove if the promotions
# already take place elsewhere.
def promote_literal_if_needed(value, from_type, to_type):
    """Promote a literal value to a new type if needed"""
    # If types are the same, no promotion needed
    if from_type == to_type:
        return value

    # Handle integer promotions
    if is_integer_type(from_type) and is_integer_type(to_type):
        # For unsigned types, mask to appropriate size
        if is_unsigned_type(to_type):
            return truncate_to_unsigned(value, to_type)
        return value

    # Handle float/double promotions
    if is_float_type(from_type) and is_float_type(to_type):
        return float(value)

    # Handle int to float promotions
    if is_integer_type(from_type) and is_float_type(to_type):
        return float(value)

    return value

