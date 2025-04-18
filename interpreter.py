# Implementation of interpreter for the AST with C-style type handling

from __future__ import division  # Use true division
from shared import *
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

# Binary arithmetic operators with C-style type promotion
def add(left, right, left_type, right_type):
    """Addition with C semantics"""
    if is_float_type(left_type) or is_float_type(right_type):
        # Convert to double if either operand is double
        if TYPE_DOUBLE in (left_type, right_type):
            return float(left) + float(right)
        # Otherwise use float precision
        return float(left) + float(right)

    # String concatenation
    if left_type == TYPE_STRING and right_type == TYPE_STRING:
        return str(left) + str(right)

    # Integer addition - follow C promotion rules
    result = int(left) + int(right)

    # Handle overflow according to type
    if is_unsigned_type(left_type) or is_unsigned_type(right_type):
        # For unsigned types, result wraps around
        return truncate_to_unsigned(result, max(left_type, right_type))

    return result

def subtract(left, right, left_type, right_type):
    """Subtraction with C semantics"""
    if is_float_type(left_type) or is_float_type(right_type):
        if TYPE_DOUBLE in (left_type, right_type):
            return float(left) - float(right)
        return float(left) - float(right)

    result = int(left) - int(right)

    if is_unsigned_type(left_type) or is_unsigned_type(right_type):
        return truncate_to_unsigned(result, max(left_type, right_type))

    return result

def multiply(left, right, left_type, right_type):
    """Multiplication with C semantics"""
    if is_float_type(left_type) or is_float_type(right_type):
        if TYPE_DOUBLE in (left_type, right_type):
            return float(left) * float(right)
        return float(left) * float(right)

    result = int(left) * int(right)

    if is_unsigned_type(left_type) or is_unsigned_type(right_type):
        return truncate_to_unsigned(result, max(left_type, right_type))

    return result

def divide(left, right, left_type, right_type):
    """Division with C semantics"""
    if right == 0:
        raise ZeroDivisionError("Division by zero")

    if is_float_type(left_type) or is_float_type(right_type):
        if TYPE_DOUBLE in (left_type, right_type):
            return float(left) / float(right)
        return float(left) / float(right)

    # Integer division - follow C rules
    if is_unsigned_type(left_type) and is_unsigned_type(right_type):
        # Both unsigned, use integer division
        return int(left) // int(right)
    elif is_signed_type(left_type) and is_signed_type(right_type):
        # Both signed, use C truncation toward zero
        result = abs(left) // abs(right)
        if (left < 0) != (right < 0):  # If signs differ
            return -result
        return result
    else:
        # Mixed signed/unsigned - follow C promotion rules
        # Treat as unsigned if either operand is unsigned
        return int(left) // int(right)

def modulo(left, right, left_type, right_type):
    """Modulus with C semantics"""
    if right == 0:
        raise ZeroDivisionError("Modulo by zero")

    if is_float_type(left_type) or is_float_type(right_type):
        raise TypeError("Modulo not defined for floating point types")

    # Integer modulo - follow C rules
    if is_unsigned_type(left_type) and is_unsigned_type(right_type):
        # Both unsigned
        return int(left) % int(right)
    elif is_signed_type(left_type) and is_signed_type(right_type):
        # Both signed - C99 requires sign of result to match dividend
        result = abs(left) % abs(right)
        if left < 0:
            return -result
        return result
    else:
        # Mixed signed/unsigned - follow C promotion rules
        return int(left) % int(right)

def shift_left(left, right, left_type, right_type):
    """Left shift with C semantics"""
    if right < 0:
        raise ValueError("Negative shift count")

    result = int(left) << int(right)

    # Handle overflow according to type
    if is_unsigned_type(left_type):
        return truncate_to_unsigned(result, left_type)

    return result

def shift_right(left, right, left_type, right_type):
    """Right shift with C semantics"""
    if right < 0:
        raise ValueError("Negative shift count")

    # In C, right shift behavior depends on whether the left operand is signed
    if is_signed_type(left_type) and left < 0:
        # Arithmetic shift (preserve sign bit) for signed negative values
        return int(left) >> int(right)
    else:
        # Logical shift (fill with zeros) for unsigned or positive values
        return int(left) >> int(right)

# Unary operators
def negate(value, type_):
    """Unary negation with C semantics"""
    if is_float_type(type_):
        if type_ == TYPE_DOUBLE:
            return -float(value)
        return -float(value)
    return -int(value)

def logical_not(value):
    """Logical NOT with C semantics - returns 1 for false, 0 for true"""
    return 1 if not value else 0

def bitwise_not(value, type_):
    """Bitwise NOT with C semantics"""
    result = ~int(value)

    # Handle overflow according to type
    if is_unsigned_type(type_):
        return truncate_to_unsigned(result, _type)

    return result

# Comparison operators
def compare_eq(left, right, left_type, right_type):
    """Equality comparison with C semantics"""
    if is_float_type(left_type) or is_float_type(right_type):
        return 1 if float(left) == float(right) else 0
    return 1 if int(left) == int(right) else 0

def compare_ne(left, right, left_type, right_type):
    """Not equal comparison with C semantics"""
    if is_float_type(left_type) or is_float_type(right_type):
        return 1 if float(left) != float(right) else 0
    return 1 if int(left) != int(right) else 0

def compare_lt(left, right, left_type, right_type):
    """Less than comparison with C semantics"""
    if is_float_type(left_type) or is_float_type(right_type):
        return 1 if float(left) < float(right) else 0

    # For integers, handle sign differences according to C rules
    if is_signed_type(left_type) and is_unsigned_type(right_type):
        # Signed < Unsigned: if left is negative, result is true, else compare as unsigned
        if int(left) < 0:
            return 1
    elif is_unsigned_type(left_type) and is_signed_type(right_type):
        # Unsigned < Signed: if right is negative, result is false, else compare as unsigned
        if int(right) < 0:
            return 0

    return 1 if int(left) < int(right) else 0

def compare_le(left, right, left_type, right_type):
    """Less than or equal comparison with C semantics"""
    if is_float_type(left_type) or is_float_type(right_type):
        return 1 if float(left) <= float(right) else 0

    # For integers, handle sign differences according to C rules
    if is_signed_type(left_type) and is_unsigned_type(right_type):
        # Signed <= Unsigned: if left is negative, result is true, else compare as unsigned
        if int(left) < 0:
            return 1
    elif is_unsigned_type(left_type) and is_signed_type(right_type):
        # Unsigned <= Signed: if right is negative, result is false, else compare as unsigned
        if int(right) < 0:
            return 0

    return 1 if int(left) <= int(right) else 0

def compare_gt(left, right, left_type, right_type):
    """Greater than comparison with C semantics"""
    return 1 - compare_le(left, right, left_type, right_type)

def compare_ge(left, right, left_type, right_type):
    """Greater than or equal comparison with C semantics"""
    return 1 - compare_lt(left, right, left_type, right_type)

# Logical operators
def logical_and(left, right):
    """Logical AND with C semantics"""
    return 1 if left and right else 0

def logical_or(left, right):
    """Logical OR with C semantics"""
    return 1 if left or right else 0

# Bitwise operators
def bitwise_and(left, right, left_type, right_type):
    """Bitwise AND with C semantics"""
    result = int(left) & int(right)

    # Handle overflow according to result type
    result_type = max(left_type, right_type) #depends on right order of type constants
    if is_unsigned_type(result_type):
        return truncate_to_unsigned(result, result_type)

    return result

def bitwise_or(left, right, left_type, right_type):
    """Bitwise OR with C semantics"""
    result = int(left) | int(right)

    result_type = max(left_type, right_type)
    if is_unsigned_type(result_type):
        return truncate_to_unsigned(result, result_type)

    return result

def bitwise_xor(left, right, left_type, right_type):
    """Bitwise XOR with C semantics"""
    result = int(left) ^ int(right)

    result_type = max(left_type, right_type)

    if is_unsigned_type(result_type):
        return truncate_to_unsigned(result, result_type)

    return result

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
            AST_NODE_FUNCTION_CALL: self.visit_function_call,
            AST_NODE_RETURN: self.visit_return,
            AST_NODE_COMPARE: self.visit_compare,
            AST_NODE_LOGICAL: self.visit_logical,
            AST_NODE_BITOP: self.visit_bitop,
            # Struct-related nodes
            AST_NODE_STRUCT_DEF: self.visit_struct_def,
            AST_NODE_METHOD_DEF: self.visit_method_def,
            AST_NODE_STRUCT_INIT: self.visit_struct_init,
            AST_NODE_MEMBER_ACCESS: self.visit_member_access,
            AST_NODE_METHOD_CALL: self.visit_method_call,
            AST_NODE_NEW: self.visit_new,
            AST_NODE_DEL: self.visit_del
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

            # Then process method definitions
            for node in program:
                if node.node_type == AST_NODE_METHOD_DEF:
                    interpreter.evaluate(node)

            # Execute global variable declarations
            for node in program:
                if node.node_type == AST_NODE_VAR_DECL:
                    interpreter.evaluate(node)

            # Register functions in the function map (but don't execute them)
            for node in program:
                if node.node_type == AST_NODE_FUNCTION_DECL:
                    interpreter.environment.register_function(node.name, node)

            # Check if main function exists
            if not interpreter.environment.has_function("main"):
                return {'success': False, 'error': "No 'main' function defined", 'ast': ast}

            # Get main function
            main_func = interpreter.environment.get_function("main")

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

        for i, ((param_name, param_type), arg_value) in enumerate(zip(method_params, args)):
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
        # Store function in the environment
        self.environment.register_function(node.name, node)
        return 0

    def visit_function_call(self, node):
        """Evaluate a function call node"""
        # Get function from function map
        if not self.environment.has_function(node.name):
            raise CompilerException("Function '%s' is not defined" % node.name)

        func = self.environment.get_function(node.name)
        if not hasattr(func, 'node_type') or func.node_type != AST_NODE_FUNCTION_DECL:
            raise CompilerException("'%s' is not a function" % node.name)

        # Enter a new scope for function execution
        self.environment.enter_scope()

        # Evaluate arguments and bind to parameters
        if len(node.args) != len(func.params):
            self.environment.leave_scope()  # Clean up before raising exception
            raise CompilerException("Function '%s' expects %d arguments, got %d" % 
                                  (node.name, len(func.params), len(node.args)))

        for (param_name, param_type), arg in zip(func.params, node.args):
            arg_value = self.evaluate(arg)
            if not can_promote(arg.expr_type, param_type):
                self.environment.leave_scope()
                raise CompilerException("Type mismatch in argument to function '%s': cannot convert %s to %s" %
                                      (node.name, var_type_to_string(arg.expr_type), var_type_to_string(param_type)))
            self.environment.set(param_name, arg_value)

        result = None  # Default return value for void functions

        try:
            # Execute function body
            for stmt in func.body:
                self.evaluate(stmt)

            # If no return statement was encountered and function is not void,
            # we should raise an error
            if func.return_type != TYPE_VOID:
                self.environment.leave_scope()  # Clean up before raising exception
                raise CompilerException("Function '%s' has non-void return type but reached end of function without return" % node.name)

        except ReturnException as ret:
            # Check return value type against function's return type
            if func.return_type == TYPE_VOID and ret.value is not None:
                self.environment.leave_scope()  # Clean up before raising exception
                raise CompilerException("Void function '%s' returned a value" % node.name)

            result = ret.value

        # Leave function scope
        self.environment.leave_scope()
        return result

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

    def visit_method_def(self, node):
        """Visit a method definition node - Nothing to do at runtime"""
        # Method definitions are handled at parse time
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
            # Create a temporary scope for the constructor
            self.environment.enter_scope()
            self.environment.set("self", instance)

            # Evaluate and pass constructor arguments
            args = [self.evaluate(arg) for arg in node.args]
            self.check_and_set_params("Constructor for '%s'" % node.struct_name, init_method.params, args, node.args)

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

    def visit_method_call(self, node):
        """Visit a method call node (obj.method())"""
        # Evaluate the object
        obj = self.evaluate(node.obj)

        # Handle nil reference
        if obj is None:
            raise CompilerException("Attempt to call method '%s' on a nil reference" % node.method_name)

        # Make sure it's a struct instance
        if not isinstance(obj, StructInstance):
            raise CompilerException("Cannot call method '%s' on non-struct value" % node.method_name)

        # Get the method
        method = type_registry.get_method(obj.struct_name, node.method_name)
        if not method:
            raise CompilerException("Method '%s' not found in struct '%s'" % (node.method_name, obj.struct_name))

        # Create a new scope for method execution
        self.environment.enter_scope()

        # Set 'self' to the object instance
        self.environment.set("self", obj)

        # Evaluate and pass method arguments
        args = [self.evaluate(arg) for arg in node.args]
        self.check_and_set_params("Method '%s'" % node.method_name, method.params, args, node.args)

        # Execute method body
        result = None
        try:
            for stmt in method.body:
                self.evaluate(stmt)

            # If we reach here without a return and the method isn't void, it's an error
            if method.return_type != TYPE_VOID:
                self.environment.leave_scope()
                raise CompilerException("Method '%s' has non-void return type but reached end without returning" % 
                                      (node.method_name))

        except ReturnException as ret:
            # Check return value type against method's return type
            if method.return_type == TYPE_VOID and ret.value is not None:
                self.environment.leave_scope()
                raise CompilerException("Void method '%s' returned a value" % node.method_name)

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
        self.function_map = {}  # Map of function names to function nodes

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

    def register_function(self, name, func_node):
        """Register a function in the function map"""
        self.function_map[name] = func_node

    def has_function(self, name):
        """Check if a function exists in the function map"""
        return name in self.function_map

    def get_function(self, name):
        """Get a function from the function map"""
        return self.function_map[name]
