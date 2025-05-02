# Implementation of interpreter for the AST with C-style type handling

from __future__ import division  # Use true division
from shared import *
from cops import add, subtract, multiply, divide, modulo, shift_left, shift_right, negate, logical_not, bitwise_not, compare_eq, compare_ne, compare_lt, compare_le, compare_gt, compare_ge, logical_and, logical_or, bitwise_and, bitwise_or, bitwise_xor, truncate_to_unsigned
from type_registry import get_registry
from scope import EnvironmentStack

# import registry singleton
registry = get_registry()

# Reference type tags - explicitly defined for clarity
TAG_DIRECT_VALUE = 0  # Regular value (int, string, struct)
TAG_HEAP_REF = 1      # Heap reference (created with new)
TAG_STACK_REF = 2     # Stack reference (parameters passed byref)

# Simple container for all runtime values with reference tracking
class Variable(object):
    """Container for all runtime values with reference tracking"""
    def __init__(self, value, expr_type, tag=TAG_DIRECT_VALUE, ref_data=None):
        self.value = value        # The actual value or target
        self.expr_type = expr_type # Type ID
        self.tag = tag            # Reference type tag
        self.ref_data = ref_data  # For heap: heap_id, For stack: (var_name, scope_id)

# Struct instance class for runtime
class StructInstance:
    """Object to store a struct's fields at runtime"""

    def __init__(self, struct_id, struct_name):
        self.struct_id = struct_id
        self.struct_name = struct_name
        self.fields = {}  # field_name -> Variable instance

    def __repr__(self):
        #if self.fields is None: self.fields = {}
        fields_str = ", ".join(["%s=%s" % (name, value.value if value else "None") for name, value in self.fields.items()])
        return "%s(%s)" % (self.struct_name, fields_str)

class Interpreter(object):
    def __init__(self, environment=None):
        """Initialize the interpreter with an optional environment"""
        self.environment = environment if environment else EnvironmentStack()
        # Initialize heap management
        self.heap_objects = {}  # obj_id -> (Variable, is_freed)
        self.next_heap_id = 1

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
            AST_NODE_GENERIC_INITIALIZER: self.visit_generic_initializer,
        }

    def reset(self):
        registry.reset()
        if self.environment: self.environment.reset()
        # Reset heap tracking
        self.heap_objects = {}
        self.next_heap_id = 1

    def make_direct_value(self, value, expr_type):
        """Create a direct value variable"""
        return Variable(value, expr_type, TAG_DIRECT_VALUE)
        
    def make_heap_ref(self, target, expr_type, heap_id):
        """Create a heap reference variable"""
        return Variable(target, expr_type, TAG_HEAP_REF, heap_id)
        
    def make_stack_ref(self, var_name, scope_id, expr_type):
        """Create a stack reference variable"""
        return Variable(None, expr_type, TAG_STACK_REF, (var_name, scope_id))
        
    def dereference(self, var, skip_checks=False):
        """Get actual value from a variable, following references if needed"""
        # Direct values are returned as-is
        if var.tag == TAG_DIRECT_VALUE:
            return var
            
        if var.tag == TAG_HEAP_REF:
            # Check heap reference validity
            heap_id = var.ref_data
            if heap_id not in self.heap_objects:
                raise CompilerException("Invalid heap reference")
                
            obj, is_freed = self.heap_objects[heap_id]
            if is_freed and not skip_checks:
                raise CompilerException("Use after free")
                
            return obj
            
        if var.tag == TAG_STACK_REF:
            # Resolve stack reference
            var_name, scope_id = var.ref_data
            
            # Check if scope still exists
            scope_exists = scope_id <= self.environment.stackptr
            if not scope_exists:
                raise CompilerException("Reference to variable from destroyed scope")
            
            # Find the variable in the appropriate scope
            for i in range(scope_id, -1, -1):
                if var_name in self.environment.stack[i]:
                    return self.environment.stack[i][var_name]
                    
            raise CompilerException("Referenced variable '%s' not found" % var_name)
            
        # This should never happen if code is consistent
        raise CompilerException("Unknown reference tag")
    
    def get_raw_value(self, var):
        """Get the underlying raw value from a Variable, following references"""
        deref_var = self.dereference(var)
        return deref_var.value
    
    def extract_raw_values_from_env(self, env):
        """Extract raw values from an environment containing Variables"""
        raw_env = {}
        for var_name, var_obj in env.items():
            # All objects in the environment should be Variables
            deref_var = self.dereference(var_obj, skip_checks=True)
            raw_env[var_name] = deref_var.value
        return raw_env
        
    def prepare_result_environments(self):
        """Prepare result environments with raw values for testing"""
        raw_global_env = self.extract_raw_values_from_env(self.environment.stack[0])
        raw_main_env = self.extract_raw_values_from_env(self.environment.stack[1])
        return raw_global_env, raw_main_env
    
    def run(self, text):
        from lexer import Lexer
        from compiler import Parser

        lexer = Lexer(text)
        parser = Parser(lexer)
        ast = None
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

            main_func_id = registry.lookup_function("main")

            if main_func_id == -1:
                return {'success': False, 'error': "No 'main' function defined", 'ast': ast}

            main_func = registry.get_func_from_id(main_func_id).ast_node

            # Make sure main has no parameters
            if len(main_func.params) > 0:
                return {'success': False, 'error': "Function 'main' cannot have parameters", 'ast': ast}

            # Create a new scope for main function
            interpreter.environment.enter_scope()

            try:
                # Execute main function body
                for stmt in main_func.body:
                    interpreter.evaluate(stmt)
                    
                # Convert environment to raw values for testing
                raw_global_env, raw_main_env = self.prepare_result_environments()

                # Return both global and main's environment
                return {
                    'success': True,
                    'global_env': raw_global_env,
                    'main_env': raw_main_env,
                    'ast': ast
                }
            except ReturnException as ret:
                # If main returns a value, include it in the result
                
                # Convert environment to raw values for testing
                raw_global_env, raw_main_env = self.prepare_result_environments()
                
                return {
                    'success': True,
                    'result': ret.value.value if ret.value else None,  # Extract raw value for result
                    'global_env': raw_global_env,
                    'main_env': raw_main_env,
                    'ast': ast
                }
        except CompilerException as e:
            return {'success': False, 'error': str(e), 'ast': ast}

    def evaluate(self, node):
        """Main entry point to evaluate an AST node"""
        if node is None:
            return self.make_direct_value(None, TYPE_VOID)

        # If given a list of nodes (program), execute each one
        if isinstance(node, list):
            result = self.make_direct_value(None, TYPE_VOID)
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
            args: List of evaluated argument values (Variable instances)
            arg_nodes: List of argument AST nodes for type information
        """
        if len(args) != len(method_params):
            self.environment.leave_scope()
            raise CompilerException("%s expects %d arguments, got %d" %
                                   (method_name, len(method_params), len(args)))

        for i in range(len(method_params)):
            param_name, param_type, is_byref = method_params[i]
            arg_value = args[i]
            # Determine if we need to type check this parameter
            skip_type_check = is_byref  # Skip type check for byref params, already verified

            # Make sure arg_nodes has an entry for this parameter
            has_node_for_type_check = i < len(arg_nodes)

            # Only perform type checking if needed and possible
            if not skip_type_check and has_node_for_type_check and not can_promote(arg_nodes[i].expr_type, param_type):
                self.environment.leave_scope()
                raise CompilerException("Type mismatch in %s argument: cannot convert %s to %s" %
                                       (method_name, registry.var_type_to_string(arg_nodes[i].expr_type),
                                        registry.var_type_to_string(param_type)))
            self.environment.set(param_name, arg_value)

    def visit_number(self, node):
        """Evaluate a number node (integer or float)"""
        return self.make_direct_value(node.value, node.expr_type)

    def visit_string(self, node):
        """Evaluate a string node"""
        return self.make_direct_value(node.value, TYPE_STRING)

    def visit_variable(self, node):
        """Evaluate a variable reference node"""
        var = self.environment.get(node.name)
        if var is None:
            raise CompilerException("Variable '%s' is not defined" % node.name)
        return var

    def visit_binary_op(self, node):
        """Evaluate a binary operation node"""
        left_var = self.evaluate(node.left)
        right_var = self.evaluate(node.right)

        # Special case for assignment to struct field (obj.field = value)
        if node.operator == '=' and node.left.node_type == AST_NODE_MEMBER_ACCESS:
            # First evaluate the object to get the struct instance
            obj_var = self.evaluate(node.left.obj)
            
            # Dereference if it's a reference
            obj_var = self.dereference(obj_var)
            obj = obj_var.value  # Get the struct instance

            # Check if obj is a struct instance
            if not isinstance(obj, StructInstance):
                raise CompilerException("Cannot assign to field of non-struct value")

            # Set the field value
            obj.fields[node.left.member_name] = right_var
            return right_var

        # Special case for assignment to function call result (func() = value)
        elif node.operator == '=' and node.left.node_type == AST_NODE_CALL:
            # Function calls that return references can be assigned to
            if left_var.tag not in (TAG_STACK_REF, TAG_HEAP_REF):
                raise CompilerException("Cannot assign to non-reference function result")

            # Assign through the reference
            self.assign_through_reference(left_var, right_var)
            return right_var

        # Dereference operands if they are references
        left_var = self.dereference(left_var)
        right_var = self.dereference(right_var)

        # Get raw values for operation
        left_val = left_var.value
        right_val = right_var.value

        if node.operator == '+':
            # Handle string concatenation
            if node.left.expr_type == TYPE_STRING and node.right.expr_type == TYPE_STRING:
                result = left_val + right_val
                return self.make_direct_value(result, TYPE_STRING)

            result = add(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, node.expr_type)
        elif node.operator == '-':
            result = subtract(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, node.expr_type)
        elif node.operator == '*':
            result = multiply(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, node.expr_type)
        elif node.operator == '/':
            result = divide(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, node.expr_type)
        elif node.operator == '%':
            result = modulo(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, node.expr_type)
        elif node.operator == 'shl':
            result = shift_left(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, node.expr_type)
        elif node.operator == 'shr':
            result = shift_right(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, node.expr_type)
        elif node.operator == '=':
            raise CompilerException("= Operator for binary operator used in unspecified context!")

        raise CompilerException("Unknown binary operator: %s" % node.operator)

    def visit_unary_op(self, node):
        """Evaluate a unary operation node"""
        value_var = self.evaluate(node.operand)
        
        # Dereference if it's a reference
        value_var = self.dereference(value_var)
        value = value_var.value

        if node.operator == '-':
            result = negate(value, node.operand.expr_type)
            return self.make_direct_value(result, node.expr_type)
        elif node.operator == '!':
            result = logical_not(value)
            return self.make_direct_value(result, TYPE_INT)
        elif node.operator == 'bitnot':
            result = bitwise_not(value, node.operand.expr_type)
            return self.make_direct_value(result, node.expr_type)

        raise CompilerException("Unknown unary operator: %s" % node.operator)

    def assign_through_reference(self, ref_var, value_var):
        """Assign a value through a reference"""
        if ref_var.tag == TAG_STACK_REF:
            # Stack reference - update the original variable
            var_name, scope_id = ref_var.ref_data

            # Find and update in appropriate scope
            for i in range(scope_id, -1, -1):
                if var_name in self.environment.stack[i]:
                    self.environment.stack[i][var_name] = value_var
                    return True

            raise CompilerException("Reference target '%s' not found" % var_name)

        elif ref_var.tag == TAG_HEAP_REF:
            # Heap reference - update the heap object
            heap_id = ref_var.ref_data
            if heap_id in self.heap_objects:
                _, is_freed = self.heap_objects[heap_id]
                if is_freed:
                    raise CompilerException("Use after free in assignment")
                self.heap_objects[heap_id] = (value_var, is_freed)
                return True
            raise CompilerException("Invalid heap reference in assignment")
        raise CompilerException("Cannot assign through non-reference value")

    def visit_assign(self, node):
        """Evaluate an assignment node"""
        value_var = self.evaluate(node.expr)
        var_obj = None

        # Check if we're assigning to a variable passed by reference
        var_obj = self.environment.get(node.var_name)
        if var_obj is not None:

            # If it's a reference, we need to assign through it
            if var_obj.tag in (TAG_STACK_REF, TAG_HEAP_REF):
                return self.assign_through_reference(var_obj, value_var)

        # Check if type promotion is needed and allowed
        if node.expr_type != node.expr.expr_type:
            if not can_promote(node.expr.expr_type, node.expr_type):
                raise CompilerException("Cannot assign %s to %s" % 
                                      (var_type_to_string(node.expr.expr_type), var_type_to_string(node.expr_type)))

        # Handle number literal promotion
        if node.expr.node_type == AST_NODE_NUMBER:
            raw_value = value_var.value
            promoted = promote_literal_if_needed(raw_value, node.expr.expr_type, node.expr_type)
            if promoted != raw_value:
                value_var = self.make_direct_value(promoted, node.expr_type)

        self.environment.set(node.var_name, value_var)
        return value_var

    def visit_compound_assign(self, node):
        """Evaluate a compound assignment node (+=, -=, etc.)"""
        var_obj = self.environment.get(node.var_name)
        current_var = self.dereference(var_obj)
        expr_var = self.evaluate(node.expr)
        expr_var = self.dereference(expr_var)

        # Get raw values
        current_value = current_var.value
        expr_value = expr_var.value

        if node.op_type == TT_PLUS_ASSIGN:
            # Handle string concatenation for += operator
            if node.expr_type == TYPE_STRING:
                if node.expr.expr_type != TYPE_STRING:
                    raise CompilerException("Cannot use += with string and %s" % var_type_to_string(node.expr.expr_type))
                result = current_value + expr_value
                result_var = self.make_direct_value(result, TYPE_STRING)
            else:
                result = add(current_value, expr_value, node.expr_type, node.expr.expr_type)
                result_var = self.make_direct_value(result, node.expr_type)
        elif node.op_type == TT_MINUS_ASSIGN:
            result = subtract(current_value, expr_value, node.expr_type, node.expr.expr_type)
            result_var = self.make_direct_value(result, node.expr_type)
        elif node.op_type == TT_MULT_ASSIGN:
            result = multiply(current_value, expr_value, node.expr_type, node.expr.expr_type)
            result_var = self.make_direct_value(result, node.expr_type)
        elif node.op_type == TT_DIV_ASSIGN:
            result = divide(current_value, expr_value, node.expr_type, node.expr.expr_type)
            result_var = self.make_direct_value(result, node.expr_type)
        elif node.op_type == TT_MOD_ASSIGN:
            result = modulo(current_value, expr_value, node.expr_type, node.expr.expr_type)
            result_var = self.make_direct_value(result, node.expr_type)
        else:
            raise CompilerException("Unknown compound assignment operator: %s" % token_name(node.op_type))

        # If it's a reference, we need to assign through it
        if var_obj.tag in (TAG_STACK_REF, TAG_HEAP_REF):
            self.assign_through_reference(var_obj, result_var)
        else:
             self.environment.set(node.var_name, result_var)
        return result_var

    def visit_print(self, node):
        """Evaluate a print statement node"""
        value_var = self.evaluate(node.expr)
        
        # Dereference if it's a reference
        value_var = self.dereference(value_var)
        
        # Print the raw value
        print(value_var.value)
        return value_var

    def visit_if(self, node):
        """Evaluate an if statement node"""
        condition_var = self.evaluate(node.condition)
        
        # Dereference if it's a reference
        condition_var = self.dereference(condition_var)
        condition = condition_var.value
        
        if condition:
            for stmt in node.then_body:
                self.evaluate(stmt)
        elif node.else_body:
            for stmt in node.else_body:
                self.evaluate(stmt)
        return self.make_direct_value(0, TYPE_INT)

    def visit_while(self, node):
        """Evaluate a while loop node"""
        while True:
            condition_var = self.evaluate(node.condition)
            
            # Dereference if it's a reference
            condition_var = self.dereference(condition_var)
            condition = condition_var.value
            
            if not condition:
                break
                
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
        return self.make_direct_value(0, TYPE_INT)

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
        value_var = self.evaluate(node.expr)

        # Check reference kind consistency
        if value_var.tag != TAG_DIRECT_VALUE and node.ref_kind != REF_KIND_NONE:
            # Reference compatibility is fine - references can be assigned to reference variables
            pass

        # types are already checked in compiler
        self.environment.set(node.var_name, value_var)
        return value_var

    def visit_function_decl(self, node):
        """Evaluate a function declaration node"""
        # Functions are already registered in the type registry during parsing
        return self.make_direct_value(0, TYPE_INT)

    def visit_return(self, node):
        """Evaluate a return statement node"""
        value_var = None if node.expr is None else self.evaluate(node.expr)
        # Check if function returns by reference
        if node.ref_kind & REF_KIND_GENERIC:
            # Special case: If returning a variable (not an expression) from a byref function,
            # automatically create a reference to that variable
            if value_var and value_var.tag == TAG_DIRECT_VALUE:
                # Only do this for simple variable expressions
                if node.expr and node.expr.node_type == AST_NODE_VARIABLE:
                    var_name = node.expr.name

                    # Find which scope contains this variable
                    for i in range(self.environment.stackptr, -1, -1):
                        if var_name in self.environment.stack[i]:
                            # Create a reference to the variable in its correct scope
                            value_var = self.make_stack_ref(var_name, i, node.expr.expr_type)
                            break

            # Validate that we're actually returning a reference
            if value_var is None or value_var.tag == TAG_DIRECT_VALUE:
                raise CompilerException("Function with 'byref' return type must return a reference")

        elif value_var and value_var.tag != TAG_DIRECT_VALUE:
            # Non-reference return function - dereference
            value_var = self.dereference(value_var)

        # Throw a special exception to unwind the call stack
        raise ReturnException(value_var)

    def visit_compare(self, node):
        """Evaluate a comparison node"""
        left_var = self.evaluate(node.left)
        right_var = self.evaluate(node.right)

        # Dereference operands if they are references
        left_var = self.dereference(left_var)
        right_var = self.dereference(right_var)

        # Get raw values
        left_val = left_var.value
        right_val = right_var.value

        # Handle string comparison operations
        if node.left.expr_type == TYPE_STRING and node.right.expr_type == TYPE_STRING:
            if node.operator == '==':
                result = 1 if left_val == right_val else 0
                return self.make_direct_value(result, TYPE_INT)
            elif node.operator == '!=':
                result = 1 if left_val != right_val else 0
                return self.make_direct_value(result, TYPE_INT)
            # Other comparison operators are not supported for strings
            elif node.operator in ['>', '>=', '<', '<=']:
                raise CompilerException("Operator %s not supported for strings" % node.operator)
            else:
                # Unknown operator
                raise CompilerException("Unknown comparison operator: %s" % node.operator)

        if node.operator == '==':
            result = compare_eq(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, TYPE_INT)
        elif node.operator == '!=':
            result = compare_ne(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, TYPE_INT)
        elif node.operator == '>=':
            result = compare_ge(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, TYPE_INT)
        elif node.operator == '>':
            result = compare_gt(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, TYPE_INT)
        elif node.operator == '<':
            result = compare_lt(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, TYPE_INT)
        elif node.operator == '<=':
            result = compare_le(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, TYPE_INT)

        raise CompilerException("Unknown comparison operator: %s" % node.operator)

    def visit_logical(self, node):
        """Evaluate a logical operation node"""
        left_var = self.evaluate(node.left)
        
        # Dereference if it's a reference
        left_var = self.dereference(left_var)
        left_val = left_var.value

        if node.operator == 'and':
            # Short circuit evaluation
            if not left_val:
                return self.make_direct_value(0, TYPE_INT)
                
            right_var = self.evaluate(node.right)
            right_var = self.dereference(right_var)
            right_val = right_var.value
            
            result = logical_and(left_val, right_val)
            return self.make_direct_value(result, TYPE_INT)
        elif node.operator == 'or':
            # Short circuit evaluation
            if left_val:
                return self.make_direct_value(1, TYPE_INT)
                
            right_var = self.evaluate(node.right)
            right_var = self.dereference(right_var)
            right_val = right_var.value
            
            result = logical_or(left_val, right_val)
            return self.make_direct_value(result, TYPE_INT)

        raise CompilerException("Unknown logical operator: %s" % node.operator)

    def visit_bitop(self, node):
        """Evaluate a bitwise operation node"""
        left_var = self.evaluate(node.left)
        right_var = self.evaluate(node.right)
        
        # Dereference operands if they are references
        left_var = self.dereference(left_var)
        right_var = self.dereference(right_var)
        
        # Get raw values
        left_val = left_var.value
        right_val = right_var.value

        if node.operator == '&':
            result = bitwise_and(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, node.expr_type)
        elif node.operator == '|':
            result = bitwise_or(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, node.expr_type)
        elif node.operator == 'xor':
            result = bitwise_xor(left_val, right_val, node.left.expr_type, node.right.expr_type)
            return self.make_direct_value(result, node.expr_type)

        raise CompilerException("Unknown bitwise operator: %s" % node.operator)

    # Struct-related visitor methods
    def visit_struct_def(self, node):
        """Visit a struct definition node - Nothing to do at runtime"""
        # Struct definitions are handled at parse time
        return self.make_direct_value(None, TYPE_VOID)

    def visit_struct_init(self, node):
        """Visit a struct initialization node"""
        # Create a new struct instance
        instance = StructInstance(node.struct_id, node.struct_name)

        # Initialize fields with default values first
        all_fields = registry.get_all_fields(node.struct_name)
        for field_name, field_type in all_fields:
            # Set default value based on type
            if registry.is_struct_type(field_type):
                # For now, we don't auto-initialize nested structs
                instance.fields[field_name] = self.make_direct_value(None, field_type)
            elif field_type == TYPE_STRING:
                instance.fields[field_name] = self.make_direct_value("", TYPE_STRING)
            elif is_float_type(field_type):
                instance.fields[field_name] = self.make_direct_value(0.0, field_type)
            else:
                instance.fields[field_name] = self.make_direct_value(0, field_type)

        # Call constructor if it exists and there are args or it's "init"
        init_method = registry.get_method(node.struct_id, "init")
        if init_method:
            # Process arguments with proper byref handling
            # THIS NEEDS TO BE DONE BEFORE ENTER_SCOPE ELSE WE MAY SHADOW VARIABLES
            args = []

            for i in range(len(init_method.params)):
                if i == 0:  # First parameter is always 'self'
                    arg_value = self.make_direct_value(instance, node.struct_id)
                else:
                    _, _, is_byref = init_method.params[i]
                    arg_value = self.process_argument(node.args[i], is_byref)
                args.append(arg_value)

            # Create a temporary scope for the constructor
            self.environment.enter_scope()

            self.check_and_set_params("Constructor for '%s'" % node.struct_name, init_method.params, args, node.args)

            # Execute constructor body
            try:
                for stmt in init_method.body:
                    self.evaluate(stmt)
            except ReturnException:
                # Ignore return value from constructor
                pass

            self.environment.leave_scope()

        # Return instance wrapped in a Variable
        return self.make_direct_value(instance, node.struct_id)

    def visit_member_access(self, node):
        """Visit a member access node (obj.field)"""
        # Evaluate the object expression
        obj_var = self.evaluate(node.obj)
        
        # Dereference if it's a reference
        obj_var = self.dereference(obj_var)
        obj = obj_var.value  # Get the actual struct instance

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

    def process_argument(self, arg_node, is_byref):
        """Process a function/method argument based on byref flag"""
        # First evaluate the expression regardless of byref status
        result = self.evaluate(arg_node)

        if not is_byref:
            # For regular parameters, just return the evaluated result
            return result

        # For byref parameters, we need an lvalue or a reference
        if result.tag in (TAG_HEAP_REF, TAG_STACK_REF):
            # Already a reference, can pass directly
            return result

        # Direct value from a variable - create a reference
        if arg_node.node_type == AST_NODE_VARIABLE:
            var_name = arg_node.name
            scope_id = self.environment.stackptr
            return self.make_stack_ref(var_name, scope_id, arg_node.expr_type)

        # Create a temporary variable for the expression result
        # This allows passing expression results to byref parameters
        temp = self.make_direct_value(result.value, result.expr_type)

        # Store this temporary in the current function's scope with a unique name
        # We don't need to track this name as it's just for the duration of the call
        # WARNING: Returning this temporary reference from the function would be unsafe
        # hack hack - we introduce a "static int" into the class
        # Add a temp counter at class level if not present
        if not hasattr(self, 'temp_counter'): self.temp_counter = 0
        temp_name = "__temp_%d"%self.temp_counter
        self.temp_counter += 1
        self.environment.set(temp_name, temp)

        # Create and return a reference to this temporary
        return self.make_stack_ref(temp_name, self.environment.stackptr, result.expr_type)

    def visit_call(self, node):
        """Unified handler for function and method calls using only the type registry"""
        # Determine if this is a method call (has an object context)
        is_method_call = node.obj is not None
        context_name = "Method '%s'" % node.name if is_method_call else "Function '%s'" % node.name

        if is_method_call:
            # Evaluate object and perform validation for method calls
            obj = self.evaluate(node.obj)
            obj_deref = self.dereference(obj)
            obj_value = obj_deref.value

            # Handle nil reference
            if obj is None:
                raise CompilerException("Attempt to call method '%s' on a nil reference" % node.name)

            # Make sure it's a struct instance
            if not isinstance(obj_value, StructInstance):
                raise CompilerException("Cannot call method '%s' on non-struct value" % node.name)

            # Get the method ID from registry
            method_id = registry.lookup_function(node.name, obj_value.struct_id)
            if method_id == -1:
                raise CompilerException("Method '%s' not found in struct '%s'" % (node.name, obj.struct_name))

        else:
            # Get function ID from registry
            method_id = registry.lookup_function(node.name)
            if method_id == -1:
                raise CompilerException("Function '%s' is not defined" % node.name)

        # Get function details from registry
        func_obj = registry.get_func_from_id(method_id)

        # Process all parameters (including self for methods)
        # THIS NEEDS TO BE DONE BEFORE ENTER_SCOPE ELSE WE MAY SHADOW VARIABLES

        args = []
        arg_nodes = []  # For type checking

        for i in range(len(func_obj.params)):
            _, _, is_byref = func_obj.params[i]

            if i == 0 and is_method_call:
                # First parameter for methods is 'self'
                arg_value = obj  # Already evaluated above
                # Make sure we use the dereferenced object for 'self'
                #arg_value = self.make_direct_value(obj_value, obj_deref.expr_type)
                arg_nodes.append(node.obj)
            else:
                arg_value = self.process_argument(node.args[i], is_byref)
                arg_nodes.append(node.args[i])

            args.append(arg_value)

        # Enter function scope
        self.environment.enter_scope()

        self.check_and_set_params(context_name, func_obj.params, args, arg_nodes)
        # Default return value for void functions
        result = self.make_direct_value(None, TYPE_VOID)

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

            result = ret.value if ret.value else self.make_direct_value(None, TYPE_VOID)

            # Check if this is a reference-returning function
            if node.ref_kind & REF_KIND_GENERIC:
                # Ensure reference is valid - should already be checked in visit_return
                if result.tag == TAG_DIRECT_VALUE:
                    raise CompilerException("Function '%s' with byref return type must return a reference" % 
                                           (func_obj.name))

                # No need to create new reference, already properly validated
            elif result.tag != TAG_DIRECT_VALUE:
                # For non-ref function with reference result, dereference
                result = self.dereference(result)

        # Clean up scope
        self.environment.leave_scope()
        return result

    def visit_new(self, node):
        """Visit a new expression for heap allocation"""
        # Evaluate the struct initialization
        instance_var = self.evaluate(node.struct_init)
        instance = instance_var.value
        
        # Allocate on heap
        heap_id = self.next_heap_id
        self.next_heap_id += 1
        self.heap_objects[heap_id] = (instance_var, False)  # Not freed
        
        # Return a heap reference
        return self.make_heap_ref(instance, node.expr_type, heap_id)

    def visit_del(self, node):
        """Visit a del statement for heap deallocation"""
        # Evaluate the expression
        obj_var = self.evaluate(node.expr)
        
        # Check if it's a heap reference
        if obj_var.tag != TAG_HEAP_REF:
            raise CompilerException("'del' can only be used with heap references (created with 'new')")
        
        # Get heap ID
        heap_id = obj_var.ref_data

        if heap_id not in self.heap_objects:
            raise CompilerException("Invalid heap reference or double free")

        instance_var, is_freed = self.heap_objects[heap_id]
        if is_freed:
            raise CompilerException("Double free detected")
            
        # Get the actual instance
        instance = instance_var.value

        # Handle nil reference
        if instance is None:
            raise CompilerException("Attempt to delete a nil reference")

        # Call destructor if it exists
        if isinstance(instance, StructInstance):
            fini_method = registry.get_method(instance.struct_id, "fini")
            if fini_method:
                # Create a temporary scope for the destructor
                self.environment.enter_scope()
                self.environment.set("self", instance_var)

                # Execute destructor body
                try:
                    for stmt in fini_method.body:
                        self.evaluate(stmt)
                except ReturnException:
                    # Ignore return from destructor
                    pass

                self.environment.leave_scope()

        # Mark as freed (destructor must run first)
        self.heap_objects[heap_id] = (instance_var, True)

        return self.make_direct_value(None, TYPE_VOID)

    def visit_generic_initializer(self, node):
        """Visit a generic initializer node handling tuples, structs, and arrays"""
        # Get struct name from type registry
        struct_name = registry.get_struct_name(node.expr_type)
        if not struct_name:
            raise CompilerException("Unknown initializer type")

        # Create a struct instance
        instance = StructInstance(node.expr_type, struct_name)

        # Get struct fields for validation in LINEAR mode
        all_fields = []
        if node.subtype == INITIALIZER_SUBTYPE_LINEAR:
            all_fields = registry.get_all_fields(struct_name)

        # Initialize fields based on initializer subtype
        if node.subtype == INITIALIZER_SUBTYPE_TUPLE:
            # For tuples, use numerical field names (_0, _1, etc.)
            for i, elem in enumerate(node.elements):
                field_name = "_%d" % i
                value_var = self.evaluate(elem)
                instance.fields[field_name] = value_var

        elif node.subtype == INITIALIZER_SUBTYPE_LINEAR:
            # For linear initializers, assign values to fields in order
            for i, (field_name, field_type) in enumerate(all_fields):
                if i < len(node.elements):
                    value_var = self.evaluate(node.elements[i])
                    instance.fields[field_name] = value_var

        elif node.subtype == INITIALIZER_SUBTYPE_NAMED:
            # Reserved for future C99-style named initializers
            raise CompilerException("Named initializers not yet implemented")

        return self.make_direct_value(instance, node.expr_type)

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
