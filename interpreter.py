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
        # Initialize heap management via reset
        self.reset()

        # Create node type to visitor method mapping
        self.visitor_map = {
            AST_NODE_NUMBER: self.visit_number,
            AST_NODE_STRING: self.visit_string,
            AST_NODE_VARIABLE: self.visit_variable,
            AST_NODE_BINARY_OP: self.visit_binary_op,
            AST_NODE_UNARY_OP: self.visit_unary_op,
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
            AST_NODE_MEMBER_ACCESS: self.visit_member_access,
            AST_NODE_ARRAY_ACCESS: self.visit_array_access,
            AST_NODE_NEW: self.visit_new,
            AST_NODE_DEL: self.visit_del,
            AST_NODE_NIL: self.visit_nil,
            AST_NODE_GENERIC_INITIALIZER: self.visit_generic_initializer,
            AST_NODE_ARRAY_RESIZE: self.visit_array_resize,
        }

    def reset(self):
        registry.reset()
        if self.environment: self.environment.reset()
        # Reset heap tracking
        self.heap_objects = {} # obj_id -> (Variable, is_freed)
        # Reserve heap ID 0 for nil references - simplifies comparison logic
        # We use the integer value 0 for it so our C-style operators don't hiccup on None
        self.heap_objects[0] = (self.make_direct_value(0, TYPE_VOID), False)
        self.next_heap_id = 1
        # Store last evaluated token for error reporting (line number)
        self.last_token = None

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
                raise CompilerException("Invalid heap reference", self.last_token)
                
            obj, is_freed = self.heap_objects[heap_id]
            if is_freed and not skip_checks:
                raise CompilerException("Use after free", self.last_token)
                
            return obj
            
        if var.tag == TAG_STACK_REF:
            # Resolve stack reference
            var_name, scope_id = var.ref_data
            
            # Check if scope still exists
            scope_exists = scope_id <= self.environment.stackptr
            if not scope_exists:
                raise CompilerException("Reference to variable from destroyed scope", self.last_token)
            
            # Find the variable in the appropriate scope
            for i in range(scope_id, -1, -1):
                if var_name in self.environment.stack[i]:
                    return self.environment.stack[i][var_name]
                    
            raise CompilerException("Referenced variable '%s' not found" % var_name, self.last_token)
            
        # This should never happen if code is consistent
        raise CompilerException("Unknown reference tag", self.last_token)

    def create_default_value(self, type_id):
        """Create a default value Variable for a given type"""
        if type_id == TYPE_STRING:
            return self.make_direct_value("", TYPE_STRING)
        elif is_float_type(type_id):
            return self.make_direct_value(0.0, type_id)
        elif registry.is_struct_type(type_id):
            # Special case: dynamic arrays should default to nil
            if registry.is_array_type(type_id) and registry.get_array_size(type_id) == 0:
                return self.make_direct_value(None, type_id)  # Dynamic arrays default to nil
            # Create a properly initialized struct instance instead of nil
            struct_name = registry.get_struct_name(type_id)
            instance = StructInstance(type_id, struct_name)
            # Initialize its fields recursively
            all_fields = registry.get_struct_fields(type_id)
            for field_name, field_type in all_fields:
                instance.fields[field_name] = self.create_default_value(field_type)
            # Return the initialized struct
            return self.make_direct_value(instance, type_id)
        else:
            return self.make_direct_value(0, type_id)  # Default for other types

    def visit_nil(self, node):
        """Visit a nil literal node"""
        # We use a special heap-id of 0 to denote the nil status
        return self.make_heap_ref(None, node.expr_type, 0)

    def deep_copy(self, var):
        """Create a deep copy of a variable and its value"""
        # Dereference if it's a reference first
        var = self.dereference(var)

        # Handle strings (also immutable in Python)
        if var.expr_type == TYPE_STRING:
            return self.make_direct_value(var.value, TYPE_STRING)

        # Handle primitive types (they're immutable, so no need for deep copy)
        if registry.is_primitive_type(var.expr_type):
            return self.make_direct_value(var.value, var.expr_type)

        # Handle struct instances (including arrays)
        if isinstance(var.value, StructInstance):
            # Create a new instance with the same type
            new_instance = StructInstance(var.expr_type, var.value.struct_name)

            # Recursively copy all fields
            for field_name, field_value in var.value.fields.items():
                # Deep copy each field value
                new_instance.fields[field_name] = self.deep_copy(field_value)

            # Return as a direct value
            return self.make_direct_value(new_instance, var.expr_type)

        # Handle nil references
        if var.value is None:
            raise CompilerException("Attempt to access None", self.last_token)
            # return self.make_direct_value(None, var.expr_type)

        # If we get here, we encountered an unknown type
        raise CompilerException("Cannot deep copy value of type %s"%registry.var_type_to_string(var.expr_type), self.last_token)

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

        self.last_token = node.token

        # Dispatch to the appropriate visitor method
        if node.node_type in self.visitor_map:
            visitor = self.visitor_map[node.node_type]
            return visitor(node)

        raise CompilerException("No visit method defined for node type: %s" % ast_node_type_to_string(node.node_type), node.token)

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
                                   (method_name, len(method_params), len(args)), self.last_token)

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
                                        registry.var_type_to_string(param_type)), self.last_token)
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
            raise CompilerException("Variable '%s' is not defined" % node.name, self.last_token)
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
                raise CompilerException("Cannot assign to field of non-struct value", self.last_token)

            # Set the field value
            obj.fields[node.left.member_name] = right_var
            return right_var

        # Special case for assignment to array element (arr[idx] = value)
        elif node.operator == '=' and node.left.node_type == AST_NODE_ARRAY_ACCESS:
            # Use the existing array_access visitor to get the array element
            # This already handles all the array access logic consistently
            array_node = node.left

            # Evaluate the array
            arr_var = self.evaluate(array_node.array)
            # Dereference as needed
            arr_var = self.dereference(arr_var)

            arr = arr_var.value

            # Evaluate the index
            idx_var = self.evaluate(array_node.index)
            idx_var = self.dereference(idx_var)
            idx = idx_var.value

            # Make sure it's a struct instance (arrays are represented as structs)
            if not isinstance(arr, StructInstance):
                raise CompilerException("Cannot perform array indexing on non-array value", self.last_token)

            # Set the element value
            field_name = "_%d" % idx
            arr.fields[field_name] = right_var
            return right_var

        # Special case for assignment to function call result (func() = value)
        elif node.operator == '=' and node.left.node_type == AST_NODE_CALL:
            # Function calls that return references can be assigned to
            if left_var.tag not in (TAG_STACK_REF, TAG_HEAP_REF):
                raise CompilerException("Cannot assign to non-reference function result", self.last_token)

            # Assign through the reference
            self.assign_through_reference(left_var, right_var)
            return right_var

        # Handle variable assignment (variable = value)
        elif node.operator == '=' and node.left.node_type == AST_NODE_VARIABLE:
            # Extract variable name from the left node
            var_name = node.left.name
            value_var = right_var
            var_obj = None

            # Check if we're assigning to a variable passed by reference
            var_obj = self.environment.get(var_name)
            if var_obj is not None:
                # Special case 1: Variable is a nil reference (heap ID 0)
                # Special case 2: Right-side is a heap allocation operation (new or array resize)
                is_nil_ref = (var_obj.tag == TAG_HEAP_REF and var_obj.ref_data == 0)
                is_heap_alloc = (node.right.node_type == AST_NODE_NEW or 
                                node.right.node_type == AST_NODE_ARRAY_RESIZE)

                if (is_nil_ref or is_heap_alloc) and var_obj.tag == TAG_HEAP_REF:
                    # Direct assignment - update the reference itself
                    self.environment.set(var_name, value_var)
                    return value_var

                # If it's a reference, we need to assign through it
                if var_obj.tag in (TAG_STACK_REF, TAG_HEAP_REF):
                    return self.assign_through_reference(var_obj, value_var)

            # Check if type promotion is needed and allowed
            if node.expr_type != node.right.expr_type:
                if not can_promote(node.right.expr_type, node.expr_type):
                    raise CompilerException("Cannot assign %s to %s" %
                                          (var_type_to_string(node.right.expr_type), var_type_to_string(node.expr_type)), self.last_token)

            # Handle number literal promotion
            if node.right.node_type == AST_NODE_NUMBER:
                raw_value = value_var.value
                promoted = promote_literal_if_needed(raw_value, node.right.expr_type, node.expr_type)
                if promoted != raw_value:
                    value_var = self.make_direct_value(promoted, node.expr_type)

            self.environment.set(var_name, value_var)
            return value_var

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
            raise CompilerException("= Operator for binary operator used in unspecified context!", self.last_token)

        raise CompilerException("Unknown binary operator: %s" % node.operator, self.last_token)

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

        raise CompilerException("Unknown unary operator: %s" % node.operator, self.last_token)

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

            raise CompilerException("Reference target '%s' not found" % var_name, self.last_token)

        elif ref_var.tag == TAG_HEAP_REF:
            # Heap reference - update the heap object
            heap_id = ref_var.ref_data

            # Special case for nil reference (heap ID 0)
            if heap_id == 0:
                raise CompilerException("Cannot assign through a nil reference", self.last_token)

            if heap_id in self.heap_objects:
                _, is_freed = self.heap_objects[heap_id]
                if is_freed:
                    raise CompilerException("Use after free in assignment", self.last_token)
                self.heap_objects[heap_id] = (value_var, is_freed)
                return True
            raise CompilerException("Invalid heap reference in assignment", self.last_token)
        raise CompilerException("Cannot assign through non-reference value", self.last_token)

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

        if node.ref_kind == REF_KIND_NONE and value_var.tag != TAG_DIRECT_VALUE:
            # Dereference and make a deep copy to maintain proper value semantics
            value_var = self.deep_copy(value_var)

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
                raise CompilerException("Function with 'byref' return type must return a reference", self.last_token)

        elif value_var and value_var.tag != TAG_DIRECT_VALUE:
            # Non-reference return function - dereference
            value_var = self.dereference(value_var)

        # Throw a special exception to unwind the call stack
        raise ReturnException(value_var)

    def visit_compare(self, node):
        """Evaluate a comparison node"""
        left_var = self.evaluate(node.left)
        right_var = self.evaluate(node.right)

        # Special case: If both operands are heap references, compare their heap IDs
        if node.operator in ('==', '!=') and left_var.tag == TAG_HEAP_REF and right_var.tag == TAG_HEAP_REF:
            # Compare heap IDs (pointer equality)
            are_equal = (left_var.ref_data == right_var.ref_data)
            if node.operator == '==':
                return self.make_direct_value(1 if are_equal else 0, TYPE_INT)
            else:  # !=
                return self.make_direct_value(0 if are_equal else 1, TYPE_INT)

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
                raise CompilerException("Operator %s not supported for strings" % node.operator, self.last_token)
            else:
                # Unknown operator
                raise CompilerException("Unknown comparison operator: %s" % node.operator, self.last_token)

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

        raise CompilerException("Unknown comparison operator: %s" % node.operator, self.last_token)

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

        raise CompilerException("Unknown logical operator: %s" % node.operator, self.last_token)

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

        raise CompilerException("Unknown bitwise operator: %s" % node.operator, self.last_token)

    # Struct-related visitor methods
    def visit_struct_def(self, node):
        """Visit a struct definition node - Nothing to do at runtime"""
        # Struct definitions are handled at parse time
        return self.make_direct_value(None, TYPE_VOID)

    def visit_member_access(self, node):
        """Visit a member access node (obj.field)"""
        # Evaluate the object expression
        obj_var = self.evaluate(node.obj)

        # Dereference if it's a reference
        obj_var = self.dereference(obj_var)
        obj = obj_var.value  # Get the actual struct instance

        # Handle nil reference
        if obj is None:
            raise CompilerException("Attempt to access field of a nil reference", self.last_token)

        # Make sure it's a struct instance
        if not isinstance(obj, StructInstance):
            raise CompilerException("Cannot access member '%s' on non-struct value" % node.member_name, self.last_token)

        # Get the field value
        if node.member_name not in obj.fields:
            raise CompilerException("Field '%s' not found in struct '%s'" % (node.member_name, obj.struct_name), self.last_token)

        return obj.fields[node.member_name]

    def visit_array_access(self, node):
        """Evaluate an array access node (arr[idx])"""
        # Evaluate the array expression
        arr_var = self.evaluate(node.array)

        # Evaluate the index expression
        idx_var = self.evaluate(node.index)

        # Dereference if it's a reference
        arr_var = self.dereference(arr_var)
        idx_var = self.dereference(idx_var)

        # Get the raw values
        arr = arr_var.value
        idx = idx_var.value

        # Handle nil reference
        if arr is None:
            raise CompilerException("Attempt to access element of a nil reference", self.last_token)

        # Make sure it's a struct instance (arrays are represented as structs)
        if not isinstance(arr, StructInstance):
            raise CompilerException("xCannot perform array indexing on non-array value", self.last_token)

        # Convert index to field name format (arrays use _0, _1, etc. like tuples)
        field_name = "_%d" % idx

        # Get the element value
        if field_name not in arr.fields:
            raise CompilerException("Array index %d out of bounds" % idx, self.last_token)

        return arr.fields[field_name]

    def process_argument(self, arg_node, is_byref):
        """Process a function/method argument based on byref flag"""
        # First evaluate the expression regardless of byref status
        result = self.evaluate(arg_node)

        if not is_byref:
            # For regular parameters, create a deep copy for value semantics
            return self.deep_copy(result)

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

        # Special case for constructor calls
        is_constructor_call = (is_method_call and node.name == "init" and
                              node.obj.node_type == AST_NODE_VARIABLE and
                              node.obj.name == "__dunno__")

        if is_constructor_call:
            # This is a constructor call with our special marker
            struct_id = node.obj.expr_type
            struct_name = registry.get_struct_name(struct_id)

            # Create a new instance (similar to visit_struct_init)
            instance = StructInstance(struct_id, struct_name)

            # Initialize fields with default values
            all_fields = registry.get_struct_fields(struct_id)
            for field_name, field_type in all_fields:
                instance.fields[field_name] = self.create_default_value(field_type)

            # Create a direct value for the new instance
            obj = self.make_direct_value(instance, struct_id)
            obj_value = obj

            # Get the method ID from registry
            method_id = registry.lookup_function(node.name, struct_id)
            if method_id == -1:
                raise CompilerException("Method '%s' not found in struct '%s'" % ("init", struct_name), self.last_token)

        elif is_method_call:
            # Evaluate object and perform validation for method calls
            obj = self.evaluate(node.obj)
            obj_deref = self.dereference(obj)
            obj_value = obj_deref.value

            # Handle nil reference
            if obj is None:
                raise CompilerException("Attempt to call method '%s' on a nil reference" % node.name, self.last_token)

            # Make sure it's a struct instance
            if not isinstance(obj_value, StructInstance):
                raise CompilerException("Cannot call method '%s' on non-struct value" % node.name, self.last_token)

            # Get the method ID from registry
            method_id = registry.lookup_function(node.name, obj_value.struct_id)
            if method_id == -1:
                raise CompilerException("Method '%s' not found in struct '%s'" % (node.name, obj.struct_name), self.last_token)

        else:
            # Get function ID from registry
            method_id = registry.lookup_function(node.name)
            if method_id == -1:
                raise CompilerException("Function '%s' is not defined" % node.name, self.last_token)

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
                raise CompilerException("%s has non-void return type but reached end without returning" % context_name, self.last_token)

        except ReturnException as ret:
            # Check return value type against function's return type
            if func_obj.return_type == TYPE_VOID and ret.value is not None:
                self.environment.leave_scope()  # Clean up before raising exception
                raise CompilerException("Void %s returned a value" %
                                      ("method" if is_method_call else "function"), self.last_token)

            result = ret.value if ret.value else self.make_direct_value(None, TYPE_VOID)

            # Check if this is a reference-returning function
            if node.ref_kind & REF_KIND_GENERIC:
                # Ensure reference is valid - should already be checked in visit_return
                # constructor is special in that it returns byref or byval depending on whether wrapped in a NewNode
                if result.tag == TAG_DIRECT_VALUE and not is_constructor_call:
                    raise CompilerException("Function '%s' with byref return type must return a reference" % 
                                           (func_obj.name), self.last_token)

                # No need to create new reference, already properly validated
            elif result.tag != TAG_DIRECT_VALUE:
                # For non-ref function with reference result, dereference
                result = self.dereference(result)

        # Clean up scope
        self.environment.leave_scope()

        # For constructor calls, always return the newly created instance
        if is_constructor_call:
            # Check if this constructor call should return a reference (REF_KIND_HEAP)
            # or a value (REF_KIND_NONE) based on the node's ref_kind
            if node.ref_kind == REF_KIND_HEAP:
                # Constructor is within a NewNode, return as heap reference
                return obj
            else:
                # Constructor is standalone, return by value
                return self.dereference(obj)

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
            raise CompilerException("'del' can only be used with heap references (created with 'new')", self.last_token)
        
        # Get heap ID
        heap_id = obj_var.ref_data

        if heap_id not in self.heap_objects:
            raise CompilerException("Invalid heap reference or double free", self.last_token)

        instance_var, is_freed = self.heap_objects[heap_id]
        if is_freed:
            raise CompilerException("Double free detected", self.last_token)
            
        # Get the actual instance
        instance = instance_var.value

        # Handle nil reference
        if instance is None:
            raise CompilerException("Attempt to delete a nil reference", self.last_token)

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
            raise CompilerException("Unknown initializer type", self.last_token)

        # Create a struct instance
        instance = StructInstance(node.expr_type, struct_name)

        # Get struct fields for validation in LINEAR mode
        all_fields = []
        if node.subtype == INITIALIZER_SUBTYPE_LINEAR:
            # Handle different LINEAR initializers based on type
            if registry.is_array_type(node.expr_type):
                # For arrays, use numerical field names (_0, _1, etc.) - similar to tuples
                for i, elem in enumerate(node.elements):
                    field_name = "_%d" % i
                    value = self.evaluate(elem)
                    instance.fields[field_name] = value

                # If this is a fixed-size array, pre-initialize missing elements
                array_size = registry.get_array_size(node.expr_type)
                if array_size != 0 and array_size > len(node.elements):
                    element_type = registry.get_array_element_type(node.expr_type)
                    # Initialize remaining elements with default values
                    for i in range(len(node.elements), array_size):
                        field_name = "_%d" % i
                        instance.fields[field_name] = self.create_default_value(element_type)
            else:
                # For structs, get field definitions
                all_fields = registry.get_struct_fields(node.expr_type)

        if node.subtype == INITIALIZER_SUBTYPE_LINEAR:
            # For linear initializers, assign values to fields in order
            for i, (field_name, field_type) in enumerate(all_fields):
                if i < len(node.elements):
                    value_var = self.evaluate(node.elements[i])
                    instance.fields[field_name] = value_var

        elif node.subtype == INITIALIZER_SUBTYPE_NAMED:
            # Reserved for future C99-style named initializers
            raise CompilerException("Named initializers not yet implemented", self.last_token)

        return self.make_direct_value(instance, node.expr_type)

    def visit_array_resize(self, node):
        """Visit an array resize operation: new(array, size)"""
        # Evaluate the array expression
        array_var = self.evaluate(node.array_expr)

        # Evaluate the size expression
        size_var = self.evaluate(node.size_expr)
        size_var = self.dereference(size_var)
        new_size = size_var.value

        # Validate size
        if not isinstance(new_size, int) or new_size < 0:
            raise CompilerException("Array size must be a non-negative integer", self.last_token)

        # Get element type from the array type
        element_type = registry.get_array_element_type(node.expr_type)

        # Create a new array instance
        new_array = StructInstance(node.expr_type, registry.get_struct_name(node.expr_type))

        # Copy existing elements from old array if not nil
        if array_var.tag == TAG_HEAP_REF and array_var.ref_data != 0:  # Not nil
            old_heap_id = array_var.ref_data
            old_array = self.dereference(array_var).value

            # Copy elements up to the minimum of old and new size
            for i in range(min(len(old_array.fields), new_size)):
                field_name = "_%d" % i
                if field_name in old_array.fields:
                    new_array.fields[field_name] = old_array.fields[field_name]

            # Mark old array as freed after copying (following realloc semantics)
            if old_heap_id in self.heap_objects:
                _, was_freed = self.heap_objects[old_heap_id]
                if not was_freed:
                    self.heap_objects[old_heap_id] = (self.heap_objects[old_heap_id][0], True)

        # Initialize any new elements to default values
        for i in range(len(new_array.fields), new_size):
            field_name = "_%d" % i
            new_array.fields[field_name] = self.create_default_value(element_type)

        # Allocate on heap
        heap_id = self.next_heap_id
        self.next_heap_id += 1

        # Store new array on heap
        new_array_var = self.make_direct_value(new_array, node.expr_type)
        self.heap_objects[heap_id] = (new_array_var, False)  # Not freed

        # Return heap reference to new array
        return self.make_heap_ref(new_array, node.expr_type, heap_id)


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
