# AST Expression Flattener
# Transforms complex expressions into simpler ones with temporaries

from shared import *
from compiler import VariableNode, BinaryOpNode, UnaryOpNode, IfNode, WhileNode, BreakNode, ContinueNode, ExprStmtNode, VarDeclNode, CallNode
import copy

class AstExpressionFlattener:
    """
    Transforms AST for C code generation:
    1. Flattens complex expressions by introducing temporary variables
    2. Prepares struct handling for C backend by setting appropriate reference kinds
    """

    def __init__(self, registry):
        """Initialize with registry for type information"""
        self.registry = registry
        self.current_function = None
        self._reset_temps()

    def _reset_temps(self):
        """Reset temporary variable counter when processing a new function"""
        self.temp_counter = 0

    def inject_ref_kind(self, node, make_copy=True):
        """
        Inject appropriate reference kind for arrays and non-byval struct instances.
        Returns modified node or original if no change needed.
        """
        if node.node_type == AST_NODE_VAR_DECL:
            type_id = node.var_type
        else:
            type_id = node.expr_type

        if make_copy: node = copy.copy(node)
        if node.ref_kind == REF_KIND_NONE and (
                self.registry.is_array_type(type_id) or
                (self.registry.is_struct_type(type_id) and
                 not self.registry.is_tuple_type(type_id) and
                 not self.registry.is_enum_type(type_id))):
                node.ref_kind = REF_KIND_GENERIC

        return node

    def create_variable_node(self, token, name, type_id):
        """Create a variable node with appropriate ref_kind for struct instances"""
        var_node = VariableNode(token, name, type_id)
        return self.inject_ref_kind(var_node, make_copy=False)

    def create_var_decl_node(self, token, decl_type, name, type_id, expr=None):
        """Create a variable declaration node with appropriate ref_kind for struct instances"""
        var_decl = VarDeclNode(token, decl_type, name, type_id, expr)
        return self.inject_ref_kind(var_decl, make_copy=False)

    def copy_with_ref_kind(self, node):
        """
        Make a copy of a node and inject struct ref_kind if needed.
        This function always makes a copy.
        """
        return self.inject_ref_kind(node, make_copy=True)

    def flatten_function(self, func_node):
        """Entry point: Transform a function by flattening expressions in its body"""
        self._reset_temps()

        # Track the current function context
        old_function = self.current_function
        self.current_function = func_node

        func_id = self.registry.lookup_function(func_node.name, func_node.parent_struct_id)
        assert(func_id != -1)
        func_obj = self.registry.get_func_from_id(func_id)

        # If this is a constructor, make sure it's marked as returning by reference for C codegen
        # regardless of the semantics in the language
        if func_node.parent_struct_id != -1 and func_node.name == "init":
            # Update function object in registry
            func_obj.is_ref_return = True
            # Update AST node
            func_node.return_ref_kind = REF_KIND_GENERIC

        # Create a copy of the function with a transformed body
        new_func = copy.copy(func_node)
        new_func.body = self.flatten_statements(func_node.body)

        # update registry with new AST
        func_obj.ast_node = new_func

        # Restore previous function context
        self.current_function = old_function

        return new_func

    def flatten_statements(self, statements):
        """Transform a list of statements, flattening expressions"""
        result = []

        for stmt in statements:
            flattened = self.flatten_statement(stmt)

            # Some statement transformations might return multiple statements
            if isinstance(flattened, list):
                result.extend(flattened)
            else:
                result.append(flattened)

        return result

    def flatten_statement(self, stmt):
        """Transform a single statement, flattening expressions"""
        if stmt is None:
            return None

        # Dispatch based on statement type
        if stmt.node_type == AST_NODE_EXPR_STMT:
            return self.flatten_expr_stmt(stmt)
        elif stmt.node_type == AST_NODE_VAR_DECL:
            return self.flatten_var_decl(stmt)
        elif stmt.node_type == AST_NODE_IF:
            return self.flatten_if(stmt)
        elif stmt.node_type == AST_NODE_WHILE:
            return self.flatten_while(stmt)
        elif stmt.node_type == AST_NODE_RETURN:
            return self.flatten_return(stmt)
        elif stmt.node_type == AST_NODE_PRINT:
            return self.flatten_print(stmt)
        elif stmt.node_type == AST_NODE_BREAK or stmt.node_type == AST_NODE_CONTINUE:
            return copy.copy(stmt)  # No expressions to flatten
        elif stmt.node_type == AST_NODE_DEL:
            return self.flatten_del(stmt)
        else:
            # For any other statement types
            return copy.copy(stmt)

    def flatten_expr_stmt(self, stmt):
        """Transform an expression statement"""
        expr, hoisted_stmts = self.flatten_expr(stmt.expr)

        if not hoisted_stmts:
            # If no hoisted statements, just update the expression
            new_stmt = copy.copy(stmt)
            new_stmt.expr = expr
            return new_stmt
        else:
            # If we have hoisted statements, append the transformed statement
            result = list(hoisted_stmts)  # Copy the list
            new_stmt = copy.copy(stmt)
            new_stmt.expr = expr
            result.append(new_stmt)
            return result

    def flatten_var_decl(self, stmt):
        """Transform a variable declaration"""
        new_stmt = self.copy_with_ref_kind(stmt)

        if not stmt.expr:
            # No initializer, nothing to flatten
            return new_stmt

        expr, hoisted_stmts = self.flatten_expr(stmt.expr)

        if not hoisted_stmts:
            # No hoisted statements, just update the expression
            new_stmt.expr = expr
            return new_stmt
        else:
            # Hoist statements before the declaration
            result = list(hoisted_stmts)  # Copy the list
            new_stmt.expr = expr
            result.append(new_stmt)
            return result

    def flatten_if(self, stmt):
        """Transform an if statement"""
        # Transform the condition
        condition, hoisted_stmts = self.flatten_expr(stmt.condition)

        # Transform the then body
        then_body = self.flatten_statements(stmt.then_body)

        # Transform the else body if it exists
        else_body = None
        if stmt.else_body:
            else_body = self.flatten_statements(stmt.else_body)

        # Create a new if node
        new_if = copy.copy(stmt)
        new_if.condition = condition
        new_if.then_body = then_body
        new_if.else_body = else_body

        # If we have hoisted statements, they go before the if
        if hoisted_stmts:
            return hoisted_stmts + [new_if]
        else:
            return new_if

    def flatten_while(self, stmt):
        """Transform a while statement"""
        # Transform the condition
        condition, hoisted_stmts = self.flatten_expr(stmt.condition)

        # Transform the loop body
        body = self.flatten_statements(stmt.body)

        # Create a new while node
        new_while = copy.copy(stmt)
        new_while.condition = condition
        new_while.body = body

        # If we have hoisted statements for the condition, special handling is needed
        if hoisted_stmts:
            # For complex conditions, we need to use a different approach:
            # Create an infinite loop with the condition check inside

            # Create the condition check with break
            condition_check = IfNode(stmt.token,
                                    UnaryOpNode(stmt.token, "!", condition, TYPE_INT),
                                    [BreakNode(stmt.token)],
                                    None)

            # The body becomes: condition setup, condition check, original body
            new_body = hoisted_stmts + [condition_check] + body

            # Create an infinite loop
            return WhileNode(stmt.token, NumberNode(stmt.token, 1, TYPE_INT), new_body)
        else:
            # Simple condition, no special handling needed
            return new_while

    def flatten_return(self, stmt):
        """Transform a return statement"""
        if not stmt.expr:
            # No expression, nothing to flatten
            return copy.copy(stmt)

        expr, hoisted_stmts = self.flatten_expr(stmt.expr)

        new_stmt = copy.copy(stmt)
        new_stmt.expr = expr

        # Special handling for constructor returns - set reference kind
        if (self.current_function and
            self.current_function.parent_struct_id != -1 and
            self.current_function.name == "init" and
            new_stmt.ref_kind == REF_KIND_NONE):
            new_stmt.ref_kind = REF_KIND_GENERIC

        if not hoisted_stmts:
            # No hoisted statements, just update the expression
            return new_stmt
        else:
            # Hoist statements before the return
            result = list(hoisted_stmts)  # Copy the list
            result.append(new_stmt)
            return result

    def flatten_print(self, stmt):
        """Transform a print statement"""
        expr, hoisted_stmts = self.flatten_expr(stmt.expr)

        if not hoisted_stmts:
            # No hoisted statements, just update the expression
            new_stmt = copy.copy(stmt)
            new_stmt.expr = expr
            return new_stmt
        else:
            # Hoist statements before the print
            result = list(hoisted_stmts)  # Copy the list
            new_stmt = copy.copy(stmt)
            new_stmt.expr = expr
            result.append(new_stmt)
            return result

    def flatten_del(self, stmt):
        """Transform a delete statement"""
        expr, hoisted_stmts = self.flatten_expr(stmt.expr)

        if not hoisted_stmts:
            # No hoisted statements, just update the expression
            new_stmt = copy.copy(stmt)
            new_stmt.expr = expr
            return new_stmt
        else:
            # Hoist statements before the del
            result = list(hoisted_stmts)  # Copy the list
            new_stmt = copy.copy(stmt)
            new_stmt.expr = expr
            result.append(new_stmt)
            return result

    def flatten_expr(self, node):
        """
        Transform an expression, possibly introducing temporary variables.
        Returns (new_expr, hoisted_statements).
        """
        if node is None:
            return None, []

        # Dispatch based on expression type
        if self.is_constructor_node(node):
            return self.flatten_constructor(node)
        elif node.node_type == AST_NODE_CALL:
            return self.flatten_call(node)
        elif node.node_type == AST_NODE_BINARY_OP:
            return self.flatten_binary_op(node)
        elif node.node_type == AST_NODE_UNARY_OP:
            return self.flatten_unary_op(node)
        elif node.node_type == AST_NODE_COMPARE:
            return self.flatten_compare(node)
        elif node.node_type == AST_NODE_LOGICAL:
            return self.flatten_logical(node)
        elif node.node_type == AST_NODE_BITOP:
            return self.flatten_bitop(node)
        elif node.node_type == AST_NODE_MEMBER_ACCESS:
            return self.flatten_member_access(node)
        elif node.node_type == AST_NODE_ARRAY_ACCESS:
            return self.flatten_array_access(node)
        elif node.node_type == AST_NODE_NEW:
            return self.flatten_new(node)
        elif node.node_type == AST_NODE_GENERIC_INITIALIZER:
            return self.flatten_generic_initializer(node)
        elif node.node_type == AST_NODE_ARRAY_RESIZE:
            return self.flatten_array_resize(node)
        elif node.node_type == AST_NODE_VARIABLE:
            return self.copy_with_ref_kind(node), []
        else:
            # For simple expressions like variables, constants, etc.
            return copy.copy(node), []

    def flatten_constructor(self, node):
        """Handle constructor calls like __dunno__.init()"""
        # Preserve the reference kind if already set (e.g., REF_KIND_HEAP for new expressions)
        preserved_ref_kind = node.ref_kind

        # Create a temporary variable for the object
        struct_type = node.expr_type
        temp_name = self.get_temp_name()

        # Process constructor arguments if any
        arg_exprs = []
        hoisted_stmts = []

        # For constructor calls, the first arg is a placeholder 'self'
        # We need to replace this with our new temporary
        temp_var = self.create_variable_node(node.token, temp_name, struct_type)
        arg_exprs.append(temp_var)

        # Process the remaining arguments (skip the first 'self' placeholder)
        for arg in node.args[1:]:
            arg_expr, arg_hoisted = self.flatten_expr(arg)
            arg_exprs.append(arg_expr)
            hoisted_stmts.extend(arg_hoisted)

        # Create a variable declaration for the temp (initialized to default)
        var_decl = self.create_var_decl_node(node.token, TT_VAR, temp_name, struct_type, None)

        # Create a constructor call
        struct_name = self.registry.get_struct_name(struct_type)
        init_call = CallNode(node.token, "%s_init" % struct_name, arg_exprs, TYPE_VOID)
        init_stmt = ExprStmtNode(node.token, init_call)

        # For C code generation, set reference kind appropriately
        # If it was REF_KIND_HEAP (inside new), preserve it, otherwise set to REF_KIND_GENERIC
        if preserved_ref_kind == REF_KIND_HEAP:
            temp_var.ref_kind = REF_KIND_HEAP
        else:
            temp_var.ref_kind = REF_KIND_GENERIC

        # Order is important: declare temp, hoist setup, call constructor
        hoisted_stmts.insert(0, var_decl)
        hoisted_stmts.append(init_stmt)

        # Return the temp var as the expression
        return temp_var, hoisted_stmts

    def flatten_call(self, node):
        """Handle function calls and method calls"""
        # Function call
        if not node.obj:
            # Process arguments
            arg_exprs = []
            hoisted_stmts = []
            
            for arg in node.args:
                arg_expr, arg_hoisted = self.flatten_expr(arg)
                arg_exprs.append(arg_expr)
                hoisted_stmts.extend(arg_hoisted)
            
            # Create a new call node with the processed arguments
            new_call = copy.copy(node)
            new_call.args = arg_exprs
            
            return new_call, hoisted_stmts
            
        # Method call
        else:
            # Process the object expression
            obj_expr, obj_hoisted = self.flatten_expr(node.obj)

            # Process arguments
            arg_exprs = []
            arg_hoisted = []

            for arg in node.args:
                arg_expr, hoisted = self.flatten_expr(arg)
                arg_exprs.append(arg_expr)
                arg_hoisted.extend(hoisted)

            # Combine hoisted statements in correct order
            hoisted_stmts = obj_hoisted + arg_hoisted

            # For method chaining, we need a temporary for the result
            if node.expr_type != TYPE_VOID:
                temp_name = self.get_temp_name()
                temp_var = self.create_variable_node(node.token, temp_name, node.expr_type)

                # Create the method call
                new_call = copy.copy(node)
                new_call.obj = obj_expr
                new_call.args = arg_exprs

                # Create assignment to temporary
                var_decl = self.create_var_decl_node(node.token, TT_VAR, temp_name, node.expr_type, new_call)
                hoisted_stmts.append(var_decl)

                return temp_var, hoisted_stmts
            else:
                # No result needed, just call the method
                new_call = copy.copy(node)
                new_call.obj = obj_expr
                new_call.args = arg_exprs

                return new_call, hoisted_stmts

    def flatten_binary_op(self, node):
        """Handle binary operations"""
        left_expr, left_hoisted = self.flatten_expr(node.left)
        right_expr, right_hoisted = self.flatten_expr(node.right)
        
        # If both operands are simple, no need for a temporary
        if not left_hoisted and not right_hoisted:
            new_op = copy.copy(node)
            new_op.left = left_expr
            new_op.right = right_expr
            return new_op, []
        
        # Otherwise, we need to hoist the operands
        # Order matters: left side evaluated first
        hoisted_stmts = left_hoisted + right_hoisted
        
        new_op = copy.copy(node)
        new_op.left = left_expr
        new_op.right = right_expr
        
        return new_op, hoisted_stmts
    
    def flatten_unary_op(self, node):
        """Handle unary operations"""
        expr, hoisted_stmts = self.flatten_expr(node.operand)
        
        new_op = copy.copy(node)
        new_op.operand = expr
        
        return new_op, hoisted_stmts
    
    def flatten_compare(self, node):
        """Handle comparison operations"""
        left_expr, left_hoisted = self.flatten_expr(node.left)
        right_expr, right_hoisted = self.flatten_expr(node.right)
        
        # Combine hoisted statements in correct order
        hoisted_stmts = left_hoisted + right_hoisted
        
        new_op = copy.copy(node)
        new_op.left = left_expr
        new_op.right = right_expr
        
        return new_op, hoisted_stmts
    
    def flatten_logical(self, node):
        """Handle logical operations"""
        # For logical operators, we need to handle short-circuit evaluation correctly
        # But since we're just flattening expressions and not changing control flow,
        # we can just handle the hoisted statements in order
        left_expr, left_hoisted = self.flatten_expr(node.left)
        right_expr, right_hoisted = self.flatten_expr(node.right)
        
        # Combine hoisted statements in correct order
        hoisted_stmts = left_hoisted + right_hoisted
        
        new_op = copy.copy(node)
        new_op.left = left_expr
        new_op.right = right_expr
        
        return new_op, hoisted_stmts
    
    def flatten_bitop(self, node):
        """Handle bitwise operations"""
        left_expr, left_hoisted = self.flatten_expr(node.left)
        right_expr, right_hoisted = self.flatten_expr(node.right)
        
        # Combine hoisted statements in correct order
        hoisted_stmts = left_hoisted + right_hoisted
        
        new_op = copy.copy(node)
        new_op.left = left_expr
        new_op.right = right_expr
        
        return new_op, hoisted_stmts
    
    def flatten_member_access(self, node):
        """Handle member access"""
        obj_expr, obj_hoisted = self.flatten_expr(node.obj)
        
        new_access = copy.copy(node)
        new_access.obj = obj_expr
        
        return new_access, obj_hoisted
    
    def flatten_array_access(self, node):
        """Handle array access"""
        array_expr, array_hoisted = self.flatten_expr(node.array)
        index_expr, index_hoisted = self.flatten_expr(node.index)
        
        # Combine hoisted statements in correct order
        hoisted_stmts = array_hoisted + index_hoisted
        
        new_access = copy.copy(node)
        new_access.array = array_expr
        new_access.index = index_expr
        
        return new_access, hoisted_stmts
    
    def flatten_new(self, node):
        """Handle new expressions"""
        # We need to flatten the struct_init part
        struct_init, hoisted_stmts = self.flatten_expr(node.struct_init)
        
        new_node = copy.copy(node)
        new_node.struct_init = struct_init
        
        return new_node, hoisted_stmts
    
    def flatten_generic_initializer(self, node):
        """Handle generic initializers (arrays, structs)"""
        # Process all elements in the initializer
        new_elements = []
        hoisted_stmts = []
        
        for elem in node.elements:
            elem_expr, elem_hoisted = self.flatten_expr(elem)
            new_elements.append(elem_expr)
            hoisted_stmts.extend(elem_hoisted)
        
        new_init = copy.copy(node)
        new_init.elements = new_elements
        
        return new_init, hoisted_stmts
    
    def flatten_array_resize(self, node):
        """Handle array resize operations"""
        array_expr, array_hoisted = self.flatten_expr(node.array_expr)
        size_expr, size_hoisted = self.flatten_expr(node.size_expr)
        
        # Combine hoisted statements in correct order
        hoisted_stmts = array_hoisted + size_hoisted
        
        new_resize = copy.copy(node)
        new_resize.array_expr = array_expr
        new_resize.size_expr = size_expr
        
        return new_resize, hoisted_stmts
    
    def is_constructor_node(self, node):
        """Determine if a node is a constructor call"""
        return (node.node_type == AST_NODE_CALL and
                node.obj and node.obj.node_type == AST_NODE_VARIABLE and
                node.obj.name == "__dunno__" and node.name == "init")
    
    def get_temp_name(self):
        """Generate a unique temporary variable name"""
        temp_name = "__temp_%d" % self.temp_counter
        self.temp_counter += 1
        return temp_name
