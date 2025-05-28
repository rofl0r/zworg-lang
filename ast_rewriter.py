# AST Type Rewriter for generic type instantiation
# Walks an AST and replaces generic type IDs with concrete type IDs

from shared import *
import copy

# Special constant to match all node types
AST_NODE_ALL = -1

class AstTypeRewriter:
    """
    Walks an AST and replaces type IDs according to a mapping.
    
    Usage:
        rewriter = AstTypeRewriter(type_mapping)
        new_ast = rewriter.rewrite(ast_node, node_types_to_rewrite)
    """
    
    def __init__(self, type_mapping):
        """
        Initialize with a mapping from old type IDs to new type IDs
        
        Args:
            type_mapping: Dict mapping old type IDs to new type IDs
        """
        self.type_mapping = type_mapping
        
        # Create mapping of node types to handler methods
        self.handlers = {
            AST_NODE_NUMBER: self.rewrite_expr_node,
            AST_NODE_STRING: self.rewrite_expr_node,
            AST_NODE_VARIABLE: self.rewrite_expr_node,
            AST_NODE_BINARY_OP: self.rewrite_binary_op,
            AST_NODE_UNARY_OP: self.rewrite_unary_op,
            AST_NODE_PRINT: self.rewrite_print,
            AST_NODE_IF: self.rewrite_if,
            AST_NODE_WHILE: self.rewrite_while,
            AST_NODE_BREAK: self.rewrite_simple_node,
            AST_NODE_CONTINUE: self.rewrite_simple_node,
            AST_NODE_EXPR_STMT: self.rewrite_expr_stmt,
            AST_NODE_VAR_DECL: self.rewrite_var_decl,
            AST_NODE_FUNCTION_DECL: self.rewrite_function_decl,
            AST_NODE_CALL: self.rewrite_call,
            AST_NODE_RETURN: self.rewrite_return,
            AST_NODE_COMPARE: self.rewrite_compare,
            AST_NODE_LOGICAL: self.rewrite_logical,
            AST_NODE_BITOP: self.rewrite_bitop,
            AST_NODE_STRUCT_DEF: self.rewrite_struct_def,
            AST_NODE_MEMBER_ACCESS: self.rewrite_member_access,
            AST_NODE_ARRAY_ACCESS: self.rewrite_array_access,
            AST_NODE_NEW: self.rewrite_new,
            AST_NODE_DEL: self.rewrite_del,
            AST_NODE_NIL: self.rewrite_expr_node,
            AST_NODE_GENERIC_INITIALIZER: self.rewrite_generic_initializer,
            AST_NODE_ARRAY_RESIZE: self.rewrite_array_resize,
        }
    
    def rewrite(self, node, node_types=None):
        """
        Rewrite type IDs in the given AST node and its children
        """
        if node is None:
            return None
            
        # Check if we should rewrite this node type
        should_rewrite = (
            node_types is None or 
            AST_NODE_ALL in node_types or 
            node.node_type in node_types
        )
        
        # Get the appropriate handler for this node type
        handler = self.handlers.get(node.node_type, self.rewrite_unknown)
        
        # Call the handler with the should_rewrite flag
        return handler(node, should_rewrite)
    
    def replace_type_if_needed(self, type_id, should_rewrite):
        """Replace a type ID if it's in the mapping and should be rewritten"""
        if not should_rewrite:
            return type_id
            
        # Direct mapping
        if type_id in self.type_mapping:
            return self.type_mapping.get(type_id)

        # Handle array types - using lazy import to avoid circular dependency
        from type_registry import get_registry
        registry = get_registry()
        element_type = registry.get_array_element_type(type_id)
        if element_type in self.type_mapping:
            # Found a mapping for the element type
            new_element_type = self.type_mapping[element_type]
            # Get array size if any
            array_size = registry.get_array_size(type_id)
            # Register and return a new array type with the replaced element type
            return registry.register_array_type(new_element_type, array_size)

        return type_id
    
    def rewrite_expr_node(self, node, should_rewrite):
        """Rewrite an expression node (number, string, variable, nil)"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        return new_node
    
    def rewrite_simple_node(self, node, should_rewrite):
        """Rewrite a node with no children (break, continue)"""
        return copy.copy(node)
    
    def rewrite_binary_op(self, node, should_rewrite):
        """Rewrite a binary operation node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        new_node.left = self.rewrite(node.left, node_types=[AST_NODE_ALL])
        new_node.right = self.rewrite(node.right, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_unary_op(self, node, should_rewrite):
        """Rewrite a unary operation node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        new_node.operand = self.rewrite(node.operand, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_print(self, node, should_rewrite):
        """Rewrite a print statement node"""
        new_node = copy.copy(node)
        new_node.expr = self.rewrite(node.expr, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_if(self, node, should_rewrite):
        """Rewrite an if statement node"""
        new_node = copy.copy(node)
        new_node.condition = self.rewrite(node.condition, node_types=[AST_NODE_ALL])
        
        # Rewrite then body
        new_node.then_body = []
        for stmt in node.then_body:
            new_node.then_body.append(self.rewrite(stmt, node_types=[AST_NODE_ALL]))
        
        # Rewrite else body if it exists
        if node.else_body:
            new_node.else_body = []
            for stmt in node.else_body:
                new_node.else_body.append(self.rewrite(stmt, node_types=[AST_NODE_ALL]))
        
        return new_node
    
    def rewrite_while(self, node, should_rewrite):
        """Rewrite a while loop node"""
        new_node = copy.copy(node)
        new_node.condition = self.rewrite(node.condition, node_types=[AST_NODE_ALL])
        
        # Rewrite body
        new_node.body = []
        for stmt in node.body:
            new_node.body.append(self.rewrite(stmt, node_types=[AST_NODE_ALL]))
        
        return new_node
    
    def rewrite_expr_stmt(self, node, should_rewrite):
        """Rewrite an expression statement node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        new_node.expr = self.rewrite(node.expr, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_var_decl(self, node, should_rewrite):
        """Rewrite a variable declaration node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        if node.expr:
            new_node.expr = self.rewrite(node.expr, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_function_decl(self, node, should_rewrite):
        """Rewrite a function declaration node"""
        new_node = copy.copy(node)

        # If this is a method of a generic struct, translate its parent struct ID
        new_node.parent_struct_id = self.replace_type_if_needed(new_node.parent_struct_id, should_rewrite)

        # Rewrite return type
        new_node.return_type = self.replace_type_if_needed(new_node.return_type, should_rewrite)

        # Rewrite parameter types
        if hasattr(new_node, 'params'):
            new_params = []
            for name, type_id, is_byref in new_node.params:
                new_type_id = self.replace_type_if_needed(type_id, should_rewrite)
                new_params.append((name, new_type_id, is_byref))
            new_node.params = new_params

        # Rewrite function body
        if hasattr(new_node, 'body'):
            new_node.body = []
            for stmt in node.body:
                new_node.body.append(self.rewrite(stmt, node_types=[AST_NODE_ALL]))

        return new_node

    def rewrite_call(self, node, should_rewrite):
        """Rewrite a function/method call node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        
        # Rewrite object if it's a method call
        if node.obj:
            new_node.obj = self.rewrite(node.obj, node_types=[AST_NODE_ALL])
        
        # Rewrite arguments
        new_node.args = []
        for arg in node.args:
            new_node.args.append(self.rewrite(arg, node_types=[AST_NODE_ALL]))
        
        return new_node
    
    def rewrite_return(self, node, should_rewrite):
        """Rewrite a return statement node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        if node.expr:
            new_node.expr = self.rewrite(node.expr, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_compare(self, node, should_rewrite):
        """Rewrite a comparison node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        new_node.left = self.rewrite(node.left, node_types=[AST_NODE_ALL])
        new_node.right = self.rewrite(node.right, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_logical(self, node, should_rewrite):
        """Rewrite a logical operation node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        new_node.left = self.rewrite(node.left, node_types=[AST_NODE_ALL])
        new_node.right = self.rewrite(node.right, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_bitop(self, node, should_rewrite):
        """Rewrite a bitwise operation node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        new_node.left = self.rewrite(node.left, node_types=[AST_NODE_ALL])
        new_node.right = self.rewrite(node.right, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_struct_def(self, node, should_rewrite):
        """Rewrite a struct definition node"""
        new_node = copy.copy(node)
        
        if hasattr(new_node, 'fields'):
            # Rewrite field types
            new_fields = []
            for name, type_id in new_node.fields:
                new_type_id = self.replace_type_if_needed(type_id, should_rewrite)
                new_fields.append((name, new_type_id))
            new_node.fields = new_fields
        
        return new_node
    
    def rewrite_member_access(self, node, should_rewrite):
        """Rewrite a member access node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        new_node.obj = self.rewrite(node.obj, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_array_access(self, node, should_rewrite):
        """Rewrite an array access node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        new_node.array = self.rewrite(node.array, node_types=[AST_NODE_ALL])
        new_node.index = self.rewrite(node.index, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_new(self, node, should_rewrite):
        """Rewrite a new expression node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        new_node.struct_init = self.rewrite(node.struct_init, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_del(self, node, should_rewrite):
        """Rewrite a del statement node"""
        new_node = copy.copy(node)
        new_node.expr = self.rewrite(node.expr, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_generic_initializer(self, node, should_rewrite):
        """Rewrite a generic initializer node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        
        # Rewrite elements
        if hasattr(new_node, 'elements'):
            new_node.elements = []
            for elem in node.elements:
                new_node.elements.append(self.rewrite(elem, node_types=[AST_NODE_ALL]))
        
        return new_node
    
    def rewrite_array_resize(self, node, should_rewrite):
        """Rewrite an array resize node"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        new_node.array_expr = self.rewrite(node.array_expr, node_types=[AST_NODE_ALL])
        new_node.size_expr = self.rewrite(node.size_expr, node_types=[AST_NODE_ALL])
        return new_node
    
    def rewrite_unknown(self, node, should_rewrite):
        """Fallback for unknown node types"""
        new_node = copy.copy(node)
        new_node.expr_type = self.replace_type_if_needed(new_node.expr_type, should_rewrite)
        return new_node
