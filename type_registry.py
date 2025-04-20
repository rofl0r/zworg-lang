# Type registry for struct types
from shared import *

# In type_registry.py
class Function:
    def __init__(self, name, return_type, params, parent_struct_id=-1, ast_node=None):
        self.name = name # this is the "qualified name" containing struct, prefix in case of a method
        self.return_type = return_type
        self.params = params  # List of (name, type) tuples
        self.parent_struct_id = parent_struct_id  # -1 for global functions
        self.ast_node = ast_node  # Store AST nodes directly

# Function Storage
_functions = {}  # funcid -> Function object
_next_funcid = 1  # Start with positive numbers so -1 can mean "not found"
_func_map = {}  # (struct_id, func_name) -> funcid  # struct_id=-1 for regular functions

# Global registry of structs
_struct_registry = {}  # name -> (type_id, parent_id, fields, methods)
_next_struct_id = TYPE_STRUCT_BASE

def register_function(name, return_type, params, parent_struct_id=-1, ast_node=None):
    global _next_funcid
    funcid = _next_funcid
    _next_funcid += 1

    # Create qualified name for methods (for debugging)
    qualified_name = name
    if parent_struct_id != -1:
        struct_name = get_struct_name(parent_struct_id)
        qualified_name = "%s.%s" % (struct_name, name)

    func = Function(qualified_name, return_type, params, parent_struct_id, ast_node)
    _functions[funcid] = func

    # Add to unified lookup map
    _func_map[(parent_struct_id, name)] = funcid

    return funcid

def get_func_from_id(funcid):
    return _functions[funcid]

# Helper functions that fail fast
def is_method(funcid):
    """Check if a function ID represents a method"""
    return _functions[funcid].parent_struct_id != -1

def set_function_ast_node(funcid, ast_node):
    """Set the body AST nodes of a function by ID"""
    _functions[funcid].ast_node = ast_node

# Unified lookup function returning -1 if not found
def lookup_function(name, struct_id=-1):
    """Look up a function ID by name and optional struct ID"""
    key = (struct_id, name)
    return _func_map.get(key, -1)  # Return -1 if not found

# Reset registry (for testing)
def reset_functions():
    """Reset the function registry"""
    global _next_funcid, _functions, _func_map
    _functions = {}
    _next_funcid = 1
    _func_map = {}

def reset_registry():
    """Reset the type registry to initial state"""
    global _struct_registry, _next_struct_id
    _struct_registry.clear()
    _next_struct_id = TYPE_STRUCT_BASE
    reset_functions()

def register_struct(name, parent_name=None, token=None):
    """Register a new struct type, return its ID"""
    global _next_struct_id

    if name in _struct_registry:
        raise CompilerException("Struct '%s' is already defined" % name, token)

    type_id = _next_struct_id
    _next_struct_id += 1

    parent_id = None
    if parent_name:
        if parent_name not in _struct_registry:
            raise CompilerException("Parent struct '%s' is not defined" % parent_name, token)
        parent_id = _struct_registry[parent_name][0]  # Get parent's type_id

    # (type_id, parent_id, fields, methods)
    _struct_registry[name] = (type_id, parent_id, [], {})
    return type_id

def add_field(struct_name, field_name, field_type, token=None):
    """Add a field to a struct definition"""
    if struct_name not in _struct_registry:
        raise CompilerException("Struct '%s' is not defined" % struct_name, token)

    _, _, fields, _ = _struct_registry[struct_name]

    # Check for duplicate field
    for name, _ in fields:
        if name == field_name:
            raise CompilerException("Field '%s' is already defined in struct '%s'" % (field_name, struct_name), token)

    fields.append((field_name, field_type))

def add_method(struct_name, method_name, method_node, token=None):
    """Add a method to a struct definition"""
    if struct_name not in _struct_registry:
        raise CompilerException("Struct '%s' is not defined" % struct_name, token)

    _, _, _, methods = _struct_registry[struct_name]

    # Check for duplicate method
    if method_name in methods:
        raise CompilerException("Method '%s' is already defined in struct '%s'" % (method_name, struct_name), token)

    methods[method_name] = method_node

def get_struct_id(struct_name):
    """Get the type ID for a struct"""
    if struct_name not in _struct_registry:
        return None
    return _struct_registry[struct_name][0]

def get_struct_parent(struct_name):
    """Get the parent struct name"""
    if struct_name not in _struct_registry:
        return None
    parent_id = _struct_registry[struct_name][1]
    if parent_id is None:
        return None

    # Find parent name from ID
    for name, (type_id, _, _, _) in _struct_registry.items():
        if type_id == parent_id:
            return name
    return None

def get_all_fields(struct_name):
    """Get all fields including those from parent structs"""
    if struct_name not in _struct_registry:
        return []

    fields = []

    # First get parent fields if any
    parent_name = get_struct_parent(struct_name)
    if parent_name:
        fields.extend(get_all_fields(parent_name))

    # Add fields from the current struct
    _, _, struct_fields, _ = _struct_registry[struct_name]
    fields.extend(struct_fields)

    return fields

def get_field_type(struct_name, field_name):
    """Get the type of a field in a struct (including parent fields)"""
    for name, type_ in get_all_fields(struct_name):
        if name == field_name:
            return type_
    return None

def has_method(struct_name, method_name):
    """Check if a struct has a method (including parent methods)"""
    if struct_name not in _struct_registry:
        return False

    # Check current struct
    _, _, _, methods = _struct_registry[struct_name]
    if method_name in methods:
        return True

    # Check parent struct
    parent_name = get_struct_parent(struct_name)
    if parent_name:
        return has_method(parent_name, method_name)

    return False

def get_method(struct_name, method_name):
    """Get a method from a struct or its parents"""
    if struct_name not in _struct_registry:
        return None

    # Check current struct
    _, _, _, methods = _struct_registry[struct_name]
    if method_name in methods:
        return methods[method_name]

    # Check parent struct
    parent_name = get_struct_parent(struct_name)
    if parent_name:
        return get_method(parent_name, method_name)

    return None

def struct_exists(struct_name):
    """Check if a struct exists"""
    return struct_name in _struct_registry

def get_struct_name(type_id):
    """Get struct name from type ID"""
    # Handle reference types
    base_type = get_base_type(type_id)

    for name, (tid, _, _, _) in _struct_registry.items():
        if tid == base_type:
            return name
    return None

