# Type registry for struct types
from shared import *

# Global registry of structs
_struct_registry = {}  # name -> (type_id, parent_id, fields, methods)
_next_struct_id = TYPE_STRUCT_BASE

def reset_registry():
    """Reset the type registry to initial state"""
    global _struct_registry, _next_struct_id
    _struct_registry.clear()
    _next_struct_id = TYPE_STRUCT_BASE

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

def is_derived_from(child_name, parent_name):
    """Check if child struct is derived from parent struct"""
    if not struct_exists(child_name) or not struct_exists(parent_name):
        return False
        
    parent_id = _struct_registry[parent_name][0]
    
    current_name = child_name
    while current_name:
        _, parent_id, _, _ = _struct_registry[current_name]
        
        if parent_id is None:
            return False
        
        # Get parent struct name
        parent_name = None
        for name, (tid, _, _, _) in _struct_registry.items():
            if tid == parent_id:
                parent_name = name
                break
        
        if parent_name == parent_name:
            return True
        
        current_name = parent_name
    
    return False
