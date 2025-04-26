# Type registry for struct types
from shared import *

# Type descriptor classes
class TypeDescriptor(object):
    """Base descriptor for all types"""
    def __init__(self, kind):
        self.kind = kind

class PrimitiveDescriptor(TypeDescriptor):
    """Descriptor for primitive types"""
    def __init__(self, primitive_type):
        TypeDescriptor.__init__(self, TypeRegistry.TYPE_KIND_PRIMITIVE)
        self.primitive_type = primitive_type

class StructDescriptor(TypeDescriptor):
    """Descriptor for struct types"""
    def __init__(self, name, parent_id=-1):
        TypeDescriptor.__init__(self, TypeRegistry.TYPE_KIND_STRUCT)
        self.name = name
        self.parent_id = parent_id
        self.fields = []  # List of (name, type_id) tuples

class ArrayDescriptor(TypeDescriptor):
    """Descriptor for array types"""
    def __init__(self, element_type_id, size=None):
        TypeDescriptor.__init__(self, TypeRegistry.TYPE_KIND_ARRAY)
        self.element_type_id = element_type_id
        self.size = size  # None = dynamic size, int = fixed size

class Function:
    def __init__(self, name, return_type, params, parent_struct_id=-1, ast_node=None):
        self.name = name  # Qualified name containing struct prefix for methods
        self.return_type = return_type
        self.params = params  # List of (name, type) tuples
        self.parent_struct_id = parent_struct_id  # -1 for global functions
        self.ast_node = ast_node  # Store AST nodes directly

class TypeRegistry:
    # Type kind constants
    TYPE_KIND_PRIMITIVE = 0
    TYPE_KIND_STRUCT = 1
    TYPE_KIND_ARRAY = 2

    def __init__(self):
        """Initialize the registry"""
        self.reset()

    def reset(self):
        """Reset registry to initial state (for testing)"""
        # Type descriptor system
        self._type_descriptors = {}  # type_id -> TypeDescriptor

        # Legacy struct storage (maintained for compatibility)
        self._struct_registry = {}    # name -> (type_id, parent_id, fields)
        self._struct_id_to_name = {}  # type_id -> name
        self._next_struct_id = TYPE_STRUCT_BASE

        # Array type cache
        self._array_cache = {}  # (element_type_id, size) -> type_id

        # Function storage
        self._functions = {}    # funcid -> Function object
        self._next_funcid = 1   # Start with 1 so -1 means "not found"
        self._func_map = {}     # (struct_id, func_name) -> funcid

        # Initialize primitive types
        self._initialize_primitive_types()

    def _initialize_primitive_types(self):
        """Register all primitive types with descriptors"""
        primitive_types = [
            TYPE_UNKNOWN, TYPE_VOID, TYPE_STRING,
            TYPE_I8, TYPE_U8, TYPE_I16, TYPE_U16, TYPE_I32, TYPE_U32,
            TYPE_I64, TYPE_U64, TYPE_INT, TYPE_UINT, TYPE_LONG, TYPE_ULONG,
            TYPE_LONGLONG, TYPE_ULONGLONG, TYPE_FLOAT, TYPE_DOUBLE
        ]

        for type_id in primitive_types:
            self._type_descriptors[type_id] = PrimitiveDescriptor(type_id)

    # Array type methods
    def register_array_type(self, element_type_id, size=None):
        """Register an array type with element type and optional size"""
        # Check cache for existing identical array type
        cache_key = (element_type_id, size)
        if cache_key in self._array_cache:
            return self._array_cache[cache_key]

        # Create new array type ID
        type_id = self._next_struct_id
        self._next_struct_id += 1

        # Create descriptor
        descriptor = ArrayDescriptor(element_type_id, size)
        self._type_descriptors[type_id] = descriptor

        # Add to array cache
        self._array_cache[cache_key] = type_id

        # Create debug name
        element_type_name = self.type_to_string(element_type_id)
        size_str = str(size) if size is not None else ""
        array_name = "_array_%s_%s" % (element_type_name, size_str)
        self._struct_id_to_name[type_id] = array_name

        return type_id

    def is_array_type(self, type_id):
        """Check if a type is an array"""
        # Get base type if this is a reference
        base_type = self.get_base_type(type_id)
        descriptor = self._type_descriptors.get(base_type)
        return descriptor and descriptor.kind == self.TYPE_KIND_ARRAY

    def get_array_element_type(self, array_type_id):
        """Get element type for an array"""
        # Get base type if this is a reference
        base_type = self.get_base_type(array_type_id)
        descriptor = self._type_descriptors.get(base_type)
        if descriptor and descriptor.kind == self.TYPE_KIND_ARRAY:
            return descriptor.element_type_id
        return -1

    def get_array_size(self, array_type_id):
        """Get size for an array (None if dynamic)"""
        # Get base type if this is a reference
        base_type = self.get_base_type(array_type_id)
        descriptor = self._type_descriptors.get(base_type)
        if descriptor and descriptor.kind == self.TYPE_KIND_ARRAY:
            return descriptor.size
        return -1

    # Struct methods
    def register_struct(self, name, parent_name=None, token=None):
        """Register a new struct type, return its ID"""
        # Check if already registered
        if name in self._struct_registry:
            if token:
                raise CompilerException("Struct '%s' is already defined" % name, token)
            return self._struct_registry[name][0]

        # Get parent ID if specified
        parent_id = -1
        if parent_name:
            if parent_name not in self._struct_registry:
                if token:
                    raise CompilerException("Parent struct '%s' is not defined" % parent_name, token)
                return -1
            parent_id = self._struct_registry[parent_name][0]

        # Create new type ID
        type_id = self._next_struct_id
        self._next_struct_id += 1

        # Create descriptor for new system
        descriptor = StructDescriptor(name, parent_id)
        self._type_descriptors[type_id] = descriptor

        # Maintain compatibility with existing system
        self._struct_registry[name] = (type_id, parent_id, [])
        self._struct_id_to_name[type_id] = name

        return type_id

    def is_struct_type(self, type_id):
        """Check if a type is a struct type using descriptor information"""
        # Get base type (in case it's a reference)
        base_type = self.get_base_type(type_id)
        descriptor = self._type_descriptors.get(base_type)
        return descriptor is not None and descriptor.kind == self.TYPE_KIND_STRUCT

    def add_field(self, struct_name, field_name, field_type, token=None):
        """Add a field to a struct definition"""
        if struct_name not in self._struct_registry:
            if token:
                raise CompilerException("Struct '%s' is not defined" % struct_name, token)
            return False

        type_id, _, fields = self._struct_registry[struct_name]

        # Check for duplicate field
        for name, _ in fields:
            if name == field_name:
                if token:
                    raise CompilerException("Field '%s' is already defined in struct '%s'" % 
                                          (field_name, struct_name), token)
                return False

        # Add field to both systems
        fields.append((field_name, field_type))

        # Also update descriptor
        descriptor = self._type_descriptors.get(type_id)
        if descriptor and descriptor.kind == self.TYPE_KIND_STRUCT:
            descriptor.fields.append((field_name, field_type))

        return True

    def get_struct_id(self, struct_name):
        """Get the type ID for a struct"""
        if struct_name not in self._struct_registry:
            return -1
        return self._struct_registry[struct_name][0]

    def get_struct_parent(self, struct_name):
        """Get the parent struct name"""
        if struct_name not in self._struct_registry:
            return None
        parent_id = self._struct_registry[struct_name][1]
        if parent_id == -1:
            return None

        return self._struct_id_to_name.get(parent_id, None)

    def get_all_fields(self, struct_name):
        """Get all fields including those from parent structs"""
        if struct_name not in self._struct_registry:
            return []

        fields = []

        # First get parent fields if any
        parent_name = self.get_struct_parent(struct_name)
        if parent_name:
            fields.extend(self.get_all_fields(parent_name))

        # Add fields from the current struct
        _, _, struct_fields = self._struct_registry[struct_name]
        fields.extend(struct_fields)

        return fields

    def get_field_type(self, struct_name, field_name):
        """Get the type of a field in a struct (including parent fields)"""
        for name, type_ in self.get_all_fields(struct_name):
            if name == field_name:
                return type_
        return None

    def get_struct_name(self, type_id):
        """Get struct name from type ID"""
        # Handle reference types
        base_type = self.get_base_type(type_id)

        # Try descriptor system first
        descriptor = self._type_descriptors.get(base_type)
        if descriptor:
            if descriptor.kind == self.TYPE_KIND_STRUCT:
                return descriptor.name
            elif descriptor.kind == self.TYPE_KIND_ARRAY:
                # For arrays, return the name from the ID-to-name map
                return self._struct_id_to_name.get(base_type, None)

        # Fall back to legacy system
        return self._struct_id_to_name.get(base_type, None)

    def struct_exists(self, struct_name):
        """Check if a struct exists"""
        return struct_name in self._struct_registry

    # Function methods
    def register_function(self, name, return_type, params, parent_struct_id=-1, ast_node=None):
        """Register a function, returning its ID"""
        func_id = self._next_funcid
        self._next_funcid += 1

        # Create qualified name for methods (for debugging)
        qualified_name = name
        if parent_struct_id != -1:
            struct_name = self.get_struct_name(parent_struct_id)
            qualified_name = "%s.%s" % (struct_name, name)

        func = Function(qualified_name, return_type, params, parent_struct_id, ast_node)
        self._functions[func_id] = func

        # Add to unified lookup map
        self._func_map[(parent_struct_id, name)] = func_id

        return func_id

    def get_func_from_id(self, func_id):
        """Get function object by ID"""
        return self._functions.get(func_id)

    def set_function_ast_node(self, func_id, ast_node):
        """Set the AST node for a function"""
        if func_id in self._functions:
            self._functions[func_id].ast_node = ast_node

    def lookup_function(self, name, struct_id=-1, check_parents=True):
        """Look up a function by name and struct ID"""
        # Try direct lookup first
        key = (struct_id, name)
        func_id = self._func_map.get(key, -1)

        # If found or this is a global function, return the result
        if func_id != -1 or struct_id == -1 or not check_parents:
            return func_id

        # If not found and this is a struct, check parent structs
        struct_name = self.get_struct_name(struct_id)
        if struct_name:
            parent_name = self.get_struct_parent(struct_name)
            if parent_name:
                parent_id = self.get_struct_id(parent_name)
                return self.lookup_function(name, parent_id, check_parents)

        return -1  # Not found anywhere

    def get_method(self, struct_id, method_name):
        """Get a method from a struct or its parents"""
        if struct_id == -1:
            return None

        func_id = self.lookup_function(method_name, struct_id)
        if func_id == -1:
            return None
        return self._functions[func_id].ast_node

    def reset_functions(self):
        """Reset function registry"""
        self._functions = {}
        self._next_funcid = 1
        self._func_map = {}

    # This method can be an alias for var_type_to_string for internal usage
    def type_to_string(self, type_id):
        """Alias for var_type_to_string for backward compatibility"""
        return self.var_type_to_string(type_id)

    def var_type_to_string(self, var_type):
        """Convert a type ID to a string representation (without reference info)"""
        # Basic primitive types
        if var_type in TYPE_TO_STRING_MAP and not self.is_struct_type(var_type):
            return TYPE_TO_STRING_MAP[var_type]

        # Check descriptor system for struct and array types
        descriptor = self._type_descriptors.get(var_type)
        if descriptor:
            if descriptor.kind == self.TYPE_KIND_STRUCT:
                return self.get_struct_name(var_type) or "unknown_struct"

            elif descriptor.kind == self.TYPE_KIND_ARRAY:
                elem_type = descriptor.element_type_id
                size = descriptor.size
                elem_type_str = self.var_type_to_string(elem_type)  # Safe recursive call

                if size is not None:
                    return "%s[%s]" % (elem_type_str, size)
                else:
                    return "%s[]" % elem_type_str

        # Fall back to struct name lookup
        struct_name = self._struct_id_to_name.get(var_type)
        if struct_name:
            return struct_name

        raise CompilerException("unknown_type_%d" % var_type)

    def format_type_with_ref_kind(self, type_id, ref_kind=REF_KIND_NONE):
        """Format a type with its reference kind wrapper"""
        # Get the base type string
        type_str = self.var_type_to_string(type_id)
        # Add reference kind wrapper if needed
        if ref_kind == REF_KIND_HEAP:
            return "heap_ref<%s>" % type_str
        elif ref_kind == REF_KIND_STACK:
            return "stack_ref<%s>" % type_str
        return type_str

    # Type helpers for reference types
    def is_ref_type(self, type_):
        """Check if a type is a reference type"""
        return (type_ & REF_TYPE_FLAG) != 0

    def get_base_type(self, type_):
        """Get the base type of a reference type"""
        if self.is_ref_type(type_):
            return type_ & ~REF_TYPE_FLAG
        return type_

    def make_ref_type(self, type_):
        """Convert a type to its reference equivalent"""
        if self.is_ref_type(type_):
            return type_  # Already a reference
        return type_ | REF_TYPE_FLAG


# Singleton instance
_registry = None

def get_registry():
    """Get the global type registry instance"""
    global _registry
    if _registry is None:
        _registry = TypeRegistry()
    return _registry

def get_struct_name(type_id):
    return get_registry().get_struct_name(type_id)

