# Type registry for struct types
from shared import *
from ast_rewriter import AstTypeRewriter, AST_NODE_ALL

PRIMITIVE_TYPES = [
    TYPE_UNKNOWN, TYPE_VOID, TYPE_NIL, TYPE_STRING,
    TYPE_I8, TYPE_U8, TYPE_I16, TYPE_U16, TYPE_I32, TYPE_U32,
    TYPE_I64, TYPE_U64, TYPE_INT, TYPE_UINT, TYPE_LONG, TYPE_ULONG,
    TYPE_LONGLONG, TYPE_ULONGLONG, TYPE_FLOAT, TYPE_DOUBLE
]

# Generic type ID range
TYPE_GENERIC_BASE = 50  # Starting ID for generic parameters
TYPE_GENERIC_MAX = 99   # Maximum ID for generic parameters (49 possible parameters)

# Type descriptor classes
class TypeDescriptor(object):
    """Base descriptor for all types"""
    def __init__(self, kind, name=None):
        self.kind = kind
        self.name = name

class PrimitiveDescriptor(TypeDescriptor):
    """Descriptor for primitive types"""
    def __init__(self, primitive_type, name=None):
        TypeDescriptor.__init__(self, TypeRegistry.TYPE_KIND_PRIMITIVE, name)
        self.primitive_type = primitive_type

class StructDescriptor(TypeDescriptor):
    """Descriptor for struct types"""
    def __init__(self, name, parent_id=-1):
        TypeDescriptor.__init__(self, TypeRegistry.TYPE_KIND_STRUCT, name)
        self.parent_id = parent_id
        self.fields = []  # List of (name, type_id) tuples
        self.param_mapping = {}  # Maps generic param names to type IDs
        self.instantiations = {}  # Maps concrete type tuples to concrete struct IDs

class ArrayDescriptor(TypeDescriptor):
    """Descriptor for array types"""
    def __init__(self, element_type_id, size=None, name=None):
        TypeDescriptor.__init__(self, TypeRegistry.TYPE_KIND_ARRAY, name)
        self.element_type_id = element_type_id
        self.size = size  # None = dynamic size, int = fixed size

class Function:
    def __init__(self, name, return_type, params, parent_struct_id=-1, ast_node=None, is_ref_return=False):
        self.name = name  # Qualified name containing struct prefix for methods
        self.return_type = return_type
        self.is_ref_return = is_ref_return
        self.params = params  # List of (name, type, is_byref) tuples
        self.parent_struct_id = parent_struct_id  # -1 for global functions
        self.ast_node = ast_node  # Store AST nodes directly

class TypeRegistry:
    # Type kind constants
    TYPE_KIND_PRIMITIVE = 0
    TYPE_KIND_STRUCT = 1
    TYPE_KIND_TUPLE = 2
    TYPE_KIND_ENUM = 3
    TYPE_KIND_ARRAY = 4

    def __init__(self):
        """Initialize the registry"""
        self.reset()

    def reset(self):
        """Reset registry to initial state (for testing)"""
        # Type descriptor system
        self._type_descriptors = {}  # type_id -> TypeDescriptor
        self._type_name_to_id = {}    # name -> type_id
        self._next_struct_id = TYPE_STRUCT_BASE

        # Array type cache
        self._array_cache = {}  # (element_type_id, size) -> type_id

        # Function storage
        self._functions = {}    # funcid -> Function object
        self._next_funcid = 1   # Start with 1 so -1 means "not found"
        self._methods = {}      # (struct_id) -> dict(funcname) -> funcid

        # Initialize primitive types
        self._initialize_primitive_types()

    def _initialize_primitive_types(self):
        """Register all primitive types with descriptors"""
        for type_id in PRIMITIVE_TYPES:
            name = TYPE_TO_STRING_MAP[type_id]
            self._type_descriptors[type_id] = PrimitiveDescriptor(type_id, name)
            self._type_name_to_id[name] = type_id

        # Pre-register generic parameter IDs
        for i in range(TYPE_GENERIC_BASE, TYPE_GENERIC_MAX + 1):
            self._type_descriptors[i] = TypeDescriptor(TypeRegistry.TYPE_KIND_PRIMITIVE, "T%d"%(i-TYPE_GENERIC_BASE))

    def is_primitive_type(self, type_id):
        return type_id in PRIMITIVE_TYPES

    def get_tuple_prefix(self):
	return "__zw_tuple_"

    def get_enum_prefix(self):
	return "__zw_enum_"

    def is_tuple_type(self, type_id):
        descriptor = self._type_descriptors.get(type_id)
        return descriptor is not None and descriptor.kind == self.TYPE_KIND_TUPLE

    def is_enum_type(self, type_id):
        descriptor = self._type_descriptors.get(type_id)
        return descriptor is not None and descriptor.kind == self.TYPE_KIND_ENUM

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

        # Create debug name
        element_type_name = self.var_type_to_string(element_type_id)
        size_str = str(size) if size is not None else ""
        array_name = "_array_%s_%s" % (element_type_name, size_str)

        # Create descriptor
        descriptor = ArrayDescriptor(element_type_id, size, array_name)
        self._type_descriptors[type_id] = descriptor

        # Add to array cache
        self._array_cache[cache_key] = type_id

        return type_id

    def is_array_type(self, type_id):
        """Check if a type is an array"""
        # Get base type if this is a reference
        descriptor = self._type_descriptors.get(type_id)
        return descriptor and descriptor.kind == self.TYPE_KIND_ARRAY

    def get_array_element_type(self, array_type_id):
        """Get element type for an array"""
        descriptor = self._type_descriptors.get(array_type_id)
        if descriptor and descriptor.kind == self.TYPE_KIND_ARRAY:
            return descriptor.element_type_id
        return -1

    def get_array_size(self, array_type_id):
        """Get size for an array (None if dynamic)"""
        descriptor = self._type_descriptors.get(array_type_id)
        if descriptor and descriptor.kind == self.TYPE_KIND_ARRAY:
            return descriptor.size
        return -1

    # Struct methods
    def register_struct(self, name, parent_name=None, token=None, generic_params=None):
        """Register a new struct type, return its ID"""
        # Check if already registered
        if name in self._type_name_to_id:
            if token:
                raise CompilerException("Struct '%s' is already defined" % name, token)
            return self._type_name_to_id[name]

        # Get parent ID if specified
        parent_id = -1
        if parent_name:
            if parent_name not in self._type_name_to_id:
                if token:
                    raise CompilerException("Parent struct '%s' is not defined" % parent_name, token)
                return -1
            parent_id = self._type_name_to_id[parent_name]

        # Create new type ID
        type_id = self._next_struct_id
        self._next_struct_id += 1

        # Create descriptor for new system
        descriptor = StructDescriptor(name, parent_id)

        # if it's a tuple, mark it with a special descriptor kind
        # so we don't need to do lame string comparison over and over
        if name.startswith(self.get_tuple_prefix()):
            descriptor.kind = self.TYPE_KIND_TUPLE
        elif name.startswith(self.get_enum_prefix()):
            descriptor.kind = self.TYPE_KIND_ENUM

        self._type_descriptors[type_id] = descriptor

        # Set up generic parameters if provided
        if generic_params:
            param_index = 0
            for param_name in generic_params:
                param_id = TYPE_GENERIC_BASE + param_index
                descriptor.param_mapping[param_name] = param_id
                param_index += 1

        self._type_name_to_id[name] = type_id

        return type_id

    def _is_struct_descriptor(self, descriptor):
        return descriptor is not None and descriptor.kind in [self.TYPE_KIND_STRUCT, self.TYPE_KIND_TUPLE, self.TYPE_KIND_ENUM]

    def is_struct_type(self, type_id):
        """Check if a type is a struct type using descriptor information"""
        descriptor = self._type_descriptors.get(type_id)
        return self._is_struct_descriptor(descriptor)

    def is_generic_struct(self, struct_id):
        """Check if a struct is generic (has type parameters)"""
        descriptor = self._type_descriptors.get(struct_id)
        if not self._is_struct_descriptor(descriptor):
            return False
        return len(descriptor.param_mapping) > 0

    def is_generic_param(self, type_id):
        """Check if a type ID represents a generic parameter"""
        return TYPE_GENERIC_BASE <= type_id <= TYPE_GENERIC_MAX

    def resolve_generic_type(self, type_id, type_mapping):
        """Resolve a potentially generic type to a concrete type
        
        Args:
            type_id: The type ID to resolve, could be generic or concrete
            type_mapping: Dict mapping generic param IDs to concrete type IDs
            
        Returns:
            The concrete type ID corresponding to the input type
        """
        # Step 1: Direct replacement for generic parameters
        if type_id in type_mapping:
            return type_mapping[type_id]
        
        # Step 2: Handle array types that might contain generic elements
        if self.is_array_type(type_id):
            # Get element type and resolve it
            element_type = self.get_array_element_type(type_id)
            resolved_element = self.resolve_generic_type(element_type, type_mapping)
            
            # If element type changed, create new array type
            if resolved_element != element_type:
                # Get array size (if fixed)
                array_size = self.get_array_size(type_id)
                # Create new array with resolved element type
                return self.register_array_type(resolved_element, array_size)
        
        # Step 3: If we get here, it's not a generic type or already concrete
        return type_id

    def instantiate_generic_struct(self, generic_struct_id, concrete_types):
        """Create a concrete instance of a generic struct
        
        Args:
            generic_struct_id: The type ID of the generic struct template
            concrete_types: List of concrete type IDs to use for instantiation
            
        Returns:
            Type ID of the concrete instantiated struct
        """
        # Step 1: Validate the generic struct
        descriptor = self._type_descriptors.get(generic_struct_id)
        if not self._is_struct_descriptor(descriptor):
            return -1
        
        # Step 2: Check if it's actually a generic struct
        if len(descriptor.param_mapping) == 0:
            return generic_struct_id  # Not generic, return as is
        
        # Step 3: Ensure correct number of concrete types provided
        if len(concrete_types) != len(descriptor.param_mapping):
            return -1
        
        # Step 4: Create hashable key for caching
        concrete_tuple = tuple(concrete_types)
        
        # Step 5: Check cache for existing instantiation
        if concrete_tuple in descriptor.instantiations:
            return descriptor.instantiations[concrete_tuple]
            
        # Step 6: Generate concrete struct name
        generic_name = descriptor.name
        
        # Convert each concrete type to a string name
        type_names = []
        for type_id in concrete_types:
            type_names.append(self.var_type_to_string(type_id))
        
        concrete_name = "%s_%s" % (generic_name, "_".join(type_names))
        
        # Step 7: Create mapping from generic params to concrete types
        type_mapping = {}
        param_index = 0
        for param_name, param_id in descriptor.param_mapping.items():
            if param_index < len(concrete_types):
                type_mapping[param_id] = concrete_types[param_index]
            param_index += 1
        
        # Step 8: Create the concrete struct
        concrete_id = self.register_struct(concrete_name, parent_name=self.get_struct_name(descriptor.parent_id))
        
        # Step 9: Copy and transform fields
        field_index = 0
        while field_index < len(descriptor.fields):
            field_name, field_type = descriptor.fields[field_index]
            
            # Resolve field type if it's generic
            concrete_field_type = self.resolve_generic_type(field_type, type_mapping)
            self.add_field(concrete_name, field_name, concrete_field_type)
            
            field_index += 1
        
        # Step 10: Cache the instantiation
        descriptor.instantiations[concrete_tuple] = concrete_id
        
        # Step 11: Instantiate all methods for this concrete struct
        self._instantiate_all_methods(generic_struct_id, concrete_id, type_mapping)
        
        return concrete_id

    def instantiate_generic_method(self, method_name, generic_struct_id, concrete_struct_id, type_mapping=None):
        """Create a concrete method implementation from a generic method

        Args:
            method_name: Name of the method to instantiate
            generic_struct_id: Type ID of the generic struct template
            concrete_struct_id: Type ID of the concrete struct instance
            type_mapping: Optional mapping from generic type IDs to concrete type IDs

        Returns:
            ID of the concrete instantiated method
        """
        # Step 1: Find the generic method
        generic_method_id = self.lookup_function(method_name, generic_struct_id)
        if generic_method_id == -1:
            return -1

        # Step 2: Get the generic struct descriptor and method
        generic_descriptor = self._type_descriptors.get(generic_struct_id)
        if not self._is_struct_descriptor(generic_descriptor):
            return -1

        generic_method = self.get_func_from_id(generic_method_id)
        if generic_method is None:
            return -1

        # Step 3: Check if method is already instantiated
        existing_method_id = self.lookup_function(method_name, concrete_struct_id, check_parents=False)
        if existing_method_id != -1:
            return existing_method_id

        # Step 4: Get or create type mapping
        if type_mapping is None:
            # Get the concrete struct descriptor
            concrete_descriptor = self._type_descriptors.get(concrete_struct_id)
            if concrete_descriptor is None:
                return -1

            # Find which concrete types were used for instantiation by querying the registry
            # This is a proper O(1) lookup that doesn't rely on string parsing
            for concrete_tuple, instantiated_id in generic_descriptor.instantiations.items():
                if instantiated_id == concrete_struct_id:
                    # Found the matching instantiation
                    concrete_types = concrete_tuple

                    # Create the mapping from generic param IDs to concrete type IDs
                    type_mapping = {}
                    param_index = 0
                    for param_name, param_id in generic_descriptor.param_mapping.items():
                        if param_index < len(concrete_types):
                            type_mapping[param_id] = concrete_types[param_index]
                        param_index += 1
                    break

            # If we couldn't find a type mapping, this isn't a proper instantiation
            if type_mapping is None:
                return -1

        # Step 5: Transform return type
        concrete_return_type = self.resolve_generic_type(generic_method.return_type, type_mapping)

        # Step 6: Transform parameter types
        concrete_params = []
        param_index = 0
        while param_index < len(generic_method.params):
            param_name, param_type, is_byref = generic_method.params[param_index]

            # For 'self' parameter, use the concrete struct type directly
            if param_name == "self" and param_index == 0:
                concrete_params.append((param_name, concrete_struct_id, is_byref))
            else:
                concrete_param_type = self.resolve_generic_type(param_type, type_mapping)
                concrete_params.append((param_name, concrete_param_type, is_byref))

            param_index += 1

        # Step 7: Rewrite AST with all types exchanged
        complete_type_mapping = dict(type_mapping)  # Make a copy
        # Add mapping from generic struct ID to concrete struct ID
        complete_type_mapping[generic_struct_id] = concrete_struct_id

        rewriter = AstTypeRewriter(complete_type_mapping)
        concrete_ast_node = rewriter.rewrite(generic_method.ast_node, node_types=[AST_NODE_ALL])

        # Step 8: Register the new concrete function
        return self.register_function(
            method_name,
            concrete_return_type,
            concrete_params,
            concrete_struct_id,
            concrete_ast_node,
            generic_method.is_ref_return
        )

    def _instantiate_all_methods(self, generic_struct_id, concrete_struct_id, type_mapping):
        """Instantiate all methods from a generic struct for a concrete struct

        Args:
            generic_struct_id: ID of the generic struct template
            concrete_struct_id: ID of the concrete struct instance
            type_mapping: Dict mapping generic type IDs to concrete type IDs
        """
        # Find all methods defined for the generic struct
        generic_methods = []
        if generic_struct_id in self._methods:
            generic_methods = list(self._methods[generic_struct_id].keys())

        # For each method of the generic struct, instantiate a concrete version
        for method_name in generic_methods:
            self.instantiate_generic_method(
                method_name,
                generic_struct_id,
                concrete_struct_id,
                type_mapping
            )

    def get_generic_param_id(self, struct_id, param_name):
        """Get the type ID for a generic parameter by name

        Args:
            struct_id: The ID of the generic struct
            param_name: The name of the parameter (e.g. "K", "V")

        Returns:
            The type ID corresponding to the parameter, or -1 if not found
        """
        descriptor = self._type_descriptors.get(struct_id)
        if self._is_struct_descriptor(descriptor):
            return descriptor.param_mapping.get(param_name, -1)
        return -1

    def _get_type_id_from_name(self, type_name):
        """Helper to get type ID from name for both primitive types and structs
        Args:
            type_name: String name of a type
        Returns:
            Type ID or -1 if not found
        """
        if type_name in self._type_name_to_id:
            return self._type_name_to_id[type_name]

        return -1

    def _get_name_from_type_id(self, type_id):
        """Get the name of a type from its ID using the descriptor system"""
        desc = self._type_descriptors.get(type_id, None)
        return desc.name if desc else None

    def register_typedef(self, alias_name, target_type_id, token=None):
        """Register a type alias"""
        # Check if alias already exists
        if alias_name in self._type_name_to_id:
            if token:
                raise CompilerException("Type alias '%s' is already defined" % alias_name, token)
            return False

        # Add the alias to struct_registry, pointing to the same ID as the target
        self._type_name_to_id[alias_name] = target_type_id
        return True

    def add_field(self, struct_name, field_name, field_type, token=None):
        """Add a field to a struct definition"""
        type_id = self._type_name_to_id.get(struct_name, -1)

        if type_id == -1:
            if token:
                raise CompilerException("Struct '%s' is not defined" % struct_name, token)
            return False

        # Get the struct descriptor
        descriptor = self._type_descriptors.get(type_id)
        if not self._is_struct_descriptor(descriptor):
            if token:
                raise CompilerException("Type '%s' is not a struct" % struct_name, token)
            return False

        # Check for duplicate fields
        for name, _ in descriptor.fields:
            if name == field_name:
                if token:
                    raise CompilerException("Field '%s' is already defined in struct '%s'" % 
                                           (field_name, struct_name), token)
                return False

        # Add the field to the descriptor
        descriptor.fields.append((field_name, field_type))
        return True

    def get_struct_id(self, struct_name):
        """Get the type ID for a struct"""
        return self._get_type_id_from_name(struct_name)

    def get_struct_parent_id(self, struct_id):
        """Get the parent struct id from a struct id"""
        descriptor = self._type_descriptors.get(struct_id)
        if descriptor is None or descriptor.kind != self.TYPE_KIND_STRUCT:
            return -1
        return descriptor.parent_id

    def get_struct_parent(self, struct_name):
        """Get the parent struct name"""
        type_id = self._get_type_id_from_name(struct_name)
        if type_id == -1: return None
        parent_id = self.get_struct_parent_id(type_id)
        if parent_id == -1: return None
        return self._get_name_from_type_id(parent_id)

    def is_subtype_of(self, child_type, parent_type):
        """Check if child_type is a subtype of parent_type (same or inherits)"""
        # Same type is considered a subtype
        if child_type == parent_type:
            return True

        # Only struct types can have inheritance
        if not self.is_struct_type(child_type) or not self.is_struct_type(parent_type):
            return False

        # Follow the inheritance chain for the child
        current = child_type
        while True:
            current_parent = self.get_struct_parent_id(current)
            if current_parent == -1:
                # Reached top of hierarchy without finding parent
                return False
            if current_parent == parent_type:
                return True
            current = current_parent

    def get_struct_fields(self, type_id, include_parents=True):
        """
        Get fields for a struct type, optionally including parent fields

        Args:
            type_id: The struct type ID
            include_parents: Whether to include fields from parent structs
        Returns:
            List of (field_name, field_type) tuples in inheritance order
        """
        # Get descriptor for this struct
        descriptor = self._type_descriptors.get(type_id)
        if not self._is_struct_descriptor(descriptor):
            return []  # Not a struct type

        # Initialize result list
        result = []

        # If we should include parent fields and this struct has a parent
        if include_parents and descriptor.parent_id != -1:
            # First get all parent fields recursively
            parent_fields = self.get_struct_fields(descriptor.parent_id, True)
            result.extend(parent_fields)

        # Now add this struct's own fields
        result.extend(descriptor.fields)
        return result

    def get_field_type(self, struct_name, field_name):
        """Get the type of a field in a struct (including parent fields)"""
        # Use legacy system to get the struct ID
        if struct_name not in self._type_name_to_id:
            return None
        type_id = self._type_name_to_id[struct_name]
        for name, type_ in self.get_struct_fields(type_id, include_parents=True):
            if name == field_name:
                return type_
        return None

    def get_struct_name(self, type_id):
        """Get struct name from type ID"""
        descriptor = self._type_descriptors.get(type_id)
        return descriptor.name if descriptor else None

    def struct_exists(self, struct_name):
        """Check if a struct exists"""
        return struct_name in self._type_name_to_id

    # Function methods
    def register_function(self, name, return_type, params, parent_struct_id=-1, ast_node=None, is_ref_return=False):
        """Register a function, returning its ID"""
        func_id = self._next_funcid
        self._next_funcid += 1

        # Create qualified name for methods (for debugging)
        qualified_name = name
        if parent_struct_id != -1:
            struct_name = self.get_struct_name(parent_struct_id)
            qualified_name = "%s.%s" % (struct_name, name)

        func = Function(qualified_name, return_type, params, parent_struct_id, ast_node, is_ref_return=is_ref_return)
        self._functions[func_id] = func

        if parent_struct_id not in self._methods:
            self._methods[parent_struct_id] = {}
        self._methods[parent_struct_id][name] = func_id

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
        if struct_id in self._methods and name in self._methods[struct_id]:
            return self._methods[struct_id][name]

        # If not found and this is a struct, check parent structs
        if struct_id != -1 and check_parents:
            parent_id = self.get_struct_parent_id(struct_id)
            if parent_id != -1:
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
        self._methods = {}

    def var_type_to_string(self, var_type):
        """Convert a type ID to a string representation (without reference info)"""
        descriptor = self._type_descriptors.get(var_type)
        if not descriptor:
            raise CompilerException("unknown_type_%d" % var_type)
        return descriptor.name

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

