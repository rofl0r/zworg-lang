# C code generator
# Python 2.7 compatible

from shared import *
from type_registry import get_registry
from ast_flattener import AstExpressionFlattener
try:
    from StringIO import StringIO  # Python 2
except ImportError:
    from io import StringIO  # Python 3
import os

# import registry singleton
registry = get_registry()

# C runtime with typedefs and standard includes
c_runtime = """#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef unsigned char u8;
typedef signed char i8;
typedef unsigned short u16;
typedef signed short i16;
typedef unsigned int u32;
typedef signed int i32;
typedef unsigned long long u64;
typedef signed long long i64;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef long long longlong;
typedef unsigned long long ulonglong;
typedef char* string;

#include "handle_alloc.c"

/* Global handle allocator */
static struct handle_allocator ha;

/* Array helper functions and macros */
#ifdef ZWORG_BOUNDS_CHECK
static inline void* zw_array_element_ptr_internal(handle arr, size_t idx, size_t elem_size) {
    struct array_meta *meta = allocator_get_ptr(ha.allocators, arr.idx);
    if (idx * elem_size >= meta->len) {
        fprintf(stderr, "Array bounds check failed: index %zu out of bounds %zu\\n",
                idx, meta->len / elem_size);
        abort();
    }
    return &((char*)ha_array_get_ptr(&ha, arr))[idx * elem_size];
}

#define ZW_ARRAY_ACCESS(elem_type, arr_handle, index) \
    (*(elem_type*)zw_array_element_ptr_internal(arr_handle, index, sizeof(elem_type)))

#define ZW_ARRAY_ELEMENT_PTR(elem_type, arr_handle, index) \
    ((elem_type*)zw_array_element_ptr_internal(arr_handle, index, sizeof(elem_type)))
#else
#define ZW_ARRAY_ACCESS(elem_type, arr_handle, index) \
    (((elem_type*)ha_array_get_ptr(&ha, (arr_handle)))[(index)])

#define ZW_ARRAY_ELEMENT_PTR(elem_type, arr_handle, index) \
    (&((elem_type*)ha_array_get_ptr(&ha, (arr_handle)))[(index)])
#endif

#define ZW_ARRAY_LEN(elem_type, arr_handle) \
    (((struct array_meta*)allocator_get_ptr(ha.allocators, (arr_handle).idx))->len / sizeof(elem_type))

"""

main_header = """
int main(int argc, char **argv) {
	int stack_base;
	ha_init(&ha, &stack_base);
	/* ZW__HEADER_INJECT_GLOBALS */
	zw_main();
	return 0;
}
"""

# Map operators to C operators
C_OP_MAP = {
    '=': '=',
    '+': '+', '-': '-', '*': '*', '/': '/', '%': '%',
    # bits
    '&': '&', '|': '|', 'xor': '^',
    'shl': '<<', 'shr': '>>',
    # unary
    '-': '-', '!': '!', 'bitnot': '~',
    # comparison
    '==': '==', '!=': '!=', '<': '<', '<=': '<=', '>': '>', '>=': '>=',
    # logical
    'and': '&&', 'or': '||',
}

def make_reserved_identifier(name):
    """Transform an identifier into a reserved one by adding zw_ prefix"""
    return "zw_" + name

def is_byval_struct_type(type_id):
    return registry.is_tuple_type(type_id) or registry.is_enum_type(type_id)

def is_handle(type_id, ref_kind):
    """Determine if a type with given ref_kind should be treated as a handle"""
    # currently nil is internally (for simplicity) treated the same as
    # primitive types. we might revise this later.
    if type_id == TYPE_NIL: return True
    # Primitive types are never handles, regardless of ref_kind
    if registry.is_primitive_type(type_id): return False
    # Non-primitive types with not-NONE ref_kind are handles
    if is_byval_struct_type(type_id):
        assert(ref_kind == REF_KIND_NONE)
    return ref_kind != REF_KIND_NONE

def type_to_c(type_id, use_handles=False):
    """Convert type_id to a C type string."""
    if registry.is_primitive_type(type_id) or type_id == TYPE_STRING:
        # For primitive types, use the type name directly
        type_name = registry.var_type_to_string(type_id)
        return type_name

    elif registry.is_array_type(type_id) and registry.get_array_size(type_id) == 0:
        return "handle" # Dynamic arrays

    elif registry.is_struct_type(type_id) or registry.is_array_type(type_id):
        if is_byval_struct_type(type_id):
            use_handles = False
        if use_handles:
            return "handle"
        return "struct " + registry.get_struct_name(type_id)

    return "void"  # Default fallback

def unwrap_var(node):
    """unwrap a variablenode inside a new expr or a varnode itself"""
    if node.node_type == AST_NODE_NEW:
       return node.struct_init
    elif node.node_type == AST_NODE_VARIABLE:
       return node
    return None

def make_struct_decl(struct_id):
    """
    Generate a C struct definition from registry data
    Returns a string with the complete struct declaration
    """
    struct_name = registry.get_struct_name(struct_id)
    fields = registry.get_struct_fields(struct_id)
    result = "struct %s {\n" % struct_name

    for field_name, field_type in fields:
        field_c_type = type_to_c(field_type, use_handles=False)
        result += "\t%s %s;\n" % (field_c_type, field_name)

    if registry.is_array_type(struct_id):
        field_c_type = type_to_c(registry.get_array_element_type(struct_id))
        field_arr_dims = registry.get_array_size(struct_id)
        result += "\t%s %s[%d];\n" % (field_c_type, "data", field_arr_dims)

    result += "};\n\n"
    return result

def tuple_decl(type_id):
    return make_struct_decl(type_id)

def needs_tuple_decl(type_id):
    return registry.is_tuple_type(type_id) or (registry.is_array_type(type_id) and registry.get_array_size(type_id) != 0)

class VarLifecycle:
    """Track lifecycle information for a variable"""
    def __init__(self, type_id):
        self.type_id = type_id
        self.escapes = False  # True if assigned to struct member or passed as steal
        self.needs_storage = True # we don't need to emit an initializer expression if this is e.g. a direct result from func return
        self.static_data = None  # For initializers with static data (can apply to arrays or structs)
        self.heap_alloc = False # Whether the variable is allocated on heap anyway

    def __repr__(self):
        return "Var(%s)" % ("escapes" if self.escapes else "local")

class ScopeLevel:
    """Represents a single scope level with its variables and metadata"""
    def __init__(self, function_name, scope_id):
        self.function_name = function_name
        self.scope_id = scope_id
        self.var_dict = {}  # name -> VarLifecycle for fast lookup
        self.var_order = []  # List of names in declaration order

    def add_variable(self, name, var_lifecycle):
        """Add a variable to this scope level"""
        self.var_dict[name] = var_lifecycle
        self.var_order.append(name)

    def get_vars_in_order(self):
        """Get variables in declaration order as (name, lifecycle) tuples"""
        return [(name, self.var_dict[name]) for name in self.var_order]

class ScopeManager:
    """Manage scopes and variable lifecycles during code generation"""
    def __init__(self):
        self.reset()
        self.tuple_decls = {} # tuple_type_id: struct decl
        self.tuple_order = [] # tuple_id

    def reset(self):
        """Reset all scope tracking"""
        self.scope_stack = []  # Stack of ScopeLevel objects
        self.function_counters = {}  # {function_name: next_scope_id}

    def declare_tuple_type(self, tup_id, decl):
        if not self.is_tuple_processed(tup_id):
            if not tup_id in self.tuple_decls:
                self.tuple_order.append(tup_id)
            self.tuple_decls[tup_id] = decl

    def get_tuple_decls(self):
        result = ''
        for tid in self.tuple_order:
            if not self.is_tuple_processed(tid):
                result += self.tuple_decls[tid]
        return result

    def mark_tuple_processed(self, tup_id):
        self.declare_tuple_type(tup_id, "")

    # when we emitted tuple decls previous to a struct def using it,
    def is_tuple_processed(self, tup_id):
        return tup_id in self.tuple_decls and self.tuple_decls[tup_id] == ""

    def current_scope(self):
        """Get current scope or None if at global level"""
        if not self.scope_stack:
            return None
        return self.scope_stack[-1]

    def enter_scope(self, function_name):
        """Enter a new scope in the given function"""
        # Get next scope ID for this function
        if function_name not in self.function_counters:
            self.function_counters[function_name] = 0

        scope_id = self.function_counters[function_name]
        self.function_counters[function_name] += 1

        # Create new scope and push it to the stack
        new_scope = ScopeLevel(function_name, scope_id)
        self.scope_stack.append(new_scope)

        return new_scope

    def leave_scope(self, helper_header_output):
        """Leave current scope and generate both START and END macros"""
        if not self.scope_stack:
            return None

        scope = self.scope_stack.pop()
        return scope

    def add_variable(self, name, type_id):
        """Add a variable to the current scope"""
        current = self.current_scope()
        if not current:
            return None  # At global scope, not tracking

        var = VarLifecycle(type_id)
        current.add_variable(name, var)
        return var

    def find_variable(self, name):
        """Find variable in current or parent scopes - O(1) lookup"""
        # Search from innermost scope outward
        for scope in reversed(self.scope_stack):
            if name in scope.var_dict:
                return scope.var_dict[name]
        return None

    def set_static_data(self, name, static_data):
        """Set static data for initialization (arrays or structs)"""
        var = self.find_variable(name)
        var.static_data = static_data

    def mark_variable_escaping(self, name):
        """Mark a variable as escaping"""
        var = self.find_variable(name)
        if var:
            var.escapes = True

    def mark_variable_heap_alloc(self, name):
        """Mark a variable as requiring a heap allocation because user said so"""
        var = self.find_variable(name)
        if var:
            var.heap_alloc = True

    def mark_variable_no_storage(self, name):
        """Mark a variable not needing storage"""
        var = self.find_variable(name)
        if var:
            var.needs_storage = False

    def indent_level(self):
        """Get indent level based on scope depth"""
        return len(self.scope_stack)

class CCodeGenerator:
    def __init__(self):
        self.output = StringIO()
        self.helper_header = StringIO()
        self.indent_level = 0
        self.current_function = None # C name of the current func
        self.current_func_obj = None # Type registry func obj
        # Stack to track variables for testing
        self.test_printfs = []
        # Stack to track initializer context
        self.initializer_context = []  # Stack of (container_type, current_index) tuples
        self.global_initializers = []  # Store initializers for global handle-based structs
        self.scope_manager = ScopeManager()
        self.flattener = AstExpressionFlattener(registry)

    def indent(self):
        return '\t' * self.indent_level

    def needs_dereference_with_ref_kind(self, lhs_ref_kind, rhs_ref_kind):
        """
        Check dereferencing needs based on explicit reference kinds
        Same return values as needs_dereference.
        """
        # Basic dereferencing rules
        left_needs = lhs_ref_kind != REF_KIND_NONE and rhs_ref_kind == REF_KIND_NONE
        right_needs = lhs_ref_kind == REF_KIND_NONE and rhs_ref_kind != REF_KIND_NONE
        # Reference kind hierarchy rules
        # REF_KIND_STACK is lesser than REF_KIND_GENERIC and REF_KIND_HEAP
        if not left_needs:
            left_needs = (lhs_ref_kind == REF_KIND_GENERIC or lhs_ref_kind == REF_KIND_HEAP) and rhs_ref_kind == REF_KIND_STACK
        if not right_needs:
            right_needs = (rhs_ref_kind == REF_KIND_GENERIC or rhs_ref_kind == REF_KIND_HEAP) and lhs_ref_kind == REF_KIND_STACK

        if left_needs and right_needs:
            return 2  # Both sides need dereferencing
        elif left_needs:
            return -1  # Left side needs dereferencing
        elif right_needs:
            return 1  # Right side needs dereferencing
        else:
            return 0  # No dereferencing needed

    def needs_dereference(self, lhs_node, rhs_node):
        """
        Check dereferencing needs when combining LHS and RHS
        Returns:
          -1: Left side needs dereferencing
           0: No dereferencing needed
           1: Right side needs dereferencing
           2: Both sides need dereferencing
        """
        # Get the reference kinds
        lhs_ref_kind = lhs_node.ref_kind
        rhs_ref_kind = rhs_node.ref_kind

        # Special case for NEW node - look at struct_init
        if rhs_node.node_type == AST_NODE_NEW:
            rhs_ref_kind = rhs_node.struct_init.ref_kind

        return self.needs_dereference_with_ref_kind(lhs_ref_kind, rhs_ref_kind)

    def dereference(self, type_id, expr):
        """Generate code to dereference a type"""
        # we should never get a tuple type byref
        assert(not is_byval_struct_type(type_id))
        if registry.is_struct_type(type_id):
            c_type = type_to_c(type_id, use_handles=False)
            return '*((%s*)ha_obj_get_ptr(&ha, %s))' % (c_type, expr)
        return "*(%s)"%expr

    def push_initializer_context(self, container_type):
        """Push a new initializer context onto the stack"""
        self.initializer_context.append((container_type, 0))

    def pop_initializer_context(self):
        """Pop the current initializer context from the stack"""
        if self.initializer_context:
            return self.initializer_context.pop()
        return None

    def current_initializer_context(self):
        """Get the current initializer context or None"""
        if self.initializer_context:
            return self.initializer_context[-1]
        return None

    def advance_initializer_index(self):
        """Advance the current index in the initializer context"""
        if self.initializer_context:
            container_type, index = self.initializer_context[-1]
            self.initializer_context[-1] = (container_type, index + 1)

    def get_container_field_ref_kind(self, container_type):
        """Get the ref_kind for fields in a container type"""
        if registry.is_array_type(container_type) and registry.get_array_size(container_type) == 0:
            return REF_KIND_GENERIC
        return REF_KIND_NONE

    def generate_header(self):
        """Generate C typedefs and standard includes"""
        self.output.write(c_runtime)

    def generate(self, ast):
        """Generate C code from the AST"""
        self.generate_header()

        # First, collect all structs, functions and global variables
        functions = []
        globals = []
        structs = []

        for node in ast:
            if node.node_type == AST_NODE_FUNCTION_DECL:
                if not registry.is_generic_struct(node.parent_struct_id):
                    functions.append(node)
            elif node.node_type == AST_NODE_VAR_DECL:
                globals.append(node)
            elif node.node_type == AST_NODE_STRUCT_DEF:
                if not registry.is_generic_struct(node.struct_id):
                    structs.append(node)

        # Generate struct definitions first
        for struct in structs:
            self.generate_struct_def(struct.struct_id)

        # marker to insert tuples that need to come after struct defs
        self.output.write("\n/* ZW__HEADER_INJECT_TUPLES */\n")

        # flatten functions and modify constructor return values
        for i, func in enumerate(functions):
            functions[i] = self.rewrite_function(func)

        # Generate function prototypes
        for func in functions:
            self.generate_function_prototype(func)

        # Generate global variables
        for var_decl in globals:
            self.generate_global_var(var_decl)

        # create code and struct defs for concrete generics
        # TODO: these might not be emitted in the correct order
        self.generate_specializations()

        # Generate function definitions
        for func in functions:
            self.generate_function(func)

        # Generate the real C main function at the end
        self.generate_c_main()

        # Inject macro definitions into the output
        result = self.output.getvalue()

        marker = "/* ZW__HEADER_INJECT_TUPLES */"
        if marker in result:
            # need to patch in tuple struct types on the fly since they're not declared in the AST
            tup_str = self.scope_manager.get_tuple_decls()
            result = result.replace(marker, tup_str)

        marker = "/* ZW__HEADER_INJECT_GLOBALS */"
        if marker in result:
            result = result.replace(marker, '\n\t'.join(self.global_initializers))

        return result

    def generate_specializations(self):
        """Generate code for all generic specializations"""
        # Get all specializations in creation order
        for generic_id, concrete_id, concrete_tuple in registry.get_specializations():
            self.generate_struct_def(concrete_id)
            # Generate all methods using the get_struct_methods function
            for func_id in registry.get_struct_methods(concrete_id):
                func = registry.get_func_from_id(func_id)
                if func and func.ast_node:
                    # Flatten the method AST
                    flattened = self.rewrite_function(func.ast_node)
                    # Generate the method code
                    self.generate_function(flattened)

    def rewrite_function(self, node):
        """ decompose complex expressions in func body, apply other AST patches"""
        func_id = registry.lookup_function(node.name, node.parent_struct_id)
        func_obj = registry.get_func_from_id(func_id)
        # this call may modify is_ref_return, so it must come before hack, hack
        flat_func = self.flattener.flatten_function(func_obj.ast_node)
        # hack, hack: turn the func_obj into a pseudo-node with ref_kind attribute
        # so it can be used "as-if" a node in needs_dereference checks
        func_obj.ref_kind = REF_KIND_GENERIC if func_obj.is_ref_return else REF_KIND_NONE
        if os.getenv("DEBUG"): print(flat_func)
        # return rewritten FuncDecl
        return flat_func

    def generate_c_main(self):
        """Generate the C main function that initializes the runtime and calls zw_main"""
        self.output.write(main_header)

    def generate_struct_def(self, struct_id):
        """Generate C code for a struct definition"""
        # we first track all tuples we need, then emit those before the
        # the actual struct def, and mark them as processed so they dont
        # get declared again.
        tuples_needed = []
        result = 'struct %s {\n' % registry.get_struct_name(struct_id)
        fields = registry.get_struct_fields(struct_id, include_parents=True)

        # Generate struct members
        for field_name, field_type in fields:
            # deal with tuple structs we may find inside another struct
            if registry.is_tuple_type(field_type):
                if not self.scope_manager.is_tuple_processed(field_type):
                    tuples_needed.append(field_type)
                    # mark as processed
                    self.scope_manager.mark_tuple_processed(field_type)

            c_type = type_to_c(field_type, use_handles=False)
            result += '    %s %s;\n' % (c_type, field_name)

        result += '};\n\n'
        tup_str = ''
        for tid in tuples_needed:
            tup_str += tuple_decl(tid)

        self.output.write(tup_str + result);

    def generate_array_resize(self, node):
        """Generate C code for an array resize expression (new array allocator)"""
        # Generate the array and size expressions
        array_expr = self.generate_expression(node.array_expr)
        size_expr = self.generate_expression(node.size_expr)

        # Get the element type for the array
        elem_type_id = registry.get_array_element_type(node.array_expr.expr_type)
        element_type = type_to_c(elem_type_id, use_handles=False)

        # Calculate total size by multiplying element size with the requested count
        return "ha_array_realloc(&ha, %s, sizeof(%s) * %s)" % (array_expr, element_type, size_expr)

    def generate_array_access(self, node):
        """Generate C code for an array access expression"""
        array_expr = self.generate_expression(node.array)
        index_expr = self.generate_expression(node.index)

        # Get the element type for the array
        elem_type_id = registry.get_array_element_type(node.array.expr_type)
        # since an array access is always by-value, never use handles
        elem_type = type_to_c(elem_type_id, use_handles=False)
        if node.array.ref_kind != REF_KIND_NONE:
            # Generate array access with bounds checking
            return "ZW_ARRAY_ACCESS(%s, %s, %s)" % (elem_type, array_expr, index_expr)
        return "%s.data[%s]"%(array_expr, index_expr)

    def generate_var_decl(self, node, is_global=False):
        """Generate C code for a variable declaration (both local and global)"""
        var_type = type_to_c(node.var_type)
        is_struct = registry.is_struct_type(node.var_type)
        is_array = registry.is_array_type(node.var_type)

        # potentially need to inject tuple/array struct declaration
        if needs_tuple_decl(node.var_type):
            self.scope_manager.declare_tuple_type(node.var_type, tuple_decl(node.var_type))

        # in this context we want only by-ref struct types
        needs_handle = is_handle(node.var_type, node.ref_kind) or (node.expr and node.expr.node_type == AST_NODE_NEW)
        #is_handle_struct = registry.is_struct_type(node.var_type) and node.ref_kind != REF_KIND_NONE

        # Check if the variable is const (not TT_VAR)
        if node.decl_type != TT_VAR and not needs_handle:
            var_type = 'const ' + var_type

        init_expr = ''

        if node.expr:
            # Generate the expression
            expr_code = self.generate_expression(node.expr)

            # Add explicit cast if needed for float type conversion
            if node.var_type == TYPE_FLOAT and node.expr.expr_type == TYPE_DOUBLE:
                init_expr = '(float)(%s)' % expr_code
            else:
                init_expr = expr_code

        deref_needed = 0
        if node.expr:
            deref_needed = self.needs_dereference(node, node.expr)

        needs_storage = True

        # Register variable in scope manager
        if not is_global:
            self.scope_manager.add_variable(node.var_name, node.var_type)
            if needs_handle and node.expr:
                # if handle variable gets assigned result of another expression, we
                # dont need to allocate storage for it.
                # Only mark as not needing storage if BOTH sides have ref_kind != REF_KIND_NONE
                # In other words, we're just assigning handles
                if deref_needed == 0:
                    needs_storage = False
                    self.scope_manager.mark_variable_no_storage(node.var_name)
                if node.ref_kind == REF_KIND_HEAP:
                    self.scope_manager.mark_variable_heap_alloc(node.var_name)

        if init_expr:
            if not needs_handle and is_array:
                if deref_needed > 0:
                    assert(not is_global)
                    self.output.write(self.indent() + '%s %s;\n' % (var_type, node.var_name))
                    self.output.write(self.indent() + "memcpy(%s.data, ha_obj_get_ptr(&ha, %s), sizeof(%s.data));\n" % (
                        node.var_name, init_expr, node.var_name))
                else:
                    self.output.write(self.indent() + '%s %s = {%s};\n' % (var_type, node.var_name, init_expr))
            elif needs_handle and node.expr.expr_type == TYPE_NIL:
                if is_global:
                    self.output.write(self.indent() + 'handle %s;\n'%node.var_name)
                    self.global_initializers.append('%s = handle_nil;\n'%node.var_name)
                else:
                    self.output.write(self.indent() + 'handle %s = handle_nil;\n'%(node.var_name))
            elif needs_handle:
                assert(not is_global)
                skip_copy = False

                if not needs_storage:
                    self.output.write(self.indent() + 'handle %s = %s;\n'%(node.var_name, init_expr))
                    skip_copy = True
                elif is_struct:
                    self.output.write(self.indent() + 'handle %s = ha_obj_alloc(&ha, sizeof(%s));\n'%(node.var_name, var_type))
                elif is_array:
                    elem_type_id = registry.get_array_element_type(node.var_type)
                    elem_type_c = type_to_c(elem_type_id, use_handles=False)
                    array_size = registry.get_array_size(node.var_type)
                    self.output.write(self.indent() + 'handle %s = ha_array_alloc(&ha, sizeof(%s)*%d, (void*)0);\n'%(node.var_name, elem_type_c, array_size))
                    self.output.write(self.indent() + "memcpy(ha_obj_get_ptr(&ha, %s), &(%s[%d])%s, sizeof(%s));\n" % (node.var_name, elem_type_c, array_size, init_expr, var_type))
                    skip_copy = True
                elif registry.is_primitive_type(node.var_type) and node.ref_kind == REF_KIND_HEAP:
                    self.output.write(self.indent() + '%s *%s = realloc((void*)0, sizeof(%s));\n'%(var_type, node.var_name, var_type))
                    l_deref = '*' if deref_needed == -1 or deref_needed == 2 else ''
                    r_deref = '*' if deref_needed > 0 else ''
                    self.output.write(self.indent() + '%s%s = %s%s;\n'%(l_deref, node.var_name, r_deref, init_expr))
                    skip_copy = True
                else:
                    assert(0)
#                if deref_needed == -1 or deref_needed == 2:
#                    init_expr = self.dereference(node.expr.expr_type, node.var_name)
                if not skip_copy:
                    temp_code = init_expr
                    if deref_needed > 0:
                        temp_code = self.dereference(node.expr.expr_type, init_expr)
                    else:
                        temp_code = "&" + init_expr
                    self.output.write(self.indent() + "memcpy(ha_obj_get_ptr(&ha, %s), %s, sizeof(%s));\n" % (node.var_name, temp_code, var_type))
            else:
                self.output.write(self.indent() + '%s %s = %s;\n' % (var_type, node.var_name, init_expr))
        elif needs_handle:  # zw__temp_
            if is_struct:
                if node.ref_kind == REF_KIND_HEAP:
                    self.output.write(self.indent() + 'handle %s = ha_obj_alloc(&ha, sizeof(%s));\n'%(node.var_name, var_type))
                else:
                    storage = make_reserved_identifier("%s_storage" % node.var_name)
                    self.output.write(self.indent() + '%s %s;\n'%(var_type, storage))
                    self.output.write(self.indent() + 'handle %s = ha_stack_alloc(&ha, sizeof(%s), &%s);\n'%(node.var_name, var_type, storage))
            elif is_array:
                elem_type_id = registry.get_array_element_type(node.var_type)
                elem_type_c = type_to_c(elem_type_id, use_handles=False)
                array_size = registry.get_array_size(node.var_type)
                self.output.write(self.indent() + 'handle %s = ha_array_alloc(&ha, sizeof(%s)*%d, (void*)0);\n'%(node.var_name, elem_type_c, array_size))
        else:
            self.output.write(self.indent() + '%s %s;\n' % (var_type, node.var_name))

        if not needs_handle and (is_array or (is_struct and not is_byval_struct_type(node.var_type))):
            # create a handle for the object in case we need it later
            handle_name = make_reserved_identifier("%s_handle" % node.var_name)
            if not is_global:
                self.output.write(self.indent() + 'handle %s = ha_stack_alloc(&ha, sizeof(%s), &%s);\n'%
                    (handle_name, node.var_name, node.var_name))
            else:
                self.output.write(self.indent() + 'handle %s;\n'%(handle_name))
                self.global_initializers.append('struct raw_ptr_data %s_raw = {&%s};\n'%(handle_name, node.var_name))
                self.global_initializers.append('%s = ha_raw_handle(&ha, &%s_raw);\n'%(handle_name, handle_name))

        # Add test printf for main function or global variables
        # But only for primitive types and strings, not structs
        if not is_struct and not is_array and (self.current_function == 'main' or is_global):
            if needs_handle and not registry.is_primitive_type: pass
            # only add variables available in the main function scope
            elif is_global or self.scope_manager.indent_level() == 1:
                deref = ''
                if needs_handle and registry.is_primitive_type: deref='*'
                self.add_test_printf(deref+node.var_name, node.var_type)

    def generate_function_prototype_string(self, node):
        """Generate C function prototype string without ending ;"""
        func_name = self.get_method_name_from_node(node)
        func_id = registry.lookup_function(node.name, node.parent_struct_id)
        func_obj = registry.get_func_from_id(func_id)

        # tell scope manager about tuple types we encounter in the wild
        if needs_tuple_decl(node.return_type):
            self.scope_manager.declare_tuple_type(node.return_type, tuple_decl(node.return_type))

        # Generate return type
        ret_type = type_to_c(node.return_type, use_handles=registry.is_struct_type(node.return_type) and func_obj.is_ref_return)

        # Generate parameter list
        params = []
        for param in node.params:
            # Unpack the tuple (name, type, is_byref)
            param_name, param_type, is_byref = param

            c_type = type_to_c(param_type, use_handles=is_byref)

            if needs_tuple_decl(param_type):
                self.scope_manager.declare_tuple_type(param_type, tuple_decl(param_type))

            if registry.is_array_type(param_type):
                if not is_byref: c_type = registry.get_struct_name(param_type)

            # Handle byref params
            # Only add * for primitive types, not for handles
            # Since handles are already reference types
            if is_byref and not is_handle(param_type, REF_KIND_GENERIC):
                c_type += '*'
            params.append('%s %s' % (c_type, param_name))

        param_str = ', '.join(params) if params else 'void'

        # Function prototype
        if func_name == "main": func_name = make_reserved_identifier("main")
        return('%s %s(%s)' % (ret_type, func_name, param_str))

    def generate_function_prototype(self, node):
        self.output.write(self.generate_function_prototype_string(node) + ';\n')

    def generate_global_var(self, node):
        """Generate C code for a global variable"""
        self.generate_var_decl(node, is_global=True)

    def add_test_printf(self, var_name, type_id):
        """Add printf for testing variable value"""
        # Choose format string based on type
        cast = ''
        if type_id == TYPE_STRING:
            fmt = '%s'
        elif is_float_type(type_id):
            fmt = '%.16f'
        else:
            fmt = '%jd'
            cast = '(intmax_t)'
        deref = ''
        if var_name.startswith('*'):
            deref = '*'
            var_name = var_name[1:]
        self.test_printfs.append('dprintf(99, "%s:%s\\n", %s%s%s);' % (var_name, fmt, cast, deref, var_name))

    def generate_function(self, node):
        """Generate C code for a function definition"""
        # Function signature
        self.output.write("%s {\n"%self.generate_function_prototype_string(node))

        # make C-compatible method name
        func_name = self.get_method_name_from_node(node)
        self.current_function = func_name

        # Reset scope manager for this function
        self.scope_manager.reset()

        # Clear test printfs for this function
        self.test_printfs = []

        # Get function body from registry
        func_id = registry.lookup_function(node.name, node.parent_struct_id)
        func_obj = registry.get_func_from_id(func_id)
        self.current_func_obj = func_obj
        body = func_obj.ast_node.body

        # Function body
        self.indent_level += 1
        self.scope_manager.enter_scope(func_name)

        for stmt in body:
            self.generate_statement(stmt)

        # For main function, add test printfs
        if func_name == 'main':
            for printf_stmt in self.test_printfs:
                self.output.write(self.indent() + printf_stmt + '\n')

        # Generate scope end macro and leave scope
        scope = self.scope_manager.leave_scope(self.helper_header)
        self.indent_level -= 1

        # Close function
        self.output.write('}\n')
        self.current_function = None
        self.current_func_obj = None

    def generate_statement(self, node):
        """Generate C code for a statement"""
        if node.node_type == AST_NODE_EXPR_STMT:
            self.output.write(self.indent())
            self.output.write('%s;\n' % self.generate_expression(node.expr))

        elif node.node_type == AST_NODE_VAR_DECL:
            self.generate_var_decl(node, is_global=False)

        elif node.node_type == AST_NODE_RETURN:
            self.output.write(self.indent())
            if node.expr:
                # Use current_func_obj as pseudo-node to check if return needs dereferencing
                deref_needed = self.needs_dereference(self.current_func_obj, node.expr)
                expr_code = self.generate_expression(node.expr)
                if registry.is_struct_type(node.expr.expr_type) and deref_needed != 0:
                    self.output.write('return %s;\n' % self.dereference(node.expr.expr_type, expr_code))
                else:
                    self.output.write('return %s;\n' % expr_code)
                    if node.expr.node_type == AST_NODE_VARIABLE and self.current_func_obj.is_ref_return:
                        self.scope_manager.mark_variable_escaping(node.expr.name)
            else:
                self.output.write('return;\n')

        elif node.node_type == AST_NODE_IF:
            self.generate_if_statement(node)

        elif node.node_type == AST_NODE_WHILE:
            self.generate_while_statement(node)

        elif node.node_type == AST_NODE_BREAK:
            self.output.write(self.indent() + 'break;\n')

        elif node.node_type == AST_NODE_CONTINUE:
            self.output.write(self.indent() + 'continue;\n')

    def generate_if_statement(self, node):
        """Generate C code for an if statement"""
        condition = self.generate_expression(node.condition)

        # Special case for handle variables used as conditions
        if node.condition.node_type == AST_NODE_VARIABLE and is_handle(node.condition.expr_type, node.condition.ref_kind):
            condition = "(handle_cmp(%s, handle_nil))" % condition

        # Write if condition
        self.output.write(self.indent() + 'if (%s) {\n' % condition)
        self.indent_level += 1
        scope = self.scope_manager.enter_scope(self.current_function)

        # Generate then body
        for stmt in node.then_body:
            self.generate_statement(stmt)

        scope = self.scope_manager.leave_scope(self.helper_header)
        self.indent_level -= 1

        # Generate else part if it exists
        if node.else_body:
            self.output.write(self.indent() + '} else {\n')
            self.indent_level += 1
            scope = self.scope_manager.enter_scope(self.current_function)

            for stmt in node.else_body:
                self.generate_statement(stmt)

            # Generate scope end macro and leave scope
            scope = self.scope_manager.leave_scope(self.helper_header)
            self.indent_level -= 1

        self.output.write(self.indent() + '}\n')

    def generate_while_statement(self, node):
        """Generate C code for a while loop"""
        condition = self.generate_expression(node.condition)

        # Special case for handle variables used as conditions
        if node.condition.node_type == AST_NODE_VARIABLE and is_handle(node.condition.expr_type, node.condition.ref_kind):
            condition = "(handle_cmp(%s, handle_nil))" % condition

        # Write while header
        self.output.write(self.indent() + 'while (%s) {\n' % condition)
        self.indent_level += 1
        scope = self.scope_manager.enter_scope(self.current_function)

        # Generate loop body
        for stmt in node.body:
            self.generate_statement(stmt)

        # Generate scope end macro and leave scope
        scope = self.scope_manager.leave_scope(self.helper_header)
        self.indent_level -= 1
        self.output.write(self.indent() + '}\n')

    def generate_generic_initializer(self, node):
        """Generate code for a generic initializer node (structs, arrays, tuples)"""
        # tell scope manager about tuple types we encounter in the wild
        if needs_tuple_decl(node.expr_type):
            self.scope_manager.declare_tuple_type(node.expr_type, tuple_decl(node.expr_type))

        # Push initializer context before we start processing elements
        self.push_initializer_context(node.expr_type)

        # Get type information for casting
        type_cast = ""
        if registry.is_struct_type(node.expr_type):
            struct_string = type_to_c(node.expr_type, use_handles=False)
            type_cast = "(%s) " % struct_string

        # Start the initializer
        result = "%s{"%type_cast

        if node.subtype == INITIALIZER_SUBTYPE_LINEAR and registry.is_array_type(node.expr_type):
            # For arrays, output elements in order
            for i, elem in enumerate(node.elements):
                if i > 0:
                    result += ", "
                result += self.generate_expression(elem)
                self.advance_initializer_index()

            # If this is a fixed-size array, output zeros for missing elements
            array_size = registry.get_array_size(node.expr_type)
            if array_size != 0 and array_size > len(node.elements):
                if len(node.elements) > 0:
                    result += ", "

                # Add zeros for remaining elements
                element_type = registry.get_array_element_type(node.expr_type)
                for i in range(len(node.elements), array_size):
                    if i > len(node.elements):
                        result += ", "
                    # Output appropriate default value based on element type
                    if element_type == TYPE_INT or element_type == TYPE_CHAR:
                        result += "0"
                    elif element_type == TYPE_FLOAT:
                        result += "0.0"
                    elif registry.is_struct_type(element_type):
                        # For struct elements, output nested initializer with all zeros
                        result += "{0}"
                    else:
                        result += "NULL"

        elif node.subtype == INITIALIZER_SUBTYPE_LINEAR:
            # For struct initializers, output elements in order
            for i, elem in enumerate(node.elements):
                if i > 0:
                    result += ", "
                result += self.generate_expression(elem)
                self.advance_initializer_index()

        elif node.subtype == INITIALIZER_SUBTYPE_NAMED:
            # Reserved for future C99-style named initializers
            result += "/* Named initializers not yet implemented */"

        # Close the initializer
        result += "}"
        # Pop initializer context when done
        self.pop_initializer_context()
        return result

    def get_method_name(self, struct_name, func_name):
        if struct_name:
            func_name = "%s_%s" % (struct_name, func_name)
        return func_name

    def get_method_name_from_node(self, node):
        func_name = node.name
        struct_name = None

        # Check if this is a method call
        if node.node_type == AST_NODE_CALL:
            struct_id = -1
            if node.obj: struct_id = node.obj.expr_type
            func_id = registry.lookup_function(func_name, struct_id, check_parents = node.obj is not None)
            func_obj = registry.get_func_from_id(func_id)
            # get the struct that implements the method
            struct_id = func_obj.parent_struct_id
            if struct_id != -1: struct_name = registry.get_struct_name(struct_id)
        elif node.node_type == AST_NODE_FUNCTION_DECL:
                if node.parent_struct_id != -1:
                    struct_name = registry.get_struct_name(node.parent_struct_id)
        else:
            assert(0)
        return self.get_method_name(struct_name, func_name)


    def generate_expression(self, node):
        """Generate C code for an expression"""
        if node is None:
            return ""

        if node.node_type == AST_NODE_NUMBER:
            # Add suffix for specific literal types
            if hasattr(node, 'literal_suffix') and node.literal_suffix:
                return str(node.value) + node.literal_suffix
            return str(node.value)

        elif node.node_type == AST_NODE_STRING:
            # Escape string literal
            escaped = node.value.replace('"', '\\"').replace('\n', '\\n')
            return '"%s"' % escaped

        elif node.node_type == AST_NODE_NEW:
            # The only action needed is to forward to the struct initializer
            # The ref_kind is already set to REF_KIND_HEAP
            assert node.ref_kind == REF_KIND_HEAP  # Sanity check
            return self.generate_expression(node.struct_init)

        elif node.node_type == AST_NODE_VARIABLE:
            # Special case for initializers - check if we're in a container context
            current_context = self.current_initializer_context()
            if current_context:
                container_type, index = current_context

                # Get the appropriate ref_kind for fields in this container
                field_ref_kind = self.get_container_field_ref_kind(container_type)

                # Check if we need to dereference based on ref_kinds
                deref_needed = self.needs_dereference_with_ref_kind(field_ref_kind, node.ref_kind)
                if deref_needed == 1 or deref_needed == 2:  # Right side needs dereferencing
                    return self.dereference(node.expr_type, node.name)

            return node.name

        elif node.node_type == AST_NODE_BINARY_OP:
            left = self.generate_expression(node.left)
            right = self.generate_expression(node.right)
            op = C_OP_MAP.get(node.operator, node.operator)
            deref_needed = self.needs_dereference(node.left, node.right)
            if op == '=':
                if node.right.node_type == AST_NODE_VARIABLE and node.left.node_type == AST_NODE_MEMBER_ACCESS:
                    var_name = node.right.name
                    self.scope_manager.mark_variable_escaping(var_name)
                elif node.right.node_type == AST_NODE_GENERIC_INITIALIZER:
                    # For arrays, use compound literal and memcpy
                    if registry.is_array_type(node.left.expr_type):
                        # Get array info
                        elem_type_id = registry.get_array_element_type(node.left.expr_type)
                        elem_type = type_to_c(elem_type_id, use_handles=False)
                        array_size = registry.get_array_size(node.left.expr_type)

                        # Generate initializer directly
                        init_expr = self.generate_expression(node.right)

                        if node.left.ref_kind == REF_KIND_NONE:
                            return "(memcpy(&%s, (%s[%d])%s, sizeof(%s) * %d), %s)" % (
                                left, elem_type, array_size, init_expr, elem_type, array_size, left)
                        else:
                            # Use compound literal with memcpy for the assignment
                            return "(memcpy(ha_array_get_ptr(&ha, %s), (%s[%d])%s, sizeof(%s) * %d), %s)" % (
                                left, elem_type, array_size, init_expr, elem_type, array_size, left)

                    # For struct types, use compound literal and memcpy
                    elif registry.is_struct_type(node.left.expr_type) and not is_byval_struct_type(node.left.expr_type):
                        struct_type = type_to_c(node.left.expr_type, use_handles=False)

                        # Generate initializer directly
                        init_expr = self.generate_expression(node.right)

                        # Check if we're assigning to an array element
                        if node.left.node_type == AST_NODE_ARRAY_ACCESS:
                            # Array element assignment needs direct element pointer
                            array_expr = self.generate_expression(node.left.array)
                            index_expr = self.generate_expression(node.left.index)

                            # Get the element type for the array
                            elem_type = type_to_c(node.left.expr_type, use_handles=False)

                            if node.left.array.ref_kind == REF_KIND_NONE:
                                # Use ZW_ARRAY_ELEMENT_PTR to get pointer to the array element
                                return "(memcpy(&%s.data[%s], &%s, sizeof(%s)), %s)" % (
                                    array_expr, index_expr, init_expr, struct_type, left)
                            else:
                                # Use ZW_ARRAY_ELEMENT_PTR to get pointer to the array element
                                return "(memcpy(ZW_ARRAY_ELEMENT_PTR(%s, %s, %s), &(%s)%s, sizeof(%s)), %s)" % (
                                    elem_type, array_expr, index_expr, struct_type, init_expr, struct_type, left)

                        if node.left.ref_kind == REF_KIND_NONE:
                            return "(memcpy(&%s, &%s, sizeof(%s)), %s)" % (
                                left, init_expr, struct_type, left)
                        else:
                            # Use properly typed compound literal with memcpy for the assignment
                            return "(memcpy(ha_obj_get_ptr(&ha, %s), &%s, sizeof(%s)), %s)" % (
                                left, init_expr, struct_type, left)

                elif registry.is_array_type(node.right.expr_type) and deref_needed > 0:
                    assert(registry.is_array_type(node.left.expr_type))
                    if node.left.ref_kind != REF_KIND_NONE and node.right.ref_kind != REF_KIND_NONE:
                         return "(ha_array_copy(&ha, %s, %s), %s)" % (left, right, left)
                    elif node.left.ref_kind == REF_KIND_NONE:
                         return "(memcpy(%s.data, ha_obj_get_ptr(&ha, %s), sizeof(%s.data)), %s)" % (left, right, left, left)
                    else:
                         assert(0)

            if deref_needed == -1 or deref_needed == 2:
                left = self.dereference(node.left.expr_type, left)
            if deref_needed > 0:
                right = self.dereference(node.right.expr_type, right)

            return '(%s %s %s)' % (left, op, right)

        elif node.node_type == AST_NODE_UNARY_OP:
            operand = self.generate_expression(node.operand)
            op = C_OP_MAP.get(node.operator, node.operator)
            return '%s(%s)' % (op, operand)

        elif node.node_type == AST_NODE_CALL:
            func_name = self.get_method_name_from_node(node)
            func_id = registry.lookup_function(node.name, node.obj.expr_type if node.obj else -1)
            func_obj = registry.get_func_from_id(func_id)
            args = []
            is_constructor = node.obj and node.name == "init"
            for i, arg in enumerate(node.args):
                param_name, param_type, is_byref = func_obj.params[i]
                arg_varnode = unwrap_var(arg)
                # if this is the self parameter of a method, and it's an array access
                if i == 0 and node.obj and arg.node_type == AST_NODE_ARRAY_ACCESS:
                    elem_type_id = registry.get_array_element_type(arg.array.expr_type)
                    array_expr = self.generate_expression(arg.array)
                    index_expr = self.generate_expression(arg.index)
                    # Use compound literal to create a direct handle to the array element
                    c_type = type_to_c(elem_type_id, use_handles=False)
                    handle_name = array_expr
                    if arg.array.ref_kind == REF_KIND_NONE:
                         handle_name = make_reserved_identifier("%s_handle" % array_expr)
                    handle_expr = "ha_array_elem_handle(&ha, &(struct array_elem_handle_data){%s, %s * sizeof(%s)})" % (
                        handle_name, index_expr, c_type)
                    args.append(handle_expr)
                    continue
                if arg_varnode and is_byref and arg.ref_kind == REF_KIND_NONE and is_handle(param_type, REF_KIND_GENERIC):
                    args.append(make_reserved_identifier("%s_handle" % arg.name))
                elif arg_varnode and not is_byref and arg.ref_kind != REF_KIND_NONE and is_handle(param_type, REF_KIND_GENERIC):
                    args.append(self.dereference(param_type, self.generate_expression(arg)))
                else:
                    args.append(self.generate_expression(arg))
                if arg.node_type == AST_NODE_VARIABLE:
                    # never mark self in constructor calls as escaping
                    if is_constructor and i == 0: pass
                    # never mark temp vars inserted by ast_flattener as escaping
                    elif arg.name.startswith("zw__temp_"): pass
                    else:
                        self.scope_manager.mark_variable_escaping(arg.name)

            call_expr = '%s(%s)' % (func_name, ', '.join(args))
            # Special case for initializers - check if we're in a container context
            current_context = self.current_initializer_context()
            if current_context and node.ref_kind != REF_KIND_NONE:
                container_type, index = current_context

                # Get the appropriate ref_kind for fields in this container
                field_ref_kind = self.get_container_field_ref_kind(container_type)

                # Check if we need to dereference based on ref_kinds
                if self.needs_dereference_with_ref_kind(field_ref_kind, node.ref_kind) > 0:
                    return self.dereference(node.expr_type, call_expr)

            return call_expr

        elif node.node_type == AST_NODE_COMPARE:
            left = self.generate_expression(node.left)
            right = self.generate_expression(node.right)
            op = C_OP_MAP.get(node.operator, node.operator)
            if (is_handle(node.left.expr_type, node.left.ref_kind) and
                is_handle(node.right.expr_type, node.right.ref_kind)):
                assert(op in ["==", "!="])
                if op == '==': return "(!handle_cmp(%s, %s))"%(left, right)
                if op == '!=': return "(handle_cmp(%s, %s))"%(left, right)
            return '(%s %s %s)' % (left, op, right)

        elif node.node_type == AST_NODE_BITOP:
            left = self.generate_expression(node.left)
            right = self.generate_expression(node.right)
            op = C_OP_MAP.get(node.operator, node.operator)
            return '(%s %s %s)' % (left, op, right)

        elif node.node_type == AST_NODE_LOGICAL:
            left = self.generate_expression(node.left)
            right = self.generate_expression(node.right)
            op = C_OP_MAP.get(node.operator, node.operator)
            return '(%s %s %s)' % (left, op, right)

        elif node.node_type == AST_NODE_MEMBER_ACCESS:
            # Generate member access expression
            expr = self.generate_expression(node.obj)
            if node.obj.ref_kind == REF_KIND_NONE:
                return "%s.%s"%(expr, node.member_name)
            struct_string = type_to_c(node.obj.expr_type, use_handles=False)
            return '((%s*)(ha_obj_get_ptr(&ha, %s)))->%s' % (struct_string, expr, node.member_name)

        elif node.node_type == AST_NODE_GENERIC_INITIALIZER:
            return self.generate_generic_initializer(node)

        elif node.node_type == AST_NODE_ARRAY_ACCESS:
            return self.generate_array_access(node)

        elif node.node_type == AST_NODE_ARRAY_RESIZE:
            return self.generate_array_resize(node)

        elif node.node_type == AST_NODE_NIL:
            return "handle_nil"

        return "/* Unknown expression */"

def generate_c_code(ast):
    """Entry point function to generate C code from an AST"""
    generator = CCodeGenerator()
    return generator.generate(ast)
