# C code generator
# Python 2.7 compatible

from shared import *
from type_registry import get_registry
from ast_flattener import AstExpressionFlattener
try:
    from StringIO import StringIO  # Python 2
except ImportError:
    from io import StringIO  # Python 3

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

/* Variable declaration and cleanup code */
/* ZW__HEADER_INJECT_MACROS */

"""

main_header = """
int main(int argc, char **argv) {
	int stack_base;
	ha_init(&ha, &stack_base);
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

def type_to_c(type_id, use_handles=True):
    """Convert type_id to a C type string"""
    if registry.is_primitive_type(type_id):
        # For primitive types, use the type name directly
        type_name = registry.var_type_to_string(type_id)
        return type_name

    elif registry.is_array_type(type_id):
        # For arrays, get element type and make it a pointer
        elem_type_id = registry.get_array_element_type(type_id)
        elem_type = type_to_c(elem_type_id)
        return "%s*" % elem_type

    elif registry.is_struct_type(type_id):
        if is_byval_struct_type(type_id):
            use_handles = False
        if use_handles:
            return "handle"
        return "struct " + registry.get_struct_name(type_id)

    return "void"  # Default fallback

def tuple_decl(type_id):
    fields = registry.get_struct_fields(type_id)
    result = "struct %s {"%registry.get_struct_name(type_id)
    for field_name, field_type in fields:
        field_c_type = type_to_c(field_type, use_handles=False)
        result += "%s %s; " % (field_c_type, field_name)
    result += "};\n"
    return result

class VarLifecycle:
    """Track lifecycle information for a variable"""
    def __init__(self, type_id, is_struct=False):
        self.type_id = type_id
        self.is_struct = is_struct  # Whether this is a struct type
        self.escapes = False  # True if assigned to struct member or passed as steal
        self.needs_storage = True # we don't need to emit an initializer expression if this is e.g. a direct result from func return

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

    def get_macro_name(self, prefix):
        """Get the macro name for this scope"""
        return "%s_%s_%d" % (prefix, self.function_name, self.scope_id)

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

        # Generate START macro with all variable declarations
        start_macro_name = scope.get_macro_name("ZW_SCOPE_START")
        helper_header_output.write("#define %s \\\n" % start_macro_name)

        # Add variable declarations - only allocate storage for variables that need it
        for name, var in scope.get_vars_in_order():
            if var.is_struct and var.needs_storage:
                if var.escapes:
                    # Heap allocation for escaping variables
                    struct_string = type_to_c(var.type_id, use_handles=False)
                    helper_header_output.write("\thandle %s = ha_obj_alloc(&ha, sizeof(%s)); \\\n" %
                                            (name, struct_string))
                else:
                    # Stack allocation for non-escaping variables
                    storage_name = make_reserved_identifier("%s_storage" % name)
                    helper_header_output.write("\t%s %s = {0}; \\\n" % (type_to_c(var.type_id, use_handles=False), storage_name))
                    helper_header_output.write("\thandle %s = ha_stack_alloc(&ha, sizeof(%s), &%s); \\\n" %
                                            (name, storage_name, storage_name))
            elif var.is_struct:
                # Just declare handle for variables getting value from elsewhere
                helper_header_output.write("\thandle %s; \\\n" % name)

        helper_header_output.write("\n")

        # Generate END macro with cleanup code
        end_macro_name = scope.get_macro_name("ZW_SCOPE_END")
        helper_header_output.write("#define %s \\\n" % end_macro_name)

        # Add cleanup code for struct variables in reverse declaration order
        cleanup_needed = False
        for name, var in reversed(scope.get_vars_in_order()):
            if var.is_struct and var.escapes and var.needs_storage:
                helper_header_output.write("\tha_obj_free(&ha, %s); \\\n" % name)
                cleanup_needed = True

        helper_header_output.write("\n")

        return scope

    def add_variable(self, name, type_id, is_struct=False):
        """Add a variable to the current scope"""
        current = self.current_scope()
        if not current:
            return None  # At global scope, not tracking

        var = VarLifecycle(type_id, is_struct)
        current.add_variable(name, var)
        return var

    def find_variable(self, name):
        """Find variable in current or parent scopes - O(1) lookup"""
        # Search from innermost scope outward
        for scope in reversed(self.scope_stack):
            if name in scope.var_dict:
                return scope.var_dict[name]
        return None

    def mark_variable_escaping(self, name):
        """Mark a variable as escaping"""
        var = self.find_variable(name)
        if var:
            var.escapes = True

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
        self.scope_manager = ScopeManager()
        self.flattener = AstExpressionFlattener(registry)

    def indent(self):
        return '\t' * self.indent_level

    def needs_dereference_with_ref_kind(self, lhs_ref_kind, rhs_ref_kind):
        """
        Check dereferencing needs based on explicit reference kinds
        Same return values as needs_dereference.
        """
        left_needs = lhs_ref_kind != REF_KIND_NONE and rhs_ref_kind == REF_KIND_NONE
        right_needs = lhs_ref_kind == REF_KIND_NONE and rhs_ref_kind != REF_KIND_NONE

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
            struct_string = type_to_c(type_id, use_handles=False)
            return '*((%s*)ha_obj_get_ptr(&ha, %s))' % (struct_string, expr)
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
        # All structs that use handles have ref_kind != REF_KIND_NONE
        # Tuples and other value types have REF_KIND_NONE
        if registry.is_struct_type(container_type) and not is_byval_struct_type(container_type):
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
                functions.append(node)
            elif node.node_type == AST_NODE_VAR_DECL:
                globals.append(node)
            elif node.node_type == AST_NODE_STRUCT_DEF:
                structs.append(node)

        # Generate struct definitions first
        for struct in structs:
            self.generate_struct_def(struct)

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

        # Generate function definitions
        for func in functions:
            self.generate_function(func)

        # Generate the real C main function at the end
        self.generate_c_main()

        # Inject macro definitions into the output
        result = self.output.getvalue()
        marker = "/* ZW__HEADER_INJECT_MACROS */"
        if marker in result:
            result = result.replace(marker, self.helper_header.getvalue())

        marker = "/* ZW__HEADER_INJECT_TUPLES */"
        if marker in result:
            # need to patch in tuple struct types on the fly since they're not declared in the AST
            tup_str = self.scope_manager.get_tuple_decls()
            result = result.replace(marker, tup_str)

        return result

    def rewrite_function(self, node):
        """ decompose complex expressions in func body, apply other AST patches"""
        func_id = registry.lookup_function(node.name, node.parent_struct_id)
        func_obj = registry.get_func_from_id(func_id)
        # this call may modify is_ref_return, so it must come before hack, hack
        flat_func = self.flattener.flatten_function(func_obj.ast_node)
        # hack, hack: turn the func_obj into a pseudo-node with ref_kind attribute
        # so it can be used "as-if" a node in needs_dereference checks
        func_obj.ref_kind = REF_KIND_GENERIC if func_obj.is_ref_return else REF_KIND_NONE
        # return rewritten FuncDecl
        return flat_func

    def generate_c_main(self):
        """Generate the C main function that initializes the runtime and calls zw_main"""
        self.output.write(main_header)

    def generate_struct_def(self, node):
        """Generate C code for a struct definition"""
        # we first track all tuples we need, then emit those before the
        # the actual struct def, and mark them as processed so they dont
        # get declared again.
        tuples_needed = []
        result = 'struct %s {\n' % node.name

        # Generate struct members
        for field_name, field_type in node.fields:
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

    def generate_var_decl(self, node, is_global=False):
        """Generate C code for a variable declaration (both local and global)"""
        var_type = type_to_c(node.var_type)

        # potentially need to inject tuple declaration
        if registry.is_tuple_type(node.var_type):
            self.scope_manager.declare_tuple_type(node.var_type, tuple_decl(node.var_type))

        # in this context we want is_struct to exclude byval struct type
        is_struct = registry.is_struct_type(node.var_type) and not is_byval_struct_type(node.var_type)

        # Check if the variable is const (not TT_VAR)
        if node.decl_type != TT_VAR:
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

        # Register variable in scope manager
        if not is_global:
            self.scope_manager.add_variable(node.var_name, node.var_type, is_struct=is_struct)
            if is_struct and not is_byval_struct_type(node.var_type) and node.expr:
                # if variable gets assigned result of another expression, we
                # dont need to allocate storage for it.
                # Only mark as not needing storage if BOTH sides have ref_kind != REF_KIND_NONE
                # In other words, we're just assigning handles
                if deref_needed == 0:
                    self.scope_manager.mark_variable_no_storage(node.var_name)
                # small hack - to get heap alloc for REF_KIND_HEAP, mark as
                # escaping. else we'd need to keep track of the ref_kind in
                # scope manager too - which would be advantageous in the case
                # there's a heap variable which isn't escaping, we can emit
                # an info for the user that heap alloc is overkill
                if node.ref_kind == REF_KIND_HEAP:
                    self.scope_manager.mark_variable_escaping(node.var_name)

        if init_expr:
            if is_struct:
                # If left is a reference, and initializer's isn't, we need to dereference LHS
                if deref_needed == -1 or deref_needed == 2:
                    self.output.write(self.indent() + '%s = %s;\n' %
                                     (self.dereference(node.var_type, node.var_name), expr_code))
                else:
                    # Direct handle assignment for expressions that return handles
                    self.output.write(self.indent() + '%s = %s;\n' % (node.var_name, expr_code))
            else:
                self.output.write(self.indent() + '%s %s = %s;\n' % (var_type, node.var_name, init_expr))
        elif not is_struct: # struct already declared via macro
            self.output.write(self.indent() + '%s %s;\n' % (var_type, node.var_name))

        # Add test printf for main function or global variables
        # But only for primitive types and strings, not structs
        if not is_struct and (self.current_function == 'main' or is_global):
            if not is_byval_struct_type(node.var_type):
                self.add_test_printf(node.var_name, node.var_type)

    def generate_function_prototype_string(self, node):
        """Generate C function prototype string without ending ;"""
        func_name = self.get_method_name_from_node(node)
        func_id = registry.lookup_function(node.name, node.parent_struct_id)
        func_obj = registry.get_func_from_id(func_id)

        # tell scope manager about tuple types we encounter in the wild
        if registry.is_tuple_type(node.return_type):
            self.scope_manager.declare_tuple_type(node.return_type, tuple_decl(node.return_type))

        # Generate return type
        ret_type = type_to_c(node.return_type, use_handles=registry.is_struct_type(node.return_type) and func_obj.is_ref_return)

        # Generate parameter list
        params = []
        for param in node.params:
            # Unpack the tuple (name, type, is_byref)
            param_name, param_type, is_byref = param

            if registry.is_tuple_type(param_type):
                self.scope_manager.declare_tuple_type(param_type, tuple_decl(param_type))

            c_type = type_to_c(param_type)

            # Handle byref params
            # Only add * for primitive types, not for handles
            # Since handles are already reference types
            is_struct = registry.is_struct_type(param_type)
            if is_byref and not is_struct:
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
        self.test_printfs.append('dprintf(99, "%s:%s\\n", %s%s);' % (var_name, fmt, cast, var_name))

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
        self.output.write(self.indent() + '%s;\n' % self.scope_manager.current_scope().get_macro_name("ZW_SCOPE_START"))

        for stmt in body:
            self.generate_statement(stmt)

        # For main function, add test printfs
        if func_name == 'main':
            for printf_stmt in self.test_printfs:
                self.output.write(self.indent() + printf_stmt + '\n')

        # Generate scope end macro and leave scope
        scope = self.scope_manager.leave_scope(self.helper_header)
        self.output.write(self.indent() + '%s;\n' % scope.get_macro_name("ZW_SCOPE_END"))
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

        # Write if condition
        self.output.write(self.indent() + 'if (%s) {\n' % condition)
        self.indent_level += 1
        scope = self.scope_manager.enter_scope(self.current_function)
        self.output.write(self.indent() + '%s;\n' % scope.get_macro_name("ZW_SCOPE_START"))

        # Generate then body
        for stmt in node.then_body:
            self.generate_statement(stmt)

        scope = self.scope_manager.leave_scope(self.helper_header)
        self.output.write(self.indent() + '%s;\n' % scope.get_macro_name("ZW_SCOPE_END"))
        self.indent_level -= 1

        # Generate else part if it exists
        if node.else_body:
            self.output.write(self.indent() + '} else {\n')
            self.indent_level += 1
            scope = self.scope_manager.enter_scope(self.current_function)
            self.output.write(self.indent() + '%s;\n' % scope.get_macro_name("ZW_SCOPE_START"))

            for stmt in node.else_body:
                self.generate_statement(stmt)

            # Generate scope end macro and leave scope
            scope = self.scope_manager.leave_scope(self.helper_header)
            self.output.write(self.indent() + '%s;\n' % scope.get_macro_name("ZW_SCOPE_END"))
            self.indent_level -= 1

        self.output.write(self.indent() + '}\n')

    def generate_while_statement(self, node):
        """Generate C code for a while loop"""
        condition = self.generate_expression(node.condition)

        # Write while header
        self.output.write(self.indent() + 'while (%s) {\n' % condition)
        self.indent_level += 1
        scope = self.scope_manager.enter_scope(self.current_function)
        self.output.write(self.indent() + '%s;\n' % scope.get_macro_name("ZW_SCOPE_START"))

        # Generate loop body
        for stmt in node.body:
            self.generate_statement(stmt)

        # Generate scope end macro and leave scope
        scope = self.scope_manager.leave_scope(self.helper_header)
        self.output.write(self.indent() + '%s;\n' % scope.get_macro_name("ZW_SCOPE_END"))
        self.indent_level -= 1
        self.output.write(self.indent() + '}\n')

    def generate_generic_initializer(self, node):
        """Generate code for a generic initializer node (structs, arrays, tuples)"""
        # tell scope manager about tuple types we encounter in the wild
        if registry.is_tuple_type(node.expr_type):
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
            if array_size is not None and array_size > len(node.elements):
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
            if node.obj:
                struct_name = registry.get_struct_name(node.obj.expr_type)
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

            # Set appropriate ref_kind for struct instances that use handles
            if node.ref_kind == REF_KIND_NONE:
                node.ref_kind = self.get_container_field_ref_kind(node.expr_type)

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
            args = []
            for arg in node.args:
                args.append(self.generate_expression(arg))
                if arg.node_type == AST_NODE_VARIABLE:
                    self.scope_manager.mark_variable_escaping(arg.name)
            return '%s(%s)' % (func_name, ', '.join(args))

        elif node.node_type == AST_NODE_COMPARE:
            left = self.generate_expression(node.left)
            right = self.generate_expression(node.right)
            op = C_OP_MAP.get(node.operator, node.operator)
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
            # we can't rely on the compiler setting the field ref_kind
            # for nested struct access to REF_KIND_NONE, because we patch in
            # the ref_kind for struct variables after the fact. we need to treat all struct
            # accesses with dereferencing, except for the case of inner structs.
            if node.obj.node_type == AST_NODE_MEMBER_ACCESS or is_byval_struct_type(node.obj.expr_type):
                return "%s.%s"%(expr, node.member_name)
            struct_string = type_to_c(node.obj.expr_type, use_handles=False)
            return '((%s*)(ha_obj_get_ptr(&ha, %s)))->%s' % (struct_string, expr, node.member_name)

        elif node.node_type == AST_NODE_GENERIC_INITIALIZER:
            return self.generate_generic_initializer(node)

        elif node.node_type == AST_NODE_NIL:
            return "handle_nil"

        return "/* Unknown expression */"

def generate_c_code(ast):
    """Entry point function to generate C code from an AST"""
    generator = CCodeGenerator()
    return generator.generate(ast)
