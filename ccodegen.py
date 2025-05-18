# C code generator
# Python 2.7 compatible

from shared import *
from type_registry import get_registry
try:
    from StringIO import StringIO  # Python 2
except ImportError:
    from io import StringIO  # Python 3

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

class CCodeGenerator:
    def __init__(self, registry):
        self.registry = registry
        self.output = StringIO()
        self.indent_level = 0
        self.current_function = None
        # Stack to track variables for testing
        self.test_printfs = []

    def indent(self):
        return '\t' * self.indent_level

    def make_reserved_identifier(self, name):
        """Transform an identifier into a reserved one by adding zw_ prefix"""
        return "zw_" + name

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

        return self.output.getvalue()

    def generate_c_main(self):
        """Generate the C main function that initializes the runtime and calls zw_main"""
        self.output.write(main_header)

    def generate_struct_def(self, node):
        """Generate C code for a struct definition"""
        self.output.write('struct %s {\n' % node.name)

        # Generate struct members
        for field_name, field_type in node.fields:
            c_type = self.type_to_c(field_type)
            self.output.write('    %s %s;\n' % (c_type, field_name))

        self.output.write('};\n\n')

    def generate_var_decl(self, node, is_global=False):
        """Generate C code for a variable declaration (both local and global)"""
        var_type = self.type_to_c(node.var_type)

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

        if init_expr:
            self.output.write('%s %s = %s;\n' % (var_type, node.var_name, init_expr))
        else:
            self.output.write('%s %s;\n' % (var_type, node.var_name))

        # Add test printf for main function or global variables
        # But only for primitive types and strings, not structs
        is_struct = self.registry.is_struct_type(node.var_type)
        if not is_struct and (self.current_function == 'main' or is_global):
            self.add_test_printf(node.var_name, node.var_type)

    def generate_function_prototype(self, node):
        """Generate C function prototype"""
        func_name = self.get_method_name_from_node(node)

        # Generate return type
        ret_type = self.type_to_c(node.return_type) #if hasattr(node, 'return_type') else 'void'

        # Generate parameter list
        params = []
        for param in node.params:
            # Unpack the tuple (name, type, is_byref)
            param_name, param_type, is_byref = param
            c_type = self.type_to_c(param_type)

            # Handle byref params
            # Only add * for primitive types, not for handles
            # Since handles are already reference types
            is_struct = self.registry.is_struct_type(param_type)
            if is_byref and not is_struct:
                c_type += '*'
            params.append('%s %s' % (c_type, param_name))

        param_str = ', '.join(params) if params else 'void'

        # Function prototype
        if func_name == "main": func_name = self.make_reserved_identifier("main")
        self.output.write('%s %s(%s);\n' % (ret_type, func_name, param_str))

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

    def type_to_c(self, type_id):
        """Convert type_id to a C type string"""
        if self.registry.is_primitive_type(type_id):
            # For primitive types, use the type name directly
            type_name = self.registry.var_type_to_string(type_id)
            return type_name

        elif self.registry.is_array_type(type_id):
            # For arrays, get element type and make it a pointer
            elem_type_id = self.registry.get_array_element_type(type_id)
            elem_type = self.type_to_c(elem_type_id)
            return "%s*" % elem_type

        elif self.registry.is_struct_type(type_id):
            return "handle"

        return "void"  # Default fallback

    def generate_function(self, node):
        """Generate C code for a function definition"""
        # make C-compatible method name
        func_name = self.get_method_name_from_node(node)
        self.current_function = func_name

        # Clear test printfs for this function
        self.test_printfs = []

        # Get function body from registry
        func_id = self.registry.lookup_function(node.name, node.parent_struct_id)
        if func_id == -1:
            # Should not happen, but handle it gracefully
            return

        func_obj = self.registry.get_func_from_id(func_id)
        body = func_obj.ast_node.body

        # Generate return type
        ret_type = self.type_to_c(func_obj.return_type)

        # Generate parameter list
        params = []
        for param in func_obj.params:
            param_name, param_type, is_byref = param
            param_type_str = self.type_to_c(param_type)
            # Handle byref params
            is_struct = self.registry.is_struct_type(param_type)
            if is_byref and not is_struct:
                param_type_str += '*'
            params.append('%s %s' % (param_type_str, param_name))

        param_str = ', '.join(params) if params else 'void'

        # Function signature
        # Special case for main - use internal identifier
        out_func_name = func_name
        if func_name == 'main':
            out_func_name = self.make_reserved_identifier('main')
        self.output.write('\n%s %s(%s) {\n' % (ret_type, out_func_name, param_str))

        # Function body
        self.indent_level += 1
        for stmt in body:
            self.generate_statement(stmt)

        # For main function, add test printfs
        if func_name == 'main':
            for printf_stmt in self.test_printfs:
                self.output.write(self.indent() + printf_stmt + '\n')
            # Add return 0 at the end of main
            self.output.write(self.indent() + 'return 0;\n')

        self.indent_level -= 1

        # Close function
        self.output.write('}\n')
        self.current_function = None

    def generate_statement(self, node):
        """Generate C code for a statement"""
        if node.node_type == AST_NODE_EXPR_STMT:
            self.output.write(self.indent())
            self.output.write('%s;\n' % self.generate_expression(node.expr))

        elif node.node_type == AST_NODE_VAR_DECL:
            self.output.write(self.indent())
            self.generate_var_decl(node, is_global=False)

        elif node.node_type == AST_NODE_RETURN:
            self.output.write(self.indent())
            if hasattr(node, 'expr') and node.expr:
                self.output.write('return %s;\n' % self.generate_expression(node.expr))
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

        # Generate then body
        for stmt in node.then_body:
            self.generate_statement(stmt)

        self.indent_level -= 1

        # Generate else part if it exists
        if node.else_body:
            self.output.write(self.indent() + '} else {\n')
            self.indent_level += 1

            for stmt in node.else_body:
                self.generate_statement(stmt)

            self.indent_level -= 1

        self.output.write(self.indent() + '}\n')

    def generate_while_statement(self, node):
        """Generate C code for a while loop"""
        condition = self.generate_expression(node.condition)

        # Write while header
        self.output.write(self.indent() + 'while (%s) {\n' % condition)
        self.indent_level += 1

        # Generate loop body
        for stmt in node.body:
            self.generate_statement(stmt)

        self.indent_level -= 1
        self.output.write(self.indent() + '}\n')

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
                struct_name = self.registry.get_struct_name(node.obj.expr_type)
        elif node.node_type == AST_NODE_FUNCTION_DECL:
                if node.parent_struct_id != -1:
                    struct_name = self.registry.get_struct_name(node.parent_struct_id)
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

        elif node.node_type == AST_NODE_VARIABLE:
            return node.name

        elif node.node_type == AST_NODE_BINARY_OP:
            left = self.generate_expression(node.left)
            right = self.generate_expression(node.right)
            op = C_OP_MAP.get(node.operator, node.operator)
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

        elif node.node_type == AST_NODE_STRUCT_INIT:
            # Generate struct initialization
            return 'ha_obj_alloc(&ha, sizeof(struct %s))' % node.struct_name

        elif node.node_type == AST_NODE_MEMBER_ACCESS:
            # Generate member access expression
            expr = self.generate_expression(node.obj)
            struct_name = self.registry.get_struct_name(node.obj.expr_type)
            return '((struct %s*)(ha_obj_get_ptr(&ha, %s)))->%s' % (struct_name, expr, node.member_name)

        return "/* Unknown expression */"

def generate_c_code(ast, registry):
    """Entry point function to generate C code from an AST"""
    generator = CCodeGenerator(registry)
    return generator.generate(ast)
