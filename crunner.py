# crunner.py - Executor for C code generator
import sys
import subprocess
import os
import fcntl

from shared import *
from lexer import Lexer
from compiler import Parser
from type_registry import get_registry
from ccodegen import generate_c_code

# subprocess.Popen is compatible with both Python 2 and 3, but with some differences:
# 1. In Python 3, communicate() returns bytes that need to be decoded
# 2. In Python 2, communicate() returns str objects directly

def run_with_fd_redirect(executable, fd_num):
    """
    Run executable with a specified fd redirected to a pipe that we can read from.
    Returns (exit_code, stdout, stderr, fd_output)
    """
    # Create a pipe
    read_fd, write_fd = os.pipe()

    # Set the read end to non-blocking
    flags = fcntl.fcntl(read_fd, fcntl.F_GETFL)
    fcntl.fcntl(read_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    # Run the subprocess with the write_fd duplicated to fd_num
    proc = subprocess.Popen(
        executable,
        stderr=subprocess.PIPE,
        close_fds=False,  # Don't close all fds
        preexec_fn=lambda: os.dup2(write_fd, fd_num)  # Map our write_fd to fd_num in child
    )

    # Close our end of the write pipe (child process still has it open)
    os.close(write_fd)

    # Get stdout/stderr
    stdout, stderr = proc.communicate()

    # Read from our pipe
    fd_output = b"" if sys.version_info[0] >= 3 else ""
    try:
        while True:
            try:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
                fd_output += chunk
            except (OSError, IOError):  # IOError for Python 2, OSError for Python 3
                # No more data available
                break
    finally:
        os.close(read_fd)

    # Convert to string if needed
    if hasattr(fd_output, 'decode'):
        fd_output = fd_output.decode('utf-8')
    if hasattr(stdout, 'decode'):
        stdout = stdout.decode('utf-8')
    if hasattr(stderr, 'decode'):
        stderr = stderr.decode('utf-8')

    return (proc.returncode, stdout, stderr, fd_output)

class CRunner:
    """Drop-in replacement for Interpreter that uses the C backend"""

    def __init__(self):
        self.registry = get_registry()

    def reset(self):
        self.registry.reset()

    def run(self, code):
        """Run Zworg code through C backend"""
        # Parse the program
        lexer = Lexer(code)
        parser = Parser(lexer)

        program = None
        c_code = None
        try:
            program = parser.parse()
            if not program:
                return {'success': False, 'error': "Parse error or empty program", 'ast': None}

            # Generate C code
            c_code = generate_c_code(program)

            # Write C code to temporary file
            with open("tmp.c", "w") as f:
                f.write(c_code)

            # Compile the C code
            compile_proc = subprocess.Popen(["cc", "-std=gnu99", "tmp.c"],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
            compile_out, compile_err = compile_proc.communicate()

            if compile_proc.returncode != 0:
                return {
                    'success': False,
                    'error': "Compilation error: " + (compile_err.decode('utf-8') if hasattr(compile_err, 'decode') else compile_err),
                    'ast': program,
                    'c_code': c_code,
                }

            # Run the compiled program with fd 99 directly piped
            returncode, stdout, stderr, output = run_with_fd_redirect("./a.out", 99)

            if returncode != 0:
                return {
                    'success': False,
                    'error': "Runtime error: " + stderr,
                    'ast': program,
                    'c_code': c_code
                }

            # Parse the variable values
            main_env = {}
            global_env = {}
            for line in output.strip().split("\n"):
                if not line or ":" not in line:
                    continue
                name, value = line.split(":", 1)
                # Convert to appropriate type for comparison
                try:
                    if value == "":  # Empty string
                        main_env[name] = ""
                    elif '.' in value:  # Float
                        main_env[name] = float(value)
                    else:  # Integer
                        main_env[name] = int(value)
                except ValueError:
                    # Leave as string if not numeric
                    main_env[name] = value

            return {
                'success': True,
                'main_env': main_env,
                'global_env': global_env,
                'c_code': c_code,
                'ast': program,
            }

        except CompilerException as e:
            import traceback
            return {'success': False, 'error': str(e), 'traceback': traceback.format_exc(), 'ast': program, 'c_code': c_code}

