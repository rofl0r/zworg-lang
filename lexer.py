from shared import *

class Token:
    def __init__(self, type, value, line=0, column=0):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __str__(self):
        return 'Token(%s, %s)' % (token_name(self.type), self.value)

class Lexer:
    def __init__(self, text):
        self.text = text
        self.line = 1      # Current line number (1-based)
        self.column = 1    # Current column number (1-based)
        self.pos = 0
        self.current_char = text[0] if text else None

        # Single-character tokens
        self.simple_tokens = {
            ',': TT_COMMA,
            ';': TT_SEMI,
            '(': TT_LPAREN,
            ')': TT_RPAREN,
            '.': TT_DOT,
            '&': TT_BITAND,
            '|': TT_BITOR,
        }

        # Tokens that might be followed by '=' to form a compound token
        self.compound_tokens = {
            '+': (TT_PLUS, TT_PLUS_ASSIGN),
            '-': (TT_MINUS, TT_MINUS_ASSIGN),
            '*': (TT_MULT, TT_MULT_ASSIGN),
            '%': (TT_MOD, TT_MOD_ASSIGN),
            '=': (TT_ASSIGN, TT_EQ),
            '!': (TT_NOT, TT_NE),
            '>': (TT_GT, TT_GE),
            '<': (TT_LT, TT_LE),
            ':': (TT_COLON, TT_TYPE_ASSIGN),
        }

        # Map of characters to their respective handler methods
        self.op_map = {
            '"': self.handle_string,
            '/': self.handle_div,
        }

    def make_token(self, token_type, value, do_advance=True):
        """Helper to create a token with current line and column info"""
        token = Token(token_type, value, self.line, self.column)
        if do_advance: self.advance()
        return token

    def error(self, message="Invalid character"):
        raise CompilerException('%s at line %d, column %d: "%s"' %
                       (message, self.line, self.column, self.current_char))

    def advance(self):
        # Update line and column tracking
        if self.current_char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def peek(self):
        """Look at the next character without advancing"""
        peek_pos = self.pos + 1
        return self.text[peek_pos] if peek_pos < len(self.text) else None

    def skip_whitespace(self):
        while self.current_char and self.current_char != '\n' and self.current_char.isspace():
            self.advance()

    def skip_until(self, terminator):
        """Skip chars until the terminator char is found or EOF w/o consuming the terminator"""
        while self.current_char is not None and self.current_char != terminator:
            self.advance()

    def number(self):
        """Parse a number (integer or float) with C-style suffixes"""
        start = self.pos
        start_line = self.line
        start_column = self.column
        is_hex = False
        is_float = False

        # Check for hex prefix
        if self.current_char == '0' and self.peek() in ['x', 'X']:
            is_hex = True
            self.advance()  # Skip '0'
            self.advance()  # Skip 'x'
            # Read hex digits
            while self.current_char and self.current_char in '0123456789abcdefABCDEF':
                self.advance()
        else:
            # Read decimal integer or float
            while self.current_char and self.current_char.isdigit():
                self.advance()

            # Check for decimal point
            if self.current_char == '.':
                is_float = True
                self.advance()
                # Must have at least one digit after decimal
                if not (self.current_char and self.current_char.isdigit()):
                    self.error("Invalid float literal: requires digits after decimal point")
                while self.current_char and self.current_char.isdigit():
                    self.advance()

        # Parse any suffix (u, l, ul, f)
        if not is_float and self.current_char in ['u', 'l', 'U', 'L']:
            num_str = self.text[start:self.pos]
            num_value = int(num_str, 16 if is_hex else 10)
            suffix = self.parse_int_suffix()

            # Create appropriate token based on suffix
            if suffix.lower() == 'u':
                return Token(TT_UINT_LITERAL, num_value, start_line, start_column)
            if suffix.lower() == 'l':
                return Token(TT_LONG_LITERAL, num_value, start_line, start_column)
            if suffix.lower() in ['ul', 'lu']:
                return Token(TT_ULONG_LITERAL, num_value, start_line, start_column)
            
            self.error("Invalid integer literal suffix: '%s'" % suffix)

        # Handle float literal
        if is_float:
            value_str = self.text[start:self.pos]
            value = float(value_str)
            # Check for float suffix
            if self.current_char in ['f', 'F']:
                self.advance()
                return Token(TT_FLOAT_LITERAL, value, start_line, start_column)
            elif self.current_char in ['d', 'D']:
                self.advance()
                return Token(TT_DOUBLE_LITERAL, value, start_line, start_column)
            # No suffix = double by default
            return Token(TT_DOUBLE_LITERAL, value, start_line, start_column)

        # Plain integer literal
        value_str = self.text[start:self.pos]
        value = int(value_str, 16 if is_hex else 10)
        return Token(TT_INT_LITERAL, value, start_line, start_column)

    def parse_int_suffix(self):
        """Parse integer literal suffixes (u, l, ul, lu)"""
        first = self.current_char.lower()
        self.advance()

        if first == 'u':
            if self.current_char and self.current_char.lower() == 'l':
                self.advance()
                return 'ul'
            return 'u'
        elif first == 'l':
            if self.current_char and self.current_char.lower() == 'u':
                self.advance()
                return 'ul'  # standardize to 'ul' even if written as 'lu'
            return 'l'
        
        self.error("Expected integer literal suffix")

    def handle_string(self):
        """Handle string literals"""
        start_line = self.line
        start_column = self.column
        self.advance()  # Skip opening quote

        result = ""
        while self.current_char is not None and self.current_char != '"':
            # Handle escape sequences
            if self.current_char == '\\':
                self.advance()
                if self.current_char == 'n':
                    result += '\n'
                elif self.current_char == 't':
                    result += '\t'
                elif self.current_char == 'r':
                    result += '\r'
                elif self.current_char == '"':
                    result += '"'
                elif self.current_char == '\\':
                    result += '\\'
                else:
                    self.error("Invalid escape sequence")
                self.advance()
            else:
                result += self.current_char
                self.advance()

        if self.current_char is None:
            self.error("Unterminated string literal")

        self.advance()  # Skip closing quote
        return Token(TT_STRING_LITERAL, result, start_line, start_column)

    def identifier(self):
        """Parse an identifier or keyword"""
        start = self.pos
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            self.advance()
        value_str = self.text[start:self.pos]

        # Check for type keywords first
        if value_str in ['i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64']:
            # Map size-specific integer types to token types
            type_map = {
                'i8': TT_TYPE_I8, 'u8': TT_TYPE_U8,
                'i16': TT_TYPE_I16, 'u16': TT_TYPE_U16,
                'i32': TT_TYPE_I32, 'u32': TT_TYPE_U32,
                'i64': TT_TYPE_I64, 'u64': TT_TYPE_U64
            }
            return self.make_token(type_map[value_str], value_str, do_advance=False)

        # Then check other keywords
        token_type = KEYWORDS.get(value_str, TT_IDENT)
        return self.make_token(token_type, value_str, do_advance=False)

    def handle_compound_op(self, base_value, base_type, compound_type):
        token = self.make_token(base_type, base_value)
        if self.current_char == '=':
            token.type = compound_type
            token.value += '='
            self.advance()
        return token

    def handle_div(self):
        """Handle division operator or comments"""
        if self.peek() == '/':  # Line comment
            self.advance()  # Skip first '/'
            self.advance()  # Skip second '/'
            self.skip_until('\n')  # Skip until end of line
            return self.next_token()  # Return next token after comment
        elif self.peek() == '*':  # Block comment
            self.advance()  # Skip '/'
            self.advance()  # Skip '*'
            while True:
                if self.current_char is None:
                    self.error("Unterminated block comment")
                if self.current_char == '*' and self.peek() == '/':
                    self.advance()  # Skip '*'
                    self.advance()  # Skip '/'
                    return self.next_token()
                self.advance()
        else:  # Division operator
            token = self.make_token(TT_DIV, '/')
            if self.current_char == '=':
                token.type = TT_DIV_ASSIGN
                token.value = '/='
                self.advance()
            return token

    def next_token(self):
        while self.current_char:
            if self.current_char == '\n':
                return self.make_token(TT_NEWLINE, '\n')

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == '"':
                return self.handle_string()

            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()

            if self.current_char in self.simple_tokens:
                return self.make_token(self.simple_tokens[self.current_char], self.current_char)

            if self.current_char in self.compound_tokens:
                bt, ct = self.compound_tokens[self.current_char]
                return self.handle_compound_op(self.current_char, bt, ct)

            if self.current_char in self.op_map:
                return self.op_map[self.current_char]()

            self.error()

        return self.make_token(TT_EOF, None, do_advance=False)
