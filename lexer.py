from shared import *

class Token:
    def __init__(self, type, value, line=0, column=0):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __str__(self):
        return 'Token(%d, %s)' % (self.type, self.value)

class Lexer:
    def __init__(self, text):
        self.text = text
        self.line = 1      # Current line number (1-based)
        self.column = 1    # Current column number (1-based)
        self.pos = 0
        self.current_char = text[0] if text else None

        self.simple_tokens = {
            ',': TT_COMMA,
            ';': TT_SEMI,
            '(': TT_LPAREN,
            ')': TT_RPAREN,
            # FIXME: do we need those? we have bitand/bitor keywords
            '&': TT_BITAND,
            '|': TT_BITOR,
        }
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

    def skip_whitespace(self):
        while self.current_char and self.current_char != '\n' and self.current_char.isspace():
            self.advance()

    def skip_until(self, terminator):
        """Skip chars until the terminator char is found or EOF w/o consuming the terminator"""
        while self.current_char is not None and self.current_char != terminator:
            self.advance()

    def number(self):
        """Parse a number (integer or float)"""
        start = self.pos

        # Track the start position for creating the token later
        start_line = self.line
        start_column = self.column

        # Read the first part of the number (digits before decimal point)
        while self.current_char and self.current_char.isdigit():
            self.advance()

        # Check for suffix first (u, l, ul, lu) - needs to be checked before decimal point
        if self.current_char in ['u', 'l']:
            num_value = int(self.text[start:self.pos])
            suffix = self.parse_int_suffix()

            # Create appropriate token based on suffix
            if suffix == 'u': return Token(TT_UINT_LITERAL, num_value, start_line, start_column)
            if suffix == 'l': return Token(TT_LONG_LITERAL, num_value, start_line, start_column)
            if suffix == 'ul': return Token(TT_ULONG_LITERAL, num_value, start_line, start_column)

            # If we get here, an invalid suffix was used
            self.error("Invalid integer literal suffix: '%s'" % suffix)

        # Check for decimal point (for float literals)
        if self.current_char == '.':
            self.advance()

            # For our restricted syntax, there MUST be at least one digit after the decimal
            if not (self.current_char and self.current_char.isdigit()):
                self.error("Invalid float literal: requires digits after decimal point")

            # Read digits after decimal point
            while self.current_char and self.current_char.isdigit():
                self.advance()

            # Create a float token
            value_str = self.text[start:self.pos]
            value = float(value_str)
            return Token(TT_FLOAT_LITERAL, value, start_line, start_column)
        else:
            # Create an integer token
            value_str = self.text[start:self.pos]
            value = int(value_str)
            return Token(TT_INT_LITERAL, value, start_line, start_column)

    def parse_int_suffix(self):
        """Parse integer literal suffixes (u, l, ul)"""
        if self.current_char == 'u':
            self.advance()
            if self.current_char == 'l':
                self.advance()
                return 'ul'
            return 'u'
        elif self.current_char == 'l':
            self.advance()
            if self.current_char == 'u':
                self.advance()
                return 'ul'  # standardize to 'ul' even if input was 'lu'
            return 'l'
        self.error("Expected integer literal suffix")

    def handle_string(self):
        """Handle string literals"""
        # Record starting position for error reporting
        start_line = self.line
        start_column = self.column

        # Skip the opening quote
        self.advance()

        # Start collecting string content
        result = ""
        while self.current_char is not None and self.current_char != '"':
            result += self.current_char
            self.advance()

        # Check if we ended because of a closing quote or end of input
        if self.current_char is None:
            self.error("Unterminated string literal")

        # Skip the closing quote
        self.advance()

        # Create a string token
        return Token(TT_STRING_LITERAL, result, start_line, start_column)

    def identifier(self):
        """
        Parse an identifier or keyword using the global KEYWORDS hashtable.

        Valid identifiers start with a letter or underscore and can contain 
        letters, digits, or underscores.

        Keywords are checked against the global KEYWORDS dictionary.
        """
        start = self.pos
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            self.advance()
        value_str = self.text[start:self.pos]

        # Look up in the global KEYWORDS hashtable, default to TT_IDENT if not found
        token_type = KEYWORDS.get(value_str, TT_IDENT)
        return self.make_token(token_type, value_str, do_advance=False)

    def handle_compound_op(self, base_value, base_type, compound_type):
        token = self.make_token(base_type, base_value)
        if self.current_char == '=':
            token.type = compound_type
            token.value += '='
            self.advance()
        return token

    # Handlers for various operators
    def handle_div(self):
        token = self.make_token(TT_DIV, '/')
        # Handle C++-style comments
        if self.current_char == '/':
            # Skip first '/'
            self.advance()  # Skip second '/'
            self.skip_until('\n')  # Skip until end of line, but don't consume the newline
            return self.next_token()  # Return the next token after the comment
        elif self.current_char == '=':
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

            # Use op_map for operators
            if self.current_char in self.op_map:
                return self.op_map[self.current_char]()

            self.error()

        return self.make_token(TT_EOF, None, do_advance=False)

