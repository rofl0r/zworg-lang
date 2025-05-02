class EnvironmentStack:
    """Stack-based environment implementation with support for scopes"""
    def reset(self):
        self.stack = [{}]  # Start with global scope at index 0
        self.stackptr = 0

    def __init__(self):
        self.reset()

    def enter_scope(self):
        """Enter a new scope - reuse existing or create new one"""
        self.stackptr += 1
        if self.stackptr >= len(self.stack):
            self.stack.append({})
        else:
            # Reuse existing dict but clear it
            self.stack[self.stackptr].clear()

    def leave_scope(self):
        """Leave current scope and return to previous"""
        if self.stackptr > 0:
            self.stackptr -= 1

    def get(self, name, all_scopes=True):
        """Get a variable value looking through all accessible scopes"""
        if not all_scopes:
            if name in self.stack[-1]:
                return self.stack[-1][name]
            return None

        # Search from current scope down to global
        for i in range(self.stackptr, -1, -1):
            if name in self.stack[i]:
                return self.stack[i][name]
        return None

    def set(self, name, value):
        """Set a variable in the current scope. value is a Variable obj."""
        self.stack[self.stackptr][name] = value

