# Test framework for compiler.py
from interpreter import Interpreter
import os, sys

def printc(color, text, file=sys.stdout):
	cols = {
		"default": 98,
		"white" : 97,
		"cyan" : 96,
		"magenta" : 95,
		"blue" : 94,
		"yellow" : 93,
		"green" : 92,
		"red" : 91,
		"gray" : 90,
		"end" : 0
	}
	colstr = "\033[%dm"
	file.write( "%s%s%s" % (colstr%cols[color], text, colstr%cols['end']) )

def test():
    # Test cases with expected final env state
    test_cases = [
        # Regular test cases (expected to succeed)
        # Each has "code" and "expected_env"
    {
        "name": "Method with incorrect return type",
        "code": """
            struct Point do
                x: int
            end

            def Point.get_x(): string do
                return self.x;  // int returned from string function
            end

            def main() do
            end
        """,
        "expected_error": "Type mismatch"
    },
        {
           "name": "test struct 1",
           "code": """
    struct Point do
        x: int
        y: int
    end

    def Point.init(x_val: int, y_val: int) do
        self.x = x_val
        self.y = y_val
    end

    def Point.magnitude(): int do
        return self.x * self.x + self.y * self.y;
    end

    def main() do
        var p := Point(3, 4);
        var x := p.x;      // 3
        var y := p.y;      // 4
        var m := p.magnitude();  // 25
        p.x = 5;
        var z := p.x      // 5
    end
           """,
           "expected_env": {"x": 3, "y": 4, "m": 25, "z":5}
        },
        {
           "name": "test struct inheritance",
           "code": """
    struct Shape do
        name: string;
    end

    def Shape.init(n: string) do
        self.name = n;
    end

    def Shape.describe(): string do
        return self.name;
    end

    struct Circle(Shape) do
        radius: float;
    end

    def Circle.init(n: string, r: float) do
        self.name = n;
        self.radius = r;
    end

    def Circle.area(): float do
        return 3.14 * self.radius * self.radius;
    end

    def main() do
        var c := Circle("My Circle", 2.5);
        var i:= c.name;        // My Circle (inherited field)
        var j:= c.describe();  // My Circle (inherited method)
        var k:= c.radius      // 2.5
        var l:= c.area();      // ~19.625
    end
           """,
           "expected_env": {"i": "My Circle", "j": "My Circle", "k": 2.5, "l":19.625}
        },
        {
           "name": "test struct inheritance",
           "code": """
    struct Counter do
        count: int;
    end

    def Counter.init() do
        self.count = 0;
        print "Counter created";
    end

    def Counter.fini() do
        print "Counter destroyed, final count:";
        print self.count;
    end

    def Counter.increment() do
        self.count = self.count + 1;
    end

    def main() do
        // Stack allocation
        var stackCounter := Counter();
        stackCounter.increment();
        stackCounter.increment();
        var x:= stackCounter.count;  // 2

        // Heap allocation
        var heapCounter := new Counter();
        heapCounter.increment();
        heapCounter.increment();
        heapCounter.increment();
        var y:= heapCounter.count   // 3

        // Explicitly destroy heap-allocated object
        del heapCounter;           // Calls fini() method

        // Stack allocated objects are automatically cleaned up
    end
           """,
           "expected_env": {"x": 2, "y": 3}
        },

        {
           "name": "test funccall, type promotion",
           "code": """
                def add(a:long, b:int):long do return a + b; end
                def main() do var result := add(add(1l, 3), 1); end
           """,
           "expected_env": {"result": 5}
        },
        {
           "name": "test funccall, type promotion 2",
           "code": """
                def add(a:int, b:long):long do return a + b; end
                def main() do var result := add(1, add(1, 3l)); end
           """,
           "expected_env": {"result": 5}
        },
        {
           "name": "empty function",
           "code": """
               def main() do
               end
           """,
           "expected_env": {}
        },
        {
            # Tests variable declaration with type inference (:=)
            "code": "def main() do var x := 5; end",
            "expected_env": {"x": 5}
        },
        {
            # Tests float literal with type inference
            "code": "def main() do var y := 3.14; end",
            "expected_env": {"y": 3.14}
        },
        {
            # Tests explicit int type annotation
            "code": "def main() do var x: int = 10; end",
            "expected_env": {"x": 10}
        },
        {
            # Tests explicit float type annotation
            "code": "def main() do var y: float = 2.718; end",
            "expected_env": {"y": 2.718}
        },
        {
            # Tests let with type inference
            "code": "def main() do let x := 42; end",
            "expected_env": {"x": 42}
        },
        {
            # Tests assignment of same type variable
            "code": "def main() do var x := 5; var y := x; end",
            "expected_env": {"x": 5, "y": 5}
        },
        {
            # Tests assignment expression in while condition (not a comparison)
            "code": "def main() do var x := 0; var y := 0; while x = y do y = y + 1; end; end",
            "expected_env": {"x": 0, "y": 0}
        },
        {
            # Tests operator precedence in expressions (* has higher precedence than +)
            "code": "def main() do var x := 5 + 3 * 2; end",
            "expected_env": {"x": 11}
        },
        {
            # Tests basic if statement with equality comparison
            "code": "def main() do var x := 1; if x == 1 do print x; end end", 
            "expected_env": {"x": 1}
        },
        {
            # Tests if-else statement (true condition branch taken)
            "code": "def main() do var x := 1; var result := 0; if x == 1 do result = x; end else do result = 0; end end",
            "expected_env": {"x": 1, "result": 1}
        },
        {
            # Tests bitwise OR operator (|)
            "code": "def main() do var x := 5 | 3; end", 
            "expected_env": {"x": 7}
        },
        {
            # Tests bitwise AND operator (&)
            "code": "def main() do var x := 5 & 3; end", 
            "expected_env": {"x": 1}
        },
        {
            # Tests combination of bitwise operations with variable references
            "code": "def main() do var x := 5 | 3; var y := x & 2; end",
            "expected_env": {"x": 7, "y": 2}
        },
        {
            # Tests logical AND in if condition (evaluates to false)
            "code": "def main() do var x := 1; var y := 0; var result := 0; if x and y do result = x; end end", 
            "expected_env": {"x": 1, "y": 0, "result": 0}
        },
        {
            # Tests XOR operator with keywords
            "code": "def main() do var x := 5 xor 3; end",
            "expected_env": {"x": 6}  # 5 xor 3 = 6
        },
        {
            # Tests XOR operator with variable references
            "code": "def main() do var x := 7; var y := x xor 2; end",
            "expected_env": {"x": 7, "y": 5}  # 7 xor 2 = 5
        },
        {
            # Tests bitwise NOT unary operator
            "code": "def main() do var x := 15; var y := bitnot 3; end", 
            "expected_env": {"x": 15, "y": -4}
        },
        {
            # Tests else-if construct (first false, second true)
            "code": "def main() do var x := 1; var result := 0; if x == 0 do result = 0; end else if x == 1 do result = 1; end end",
            "expected_env": {"x": 1, "result": 1}
        },
        {
            # Tests multiple else-if branches (first & second false, third true)
            "code": "def main() do var x := 3; if x == 1 do print 1; end else if x == 2 do print 2; end else if x == 3 do print 3; end end",
            "expected_env": {"x": 3}
        },
        {
            # Tests mixed int and float operations
            "code": "def main() do var x: int = 5; var y: float = 2.5; var z := y; end", 
            "expected_env": {"x": 5, "y": 2.5, "z": 2.5}  # Variable declaration with inferred type from another variable
        },
        {
            # Tests int division
            "code": "def main() do var x := 10; var y := 3; var z := x / y; end",
            "expected_env": {"x": 10, "y": 3, "z": 3}  # Integer division
        },
        {
            # Tests float division
            "code": "def main() do var x := 10.0; var y := 3.0; var z := x / y; end",
            "expected_env": {"x": 10.0, "y": 3.0, "z": 3.3333333333333335}  # Float division
        },
        {
            # Tests assignment as expression in while condition
            "code": "def main() do var x := 10; var y := 0; while (y = y + 1) < 5 do x -= 1; end end",
            # y will end up as 5 (the condition becomes false when y = 5)
            # x will be decremented 4 times (while y is 1,2,3,4)
            "expected_env": {"x": 6, "y": 5}
        },
        {
           "name": "long variable declaration and operations",
           "code": """
               def main() do
                   var x : long = 42l;
                   var y := 10l;
                   var z : long = x + y;
                   print z;
                   z = z / 2;
                   print z;
                   z = z * 3;
                   print z;
               end
           """,
           "expected_env": {"x": 42, "y": 10, "z": 78}
        },
        {
           "name": "ulong variable declaration and operations",
           "code": """
               def main() do
                   var x : ulong = 42ul;
                   var y := 10ul;
                   var z : ulong = x + y;
                   print z;
                   z = z / 2;
                   print z;
                   z = z * 3;
                   print z;
               end
           """,
           "expected_env": {"x": 42, "y": 10, "z": 78}
        },
        {
           "name": "mixed types variable declaration - type inference",
           "code": """
               def main() do
                   var a := 0b101010;     // int
                   var b := 42u;    // uint
                   var c := 42_000_l;    // long
                   var d := 42ul;   // ulong
                   var e := 42.0;   // float
                   var f := 0xcafe_babe_ull // unsigned long long

                   // Test assignments to explicitly typed variables
                   var x : int = 10;
                   var y : uint = 20u;
                   var z : long = 30l;
                   var w : ulong = 40ul;
                   var v : float = 50.0;
                   var u : longlong = 0xcafebabe

                   print a;
                   print b;
                   print c;
                   print d;
                   print e;
               end
           """,
           "expected_env": {"a": 42, "b": 42, "c": 42000, "d": 42, "e": 42.0, "f": 0xcafebabe,
                        "x": 10, "y": 20, "z": 30, "w": 40, "v": 50.0, "u": 0xcafebabe}
        },
        {
           "name": "unsigned int division",
           "code": """
               def main() do
                   var x : uint = 10u;
                   var y : uint = 3u;
                   var z : uint = x / y;  // Should be 3 (truncated)
                   print z;
               end
           """,
           "expected_env": {"x": 10, "y": 3, "z": 3}
        },
        {
           "name": "signed division with negative numbers",
           "code": """
               def main() do
                   var x : int = -10;
                   var y : int = 3;
                   var z : int = x / y;  // Should be -3 (truncated toward zero)
                   print z;

                   var a : int = 10;
                   var b : int = -3;
                   var c : int = a / b;  // Should be -3
                   print c;

                   var m : int = -10;
                   var n : int = -3;
                   var o : int = m / n;  // Should be 3
                   print o;
               end
           """,
           "expected_env": {"x": -10, "y": 3, "z": -3, 
                        "a": 10, "b": -3, "c": -3, 
                        "m": -10, "n": -3, "o": 3}
        },
        {
           "name": "assignment in expression-condition",
           "code": """
               def main() do
                   var a : uint = 0u;

                   if a = 5u do
                       print a;
                   end

                   var b : long = 0l;
                   if b = 10l do
                       print b;
                   end

                   var c : ulong = 0ul;
                   if c = 15ul do
                       print c;
                   end
               end
           """,
           "expected_env": {"a": 5, "b": 10, "c": 15}
        },
        # String test cases
        {
            "name": "Basic string declaration",
            "code": """
                def main() do
                    var s := "Hello, world!";
                    print s;
                end
            """,
            "expected_env": {"s": "Hello, world!"}
        },
        {
            "name": "String with explicit type annotation",
            "code": """
                def main() do
                    var s : string = "Hello";
                    print s;
                end
            """,
            "expected_env": {"s": "Hello"}
        },
        {
            "name": "String concatenation",
            "code": """
                def main() do
                    var s1 := "Hello, ";
                    var s2 := "world!";
                    var s3 := s1 + s2;
                    print s3;
                end
            """,
            "expected_env": {"s1": "Hello, ", "s2": "world!", "s3": "Hello, world!"}
        },
        {
            "name": "String compound assignment",
            "code": """
                def main() do
                    var s := "Hello";
                    s += ", world!";
                    print s;
                end
            """,
            "expected_env": {"s": "Hello, world!"}
        },
        {
            "name": "String comparison equality",
            "code": """
                def main() do
                    var s1 := "abc";
                    var s2 := "abc";
                    var result := s1 == s2;
                    print result;
                end
            """,
            "expected_env": {"s1": "abc", "s2": "abc", "result": 1}
        },
        {
            "name": "String comparison inequality",
            "code": """
                def main() do
                    var s1 := "abc";
                    var s2 := "def";
                    var result := s1 != s2;
                    print result;
                end
            """,
            "expected_env": {"s1": "abc", "s2": "def", "result": 1}
        },
        {
            "name": "String in if condition",
            "code": """
                def main() do
                    var s1 := "test";
                    var s2 := "test";
                    var result := 0;
                    if s1 == s2 do
                        result = 1;
                    end
                    print result;
                end
            """,
            "expected_env": {"s1": "test", "s2": "test", "result": 1}
        },
        {
            # Tests functions and return statements
            "code": """
                def add(a:int, b:int):int do
                    return a + b;
                end

                def main() do
                    var x := 10;
                    var y := 20;
                    var result := add(x, y);
                    print result;
                end
            """,
            "expected_env": {"x": 10, "y": 20, "result": 30}
        },
        {
            # Tests function with return value specification
            "code": """
                def square(n:int):int do
                    return n * n;
                end

                def main() do
                    var x := 5;
                    var y := square(x);
                    print y;
                end
            """,
            "expected_env": {"x": 5, "y": 25}
        },
        {
            # Tests global variables
            "code": """
                var global_rw:=0
                let global_r:=42
                def main() do global_rw=10; var x:=global_rw + global_r
                end
            """,
            "expected_env": {"x": 52}
        },


        # Test cases that are expected to fail
        # Each has "code" and "expected_error"
        {
            # Tests invalid use of := operator
            "code": """
                def main() do var x:=10; x:=20 // x is already declared, so := must fail
                end
            """,
            "expected_error": "Cannot use ':=' with already declared variable 'x'. Use '=' instead"
        },
        {
            # Tests invalid redeclaration of variable
            "code": """
                def main() do var x:=10; var x:=20 // x is already declared, so "var x" must fail
                end
            """,
            "expected_error": "Variable 'x' is already declared in this scope"
        },


        {
            # Tests error when trying to assign float to int
            "code": "def main() do var x := 1; var y := 0.1; x = y; end",
            "expected_error": "Type mismatch: can't assign a value of type float to x (type int)"
        },
        {
            # Tests error when using variable without declaration
            "code": "def main() do x = 5; end",  # Missing var or let declaration
            "expected_error": "Variable 'x' is not declared"
        },
        {
            # Tests error when declaring variable without initialization
            "code": "def main() do var x; x = 5; end",  # Missing initialization
            "expected_error": "Variable declaration must include an initialization"
        },
        {
            # Tests error when using undeclared variable in a logical expression
            "code": "def main() do var x := 1; if !x or x and y do print x; end end",
            "expected_error": "Variable 'y' is not declared"
        },
        {
            # Tests error when using undeclared variable in print statement
            "code": "def main() do print z; end",
            "expected_error": "Variable 'z' is not declared"
        },
        {
            # Tests error when using undeclared variable in if condition
            "code": "def main() do if x do print 1; end end",
            "expected_error": "Variable 'x' is not declared" 
        },
        {
            # Tests error when missing semicolon between statements on the same line
            "code": "def main() do var x := 5 + 3 print x; end",
            "expected_error": "Expected semicolon between statements"
        },
        {
            # Tests error when missing semicolon between statements on the same line
            "code": "def main() do var a := 1 var b := 2 end",
            "expected_error": "Expected semicolon between statements"
        },
        {
            # Tests error when reassigning to a let-declared constant
            "code": "def main() do let x := 5; x = 10; end",
            "expected_error": "Cannot reassign to constant 'x'"
        },
        {
            # Tests error when using compound assignment on a let-declared constant
            "code": "def main() do let x := 5; x += 10; end",
            "expected_error": "Cannot reassign to constant 'x'"
        },
        {
            # Tests error when using undeclared variable in initialization
            "code": "def main() do var x := y; end",
            "expected_error": "Variable 'y' is not declared"
        },
        {
            # Tests error when using = without explicit type
            "code": "def main() do var x = 5; end",
            "expected_error": "requires explicit type annotation"
        },
        {
            # Tests error when assigning float to int variable
            "code": "def main() do var x: int = 5; var y: float = 2.5; x = y; end",
            "expected_error": "Type mismatch"
        },
        {
            # Tests error when redeclaring a variable
            "code": "def main() do var x := 5; var x := 10; end",
            "expected_error": "already declared"
        },
        {
            # Tests float literal without decimal digits
            "code": "def main() do var x := 5.; end",
            "expected_error": "Invalid float literal"
        },
        {
            # Tests else-if without proper end before else
            "code": "def main() do var x := 3; if x == 1 do print 1; else if x == 2 do print 2; else if x == 3 do print 3; end end",
            "expected_error": 'Invalid statement starting with "else" (TT_ELSE)'
        },
        {
            # Tests error when mixing int and float types in binary operation
            # This test expects a failure since our language doesn't allow implicit type conversion
            "code": "def main() do var x := 10; var y := 3.0; var z := x / y; end", 
            "expected_error": "Type mismatch in binary operation"
        },
        # Error test cases
        {
            "name": "Unterminated string literal",
            "code": """
                def main() do
                    var s := "Unterminated string;
                end
            """,
            "expected_error": "Unterminated string literal"
        },
        {
            "name": "String concatenation type error",
            "code": """
                def main() do
                    var s := "Hello";
                    var i := 123;
                    var result := s + i;
                end
            """,
            "expected_error": "Cannot concatenate string with non-string type"
        },
        {
            "name": "String comparison error - unsupported operator",
            "code": """
                def main() do
                    var s1 := "abc";
                    var s2 := "def";
                    if s1 < s2 do print "s1 is less"; end
                end
            """,
            "expected_error": "Operator < not supported for strings"
        },
        {
            "name": "String compound assignment error",
            "code": """
                def main() do
                    var s := "Hello";
                    var i := 5;
                    s += i;
                end
            """,
            "expected_error": "Type mismatch: can't assign a value of type int to s (type string)"
        },
        {
            "name": "Invalid assignment from int to string",
            "code": """
                def main() do
                    var s : string = 42;
                end
            """,
            "expected_error": "Type mismatch in initialization: can't assign int to s (type string)"
        },
        {
            "name": "Code outside functions not allowed",
            "code": "var x := 5;",
            "expected_error": "No 'main' function defined"
        },
    # Test cases for OOP parse errors
    {
        "name": "Struct with no fields",
        "code": """
            struct EmptyStruct do
            end

            def main() do
            end
        """,
        "expected_error": "Unexpected token type"
    },
    {
        "name": "Method without self",
        "code": """
            struct Point do
                x: int
            end

            def Point.bad(): int do
                return x;  // Missing self.x
            end

            def main() do
            end
        """,
        "expected_error": "Variable 'x' is not declared"
    },
    {
        "name": "Undefined method call",
        "code": """
            struct Point do
                x: int
            end

            def main() do
                var p := Point(5);
                p.nonexistent();
            end
        """,
        "expected_error": "Method 'nonexistent' not found in struct 'Point'"
    },
    {
        "name": "Undefined field access",
        "code": """
            struct Point do
                x: int
            end

            def main() do
                var p := Point(5);
                print p.y;
            end
        """,
        "expected_error": "Field 'y' not found in struct 'Point'"
    },
    {
        "name": "Invalid constructor parameter count",
        "code": """
            struct Point do
                x: int
            end

            def Point.init(x: int, y: int) do
                self.x = x;
            end

            def main() do
                var p := Point(5);  // Missing second parameter
            end
        """,
        "expected_error": "Constructor for 'Point' expects 2 arguments, got 1"
    },
    {
        "name": "Non-struct method definition",
        "code": """
            def NotAStruct.method() do
            end

            def main() do
            end
        """,
        "expected_error": "Struct 'NotAStruct' is not defined"
    },
    {
        "name": "Invalid parent struct",
        "code": """
            struct Child(NonExistentParent) do
                x: int
            end

            def main() do
            end
        """,
        "expected_error": "Parent struct 'NonExistentParent' is not defined"
    },
    {
        "name": "Constructor with non-void return type",
        "code": """
            struct Point do
                x: int
            end

            def Point.init(): int do
                self.x = 5;
                return 0;
            end

            def main() do
            end
        """,
        "expected_error": "Constructor 'init' must have void return type"
    },
    {
        "name": "Destructor with parameters",
        "code": """
            struct Point do
                x: int
            end

            def Point.fini(flag: int) do
            end

            def main() do
            end
        """,
        "expected_error": "Destructor 'fini' cannot have parameters"
    },
    {
        "name": "Del on stack object",
        "code": """
            struct Point do
                x: int
            end

            def main() do
                var p := Point(5);
                del p;  // Cannot delete stack object
            end
        """,
        "expected_error": "'del' can only be used with reference types"
    },
    {
        "name": "Missing constructor arguments",
        "code": """
            struct Point do
                x: int
                y: int
            end

            def Point.init(x: int, y: int) do
                self.x = x;
                self.y = y;
            end

            def main() do
                var p := Point();  // No arguments provided
            end
        """,
        "expected_error": "Constructor for 'Point' expects 2 arguments, got 0"
    },
    {
        "name": "New operator on built-in non-struct type",
        "code": """
            def main() do
                var x := new int(10);
            end
        """,
        "expected_error": "Expected struct name after 'new'"
    },
    {
        "name": "New operator on non-struct",
        "code": """
            def main() do
                var x := new Foo(10);
            end
        """,
        "expected_error": "Struct 'Foo' is not defined"
    },
    {
        "name": "Self parameter redefinition",
        "code": """
            struct Point do
                x: int
            end

            def Point.init(self: int) do
                self.x = 5;
            end

            def main() do
            end
        """,
        "expected_error": "Cannot use 'self' as a parameter name"
    },
    {
        "name": "Missing parentheses after new",
        "code": """
            struct Counter do
                count: int
            end

            def main() do
                var c := new Counter;  // Missing parentheses
            end
        """,
        "expected_error": "constructor invocation requires parenthesis"
    }

    ]

    # List to track failing tests
    interpreter = Interpreter()
    failed_tests = []

    # Run all test cases
    for i, test_case in enumerate(test_cases):
        interpreter.reset()

        test_num = i + 1
        print("\nTest %d:" % test_num)
        print("Input: %s" % test_case["code"])

        result = interpreter.run(test_case["code"])

        # Check if this test is expected to fail
        if "expected_error" in test_case:
            # This is a test that should fail
            if not result['success'] and test_case["expected_error"] in result['error']:
                print("Success! Failed with expected error: %s" % result['error'])
            else:
                printc("red", "Test didn't fail as expected! Result: %s" % result)
                failed_tests.append(test_num)
                # Add AST dump for unexpected failures
                if result.get('ast'):
                    print("AST dump: %s" % result['ast'])

        else:
            # This is a test that should succeed
            if result['success']:
                # For function tests, we need to get the local environment
                env = None
                if 'result' in result:
                    # This is likely from a function return
                    print("Function returned: %s" % result['result'])

                # Get environment from main function
		env = result['main_env']

                # Check if environment values match
                env_match = True
                for k, v in test_case["expected_env"].iteritems():
                    if k not in env or env[k] != v:
                        env_match = False
                        break

                if env_match:
                    print("Success! Environment matches expectations.")
                else:
                    print("Test passed but with incorrect environment values:")
                    print("  Expected env: %s" % test_case["expected_env"])
                    print("  Actual env: %s" % env)
                    failed_tests.append(test_num)
            else:
                printc("red", "Failed! Error: %s" % result['error'])
                if os.getenv("DEBUG"):
                    import time
                    time.sleep(10000)
                failed_tests.append(test_num)
                if result.get('ast'):
                    print("AST dump: %s" % result['ast'])

    # Print statistics at the end
    print("\n========== Test Results ==========")
    print("Total tests: %d" % len(test_cases))
    print("Failed test IDs: %s" % (", ".join(str(num) for num in failed_tests) if failed_tests else "None"))
    print("All tests passed: %s" % ("No" if failed_tests else "Yes"))

if __name__ == '__main__':
    test()
