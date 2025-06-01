# Test framework for compiler.py
from interpreter import Interpreter
from crunner import CRunner
import os, sys

"""
        {
           "name": "",
           "code": "",
           "expected_env": {"x": 1}
        },
"""
# Test cases with expected final env state
test_cases = [
        # Regular test cases (expected to succeed)
        # Each has "code" and "expected_env"
        {
           "name": "struct array heap mass constructor",
           "code": """
                struct Point do x: int; y: int; end
                def Point.init(x_val: int, y_val: int) do
                    self.x = x_val
                    self.y = y_val
                end
                def Point.sum() : int do
                    return self.x + self.y;
                end
                def main() do
                    var heap_init_points := new Point[2](0, 0)
                    heap_init_points[0].init(10, 20)
                    var init_sum := heap_init_points[0].sum()  // Should be 30 (10+20)
                end
           """,
           "expected_env": {"init_sum": 30}
        },
        {
           "name": "struct array self handle and constructors A",
           "code": """
                struct Point do x: int; y: int; end
                def Point.sum() : int do
                    return self.x + self.y;
                end
                def main() do
                    var stack_points :Point[3]
                    stack_points[1].x = 3
                    stack_points[1].y = 4
                    var stack_sum := stack_points[1].sum()  // Should be 7 (3+4)
                end
           """,
           "expected_env": {"stack_sum": 7}
        },
        {
           "name": "struct array self handle and constructors B",
           "code": """
                struct Point do x: int; y: int; end
                def Point.init(x_val: int, y_val: int) do
                    self.x = x_val
                    self.y = y_val
                end
                def Point.sum() : int do
                    return self.x + self.y;
                end
                def main() do
                    var stack_points :Point[]= {Point(1,2), Point(3,4), Point(5,6)}
                    var stack_sum := stack_points[1].sum()  // Should be 7 (3+4)
                end
           """,
           "expected_env": {"stack_sum": 7}
        },
        {
           "name": "struct array self handle and constructors C",
           "code": """
                struct Point do x: int; y: int; end
                def Point.init(x_val: int, y_val: int) do
                    self.x = x_val
                    self.y = y_val
                end
                def Point.sum() : int do
                    return self.x + self.y;
                end
                def main() do
                    var stack_points :Point[3]
                    stack_points[0] = Point(1,2)
                    stack_points[1] = Point(3,4)
                    stack_points[2] = Point(5,6)
                    var stack_sum := stack_points[1].sum()  // Should be 7 (3+4)
                end
           """,
           "expected_env": {"stack_sum": 7}
        },
        {
           "name": "struct array self handle with nested initializer A",
           "code": """
                struct Point do x: int; y: int; end
                def Point.sum() : int do
                    return self.x + self.y;
                end
                def main() do
                    var stack_points :Point[]= {{1,2}, {3,4}, {5,6}}
                    var stack_sum := stack_points[1].sum()  // Should be 7 (3+4)
                end
           """,
           "expected_env": {"stack_sum": 7}
        },
        {
           "name": "struct array self handle with nested initializer B",
           "code": """
                struct Point do x: int; y: int; end
                def Point.sum() : int do
                    return self.x + self.y;
                end
                def main() do
                    var stack_points :Point[3]= {{1,2}, {3,4}, {5,6}}
                    var stack_sum := stack_points[1].sum()  // Should be 7 (3+4)
                end
           """,
           "expected_env": {"stack_sum": 7}
        },
        {
            "name": "unary not with typedef primitive",
            "code": """
                typedef foo: int
                def main() do
                    var x:= 0
                    var f:foo=0
                    if not f do x+=1; end
                end
            """,
            "expected_env": {"x": 1}
        },
        {
            "name": "comparison with nil",
            "code": """
                struct TestStruct do value: int; end
                def main() do
                    var nil_array: int[] = nil
                    var x:= 0
                    if(nil_array == nil) do x += 3; end
                    if(nil_array != nil) do x += 7; end
                end
            """,
            "expected_env": {"x": 3}
        },
        {
            "name": "dynamic array resize from nil",
            "code": """
                struct TestStruct do value: int; end
                def TestStruct.init(v: int) do self.value = v; end
                def main() do
                    // Test 1: Resize nil int array
                    var nil_array: int[] = nil
                    nil_array = new(nil_array, 3)
                    nil_array[0] = 111
                    var nil_result := nil_array[0]
                    // Test 2: Resize nil struct array
                    var struct_array: TestStruct[] = nil
                    struct_array = new(struct_array, 2)
                    struct_array[0] = new TestStruct(222)
                    var struct_result := struct_array[0].value
                    // Test 3: Test that an array after new is not nil
                    var is_nil := 0
                    var x: int[] = nil
                    if x == nil do
                        is_nil = 1
                    end
                    x = new(x, 1)
                    var not_nil := 0
                    if x != nil do
                        not_nil = 1
                    end
                end
            """,
            "expected_env": {
                "nil_result": 111,
                "struct_result": 222,
                "is_nil": 1,
                "not_nil": 1
            }
        },
        {
            "name": "struct return from array",
            "code": """
                struct Foo do x:int ; end
                def Foo_ret(arr:Foo[], n:int):Foo do
                    return arr[n]
                end
                def main() do
                    var dyn:Foo[]
                    dyn = new(dyn, 10)
                    dyn[4].x = 1337
                    var f := Foo_ret(dyn, 4)
                    var x:= f.x
                end
            """,
            "expected_env": {"x": 1337}
        },
        {
            "name": "simple dynamic array resize",
            "code": """
                def main() do
                    var dyn:int[]
                    dyn = new(dyn, 10)
                    dyn[4] = 1337
                    var x:=dyn[4]
                end
            """,
            "expected_env": {"x": 1337}
        },
        {
            "name": "simple array assign",
            "code": """
                def main() do
                    var arr:int[10]
                    arr[4] = 1337
                    var x:=arr[4]
                end
            """,
            "expected_env": {"x": 1337}
        },
        {
            "name": "global arrays with and without initializer",
            "code": """
                var gnums_zeroed: int[5]
                var gnums_set: longlong[3] = {1,2,3}
                var gnums_dynamic_zeroed: u8[]
                var gnums_dynamic_set: u8[] = nil

                def main() do
                    var nz2 := gnums_zeroed[2]
                    var ns2 := gnums_set[2]
                    var gh := gnums_dynamic_zeroed
                end
            """,
            "expected_env": {"nz2": 0, "ns2": 3}
        },
        {
            "name": "fixed size array without initializer",
            "code": """
                def main() do
                    var nums: int[5]
                    var n_2 := nums[2]
                end
            """,
            "expected_env": {"n_2": 0}
        },
        {
           "name": "nested struct, stack and heap constructor",
           "code": """
		struct Foo do x:int; y:int; end;
                struct Bar do foo:Foo; z:int; end;
                def Bar.init() do end;
                def main() do var b1:=Bar(); var b2:Bar=new Bar(); var x:= b2.foo.x; var z:=b2.z; end
           """,
           "expected_env": {"x": 0, "z": 0}
        },
        {
            "name": "mixed struct and tuple initializers x",
            "code": """
                struct Point do x:int; y:int; end
                def main() do
                    var p:Point = {10, 20}
                    var t := {Point(), "coordinate"}
                    t = {p, "coordinate"}
                    var x := t._0.x
                    var s := t._1
                end
            """,
            "expected_env": {"x": 10, "s": "coordinate"}
        },
        {
            "name": "mixed struct and tuple initializers",
            "code": """
                struct Point do x:int; y:int; end
                def main() do
                    var p:Point = {10, 20}
                    var t := {p, "coordinate"}
                    var x := t._0.x
                    var s := t._1
                end
            """,
            "expected_env": {"x": 10, "s": "coordinate"}
        },
        {
            "name": "tuple initializer with type inference",
            "code": """
                def main() do
                    var t := {1, 2, 3}
                    var x:= t._0 + t._1 + t._2
                end
            """,
            "expected_env": {"x": 6}
        },
        {
            "name": "tuple initializer and return type with type inference",
            "code": """
                def ret_tup():{int, long} do return {1,2}; end
                def main() do
                    var t := ret_tup()
                    var x:= t._1
                end
            """,
            "expected_env": {"x": 2}
        },
        {
            "name": "tuple initializer with type inference and strings",
            "code": """
                def main() do
                    var t := {1, 2, 3}
                    var s := {"hello", 42}
                    var x:= t._0 + t._1 + t._2
                    var y:= s._0 + " " + "world"
                    var z:= s._1
                end
            """,
            "expected_env": {"x": 6, "y": "hello world", "z": 42}
        },
        {
            "name": "struct initializer with type annotation",
            "code": """
                struct Point do x:int; y:int; end
                def main() do
                    var p:Point = {10, 20}
                    var x:=p.x
                    var y:=p.y
                end
            """,
            "expected_env": {"x": 10, "y":20}
        },
        {
            "name": "partial struct initializer",
            "code": """
                struct Point do x:int; y:int; end
                def main() do
                    var p:Point = {10}
                    var x:=p.x
                    var y:=p.y
                end
            """,
            "expected_env": {"x": 10, "y":0}
        },
        {
            "name": "return struct initializer from function",
            "code": """
                struct Point do x:int; y:int; end
                def makePoint(a:int, b:int):Point do
                    return {a, b}
                end
                def main() do
                    var p := makePoint(10, 20)
                    var x := p.x
                    var y := p.y
                end
            """,
            "expected_env": {"x": 10, "y":20}
        },
        {
            "name": "nested initializers",
            "code": """
                struct Point do x:int; y:int; end
                struct Rect do
                    topleft:Point; bottomright:Point
                end
                def main() do
                    var r:Rect = {{1, 2}, {3, 4}}
                    var a:= r.topleft.x
                    var b:= r.topleft.y
                    var c:= r.bottomright.x
                    var d:= r.bottomright.y
                end
            """,
            "expected_env": {"a": 1, "b":2, "c":3, "d":4}
        },
        {
            "name": "nested partial initializers",
            "code": """
                struct Point do x:int; y:int; end
                struct Rect do
                    topleft:Point; bottomright:Point
                end
                def main() do
                    var r:Rect = {{1}}
                    var a:= r.topleft.x
                    var b:= r.topleft.y
                    var c:= r.bottomright.x
                    var d:= r.bottomright.y
                end
            """,
            "expected_env": {"a": 1, "b":0, "c":0, "d":0}
        },
        {
            "name": "mixed types in tuple initializer",
            "code": """
                def main() do
                    var t := {1, "hello", 2.5}
                    var a:= t._0
                    var b:= t._1
                    var c:= t._2
                end
            """,
            "expected_env": {"a": 1, "b":"hello", "c":2.5}
        },
        {
            "name": "struct initializer with multiple fields",
            "code": """
                struct Person do
                    name:string
                    age:int
                    height:float
                end
                def main() do
                    var p:Person = {"John", 30, 1.85}
                    var a:= p.name
                    var b:= p.age
                    var c:= p.height
                end
            """,
            "expected_env": {"a": "John", "b": 30, "c": 1.85}
        },
        {
            "name": "nested tuple in struct initializer",
            "code": """
                struct Container do
                    data:{int, string}
                end
                def main() do
                    var c:Container = {{1, "hello"}}
                    var i := c.data._0
                    var s := c.data._1
                end
            """,
            "expected_env": {"i": 1, "s": "hello"}
        },
        {
            "name": "struct literal in global scope",
            "code": """
                struct Color do Red:int; Green:int; Blue:int; end
                const Color:Color={0,1,2}
                def main() do
                    var x := Color.Red;
                    var y := Color.Green;
                    var z := Color.Blue
                end
            """,
            "expected_env": {"x": 0, "y": 1, "z": 2}
        },
        {
           "name": "nested initializer",
           "code": """
		struct Foo do x:int; y:int; end;
                struct Bar do foo:Foo; z:int; end;
                def main() do var b1:=Bar(); var b2:Bar=new Bar(); b2={{1,2},3}; var x:= b2.foo.x; var z:=b2.z; end
           """,
           "expected_env": {"x": 1, "z":3}
        },
        {
           "name": "method call, no constructor",
           "code": """
		struct Foo do x:int; end;
                def Foo.inc(n:int):Foo do var fn:=Foo(); fn.x=n; return fn; end
                def main() do var f:=Foo(); var f2:= f.inc(1); var x:= f2.x; end
           """,
           "expected_env": {"x": 1}
        },
        {
           "name": "method call escape analyis",
           "code": """
		struct Foo do x:int; end;
		struct Bar do x:int; end;
                def Foo.init() do end
                def Foo.inc(n:int):Foo do var fn:=Foo(); fn.x=n; return fn; end
                def main() do var f:=Foo(); var b:Bar={1}; var f2:= f.inc(1); var x:= f2.x; end
           """,
           "expected_env": {"x": 1}
        },
        {
            "name": "variable declaration with type inference (:=)",
            "code": "def main() do var x := 5; end",
            "expected_env": {"x": 5}
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
            "name": "float literal with type inference",
            "code": "def main() do var y := 3.14; end",
            "expected_env": {"y": 3.14}
        },
        {
            "name": "explicit int type annotation",
            "code": "def main() do var x: int = 10; end",
            "expected_env": {"x": 10}
        },
        {
            "name": "explicit float type annotation",
            "code": "def main() do var y: float = 2.718; end",
            "expected_env": {"y": 2.718}
        },
        {
            "name": "const with type inference",
            "code": "def main() do const x := 42; end",
            "expected_env": {"x": 42}
        },
        {
            "name": "assignment of same type variable",
            "code": "def main() do var x := 5; var y := x; end",
            "expected_env": {"x": 5, "y": 5}
        },
        {
            "name": "assignment expression in while condition (not a comparison)",
            "code": "def main() do var x := 0; var y := 0; while x = y do y = y + 1; end; end",
            "expected_env": {"x": 0, "y": 0}
        },
        {
            "name": "Tests operator precedence in expressions (* has higher precedence than +)",
            "code": "def main() do var x := 5 + 3 * 2; end",
            "expected_env": {"x": 11}
        },
        {
            "name": "basic if statement with equality comparison",
            "code": "def main() do var x := 1; if x == 1 do x = 42; end end",
            "expected_env": {"x": 42}
        },
        {
            "name": "if-else statement (true condition branch taken)",
            "code": "def main() do var x := 1; var result := 0; if x == 1 do result = x; end else do result = 0; end end",
            "expected_env": {"x": 1, "result": 1}
        },
        {
            "name": "bitwise OR operator (|)",
            "code": "def main() do var x := 5 | 3; end",
            "expected_env": {"x": 7}
        },
        {
            "name": "bitwise AND operator (&)",
            "code": "def main() do var x := 5 & 3; end",
            "expected_env": {"x": 1}
        },
        {
            "name": "combination of bitwise operations with variable references",
            "code": "def main() do var x := 5 | 3; var y := x & 2; end",
            "expected_env": {"x": 7, "y": 2}
        },
        {
            "name": "logical AND in if condition (evaluates to false)",
            "code": "def main() do var x := 1; var y := 0; var result := 0; if x and y do result = x; end end", 
            "expected_env": {"x": 1, "y": 0, "result": 0}
        },
        {
            "name": "xor operator with keywords",
            "code": "def main() do var x := 5 xor 3; end",
            "expected_env": {"x": 6}  # 5 xor 3 = 6
        },
        {
            "name": "xor operator with variable references",
            "code": "def main() do var x := 7; var y := x xor 2; end",
            "expected_env": {"x": 7, "y": 5}  # 7 xor 2 = 5
        },
        {
            "name": "bitwise NOT unary operator",
            "code": "def main() do var x := 15; var y := bitnot 3; end",
            "expected_env": {"x": 15, "y": -4}
        },
        {
            "name": "else-if construct (first false, second true)",
            "code": "def main() do var x := 1; var result := 0; if x == 0 do result = 0; end else if x == 1 do result = 1; end end",
            "expected_env": {"x": 1, "result": 1}
        },
        {
            "name": "multiple else-if branches (first & second false, third true)",
            "code": "def main() do var result:= 0; var x := 3; if x == 1 do result=1; end else if x == 2 do result=2; end else if x == 3 do result=42; end end",
            "expected_env": {"x": 3, "result":42}
        },
        {
            "name": "mixed int and float operations",
            "code": "def main() do var x: int = 5; var y: float = 2.5; var z := y; end", 
            "expected_env": {"x": 5, "y": 2.5, "z": 2.5}  # Variable declaration with inferred type from another variable
        },
        {
            "name": "int division",
            "code": "def main() do var x := 10; var y := 3; var z := x / y; end",
            "expected_env": {"x": 10, "y": 3, "z": 3}  # Integer division
        },
        {
            "name": "float division",
            "code": "def main() do var x := 10.0; var y := 3.0; var z := x / y; end",
            "expected_env": {"x": 10.0, "y": 3.0, "z": 3.3333333333333335}  # Float division
        },
        {
            "name": "Tests assignment as expression in while condition",
            "code": "def main() do var x := 10; var y := 0; while (y = y + 1) < 5 do x -= 1; end end",
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
            "name": "Struct with no fields",
            "code": """
                struct EmptyStruct do end
                def main() do end
            """,
            "expected_env": {}
        },
        {
           "name": "constructing struct without init",
           "code": """
		struct NoInit do x:int; end
                def main() do var x:= NoInit(); var y:=x.x; end
           """,
           "expected_env": {"y": 0}
        },
        {
           "name": "recursion",
           "code": """
		def fib(n:int):int do if n <= 1 do return n; end
			return fib(n-1) + fib(n-2) ; end
		def main() do var result:= fib(10); end
           """,
           "expected_env": {"result": 55}
        },
        {
           "name": "method chaining",
           "code": """
		struct Foo do x:int; end;
                def Foo.init(x:int) do self.x = x; end
                def Foo.inc(n:int):Foo do return Foo(self.x + n); end
                def main() do var x:= Foo(1).inc(1).inc(1).inc(1).x; end
           """,
           "expected_env": {"x": 4}
        },
        {
            "name": "generic map implementation",
            "code": """
                struct List<T> do
                    data: T[]      // Array to hold elements
                    len: int       // Current number of elements
                    capa: int      // Current capacity
                end
        
                def List<T>.init() do
                    self.data = nil
                    self.len = 0
                    self.capa = 0
                end
        
                def List<T>.add(value: T) do
                    // Check if we need to resize
                    if self.len + 1 > self.capa do
                        var new_capa: int
                        if self.capa == 0 do
                            new_capa = 1
                        end else do
                            new_capa = self.capa * 2  // Double capacity for efficiency
                        end
                        self.data = new(self.data, new_capa)
                        self.capa = new_capa
                    end
        
                    // Add the new element
                    self.data[self.len] = value
                    self.len = self.len + 1
                end
        
                def List<T>.get(index: int): T do
                    if index < 0 or index >= self.len do
                        // Basic bounds checking
                        print("Index out of bounds")
                    end
                    return self.data[index]
                end
        
                struct Map<K,V> do
                    keys: K[]      // Array to hold keys
                    values: V[]    // Array to hold values
                    len: int       // Current number of key-value pairs
                    capa: int      // Current capacity
                end
        
                def Map<K,V>.init() do
                    self.keys = nil
                    self.values = nil
                    self.len = 0
                    self.capa = 0
                end
        
                def Map<K,V>.set(key: K, value: V) do
                    // First check if key already exists
                    var i := 0
                    while i < self.len do
                        if self.keys[i] == key do
                            // Key found, update value
                            self.values[i] = value
                            return
                        end
                        i = i + 1
                    end
                    
                    // Key not found, need to add new entry
                    // Check if we need to resize
                    if self.len + 1 > self.capa do
                        var new_capa: int
                        if self.capa == 0 do
                            new_capa = 1
                        end else do
                            new_capa = self.capa * 2  // Double capacity for efficiency
                        end
                        self.keys = new(self.keys, new_capa)
                        self.values = new(self.values, new_capa)
                        self.capa = new_capa
                    end
        
                    // Add the new key-value pair
                    self.keys[self.len] = key
                    self.values[self.len] = value
                    self.len = self.len + 1
                end
        
                def Map<K,V>.get(key: K, default_value: V): V do
                    var i := 0
                    while i < self.len do
                        if self.keys[i] == key do
                            // Key found, return associated value
                            return self.values[i]
                        end
                        i = i + 1
                    end
                    
                    // Key not found, return default value
                    return default_value
                end
        
                def main() do
                    // Test basic map functionality with primitive types
                    var int_map := new Map<string, int>()
                    int_map.set("one", 1)
                    int_map.set("two", 2)
                    int_map.set("three", 3)
                    
                    var val1 := int_map.get("one", 0)
                    var val2 := int_map.get("two", 0)
                    var val3 := int_map.get("nonexistent", -1)  // Should get default -1
                    
                    // Update a value
                    int_map.set("one", 100)
                    var updated_val := int_map.get("one", 0)
                    
                    // Test map with List<int> as values
                    var list_map := new Map<string, List<int>>()
                    
                    // Create and populate a list
                    var numbers := new List<int>()
                    numbers.add(10)
                    numbers.add(20)
                    numbers.add(30)
                    
                    // Create another list
                    var more_numbers := new List<int>()
                    more_numbers.add(100)
                    more_numbers.add(200)
                    
                    // Add lists to map
                    list_map.set("numbers", numbers)
                    list_map.set("more_numbers", more_numbers)
                    
                    // Retrieve and verify list contents
                    var retrieved_list := list_map.get("numbers", new List<int>())
                    var list_val1 := retrieved_list.get(0)  // Should be 10
                    var list_val2 := retrieved_list.get(1)  // Should be 20
                    
                    var other_list := list_map.get("more_numbers", new List<int>())
                    var other_val := other_list.get(1)      // Should be 200
                    
                    // Modify list through map
                    var updated_list := list_map.get("numbers", new List<int>())
                    updated_list.add(40)
                    
                    // Verify list was updated
                    var final_len := list_map.get("numbers", new List<int>()).len  // Should be 4
                    var final_val := list_map.get("numbers", new List<int>()).get(3)  // Should be 40
                end
            """,
            "expected_env": {
                "val1": 1,
                "val2": 2,
                "val3": -1,
                "updated_val": 100,
                "list_val1": 10,
                "list_val2": 20,
                "other_val": 200,
                "final_len": 4,
                "final_val": 40
            }
        },
        {
            "name": "generic list implementation",
            "code": """
                struct List<T> do
                    data: T[]      // Array to hold elements
                    len: int       // Current number of elements
                    capa: int      // Current capacity
                end

                def List<T>.init() do
                    self.data = nil
                    self.len = 0
                    self.capa = 0
                end

                def List<T>.add(value: T) do
                    // Check if we need to resize
                    if self.len + 1 > self.capa do
                        var new_capa: int
                        if self.capa == 0 do
                            new_capa = 1
                        end else do
                            new_capa = self.capa * 2  // Double capacity for efficiency
                        end
                        self.data = new(self.data, new_capa)
                        self.capa = new_capa
                    end

                    // Add the new element
                    self.data[self.len] = value
                    self.len = self.len + 1
                end

                def List<T>.get(index: int): T do
                    if index < 0 or index >= self.len do
                        // Basic bounds checking
                        print("Index out of bounds")
                    end
                    return self.data[index]
                end

                struct Person do
                    name: string
                    age: int
                end

                def Person.init(n: string, a: int) do
                    self.name = n
                    self.age = a
                end

                def main() do
                    // Test with integers
                    var int_list := new List<int>()
                    int_list.add(42)
                    int_list.add(99)
                    int_list.add(123)

                    var val1 := int_list.get(0)
                    var val2 := int_list.get(1)
                    var val3 := int_list.get(2)
                    var list_len := int_list.len
                    var list_capa := int_list.capa

                    // Test with struct elements
                    var person_list := new List<Person>()
                    person_list.add(new Person("Alice", 30))
                    person_list.add(new Person("Bob", 25))

                    var name1 := person_list.get(0).name
                    var age2 := person_list.get(1).age
                    var person_len := person_list.len
                end
            """,
            "expected_env": {
                "val1": 42,
                "val2": 99,
                "val3": 123,
                "list_len": 3,
                "list_capa": 4,  # After doubling from 2 to 4 when adding the third element
                "name1": "Alice",
                "age2": 25,
                "person_len": 2
            }
        },
        {
            "name": "dynamic array resize basic functionality",
            "code": """
                def main() do
                    // Test 1: Initialize a dynamic array with new
                    var x: int[] = nil
                    x = new(x, 5)
                    // Set values to verify persistence
                    x[0] = 10
                    x[1] = 20
                    x[2] = 30
                    x[3] = 40
                    x[4] = 50
                    // Check size and values
                    var initial_size := 5
                    // Test 2: Resize to larger (verify value persistence and default init)
                    x = new(x, 8)
                    var larger_size := 8
                    var preserved_value := x[2]  // Should still be 30
                    var new_element := x[7]      // Should be 0 (default)
                    // Test 3: Resize to smaller (verify truncation)
                    x = new(x, 3)
                    var smaller_size := 3
                    var still_preserved := x[1]  // Should still be 20
                    // Test 4: Array assignment compatibility
                    var y := new int[3]
                    y[0] = 100
                    y[1] = 200
                    y[2] = 300
                    // Assign fixed array to dynamic array
                    x = y
                    var assigned_value := x[0]   // Should be 100
                    // Test 5: Dynamic array to dynamic array
                    var z: int[] = nil
                    z = new(z, 2)
                    z[0] = 42
                    z[1] = 84
                    x = z
                    var z_value := x[1]          // Should be 84
                end
            """,
            "expected_env": {
                "initial_size": 5,
                "larger_size": 8,
                "preserved_value": 30,
                "new_element": 0,
                "smaller_size": 3,
                "still_preserved": 20,
                "assigned_value": 100,
                "z_value": 84
            }
        },
        {
            "name": "dynamic array complex operations",
            "code": """
                struct Wrapper do
                    nums: int[]
                end
                def main() do
                    // Test 1: Nested dynamic arrays in structs
                    var wrapper := new Wrapper //{nums:nil}
                    wrapper.nums = new(wrapper.nums, 3)
                    wrapper.nums[0] = 11
                    wrapper.nums[1] = 22
                    wrapper.nums[2] = 33
                    var struct_array_val := wrapper.nums[1]
                    // Test 2: Multi-step resize
                    var a: int[] = nil
                    a = new(a, 2)
                    a[0] = 42
                    a[1] = 43

                    a = new(a, 4)
                    a[2] = 44
                    a[3] = 45

                    a = new(a, 1)
                    var kept_value := a[0]
                end
            """,
            "expected_env": {
                "struct_array_val": 22,
                "kept_value": 42,
            }
        },
        {
            "name": "dynamic array resize errors",
            "code": """
                def main() do
                    // Try to resize a fixed-size array
                    var fixed := new int[5]
                    fixed = new(fixed, 10)
                end
            """,
            "expected_error": "First argument to new() must be a dynamic array"
        },
        {
            "name": "dynamic array resize type error",
            "code": """
                def main() do
                    // Try to resize a non-array type
                    var x := 42
                    x = new(x, 10)
                end
            """,
            "expected_error": "First argument to new() must be a dynamic array"
        },
        {
            "name": "dynamic array incompatible assignment",
            "code": """
                def main() do
                    // Try to assign between arrays with different element types
                    var int_array: int[] = nil
                    var float_array: float[] = nil
                    int_array = new(int_array, 3)
                    float_array = new(float_array, 2)
                    int_array = float_array
                end
            """,
            "expected_error": "Type mismatch: can't assign a value of type heap_ref"
        },
        {
            "name": "heap primitive type",
            "code": """
                def main() do
                    var x:= new int
                end
            """,
            "expected_env": {"x": 0}
        },
        {
            "name": "nil initialization test",
            "code": """
                struct TestRef do value: int; end
                def main() do
                    // Test 1: Basic nil assignment and comparison
                    var ref: TestRef = nil
                    var is_nil := 0
                    if ref == nil do
                        is_nil = 1
                    end
                    // Test 2: Nil in conditional
                    var conditional_result := 0
                    if not ref do
                        conditional_result = 1
                    end
                    // Test 3: Dynamic integer array initialized to nil
                    var intArray: int[] = nil
                    var intArrayIsNil := 0
                    if intArray == nil do
                        intArrayIsNil = 1
                    end
                    // Test 4: Dynamic array of references initialized to nil
                    var refArray: TestRef[] = nil
                    var refArrayIsNil := 0
                    if refArray == nil do
                        refArrayIsNil = 1
                    end
                end
            """,
            "expected_env": {
                "is_nil": 1,
                "conditional_result": 1,
                "intArrayIsNil": 1,
                "refArrayIsNil": 1
            }
        },
        {
           "name": "struct array initialization test",
           "code": """
                struct Point do x: int; y: int; end
                def Point.init(x_val: int, y_val: int) do
                    self.x = x_val
                    self.y = y_val
                end
                def Point.sum() : int do
                    return self.x + self.y;
                end
                def main() do
                    // Test 1: Stack array with constructor calls
                    var stack_points :Point[]= {Point(1,2), Point(3,4), Point(5,6)}
                    var stack_sum := stack_points[1].sum()  // Should be 7 (3+4)

                    // Test 2: Zero-initialized array (no constructors)
                    var zero_points: Point[3]  // Zero-initialized
                    var zero_sum := zero_points[0].sum()  // Should be 0 (0+0)

                    // Test 3: Heap array without constructors
                    var heap_points := new Point[2]  // Zero-initialized
                    var heap_sum := heap_points[0].sum()  // Should be 0 (0+0)

                    // Test 4: Heap array with default constructor
                    // Note: This syntax is for future implementation
                    var heap_init_points := new Point[2](0, 0)
                    heap_init_points[0].init(10, 20)
                    var init_sum := heap_init_points[0].sum()  // Should be 30 (10+20)

                    // Return values to check initialization behavior
                    var result := {
                        stack_sum,    // 7
                        zero_sum,     // 0
                        heap_sum,     // 0
                        init_sum      // 30
                    }
                end
           """,
           "expected_env": {"stack_sum": 7, "zero_sum": 0, "heap_sum": 0, "init_sum": 30 }
        },
        {
            "name": "basic enum declaration and use",
            "code": """
                enum Color do
                    Red;
                    Green;
                    Blue
                end
                def main() do
                    var x := Color.Red;
                    var y := Color.Green;
                    var z := Color.Blue
                end
            """,
            "expected_env": {"x": 0, "y": 1, "z": 2}
        },
        {
            "name": "enum with explicit base type",
            "code": """
                enum Status:u8 do
                    Success;
                    Error;
                    Pending
                end
                def main() do
                    var s := Status.Error;
                    var p := Status.Pending
                end
            """,
            "expected_env": {"s": 1, "p": 2}
        },
        {
            "name": "enum with explicit values",
            "code": """
                enum HttpStatus do
                    OK = 200;
                    NotFound = 404;
                    ServerError = 500
                end
                def main() do
                    var ok := HttpStatus.OK;
                    var notfound := HttpStatus.NotFound;
                    var error := HttpStatus.ServerError
                end
            """,
            "expected_env": {"ok": 200, "notfound": 404, "error": 500}
        },
        {
            "name": "enum with mixed explicit and auto-increment values",
            "code": """
                enum Priority do
                    Low = 10;
                    Medium;
                    High;
                    Critical = 100
                end
                def main() do
                    var l := Priority.Low;
                    var m := Priority.Medium;
                    var h := Priority.High;
                    var c := Priority.Critical
                end
            """,
            "expected_env": {"l": 10, "m": 11, "h": 12, "c": 100}
        },
        {
            "name": "enum used in function parameters",
            "code": """
                enum Direction do
                    North;
                    East;
                    South;
                    West
                end
                def move(dir:Direction) :int do
                    if dir == Direction.North do
                        return 1
                    end
                    if dir == Direction.East do
                        return 2
                    end
                    return 0
                end
                def main() do
                    var n := move(Direction.North);
                    var e := move(Direction.East);
                    var s := move(Direction.South)
                end
            """,
            "expected_env": {"n": 1, "e": 2, "s": 0}
        },
        {
            "name": "nested enum declaration - error",
            "code": """
                def test() do
                    enum Invalid do
                        One;
                        Two
                    end
                    var x := Invalid.One
                end
                def main() do
                    test()
                end
            """,
            "expected_error": "Enum definitions are not allowed inside functions"
        },
        {
            "name": "duplicate enum name - error",
            "code": """
                enum Status do
                    Ok;
                    Error
                end
                enum Status do
                    Success;
                    Failure
                end
                def main() do
                    var x := Status.Ok
                end
            """,
            "expected_error": "Type 'Status' is already defined"
        },
        {
            "name": "non-literal enum value - error",
            "code": """
                def get_value() :int do
                    return 42
                end
                enum Invalid do
                    One = 1;
                    Two = get_value()
                end
                def main() do
                    var x := Invalid.One
                end
            """,
            "expected_error": "Only numeric literals supported for enum values"
        },
        {
            "name": "assignment to enum value - error",
            "code": """
                enum Color do
                    Red;
                    Green;
                    Blue
                end
                def main() do
                    Color.Green = 100
                end
            """,
            "expected_error": "Cannot modify field 'Green' of constant 'Color'"
        },
        {
            "name": "missing end token - error",
            "code": """
                enum Incomplete do
                    First;
                    Second;
                    Third
                def main() do
                    var x := Incomplete.First
                end
            """,
            "expected_error": "Expected enum member identifier"
        },
        {
            "name": "initializer assignments in expressions",
            "code": """
                def main() do
                    var x := 0
                    var y := 0
                    // Array initializer in assignment expression
                    if x = 1 do
                        var arr: int[3] = {10, 20, 30}
                        y = arr[1]  // y = 20
                    end
                    var nums: int[3] = {0, 0, 0}
                    // Test initializer as rhs in an if condition
                    if nums = {5, 6, 7} do
                        x = 42
                    end
                    var sum := nums[0] + nums[1] + nums[2]  // 5 + 6 + 7 = 18
                end
            """,
            "expected_env": {"x": 42, "y": 20, "sum": 18}
        },
        {
            "name": "variable assignments in expressions",
            "code": """
                def main() do
                    var x := 0
                    var y := 0
                    var z := 0
                    // Simple assignment in if condition
                    if x = 42 do
                        y = 10
                    end
                    // Assignment with initializer in if condition
                    if z = {1, 2, 3}._1 do
                        y = 20
                    end
                    var result := x + z  // Should be 42 + 2 = 44
                end
            """,
            "expected_env": {"x": 42, "y": 20, "z": 2, "result": 44}
        },
        {
            "name": "array byref parameter modification",
            "code": """
                def modifyArray(byref arr: int[3]) do
                    arr[0] = 99
                    arr[1] = 88
                    arr[2] = 77
                end
                def main() do
                    var nums: int[3] = {1, 2, 3}
                    modifyArray(nums)
                    var n_0 := nums[0]
                    var n_1 := nums[1]
                    var n_2 := nums[2]
                end
            """,
            "expected_env": {"n_0": 99, "n_1": 88, "n_2": 77}
        },
        {
            "name": "array byval parameter (no changes to original)",
            "code": """
                def modifyArrayByVal(arr: int[3]) do
                    arr[0] = 99  // Should not affect original
                    arr[1] = 88
                end
                def main() do
                    var nums: int[3] = {1, 2, 3}
                    modifyArrayByVal(nums)
                    var n_0 := nums[0]  // Should still be 1
                    var n_1 := nums[1]  // Should still be 2
                end
            """,
            "expected_env": {"n_0": 1, "n_1": 2}
        },
        {
            "name": "array from heap with byref parameter",
            "code": """
                struct ArrayHolder do values: int[3]; end
                def modifyHeapArray(byref arr: int[3]) do
                    arr[0] = 99
                    arr[1] = 88
                end
                def main() do
                    var holder := new ArrayHolder()
                    holder.values = {1, 2, 3}
                    modifyHeapArray(holder.values)
                    var n_0 := holder.values[0]
                    var n_1 := holder.values[1]
                end
            """,
            "expected_env": {"n_0": 99, "n_1": 88}
        },
        {
            "name": "array byref vs byval parameters",
            "code": """
                def multiplyArrays(source: int[3], byref target: int[3]) do
                    // Multiply each element in source with target and store in target
                    target[0] = source[0] * target[0]
                    target[1] = source[1] * target[1]
                    target[2] = source[2] * target[2]
                end
                def main() do
                    var arr1: int[3] = {2, 3, 4}
                    var arr2: int[3] = {5, 6, 7}
                    // Pass arr1 by value, arr2 by reference
                    multiplyArrays(arr1, arr2)
                    // Try to modify arr1 to prove it was passed by value
                    arr1[0] = 99
                    // Store results for verification
                    var a1_0 := arr1[0]  // Should be 99 (local modification)
                    var a2_0 := arr2[0]  // Should be 10 (2*5)
                    var a2_1 := arr2[1]  // Should be 18 (3*6)
                    var a2_2 := arr2[2]  // Should be 28 (4*7)
                end
            """,
            "expected_env": {
                "a1_0": 99,
                "a2_0": 10,
                "a2_1": 18,
                "a2_2": 28
            }
        },
        {
            "name": "array element assignment",
            "code": """
                def main() do
                    var nums: int[5] = {1, 2, 3, 4, 5}
                    nums[2] = 99     // Modify the 3rd element
                    var inferred: int[] = {10, 20, 30}
                    inferred[1] = 50  // Modify the 2nd element
                    var n_2 := nums[2]
                    var i_1 := inferred[1]
                end
            """,
            "expected_env": {"n_2": 99, "i_1": 50}
        },
        {
            "name": "basic typedef test",
            "code": """
                typedef Integer: int
                typedef Points: int[5]
                def main() do
                    var a: Integer = 42
                    var points: Points = {10, 20, 30, 40, 50}
                    points[2] = 99  // Modify using array semantics
                    var raw_int: int = a + 10
                    var point_val := points[2]
                    var type_compat := a + point_val  // Should work since both are ultimately ints
                end
            """,
            "expected_env": {"a": 42, "raw_int": 52, "point_val": 99, "type_compat": 141}
        },
        {
           "name": "struct copy",
           "code": """
		struct Point do x:int; y:int end;
		def main() do
			var p:Point={20,30};
			var q:Point={1,2};
			q = p
			var y:= q.y
		end
           """,
           "expected_env": {"y": 30}
        },
        {
            "name": "byref return basic test",
            "code": """
                def get_ref(byref x:int):byref int do
                    return x;
                end
                def main() do
                    var a:int = 10;
                    var b:int = 20;
                    get_ref(a) = 42;  // Modify a through returned reference
                end
            """,
            "expected_env": {"a": 42, "b": 20}
        },
        {
            "name": "byref parameter modification",
            "code": """
                def modify(byref x:int) do
                    x = 42;
                end
                def main() do
                    var x:int = 10;
                    modify(x);
                end
            """,
            "expected_env": {"x": 42}
        },
        {
            "name": "byref return from global",
            "code": """
                var global:int = 5;
                def get_global_ref() :byref int do
                    return global;
                end
                def main() do
                    get_global_ref() = 100;
                    var global_x := global
                end
            """,
            "expected_env": {"global_x": 100}
        },
        {
            "name": "assign to byref struct return from heap",
            "code": """
                struct Point do x:int; y:int; end
                def get_ref() :byref Point do
                    return new Point();
                end
                def main() do
                    get_ref() = {100, 200};
                end
            """,
            "expected_env": {}
        },
        {
            "name": "assign to byref return from heap",
            "code": """
                def get_ref() :byref int do
                    return new int;
                end
                def main() do
                    get_ref() = 100;
                end
            """,
            "expected_env": {}
        },
        {
            "name": "byref reference forwarding",
            "code": """
                def forward(byref x:int) :byref int do
                    return x;
                end
                def main() do
                    var x:int = 10;
                    forward(x) = 99;
                end
            """,
            "expected_env": {"x": 99}
        },
        {
            "name": "byref return local variable - error",
            "code": """
                def unsafe() :byref int do
                    var local:int = 10;
                    return local;
                end
                def main() do
                    var ref := unsafe();
                    print(ref);
                end
            """,
            "expected_error": "Cannot return a reference to a local variable"
        },
        {
            "name": "byref return from heap - allowed",
            "code": """
                struct Point do x:int; y:int; end
                def get_heap_ref() :byref Point do
                    return new Point();
                end
                def main() do
                    var p := get_heap_ref();
                    p.x = 42;
                    var x:= p.x
                end
            """,
            "expected_env": {"x": 42}
        },
        {
            "name": "byref return chaining",
            "code": """
                def get_ref2(byref x:int) :byref int do
                    return x;
                end
                def get_ref1(byref x:int) :byref int do
                    return get_ref2(x);
                end
                def main() do
                    var value:int = 10;
                    get_ref1(value) = 55;
                    print(value);
                end
            """,
            "expected_env": {"value": 55}
        },
        {
            "name": "byref return with compound assignment",
            "code": """
                def get_ref(byref x:int) :byref int do
                    return x;
                end
                def main() do
                    var num:int = 10;
                    get_ref(num) += 5;  // Should increment by 5
                    print(num);
                end
            """,
            "expected_env": {"num": 15}
        },
        {
            "name": "byref return with struct fields",
            "code": """
                struct Point do x:int; y:int; end
                def get_x_ref(byref p:Point) :byref int do
                    return p.x;
                end
                def main() do
                    var p:Point = {10, 20};
                    get_x_ref(p) = 99;
                    print(p.x);
                    print(p.y);
                end
            """,
            "expected_error": "return type must return a reference"
        },
        {
            "name": "reference modification through assignment",
            "code": """
                def get_ref(byref x:int) :byref int do
                    return x;
                end
                def main() do
                    var x:int = 10;
                    var ref := get_ref(x);
                    ref = 42;
                end
            """,
            "expected_env": {"x": 42}
        },
        {
            "name": "reference retargeting behavior",
            "code": """
                def get_ref(byref x:int) :byref int do
                    return x;
                end
                def main() do
                    var x:int = 10;
                    var y:int = 20;
                    var ref := get_ref(x);
                    ref = y;  // Should modify x, not retarget ref
                    y = 30;   // Changing y shouldn't affect x
                end
            """,
            "expected_env": {"x": 20, "y": 30}
        },
        {
            "name": "implicit reference with explicit typing",
            "code": """
                struct Point do x:int; y:int; end
                def get_point() :byref Point do
                    return new Point();
                end
                def main() do
                    var p:Point = get_point();
                    p.x = 42;
                    var x:= p.x;
                end
            """,
            "expected_env": {"x": 42}
        },
        {
            "name": "double byref indirection",
            "code": """
                def update_through_ref(byref x:int) do
                    x = 99;
                end
                def get_ref(byref x:int) :byref int do
                    return x;
                end
                def main() do
                    var value:int = 42;
                    update_through_ref(get_ref(value));
                end
            """,
            "expected_env": {"value": 99}
        },
        {
            "name": "nested byref functions",
            "code": """
                def modify(byref x:int) do
                    x = 42;
                end
                def wrapper(byref y:int) do
                    modify(y);
                end
                def main() do
                    var z:int = 10;
                    wrapper(z);
                end
            """,
            "expected_env": {"z": 42},
        },
        {
            "name": "references with arithmetic expressions",
            "code": """
                def get_ref(byref x:int) :byref int do
                    return x;
                end
                def main() do
                    var x:int = 10;
                    var ref := get_ref(x);
                    ref = ref * 2;
                    ref *= 2
                end
            """,
            "expected_env": {"x": 40}
        },
        {
           "name": "byref call 1",
           "code": """
		def fun(byref y:int) do y=42; end
		def main() do var x:=11; fun(x); end
           """,
           "expected_env": {"x": 42}
        },
        {
           "name": "byref call 1 with param/variable name shadowing",
           "code": """
		def fun(byref x:int) do x=42; end
		def main() do var x:=11; fun(x); end
           """,
           "expected_env": {"x": 42}
        },
        {
            "name": "heap allocation with explicit typing",
            "code": """
                struct Point do x:int; y:int; end
                def main() do
                    var p:Point = new Point();
                    p.x = 42;
                    var x:= p.x;
                end
            """,
            "expected_env": {"x": 42}
        },
        {
           "name": "parameter shadowing test",
           "code": '''
                struct Calculator do
                  base: int;
                end
                def Calculator.init(base: int) do
                  self.base = base;
                end
                def process(x: int, y: int): int do
                  // This function processes x and y
                  // If scope is wrong, it might use the outer x and y
                  return x * 10 + y;
                end
                def transform(x: int, y: int): int do
                  // Calls process with SWAPPED parameters
                  // If scope is wrong, it will use the original x and y
                  return process(y, x);  // Note the swap here
                end
                def Calculator.calculate(x: int, y: int): int do
                  // Method with same parameter names
                  // Uses a third value for the same parameters
                  return transform(x+100, y+100);  // Drastically different values
                end
                def main() do
                  var x := 1;
                  var y := 2;
                  var calc := Calculator(50);
                  // Each of these calls has 'x' and 'y' parameters
                  // using shadowed parameters from any wrong scope would result in wrong values
                  var result := calc.calculate(x, y);
                  // Calculation if correct:
                  // 1. calculate is called with x=1, y=2
                  // 2. transform is called with x=101, y=102
                  // 3. process is called with x=102, y=101 (swapped)
                  // 4. process returns 102*10 + 101 = 1021
                  // Incorrect scope handling would yield different values:
                  // - If using x,y from main scope: 1*10 + 2 = 12
                  // - If using x,y from calculate: 1*10 + 2 = 12
                  // - If using x,y from transform but not swapping: 101*10 + 102 = 1112
                end
           ''',
           "expected_env": {"x": 1, "y": 2, "result": 1121}
        },
        {
           "name": "constructor of differing types",
           "code": """
                struct Foo do x: int; label: string; end
                def Foo.init(x: int, name: string) do
                    self.x = x
                    self.label = name;
                end
                def main() do
                    var f := Foo(4, "origin")
                    var x := f.x
                end
           """,
           "expected_env": {"x": 4}
        },
        {
            "name": "operator precedence - unary op vs member access",
            "code": """
                struct BitHolder do value: int; end
                def BitHolder.init(initial: int) do self.value = initial; end
                def BitHolder.apply_not(): int do
                    return bitnot self.value  // This should apply bitnot to self.value, not to self
                end
                def BitHolder.nested_test(): int do
                    self.value = 5
                    return bitnot self.value & 3  // Should be equivalent to (bitnot self.value) & 3, not bitnot (self.value & 3)
                end
                def main() do
                    var b := BitHolder(5)  // value = 5 (binary: 101)
                    var not_result := b.apply_not()  // bitnot 5 should be -6 (binary: ~101 = 11111...1010)
                    var nested_result := b.nested_test()  // Should be (bitnot 5) & 3 = -6 & 3 = 2 (binary: ~101 & 11 = 10)
                end
            """,
            "expected_env": {"not_result": -6, "nested_result": 2}
        },
        {
            "name": "binary expression starting with identifier",
            "code": """
                def main() do
                    var a := 10
                    var b := 20
                    a + b  // Expression statement with result discarded
                end
            """,
            "expected_env": {"a": 10, "b": 20}
        },
        {
            "name": "minimal compound assignment to field issue",
            "code": """
                struct Point do
                    x: float
                end
                def Point.init(val: float) do self.x = val; end
                def main() do
                    var p := Point(1.0)
                    p.x += 2.5
                    const x := p.x
                end
            """,
            "expected_env": {'x': 3.5}
        },
        {
            "name": "reproducer for method chaining issue in statement",
            "code": """
                struct Obj do x: int; end
                def Obj.init(val: int) do self.x = val; end
                def Obj.toggle(): Obj do return self; end
                def main() do
                    var obj := Obj(1)
                    obj.toggle().toggle()
                    const x := obj.x
                end
            """,
            "expected_env": {"x": 1}
        },
        {
            "name": "comprehensive AST node coverage test",
            "code": """
                // Struct definition with inheritance
                struct BaseObject do
                    id: int
                end

                def BaseObject.init(id_val: int) do
                    self.id = id_val
                end

                def BaseObject.getId(): int do
                    return self.id
                end

                struct ComplexObject: BaseObject do
                    name: string
                    value: float
                    enabled: int
                end

                def ComplexObject.init(id_val: int, name_val: string, value_val: float) do
                    self.id = id_val
                    self.name = name_val
                    self.value = value_val
                    self.enabled = 1
                end

                def ComplexObject.toggle(): ComplexObject do
                    self.enabled = bitnot self.enabled & 1
                    return self
                end

                def ComplexObject.fini() do
                    print "Destroying object with id:";
                    print self.id;
                end

                // Global variables
                var global_counter := 0
                const PI := 3.14159

                // Helper function returning tuple
                def divmod(a: int, b: int): {int, int} do
                    return {a/b, a%b};
                end

                // Function with multiple parameters and return value
                def calculate(x: int, y: float, op: string): float do
                    if op == "add" do
                        return x + y;
                    end else if op == "subtract" do
                        return x - y;
                    end else if op == "multiply" do
                        return x * y;
                    end else if op == "divide" do
                        if y == 0.0 do
                            return 0.0;
                        end
                        return x / y;
                    end
                    return 0.0;
                end

                def main() do
                    // Variable declarations with different types
                    var a := 10
                    var b: float = 20.5
                    var c: int = 0

                    // Logical operations
                    var logical_and := a > 5 and b < 30.0
                    var logical_or := a < 5 or b > 10.0
                    var logical_not := !(a == c)

                    // Binary operations
                    var sum := a + b
                    var diff := b - a
                    var product := a * b
                    var quotient := b / a
                    var remainder := a % 3

                    // Bitwise operations
                    var bit_and := a & 3
                    var bit_or := a | 5
                    var bit_xor := a xor 7
                    var bit_not := bitnot a
                    var bit_shift_left := a shl 2
                    var bit_shift_right := a shr 1
                    
                    // Compound assignments
                    var compound := 5
                    compound += 3
                    compound -= 1
                    compound *= 2
                    compound /= 2
                    compound %= 3
                    
                    // While loop with break and continue
                    var i := 0
                    var loop_result := 0
                    while i < 10 do
                        i += 1
                        if i == 3 do
                            continue
                        end
                        if i == 8 do
                            break
                        end
                        loop_result += i
                    end
                    // Tuple usage
                    var division_result := divmod(25, 7)
                    var quotient_from_tuple := division_result._0
                    var remainder_from_tuple := division_result._1

                    // Function calls with different parameter types
                    var calc_result := calculate(5, 2.5, "multiply")

                    // Struct initialization and method calls
                    var obj := ComplexObject(42, "test object", 3.14)
                    var obj_id := obj.getId()    // Inherited method
                    var initial_enabled := obj.enabled
                    obj.toggle()                 // Method call
                    var toggled_enabled := obj.enabled
                    obj.toggle().toggle()        // Method chaining
                    var double_toggled := obj.enabled

                    // Heap allocation and deallocation
                    var heap_obj := new ComplexObject(99, "heap object", 7.5)
                    var heap_obj_id := heap_obj.id
                    var heap_name := heap_obj.name
                    heap_obj.value += 2.5
                    var heap_value := heap_obj.value
                    del heap_obj   // Should trigger fini() method

                    // Global variable access and modification
                    global_counter += 1
                    var pi_value := PI

                    // If-else statements
                    var condition_result := 0
                    if a > 5 do
                        condition_result = 1
                    end else if a < 3 do
                        condition_result = 2
                    end else do
                        condition_result = 3
                    end

                    // Expression statement
                    a + b  // Result is discarded
                    print "End of AstNode coverage test"
                end
            """,
            "expected_env": {
                "a": 10,
                "b": 20.5,
                "c": 0,
                "logical_and": 1,
                "logical_or": 1,
                "logical_not": 1,
                "sum": 30.5,
                "diff": 10.5,
                "product": 205.0,
                "quotient": 2.05,
                "remainder": 1,
                "bit_and": 2,
                "bit_or": 15,
                "bit_xor": 13,
                "bit_not": -11,
                "bit_shift_left": 40,
                "bit_shift_right": 5,
                "compound": 1,
                "i": 8,
                "loop_result": 25,
                "quotient_from_tuple": 3,
                "remainder_from_tuple": 4,
                "calc_result": 12.5,
                "obj_id": 42,
                "initial_enabled": 1,
                "toggled_enabled": 0,
                "double_toggled": 0,
                "heap_obj_id": 99,
                "heap_name": "heap object",
                "heap_value": 10.0,
                "pi_value": 3.14159,
                "condition_result": 1,
                "global_counter": 1
            }
        },
        {
            "name": "operations involving signed and unsigned integers",
            "code": """
		// results havent been verified, they're for coverage testing
		def main() do
			var f1:= 10; var f2:= 2;
			var d1:uint=10; var d2:uint=2
			var frl1 := f1 + f2;
			var frr1 := f2 + f1;
			var mrl1 := f1 + d2;
			var mrr1 := d1 + f2;
			var dr1  := d1 + d2;
			var frl2 := f1 - f2;
			var frr2 := f2 - f1;
			var mrl2 := f1 - d2;
			var mrr2 := d1 - f2;
			var dr2  := d1 - d2;
			var frl3 := f1 * f2;
			var frr3 := f2 * f1;
			var mrl3 := f1 * d2;
			var mrr3 := d1 * f2;
			var dr3  := d1 * d2;
			var frl4 := f1 / f2;
			var frr4 := f2 / f1;
			var mrl4 := f1 / d2;
			var mrr4 := d1 / f2;
			var dr4  := d1 / d2;
			var frl5 := f1 % f2;
			var frr5 := f2 % f1;
			var mrl5 := f1 % d2;
			var mrr5 := d1 % f2;
			var dr5  := d1 % d2;
			var frl6 := f1 == f2;
			var frr6 := f2 == f1;
			var mrl6 := f1 == d2;
			var mrr6 := d1 == f2;
			var dr6  := d1 == d2;
			var frl7 := f1 != f2;
			var frr7 := f2 != f1;
			var mrl7 := f1 != d2;
			var mrr7 := d1 != f2;
			var dr7  := d1 != d2;
			var frl8 := f1 > f2;
			var frr8 := f2 > f1;
			var mrl8 := f1 > d2;
			var mrr8 := d1 > f2;
			var dr8  := d1 > d2;
			var frl9 := f1 >= f2;
			var frr9 := f2 >= f1;
			var mrl9 := f1 >= d2;
			var mrr9 := d1 >= f2;
			var dr9  := d1 >= d2;
			var frl10 := f1 < f2;
			var frr10 := f2 < f1;
			var mrl10 := f1 < d2;
			var mrr10 := d1 < f2;
			var dr10  := d1 < d2;
			var frl11 := f1 <= f2;
			var frr11 := f2 <= f1;
			var mrl11 := f1 <= d2;
			var mrr11 := d1 <= f2;
			var dr11  := d1 <= d2;
			var frl12 := f1 shr f2;
			var frr12 := f2 shr f1;
			var mrl12 := f1 shr d2;
			var mrr12 := d1 shr f2;
			var dr12  := d1 shr d2;
			var frl13 := f1 shl f2;
			var frr13 := f2 shl f1;
			var mrl13 := f1 shl d2;
			var mrr13 := d1 shl f2;
			var dr13  := d1 shl d2;
			var frl14 := bitnot f1;
			var frr14 := bitnot f2;
			var mrl14 := bitnot d1;
			var mrr14 := bitnot d2;
			//var dr14  := d1 shl d2;
			var frl15 := f1 | f2;
			var frr15 := f2 | f1;
			var mrl15 := f1 | d2;
			var mrr15 := d1 | f2;
			var dr15  := d1 | d2;
			var frl16 := f1 & f2;
			var frr16 := f2 & f1;
			var mrl16 := f1 & d2;
			var mrr16 := d1 & f2;
			var dr16  := d1 & d2;
			var frl17 := f1 xor f2;
			var frr17 := f2 xor f1;
			var mrl17 := f1 xor d2;
			var mrr17 := d1 xor f2;
			var dr17  := d1 xor d2;
			var frl18 := f1 or f2;
			var frr18 := f2 or f1;
			var mrl18 := f1 or d2;
			var mrr18 := d1 or f2;
			var dr18  := d1 or d2;
			var frl19 := f1 and f2;
			var frr19 := f2 and f1;
			var mrl19 := f1 and d2;
			var mrr19 := d1 and f2;
			var dr19  := d1 and d2;
		end
            """,
            "expected_env": {'frr10': 1, 'frr11': 1, 'frr12': 0, 'frr13': 2048, 'frr14': -3, 'frr15': 10, 'frr16': 2, 'frr17': 8, 'frr18': 1, 'mrl1': 12, 'dr2': 8, 'mrl2': 8, 'd2': 2, 'd1': 10, 'frr2': -8, 'frr3': 20, 'frr1': 12, 'frr6': 0, 'frr7': 1, 'frr4': 0, 'frr5': 2, 'frr8': 0, 'frr9': 0, 'frr19': 1, 'dr9': 1, 'dr8': 1, 'mrl9': 1, 'mrl8': 1, 'dr1': 12, 'mrl4': 5, 'dr3': 20, 'mrl6': 0, 'dr5': 0, 'dr4': 5, 'dr7': 1, 'dr6': 0, 'frl1': 12, 'frl2': 8, 'frl3': 20, 'frl4': 5, 'frl5': 0, 'frl6': 0, 'frl7': 1, 'frl8': 1, 'frl9': 1, 'dr15': 10, 'dr17': 8, 'dr16': 2, 'dr11': 0, 'dr10': 0, 'dr13': 40, 'dr12': 2, 'dr19': 1, 'dr18': 1, 'mrr17': 8, 'mrr16': 2, 'mrr15': 10, 'mrr14': 4294967293, 'mrr13': 40, 'mrr12': 2, 'mrr11': 0, 'mrr10': 0, 'mrr19': 1, 'mrr18': 1, 'mrr3': 20, 'mrr2': 8, 'mrr1': 12, 'mrr7': 1, 'mrr6': 0, 'mrr5': 0, 'mrr4': 5, 'mrr9': 1, 'mrr8': 1, 'frl18': 1, 'frl19': 1, 'frl12': 2, 'frl13': 40, 'frl10': 0, 'frl11': 0, 'frl16': 2, 'frl17': 8, 'frl14': -11, 'frl15': 10, 'f1': 10, 'f2': 2, 'mrl3': 20, 'mrl19': 1, 'mrl18': 1, 'mrl15': 10, 'mrl14': 4294967285, 'mrl17': 8, 'mrl16': 2, 'mrl11': 0, 'mrl10': 0, 'mrl13': 40, 'mrl12': 2, 'mrl5': 0, 'mrl7': 1}
        },
        {
            "name": "operations involving double and float",
            "code": """
		// results havent been verified, they're for coverage testing
		def main() do
			var f1:= 10.0; var f2:= 2.0;
			var d1:double = 10.0; var d2:double=2.0
			var frl1 := f1 + f2;
			var frr1 := f2 + f1;
			var mrl1 := f1 + d2;
			var mrr1 := d1 + f2;
			var dr1  := d1 + d2;
			var frl2 := f1 - f2;
			var frr2 := f2 - f1;
			var mrl2 := f1 - d2;
			var mrr2 := d1 - f2;
			var dr2  := d1 - d2;
			var frl3 := f1 * f2;
			var frr3 := f2 * f1;
			var mrl3 := f1 * d2;
			var mrr3 := d1 * f2;
			var dr3  := d1 * d2;
			var frl4 := f1 / f2;
			var frr4 := f2 / f1;
			var mrl4 := f1 / d2;
			var mrr4 := d1 / f2;
			var dr4  := d1 / d2;
			//var frl5 := f1 % f2;
			//var frr5 := f2 % f1;
			//var mrl5 := f1 % d2;
			//var mrr5 := d1 % f2;
			//var dr5  := d1 % d2;
			var frl6 := f1 == f2;
			var frr6 := f2 == f1;
			var mrl6 := f1 == d2;
			var mrr6 := d1 == f2;
			var dr6  := d1 == d2;
			var frl7 := f1 != f2;
			var frr7 := f2 != f1;
			var mrl7 := f1 != d2;
			var mrr7 := d1 != f2;
			var dr7  := d1 != d2;
			var frl8 := f1 > f2;
			var frr8 := f2 > f1;
			var mrl8 := f1 > d2;
			var mrr8 := d1 > f2;
			var dr8  := d1 > d2;
			var frl9 := f1 >= f2;
			var frr9 := f2 >= f1;
			var mrl9 := f1 >= d2;
			var mrr9 := d1 >= f2;
			var dr9  := d1 >= d2;
			var frl10 := f1 < f2;
			var frr10 := f2 < f1;
			var mrl10 := f1 < d2;
			var mrr10 := d1 < f2;
			var dr10  := d1 < d2;
			var frl11 := f1 <= f2;
			var frr11 := f2 <= f1;
			var mrl11 := f1 <= d2;
			var mrr11 := d1 <= f2;
			var dr11  := d1 <= d2;
		end
            """,
            "expected_env": {'frr10': 1, 'f1': 10.0, 'f2': 2.0, 'mrl6': 0, 'frr11': 1, 'dr4': 5.0, 'frr9': 0, 'dr7': 1, 'mrr11': 0, 'mrr10': 0, 'dr6': 0, 'd2': 2.0, 'd1': 10.0, 'frr2': -8.0, 'frr3': 20.0, 'frr1': 12.0, 'frr6': 0, 'frr7': 1, 'frr4': 0.2, 'frl1': 12.0, 'dr1': 12.0, 'mrl4': 5.0, 'dr3': 20.0, 'dr2': 8.0, 'mrl1': 12.0, 'mrr1': 12.0, 'mrl3': 20.0, 'mrl2': 8.0, 'mrr3': 20.0, 'mrr2': 8.0, 'frl2': 8.0, 'frl3': 20.0, 'frl4': 5.0, 'mrr6': 0, 'frl6': 0, 'mrr4': 5.0, 'frl8': 1, 'mrr7': 1, 'mrr9': 1, 'mrr8': 1, 'dr9': 1, 'dr8': 1, 'frl7': 1, 'mrl11': 0, 'mrl10': 0, 'frl9': 1, 'mrl9': 1, 'mrl8': 1, 'frl10': 0, 'frl11': 0, 'frr8': 0, 'dr11': 0, 'dr10': 0, 'mrl7': 1}
        },
        {
            "name": "mixing int and float types in binary operation",
            "code": "def main() do var x := 10; var y := 3.0; var z := x / y; end", 
            "expected_env": {"y": 3.0, "x": 10, "z": 3.3333333333333335}
        },
        {
           "name": "calling methods on temporary struct instance",
           "code": """
        struct Vector do x: int; y: int ; end

        def Vector.init(a: int, b: int) do
            self.x = a; self.y = b;
        end

        def Vector.add(other: Vector): Vector do
            return Vector(self.x + other.x, self.y + other.y);
        end

        def Vector.length(): int do
            return self.x * self.x + self.y * self.y;
        end

        def main() do
            var v1 := Vector(3, 4);
            var v2 := Vector(1, 2);
            // Method call on the result of a parenthesized expression
            // nud() sees the opening parenthesis
            var len := (v1.add(v2)).length();
        end
           """,
           "expected_env": {"len": 52}
        },
        {
           "name": "nested method call 1",
           "code": """
                struct Vector do
                    x: int
                    y: int
                end

                def Vector.init(a: int, b: int) do
                    self.x = a
                    self.y = b
                end

                def Vector.length(): int do
                    return self.x * self.x + self.y * self.y
                end

                struct Point do
                    pos: Vector
                    label: string
                end

                def Point.init(x: int, y: int, name: string) do
                    self.pos = Vector(x, y);
                    self.label = name;
                end

                def main() do
                    var p := Point(3, 4, "origin")
                    var len := p.pos.length()
                end
           """,
           "expected_env": {"len": 25}
        },

        {
           "name": "chained method calls",
           "code": """
        struct Counter do
            value: int
        end

        def Counter.init(start: int) do
            self.value = start;
        end

        def Counter.inc(): Counter do
            self.value = self.value + 1;
            return self;
        end

        def Counter.dec(): Counter do
            self.value = self.value - 1;
            return self;
        end

        def Counter.get(): int do
            return self.value;
        end

        def main() do
            var c := Counter(5);
            var result := c.inc().inc().dec().get();
        end
           """,
           "expected_env": {"result": 6}
        },

        {
           "name": "test tuple 1",
           "code": """
                def divmod(a:int, b:int):{int, int} do return {a/b, a%b}; end
                def main() do var result:= divmod(3, 2)
                var x:=result._0 ; var y:=result._1; end
           """,
           "expected_env": {"x": 1, "y": 1}
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
           "name": "test struct inheritance1",
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

    struct Circle:Shape do
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
           "name": "test struct inheritance2",
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
            "name": "functions and return statements",
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
            "name": "function with return value specification",
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
            "name": "global variables",
            "code": """
                var global_rw:=0
                const global_r:=42
                def main() do global_rw=10; var x:=global_rw + global_r
                end
            """,
            "expected_env": {"x": 52}
        },


        # Test cases that are expected to fail
        # Each has "code" and "expected_error"
        {
            "name": "invalid use of := operator",
            "code": """
                def main() do var x:=10; x:=20 // x is already declared, so := must fail
                end
            """,
            "expected_error": "Cannot use ':=' with already declared variable 'x'. Use '=' instead"
        },
        {
            "name": "non-void function with empty return",
            "code": """
                def should_return_int() :int do
                    return
                end
                def main() do
                    var x := should_return_int();
                end
            """,
            "expected_error": "Non-void function 'should_return_int' must return a value"
        },
        {
            "name": "type inference with void function",
            "code": """
                def voidfunc() do
                    return
                end
                def main() do
                    var x := voidfunc();
                end
            """,
            "expected_error": "Cannot assign void expression to variable"
        },

        {
            "name": "invalid redeclaration of variable",
            "code": """
                def main() do var x:=10; var x:=20 // x is already declared, so "var x" must fail
                end
            """,
            "expected_error": "Variable 'x' is already declared in this scope"
        },
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
            "name": "error when using variable without declaration",
            "code": "def main() do x = 5; end",  # Missing var or const declaration
            "expected_error": "Variable 'x' is not declared"
        },
        {
            "name": "error when declaring variable without initialization",
            "code": "def main() do var x; x = 5; end",  # Missing initialization
            "expected_error": "Variable declaration must include either a type annotation or initialization"
        },
        {
            "name": "error when using undeclared variable in a logical expression",
            "code": "def main() do var x := 1; if !x or x and y do print x; end end",
            "expected_error": "Variable 'y' is not declared"
        },
        {
            "name": "error when using undeclared variable in print statement",
            "code": "def main() do print z; end",
            "expected_error": "Variable 'z' is not declared"
        },
        {
            "name": "Tests error when using undeclared variable in if condition",
            "code": "def main() do if x do print 1; end end",
            "expected_error": "Variable 'x' is not declared"
        },
        {
            "name": "error when missing semicolon between statements on the same line",
            "code": "def main() do var x := 5 + 3 print x; end",
            "expected_error": "Expected semicolon between statements"
        },
        {
            "name": "error when missing semicolon between statements on the same line",
            "code": "def main() do var a := 1 var b := 2 end",
            "expected_error": "Expected semicolon between statements"
        },
        {
            "name": "error when reassigning to a const-declared constant",
            "code": "def main() do const x := 5; x = 10; end",
            "expected_error": "Cannot reassign to constant 'x'"
        },
        {
            "name": "error when using compound assignment on a const-declared constant",
            "code": "def main() do const x := 5; x += 10; end",
            "expected_error": "Cannot reassign to constant 'x'"
        },
        {
            "name": "error when using undeclared variable in initialization",
            "code": "def main() do var x := y; end",
            "expected_error": "Variable 'y' is not declared"
        },
        {
            "name": "error when using = without explicit type",
            "code": "def main() do var x = 5; end",
            "expected_error": "requires explicit type annotation"
        },
        {
            "name": "error when redeclaring a variable",
            "code": "def main() do var x := 5; var x := 10; end",
            "expected_error": "already declared"
        },
        {
            "name": "float literal without decimal digits",
            "code": "def main() do var x := 5.; end",
            "expected_error": "Invalid float literal"
        },
        {
            "name": "else-if without proper end before else",
            "code": "def main() do var x := 3; if x == 1 do print 1; else if x == 2 do print 2; else if x == 3 do print 3; end end",
            "expected_error": "Missing 'end' before 'else'"
        },
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
            "expected_error": "Type mismatch in initialization"
        },
        {
            "name": "Code outside functions not allowed",
            "code": "var x := 5;",
            "expected_error": "No 'main' function defined"
        },
    # Test cases for OOP parse errors
    {
        "name": "Method without self",
        "code": """
            struct Point do x: int; end

            def Point.bad(): int do
                return x;  // Missing self.x
            end
        """,
        "expected_error": "Variable 'x' is not declared"
    },
    {
        "name": "Undefined method call",
        "code": """
            struct Point do x: int; end
            def main() do var p := Point(); p.nonexistent(); end
        """,
        "expected_error": "Method 'nonexistent' not found in struct 'Point'"
    },
    {
        "name": "Undefined field access",
        "code": """
            struct Point do x: int; end
            def main() do var p := Point(); print p.y; end
        """,
        "expected_error": "Field 'y' not found in struct 'Point'"
    },
    {
        "name": "Invalid constructor parameter count",
        "code": """
            struct Point do x: int; end
            def Point.init(x: int, y: int) do self.x = x; end
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
            struct Child:NonExistentParent do x: int; end
            def main() do
            end
        """,
        "expected_error": "Parent struct 'NonExistentParent' is not defined"
    },
    {
        "name": "Constructor with non-void return type",
        "code": """
            struct Point do x: int; end
            def Point.init(): int do
                self.x = 5;
                return 0;
            end
        """,
        "expected_error": "Constructor 'init' must have void return type"
    },
    {
        "name": "Destructor with parameters",
        "code": """
            struct Point do x: int; end
            def Point.fini(flag: int) do end
        """,
        "expected_error": "Destructor 'fini' cannot have parameters"
    },
    {
        "name": "Del on stack object",
        "code": """
            struct Point do x: int; end
            def main() do
                var p := Point();
                del p;  // Cannot delete stack object
            end
        """,
        "expected_error": "'del' can only be used with reference types"
    },
    {
        "name": "Missing constructor arguments",
        "code": """
            struct Point do x: int;y: int; end
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
        "name": "Constructor on non-struct type",
        "code": """
            def main() do
                var x := new int(10);
            end
        """,
        "expected_error": "Constructor invocation not allowed for primitive types"
    },
    {
        "name": "New operator on non-existing struct",
        "code": """
            def main() do
                var x := new Foo(10);
            end
        """,
        "expected_error": "Type 'Foo' is not defined"
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
        "name": "new Struct without parens (using default init)",
        "code": """
            struct Counter do
                count: int
            end

            def main() do
                var c := new Counter;  // Missing parentheses
                var x := c.count
            end
        """,
        "expected_env": {"x": 0}
    },
        {
            "name": "struct initializer with wrong field count",
            "code": """
                struct Point do
                    x:int
                    y:int
                end
                def main() do
                    var p:Point = {1, 2, 3}  // Too many values
                end
            """,
            "expected_error": "Initializer for Point has 3 elements, but struct has only 2 fields"
        },
        {
            "name": "struct initializer with type mismatch",
            "code": """
                struct Point do
                    x:int
                    y:string
                end
                def main() do
                    var p:Point = {1, 2}  // y should be string
                end
            """,
            "expected_error": "Field 2 in Point initializer: cannot convert int to string"
        },
        {
            "name": "nested initializer with wrong type",
            "code": """
                struct Point do
                    x:int
                    y:int
                end
                
                struct Rect do
                    topleft:Point
                    bottomright:Point
                end
                
                def main() do
                    var r:Rect = {{1, 2}, {3, "4"}}  // Should be int, not string
                end
            """,
            "expected_error": "Field 2 in Point initializer: cannot convert string to int"
        },
        {
            "name": "initializer in return with type mismatch",
            "code": """
                struct Point do
                    x:int
                    y:int
                end
                
                def makePoint():Point do
                    return {1, "2"}  // y should be int
                end
                
                def main() do
                    var p := makePoint()
                end
            """,
            "expected_error": "Field 2 in Point initializer: cannot convert string to int"
        },
        {
            "name": "empty initializer",
            "code": """
                def main() do
                    var t := {}
                end
            """,
            "expected_error": "Empty initializers are not supported"
        },
        {
            "name": "initializer for struct with constructor",
            "code": """
                struct Point do x:int; y:int end
                def Point.init(x:int, y:int) do
                    self.x = x; self.y = y
                end
                def main() do
                    var p:Point = {10, 20}  // Error: can't use initializer when constructor exists
                end
            """,
            "expected_error": "Cannot use initializer for struct 'Point' because it has a constructor"
        },
]

def values_equal(expected, actual):
    """
    Compare two values with special handling for floating-point numbers.
    Rounds floating-point numbers to 4 decimal places before comparison.

    Args:
        expected: The expected value
        actual: The actual value

    Returns:
        bool: True if values are equal (with rounding for floats), False otherwise
    """
    # Check if both values are floating-point types
    if isinstance(expected, float) and isinstance(actual, float):
        # Round to 4 decimal places before comparing
        return round(expected, 4) == round(actual, 4)
    # For other types, use standard equality comparison
    return expected == actual

def test(use_interpreter=True):
    # Use either Interpreter or CRunner based on parameter
    interpreter = Interpreter() if use_interpreter else CRunner()

    failed_tests = []
    # Dictionary to track test hashes: hash -> [test index]
    test_hashes = {}

    # Run all test cases
    for i, test_case in enumerate(test_cases):
        # Calculate test hash and track duplicates
        test_hash = get_hash(test_case["code"])
        if test_hash not in test_hashes:
            test_hashes[test_hash] = []
        test_hashes[test_hash].append(i)

        interpreter.reset()

        test_num = i + 1
        print("\nTest %d (%s):" % ((test_num), test_case["name"]))
        # don't print the test by default as it's getting too verbose
        # print("Input: %s" % test_case["code"])

        result = interpreter.run(test_case["code"])
        if result.get('ast'):
            devnull = "%s"%(result['ast'])  # stringify ast in any case for code coverage

        # Check if this test is expected to fail
        if "expected_error" in test_case:
            # This is a test that should fail
            if not result['success'] and test_case["expected_error"] in result['error']:
                print("Success! Failed with expected error: %s" % result['error'])
            else:
                printc("red", "Test didn't fail as expected! Result: %s" % result)
                printc("red", "Input: %s" % add_line_numbers(test_case["code"]))
                failed_tests.append(test_num)
                # Add AST dump for unexpected failures
                if result.get('ast'):
                    print("AST dump: %s" % result['ast'])
                if result.get('c_code'):
                    print("C code: %s" % result['c_code'])

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
                mismatches = []
                for k in test_case["expected_env"].keys():
                    v = test_case["expected_env"][k]
                    if k not in env:
                        mismatches.append("{}: want {} got: None".format(k, v))
                    elif not values_equal(v, env[k]):
                        mismatches.append("{}: want {} got: {}".format(k, v, env[k]))

                if len(mismatches) == 0:
                    print("Success! Environment matches expectations.")
                    if result.get('c_code'):
                        print("C code: %s" % result['c_code'])
                else:
                    printc("red", "Input: %s" % add_line_numbers(test_case["code"]))
                    print("Test passed but with incorrect environment values:")
                    print("  Expected env: %s" % test_case["expected_env"])
                    print("  Actual env: %s" % env)
                    print("  Mismatches:")
                    for m in mismatches: print(m)
                    failed_tests.append(test_num)
                    if result.get('ast'):
                        print("AST dump: %s" % result['ast'])
                    if result.get('c_code'):
                        print("C code: %s" % result['c_code'])
                    if os.getenv("DEBUG"):
                        import time
                        time.sleep(10000)

            else:
                printc("red", "Failed! Error: %s" % result['error'])
                printc("red", "Input: %s" % add_line_numbers(test_case["code"]))
                failed_tests.append(test_num)
                if result.get('ast'):
                    print("AST dump: %s" % result['ast'])
                if result.get('c_code'):
                    print("C code: %s" % result['c_code'])
                if os.getenv("DEBUG"):
                    import time
                    time.sleep(10000)

    # Print statistics at the end
    print("\n========== Test Results ==========")
    print("Total tests: %d" % len(test_cases))
    print("Failed test IDs: %s" % (", ".join(str(num) for num in failed_tests) if failed_tests else "None"))
    if failed_tests:
        print("Number of failed tests: %d (%.02f%%)"%(len(failed_tests), len(failed_tests)/(len(test_cases)/100.0)))
    print("All tests passed: %s" % ("No" if failed_tests else "Yes"))
    # Print duplicate tests
    duplicates = {h: tests for h, tests in test_hashes.items() if len(tests) > 1}
    if duplicates:
        print("\n========== Duplicate Tests ==========")
        for hash_value, tests in duplicates.items():
            print("Duplicate test group:")
            for i in tests:
                print("  Test %d: %s" % (i+1, test_cases[i]["name"]))

import hashlib

def get_hash(test_code):
    """Generate MD5 hash of test code - works in Python 2.7 and 3.x"""
    if isinstance(test_code, str):
        # In Python 3, we need to encode strings to bytes
        test_code = test_code.encode('utf-8')
    return hashlib.md5(test_code).hexdigest()

def add_line_numbers(text):
    out = ''
    line = 1
    for s in text.split('\n'):
        out += "%.4d %s\n"%(line, s)
        line += 1
    return out

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

if __name__ == "__main__":
    import sys
    use_interp = len(sys.argv) <= 1 or sys.argv[1] != 'c'
    test(use_interp)
    if not use_interp:
        print("\nUsing C backend for tests")
