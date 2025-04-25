# Test framework for compiler.py
from interpreter import Interpreter
import os, sys

"""
        {
           "name": "",
           "code": "",
           "expected_env": {"x": 1}
        },
"""

def test():
    # Test cases with expected final env state
    test_cases = [
        # Regular test cases (expected to succeed)
        # Each has "code" and "expected_env"
        {
            "name": "tuple initializer with type inference",
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
            "name": "variable declaration with type inference (:=)",
            "code": "def main() do var x := 5; end",
            "expected_env": {"x": 5}
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
            "code": "def main() do var x := 1; if x == 1 do print x; end end",
            "expected_env": {"x": 1}
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
            "code": "def main() do var x := 3; if x == 1 do print 1; end else if x == 2 do print 2; end else if x == 3 do print 3; end end",
            "expected_env": {"x": 3}
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
            "expected_error": "Variable declaration must include an initialization"
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
            "expected_error": 'Invalid statement starting with "else" (TT_ELSE)'
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
            def main() do var p := Point(5); p.nonexistent(); end
        """,
        "expected_error": "Method 'nonexistent' not found in struct 'Point'"
    },
    {
        "name": "Undefined field access",
        "code": """
            struct Point do x: int; end
            def main() do var p := Point(5); print p.y; end
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
                var p := Point(5);
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
            "expected_error": "Initializer for Point has 3 elements, but struct has 2 fields"
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

    # List to track failing tests
    interpreter = Interpreter()
    failed_tests = []

    # Run all test cases
    for i, test_case in enumerate(test_cases):
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
                printc("red", "Input: %s" % test_case["code"])
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
		mismatches = []
                for k in test_case["expected_env"].keys():
                    v = test_case["expected_env"][k]
                    if k not in env:
                        mismatches.append("{}: want {} got: None".format(k, v))
                    elif env[k] != v:
                        mismatches.append("{}: want {} got: {}".format(k, v, env[k]))

                if len(mismatches) == 0:
                    print("Success! Environment matches expectations.")
                else:
                    printc("red", "Input: %s" % test_case["code"])
                    print("Test passed but with incorrect environment values:")
                    print("  Expected env: %s" % test_case["expected_env"])
                    print("  Actual env: %s" % env)
                    print("  Mismatches:")
                    for m in mismatches: print(m)
                    failed_tests.append(test_num)
                    if result.get('ast'):
                        print("AST dump: %s" % result['ast'])

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

if __name__ == '__main__':
    test()
