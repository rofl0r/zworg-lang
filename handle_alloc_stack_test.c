#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "handle_alloc.c"

void test_stack_allocator() {
    printf("Testing stack allocator...\n");

    // Initialize allocator with stack base
    int stack_anchor;
    struct handle_allocator ha;
    ha_init(&ha, &stack_anchor);

    // Allocate structures on stack
    struct {
        int x;
        int y;
    } point_on_stack = {10, 20};

    char string_on_stack[32];
    strcpy(string_on_stack, "Stack string test");

    // Create handles to stack objects
    handle stack_point_h = ha_stack_alloc(&ha, sizeof(point_on_stack), &point_on_stack);
    handle stack_string_h = ha_stack_alloc(&ha, sizeof(string_on_stack), string_on_stack);

    // Verify allocator_id is set correctly
    assert(stack_point_h.allocator_id == 0xFFFF);
    assert(stack_string_h.allocator_id == 0xFFFF);

    // Access stack objects through handle system
    void* point_ptr = ha_stack_get_ptr(&ha, stack_point_h);
    void* string_ptr = ha_stack_get_ptr(&ha, stack_string_h);

    // Verify pointers match original stack locations
    assert(point_ptr == &point_on_stack);
    assert(string_ptr == string_on_stack);

    // Test reading values through the pointers
    struct { int x; int y; } *p = point_ptr;
    assert(p->x == 10);
    assert(p->y == 20);
    assert(strcmp((char*)string_ptr, "Stack string test") == 0);

    // Test writing values through the pointers
    p->x = 30;
    p->y = 40;
    strcpy((char*)string_ptr, "Modified stack string");

    // Verify original stack values were changed
    assert(point_on_stack.x == 30);
    assert(point_on_stack.y == 40);
    assert(strcmp(string_on_stack, "Modified stack string") == 0);

    // Test mixing stack and heap objects
    handle heap_obj = ha_obj_alloc(&ha, sizeof(int) * 4);
    int* heap_ptr = ha_obj_get_ptr(&ha, heap_obj);
    for (int i = 0; i < 4; i++) {
        heap_ptr[i] = i * 100;
    }

    // Verify heap object values
    assert(heap_ptr[0] == 0);
    assert(heap_ptr[1] == 100);
    assert(heap_ptr[2] == 200);
    assert(heap_ptr[3] == 300);

    // Clean up heap object
    ha_obj_free(&ha, heap_obj);

    // No cleanup needed for stack objects

    printf("Stack allocator test passed!\n");
}

int main() {
    test_stack_allocator();
    return 0;
}
