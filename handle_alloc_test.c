#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>

#include "handle_alloc.c"

// Test structures
typedef struct {
    int x;
    int y;
} Point;

typedef struct {
    char name[32];
    int value;
    double ratio;
} LargeObject;

void test_simple_allocation() {
    printf("Testing simple allocation...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    // Allocate a single object
    handle h = ha_obj_alloc(&ha, sizeof(Point));
    assert(h.idx == 0);
    assert(h.allocator_id == 1);
    assert(h.generation == 1);

    // Check that we can access it
    Point* p = ha_obj_get_ptr(&ha, h);
    assert(p != NULL);

    // Set and retrieve values
    p->x = 42;
    p->y = 24;

    Point* p2 = ha_obj_get_ptr(&ha, h);
    assert(p2->x == 42);
    assert(p2->y == 24);

    // Free the object
    ha_obj_free(&ha, h);

    ha_destroy(&ha);
    printf("Simple allocation test passed\n");
}

void test_multiple_sizes() {
    printf("Testing multiple size classes...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    // Allocate objects of different sizes
    handle h1 = ha_obj_alloc(&ha, sizeof(Point));
    handle h2 = ha_obj_alloc(&ha, sizeof(LargeObject));

    assert(h1.allocator_id == 1);
    assert(h2.allocator_id == 2);

    // Access and modify objects
    Point* p = ha_obj_get_ptr(&ha, h1);
    p->x = 10;
    p->y = 20;

    LargeObject* lo = ha_obj_get_ptr(&ha, h2);
    strcpy(lo->name, "Test Object");
    lo->value = 100;
    lo->ratio = 3.14159;

    // Verify values
    p = ha_obj_get_ptr(&ha, h1);
    assert(p->x == 10);
    assert(p->y == 20);

    lo = ha_obj_get_ptr(&ha, h2);
    assert(strcmp(lo->name, "Test Object") == 0);
    assert(lo->value == 100);
    assert(lo->ratio == 3.14159);

    // Free objects
    ha_obj_free(&ha, h1);
    ha_obj_free(&ha, h2);

    ha_destroy(&ha);
    printf("Multiple size classes test passed\n");
}

void test_reuse_after_free() {
    printf("Testing handle reuse after free...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    // Allocate and free to set up the free list
    handle h1 = ha_obj_alloc(&ha, sizeof(Point));
    ha_obj_free(&ha, h1);

    // Next allocation should reuse the same slot
    handle h2 = ha_obj_alloc(&ha, sizeof(Point));
    assert(h2.idx == h1.idx);
    assert(h2.allocator_id == h1.allocator_id);

    // Objects should be zeroed after allocation
    Point* p = ha_obj_get_ptr(&ha, h2);
    assert(p->x == 0);
    assert(p->y == 0);

    ha_obj_free(&ha, h2);
    ha_destroy(&ha);
    printf("Reuse after free test passed\n");
}

void test_multiple_allocations() {
    printf("Testing multiple allocations and capacity growth...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    // Allocate many objects to force capacity growth
    #define NUM_OBJECTS 1000
    handle handles[NUM_OBJECTS];

    for (int i = 0; i < NUM_OBJECTS; i++) {
        handles[i] = ha_obj_alloc(&ha, sizeof(Point));
        assert(handles[i].generation == 1);

        // Set unique values
        Point* p = ha_obj_get_ptr(&ha, handles[i]);
        p->x = i;
        p->y = i * 10;
    }

    // Verify all values
    for (int i = 0; i < NUM_OBJECTS; i++) {
        Point* p = ha_obj_get_ptr(&ha, handles[i]);
        assert(p->x == i);
        assert(p->y == i * 10);
    }

    // Free every other object
    for (int i = 0; i < NUM_OBJECTS; i += 2) {
        ha_obj_free(&ha, handles[i]);
    }

    // Allocate new objects that should reuse freed slots
    for (int i = 0; i < NUM_OBJECTS; i += 2) {
        handles[i] = ha_obj_alloc(&ha, sizeof(Point));
#ifdef DEBUG_ALLOCATOR
        assert(handles[i].generation == 2);
#else
        assert(handles[i].generation == 1);
#endif

        Point* p = ha_obj_get_ptr(&ha, handles[i]);
        p->x = -i;
        p->y = -i * 10;
    }

    // Verify all values again
    for (int i = 0; i < NUM_OBJECTS; i++) {
        Point* p = ha_obj_get_ptr(&ha, handles[i]);
        if (i % 2 == 0) {
            assert(p->x == -i);
            assert(p->y == -i * 10);
        } else {
            assert(p->x == i);
            assert(p->y == i * 10);
        }
    }

    // Free all objects
    for (int i = 0; i < NUM_OBJECTS; i++) {
        ha_obj_free(&ha, handles[i]);
    }

    ha_destroy(&ha);
    printf("Multiple allocations test passed\n");
}

void test_nil_handle() {
    printf("Testing nil handle behavior...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    // Test that handle_nil is properly defined
    assert(handle_nil.idx == 0);
    assert(handle_nil.allocator_id == 0);
    assert(handle_nil.generation == 0);

    // Allocation failure simulation would return nil
    // (Not easy to simulate in this test environment)

    ha_destroy(&ha);
    printf("Nil handle test passed\n");
}

void test_different_sized_allocators() {
    printf("Testing different sized allocators...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    // Create a range of differently-sized objects
    //handle h_1byte = ha_obj_alloc(&ha, 1);
    handle h_4bytes = ha_obj_alloc(&ha, 4);
    handle h_7bytes = ha_obj_alloc(&ha, 7);
    handle h_8bytes = ha_obj_alloc(&ha, 8);
    handle h_9bytes = ha_obj_alloc(&ha, 9);
    handle h_16bytes = ha_obj_alloc(&ha, 16);
    handle h_1024bytes = ha_obj_alloc(&ha, 1024);

    // Check that each has its own allocator
    //assert(h_1byte.allocator_id != h_4bytes.allocator_id);
    assert(h_4bytes.allocator_id != h_7bytes.allocator_id);
    assert(h_7bytes.allocator_id != h_8bytes.allocator_id);
    assert(h_8bytes.allocator_id != h_9bytes.allocator_id);
    assert(h_9bytes.allocator_id != h_16bytes.allocator_id);
    assert(h_16bytes.allocator_id != h_1024bytes.allocator_id);

    // Check total allocator count
    assert(ha.count == 6+1 /* 1 for magic array allocator */);

    // Clean up
    //ha_obj_free(&ha, h_1byte);
    ha_obj_free(&ha, h_4bytes);
    ha_obj_free(&ha, h_7bytes);
    ha_obj_free(&ha, h_8bytes);
    ha_obj_free(&ha, h_9bytes);
    ha_obj_free(&ha, h_16bytes);
    ha_obj_free(&ha, h_1024bytes);

    ha_destroy(&ha);
    printf("Different sized allocators test passed\n");
}

void test_allocation_pattern() {
    printf("Testing complex allocation pattern...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    #define PATTERN_SIZE 100
    handle handles[PATTERN_SIZE];

    // Allocate all objects
    for (int i = 0; i < PATTERN_SIZE; i++) {
        handles[i] = ha_obj_alloc(&ha, sizeof(int));
        int* p = ha_obj_get_ptr(&ha, handles[i]);
        *p = i;
    }

    // Free objects in a pattern
    for (int i = 0; i < PATTERN_SIZE; i += 3) {
        ha_obj_free(&ha, handles[i]);
    }

    // Reallocate freed objects
    for (int i = 0; i < PATTERN_SIZE; i += 3) {
        handles[i] = ha_obj_alloc(&ha, sizeof(int));
        int* p = ha_obj_get_ptr(&ha, handles[i]);
        *p = i + 1000;
    }

    // Verify values
    for (int i = 0; i < PATTERN_SIZE; i++) {
        int* p = ha_obj_get_ptr(&ha, handles[i]);
        if (i % 3 == 0) {
            assert(*p == i + 1000);
        } else {
            assert(*p == i);
        }
    }

    // Clean up
    for (int i = 0; i < PATTERN_SIZE; i++) {
        ha_obj_free(&ha, handles[i]);
    }

    ha_destroy(&ha);
    printf("Complex allocation pattern test passed\n");
}

int main() {
    test_simple_allocation();
    test_multiple_sizes();
    test_reuse_after_free();
    test_multiple_allocations();
    test_nil_handle();
    test_different_sized_allocators();
    test_allocation_pattern();

    printf("\nAll tests passed!\n");
    return 0;
}
