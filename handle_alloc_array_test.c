#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "handle_alloc.c"

// Test structures
typedef struct {
    int x;
    int y;
} Point;

// Static test data for const array tests
static const int test_const_data[] = {1, 2, 3, 4, 5};
static const char *test_const_strings[] = {"hello", "world", "zworg"};

void test_array_basic() {
    printf("Testing basic array allocation...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    // Test 1: Allocate array of ints
    int array_size = 10;
    handle h_array = ha_array_alloc(&ha, sizeof(int) * array_size, NULL);
    assert(h_array.allocator_id == 0);
    assert(h_array.generation == 1);

    // Get the array pointer and test writing to it
    int* int_array = ha_array_get_ptr(&ha, h_array);
    assert(int_array != NULL);

    for (int i = 0; i < array_size; i++) {
        int_array[i] = i * 10;
    }

    int_array = ha_array_get_ptr(&ha, h_array);

    // Verify values
    for (int i = 0; i < array_size; i++) {
        assert(int_array[i] == i * 10);
    }

    // Free the array
    ha_array_free(&ha, h_array);

    ha_destroy(&ha);
    printf("Basic array allocation test passed\n");
    fflush(stdout);
}

void test_static_array() {
    printf("Testing static array allocation...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    // Test 1: Create a handle for static array
    handle h_static = ha_array_alloc(&ha, sizeof(test_const_data), (void*)test_const_data);
    assert(h_static.allocator_id == 0);
    assert(h_static.generation == 1);

    // Get the static array ptr and verify it points to our static data
    int* static_array = ha_array_get_ptr(&ha, h_static);
    assert(static_array == test_const_data);

    // Verify we can read values (but not modify them since they're const)
    assert(static_array[0] == 1);
    assert(static_array[1] == 2);
    assert(static_array[4] == 5);

    // Free the array - this should only free the handle, not the static data
    ha_array_free(&ha, h_static);

    // Test 2: String array
    handle h_strings = ha_array_alloc(&ha, sizeof(test_const_strings), (void*)test_const_strings);
    const char **strings = ha_array_get_ptr(&ha, h_strings);

    assert(strings == test_const_strings);
    assert(strcmp(strings[0], "hello") == 0);
    assert(strcmp(strings[1], "world") == 0);
    assert(strcmp(strings[2], "zworg") == 0);

    ha_array_free(&ha, h_strings);
    ha_destroy(&ha);
    printf("Static array allocation test passed\n");
}

void test_array_realloc() {
    printf("Testing array reallocation...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    // Test 1: Allocate and then reallocate
    int initial_size = 5;
    int new_size = 10;
    handle h_array = ha_array_alloc(&ha, sizeof(int) * initial_size, NULL);

    // Fill with data
    int* int_array = ha_array_get_ptr(&ha, h_array);
    for (int i = 0; i < initial_size; i++) int_array[i] = i * 100;

    // Reallocate to larger size
    handle h_resized = ha_array_realloc(&ha, h_array, sizeof(int) * new_size);
    assert(h_resized.idx == h_array.idx); // Should be the same handle

    // Verify original data is preserved
    int* resized_array = ha_array_get_ptr(&ha, h_resized);
    for (int i = 0; i < initial_size; i++) assert(resized_array[i] == i * 100);

    // Add new data in expanded area
    for (int i = initial_size; i < new_size; i++) {
        resized_array[i] = i * 100;
    }

    resized_array = ha_array_get_ptr(&ha, h_resized);

    // Verify all data
    for (int i = 0; i < new_size; i++) assert(resized_array[i] == i * 100);

    // Test 2: Reallocate a static array (should copy data)
    handle h_static = ha_array_alloc(&ha, sizeof(test_const_data), (void*)test_const_data);

    // Reallocate (this should create a copy)
    handle h_static_copy = ha_array_realloc(&ha, h_static, sizeof(int) * 10);

    // Verify it's not pointing to the original static data anymore
    int* copied_array = ha_array_get_ptr(&ha, h_static_copy);
    assert(copied_array != test_const_data);

    // Verify data was copied correctly
    for (int i = 0; i < 5; i++) {
        assert(copied_array[i] == test_const_data[i]);
    }

    // Now we can modify the copy without affecting the original
    copied_array[0] = 999; // should raise OS error if pointing to const data
    assert(test_const_data[0] == 1); // Original unchanged

    // Clean up
    ha_array_free(&ha, h_resized);
    ha_array_free(&ha, h_static);
    //ha_array_free(&ha, h_static_copy); // <- this would cause double-free

    ha_destroy(&ha);
    printf("Array reallocation test passed\n");
}

void test_mixed_allocations() {
    printf("Testing mixed object and array allocations...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    // Allocate arrays
    handle h_array1 = ha_array_alloc(&ha, sizeof(int) * 10, NULL);
    handle h_array2 = ha_array_alloc(&ha, sizeof(char) * 20, NULL);

    // Allocate regular objects
    handle h_point1 = ha_obj_alloc(&ha, sizeof(Point));
    handle h_point2 = ha_obj_alloc(&ha, sizeof(Point));

    // Use both types
    int* int_array = ha_array_get_ptr(&ha, h_array1);
    char* char_array = ha_array_get_ptr(&ha, h_array2);
    Point* p1 = ha_obj_get_ptr(&ha, h_point1);
    Point* p2 = ha_obj_get_ptr(&ha, h_point2);

    // Set values
    for (int i = 0; i < 10; i++) {
        int_array[i] = i;
    }

    strcpy(char_array, "Hello, ZWorg!");

    p1->x = 10;
    p1->y = 20;
    p2->x = 30;
    p2->y = 40;

    // Verify values
    for (int i = 0; i < 10; i++) {
        assert(int_array[i] == i);
    }

    assert(strcmp(char_array, "Hello, ZWorg!") == 0);

    assert(p1->x == 10 && p1->y == 20);
    assert(p2->x == 30 && p2->y == 40);

    // Free everything
    ha_array_free(&ha, h_array1);
    ha_array_free(&ha, h_array2);
    ha_obj_free(&ha, h_point1);
    ha_obj_free(&ha, h_point2);

    ha_destroy(&ha);
    printf("Mixed allocation test passed\n");
}

void test_edge_cases() {
    printf("Testing edge cases...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    // Test 1: Zero-sized array (should still allocate something)
    handle h_zero = ha_array_alloc(&ha, 0, NULL);
    assert(h_zero.allocator_id == 0);
    assert(h_zero.generation == 1);

    void* zero_ptr = ha_array_get_ptr(&ha, h_zero);
    assert(zero_ptr != NULL); // Should still get a valid pointer

    // Test 2: Large array
    size_t large_size = 1024 * 1024; // 1MB
    handle h_large = ha_array_alloc(&ha, large_size, NULL);

    void* large_ptr = ha_array_get_ptr(&ha, h_large);
    assert(large_ptr != NULL);

    // Write to the large array to ensure it's valid
    memset(large_ptr, 0xAB, large_size);

    // Test 3: Multiple reallocs
    handle h_multi = ha_array_alloc(&ha, 8, NULL);
    for (int i = 0; i < 10; i++) {
        h_multi = ha_array_realloc(&ha, h_multi, 8 * (i+2));
        assert(h_multi.allocator_id == 0);
    }

    // Clean up
    ha_array_free(&ha, h_zero);
    ha_array_free(&ha, h_large);
    ha_array_free(&ha, h_multi);

    ha_destroy(&ha);
    printf("Edge cases test passed\n");
}

void test_many_arrays() {
    printf("Testing many array allocations...\n");

    struct handle_allocator ha;
    ha_init(&ha);

    #define NUM_ARRAYS 1000
    handle handles[NUM_ARRAYS];

    // Allocate many small arrays
    for (int i = 0; i < NUM_ARRAYS; i++) {
        handles[i] = ha_array_alloc(&ha, sizeof(int) * 4, NULL);
        assert(handles[i].allocator_id == 0);

        int* arr = ha_array_get_ptr(&ha, handles[i]);
        for (int j = 0; j < 4; j++) {
            arr[j] = i * 1000 + j;
        }
    }

    // Verify all arrays
    for (int i = 0; i < NUM_ARRAYS; i++) {
        int* arr = ha_array_get_ptr(&ha, handles[i]);
        for (int j = 0; j < 4; j++) {
            assert(arr[j] == i * 1000 + j);
        }
    }

    // Free half the arrays
    for (int i = 0; i < NUM_ARRAYS; i += 2) {
        ha_array_free(&ha, handles[i]);
    }

    // Reallocate those slots
    for (int i = 0; i < NUM_ARRAYS; i += 2) {
        handles[i] = ha_array_alloc(&ha, sizeof(double) * 2, NULL);
        double* arr = ha_array_get_ptr(&ha, handles[i]);
        arr[0] = i + 0.5;
        arr[1] = i + 1.5;
    }

    // Verify the remaining original arrays are intact
    for (int i = 1; i < NUM_ARRAYS; i += 2) {
        int* arr = ha_array_get_ptr(&ha, handles[i]);
        for (int j = 0; j < 4; j++) {
            assert(arr[j] == i * 1000 + j);
        }
    }

    // Verify the new arrays
    for (int i = 0; i < NUM_ARRAYS; i += 2) {
        double* arr = ha_array_get_ptr(&ha, handles[i]);
        assert(arr[0] == i + 0.5);
        assert(arr[1] == i + 1.5);
    }

    // Clean up
    for (int i = 0; i < NUM_ARRAYS; i++) {
        ha_array_free(&ha, handles[i]);
    }

    ha_destroy(&ha);
    printf("Many arrays test passed\n");
}

int main() {
    test_array_basic();
    test_static_array();
    test_array_realloc();
    test_mixed_allocations();
    test_edge_cases();
    test_many_arrays();

    printf("\nAll array allocator tests passed!\n");
    return 0;
}
