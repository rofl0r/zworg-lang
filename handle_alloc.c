#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define DEBUG_ALLOCATOR

typedef struct handle
{
	uint32_t idx;
	uint16_t allocator_id;
	uint16_t generation; /* currently only used with DEBUG_ALLOCATOR */
} handle;

static const handle handle_nil = {0};

static int handle_cmp(handle a, handle b) {
	return memcmp(&a, &b, sizeof(a));
}

struct allocator
{
	uint32_t alloc_size; // this can be hardcoded in when generating allocators
	// per-struct - must be at least 4
	uint32_t capa; // capacity in items, not bytes
	uint32_t len;  // current total of allocated items
	uint32_t next; // block id of next free chunk
	unsigned char* storage; // this could be of the type itself
#ifdef DEBUG_ALLOCATOR
	unsigned char* gen_ids; // generation id to validate handle
#endif
};

void allocator_init(struct allocator *self, uint32_t alloc_size) {
	assert(alloc_size >= 4);
	memset(self, 0, sizeof(*self));
	self->alloc_size = alloc_size;
}

#define ALIGN(X, A) ((X+(A-1)) & -(A))

uint32_t allocator_alloc(struct allocator *self) {
	if(self->len+1 > self->capa) {
		size_t new_capa = (self->capa?self->capa:1UL)*2UL;
		if(ALIGN(new_capa, 4096)/self->alloc_size > new_capa)
			new_capa = ALIGN(new_capa, 4096)/self->alloc_size;
		void *new = realloc(self->storage, new_capa*self->alloc_size);
		if(!new) return -1;
		self->storage = new;
		uint32_t i;
		/* insert chain of next free slot as in-line metadata */
		unsigned char *basep = self->storage + self->capa*self->alloc_size;
		for(i = self->capa; i < new_capa; ++i, basep+=self->alloc_size) {
			uint32_t* nfp = (void*)basep;
			*nfp = i+1;
		}
#ifdef DEBUG_ALLOCATOR
		self->gen_ids = realloc(self->gen_ids, new_capa);
		/* set initial generation id to 1 */
		memset(self->gen_ids+self->capa, 1, new_capa - self->capa);
#endif
		self->next = self->capa;
		self->capa = new_capa;
	}
	unsigned char *nf = self->storage + self->next*self->alloc_size;
	uint32_t *nfp = (void*)nf;
	uint32_t ret = self->next;
	self->next = *nfp;
	++self->len;
	memset(nf, 0, self->alloc_size);
	return ret;
}

void allocator_free(struct allocator *self, uint32_t index) {
	assert(index < self->capa);
	assert(self->len >= 1);
	unsigned char *p = self->storage + index*self->alloc_size;
	uint32_t *mp = (void*)p;
	*mp = self->next;
	self->next = index;
	--self->len;
}

void *allocator_get_ptr(struct allocator *self, uint32_t index) {
	assert(index < self->capa);
	return self->storage + index*self->alloc_size;
}

struct handle_allocator {
	struct allocator *allocators;
	void *stackbase; // TODO: once we support threads, this needs to be a separate _Thread_local variable
	size_t count;
};

/* TODO: finding the allocator could be made faster for the case that there
   are hundreds of object size classes... but this will do for the moment. */
static inline size_t
find_allocator_for_size(struct handle_allocator *self, uint32_t size) {
	size_t i;
	/* start at index 1 to skip the special array allocator, which might
           have the same alloc_size than one used for objects */
	for(i=1; i<self->count; ++i) {
		if(self->allocators[i].alloc_size == size) return i;
	}
	return (size_t)-1;
}

handle ha_obj_alloc(struct handle_allocator *self, uint32_t size) {
	if(find_allocator_for_size(self, size) == (size_t)-1) {
		assert(self->count <= 0xfffc); /* currently only 64k possible sizeclasses, 0xffff is reserved for stack, 0xfffe for temporary array elem handles, 0xfffd for raw pointer handles */
		void *new = realloc(self->allocators, (self->count+1)*sizeof(struct allocator));
		if(!new) return handle_nil;
		self->allocators = new;
		allocator_init(self->allocators + self->count, size);
		++self->count;
	}
	size_t allocator_idx = find_allocator_for_size(self, size);
	if(sizeof(size_t) > sizeof(uint32_t))
		assert((allocator_idx & 0xffffffff00000000ull) == 0ull);
	assert(allocator_idx < self->count);
	uint32_t alloc_idx = allocator_alloc(self->allocators+allocator_idx);
	if(alloc_idx == -1) return handle_nil;
	uint8_t genid;
#ifdef DEBUG_ALLOCATOR
	genid = self->allocators[allocator_idx].gen_ids[alloc_idx];
#else
	// we set initial generation to 1 so we can differentiate the nil handle from a valid 0,0 handle
	genid = 1;
#endif
	struct handle h = {.idx = alloc_idx, .allocator_id = allocator_idx, .generation = genid};
	return h;
}

void ha_obj_free(struct handle_allocator *self, handle h) {
#ifndef DEBUG_ALLOCATOR
	assert(h.generation == 1);
#endif
	assert(h.allocator_id < self->count);
#ifdef DEBUG_ALLOCATOR
	assert("double-free" && h.generation == self->allocators[h.allocator_id].gen_ids[h.idx]);
	if(++self->allocators[h.allocator_id].gen_ids[h.idx] == 0)
		++self->allocators[h.allocator_id].gen_ids[h.idx];
#endif
	allocator_free(self->allocators + h.allocator_id, h.idx);
}

struct array_elem_handle_data {
	handle array_handle;
	size_t array_idx;
};

struct raw_ptr_data {
	void *ptr;
};

void *ha_array_get_ptr(struct handle_allocator *self, handle h);
void *ha_stack_get_ptr(struct handle_allocator *self, handle h);
void *ha_obj_get_ptr(struct handle_allocator *self, handle h) {
        if(h.allocator_id == 0xffff) return ha_stack_get_ptr(self, h);
        if(h.allocator_id == 0) return ha_array_get_ptr(self, h);
	if(h.allocator_id == 0xfffe) {
		h.allocator_id = 0xffff;
		struct array_elem_handle_data *d = ha_stack_get_ptr(self, h);
		char* ad = ha_obj_get_ptr(self, d->array_handle);
		return ad + d->array_idx;
	}
	if(h.allocator_id == 0xfffd) {
		h.allocator_id = 0xffff;
		struct raw_ptr_data *d = ha_stack_get_ptr(self, h);
		return d->ptr;
	}
	assert(h.allocator_id < self->count);
#ifndef DEBUG_ALLOCATOR
	assert(h.generation == 1);
#else
	assert("use-after-free" && h.generation == self->allocators[h.allocator_id].gen_ids[h.idx]);
#endif
	return allocator_get_ptr(self->allocators + h.allocator_id, h.idx);
}

#define ARRAY_META_FLAG_STATIC 1 /* when a pointer is set from const or static storage */

/* TODO: 32 bits might not be sufficient for the len on 64bit systems.
   unlike for objects, it's plausible that someone wants to open e.g. a big
   file > 4GB as a single array object.
   we might want to use a 64 bit len field, except on 32bit platforms, and
   encode ARRAY_META_FLAG_STATIC as the highest bit instead.
*/
struct array_meta {
	void *ptr;
	uint32_t len;
	uint32_t flags; // currently only 1 bit used, but...padding.
};

handle ha_array_alloc(struct handle_allocator *self, size_t size, void*existing) {
	uint32_t alloc_idx = allocator_alloc(self->allocators+0);
	if(alloc_idx == -1) return handle_nil;
	struct array_meta *meta = allocator_get_ptr(self->allocators+0, alloc_idx);
	if(existing) {
		meta->ptr = existing;
		meta->flags = ARRAY_META_FLAG_STATIC;
	} else {
		meta->ptr = realloc(0, size);
		if(!meta->ptr) {
			allocator_free(self->allocators+0, alloc_idx);
			return handle_nil;
		}
		meta->flags = 0;
	}
	meta->len = size;
	uint8_t genid;
#ifdef DEBUG_ALLOCATOR
	genid = self->allocators[0].gen_ids[alloc_idx];
#else
	// we set initial generation to 1 so we can differentiate the nil handle from a valid 0,0 handle
	genid = 1;
#endif
	struct handle h = {.idx = alloc_idx, .allocator_id = 0, .generation = genid};
	return h;
}

void ha_array_free(struct handle_allocator *self, handle h) {
	assert(h.allocator_id == 0);
#ifdef DEBUG_ALLOCATOR
	assert("double-free" && h.generation == self->allocators[h.allocator_id].gen_ids[h.idx]);
	if(++self->allocators[h.allocator_id].gen_ids[h.idx] == 0)
		++self->allocators[h.allocator_id].gen_ids[h.idx];
#endif
	struct array_meta *meta = allocator_get_ptr(self->allocators+0, h.idx);
	if((meta->flags & ARRAY_META_FLAG_STATIC) == 0)
		free(meta->ptr);
	allocator_free(self->allocators+0, h.idx);
}

handle ha_array_realloc(struct handle_allocator *self, handle h, size_t newsize) {
	assert(h.allocator_id == 0);
        if(handle_cmp(h, handle_nil) == 0)
		return ha_array_alloc(self, newsize, (void*)0);
#ifdef DEBUG_ALLOCATOR
	assert("use-after-free" && h.generation == self->allocators[h.allocator_id].gen_ids[h.idx]);
#endif
	struct array_meta *meta = allocator_get_ptr(self->allocators+0, h.idx);
	void *new;
	/* can't resize a static/const ptr */
	if(meta->flags & ARRAY_META_FLAG_STATIC) {
		new = realloc(0, newsize);
		if(new) {
			size_t min = meta->len > newsize ? newsize : meta->len;
			memcpy(new, meta->ptr, min);
		}
	} else
		new = realloc(meta->ptr, newsize);
	if(!new) return handle_nil;
	meta->len = newsize;
	meta->ptr = new;
	meta->flags = 0;
	return h;
}

void *ha_array_get_ptr(struct handle_allocator *self, handle h) {
	assert(h.allocator_id == 0);
#ifndef DEBUG_ALLOCATOR
	assert(h.generation == 1);
#else
	assert("use-after-free" && h.generation == self->allocators[h.allocator_id].gen_ids[h.idx]);
#endif
	struct array_meta *meta = allocator_get_ptr(self->allocators+0, h.idx);
	return meta->ptr;
}

static void ha_array_copy(struct handle_allocator *ha, handle dest, handle src) {
	struct array_meta *src_meta = allocator_get_ptr(ha->allocators, src.idx);
	memcpy(ha_array_get_ptr(ha, dest), ha_array_get_ptr(ha, src), src_meta->len);
}

handle ha_stack_alloc(struct handle_allocator *self, size_t size, void*existing) {
	intptr_t offset = (intptr_t)self->stackbase - (intptr_t)existing;
	handle h = {.idx = offset, .allocator_id = 0xffff, .generation = 1};
	return h;
}

void *ha_stack_get_ptr(struct handle_allocator *self, handle h) {
	assert(h.allocator_id == 0xffff);
	return (void*)((intptr_t)self->stackbase - h.idx);
}

/* get a temporary stack handle to an object inside an array handle.
   handle_data must live on the stack, and array_idx is the element's position
   in bytes from the array data start, not actual the element index. */
handle ha_array_elem_handle(struct handle_allocator *self, struct array_elem_handle_data *handle_data) {
	handle h = ha_stack_alloc(self, sizeof(*handle_data), handle_data);
	h.allocator_id = 0xfffe;
	return h;
}

handle ha_raw_handle(struct handle_allocator *self, struct raw_ptr_data* handle_data) {
	handle h = ha_stack_alloc(self, sizeof(*handle_data), handle_data);
	h.allocator_id = 0xfffd;
	return h;
}

void ha_init(struct handle_allocator *self, void *stack) {
	/* allocate the special array allocator as ID 0 */
	self->allocators = realloc(0, sizeof(struct allocator));
	/* if we can't even allocate the base allocator, we can't do anything at all  */
	assert(self->allocators);
	allocator_init(self->allocators, sizeof(struct array_meta));
	self->count = 1;
	self->stackbase = stack;
}

void ha_destroy(struct handle_allocator *self) {
	size_t i;
	assert(self->count);
	/* FIXME need to find all non-free items in the array allocator,
           extract meta and free pointers without static bit */
	for (i = 0; i < self->count; i++) {
		free(self->allocators[i].storage);
	}
	free(self->allocators);
	self->allocators = NULL;
	self->count = 0;
}
