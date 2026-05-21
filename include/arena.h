#ifndef ARENA_H
#define ARENA_H

#include <stdbool.h>
#include <stdint.h>
typedef int8_t i8;
typedef int32_t i32;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;
typedef float f32;
typedef i8 b8;
typedef i32 b32;

#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) << 20)
#define GiB(n) ((u64)(n) << 30)

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define ARENA_BASE_POS (sizeof(mem_arena))
#define ARENA_ALIGN (sizeof(void *))
#define ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))

#define PUSH_STRUCT(arena, T) (T *)arena_push((arena), sizeof(T), false)
#define PUSH_STRUCT_NZ(arena, T) (T *)arena_push((arena), sizeof(T), true)
#define PUSH_ARRAY(arena, T, n) (T *)arena_push((arena), sizeof(T) * n, false)
#define PUSH_ARRAY_NX(arena, T, n) (T *)arena_push((arena), sizeof(T) * n, true)

typedef struct {
  u64 capacity;
  u64 pos;
} mem_arena;

mem_arena *arena_create(u64 capacity);
void arena_destroy(mem_arena *arena);
void *arena_push(mem_arena *arena, u64 size, b32 non_zero);
void arena_pop(mem_arena *arena, u64 size);
void arena_pop_to(mem_arena *arena, u64 pos);
void arena_clear(mem_arena *arena);

#endif
