#include "arena.h"
#include <stdlib.h>
#include <string.h>

mem_arena *arena_create(u64 capacity) {
  mem_arena *arena = (mem_arena *)malloc(capacity);
  arena->capacity = capacity;
  arena->pos = ARENA_BASE_POS;

  return arena;
}

void arena_destroy(mem_arena *arena) { free(arena); }

void *arena_push(mem_arena *arena, u64 size, b32 non_zero) {
  u64 pos_aligned = ALIGN_UP_POW2(arena->pos, ARENA_ALIGN);
  u64 new_pos = pos_aligned + size;

  if (new_pos > arena->capacity) {
    return NULL;
  }

  arena->pos = new_pos;

  u8 *out = (u8 *)arena + pos_aligned;

  if (!non_zero) {
    memset(out, 0, size);
  }

  return out;
}

void arena_pop(mem_arena *arena, u64 size) {
  size = MIN(size, arena->pos - ARENA_BASE_POS);
  arena->pos -= size;
}

void arena_pop_to(mem_arena *arena, u64 pos) {
  u64 size = (pos < arena->pos) ? arena->pos - pos : 0;
  arena_pop(arena, size);
}
void arena_clear(mem_arena *arena) { arena_pop_to(arena, ARENA_BASE_POS); }
