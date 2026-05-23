#ifndef CNN_H
#define CNN_H

#include "arena.h"
#include "data.h"

typedef struct {
  f32 *data;
  u64 width;
  u64 height;
} filter;

typedef struct {
  filter *filters;
  u64 num_filters;
} conv;

typedef struct {
  f32 *data;
  u64 width;
  u64 height;
  u64 depth;
} feature_map;

void conv_init(conv *conv, u64 width, u64 height, u64 num_filters,
               mem_arena *arena);
feature_map *forward(conv *conv, dataset *data, mem_arena *arena,
                     u64 image_index);

f32 sum_region(filter *f, dataset *data, u64 image_index, u64 x, u64 y);

void max_pool(u64 n, feature_map *fm);

#endif
