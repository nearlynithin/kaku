#include "cnn.h"
#include "arena.h"
#include <stdlib.h>
#include <time.h>

f32 random_weight() {
  srand(1000 * time(NULL));
  f32 r = (f32)rand() / (f32)RAND_MAX;
  r = ((2.0f * r) - 1) / 9;
  return r;
}

void conv_init(conv *conv, u64 width, u64 height, u64 num_filters,
               mem_arena *arena) {

  conv->num_filters = num_filters;
  conv->filters = PUSH_ARRAY(arena, filter, num_filters);

  for (u64 i = 0; i < num_filters; i++) {
    filter *f = &conv->filters[i];
    f->width = width;
    f->height = height;
    f->data = PUSH_ARRAY(arena, f32, width * height);
    for (u64 h = 0; h < height; h++) {
      for (u64 w = 0; w < width; w++) {
        f->data[h * width + w] = random_weight();
      }
    }
  }
}

f32 *forward(conv *conv, dataset *data, mem_arena *arena, u64 image_index) {

  u64 out_h = data->rows - 2;
  u64 out_w = data->cols - 2;
  f32 *feature_maps = PUSH_ARRAY(arena, f32, conv->num_filters * out_h * out_w);

  for (u64 f = 0; f < conv->num_filters; f++) {
    for (u64 y = 0; y < out_h; y++) {
      for (u64 x = 0; x < out_w; x++) {
        f32 sum = sum_region(&conv->filters[f], data, image_index, x, y);
        feature_maps[(f * out_h * out_w) + y * out_w + x] = sum;
      }
    }
  }
  return feature_maps;
}

f32 sum_region(filter *f, dataset *data, u64 image_index, u64 x, u64 y) {
  f32 sum = 0;

  u8 *image = data->images + image_index * data->rows * data->cols;

  for (u64 kr = 0; kr < f->height; kr++) {
    for (u64 kc = 0; kc < f->width; kc++) {

      f32 pixel = (image[(y + kr) * data->cols + (x + kc)] / 255.0f) - 0.5f;
      f32 weight = f->data[kr * f->width + kc];

      sum += pixel * weight;
    }
  }
  return sum;
}
