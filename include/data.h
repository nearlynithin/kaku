#ifndef DATA_H
#define DATA_H

#include "arena.h"

typedef struct {
  u64 count;
  u64 rows;
  u64 cols;
  u8 *images;
  u8 *labels;
} dataset;

void labels_head(char *s);
void images_head(char *s);
dataset *load_dataset(char *image_path, char *label_path, mem_arena *arena);

#endif
