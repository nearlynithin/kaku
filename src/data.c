#include "data.h"

#include "arena.h"
#include <stdio.h>
#include <stdlib.h>

void print_image(u8 *img, u32 num) {
  const char *shades = ".:-=+*#%@";
  for (u32 i = 0; i < num; i++) {
    for (u32 r = 0; r < 28; r++) {
      for (u32 c = 0; c < 28; c++) {
        u8 pixel = img[(i * 28 * 28) + r * 28 + c];
        char c = shades[pixel / 20];
        putchar(c);
      }
      printf("\n");
    }
  }
}

// Read u32 big endian
u32 read_u32_be(FILE *f) {
  u8 b[4];
  fread(b, 1, 4, f);

  return ((u32)b[0] << 24) | ((u32)b[1] << 16) | ((u32)b[2] << 8) | ((u32)b[3]);
}

void labels_head(char *s) {
  mem_arena *scratch = arena_create(MiB(1));

  FILE *file = fopen(s, "rb");

  if (!file) {
    printf("Error reading file %s\n", s);
    return;
  }

  u32 magic_number = 0;
  u32 num_items = 0;

  magic_number = read_u32_be(file);
  num_items = read_u32_be(file);

  printf("Magic number : %u\n", magic_number);
  printf("Number of items : %u\n", num_items);

  unsigned char *labels = PUSH_ARRAY(scratch, unsigned char, num_items);
  fread(labels, sizeof(unsigned char), num_items, file);

  // printing first 10 labels
  for (u32 i = 0; i < 10 && i < num_items; i++) {
    printf("Label %d: %u\n", i, labels[i]);
  }

  fclose(file);
  arena_destroy(scratch);
}

void images_head(char *s) {
  mem_arena *scratch = arena_create(MiB(10));

  FILE *file = fopen(s, "rb");

  if (!file) {
    printf("Error reading file %s\n", s);
    return;
  }

  u32 magic_number = read_u32_be(file);
  u32 count = read_u32_be(file);
  u32 rows = read_u32_be(file);
  u32 cols = read_u32_be(file);

  printf("Magic number : %u\n", magic_number);
  printf("Number of items : %u\n", count);
  printf("Number of rows : %u\n", rows);
  printf("Number of columns : %u\n", cols);

  u8 *images = PUSH_ARRAY(scratch, u8, rows * cols * count);
  fread(images, sizeof(u8), rows * cols * count, file);
  // printing first 10 images
  print_image(images, 10);

  fclose(file);
  arena_destroy(scratch);
}

dataset *load_dataset(char *s, mem_arena *arena) {

  FILE *file = fopen(s, "rb");

  if (!file) {
    printf("Error reading file %s\n", s);
    exit(1);
  }

  dataset *data = PUSH_STRUCT(arena, dataset);

  u32 magic_number = read_u32_be(file);
  u32 count = read_u32_be(file);
  u32 rows = read_u32_be(file);
  u32 cols = read_u32_be(file);

  printf("Magic number : %u\n", magic_number);
  printf("Number of items : %u\n", count);
  printf("Number of rows : %u\n", rows);
  printf("Number of columns : %u\n", cols);

  u8 *images = PUSH_ARRAY(arena, u8, rows * cols * count);
  fread(images, sizeof(u8), rows * cols * count, file);

  data->images = images;
  data->rows = rows;
  data->cols = cols;
  data->count = count;

  // printing first 10 images
  // print_image(images, 10);

  fclose(file);
  return data;
}
