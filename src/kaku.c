#include "arena.h"
#include "cnn.h"
#include "data.h"

int main() {
  mem_arena *perm_arena = arena_create(MiB(20));
  mem_arena *temp_arena = arena_create(MiB(20));

  conv conv;
  conv_init(&conv, 3, 3, 8, perm_arena);
  dataset *data = load_dataset("dataset/t10k-images.idx3-ubyte", perm_arena);

  feature_map *fm = forward(&conv, data, temp_arena, 1);
  max_pool(2, fm, temp_arena);

  return 0;
}
