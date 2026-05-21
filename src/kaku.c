#include "arena.h"
#include "cnn.h"
#include "data.h"

int main() {
  mem_arena *perm_arena = arena_create(MiB(20));

  conv conv;
  conv_init(&conv, 3, 3, 8, perm_arena);
  dataset *data = load_dataset("dataset/t10k-images.idx3-ubyte", perm_arena);
  f32 *feature_maps = forward(&conv, data, perm_arena, 1);

  return 0;
}
