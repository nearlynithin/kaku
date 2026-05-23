#include "arena.h"
#include "cnn.h"
#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  srand(time(NULL));
  mem_arena *perm_arena = arena_create(MiB(20));
  mem_arena *temp_arena = arena_create(MiB(20));

  conv conv;
  conv_init(&conv, 3, 3, 8, perm_arena);
  dataset *data = load_dataset("dataset/t10k-images.idx3-ubyte", perm_arena);

  feature_map *fm = forward(&conv, data, temp_arena, 1);
  max_pool(2, fm);

  softmax *sm =
      softmax_init(fm->width * fm->height * fm->depth, 10, temp_arena);
  f32 *probs = softmax_forward(sm, fm, temp_arena);

  for (u64 i = 0; i < sm->nodes; i++) {
    printf("%llu: %f\n", i, probs[i]);
  }

  u64 best = 0;
  for (u64 i = 1; i < sm->nodes; i++) {
    if (probs[i] > probs[best])
      best = i;
  }
  printf("predicted: %llu\n", best);

  return 0;
}
