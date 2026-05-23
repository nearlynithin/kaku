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

  dataset *data = load_dataset("dataset/t10k-images.idx3-ubyte",
                               "dataset/t10k-labels.idx1-ubyte", perm_arena);

  conv conv;
  conv_init(&conv, 3, 3, 8, perm_arena);
  softmax sm;
  softmax_init(&sm, 13 * 13 * 8, 10, perm_arena);

  f32 loss = 0.0f;
  u64 num_correct = 0;

  for (u64 i = 0; i < data->count; i++) {
    arena_clear(temp_arena);

    prediction p = forward(&conv, &sm, data, i, data->labels[i], temp_arena);
    loss += p.loss;
    num_correct += p.accuracy;

    if (i % 100 == 99) {
      printf(
          "[Step %llu] Past 100 steps: Average Loss : %f | Accuracy : %llu\n",
          i + 1, (f32)loss / 100, num_correct);
      loss = 0.0f;
      num_correct = 0;
    }
  }

  return 0;
}
