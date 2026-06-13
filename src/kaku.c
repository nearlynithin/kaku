#include "arena.h"
#include "cnn.h"
#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  srand(time(NULL));
  mem_arena *perm_arena = arena_create(MiB(64));
  mem_arena *temp_arena = arena_create(MiB(64));

  dataset *train_data =
      load_dataset("dataset/train-images.idx3-ubyte",
                   "dataset/train-labels.idx1-ubyte", perm_arena);

  // dataset *test_data =
  //     load_dataset("dataset/t10k-images.idx3-ubyte",
  //                  "dataset/t10k-labels.idx1-ubyte", perm_arena);

  conv conv;
  conv_init(&conv, 3, 3, 8, perm_arena);
  softmax sm;
  softmax_init(&sm, 13 * 13 * 8, 10, perm_arena);

  f32 loss = 0.0f;
  u64 num_correct = 0;

  for (u64 i = 0; i < train_data->count; i++) {
    arena_clear(temp_arena);

    u64 label = train_data->labels[i];
    feature_map *fm = conv_forward(&conv, train_data, temp_arena, i);
    max_pool(2, fm, temp_arena);
    f32 *probs = softmax_forward(&sm, fm, temp_arena);
    prediction p = {.probabilites = probs,
                    .loss = cross_entropy_loss(probs, label),
                    .accuracy = accuracy(probs, 10, label)};
    f32 *gradient = PUSH_ARRAY(temp_arena, f32, sm.nodes);
    gradient[label] = -1.0f / p.probabilites[label];
    feature_map *grad = softmax_backprop(&sm, gradient, 0.005f, temp_arena);
    grad = max_pool_backprop(2, fm, grad, temp_arena);
    conv_backprop(&conv, grad, 0.005f, temp_arena);
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
