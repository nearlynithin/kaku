#include "arena.h"
#include "cnn.h"
#include "data.h"
#include <emscripten.h>
#include <emscripten/em_asm.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static mem_arena *perm_arena;
static mem_arena *temp_arena;
static dataset *train_data;
static conv cn;
static softmax sm;

void on_fs_ready(void) {
  if (load_model(&cn, &sm, perm_arena)) {
    printf("Loaded saved model.\n");
  } else {
    printf("No saved model found, training...\n");
    train(&cn, &sm, train_data, temp_arena);
    save_model(&cn, &sm);
    printf("Training complete, model saved.\n");
  }
}

int main() {
  srand(time(NULL));
  perm_arena = arena_create(MiB(64));
  temp_arena = arena_create(MiB(64));

  train_data = load_dataset("dataset/train-images.idx3-ubyte",
                            "dataset/train-labels.idx1-ubyte", perm_arena);

  conv_init(&cn, 3, 3, 8, perm_arena);
  softmax_init(&sm, 13 * 13 * 8, 10, perm_arena);

  EM_ASM({
    FS.mkdir("/save");
    FS.mount(IDBFS, {}, "/save");
    FS.syncfs(
        true, function(err) {
          if (err)
            console.log(err);
          Module._on_fs_ready();
        });
  });

  return 0;
}
