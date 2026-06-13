#include "arena.h"
#include "cnn.h"
#include "data.h"
#include <emscripten.h>
#include <emscripten/em_asm.h>
#include <emscripten/em_macros.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static mem_arena *perm_arena;
static mem_arena *temp_arena;
static mem_arena *data_arena;
static dataset *train_data;
static conv cn;
static softmax sm;

b8 is_model_loaded() {
  FILE *f = fopen(MODEL_FILE, "rb");
  if (!f) {
    return false;
  } else
    return true;
}

void on_fs_ready(void) {
  perm_arena = arena_create(MiB(8));
  conv_init(&cn, 3, 3, 8, perm_arena);
  softmax_init(&sm, 13 * 13 * 8, 10, perm_arena);

  if (is_model_loaded()) {
    load_model(&cn, &sm, perm_arena);
    printf("Loaded saved model.\n");
  } else {
    printf("No saved model found, training...\n");
    data_arena = arena_create(MiB(64));
    train_data = load_dataset("dataset/train-images.idx3-ubyte",
                              "dataset/train-labels.idx1-ubyte", data_arena);

    temp_arena = arena_create(MiB(16));
    train(&cn, &sm, train_data, temp_arena);
    arena_destroy(temp_arena);
    arena_destroy(data_arena);

    save_model(&cn, &sm);
    printf("Training complete, model saved.\n");
  }

  temp_arena = arena_create(MiB(2));
}

EMSCRIPTEN_KEEPALIVE
u64 predict(u8 *data) {
  arena_clear(temp_arena);
  image img = {.data = data, .width = 28, .height = 28};
  return forward_image(&cn, &sm, &img, perm_arena);
}

int main() {
  srand(time(NULL));

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
}
