#include "cnn.h"
#include "arena.h"
#include <emscripten.h>
#include <emscripten/em_asm.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

f32 random_weight() {
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

feature_map *conv_forward(conv *conv, dataset *data, mem_arena *arena,
                          u64 image_index) {

  conv->last_data = data;
  conv->last_image_index = image_index;
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

  feature_map *fm = PUSH_STRUCT(arena, feature_map);
  fm->data = feature_maps;
  fm->width = out_w;
  fm->height = out_h;
  fm->depth = conv->num_filters;
  return fm;
}

void conv_backprop(conv *conv, feature_map *d_L_d_out, f32 lr,
                   mem_arena *arena) {
  dataset *data = conv->last_data;
  u64 image_index = conv->last_image_index;

  u64 filter_size = conv->filters[0].width * conv->filters[0].height;
  f32 *filter_grad = PUSH_ARRAY(arena, f32, filter_size);
  for (u64 f = 0; f < conv->num_filters; f++) {

    memset(filter_grad, 0, filter_size * sizeof(f32));

    for (u64 y = 0; y < d_L_d_out->height; y++) {
      for (u64 x = 0; x < d_L_d_out->width; x++) {

        f32 gradient =
            d_L_d_out->data[f * d_L_d_out->width * d_L_d_out->height +
                            y * d_L_d_out->width + x];

        u8 *image = data->images + image_index * data->rows * data->cols;

        for (u64 kr = 0; kr < conv->filters[f].height; kr++) {
          for (u64 kc = 0; kc < conv->filters[f].height; kc++) {
            f32 pixel =
                (image[(y + kr) * data->cols + (x + kc)] / 255.0f) - 0.5f;

            filter_grad[kr * conv->filters[f].width + kc] += gradient * pixel;
          }
        }
      }
    }

    for (u64 i = 0; i < conv->filters[f].width * conv->filters[f].height; i++) {
      conv->filters[f].data[i] -= lr * filter_grad[i];
    }
  }
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

void max_pool(u64 n, feature_map *fm, mem_arena *arena) {

  fm->last_input = fm->data;
  fm->data = PUSH_ARRAY(arena, f32, fm->width * fm->height * fm->depth);

  fm->last_width = fm->width;
  fm->last_height = fm->height;
  fm->last_depth = fm->depth;

  u64 out_w = fm->width / 2;
  u64 out_h = fm->height / 2;

  for (u64 d = 0; d < fm->depth; d++) {
    for (u64 r = 0; r < fm->height; r += n) {
      for (u64 c = 0; c < fm->width; c += n) {

        f32 max = -FLT_MAX;
        for (u64 pr = 0; pr < n; pr++) {
          for (u64 pc = 0; pc < n; pc++) {

            u64 in_r = r + pr;
            u64 in_c = c + pc;

            f32 value = fm->last_input[(d * fm->width * fm->height) +
                                       in_r * fm->width + in_c];
            if (value > max)
              max = value;
          }
        }

        u64 out_index =
            (d * out_w * out_h) + (u64)(r / n) * out_w + (u64)(c / n);
        fm->data[out_index] = max;
      }
    }
  }
  fm->width = out_w;
  fm->height = out_h;
}

feature_map *max_pool_backprop(u64 n, feature_map *fm, feature_map *d_L_d_out,
                               mem_arena *arena) {
  feature_map *grad = PUSH_STRUCT(arena, feature_map);
  grad->width = fm->last_width;
  grad->height = fm->last_height;
  grad->depth = fm->last_depth;
  grad->data = PUSH_ARRAY(arena, f32, grad->width * grad->height * grad->depth);

  for (u64 d = 0; d < grad->depth; d++) {
    for (u64 r = 0; r < grad->height; r += n) {
      for (u64 c = 0; c < grad->width; c += n) {

        f32 max = -FLT_MAX;
        u64 max_r = 0;
        u64 max_c = 0;
        for (u64 pr = 0; pr < n; pr++) {
          for (u64 pc = 0; pc < n; pc++) {

            u64 in_r = r + pr;
            u64 in_c = c + pc;

            f32 value = fm->last_input[(d * grad->width * grad->height) +
                                       in_r * grad->width + in_c];
            if (value > max) {
              max = value;
              max_r = in_r;
              max_c = in_c;
            }
          }
        }

        u64 out_index = (d * fm->width * fm->height) +
                        (u64)(r / n) * fm->width + (u64)(c / n);
        f32 incoming = d_L_d_out->data[out_index];

        grad->data[d * grad->width * grad->height + max_r * grad->width +
                   max_c] = incoming;
      }
    }
  }
  return grad;
}

void softmax_init(softmax *sm, u64 input_len, u64 nodes, mem_arena *arena) {
  sm->input_len = input_len;
  sm->nodes = nodes;

  sm->weights = PUSH_ARRAY(arena, f32, input_len * nodes);
  sm->biases = PUSH_ARRAY(arena, f32, nodes);

  for (u64 i = 0; i < input_len * nodes; i++)
    sm->weights[i] = random_weight();
}

f32 *softmax_forward(softmax *sm, feature_map *fm, mem_arena *arena) {
  f32 *totals = PUSH_ARRAY(arena, f32, sm->nodes);
  f32 *probs = PUSH_ARRAY(arena, f32, sm->nodes);

  sm->last_input = fm->data;
  sm->last_width = fm->width;
  sm->last_height = fm->height;
  sm->last_depth = fm->depth;

  for (u64 n = 0; n < sm->nodes; n++) {
    f32 total = sm->biases[n];
    for (u64 i = 0; i < sm->input_len; i++)
      total += fm->data[i] * sm->weights[i * sm->nodes + n];

    totals[n] = total;
    probs[n] = total;
  }

  sm->last_totals = totals;

  f32 max = probs[0];
  for (u64 i = 1; i < sm->nodes; i++)
    if (probs[i] > max)
      max = probs[i];

  f32 sum = 0.0f;
  for (u64 i = 0; i < sm->nodes; i++) {
    f32 ex = expf(probs[i] - max);
    sum += ex;
    probs[i] = ex;
  }

  for (u64 i = 0; i < sm->nodes; i++)
    probs[i] = probs[i] / sum;

  return probs;
}

feature_map *softmax_backprop(softmax *sm, f32 *d_L_d_out, f32 lr,
                              mem_arena *arena) {

  u64 target = 0;
  f32 gradient = 0.0f;
  feature_map *grad = PUSH_STRUCT(arena, feature_map);
  grad->data = PUSH_ARRAY(arena, f32, sm->input_len);
  f32 *t_exp = PUSH_ARRAY(arena, f32, sm->nodes);
  f32 *d_out_d_t = PUSH_ARRAY(arena, f32, sm->nodes);
  f32 *d_L_d_t = PUSH_ARRAY(arena, f32, sm->nodes);
  f32 *d_L_d_w = PUSH_ARRAY(arena, f32, sm->nodes * sm->input_len);

  for (u64 i = 0; i < sm->nodes; i++) {
    if (d_L_d_out[i] == 0.0f)
      continue;
    target = i;
    gradient = d_L_d_out[i];

    f32 S = 0.0f;
    for (u64 n = 0; n < sm->nodes; n++) {
      t_exp[n] = expf(sm->last_totals[n]);
      S += t_exp[n];
    }

    // Gradients of out[i] against totals
    for (u64 n = 0; n < sm->nodes; n++)
      d_out_d_t[n] = -(t_exp[target] * t_exp[n]) / (S * S);

    d_out_d_t[target] = (t_exp[target] * (S - t_exp[target])) / (S * S);

    // Gradients of loss against totals
    for (u64 n = 0; n < sm->nodes; n++)
      d_L_d_t[n] = gradient * d_out_d_t[n];

    // Weight gradients
    for (u64 j = 0; j < sm->input_len; j++) {
      for (u64 n = 0; n < sm->nodes; n++) {
        d_L_d_w[j * sm->nodes + n] = sm->last_input[j] * d_L_d_t[n];
      }
    }

    grad->width = sm->last_width;
    grad->height = sm->last_height;
    grad->depth = sm->last_depth;

    for (u64 j = 0; j < sm->input_len; j++) {
      f32 sum = 0.0f;
      for (u64 n = 0; n < sm->nodes; n++)
        sum += sm->weights[j * sm->nodes + n] * d_L_d_t[n];
      grad->data[j] = sum;
    }

    for (u64 j = 0; j < sm->input_len * sm->nodes; j++)
      sm->weights[j] -= lr * d_L_d_w[j];
    for (u64 j = 0; j < sm->nodes; j++)
      sm->biases[j] -= lr * d_L_d_t[j];

    break;
  }
  return grad;
}

f32 cross_entropy_loss(f32 *probs, u64 label) { return -logf(probs[label]); }

u64 accuracy(f32 *probs, u64 nodes, u64 label) {
  u64 best = 0;
  for (u64 i = 1; i < nodes; i++) {
    if (probs[i] > probs[best])
      best = i;
  }

  return (best == label) ? 1 : 0;
}

void train(conv *conv, softmax *sm, dataset *train_data, mem_arena *arena) {

  f32 loss = 0.0f;
  u64 num_correct = 0;

  for (u64 i = 0; i < train_data->count; i++) {
    arena_clear(arena);

    u64 label = train_data->labels[i];
    feature_map *fm = conv_forward(conv, train_data, arena, i);
    max_pool(2, fm, arena);
    f32 *probs = softmax_forward(sm, fm, arena);
    prediction p = {.probabilites = probs,
                    .loss = cross_entropy_loss(probs, label),
                    .accuracy = accuracy(probs, 10, label)};
    f32 *gradient = PUSH_ARRAY(arena, f32, sm->nodes);
    gradient[label] = -1.0f / p.probabilites[label];
    feature_map *grad = softmax_backprop(sm, gradient, 0.005f, arena);
    grad = max_pool_backprop(2, fm, grad, arena);
    conv_backprop(conv, grad, 0.005f, arena);
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
}

void save_model(conv *conv, softmax *sm) {
  FILE *f = fopen(MODEL_FILE, "wb");
  if (!f) {
    printf("Failed to open %s", MODEL_FILE);
    return;
  }

  // write conv
  fwrite(&conv->num_filters, sizeof(u64), 1, f);
  for (u64 i = 0; i < conv->num_filters; i++) {
    fwrite(conv->filters[i].data, sizeof(f32),
           conv->filters[i].width * conv->filters[i].height, f);
  }

  // write softmax
  fwrite(&sm->input_len, sizeof(u64), 1, f);
  fwrite(&sm->nodes, sizeof(u64), 1, f);
  fwrite(sm->weights, sizeof(f32), sm->input_len * sm->nodes, f);
  fwrite(&sm->biases, sizeof(f32), sm->nodes, f);

  fclose(f);
  EM_ASM({
    if (Module.__syncing)
      return;
    Module.__syncing = true;

    FS.syncfs(function(err) {
      Module.__syncing = false;
      if (err)
        console.log(err);
    });
  });
}

int load_model(conv *conv, softmax *sm, mem_arena *arena) {
  FILE *f = fopen(MODEL_FILE, "rb");
  if (!f) {
    return 0;
  }

  // read conv
  fread(&conv->num_filters, sizeof(u64), 1, f);
  for (u64 i = 0; i < conv->num_filters; i++) {
    fread(conv->filters[i].data, sizeof(f32),
          conv->filters[i].width * conv->filters[i].height, f);
  }

  // read softmax
  fread(&sm->input_len, sizeof(u64), 1, f);
  fread(&sm->nodes, sizeof(u64), 1, f);

  sm->weights = PUSH_ARRAY(arena, f32, sm->input_len * sm->nodes);
  sm->biases = PUSH_ARRAY(arena, f32, sm->nodes);

  fread(sm->weights, sizeof(f32), sm->input_len * sm->nodes, f);
  fread(sm->biases, sizeof(f32), sm->nodes, f);

  fclose(f);
  return 1;
}
