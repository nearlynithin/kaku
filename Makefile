CC := emcc

CFLAGS := -O0 \
					-Iinclude \
					-Wall \
					-g

LDFLAGS := \
					 -sWASM=1 \
					 --preload-file dataset \
					 -sALLOW_MEMORY_GROWTH \
					 -sEXPORTED_FUNCTIONS=_main \
					 -sEXPORTED_RUNTIME_METHODS=ccall,cwrap \
					 -sASSERTIONS=2 \
					 -sSTACK_OVERFLOW_CHECK=2

BUILD := build
TARGET := kaku

SRC := $(wildcard src/*.c)

.PHONY: all clean

all: $(BUILD)/$(TARGET).js

$(BUILD)/$(TARGET).js: $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
	cp $(BUILD)/$(TARGET).* web/

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)/*
	rm -rf web/$(TARGET).js web/$(TARGET).wasm
