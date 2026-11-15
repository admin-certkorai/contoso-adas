# Makefile - DMS-1 driver-monitoring ECU inference firmware.
#
# Two targets:
#   make            -> native build (host portability check / local demo)
#   make cross      -> aarch64 embedded-Linux ELF (the artifact CertOne certifies)
#   make train      -> re-run the host-side trainer + int8 codegen (regenerates
#                      src/model_weights.h, data/, eval/)
#   make run        -> build native + run on the bundled sample frames
#   make clean

CC      ?= gcc
CROSS   ?= aarch64-linux-gnu-
CROSSCC  = $(CROSS)gcc

CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDLIBS   = -lm
SRC      = src/inference.c
INC      = -Isrc

NATIVE_BIN = dms_infer
CROSS_BIN  = dms_infer_aarch64

.PHONY: all cross run train clean

all: $(NATIVE_BIN)

$(NATIVE_BIN): $(SRC) src/model_weights.h src/ecu_config.h
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) $(LDLIBS)

# Cross-compile the aarch64 ECU firmware. This ELF is what the CertOne worker
# runs under qemu-aarch64 inside the attested confidential VM.
cross: $(CROSS_BIN)

$(CROSS_BIN): $(SRC) src/model_weights.h src/ecu_config.h
	$(CROSSCC) $(CFLAGS) -static $(INC) -o $@ $(SRC) $(LDLIBS)
	@echo "built aarch64 ELF: $@"
	@file $@ 2>/dev/null || true

run: $(NATIVE_BIN)
	./$(NATIVE_BIN) data/sample_frame_features.csv

# Regenerate weights + eval + sample data from the seeded host trainer.
train:
	python src/train.py

clean:
	rm -f $(NATIVE_BIN) $(CROSS_BIN)
