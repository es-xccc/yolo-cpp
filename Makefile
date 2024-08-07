CXX := g++
CC := gcc
YOLO := yolo
YOLOOBJ := objects/yolo.o
LDFLAGS := $(shell pkg-config --libs opencv) -lpthread -lhineon -ln2cube -ldputils
MODEL := $(shell pwd)/model/dpu_yolo.elf
ARCH := $(shell uname -m | sed -e s/arm.*/armv71/ -e s/aarch64.*/aarch64/ )
CFLAGS := -O2 -Wall -Wpointer-arith -std=c++11 -ffast-math

ifeq ($(ARCH),armv71)
	CFLAGS += -mcpu=cortex-a9 -mfloat-abi=hard -mfpu=neon
endif
ifeq ($(ARCH),aarch64)
	CFLAGS += -mcpu=cortex-a53
endif

SRC_DIR = sources
OBJ_DIR = objects
SRC_FILES := $(wildcard $(SRC_DIR)/yolo.cpp $(SRC_DIR)/utils.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

.PHONY: all clean

all: $(YOLO)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

$(YOLO): $(OBJ_FILES)
	$(CXX) $(CFLAGS) $^ $(MODEL) -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJ_DIR)/*.o $(YOLO)