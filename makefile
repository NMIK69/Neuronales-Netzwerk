# The name of the main executable
TARGET=nn_main

# Flags and stuff, change as required
OPTIMIZE=-O2 -ftree-vectorize -ffast-math
# Flags for all languages
CPPFLAGS=-ggdb $(OPTIMIZE) -Wall -MMD -MP 
# Flags for C only
CFLAGS=-std=c11 -Wmissing-prototypes -fopenmp -fopenmp-simd
# Flags for C++ only
CXXFLAGS=-std=c++11
# Flags for the linker
LDFLAGS=-fopenmp -fopenmp-simd
# Additional linker libs
LDLIBS=-lm 

# Compilers
CC=gcc

SRCS=$(filter-out %_test.c,$(wildcard *.c))

# We make up the objects by replacing the .c and .cpp suffixes
# with a .o suffix
OBJS=$(patsubst %.c,%.o,$(SRCS))

# The test sources are assumed to end with _test.c 
TEST_SRCS=$(wildcard *_test.c)

# The test objects
TEST_OBJS=$(patsubst %.c,%.o,$(TEST_SRCS))

# The test executables (without any suffix)
TESTS=$(patsubst %.c,%,$(TEST_SRCS))

# The dependency files
DEPS=$(SRCS:.c=.d) $(TEST_SRCS:.c=.d)


# The first target (all) is allways the default target
.PHONY: all
all: build test

# Our buld target depends on the real target
.PHONY: build
build: $(TARGET)

# Our target is built up from the objects
$(TARGET): $(OBJS)

# Our test target
# 
# Here we instruct make to generate a line with all test
# which are all sequentially executed using the $(foreach)
# macro. We conclude with 'true' because the && construct
# awaits an argument on both sides. The 'true' command
# simply returns a success value
.PHONY: test
test: $(TESTS)
	$(foreach T,$(TESTS), ./$(T) &&) true

# Cleanup all generated files
clean:
	rm -Rf $(TEST_OBJS) $(TESTS) $(OBJS) $(TARGET) $(DEPS)

-include $(DEPS)