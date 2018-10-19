OS = $(shell uname -s)

CXX := g++
CXXFLAGS := -std=c++11 -O3

ifeq ($(OS), Darwin)
  LINKARGS := install_name
else
  LINKARGS := soname
endif

SRCS = $(wildcard src/*.cpp)
OBJS = $(patsubst src/%.cpp,bin/%.o,$(SRCS))

bin/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@

ckmeans: $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -Wl,-$(LINKARGS),bin/libckmeans.so -o bin/libckmeans.so $(OBJS)

.PHONY: clean

clean:
	rm -f bin/*.o bin/libckmeans.so
