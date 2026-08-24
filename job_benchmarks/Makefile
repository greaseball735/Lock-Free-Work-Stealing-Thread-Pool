CXX ?= g++
CXXFLAGS ?= -O3 -march=native -DNDEBUG -std=c++20 -pthread
OMPFLAGS := -fopenmp

TARGET := job_bench
SRC := benchmark.cpp ../job_system.cpp

all: $(TARGET)

$(TARGET): $(SRC) mutex_pool.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I.. $(SRC) -o $@ $(OMPFLAGS) -pthread

clean:
	rm -f $(TARGET) results.csv

.PHONY: all clean
