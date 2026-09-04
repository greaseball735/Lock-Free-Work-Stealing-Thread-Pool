CXX := g++
CXXFLAGS := -std=c++20 -O3 -march=native -Wall -Wextra -Iinclude -I. -fopenmp -pthread
LDFLAGS := -fopenmp -pthread

BIN_DIR := bin
SRC_DIR := benchmarks
TEST_DIR := tests

TARGETS := $(BIN_DIR)/test_job_system \
           $(BIN_DIR)/bench_overhead \
           $(BIN_DIR)/bench_recursive \
           $(BIN_DIR)/bench_parallel_for \
           $(BIN_DIR)/bench_unbalanced \
           $(BIN_DIR)/bench_dependencies \
           $(BIN_DIR)/bench_memory_model \
           $(BIN_DIR)/generate_mandelbrot

$(BIN_DIR)/generate_mandelbrot: $(SRC_DIR)/generate_mandelbrot.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

.PHONY: all clean run test

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/test_job_system: $(TEST_DIR)/test_job_system.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

$(BIN_DIR)/bench_overhead: $(SRC_DIR)/bench_overhead.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

$(BIN_DIR)/bench_recursive: $(SRC_DIR)/bench_recursive.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

$(BIN_DIR)/bench_parallel_for: $(SRC_DIR)/bench_parallel_for.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

$(BIN_DIR)/bench_unbalanced: $(SRC_DIR)/bench_unbalanced.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

$(BIN_DIR)/bench_dependencies: $(SRC_DIR)/bench_dependencies.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

$(BIN_DIR)/bench_memory_model: $(SRC_DIR)/bench_memory_model.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

test: $(BIN_DIR)/test_job_system
	./$(BIN_DIR)/test_job_system

run: all
	@echo "================================================================"
	@echo " Running Full Benchmark Suite for Job System 2.0"
	@echo "================================================================"
	@echo ""
	./$(BIN_DIR)/test_job_system
	@echo ""
	./$(BIN_DIR)/bench_overhead 100000
	@echo ""
	./$(BIN_DIR)/bench_recursive 5000000
	@echo ""
	./$(BIN_DIR)/bench_parallel_for 3000000
	@echo ""
	./$(BIN_DIR)/bench_unbalanced 2048 500
	@echo ""
	./$(BIN_DIR)/bench_dependencies 20000
	@echo ""
	./$(BIN_DIR)/bench_memory_model 200000

clean:
	rm -rf $(BIN_DIR) *.o test_job_system test_job_system_tsan bench_*
