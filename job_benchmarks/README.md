# Job System Benchmark Suite

Compares four implementations under the same workload definitions:

- sequential
- mutex-protected global thread pool
- custom work-stealing job system
- OpenMP

Suites:

1. `empty_jobs`: scheduler/task overhead.
2. `fine_grained`: many equally sized small tasks.
3. `irregular`: heavy-tailed task durations, stressing dynamic load balance.
4. `parallel_for`: same array work while sweeping grain size.

The current custom allocator in `job_system.cpp` has a per-thread finite job pool. The harness therefore caps each custom batch at 50,000 descriptors and uses multiple batches for larger totals.

## Build

Expected layout:

```
project/
  job_system.hpp
  job_system.cpp
  job_benchmarks/
    benchmark.cpp
    mutex_pool.hpp
    Makefile
    run_bench.sh
```

Run:

```
cd job_benchmarks
./run_bench.sh
```

Set worker counts:

```
BENCH_THREADS=1,2,4,8 ./run_bench.sh
```

Set implementations:

```
BENCH_IMPLS=mutex,custom,openmp BENCH_THREADS=1,2,4,8 ./run_bench.sh
```

Run one suite:

```
./job_bench empty
./job_bench fine
./job_bench irregular
./job_bench parallel_for
```

## Recommended first experiments

### Scheduler overhead

```
BENCH_THREADS=1,2,4,8 BENCH_IMPLS=sequential,mutex,custom,openmp ./job_bench empty > empty.csv
```

### Work stealing / irregular load balance

```
BENCH_THREADS=1,2,4,8 BENCH_IMPLS=sequential,mutex,custom,openmp ./job_bench irregular > irregular.csv
```

### Grain-size crossover

```
BENCH_THREADS=1,2,4,8 BENCH_IMPLS=sequential,mutex,custom,openmp ./job_bench parallel_for > parallel_for.csv
```

## Metrics in the CSV

- `best_ms`: best wall-clock sample
- `avg_ms`: average wall-clock sample
- `throughput_mjobs_s`: millions of logical work items/sec
- `checksum`: correctness guard

For the report, derive:

- speedup = baseline_time / custom_time
- parallel efficiency = speedup / thread_count
- scheduler overhead = empty-job time / number of jobs
- grain-size crossover = first grain where scheduler overhead stops dominating

## Hardware-counter runs

Wall-clock results alone do not explain why the implementation wins. On Linux, run the same executable through `perf stat`:

```
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations ./job_bench irregular
```

For a single controlled case:

```
BENCH_THREADS=8 BENCH_IMPLS=mutex,custom ./job_bench irregular >/tmp/irregular.csv
perf stat -e cycles,instructions,cache-misses,context-switches,cpu-migrations ./job_bench irregular >/dev/null
```

Pin the benchmark to a fixed CPU set when collecting final numbers, e.g.:

```
BENCH_THREADS=1,2,4,8 taskset -c 0-7 ./job_bench parallel_for > parallel_for.csv
```

Use the same CPU set for every implementation.
