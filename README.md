# Lock-Free Work-Stealing Thread Pool

A high-performance C++20 implementation of Stefan Reinalter's Job System 2.0 architecture from the Molecular Musings series with all the good things -- work-stealing Chase-Lev deques, a custom lock-free ring-buffer allocator with cache-line padding to avoid false sharing, atomic counter-based parent/child job tracking, parallel_for, continuations, explicit x86/C++11 memory ordering, and full job dependencies. 
Design and architecture credit to the original blog series.

(part 1 of 5) one of many awesome blogs.<br>
https://blog.molecular-matters.com/2015/08/24/job-system-2-0-lock-free-work-stealing-part-1-basics/ 

Jeff Preshing’s blogs, very old but still relevant and fun to read.       
https://preshing.com/20120612/an-introduction-to-lock-free-programming/                               


<br>
<figure>
  <img src="https://github.com/user-attachments/assets/13408a70-140b-43fb-b45e-6a088a2d0585" alt="Mandelbrot visualization">
  <figcaption>A sample Mandelbrot visualization rendered using the thread pool</figcaption>
</figure>

## Architecture Overview

1. **Part 1: The Basics (Work-Stealing & Dynamic Parallelism)**
   - Per-worker double-ended queues (deques).
   - Owner threads Push and Pop from the **private LIFO end** (maximizing cache locality).
   - Thief threads Steal from the **public FIFO end** (stealing the largest tasks in divide-and-conquer trees).
   - Active waiting ("Help-first"): threads waiting on jobs actively execute local jobs or steal work rather than putting the OS thread to sleep.
   - Dynamic parallelism: subtasks can be dynamically spawned as children of active running jobs (`CreateJobAsChild`).

2. **Part 2: Specialized Allocator (Thread-Local Ring Buffer)**
   - Eliminates `new`, `delete`, and `malloc` entirely.
   - Each thread owns a thread-local circular buffer of pre-allocated `Job` instances.
   - Allocations take **0 locks, 0 atomics, and 0 system calls**.
   - No explicit deallocation needed during frames; entries are safely recycled upon ring wrap-around.

3. **Part 3: Going Lock-Free (Chase-Lev Work-Stealing Deque)**
   - Lock-free SPMC (Single-Producer Multi-Consumer) circular work-stealing deque.
   - Pop() fast path requires **0 atomic CAS operations** when more than 1 item is present.
   - Top and bottom counters are placed on distinct 64-byte cache lines to eliminate false sharing.
   - Configurable memory orderings:
     - `MemoryOrderingPolicy::Weak`: Full C++11 acquire/release semantics and sequential consistency fences, portable across x86, ARM, and POWER.
     - `MemoryOrderingPolicy::TSO_Optimized`: Uses x86 TSO (Total Store Order) hardware guarantees (Store-Store and Load-Load hardware ordering), utilizing atomic `xchg` on Pop for minimum overhead.
     - `MemoryOrderingPolicy::Relaxed_Experiment`: Experimental mode to benchmark the cost of fences on x86.

4. **Part 4: High-Level `parallel_for` & Splitters**
   - Recursive binary decomposition interleaving splitting and execution.
   - Configurable splitters:
     - `CountSplitter`: splits ranges until element count <= threshold.
     - `DataSizeSplitter`: splits ranges until the data working set fits completely within the CPU's **L1 cache** (e.g. 32 KB) or L2 cache.
     - `AutoSplitter`: distributes chunks proportionally across worker threads.

5. **Part 5: Dependencies & Continuations**
   - Direct continuation model embedded in the 128-byte `Job` structure.
   - `AddContinuation(ancestor, continuation)`: as soon as an ancestor finishes, its registered continuations are immediately dispatched to the work queues.

---

## Directory Structure

```
.
├── include/
│   └── job_system/
│       ├── job.hpp                 # 128-byte aligned Job struct with in-place data & continuations
│       ├── job_allocator.hpp       # Zero-atomic thread-local ring buffer allocator
│       ├── work_stealing_queue.hpp # Lock-free Chase-Lev deque (Weak & TSO models)
│       ├── job_system.hpp          # JobSystem coordinator, work-stealing, and active waiting
│       ├── splitters.hpp           # CountSplitter, DataSizeSplitter (cache-aware), AutoSplitter
│       └── parallel_for.hpp        # Recursive divide-and-conquer parallel_for & lambda support
├── baselines/
│   ├── mutex_thread_pool.hpp       # Classical generic thread pool (global queue + mutex + cv)
│   └── openmp_baselines.hpp        # OpenMP reference implementations
├── benchmarks/
│   ├── bench_common.hpp            # High-precision timer and comparison table formatter
│   ├── bench_overhead.cpp          # Fine-grained micro-task scheduling overhead
│   ├── bench_recursive.cpp         # Recursive divide-and-conquer (Parallel MergeSort)
│   ├── bench_parallel_for.cpp      # parallel_for with cache-size vs count splitters
│   ├── bench_unbalanced.cpp        # Irregular workload load-balancing (Mandelbrot)
│   ├── bench_dependencies.cpp      # Task dependency DAG pipelines (Part 5 continuations)
│   |── bench_memory_model.cpp      # Weak vs TSO vs Relaxed memory ordering comparison
|   └── generate_mandelbrot.cpp     # a fun test to generate the famous mandelbrot figures, simple version. 
├── tests/
│   └── test_job_system.cpp         # Unit tests and ThreadSanitizer validation
└── Makefile
```

---

## Building and Running

### Build all targets:
```bash
make all
```

### Run tests:
```bash
make test
```

### Run complete benchmark suite:
```bash
make run
```

### generate mandelbrot figure using the thread pool
```bash
# Usage: ./bin/generate_mandelbrot <width> <height> <max_iter> <output.ppm>                                                                 
    ./bin/generate_mandelbrot 3840 2160 2000 4k_mandelbrot.ppm                                                                                  
                                                                                                                                                
    # Convert PPM to PNG with ffmpeg                                                                                                            
    ffmpeg -y -i 4k_mandelbrot.ppm 4k_mandelbrot.png    
```      
