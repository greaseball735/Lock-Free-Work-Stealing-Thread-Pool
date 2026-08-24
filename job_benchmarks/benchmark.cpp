#include "../job_system.hpp"
#include "mutex_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Clock = std::chrono::steady_clock;

static volatile std::uint64_t g_sink = 0;
static constexpr int WARMUPS = 2;
static constexpr int REPS = 7;
static constexpr std::uint64_t MAX_BATCH = 50000; // fits current TLS allocator

static std::uint64_t mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static std::uint64_t cpu_work(std::uint64_t x, int rounds) {
    for (int i = 0; i < rounds; ++i)
        x = mix64(x + static_cast<std::uint64_t>(i));
    return x;
}

struct Result { double best_ms, avg_ms; };

template<class F>
static Result measure(F&& f) {
    for (int i = 0; i < WARMUPS; ++i) f();
    double best = 1e300, sum = 0.0;
    for (int i = 0; i < REPS; ++i) {
        const auto t0 = Clock::now();
        f();
        const auto t1 = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        best = std::min(best, ms);
        sum += ms;
    }
    return {best, sum / REPS};
}

static void csv(const char* bench, const char* impl, int threads,
                std::uint64_t items, std::uint64_t grain,
                Result r, std::uint64_t checksum) {
    const double mjob_s = r.best_ms > 0
        ? (static_cast<double>(items) / 1e6) / (r.best_ms / 1000.0)
        : 0.0;
    std::printf("%s,%s,%d,%llu,%llu,%.3f,%.3f,%.3f,%llu\n",
        bench, impl, threads,
        static_cast<unsigned long long>(items),
        static_cast<unsigned long long>(grain),
        r.best_ms, r.avg_ms, mjob_s,
        static_cast<unsigned long long>(checksum));
}

// -----------------------------------------------------------------------------
// Task definitions
// -----------------------------------------------------------------------------

struct WorkPayload {
    std::uint64_t x;
    int rounds;
    std::uint64_t* out;
};

static void work_job(Job*, const void* raw) {
    const auto& p = *static_cast<const WorkPayload*>(raw);
    *p.out = cpu_work(p.x, p.rounds);
}

struct UnitPayload { std::uint64_t* out; std::uint64_t value; };
static void unit_job(Job*, const void* raw) {
    const auto& p = *static_cast<const UnitPayload*>(raw);
    *p.out = p.value;
}

struct RootPayload { };
static void root_job(Job*, const void*) {}

// -----------------------------------------------------------------------------
// Helpers for custom scheduler: keep outstanding main-thread allocations below
// MAX_JOBS_PER_THREAD by using batches. Each batch is still a real parent/child
// job tree and all worker execution is done by the custom scheduler.
// -----------------------------------------------------------------------------

template<class MakeChild>
static void custom_batched(int threads, std::uint64_t total, MakeChild&& make_child) {
    JobSystem::Initialize(static_cast<std::uint32_t>(threads));

    std::uint64_t done_items = 0;
    while (done_items < total) {
        const std::uint64_t n = std::min(MAX_BATCH, total - done_items);
        Job* root = JobSystem::CreateJob(root_job);
        for (std::uint64_t i = 0; i < n; ++i)
            make_child(root, done_items + i);
        JobSystem::Run(root);
        JobSystem::Wait(root);
        done_items += n;
    }

    JobSystem::Shutdown();
}

// -----------------------------------------------------------------------------
// Suite 1: empty jobs
// -----------------------------------------------------------------------------

static void run_empty(int threads, std::uint64_t jobs, const char* impl) {
    std::vector<std::uint64_t> out(jobs);
    auto body = [&] {
        std::fill(out.begin(), out.end(), 0);

        if (std::string(impl) == "sequential") {
            for (std::uint64_t i = 0; i < jobs; ++i) out[i] = i;
        } else if (std::string(impl) == "mutex") {
            MutexThreadPool pool(threads);
            for (std::uint64_t i = 0; i < jobs; ++i)
                pool.submit([&, i] { out[i] = i; });
            pool.wait();
        } else if (std::string(impl) == "custom") {
            custom_batched(threads, jobs, [&](Job* root, std::uint64_t i) {
                UnitPayload p{&out[i], i};
                Job* j = JobSystem::CreateJobAsChild(root, unit_job, p);
                JobSystem::Run(j);
            });
        }
#ifdef _OPENMP
        else if (std::string(impl) == "openmp") {
            omp_set_num_threads(threads);
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < static_cast<long long>(jobs); ++i)
                out[i] = static_cast<std::uint64_t>(i);
        }
#endif

        std::uint64_t c = 0;
        for (auto x : out) c ^= mix64(x);
        g_sink ^= c;
    };

    const auto r = measure(body);
    csv("empty_jobs", impl, threads, jobs, 1, r, g_sink);
}

// -----------------------------------------------------------------------------
// Suite 2: homogeneous fine-grained tasks
// -----------------------------------------------------------------------------

static void run_fine(int threads, std::uint64_t jobs, int rounds, const char* impl) {
    std::vector<std::uint64_t> out(jobs);
    auto body = [&] {
        std::fill(out.begin(), out.end(), 0);

        if (std::string(impl) == "sequential") {
            for (std::uint64_t i = 0; i < jobs; ++i)
                out[i] = cpu_work(i + 1, rounds);
        } else if (std::string(impl) == "mutex") {
            MutexThreadPool pool(threads);
            for (std::uint64_t i = 0; i < jobs; ++i)
                pool.submit([&, i] { out[i] = cpu_work(i + 1, rounds); });
            pool.wait();
        } else if (std::string(impl) == "custom") {
            custom_batched(threads, jobs, [&](Job* root, std::uint64_t i) {
                WorkPayload p{i + 1, rounds, &out[i]};
                Job* j = JobSystem::CreateJobAsChild(root, work_job, p);
                JobSystem::Run(j);
            });
        }
#ifdef _OPENMP
        else if (std::string(impl) == "openmp") {
            omp_set_num_threads(threads);
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < static_cast<long long>(jobs); ++i)
                out[i] = cpu_work(static_cast<std::uint64_t>(i + 1), rounds);
        }
#endif

        std::uint64_t c = 0;
        for (auto x : out) c ^= x;
        g_sink ^= c;
    };

    const auto r = measure(body);
    csv("fine_grained", impl, threads, jobs, 1, r, g_sink);
}

// -----------------------------------------------------------------------------
// Suite 3: irregular heavy-tailed tasks
// -----------------------------------------------------------------------------

static void run_irregular(int threads, std::uint64_t jobs, const char* impl) {
    std::mt19937_64 rng(1234567);
    std::vector<int> costs(jobs);
    for (auto& c : costs) {
        const auto x = rng() % 1000;
        c = x < 900 ? 1 : (x < 990 ? 10 : (x < 999 ? 100 : 1000));
    }

    std::vector<std::uint64_t> out(jobs);
    auto body = [&] {
        std::fill(out.begin(), out.end(), 0);

        if (std::string(impl) == "sequential") {
            for (std::uint64_t i = 0; i < jobs; ++i)
                out[i] = cpu_work(i + 1, costs[i]);
        } else if (std::string(impl) == "mutex") {
            MutexThreadPool pool(threads);
            for (std::uint64_t i = 0; i < jobs; ++i)
                pool.submit([&, i] { out[i] = cpu_work(i + 1, costs[i]); });
            pool.wait();
        } else if (std::string(impl) == "custom") {
            custom_batched(threads, jobs, [&](Job* root, std::uint64_t i) {
                WorkPayload p{i + 1, costs[i], &out[i]};
                Job* j = JobSystem::CreateJobAsChild(root, work_job, p);
                JobSystem::Run(j);
            });
        }
#ifdef _OPENMP
        else if (std::string(impl) == "openmp") {
            omp_set_num_threads(threads);
            #pragma omp parallel for schedule(dynamic,1)
            for (long long i = 0; i < static_cast<long long>(jobs); ++i)
                out[i] = cpu_work(static_cast<std::uint64_t>(i + 1), costs[i]);
        }
#endif

        std::uint64_t c = 0;
        for (auto x : out) c ^= x;
        g_sink ^= c;
    };

    const auto r = measure(body);
    csv("irregular", impl, threads, jobs, 1, r, g_sink);
}

// -----------------------------------------------------------------------------
// Suite 4: parallel-for / grain-size sweep
// -----------------------------------------------------------------------------

static void element_fn(int* x) {
    const auto v = cpu_work(static_cast<std::uint64_t>(*x + 1), 20);
    *x = static_cast<int>(v & 0x7fffffffULL);
}

static void run_parallel_for(int threads, std::uint64_t n, std::uint64_t grain,
                             const char* impl) {
    std::vector<int> a(n);
    auto body = [&] {
        std::iota(a.begin(), a.end(), 0);

        if (std::string(impl) == "sequential") {
            for (std::uint64_t i = 0; i < n; ++i) element_fn(&a[i]);
        } else if (std::string(impl) == "mutex") {
            MutexThreadPool pool(threads);
            for (std::uint64_t s = 0; s < n; s += grain) {
                const auto e = std::min(n, s + grain);
                pool.submit([&, s, e] {
                    for (std::uint64_t i = s; i < e; ++i) element_fn(&a[i]);
                });
            }
            pool.wait();
        } else if (std::string(impl) == "custom") {
            JobSystem::Initialize(static_cast<std::uint32_t>(threads));
            JobSystem::ParallelFor(
                a.data(), static_cast<std::uint32_t>(n),
                element_fn, static_cast<std::uint32_t>(grain));
            JobSystem::Shutdown();
        }
#ifdef _OPENMP
        else if (std::string(impl) == "openmp") {
            omp_set_num_threads(threads);
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < static_cast<long long>(n); ++i)
                element_fn(&a[i]);
        }
#endif

        std::uint64_t c = 0;
        for (auto x : a) c ^= mix64(static_cast<std::uint32_t>(x));
        g_sink ^= c;
    };

    const auto r = measure(body);
    csv("parallel_for", impl, threads, n, grain, r, g_sink);
}

static std::vector<int> parse_ints(const char* s, std::vector<int> def) {
    if (!s || !*s) return def;
    std::vector<int> out;
    std::string x(s);
    std::size_t p = 0;
    while (p < x.size()) {
        const auto q = x.find(',', p);
        const auto tok = x.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (!tok.empty()) out.push_back(std::stoi(tok));
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out.empty() ? def : out;
}

static std::vector<std::string> parse_strings(const char* s,
                                               std::vector<std::string> def) {
    if (!s || !*s) return def;
    std::vector<std::string> out;
    std::string x(s);
    std::size_t p = 0;
    while (p < x.size()) {
        const auto q = x.find(',', p);
        const auto tok = x.substr(p, q == std::string::npos ? std::string::npos : q - p);
        if (!tok.empty()) out.push_back(tok);
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out.empty() ? def : out;
}

int main(int argc, char** argv) {
    const std::string suite = argc > 1 ? argv[1] : "all";
    const auto threads = parse_ints(std::getenv("BENCH_THREADS"), {1,2,4,8});
    const auto impls = parse_strings(
        std::getenv("BENCH_IMPLS"),
        {"sequential","mutex","custom","openmp"});

    std::puts("benchmark,impl,threads,work_items,grain,best_ms,avg_ms,throughput_mjobs_s,checksum");

    for (int t : threads) {
        for (const auto& impl_s : impls) {
            const char* impl = impl_s.c_str();

            if (suite == "all" || suite == "empty") {
                run_empty(t, 1000, impl);
                run_empty(t, 10000, impl);
                run_empty(t, 50000, impl);
            }

            if (suite == "all" || suite == "fine") {
                run_fine(t, 10000, 12, impl);
                run_fine(t, 50000, 12, impl);
            }

            if (suite == "all" || suite == "irregular") {
                run_irregular(t, 10000, impl);
                run_irregular(t, 50000, impl);
            }

            if (suite == "all" || suite == "parallel_for") {
                constexpr std::uint64_t N = 4'000'000;
                for (auto grain : {16ULL, 64ULL, 256ULL, 1024ULL, 4096ULL})
                    run_parallel_for(t, N, grain, impl);
            }
        }
    }

    std::fprintf(stderr, "sink=%llu\n", static_cast<unsigned long long>(g_sink));
    return 0;
}
