#include "job_system/job_system.hpp"
#include "job_system/parallel_for.hpp"
#include "benchmarks/bench_common.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace job_system;

struct alignas(16) Particle {
    float x, y, z, pad1;
    float vx, vy, vz, pad2;
    float ax, ay, az, mass;
    float life, drag, pad3, pad4;
};

static_assert(sizeof(Particle) == 64, "Particle must be 64 bytes for cache line alignment");

inline void UpdateParticleKernel(Particle& p, float dt) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.vz += p.az * dt;

    p.vx *= p.drag;
    p.vy *= p.drag;
    p.vz *= p.drag;

    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.z += p.vz * dt;

    p.life -= dt;
}

void BenchmarkParallelFor(size_t particleCount, int iterations) {
    std::cout << ">>> Running Data-Parallel parallel_for Benchmark ("
              << particleCount << " particles, 64-byte structs) <<<\n";

    std::vector<Particle> masterParticles(particleCount);
    for (size_t i = 0; i < particleCount; ++i) {
        masterParticles[i] = Particle{
            static_cast<float>(i % 100), static_cast<float>(i % 200), static_cast<float>(i % 300), 0.0f,
            1.5f, 2.5f, -0.5f, 0.0f,
            0.0f, -9.81f, 0.0f, 1.0f,
            100.0f, 0.995f, 0.0f, 0.0f
        };
    }

    constexpr float dt = 0.016f; // 60 FPS delta time
    std::vector<Particle> particles;

    // 1. Sequential Baseline
    auto seqBench = bench::RunRepeated("Sequential Loop", iterations, 1.0, [&]() {
        particles = masterParticles;
        for (size_t i = 0; i < particleCount; ++i) {
            UpdateParticleKernel(particles[i], dt);
        }
    });
    double seqMs = seqBench.medianMs;
    seqBench.speedupVsSeq = 1.0;

    // 2. OpenMP Static Parallel For
    auto ompStaticBench = bench::RunRepeated("OpenMP parallel for (static)", iterations, seqMs, [&]() {
        particles = masterParticles;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < particleCount; ++i) {
            UpdateParticleKernel(particles[i], dt);
        }
    });

    // 3. OpenMP Guided Parallel For
    auto ompGuidedBench = bench::RunRepeated("OpenMP parallel for (guided)", iterations, seqMs, [&]() {
        particles = masterParticles;
        #pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < particleCount; ++i) {
            UpdateParticleKernel(particles[i], dt);
        }
    });

    // 4. Job System 2.0 with CountSplitter (1024 chunks)
    DefaultJobSystem js;
    auto jsCountBench = bench::RunRepeated("Job System 2.0 (CountSplitter)", iterations, seqMs, [&]() {
        particles = masterParticles;
        parallel_for(js, particles.data(), particleCount, [](Particle* chunk, size_t count, const void*) {
            for (size_t i = 0; i < count; ++i) {
                UpdateParticleKernel(chunk[i], dt);
            }
        }, nullptr, CountSplitter(1024));
    });

    // 5. Job System 2.0 with DataSizeSplitter (32KB L1 cache working set)
    auto jsL1Bench = bench::RunRepeated("Job System 2.0 (DataSize L1 32KB)", iterations, seqMs, [&]() {
        particles = masterParticles;
        parallel_for(js, particles.data(), particleCount, [](Particle* chunk, size_t count, const void*) {
            for (size_t i = 0; i < count; ++i) {
                UpdateParticleKernel(chunk[i], dt);
            }
        }, nullptr, DataSizeSplitter(32 * 1024));
    });

    std::vector<bench::BenchmarkResult> results = {
        seqBench, ompStaticBench, ompGuidedBench, jsCountBench, jsL1Bench
    };
    bench::PrintResultsTable("parallel_for Data-Parallelism (" + std::to_string(particleCount) + " particles)", results);
}

int main(int argc, char** argv) {
    size_t count = 3000000;
    if (argc > 1) {
        count = std::stoull(argv[1]);
    }
    BenchmarkParallelFor(count, 5);
    return 0;
}
