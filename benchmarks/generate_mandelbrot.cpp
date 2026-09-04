#include "job_system/job_system.hpp"
#include "job_system/parallel_for.hpp"
#include "benchmarks/bench_common.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>

using namespace job_system;

struct RGB {
    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};
};

// Smooth cosine palette (Inigo Quilez formula: a + b * cos(2*PI*(c*t + d)))
inline RGB ColorPalette(double t) {
    if (t < 0.0) return RGB{0, 0, 0};

    // Electric Fire & Gold palette parameters
    const double a[3] = {0.5, 0.5, 0.5};
    const double b[3] = {0.5, 0.5, 0.5};
    const double c[3] = {1.0, 1.0, 1.0};
    const double d[3] = {0.0, 0.33, 0.67};

    const double PI = 3.14159265358979323846;

    double r = a[0] + b[0] * std::cos(2.0 * PI * (c[0] * t + d[0]));
    double g = a[1] + b[1] * std::cos(2.0 * PI * (c[1] * t + d[1]));
    double bl = a[2] + b[2] * std::cos(2.0 * PI * (c[2] * t + d[2]));

    return RGB{
        static_cast<uint8_t>(std::clamp(r * 255.0, 0.0, 255.0)),
        static_cast<uint8_t>(std::clamp(g * 255.0, 0.0, 255.0)),
        static_cast<uint8_t>(std::clamp(bl * 255.0, 0.0, 255.0))
    };
}

// Compute single pixel using normalized escape time for ultra-smooth gradients
inline RGB RenderPixel(int px, int py, int width, int height, int maxIter,
                       double xMin, double xMax, double yMin, double yMax) {
    double x0 = xMin + (static_cast<double>(px) / width) * (xMax - xMin);
    double y0 = yMin + (static_cast<double>(py) / height) * (yMax - yMin);

    double x = 0.0;
    double y = 0.0;
    int iter = 0;
    const double ESCAPE_RADIUS_SQ = 256.0;

    while (x * x + y * y <= ESCAPE_RADIUS_SQ && iter < maxIter) {
        double xTemp = x * x - y * y + x0;
        y = 2.0 * x * y + y0;
        x = xTemp;
        iter++;
    }

    if (iter >= maxIter) {
        return RGB{0, 0, 0}; // Points inside Mandelbrot set
    }

    // Renormalized continuous iteration count (prevents color banding)
    double log_zn = std::log(x * x + y * y) / 2.0;
    double nu = std::log(log_zn / std::log(2.0)) / std::log(2.0);
    double iter_smooth = iter + 1.0 - nu;

    double t = std::sqrt(iter_smooth / maxIter) * 3.0; // Color scale multiplier
    return ColorPalette(t);
}

struct MandelbrotRenderParams {
    RGB* imageBuffer;
    int width;
    int height;
    int maxIter;
    double xMin, xMax, yMin, yMax;
};

void RenderMandelbrot(int width, int height, int maxIter,
                      double xMin, double xMax, double yMin, double yMax,
                      const std::string& ppmPath) {
    std::cout << "Rendering Mandelbrot Fractal (" << width << "x" << height
              << ", " << maxIter << " max iterations) via Job System 2.0...\n";

    std::vector<RGB> buffer(width * height);
    DefaultJobSystem js;

    MandelbrotRenderParams params{
        buffer.data(), width, height, maxIter, xMin, xMax, yMin, yMax
    };

    bench::Timer timer;

    // Parallel execution across row ranges using Job System 2.0 parallel_for_index
    parallel_for_index(js, 0, height, [](size_t rowStart, size_t rowCount, const void* ctx) {
        const auto* p = static_cast<const MandelbrotRenderParams*>(ctx);
        for (size_t py = rowStart; py < rowStart + rowCount; ++py) {
            for (int px = 0; px < p->width; ++px) {
                p->imageBuffer[py * p->width + px] = RenderPixel(
                    px, static_cast<int>(py), p->width, p->height, p->maxIter,
                    p->xMin, p->xMax, p->yMin, p->yMax
                );
            }
        }
    }, &params, CountSplitter(16));

    double renderMs = timer.ElapsedMs();
    std::cout << "Render completed in " << renderMs << " ms using "
              << js.GetWorkerCount() << " threads (work-stealing load balanced).\n";

    // Write binary PPM format (P6)
    std::ofstream out(ppmPath, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open output PPM file: " << ppmPath << "\n";
        return;
    }

    out << "P6\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(RGB));
    out.close();

    std::cout << "Successfully saved PPM image to: " << ppmPath << "\n";
}

int main(int argc, char** argv) {
    int width = 1920;
    int height = 1080;
    int maxIter = 1000;
    std::string outputPath = "mandelbrot.ppm";

    if (argc > 1) width = std::stoi(argv[1]);
    if (argc > 2) height = std::stoi(argv[2]);
    if (argc > 3) maxIter = std::stoi(argv[3]);
    if (argc > 4) outputPath = argv[4];

    // Classic full view of Mandelbrot set
    double xMin = -2.0;
    double xMax = 1.0;
    double yMin = -1.125;
    double yMax = 1.125;

    RenderMandelbrot(width, height, maxIter, xMin, xMax, yMin, yMax, outputPath);
    return 0;
}
