#pragma once

#include <cstddef>
#include <cstdint>

namespace job_system {

/**
 * @brief Splits a range based on a fixed element threshold (Part 4 of blog series).
 */
class CountSplitter {
public:
    explicit CountSplitter(size_t threshold = 256)
        : m_threshold(threshold) {}

    template <typename T>
    bool Split(size_t count) const {
        return count > m_threshold;
    }

    size_t GetThreshold() const { return m_threshold; }

private:
    size_t m_threshold;
};

/**
 * @brief Splits a range based on total data working set size in bytes (Part 4 of blog series).
 *
 * Especially effective for keeping subtasks resident in CPU L1 (e.g., 32 KB or 48 KB)
 * or L2 (e.g., 512 KB or 1.25 MB) cache, maximizing cache hits across threads.
 */
class DataSizeSplitter {
public:
    explicit DataSizeSplitter(size_t maxBytes = 32 * 1024)
        : m_maxBytes(maxBytes) {}

    template <typename T>
    bool Split(size_t count) const {
        return (count * sizeof(T)) > m_maxBytes;
    }

    size_t GetMaxBytes() const { return m_maxBytes; }

private:
    size_t m_maxBytes;
};

/**
 * @brief Heuristic splitter that distributes work evenly across available workers.
 */
class AutoSplitter {
public:
    explicit AutoSplitter(unsigned int workerCount, size_t minChunkSize = 64)
        : m_targetChunks(workerCount * 4), m_minChunkSize(minChunkSize) {}

    template <typename T>
    bool Split(size_t count) const {
        return count > m_minChunkSize && (count > 256);
    }

private:
    size_t m_targetChunks;
    size_t m_minChunkSize;
};

} // namespace job_system
