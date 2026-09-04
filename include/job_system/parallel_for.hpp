#pragma once

#include "job_system.hpp"
#include "splitters.hpp"
#include <type_traits>
#include <cstddef>

namespace job_system {

namespace detail {

template <typename T, typename Splitter, typename JS>
struct ParallelForJobData {
    using DataType = T;
    using SplitterType = Splitter;
    using FuncType = void (*)(T*, size_t, const void*);

    T* data;
    size_t count;
    FuncType function;
    const void* userContext;
    Splitter splitter;
    JS* js;
};

template <typename JobData>
void ParallelForJobEntry(Job* job, const void* rawData) {
    const auto* jobData = static_cast<const JobData*>(rawData);
    const auto& splitter = jobData->splitter;
    auto* js = jobData->js;

    if (splitter.template Split<typename JobData::DataType>(jobData->count)) {
        // Divide-and-conquer: split range into two halves
        const size_t leftCount = jobData->count / 2;
        JobData leftData{
            jobData->data,
            leftCount,
            jobData->function,
            jobData->userContext,
            splitter,
            js
        };

        Job* left = js->CreateJobAsChild(job, &ParallelForJobEntry<JobData>, leftData);
        js->Run(left);

        const size_t rightCount = jobData->count - leftCount;
        JobData rightData{
            jobData->data + leftCount,
            rightCount,
            jobData->function,
            jobData->userContext,
            splitter,
            js
        };

        Job* right = js->CreateJobAsChild(job, &ParallelForJobEntry<JobData>, rightData);
        js->Run(right);
    } else {
        // Leaf execution: invoke kernel on the chunk
        jobData->function(jobData->data, jobData->count, jobData->userContext);
    }
}

template <typename Splitter, typename JS>
struct ParallelForIndexData {
    using DataType = char; // Dummy type for sizeof = 1 in splitter
    using SplitterType = Splitter;
    using FuncType = void (*)(size_t, size_t, const void*);

    size_t start;
    size_t count;
    FuncType function;
    const void* userContext;
    Splitter splitter;
    JS* js;
};

template <typename JobData>
void ParallelForIndexEntry(Job* job, const void* rawData) {
    const auto* jobData = static_cast<const JobData*>(rawData);
    const auto& splitter = jobData->splitter;
    auto* js = jobData->js;

    if (splitter.template Split<typename JobData::DataType>(jobData->count)) {
        const size_t leftCount = jobData->count / 2;
        JobData leftData{
            jobData->start,
            leftCount,
            jobData->function,
            jobData->userContext,
            splitter,
            js
        };

        Job* left = js->CreateJobAsChild(job, &ParallelForIndexEntry<JobData>, leftData);
        js->Run(left);

        const size_t rightCount = jobData->count - leftCount;
        JobData rightData{
            jobData->start + leftCount,
            rightCount,
            jobData->function,
            jobData->userContext,
            splitter,
            js
        };

        Job* right = js->CreateJobAsChild(job, &ParallelForIndexEntry<JobData>, rightData);
        js->Run(right);
    } else {
        jobData->function(jobData->start, jobData->count, jobData->userContext);
    }
}

} // namespace detail

/**
 * @brief High-level parallel_for executing a function across an array of data (Part 4).
 *
 * Recursively splits the array into binary subranges using the provided splitter policy.
 * Interleaves splitting with execution across worker threads via lock-free work stealing.
 */
template <typename T, typename Splitter = CountSplitter, typename JS = DefaultJobSystem>
Job* parallel_for_async(JS& js, T* data, size_t count,
                        void (*function)(T*, size_t, const void*),
                        const void* userContext = nullptr,
                        const Splitter& splitter = Splitter{}) {
    using JobData = detail::ParallelForJobData<T, Splitter, JS>;
    static_assert(sizeof(JobData) <= Job::MAX_JOB_DATA_SIZE, "JobData exceeds in-place buffer size");

    JobData jobData{data, count, function, userContext, splitter, &js};
    Job* root = js.CreateJob(&detail::ParallelForJobEntry<JobData>, jobData);
    return root;
}

template <typename T, typename Splitter = CountSplitter, typename JS = DefaultJobSystem>
void parallel_for(JS& js, T* data, size_t count,
                  void (*function)(T*, size_t, const void*),
                  const void* userContext = nullptr,
                  const Splitter& splitter = Splitter{}) {
    if (count == 0) return;
    Job* root = parallel_for_async(js, data, count, function, userContext, splitter);
    js.Run(root);
    js.Wait(root);
}

template <typename T, typename Func, typename Splitter = CountSplitter, typename JS = DefaultJobSystem,
          typename = std::enable_if_t<std::is_convertible_v<Func, void(*)(T*, size_t, const void*)>>>
void parallel_for(JS& js, T* data, size_t count,
                  Func&& function,
                  const void* userContext = nullptr,
                  const Splitter& splitter = Splitter{}) {
    void (*fp)(T*, size_t, const void*) = function;
    parallel_for(js, data, count, fp, userContext, splitter);
}

/**
 * @brief Index-based parallel_for over [start, start + count).
 *
 * Calls func(startChunk, countChunk, userContext) for each leaf range.
 */
template <typename Splitter = CountSplitter, typename JS = DefaultJobSystem>
void parallel_for_index(JS& js, size_t start, size_t count,
                        void (*function)(size_t, size_t, const void*),
                        const void* userContext = nullptr,
                        const Splitter& splitter = Splitter{}) {
    if (count == 0) return;
    using JobData = detail::ParallelForIndexData<Splitter, JS>;
    static_assert(sizeof(JobData) <= Job::MAX_JOB_DATA_SIZE, "JobData exceeds in-place buffer size");

    JobData jobData{start, count, function, userContext, splitter, &js};
    Job* root = js.CreateJob(&detail::ParallelForIndexEntry<JobData>, jobData);
    js.Run(root);
    js.Wait(root);
}

/**
 * @brief Modern C++ lambda parallel_for over an index range [0, count).
 * Kernel receives element index `size_t i`.
 */
template <typename Lambda, typename Splitter = CountSplitter, typename JS = DefaultJobSystem>
void parallel_for_each(JS& js, size_t count, Lambda&& kernel, const Splitter& splitter = Splitter{}) {
    if (count == 0) return;
    using LambdaType = std::decay_t<Lambda>;

    auto chunkHandler = [](size_t chunkStart, size_t chunkCount, const void* ctx) {
        auto& fn = *const_cast<LambdaType*>(reinterpret_cast<const LambdaType*>(ctx));
        for (size_t i = chunkStart; i < chunkStart + chunkCount; ++i) {
            fn(i);
        }
    };

    parallel_for_index(js, 0, count, chunkHandler, reinterpret_cast<const void*>(&kernel), splitter);
}

} // namespace job_system
