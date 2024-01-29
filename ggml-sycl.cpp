/*MIT license
  Copyright (C) 2024 Intel Corporation
  SPDX-License-Identifier: MIT
*/

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <float.h>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>


#include <sycl/sycl.hpp>
#include <sycl/half_type.hpp>

#include "ggml-sycl.h"
#include "ggml.h"
#include "ggml-backend-impl.h"

/*
Following definition copied from DPCT head files, which are used by ggml-sycl.cpp
*/
// COPY from DPCT head files
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <map>

#if defined(__linux__)
#include <sys/mman.h>
#elif defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#error "Only support Windows and Linux."
#endif

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif
#if defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#define DPCT_COMPATIBILITY_TEMP (900)

#if defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#else
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#if defined(_MSC_VER)
#define __dpct_noinline__ __declspec(noinline)
#else
#define __dpct_noinline__ __attribute__((noinline))
#endif

namespace dpct
{
    typedef sycl::queue *queue_ptr;
    typedef sycl::event *event_ptr;
    typedef char *device_ptr;
    typedef uint8_t byte_t;
    typedef sycl::buffer<byte_t> buffer_t;

    /// SYCL default exception handler
    inline auto exception_handler = [](sycl::exception_list exceptions)
    {
        for (std::exception_ptr const &e : exceptions)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::endl;
            }
        }
    };

    enum error_code
    {
        success = 0,
        default_error = 999
    };

    enum memcpy_direction
    {
        host_to_host,
        host_to_device,
        device_to_host,
        device_to_device,
        automatic
    };

    enum memory_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    enum class library_data_t : unsigned char
    {
        real_float = 0,
        complex_float,
        real_double,
        complex_double,
        real_half,
        complex_half,
        real_bfloat16,
        complex_bfloat16,
        real_int4,
        complex_int4,
        real_uint4,
        complex_uint4,
        real_int8,
        complex_int8,
        real_uint8,
        complex_uint8,
        real_int16,
        complex_int16,
        real_uint16,
        complex_uint16,
        real_int32,
        complex_int32,
        real_uint32,
        complex_uint32,
        real_int64,
        complex_int64,
        real_uint64,
        complex_uint64,
        real_int8_4,
        real_int8_32,
        real_uint8_4,
        library_data_t_size
    };

    template <typename T>
    struct DataType
    {
        using T2 = T;
    };
    template <typename T>
    struct DataType<sycl::vec<T, 2>>
    {
        using T2 = std::complex<T>;
    };

    static void destroy_event(event_ptr event)
    {
        delete event;
    }

    static inline unsigned int get_tid()
    {
#if defined(__linux__)
        return syscall(SYS_gettid);
#elif defined(_WIN64)
        return GetCurrentThreadId();
#else
#error "Only support Windows and Linux."
#endif
    }

    namespace detail
    {
        static void get_version(const sycl::device &dev, int &major, int &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            std::string ver;
            ver = dev.get_info<sycl::info::device::version>();
            std::string::size_type i = 0;
            while (i < ver.size())
            {
                if (isdigit(ver[i]))
                    break;
                i++;
            }
            major = std::stoi(&(ver[i]));
            while (i < ver.size())
            {
                if (ver[i] == '.')
                    break;
                i++;
            }
            i++;
            minor = std::stoi(&(ver[i]));
        }

        template <typename tag, typename T>
        class generic_error_type
        {
        public:
            generic_error_type() = default;
            generic_error_type(T value) : value{value} {}
            operator T() const { return value; }

        private:
            T value;
        };

    } // namespace detail

    /// Pitched 2D/3D memory data.
    class pitched_data
    {
    public:
        pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
        pitched_data(void *data, size_t pitch, size_t x, size_t y)
            : _data(data), _pitch(pitch), _x(x), _y(y) {}

        void *get_data_ptr() { return _data; }
        void set_data_ptr(void *data) { _data = data; }

        size_t get_pitch() { return _pitch; }
        void set_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { return _x; }
        void set_x(size_t x) { _x = x; };

        size_t get_y() { return _y; }
        void set_y(size_t y) { _y = y; }

    private:
        void *_data;
        size_t _pitch, _x, _y;
    };

    class device_info
    {
    public:
        // get interface
        const char *get_name() const { return _name; }
        char *get_name() { return _name; }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, int *>,
                                   int> = 0>
        auto get_max_work_item_sizes() const
        {
            if constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                return sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            else
            {
                return _max_work_item_sizes_i;
            }
        }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, int *>,
                                   int> = 0>
        auto get_max_work_item_sizes()
        {
            if constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                return sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            else
            {
                return _max_work_item_sizes_i;
            }
        }
        bool get_host_unified_memory() const { return _host_unified_memory; }
        int get_major_version() const { return _major; }
        int get_minor_version() const { return _minor; }
        int get_integrated() const { return _integrated; }
        int get_max_clock_frequency() const { return _frequency; }
        int get_max_compute_units() const { return _max_compute_units; }
        int get_max_work_group_size() const { return _max_work_group_size; }
        int get_max_sub_group_size() const { return _max_sub_group_size; }
        int get_max_work_items_per_compute_unit() const
        {
            return _max_work_items_per_compute_unit;
        }
        int get_max_register_size_per_work_group() const
        {
            return _max_register_size_per_work_group;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, int *>,
                                   int> = 0>
        auto get_max_nd_range_size() const
        {
            if constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                return _max_nd_range_size;
            else
                return _max_nd_range_size_i;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, int *>,
                                   int> = 0>
        auto get_max_nd_range_size()
        {
            if constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                return _max_nd_range_size;
            else
                return _max_nd_range_size_i;
        }
        size_t get_global_mem_size() const { return _global_mem_size; }
        size_t get_local_mem_size() const { return _local_mem_size; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        unsigned int get_memory_clock_rate() const { return _memory_clock_rate; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        unsigned int get_memory_bus_width() const { return _memory_bus_width; }
        uint32_t get_device_id() const { return _device_id; }
        std::array<unsigned char, 16> get_uuid() const { return _uuid; }
        /// Returns global memory cache size in bytes.
        unsigned int get_global_mem_cache_size() const
        {
            return _global_mem_cache_size;
        }

        // set interface
        void set_name(const char *name)
        {
            size_t length = strlen(name);
            if (length < 256)
            {
                std::memcpy(_name, name, length + 1);
            }
            else
            {
                std::memcpy(_name, name, 255);
                _name[255] = '\0';
            }
        }
        void set_max_work_item_sizes(const sycl::range<3> max_work_item_sizes)
        {
            for (int i = 0; i < 3; ++i)
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
        }
        [[deprecated]] void
        set_max_work_item_sizes(const sycl::id<3> max_work_item_sizes)
        {
            for (int i = 0; i < 3; ++i)
            {
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
            }
        }
        void set_host_unified_memory(bool host_unified_memory)
        {
            _host_unified_memory = host_unified_memory;
        }
        void set_major_version(int major) { _major = major; }
        void set_minor_version(int minor) { _minor = minor; }
        void set_integrated(int integrated) { _integrated = integrated; }
        void set_max_clock_frequency(int frequency) { _frequency = frequency; }
        void set_max_compute_units(int max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        void set_global_mem_size(size_t global_mem_size)
        {
            _global_mem_size = global_mem_size;
        }
        void set_local_mem_size(size_t local_mem_size)
        {
            _local_mem_size = local_mem_size;
        }
        void set_max_work_group_size(int max_work_group_size)
        {
            _max_work_group_size = max_work_group_size;
        }
        void set_max_sub_group_size(int max_sub_group_size)
        {
            _max_sub_group_size = max_sub_group_size;
        }
        void
        set_max_work_items_per_compute_unit(int max_work_items_per_compute_unit)
        {
            _max_work_items_per_compute_unit = max_work_items_per_compute_unit;
        }
        void set_max_nd_range_size(int max_nd_range_size[])
        {
            for (int i = 0; i < 3; i++)
            {
                _max_nd_range_size[i] = max_nd_range_size[i];
                _max_nd_range_size_i[i] = max_nd_range_size[i];
            }
        }
        void set_memory_clock_rate(unsigned int memory_clock_rate)
        {
            _memory_clock_rate = memory_clock_rate;
        }
        void set_memory_bus_width(unsigned int memory_bus_width)
        {
            _memory_bus_width = memory_bus_width;
        }
        void
        set_max_register_size_per_work_group(int max_register_size_per_work_group)
        {
            _max_register_size_per_work_group = max_register_size_per_work_group;
        }
        void set_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        void set_uuid(std::array<unsigned char, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        void set_global_mem_cache_size(unsigned int global_mem_cache_size)
        {
            _global_mem_cache_size = global_mem_cache_size;
        }

    private:
        char _name[256];
        int _max_work_item_sizes_i[3];
        bool _host_unified_memory = false;
        int _major;
        int _minor;
        int _integrated = 0;
        int _frequency;
        // Set estimated value 3200000 kHz as default value.
        unsigned int _memory_clock_rate = 3200000;
        // Set estimated value 64 bits as default value.
        unsigned int _memory_bus_width = 64;
        unsigned int _global_mem_cache_size;
        int _max_compute_units;
        int _max_work_group_size;
        int _max_sub_group_size;
        int _max_work_items_per_compute_unit;
        int _max_register_size_per_work_group;
        size_t _global_mem_size;
        size_t _local_mem_size;
        size_t _max_nd_range_size[3];
        int _max_nd_range_size_i[3];
        uint32_t _device_id;
        std::array<unsigned char, 16> _uuid;
    };

    static int get_major_version(const sycl::device &dev)
    {
        int major, minor;
        detail::get_version(dev, major, minor);
        return major;
    }

    static int get_minor_version(const sycl::device &dev)
    {
        int major, minor;
        detail::get_version(dev, major, minor);
        return minor;
    }

    static void get_device_info(device_info &out, const sycl::device &dev)
    {
        device_info prop;
        prop.set_name(dev.get_info<sycl::info::device::name>().c_str());

        int major, minor;
        detail::get_version(dev, major, minor);
        prop.set_major_version(major);
        prop.set_minor_version(minor);

        prop.set_max_work_item_sizes(
#if (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_item_sizes>());
#else
            // SYCL 2020-conformant code, max_work_item_sizes is a struct templated by
            // an int
            dev.get_info<sycl::info::device::max_work_item_sizes<3>>());
#endif
        prop.set_host_unified_memory(dev.has(sycl::aspect::usm_host_allocations));

        prop.set_max_clock_frequency(
            dev.get_info<sycl::info::device::max_clock_frequency>() * 1000);

        prop.set_max_compute_units(
            dev.get_info<sycl::info::device::max_compute_units>());
        prop.set_max_work_group_size(
            dev.get_info<sycl::info::device::max_work_group_size>());
        prop.set_global_mem_size(dev.get_info<sycl::info::device::global_mem_size>());
        prop.set_local_mem_size(dev.get_info<sycl::info::device::local_mem_size>());

#if (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        if (dev.has(sycl::aspect::ext_intel_memory_clock_rate))
        {
            unsigned int tmp =
                dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
            if (tmp != 0)
                prop.set_memory_clock_rate(1000 * tmp);
        }
        if (dev.has(sycl::aspect::ext_intel_memory_bus_width))
        {
            prop.set_memory_bus_width(
                dev.get_info<sycl::ext::intel::info::device::memory_bus_width>());
        }
        if (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.set_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        if (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.set_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#elif defined(_MSC_VER) && !defined(__clang__)
#pragma message("get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate default value. \
        Use 64 bits as memory_bus_width default value.")
#else
#warning "get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate default value. \
        Use 64 bits as memory_bus_width default value."
#endif

        size_t max_sub_group_size = 1;
        std::vector<size_t> sub_group_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();

        for (const auto &sub_group_size : sub_group_sizes)
        {
            if (max_sub_group_size < sub_group_size)
                max_sub_group_size = sub_group_size;
        }

        prop.set_max_sub_group_size(max_sub_group_size);

        prop.set_max_work_items_per_compute_unit(
            dev.get_info<sycl::info::device::max_work_group_size>());
        int max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.set_max_nd_range_size(max_nd_range_size);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.set_max_register_size_per_work_group(65536);

        prop.set_global_mem_cache_size(
            dev.get_info<sycl::info::device::global_mem_cache_size>());
        out = prop;
    }

    /// dpct device extension
    class device_ext : public sycl::device
    {
        typedef std::mutex mutex_type;

    public:
        device_ext() : sycl::device(), _ctx(*this) {}
        ~device_ext()
        {
            std::lock_guard<mutex_type> lock(m_mutex);
            clear_queues();
        }
        device_ext(const sycl::device &base) : sycl::device(base), _ctx(*this)
        {
            std::lock_guard<mutex_type> lock(m_mutex);
            init_queues();
        }

        int is_native_atomic_supported() { return 0; }
        int get_major_version() const
        {
            return dpct::get_major_version(*this);
        }

        int get_minor_version() const
        {
            return dpct::get_minor_version(*this);
        }

        int get_max_compute_units() const
        {
            return get_device_info().get_max_compute_units();
        }

        /// Return the maximum clock frequency of this device in KHz.
        int get_max_clock_frequency() const
        {
            return get_device_info().get_max_clock_frequency();
        }

        int get_integrated() const { return get_device_info().get_integrated(); }

        int get_max_sub_group_size() const
        {
            return get_device_info().get_max_sub_group_size();
        }

        int get_max_register_size_per_work_group() const
        {
            return get_device_info().get_max_register_size_per_work_group();
        }

        int get_max_work_group_size() const
        {
            return get_device_info().get_max_work_group_size();
        }

        int get_mem_base_addr_align() const
        {
            return get_info<sycl::info::device::mem_base_addr_align>();
        }

        size_t get_global_mem_size() const
        {
            return get_device_info().get_global_mem_size();
        }

        /// Get the number of bytes of free and total memory on the SYCL device.
        /// \param [out] free_memory The number of bytes of free memory on the SYCL device.
        /// \param [out] total_memory The number of bytes of total memory on the SYCL device.
        void get_memory_info(size_t &free_memory, size_t &total_memory)
        {
#if (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
            if (!has(sycl::aspect::ext_intel_free_memory))
            {
                std::cerr << "get_memory_info: ext_intel_free_memory is not supported." << std::endl;
                free_memory = 0;
            }
            else
            {
                free_memory = get_info<sycl::ext::intel::info::device::free_memory>();
            }
#else
            std::cerr << "get_memory_info: ext_intel_free_memory is not supported." << std::endl;
            free_memory = 0;
#if defined(_MSC_VER) && !defined(__clang__)
#pragma message("Querying the number of bytes of free memory is not supported")
#else
#warning "Querying the number of bytes of free memory is not supported"
#endif
#endif
            total_memory = get_device_info().get_global_mem_size();
        }

        void get_device_info(device_info &out) const
        {
            dpct::get_device_info(out, *this);
        }

        device_info get_device_info() const
        {
            device_info prop;
            dpct::get_device_info(prop, *this);
            return prop;
        }

        void reset()
        {
            std::lock_guard<mutex_type> lock(m_mutex);
            clear_queues();
            init_queues();
        }

        sycl::queue &in_order_queue() { return *_q_in_order; }

        sycl::queue &out_of_order_queue() { return *_q_out_of_order; }

        sycl::queue &default_queue()
        {
#ifdef DPCT_USM_LEVEL_NONE
            return out_of_order_queue();
#else
            return in_order_queue();
#endif // DPCT_USM_LEVEL_NONE
        }

        void queues_wait_and_throw()
        {
            std::unique_lock<mutex_type> lock(m_mutex);
            std::vector<std::shared_ptr<sycl::queue>> current_queues(
                _queues);
            lock.unlock();
            for (const auto &q : current_queues)
            {
                q->wait_and_throw();
            }
            // Guard the destruct of current_queues to make sure the ref count is safe.
            lock.lock();
        }

        sycl::queue *create_queue(bool enable_exception_handler = false)
        {
#ifdef DPCT_USM_LEVEL_NONE
            return create_out_of_order_queue(enable_exception_handler);
#else
            return create_in_order_queue(enable_exception_handler);
#endif // DPCT_USM_LEVEL_NONE
        }

        sycl::queue *create_in_order_queue(bool enable_exception_handler = false)
        {
            std::lock_guard<mutex_type> lock(m_mutex);
            return create_queue_impl(enable_exception_handler,
                                     sycl::property::queue::in_order());
        }

        sycl::queue *create_out_of_order_queue(bool enable_exception_handler = false)
        {
            std::lock_guard<mutex_type> lock(m_mutex);
            return create_queue_impl(enable_exception_handler);
        }

        void destroy_queue(sycl::queue *&queue)
        {
            std::lock_guard<mutex_type> lock(m_mutex);
            _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                         [=](const std::shared_ptr<sycl::queue> &q) -> bool
                                         {
                                             return q.get() == queue;
                                         }),
                          _queues.end());
            queue = nullptr;
        }
        void set_saved_queue(sycl::queue *q)
        {
            std::lock_guard<mutex_type> lock(m_mutex);
            _saved_queue = q;
        }
        sycl::queue *get_saved_queue() const
        {
            std::lock_guard<mutex_type> lock(m_mutex);
            return _saved_queue;
        }
        sycl::context get_context() const { return _ctx; }

    private:
        void clear_queues()
        {
            _queues.clear();
            _q_in_order = _q_out_of_order = _saved_queue = nullptr;
        }

        void init_queues()
        {
            _q_in_order = create_queue_impl(true, sycl::property::queue::in_order());
            _q_out_of_order = create_queue_impl(true);
            _saved_queue = &default_queue();
        }

        /// Caller should acquire resource \p m_mutex before calling this function.
        template <class... Properties>
        sycl::queue *create_queue_impl(bool enable_exception_handler,
                                       Properties... properties)
        {
            sycl::async_handler eh = {};
            if (enable_exception_handler)
            {
                eh = exception_handler;
            }
            _queues.push_back(std::make_shared<sycl::queue>(
                _ctx, *this, eh,
                sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                    sycl::property::queue::enable_profiling(),
#endif
                    properties...)));

            return _queues.back().get();
        }

        void get_version(int &major, int &minor) const
        {
            detail::get_version(*this, major, minor);
        }
        sycl::queue *_q_in_order, *_q_out_of_order;
        sycl::queue *_saved_queue;
        sycl::context _ctx;
        std::vector<std::shared_ptr<sycl::queue>> _queues;
        mutable mutex_type m_mutex;
    };

    /// device manager
    class dev_mgr
    {
    public:
        device_ext &current_device()
        {
            unsigned int dev_id = current_device_id();
            check_id(dev_id);
            return *_devs[dev_id];
        }
        device_ext &cpu_device() const
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            if (_cpu_device == -1)
            {
                throw std::runtime_error("no valid cpu device");
            }
            else
            {
                return *_devs[_cpu_device];
            }
        }
        device_ext &get_device(unsigned int id) const
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            return *_devs[id];
        }
        unsigned int current_device_id() const
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            auto it = _thread2dev_map.find(get_tid());
            if (it != _thread2dev_map.end())
                return it->second;
            return DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        void select_device(unsigned int id)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        unsigned int device_count() { return _devs.size(); }

        unsigned int get_device_id(const sycl::device &dev)
        {
            unsigned int id = 0;
            for (auto dev_item : _devs)
            {
                if (*dev_item == dev)
                {
                    break;
                }
                id++;
            }
            return id;
        }

        template <class DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<int, DeviceSelector, const sycl::device &>>
        select_device(const DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            unsigned int selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        static dev_mgr &instance()
        {
            static dev_mgr d_m;
            return d_m;
        }
        dev_mgr(const dev_mgr &) = delete;
        dev_mgr &operator=(const dev_mgr &) = delete;
        dev_mgr(dev_mgr &&) = delete;
        dev_mgr &operator=(dev_mgr &&) = delete;

    private:
        mutable std::recursive_mutex m_mutex;
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs =
                sycl::device::get_devices(sycl::info::device_type::all);
            // Collect other devices except for the default device.
            if (default_device.is_cpu())
                _cpu_device = 0;
            for (auto &dev : sycl_all_devs)
            {
                if (dev == default_device)
                {
                    continue;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                if (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        void check_id(unsigned int id) const
        {
            if (id >= _devs.size())
            {
                throw std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        const unsigned int DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<unsigned int, unsigned int> _thread2dev_map;
        int _cpu_device = -1;
    };

    static inline sycl::queue &get_default_queue()
    {
        return dev_mgr::instance().current_device().default_queue();
    }

    namespace detail
    {
        enum class pointer_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        static pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                              const void *ptr)
        {
#ifdef DPCT_USM_LEVEL_NONE
            return mem_mgr::instance().is_device_ptr(ptr)
                       ? pointer_access_attribute::device_only
                       : pointer_access_attribute::host_only;
#else
            switch (sycl::get_pointer_type(ptr, q.get_context()))
            {
            case sycl::usm::alloc::unknown:
                return pointer_access_attribute::host_only;
            case sycl::usm::alloc::device:
                return pointer_access_attribute::device_only;
            case sycl::usm::alloc::shared:
            case sycl::usm::alloc::host:
                return pointer_access_attribute::host_device;
            }
#endif
        }

        template <typename ArgT>
        inline constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((unsigned char)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned char>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_data_t>, "Unsupported ArgT");
            return (std::uint64_t)Val;
        }

        template <typename FirstT, typename... RestT>
        inline constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned char>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(sizeof...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<FirstT, library_data_t>, "Unsupported FirstT");
            return get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
        }

        class mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#if defined(__linux__)
                mapped_address_space =
                    (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#elif defined(_WIN64)
                mapped_address_space = (byte_t *)VirtualAlloc(
                    NULL,               // NULL specified as the base address parameter
                    mapped_region_size, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    PAGE_NOACCESS);     // Protection = no access
#else
#error "Only support Windows and Linux."
#endif
                next_free = mapped_address_space;
            };

        public:
            using buffer_id_t = int;

            struct allocation
            {
                buffer_t buffer;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#if defined(__linux__)
                munmap(mapped_address_space, mapped_region_size);
#elif defined(_WIN64)
                VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#else
#error "Only support Windows and Linux."
#endif
            };

            mem_mgr(const mem_mgr &) = delete;
            mem_mgr &operator=(const mem_mgr &) = delete;
            mem_mgr(mem_mgr &&) = delete;
            mem_mgr &operator=(mem_mgr &&) = delete;

            /// Allocate
            void *mem_alloc(size_t size)
            {
                if (!size)
                    return nullptr;
                std::lock_guard<std::mutex> lock(m_mutex);
                if (next_free + size > mapped_address_space + mapped_region_size)
                {
                    throw std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                // Allocation
                sycl::range<1> r(size);
                buffer_t buf(r);
                allocation A{buf, next_free, size};
                // Map allocation to device pointer
                void *result = next_free;
                m_map.emplace(next_free + size, A);
                // Update pointer to the next free space.
                next_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                return result;
            }

            /// Deallocate
            void mem_free(const void *ptr)
            {
                if (!ptr)
                    return;
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            allocation translate_ptr(const void *ptr)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                return it->second;
            }

            /// Check if the pointer represents device pointer or not.
            bool is_device_ptr(const void *ptr) const
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                return (mapped_address_space <= ptr) &&
                       (ptr < mapped_address_space + mapped_region_size);
            }

            /// Returns the instance of memory manager singleton.
            static mem_mgr &instance()
            {
                static mem_mgr m;
                return m;
            }

        private:
            std::map<byte_t *, allocation> m_map;
            mutable std::mutex m_mutex;
            byte_t *mapped_address_space;
            byte_t *next_free;
            const size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
            const size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            const size_t extra_padding = 0;

            std::map<byte_t *, allocation>::iterator get_map_iterator(const void *ptr)
            {
                auto it = m_map.upper_bound((byte_t *)ptr);
                if (it == m_map.end())
                {
                    // Not a virtual pointer.
                    throw std::runtime_error("can not get buffer from non-virtual pointer");
                }
                const allocation &alloc = it->second;
                if (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    throw std::runtime_error("invalid virtual pointer");
                }
                return it;
            }
        };

        template <class T, memory_region Memory, size_t Dimension>
        class accessor;
        template <memory_region Memory, class T = byte_t>
        class memory_traits
        {
        public:
            static constexpr sycl::access::target target =
                sycl::access::target::device;
            static constexpr sycl::access_mode mode =
                (Memory == constant) ? sycl::access_mode::read
                                     : sycl::access_mode::read_write;
            static constexpr size_t type_size = sizeof(T);
            using element_t =
                typename std::conditional<Memory == constant, const T, T>::type;
            using value_t = typename std::remove_cv<T>::type;
            template <size_t Dimension = 1>
            using accessor_t = typename std::conditional<
                Memory == local, sycl::local_accessor<value_t, Dimension>,
                sycl::accessor<T, Dimension, mode, target>>::type;
            using pointer_t = T *;
        };

        static inline void *dpct_malloc(size_t size, sycl::queue &q)
        {
#ifdef DPCT_USM_LEVEL_NONE
            return mem_mgr::instance().mem_alloc(size * sizeof(byte_t));
#else
            return sycl::malloc_device(size, q.get_device(), q.get_context());
#endif // DPCT_USM_LEVEL_NONE
        }

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        static inline void *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            return dpct_malloc(pitch * y * z, q);
        }

        /**
         * @brief Sets \p value to the first \p size elements starting from \p dev_ptr in \p q.
         * @tparam valueT The type of the element to be set.
         * @param [in] q The queue in which the operation is done.
         * @param [in] dev_ptr Pointer to the virtual device memory address.
         * @param [in] value The value to be set.
         * @param [in] size Number of elements to be set to the value.
         * @return An event representing the memset operation.
         */
        template <typename valueT>
        static inline sycl::event dpct_memset(sycl::queue &q, void *dev_ptr,
                                              valueT value, size_t size)
        {
#ifdef DPCT_USM_LEVEL_NONE
            auto &mm = mem_mgr::instance();
            assert(mm.is_device_ptr(dev_ptr));
            auto alloc = mm.translate_ptr(dev_ptr);
            size_t offset = (valueT *)dev_ptr - (valueT *)alloc.alloc_ptr;

            return q.submit([&](sycl::handler &cgh)
                            {
    auto r = sycl::range<1>(size);
    auto o = sycl::id<1>(offset);
    auto new_buffer = alloc.buffer.reinterpret<valueT>(
        sycl::range<1>(alloc.size / sizeof(valueT)));
    sycl::accessor<valueT, 1, sycl::access_mode::write,
                sycl::access::target::device>
        acc(new_buffer, cgh, r, o);
    cgh.fill(acc, value); });
#else
            return q.fill(dev_ptr, value, size);
#endif // DPCT_USM_LEVEL_NONE
        }

        /**
         * @brief Sets \p value to the 3D memory region pointed by \p data in \p q.
         * @tparam valueT The type of the element to be set.
         * @param [in] q The queue in which the operation is done.
         * @param [in] data Pointer to the pitched device memory region.
         * @param [in] value The value to be set.
         * @param [in] size 3D memory region by number of elements.
         * @return An event list representing the memset operations.
         */
        template <typename valueT>
        static inline std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_data data, valueT value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t slice = data.get_pitch() * data.get_y();
            unsigned char *data_surface = (unsigned char *)data.get_data_ptr();
            for (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned char *data_ptr = data_surface;
                for (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, data_ptr, value, size.get(0)));
                    data_ptr += data.get_pitch();
                }
                data_surface += slice;
            }
            return event_list;
        }

        /**
         * @brief Sets \p val to the pitched 2D memory region pointed by \p ptr in \p q.
         * @tparam valueT The type of the element to be set.
         * @param [in] q The queue in which the operation is done.
         * @param [in] ptr Pointer to the virtual device memory.
         * @param [in] pitch The pitch size by number of elements, including padding.
         * @param [in] val The value to be set.
         * @param [in] x The width of memory region by number of elements.
         * @param [in] y The height of memory region by number of elements.
         * @return An event list representing the memset operations.
         */
        template <typename valueT>
        static inline std::vector<sycl::event>
        dpct_memset(sycl::queue &q, void *ptr, size_t pitch, valueT val, size_t x,
                    size_t y)
        {
            return dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        static memcpy_direction deduce_memcpy_direction(sycl::queue &q, void *to_ptr,
                                                        const void *from_ptr,
                                                        memcpy_direction dir)
        {
            switch (dir)
            {
            case memcpy_direction::host_to_host:
            case memcpy_direction::host_to_device:
            case memcpy_direction::device_to_host:
            case memcpy_direction::device_to_device:
                return dir;
            case memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                static const memcpy_direction
                    direction_table[static_cast<unsigned>(pointer_access_attribute::end)]
                                   [static_cast<unsigned>(pointer_access_attribute::end)] =
                                       {{memcpy_direction::host_to_host,
                                         memcpy_direction::device_to_host,
                                         memcpy_direction::host_to_host},
                                        {memcpy_direction::host_to_device,
                                         memcpy_direction::device_to_device,
                                         memcpy_direction::device_to_device},
                                        {memcpy_direction::host_to_host,
                                         memcpy_direction::device_to_device,
                                         memcpy_direction::device_to_device}};
                return direction_table[static_cast<unsigned>(get_pointer_attribute(
                    q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
            }
            default:
                throw std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        static sycl::event
        dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t size,
                    memcpy_direction direction,
                    const std::vector<sycl::event> &dep_events = {})
        {
            if (!size)
                return sycl::event{};
#ifdef DPCT_USM_LEVEL_NONE
            auto &mm = mem_mgr::instance();
            auto real_direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);

            switch (real_direction)
            {
            case host_to_host:
                return q.submit([&](sycl::handler &cgh)
                                {
    cgh.depends_on(dep_events);
    cgh.host_task([=] { std::memcpy(to_ptr, from_ptr, size); }); });
            case host_to_device:
            {
                auto alloc = mm.translate_ptr(to_ptr);
                size_t offset = (byte_t *)to_ptr - alloc.alloc_ptr;
                return q.submit([&](sycl::handler &cgh)
                                {
    cgh.depends_on(dep_events);
    auto r = sycl::range<1>(size);
    auto o = sycl::id<1>(offset);
    sycl::accessor<byte_t, 1, sycl::access_mode::write,
                        sycl::access::target::device>
        acc(alloc.buffer, cgh, r, o);
    cgh.copy(from_ptr, acc); });
            }
            case device_to_host:
            {
                auto alloc = mm.translate_ptr(from_ptr);
                size_t offset = (byte_t *)from_ptr - alloc.alloc_ptr;
                return q.submit([&](sycl::handler &cgh)
                                {
    cgh.depends_on(dep_events);
    auto r = sycl::range<1>(size);
    auto o = sycl::id<1>(offset);
    sycl::accessor<byte_t, 1, sycl::access_mode::read,
                        sycl::access::target::device>
        acc(alloc.buffer, cgh, r, o);
    cgh.copy(acc, to_ptr); });
            }
            case device_to_device:
            {
                auto to_alloc = mm.translate_ptr(to_ptr);
                auto from_alloc = mm.translate_ptr(from_ptr);
                size_t to_offset = (byte_t *)to_ptr - to_alloc.alloc_ptr;
                size_t from_offset = (byte_t *)from_ptr - from_alloc.alloc_ptr;
                return q.submit([&](sycl::handler &cgh)
                                {
    cgh.depends_on(dep_events);
    auto r = sycl::range<1>(size);
    auto to_o = sycl::id<1>(to_offset);
    auto from_o = sycl::id<1>(from_offset);
    sycl::accessor<byte_t, 1, sycl::access_mode::write,
                        sycl::access::target::device>
        to_acc(to_alloc.buffer, cgh, r, to_o);
    sycl::accessor<byte_t, 1, sycl::access_mode::read,
                        sycl::access::target::device>
        from_acc(from_alloc.buffer, cgh, r, from_o);
    cgh.copy(from_acc, to_acc); });
            }
            default:
                throw std::runtime_error("dpct_memcpy: invalid direction value");
            }
#else
            return q.memcpy(to_ptr, from_ptr, size, dep_events);
#endif // DPCT_USM_LEVEL_NONE
        }

        // Get actual copy range and make sure it will not exceed range.
        static inline size_t get_copy_range(sycl::range<3> size, size_t slice,
                                            size_t pitch)
        {
            return slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        static inline size_t get_offset(sycl::id<3> id, size_t slice,
                                        size_t pitch)
        {
            return slice * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        static inline std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
                    sycl::range<3> to_range, sycl::range<3> from_range,
                    sycl::id<3> to_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    const std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            class host_buffer
            {
                void *_buf;
                size_t _size;
                sycl::queue &_q;
                const std::vector<sycl::event> &_deps; // free operation depends

            public:
                host_buffer(size_t size, sycl::queue &q,
                            const std::vector<sycl::event> &deps)
                    : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
                void *get_ptr() const { return _buf; }
                size_t get_size() const { return _size; }
                ~host_buffer()
                {
                    if (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::free(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t to_slice = to_range.get(1) * to_range.get(0),
                   from_slice = from_range.get(1) * from_range.get(0);
            unsigned char *to_surface =
                (unsigned char *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
            const unsigned char *from_surface =
                (const unsigned char *)from_ptr +
                get_offset(from_id, from_slice, from_range.get(0));

            if (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
            {
                return {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
            size_t size_slice = size.get(1) * size.get(0);
            switch (direction)
            {
            case host_to_host:
                for (size_t z = 0; z < size.get(2); ++z)
                {
                    unsigned char *to_ptr = to_surface;
                    const unsigned char *from_ptr = from_surface;
                    if (to_range.get(0) == from_range.get(0) &&
                        to_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                         direction, dep_events));
                    }
                    else
                    {
                        for (size_t y = 0; y < size.get(1); ++y)
                        {
                            event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size.get(0),
                                                             direction, dep_events));
                            to_ptr += to_range.get(0);
                            from_ptr += from_range.get(0);
                        }
                    }
                    to_surface += to_slice;
                    from_surface += from_slice;
                }
                break;
            case host_to_device:
            {
                host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                if (to_slice == size_slice)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_to_host, dep_events);
                }
                else
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events = dpct_memcpy(
                        q, buf.get_ptr(), from_surface, to_range, from_range,
                        sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
                        // If has padding data, not sure whether it is useless. So fill temp
                        // buffer with it.
                        std::vector<sycl::event>{
                            dpct_memcpy(q, buf.get_ptr(), to_surface, buf.get_size(),
                                        device_to_host, dep_events)});
                }
                // Copy from temp host buffer to device with only one submit.
                event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(),
                                                 buf.get_size(), host_to_device,
                                                 host_events));
                break;
            }
            case device_to_host:
            {
                host_buffer buf(get_copy_range(size, from_slice, from_range.get(0)), q,
                                event_list);
                // Copy from host temp buffer to host target with reshaping.
                event_list = dpct_memcpy(
                    q, to_surface, buf.get_ptr(), to_range, from_range, sycl::id<3>(0, 0, 0),
                    sycl::id<3>(0, 0, 0), size, host_to_host,
                    // Copy from device to temp host buffer with only one submit.
                    std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                         buf.get_size(),
                                                         device_to_host, dep_events)});
                break;
            }
            case device_to_device:
#ifdef DPCT_USM_LEVEL_NONE
            {
                auto &mm = mem_mgr::instance();
                auto to_alloc = mm.translate_ptr(to_surface);
                auto from_alloc = mm.translate_ptr(from_surface);
                size_t to_offset = (byte_t *)to_surface - to_alloc.alloc_ptr;
                size_t from_offset = (byte_t *)from_surface - from_alloc.alloc_ptr;
                event_list.push_back(q.submit([&](sycl::handler &cgh)
                                              {
    cgh.depends_on(dep_events);
    auto to_o = sycl::id<1>(to_offset);
    auto from_o = sycl::id<1>(from_offset);
    sycl::accessor<byte_t, 1, sycl::access_mode::write,
                        sycl::access::target::device>
        to_acc(to_alloc.buffer, cgh,
                get_copy_range(size, to_slice, to_range.get(0)), to_o);
    sycl::accessor<byte_t, 1, sycl::access_mode::read,
                        sycl::access::target::device>
        from_acc(from_alloc.buffer, cgh,
                get_copy_range(size, from_slice, from_range.get(0)), from_o);
    cgh.parallel_for<class dpct_memcpy_3d_detail_usmnone>(
        size,
        [=](sycl::id<3> id) {
            to_acc[get_offset(id, to_slice, to_range.get(0))] =
                from_acc[get_offset(id, from_slice, from_range.get(0))];
        }); }));
            }
#else
                event_list.push_back(q.submit([&](sycl::handler &cgh)
                                              {
    cgh.depends_on(dep_events);
    cgh.parallel_for<class dpct_memcpy_3d_detail>(
        size,
        [=](sycl::id<3> id) {
            to_surface[get_offset(id, to_slice, to_range.get(0))] =
                from_surface[get_offset(id, from_slice, from_range.get(0))];
        }); }));
#endif
            break;
            default:
                throw std::runtime_error("dpct_memcpy: invalid direction value");
            }
            return event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        static inline std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                    pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            return dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                               sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        static inline std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
                    size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            return dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        namespace deprecated
        {

            template <typename T, sycl::usm::alloc AllocKind>
            class usm_allocator
            {
            private:
                using Alloc = sycl::usm_allocator<T, AllocKind>;
                Alloc _impl;

            public:
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
                using const_void_pointer =
                    typename std::allocator_traits<Alloc>::const_void_pointer;
                using reference = typename std::allocator_traits<Alloc>::value_type &;
                using const_reference =
                    const typename std::allocator_traits<Alloc>::value_type &;
                using difference_type =
                    typename std::allocator_traits<Alloc>::difference_type;
                using size_type = typename std::allocator_traits<Alloc>::size_type;
                using propagate_on_container_copy_assignment = typename std::allocator_traits<
                    Alloc>::propagate_on_container_copy_assignment;
                using propagate_on_container_move_assignment = typename std::allocator_traits<
                    Alloc>::propagate_on_container_move_assignment;
                using propagate_on_container_swap =
                    typename std::allocator_traits<Alloc>::propagate_on_container_swap;
                using is_always_equal =
                    typename std::allocator_traits<Alloc>::is_always_equal;

                template <typename U>
                struct rebind
                {
                    typedef usm_allocator<U, AllocKind> other;
                };

                usm_allocator() : _impl(dpct::get_default_queue()) {}
                ~usm_allocator() {}
                usm_allocator(const usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { return &r; }
                const_pointer address(const_reference r) { return &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = nullptr)
                {
                    return std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                void deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() const
                {
                    return std::allocator_traits<Alloc>::max_size(_impl);
                }
                bool operator==(const usm_allocator &other) const { return _impl == other._impl; }
                bool operator!=(const usm_allocator &other) const { return _impl != other._impl; }
            };

        } // namespace deprecated

        inline void dpct_free(void *ptr,
                              const sycl::queue &q)
        {
            if (ptr)
            {
#ifdef DPCT_USM_LEVEL_NONE
                detail::mem_mgr::instance().mem_free(ptr);
#else
                sycl::free(ptr, q.get_context());
#endif // DPCT_USM_LEVEL_NONE
            }
        }

        template <typename T>
        inline auto get_memory(const void *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<void *>(x));
#ifdef DPCT_USM_LEVEL_NONE
            return dpct::get_buffer<std::remove_cv_t<T>>(new_x);
#else
            return new_x;
#endif
        }

        template <typename T>
        inline typename DataType<T>::T2 get_value(const T *s, sycl::queue &q)
        {
            using Ty = typename DataType<T>::T2;
            Ty s_h;
            if (get_pointer_attribute(q, s) == pointer_access_attribute::device_only)
                detail::dpct_memcpy(q, (void *)&s_h, (void *)s, sizeof(T), device_to_host)
                    .wait();
            else
                s_h = *reinterpret_cast<const Ty *>(s);
            return s_h;
        }

    } // namespace detail

    template <typename T>
    inline auto get_value(const T *s, sycl::queue &q)
    {
        return detail::get_value(s, q);
    }

    namespace detail
    {
        template <class Ta, class Tb, class Tc, class Ts>
        inline void gemm_impl(sycl::queue &q, oneapi::mkl::transpose a_trans,
                              oneapi::mkl::transpose b_trans, int m, int n, int k,
                              const void *alpha, const void *a, int lda, const void *b,
                              int ldb, const void *beta, void *c, int ldc)
        {
#ifndef __INTEL_MKL__
            throw std::runtime_error("The oneAPI Math Kernel Library (oneMKL) Interfaces "
                                     "Project does not support this API.");
#else
            Ts alpha_value = dpct::get_value(reinterpret_cast<const Ts *>(alpha), q);
            Ts beta_value = dpct::get_value(reinterpret_cast<const Ts *>(beta), q);
            auto data_a = get_memory<const Ta>(a);
            auto data_b = get_memory<const Tb>(b);
            auto data_c = get_memory<Tc>(c);
            oneapi::mkl::blas::column_major::gemm(
                q, a_trans, b_trans, m, n, k, alpha_value, data_a, lda,
                data_b, ldb, beta_value, data_c, ldc);
#endif
        }

        template <typename VecT, class BinaryOperation, class = void>
        class vectorized_binary
        {
        public:
            inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op)
            {
                VecT v4;
                for (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = binary_op(a[i], b[i]);
                }
                return v4;
            }
        };

        template <typename VecT, class BinaryOperation>
        class vectorized_binary<
            VecT, BinaryOperation,
            std::void_t<std::invoke_result_t<BinaryOperation, VecT, VecT>>>
        {
        public:
            inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op)
            {
                return binary_op(a, b).template as<VecT>();
            }
        };

        template <class Ta, class Tb, class Tc, class Ts>
        inline void gemm_batch_impl(sycl::queue &q, oneapi::mkl::transpose a_trans,
                                    oneapi::mkl::transpose b_trans, int m, int n, int k,
                                    const void *alpha, const void **a, int lda,
                                    const void **b, int ldb, const void *beta, void **c,
                                    int ldc, int batch_size)
        {
            struct matrix_info_t
            {
                oneapi::mkl::transpose transpose_info[2];
                Ts value_info[2];
                std::int64_t size_info[3];
                std::int64_t ld_info[3];
                std::int64_t groupsize_info;
            };

            Ts alpha_value = dpct::get_value(reinterpret_cast<const Ts *>(alpha), q);
            Ts beta_value = dpct::get_value(reinterpret_cast<const Ts *>(beta), q);

            matrix_info_t *matrix_info =
                (matrix_info_t *)std::malloc(sizeof(matrix_info_t));
            matrix_info->transpose_info[0] = a_trans;
            matrix_info->transpose_info[1] = b_trans;
            matrix_info->value_info[0] = alpha_value;
            matrix_info->value_info[1] = beta_value;
            matrix_info->size_info[0] = m;
            matrix_info->size_info[1] = n;
            matrix_info->size_info[2] = k;
            matrix_info->ld_info[0] = lda;
            matrix_info->ld_info[1] = ldb;
            matrix_info->ld_info[2] = ldc;
            matrix_info->groupsize_info = batch_size;

            sycl::event e = oneapi::mkl::blas::column_major::gemm_batch(
                q, matrix_info->transpose_info, matrix_info->transpose_info + 1,
                matrix_info->size_info, matrix_info->size_info + 1,
                matrix_info->size_info + 2, matrix_info->value_info,
                reinterpret_cast<const Ta **>(a), matrix_info->ld_info,
                reinterpret_cast<const Tb **>(b), matrix_info->ld_info + 1,
                matrix_info->value_info + 1, reinterpret_cast<Tc **>(c),
                matrix_info->ld_info + 2, 1, &(matrix_info->groupsize_info));

            q.submit([&](sycl::handler &cgh)
                     {
    cgh.depends_on(e);
    cgh.host_task([=] { std::free(matrix_info); }); });
        }

        template <class Ta, class Tb, class Tc, class Ts>
        inline void
        gemm_batch_impl(sycl::queue &q, oneapi::mkl::transpose a_trans,
                        oneapi::mkl::transpose b_trans, int m, int n,
                        int k, const void *alpha, const void *a, int lda,
                        long long int stride_a, const void *b, int ldb,
                        long long int stride_b, const void *beta, void *c,
                        int ldc, long long int stride_c, int batch_size)
        {
            Ts alpha_value = dpct::get_value(reinterpret_cast<const Ts *>(alpha), q);
            Ts beta_value = dpct::get_value(reinterpret_cast<const Ts *>(beta), q);
            auto data_a = get_memory<const Ta>(a);
            auto data_b = get_memory<const Tb>(b);
            auto data_c = get_memory<Tc>(c);
            oneapi::mkl::blas::column_major::gemm_batch(
                q, a_trans, b_trans, m, n, k, alpha_value, data_a, lda,
                stride_a, data_b, ldb, stride_b, beta_value,
                data_c, ldc, stride_c, batch_size);
        }

    } // namespace detail

    template <typename VecT, class BinaryOperation>
    inline unsigned vectorized_binary(unsigned a, unsigned b,
                                      const BinaryOperation binary_op)
    {
        sycl::vec<unsigned, 1> v0{a}, v1{b};
        auto v2 = v0.as<VecT>();
        auto v3 = v1.as<VecT>();
        auto v4 =
            detail::vectorized_binary<VecT, BinaryOperation>()(v2, v3, binary_op);
        v0 = v4.template as<sycl::vec<unsigned, 1>>();
        return v0;
    }

    static void async_dpct_memcpy(void *to_ptr, const void *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
    }

    static inline unsigned int select_device(unsigned int id)
    {
        dev_mgr::instance().select_device(id);
        return id;
    }

    template <typename T>
    T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned int mask,
                               int logical_sub_group_size = 32)
    {
        unsigned int id = g.get_local_linear_id();
        unsigned int start_index =
            id / logical_sub_group_size * logical_sub_group_size;
        unsigned int target_offset = (id % logical_sub_group_size) ^ mask;
        return sycl::select_from_group(g, x,
                                       target_offset < logical_sub_group_size
                                           ? start_index + target_offset
                                           : id);
    }

    template <typename T>
    sycl::vec<T, 4> extract_and_sign_or_zero_extend4(T val)
    {
        return sycl::vec<T, 1>(val)
            .template as<sycl::vec<
                std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>, 4>>()
            .template convert<T>();
    }

    template <typename T1, typename T2>
    using dot_product_acc_t =
        std::conditional_t<std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
                           uint32_t, int32_t>;

    template <typename T1, typename T2, typename T3>
    inline auto dp4a(T1 a, T2 b, T3 c)
    {
        dot_product_acc_t<T1, T2> res = c;
        auto va = extract_and_sign_or_zero_extend4(a);
        auto vb = extract_and_sign_or_zero_extend4(b);
        res += va[0] * vb[0];
        res += va[1] * vb[1];
        res += va[2] * vb[2];
        res += va[3] * vb[3];
        return res;
    }

    struct sub_sat
    {
        template <typename T>
        auto operator()(const T x, const T y) const
        {
            return sycl::sub_sat(x, y);
        }
    };

    template <typename S, typename T>
    inline T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        auto v2 = v0.template as<S>();
        auto v3 = v1.template as<S>();
        auto v4 = sycl::min(v2, v3);
        v0 = v4.template as<sycl::vec<T, 1>>();
        return v0;
    }

    inline float pow(const float a, const int b) { return sycl::pown(a, b); }
    inline double pow(const double a, const int b) { return sycl::pown(a, b); }
    inline float pow(const float a, const float b) { return sycl::pow(a, b); }
    inline double pow(const double a, const double b) { return sycl::pow(a, b); }
    template <typename T, typename U>
    inline typename std::enable_if_t<std::is_floating_point_v<T>, T>
    pow(const T a, const U b)
    {
        return sycl::pow(a, static_cast<T>(b));
    }
    template <typename T, typename U>
    inline typename std::enable_if_t<!std::is_floating_point_v<T>, double>
    pow(const T a, const U b)
    {
        return sycl::pow(static_cast<double>(a), static_cast<double>(b));
    }

    inline double min(const double a, const float b)
    {
        return sycl::fmin(a, static_cast<double>(b));
    }
    inline double min(const float a, const double b)
    {
        return sycl::fmin(static_cast<double>(a), b);
    }
    inline float min(const float a, const float b) { return sycl::fmin(a, b); }
    inline double min(const double a, const double b) { return sycl::fmin(a, b); }
    inline std::uint32_t min(const std::uint32_t a, const std::int32_t b)
    {
        return sycl::min(a, static_cast<std::uint32_t>(b));
    }
    inline std::uint32_t min(const std::int32_t a, const std::uint32_t b)
    {
        return sycl::min(static_cast<std::uint32_t>(a), b);
    }
    inline std::int32_t min(const std::int32_t a, const std::int32_t b)
    {
        return sycl::min(a, b);
    }
    inline std::uint32_t min(const std::uint32_t a, const std::uint32_t b)
    {
        return sycl::min(a, b);
    }
    inline std::uint64_t min(const std::uint64_t a, const std::int64_t b)
    {
        return sycl::min(a, static_cast<std::uint64_t>(b));
    }
    inline std::uint64_t min(const std::int64_t a, const std::uint64_t b)
    {
        return sycl::min(static_cast<std::uint64_t>(a), b);
    }
    inline std::int64_t min(const std::int64_t a, const std::int64_t b)
    {
        return sycl::min(a, b);
    }
    inline std::uint64_t min(const std::uint64_t a, const std::uint64_t b)
    {
        return sycl::min(a, b);
    }
    inline std::uint64_t min(const std::uint64_t a, const std::int32_t b)
    {
        return sycl::min(a, static_cast<std::uint64_t>(b));
    }
    inline std::uint64_t min(const std::int32_t a, const std::uint64_t b)
    {
        return sycl::min(static_cast<std::uint64_t>(a), b);
    }
    inline std::uint64_t min(const std::uint64_t a, const std::uint32_t b)
    {
        return sycl::min(a, static_cast<std::uint64_t>(b));
    }
    inline std::uint64_t min(const std::uint32_t a, const std::uint64_t b)
    {
        return sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    inline double max(const double a, const float b)
    {
        return sycl::fmax(a, static_cast<double>(b));
    }
    inline double max(const float a, const double b)
    {
        return sycl::fmax(static_cast<double>(a), b);
    }
    inline float max(const float a, const float b) { return sycl::fmax(a, b); }
    inline double max(const double a, const double b) { return sycl::fmax(a, b); }
    inline std::uint32_t max(const std::uint32_t a, const std::int32_t b)
    {
        return sycl::max(a, static_cast<std::uint32_t>(b));
    }
    inline std::uint32_t max(const std::int32_t a, const std::uint32_t b)
    {
        return sycl::max(static_cast<std::uint32_t>(a), b);
    }
    inline std::int32_t max(const std::int32_t a, const std::int32_t b)
    {
        return sycl::max(a, b);
    }
    inline std::uint32_t max(const std::uint32_t a, const std::uint32_t b)
    {
        return sycl::max(a, b);
    }
    inline std::uint64_t max(const std::uint64_t a, const std::int64_t b)
    {
        return sycl::max(a, static_cast<std::uint64_t>(b));
    }
    inline std::uint64_t max(const std::int64_t a, const std::uint64_t b)
    {
        return sycl::max(static_cast<std::uint64_t>(a), b);
    }
    inline std::int64_t max(const std::int64_t a, const std::int64_t b)
    {
        return sycl::max(a, b);
    }
    inline std::uint64_t max(const std::uint64_t a, const std::uint64_t b)
    {
        return sycl::max(a, b);
    }
    inline std::uint64_t max(const std::uint64_t a, const std::int32_t b)
    {
        return sycl::max(a, static_cast<std::uint64_t>(b));
    }
    inline std::uint64_t max(const std::int32_t a, const std::uint64_t b)
    {
        return sycl::max(static_cast<std::uint64_t>(a), b);
    }
    inline std::uint64_t max(const std::uint64_t a, const std::uint32_t b)
    {
        return sycl::max(a, static_cast<std::uint64_t>(b));
    }
    inline std::uint64_t max(const std::uint32_t a, const std::uint64_t b)
    {
        return sycl::max(static_cast<std::uint64_t>(a), b);
    }

    inline void
    has_capability_or_fail(const sycl::device &dev,
                           const std::initializer_list<sycl::aspect> &props)
    {
        for (const auto &it : props)
        {
            if (dev.has(it))
                continue;
            switch (it)
            {
            case sycl::aspect::fp64:
                throw std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                break;
            case sycl::aspect::fp16:
                throw std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                break;
            default:
#define __SYCL_ASPECT(ASPECT, ID) \
    case sycl::aspect::ASPECT:    \
        return #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
                auto getAspectNameStr = [](sycl::aspect AspectNum) -> std::string
                {
                    switch (AspectNum)
                    {
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
                    default:
                        return "unknown aspect";
                    }
                };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
                throw std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            break;
        }
    }

    static inline unsigned int get_current_device_id()
    {
        return dev_mgr::instance().current_device_id();
    }

    static inline device_ext &get_current_device()
    {
        return dev_mgr::instance().current_device();
    }

    static inline sycl::queue &get_in_order_queue()
    {
        return dev_mgr::instance().current_device().in_order_queue();
    }

    static sycl::event
    dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t size,
                memcpy_direction direction,
                const std::vector<sycl::event> &dep_events = {})
    {
        if (!size)
            return sycl::event{};
#ifdef DPCT_USM_LEVEL_NONE
        auto &mm = mem_mgr::instance();
        auto real_direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);

        switch (real_direction)
        {
        case host_to_host:
            return q.submit([&](sycl::handler &cgh)
                            {
        cgh.depends_on(dep_events);
        cgh.host_task([=] { std::memcpy(to_ptr, from_ptr, size); }); });
        case host_to_device:
        {
            auto alloc = mm.translate_ptr(to_ptr);
            size_t offset = (byte_t *)to_ptr - alloc.alloc_ptr;
            return q.submit([&](sycl::handler &cgh)
                            {
        cgh.depends_on(dep_events);
        auto r = sycl::range<1>(size);
        auto o = sycl::id<1>(offset);
        sycl::accessor<byte_t, 1, sycl::access_mode::write,
                            sycl::access::target::device>
            acc(alloc.buffer, cgh, r, o);
        cgh.copy(from_ptr, acc); });
        }
        case device_to_host:
        {
            auto alloc = mm.translate_ptr(from_ptr);
            size_t offset = (byte_t *)from_ptr - alloc.alloc_ptr;
            return q.submit([&](sycl::handler &cgh)
                            {
        cgh.depends_on(dep_events);
        auto r = sycl::range<1>(size);
        auto o = sycl::id<1>(offset);
        sycl::accessor<byte_t, 1, sycl::access_mode::read,
                            sycl::access::target::device>
            acc(alloc.buffer, cgh, r, o);
        cgh.copy(acc, to_ptr); });
        }
        case device_to_device:
        {
            auto to_alloc = mm.translate_ptr(to_ptr);
            auto from_alloc = mm.translate_ptr(from_ptr);
            size_t to_offset = (byte_t *)to_ptr - to_alloc.alloc_ptr;
            size_t from_offset = (byte_t *)from_ptr - from_alloc.alloc_ptr;
            return q.submit([&](sycl::handler &cgh)
                            {
        cgh.depends_on(dep_events);
        auto r = sycl::range<1>(size);
        auto to_o = sycl::id<1>(to_offset);
        auto from_o = sycl::id<1>(from_offset);
        sycl::accessor<byte_t, 1, sycl::access_mode::write,
                            sycl::access::target::device>
            to_acc(to_alloc.buffer, cgh, r, to_o);
        sycl::accessor<byte_t, 1, sycl::access_mode::read,
                            sycl::access::target::device>
            from_acc(from_alloc.buffer, cgh, r, from_o);
        cgh.copy(from_acc, to_acc); });
        }
        default:
            throw std::runtime_error("dpct_memcpy: invalid direction value");
        }
#else
        return q.memcpy(to_ptr, from_ptr, size, dep_events);
#endif // DPCT_USM_LEVEL_NONE
    }

    // Get actual copy range and make sure it will not exceed range.
    static inline size_t get_copy_range(sycl::range<3> size, size_t slice,
                                        size_t pitch)
    {
        return slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    static inline size_t get_offset(sycl::id<3> id, size_t slice,
                                    size_t pitch)
    {
        return slice * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    static inline std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
                sycl::range<3> to_range, sycl::range<3> from_range,
                sycl::id<3> to_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                const std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        class host_buffer
        {
            void *_buf;
            size_t _size;
            sycl::queue &_q;
            const std::vector<sycl::event> &_deps; // free operation depends

        public:
            host_buffer(size_t size, sycl::queue &q,
                        const std::vector<sycl::event> &deps)
                : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
            void *get_ptr() const { return _buf; }
            size_t get_size() const { return _size; }
            ~host_buffer()
            {
                if (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::free(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t to_slice = to_range.get(1) * to_range.get(0),
               from_slice = from_range.get(1) * from_range.get(0);
        unsigned char *to_surface =
            (unsigned char *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
        const unsigned char *from_surface =
            (const unsigned char *)from_ptr +
            get_offset(from_id, from_slice, from_range.get(0));

        if (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
        {
            return {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
        size_t size_slice = size.get(1) * size.get(0);
        switch (direction)
        {
        case host_to_host:
            for (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned char *to_ptr = to_surface;
                const unsigned char *from_ptr = from_surface;
                if (to_range.get(0) == from_range.get(0) &&
                    to_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                     direction, dep_events));
                }
                else
                {
                    for (size_t y = 0; y < size.get(1); ++y)
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size.get(0),
                                                         direction, dep_events));
                        to_ptr += to_range.get(0);
                        from_ptr += from_range.get(0);
                    }
                }
                to_surface += to_slice;
                from_surface += from_slice;
            }
            break;
        case host_to_device:
        {
            host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            if (to_slice == size_slice)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_to_host, dep_events);
            }
            else
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events = dpct_memcpy(
                    q, buf.get_ptr(), from_surface, to_range, from_range,
                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
                    // If has padding data, not sure whether it is useless. So fill temp
                    // buffer with it.
                    std::vector<sycl::event>{
                        dpct_memcpy(q, buf.get_ptr(), to_surface, buf.get_size(),
                                    device_to_host, dep_events)});
            }
            // Copy from temp host buffer to device with only one submit.
            event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(),
                                             buf.get_size(), host_to_device,
                                             host_events));
            break;
        }
        case device_to_host:
        {
            host_buffer buf(get_copy_range(size, from_slice, from_range.get(0)), q,
                            event_list);
            // Copy from host temp buffer to host target with reshaping.
            event_list = dpct_memcpy(
                q, to_surface, buf.get_ptr(), to_range, from_range, sycl::id<3>(0, 0, 0),
                sycl::id<3>(0, 0, 0), size, host_to_host,
                // Copy from device to temp host buffer with only one submit.
                std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                     buf.get_size(),
                                                     device_to_host, dep_events)});
            break;
        }
        case device_to_device:
#ifdef DPCT_USM_LEVEL_NONE
        {
            auto &mm = mem_mgr::instance();
            auto to_alloc = mm.translate_ptr(to_surface);
            auto from_alloc = mm.translate_ptr(from_surface);
            size_t to_offset = (byte_t *)to_surface - to_alloc.alloc_ptr;
            size_t from_offset = (byte_t *)from_surface - from_alloc.alloc_ptr;
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        auto to_o = sycl::id<1>(to_offset);
        auto from_o = sycl::id<1>(from_offset);
        sycl::accessor<byte_t, 1, sycl::access_mode::write,
                            sycl::access::target::device>
            to_acc(to_alloc.buffer, cgh,
                    get_copy_range(size, to_slice, to_range.get(0)), to_o);
        sycl::accessor<byte_t, 1, sycl::access_mode::read,
                            sycl::access::target::device>
            from_acc(from_alloc.buffer, cgh,
                    get_copy_range(size, from_slice, from_range.get(0)), from_o);
        cgh.parallel_for<class dpct_memcpy_3d_detail_usmnone>(
            size,
            [=](sycl::id<3> id) {
                to_acc[get_offset(id, to_slice, to_range.get(0))] =
                    from_acc[get_offset(id, from_slice, from_range.get(0))];
            }); }));
        }
#else
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<class dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                to_surface[get_offset(id, to_slice, to_range.get(0))] =
                    from_surface[get_offset(id, from_slice, from_range.get(0))];
            }); }));
#endif
        break;
        default:
            throw std::runtime_error("dpct_memcpy: invalid direction value");
        }
        return event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    static inline std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        return dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                           sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    static inline std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
                size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        return dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    inline void gemm(sycl::queue &q, oneapi::mkl::transpose a_trans,
                     oneapi::mkl::transpose b_trans, int m, int n, int k,
                     const void *alpha, const void *a, library_data_t a_type,
                     int lda, const void *b, library_data_t b_type, int ldb,
                     const void *beta, void *c, library_data_t c_type, int ldc,
                     library_data_t scaling_type)
    {
        if (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        else if (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t key =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        switch (key)
        {
        case detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<float, float, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_impl<double, double, double, double>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_impl<std::complex<float>, std::complex<float>,
                              std::complex<float>, std::complex<float>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_impl<std::complex<double>, std::complex<double>,
                              std::complex<double>, std::complex<double>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float,
                              float>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b,
                                     ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            float alpha_value =
                dpct::get_value(reinterpret_cast<const float *>(alpha), q);
            float beta_value =
                dpct::get_value(reinterpret_cast<const float *>(beta), q);
            sycl::half alpha_half(alpha_value);
            sycl::half beta_half(beta_value);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                              oneapi::mkl::bfloat16, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            float alpha_float =
                dpct::get_value(reinterpret_cast<const std::int32_t *>(alpha), q);
            float beta_float =
                dpct::get_value(reinterpret_cast<const std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, float>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            break;
        }
        default:
            throw std::runtime_error("the combination of data type is unsupported");
        }
    } // gemm()

    /// Computes a batch of matrix-matrix product with general matrices.
    /// \param [in] q The queue where the routine should be executed.
    /// \param [in] a_trans Specifies the operation applied to A.
    /// \param [in] b_trans Specifies the operation applied to B.
    /// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
    /// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
    /// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
    /// \param [in] alpha Scaling factor for the matrix-matrix product.
    /// \param [in] a Input matrix A.
    /// \param [in] a_type Data type of the matrix A.
    /// \param [in] lda Leading dimension of A.
    /// \param [in] b Input matrix B.
    /// \param [in] b_type Data type of the matrix B.
    /// \param [in] ldb Leading dimension of B.
    /// \param [in] beta Scaling factor for matrix C.
    /// \param [in, out] c Input/Output matrix C.
    /// \param [in] c_type Data type of the matrix C.
    /// \param [in] ldc Leading dimension of C.
    /// \param [in] batch_size Specifies the number of matrix multiply operations to perform.
    /// \param [in] scaling_type Data type of the scaling factors.
    inline void gemm_batch(sycl::queue &q, oneapi::mkl::transpose a_trans,
                           oneapi::mkl::transpose b_trans, int m, int n, int k,
                           const void *alpha, const void *a[],
                           library_data_t a_type, int lda, const void *b[],
                           library_data_t b_type, int ldb, const void *beta,
                           void *c[], library_data_t c_type, int ldc,
                           int batch_size, library_data_t scaling_type)
    {
#ifdef DPCT_USM_LEVEL_NONE
        throw std::runtime_error("this API is unsupported when USM level is none");
#else
        if (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        else if (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t key =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        switch (key)
        {
        case detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<float, float, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<double, double, double, double>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<float>, std::complex<float>,
                                    std::complex<float>, std::complex<float>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<double>, std::complex<double>,
                                    std::complex<double>, std::complex<double>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, b, ldb, beta, c, ldc,
                                                batch_size);
            break;
        }
#ifdef __INTEL_MKL__
        case detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                                    oneapi::mkl::bfloat16, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float,
                                    float>(q, a_trans, b_trans, m, n, k, alpha, a, lda,
                                           b, ldb, beta, c, ldc, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            float alpha_float =
                dpct::get_value(reinterpret_cast<const std::int32_t *>(alpha), q);
            float beta_float =
                dpct::get_value(reinterpret_cast<const std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    float>(q, a_trans, b_trans, m, n, k, &alpha_float,
                                           a, lda, b, ldb, &beta_float, c, ldc,
                                           batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                batch_size);
            break;
        }
#endif
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            float alpha_value =
                dpct::get_value(reinterpret_cast<const float *>(alpha), q);
            float beta_value =
                dpct::get_value(reinterpret_cast<const float *>(beta), q);
            sycl::half alpha_half(alpha_value);
            sycl::half beta_half(beta_value);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc,
                batch_size);
            break;
        }
        default:
            throw std::runtime_error("the combination of data type is unsupported");
        }
#endif
    }

    /// Computes a batch of matrix-matrix product with general matrices.
    /// \param [in] q The queue where the routine should be executed.
    /// \param [in] a_trans Specifies the operation applied to A.
    /// \param [in] b_trans Specifies the operation applied to B.
    /// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
    /// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
    /// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
    /// \param [in] alpha Scaling factor for the matrix-matrix product.
    /// \param [in] a Input matrix A.
    /// \param [in] a_type Data type of the matrix A.
    /// \param [in] lda Leading dimension of A.
    /// \param [in] stride_a Stride between the different A matrices.
    /// \param [in] b Input matrix B.
    /// \param [in] b_type Data type of the matrix B.
    /// \param [in] ldb Leading dimension of B.
    /// \param [in] stride_b Stride between the different B matrices.
    /// \param [in] beta Scaling factor for matrix C.
    /// \param [in, out] c Input/Output matrix C.
    /// \param [in] c_type Data type of the matrix C.
    /// \param [in] ldc Leading dimension of C.
    /// \param [in] stride_c Stride between the different C matrices.
    /// \param [in] batch_size Specifies the number of matrix multiply operations to perform.
    /// \param [in] scaling_type Data type of the scaling factors.
    inline void gemm_batch(sycl::queue &q, oneapi::mkl::transpose a_trans,
                           oneapi::mkl::transpose b_trans, int m, int n, int k,
                           const void *alpha, const void *a, library_data_t a_type,
                           int lda, long long int stride_a, const void *b,
                           library_data_t b_type, int ldb, long long int stride_b,
                           const void *beta, void *c, library_data_t c_type,
                           int ldc, long long int stride_c, int batch_size,
                           library_data_t scaling_type)
    {
        if (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        else if (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t key =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        switch (key)
        {
        case detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<float, float, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<double, double, double, double>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<float>, std::complex<float>,
                                    std::complex<float>, std::complex<float>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<double>, std::complex<double>,
                                    std::complex<double>, std::complex<double>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_size);
            break;
        }
#ifdef __INTEL_MKL__
        case detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16,
                                    oneapi::mkl::bfloat16, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float,
                                    float>(q, a_trans, b_trans, m, n, k, alpha, a, lda,
                                           stride_a, b, ldb, stride_b, beta, c, ldc,
                                           stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
#endif
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            float alpha_value =
                dpct::get_value(reinterpret_cast<const float *>(alpha), q);
            float beta_value =
                dpct::get_value(reinterpret_cast<const float *>(beta), q);
            sycl::half alpha_half(alpha_value);
            sycl::half beta_half(beta_value);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_size);
            break;
        }
        default:
            throw std::runtime_error("the combination of data type is unsupported");
        }
    }

    static inline void
    async_dpct_memcpy(void *to_ptr, size_t to_pitch, const void *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                            direction);
    }

    using err0 = detail::generic_error_type<struct err0_tag, int>;
    using err1 = detail::generic_error_type<struct err1_tag, int>;

} // COPY from DPCT head files


static int g_ggml_sycl_debug=0;
#define GGML_SYCL_DEBUG(...) do{if(g_ggml_sycl_debug) printf(__VA_ARGS__);}while(0)

#define CHECK_TRY_ERROR(expr)                                                  \
  [&]() {                                                                      \
    try {                                                                      \
      expr;                                                                    \
      return dpct::success;                                                    \
    } catch (std::exception const &e) {                                        \
      std::cerr << e.what()<< "\nException caught at file:" << __FILE__        \
        << ", line:" << __LINE__ <<", func:"<<__func__<< std::endl;            \
      return dpct::default_error;                                              \
    }                                                                          \
  }()

// #define DEBUG_SYCL_MALLOC

static int g_work_group_size = 0;
// typedef sycl::half ggml_fp16_t;

#define __SYCL_ARCH__ DPCT_COMPATIBILITY_TEMP
#define VER_4VEC   610          //todo for hardward optimize.
#define VER_GEN9      700       //todo for hardward optimize.
#define VER_GEN12 1000000       //todo for hardward optimize.
#define VER_GEN13      (VER_GEN12 + 1030)   //todo for hardward optimize.

#define GGML_SYCL_MAX_NODES 8192 //TODO: adapt to hardwares


//define for XMX in Intel GPU
//TODO: currently, it's not used for XMX really.
#define SYCL_USE_XMX

// max batch size to use MMQ kernels when tensor cores are available
#define XMX_MAX_BATCH_SIZE 32


#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static_assert(sizeof(sycl::half) == sizeof(ggml_fp16_t), "wrong fp16 size");

static void crash(){
    int *ptr = NULL;
    *ptr = 0;
}

static void ggml_sycl_error(const char * stmt, const char * func, const char * file, const int line, const char * msg) {
    fprintf(stderr, "SYCL error: %s: %s\n", stmt, msg);
    fprintf(stderr, "  in function %s at %s:%d\n", func, file, line);
    GGML_ASSERT(!"SYCL error");
}

#define SYCL_CHECK(err) do {                                                   \
    auto err_ = (err); if (err_ != 0) ggml_sycl_error(                         \
        #err, __func__, __FILE__, __LINE__,                                    \
        "Meet error in this line code!");   \
} while (0)

#if DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#else
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
typedef sycl::half dfloat; // dequantize float
typedef sycl::half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef sycl::float2 dfloat2;
#endif //GGML_SYCL_F16

bool   ggml_sycl_loaded(void);
void * ggml_sycl_host_malloc(size_t size);
void   ggml_sycl_host_free(void * ptr);
bool   ggml_sycl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_sycl_set_tensor_split(const float * tensor_split);
void   ggml_sycl_transform_tensor(void * data, struct ggml_tensor * tensor);
void   ggml_sycl_free_data(struct ggml_tensor * tensor);
void   ggml_sycl_assign_buffers(struct ggml_tensor * tensor);
void   ggml_sycl_assign_buffers_no_scratch(struct ggml_tensor * tensor);
void   ggml_sycl_assign_buffers_force_inplace(struct ggml_tensor * tensor);
void   ggml_sycl_assign_buffers_no_alloc(struct ggml_tensor * tensor);
void   ggml_sycl_assign_scratch_offset(struct ggml_tensor * tensor, size_t offset);
void   ggml_sycl_copy_to_device(struct ggml_tensor * tensor);
void   ggml_sycl_set_main_device(int main_device);
void   ggml_sycl_set_mul_mat_q(bool mul_mat_q);
void   ggml_sycl_set_scratch_size(size_t scratch_size);
void   ggml_sycl_free_scratch(void);
int    ggml_sycl_get_device_count(void);
void   ggml_sycl_get_device_description(int device, char * description, size_t description_size);
bool   ggml_backend_is_sycl(ggml_backend_t backend);
int    ggml_backend_sycl_get_device(ggml_backend_t backend);
int    get_main_device();
void   print_ggml_tensor(const char*name, struct ggml_tensor *src);
void   log_tensor_with_cnt(const char* name, struct ggml_tensor * src, int stop_cnt);

static __dpct_inline__ int get_int_from_int8(const int8_t *x8, const int &i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;

    return x32;
}

static __dpct_inline__ int get_int_from_uint8(const uint8_t *x8,
                                              const int &i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;

    return x32;
}

static __dpct_inline__ int get_int_from_int8_aligned(const int8_t *x8,
                                                     const int &i32) {
    return *((const int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

static __dpct_inline__ int get_int_from_uint8_aligned(const uint8_t *x8,
                                                      const int &i32) {
    return *((const int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

template <typename T>
using to_t_sycl_t = void (*)(const void *__restrict__ x, T *__restrict__ y,
                             int k, dpct::queue_ptr stream);
typedef to_t_sycl_t<float> to_fp32_sycl_t;
typedef to_t_sycl_t<sycl::half> to_fp16_sycl_t;

typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, dfloat2 & v);
typedef void (*dot_kernel_k_t)(const void * __restrict__ vx, const int ib, const int iqs, const float * __restrict__ y, float & v);
typedef void (*cpy_kernel_t)(const char * cx, char * cdst);
typedef void (*ggml_sycl_func_t)(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
typedef void (*ggml_sycl_op_mul_mat_t)(
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);
typedef void (*ggml_sycl_op_flatten_t)(const ggml_tensor *src0,
                                       const ggml_tensor *src1,
                                       ggml_tensor *dst, const float *src0_dd,
                                       const float *src1_dd, float *dst_dd,
                                       const dpct::queue_ptr &main_stream);

// QK = number of values after dequantization
// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
typedef struct dpct_type_471834 {
    sycl::half d;           // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
typedef struct dpct_type_143705 {
    sycl::half2 dm;         // dm.x = delta, dm.y = min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(ggml_fp16_t) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
typedef struct dpct_type_673649 {
    sycl::half d;           // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
typedef struct dpct_type_135589 {
    sycl::half2 dm;         // dm.x = delta, dm.y = min
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
typedef struct dpct_type_122878 {
    sycl::half d;           // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
typedef struct dpct_type_143721 {
    sycl::half2 ds;         // ds.x = delta, ds.y = sum
    int8_t  qs[QK8_0];      // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_fp16_t) + QK8_0, "wrong q8_1 block size/padding");

typedef float (*vec_dot_q_sycl_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs);
typedef void (*allocate_tiles_sycl_t)(int **x_ql, sycl::half2 **x_dm,
                                      int **x_qh, int **x_sc);
typedef void (*load_tiles_sycl_t)(const void *__restrict__ vx,
                                  int *__restrict__ x_ql,
                                  sycl::half2 *__restrict__ x_dm,
                                  int *__restrict__ x_qh,
                                  int *__restrict__ x_sc, const int &i_offset,
                                  const int &i_max, const int &k,
                                  const int &blocks_per_row);
typedef float (*vec_dot_q_mul_mat_sycl_t)(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ms,
    const int &i, const int &j, const int &k);

//================================= k-quants

#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif

#define QR2_K 4
#define QI2_K (QK_K / (4*QR2_K))
typedef struct dpct_type_619598 {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    sycl::half2 dm;          // super-block scale for quantized scales/mins
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_fp16_t) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

#define QR3_K 4
#define QI3_K (QK_K / (4*QR3_K))
typedef struct dpct_type_138576 {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
#ifdef GGML_QKK_64
    uint8_t scales[2]; // scales, quantized with 8 bits
#else
    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits
#endif
    sycl::half d; // super-block scale
} block_q3_K;
//static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + K_SCALE_SIZE, "wrong q3_K block size/padding");

#define QR4_K 2
#define QI4_K (QK_K / (4*QR4_K))
#ifdef GGML_QKK_64
typedef struct {
    half    dm[2];             // super-block scales/mins
    uint8_t scales[2];         // 4-bit block scales/mins
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == sizeof(half2) + QK_K/2 + 2, "wrong q4_K block size/padding");
#else
typedef struct dpct_type_154943 {
    sycl::half2 dm;            // super-block scale for quantized scales/mins
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + 3*QK_K/64 + QK_K/2, "wrong q4_K block size/padding");
#endif

#define QR5_K 2
#define QI5_K (QK_K / (4*QR5_K))
#ifdef GGML_QKK_64
typedef struct {
    half d;                  // super-block scale
    int8_t scales[QK_K/16];  // block scales
    uint8_t qh[QK_K/8];      // quants, high bit
    uint8_t qs[QK_K/2];      // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == sizeof(ggml_fp16_t) + QK_K/2 + QK_K/8 + QK_K/16, "wrong q5_K block size/padding");
#else
typedef struct dpct_type_866817 {
    sycl::half2 dm;               // super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");
#endif

#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))
typedef struct dpct_type_107281 {
    uint8_t ql[QK_K/2];   // quants, lower 4 bits
    uint8_t qh[QK_K/4];   // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales
    sycl::half d;            // delta
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_fp16_t) + 13*QK_K/16, "wrong q6_K block size/padding");

#define WARP_SIZE 32
#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#define SYCL_GELU_BLOCK_SIZE 256
#define SYCL_SILU_BLOCK_SIZE 256
#define SYCL_TANH_BLOCK_SIZE 256
#define SYCL_RELU_BLOCK_SIZE 256
#define SYCL_SQR_BLOCK_SIZE 256
#define SYCL_CPY_BLOCK_SIZE 32
#define SYCL_SCALE_BLOCK_SIZE 256
#define SYCL_CLAMP_BLOCK_SIZE 256
#define SYCL_ROPE_BLOCK_SIZE 256
#define SYCL_SOFT_MAX_BLOCK_SIZE 1024
#define SYCL_ALIBI_BLOCK_SIZE 32
#define SYCL_DIAG_MASK_INF_BLOCK_SIZE 32
#define SYCL_QUANTIZE_BLOCK_SIZE 256
#define SYCL_DEQUANTIZE_BLOCK_SIZE 256
#define SYCL_GET_ROWS_BLOCK_SIZE 256
#define SYCL_UPSCALE_BLOCK_SIZE 256
#define SYCL_CONCAT_BLOCK_SIZE 256
#define SYCL_PAD_BLOCK_SIZE 256
#define SYCL_ACC_BLOCK_SIZE 256
#define SYCL_IM2COL_BLOCK_SIZE 256

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_SYCL_DMMV_X
#define GGML_SYCL_DMMV_X 32
#endif
#ifndef GGML_SYCL_MMV_Y
#define GGML_SYCL_MMV_Y 1
#endif

#ifndef K_QUANTS_PER_ITERATION
#define K_QUANTS_PER_ITERATION 2
#else
static_assert(K_QUANTS_PER_ITERATION == 1 || K_QUANTS_PER_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#endif

#ifndef GGML_SYCL_PEER_MAX_BATCH_SIZE
#define GGML_SYCL_PEER_MAX_BATCH_SIZE 128
#endif // GGML_SYCL_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

#define MAX_STREAMS 8
static dpct::queue_ptr g_syclStreams[GGML_SYCL_MAX_DEVICES][MAX_STREAMS] = {
    {0}};

struct ggml_tensor_extra_gpu {
    void * data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split tensors
    dpct::event_ptr
        events[GGML_SYCL_MAX_DEVICES]
              [MAX_STREAMS]; // events for synchronizing multiple GPUs
};

inline dpct::err0 ggml_sycl_set_device(const int device) try {
    int current_device;

    SYCL_CHECK(CHECK_TRY_ERROR(
        current_device = dpct::dev_mgr::instance().current_device_id()));

    // GGML_SYCL_DEBUG("ggml_sycl_set_device device=%d, current_device=%d\n", device, current_device);
    if (device == current_device) {
        return 0;
    }

    return CHECK_TRY_ERROR(dpct::select_device(device));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

static int g_device_count = -1;
static int g_all_sycl_device_count = -1;
static int g_main_device = -1;
static int g_main_device_index = -1;

static float g_tensor_split[GGML_SYCL_MAX_DEVICES] = {0};

struct sycl_device_capabilities {
    int     cc;                 // compute capability
    bool    vmm;                // virtual memory support
    size_t  vmm_granularity;    // granularity of virtual memory
    int device_id;
};

static sycl_device_capabilities g_device_caps[GGML_SYCL_MAX_DEVICES] = { {0, false, 0, -1} };

struct sycl_device_id2index {
    int index;
};

static sycl_device_id2index g_sycl_device_id2index[GGML_SYCL_MAX_DEVICES] = { {-1} };

static void * g_scratch_buffer = nullptr;
static size_t g_scratch_size = 0; // disabled by default
static size_t g_scratch_offset = 0;

static dpct::queue_ptr g_sycl_handles[GGML_SYCL_MAX_DEVICES] = {nullptr};

int get_main_device(){
    return g_main_device;
}

[[noreturn]]
static void bad_arch(const sycl::stream &stream_ct1) {
    stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                  "current GPU architecture.\n";
    // __trap();
    std::exit(1);

    (void) bad_arch; // suppress unused function warning
}

void log_ggml_var_device(const char*name, float *src, size_t total_elements, bool src_on_device){
    if(!g_ggml_sycl_debug) return;
    if(!src){
        printf("GGML Tensor:%s skip to save for NULL pointer\n", name);
        return;
    }
    char filename[1024];
    sprintf(filename, "%s.txt", name);
    printf("GGML Tensor:%s save to %s\n", name, filename);

    size_t total_size = total_elements*sizeof(float);
    float *local_buf = NULL;
    // printf("total_size %d2, src_on_device %d\n", total_size, src_on_device);
    if(src_on_device) {
        local_buf = (float *) ggml_sycl_host_malloc(total_size);
        // printf("local buf %p size %d bytes\n", local_buf, total_size);
        ggml_sycl_set_device(g_main_device);
        dpct::queue_ptr main_stream = g_syclStreams[g_main_device_index][0];
        main_stream->memcpy(local_buf, src, total_size);
    }
    else {
        local_buf = (float *)src;
        // printf("local buf from src-> data %p\n", local_buf);
    }

    std::ofstream logfile;
    logfile.open(filename);
    // printf("local buf element %d\n", total_elements);
    for(int i=0; i<total_elements; i++){
        if((i+1)%20 ==0) logfile <<std::endl;
        else logfile << local_buf[i] <<" ";
    }
    logfile <<std::endl;
    logfile.close();

    if(src_on_device) ggml_sycl_host_free(local_buf);
}

//todo: debug for crash in some case
void print_ggml_tensor(const char*name, struct ggml_tensor *src){
    if(!g_ggml_sycl_debug) return;
    if(!src){
        printf("GGML Tensor:%s skip to save for NULL pointer\n", name);
        return;
    }

    size_t total_elements = ggml_nelements(src);

    const bool src_on_device = src->backend == GGML_BACKEND_GPU || src->backend == GGML_BACKEND_GPU_SPLIT;
    float *src_data =NULL;
    if(src_on_device) {
        ggml_tensor_extra_gpu * src_extra = (ggml_tensor_extra_gpu *)  src->extra;
        src_data = (float*)src_extra->data_device[g_main_device_index];
    }
    else {
        src_data = (float *)src->data;
    }

    log_ggml_var_device(name, src_data, total_elements, src_on_device);
}

static int log_file_name_idx=0;
void log_tensor_with_cnt(const char* name, struct ggml_tensor * src, int stop_cnt) {
    stop_cnt = 4;
    if(log_file_name_idx>=stop_cnt) return;
    char filename[1280];
    sprintf(filename, "%s_%07d", name, log_file_name_idx);
    log_file_name_idx++;
    print_ggml_tensor(filename, src);
    // print_ggml_tensor("ggml_sycl_rms_norm_src0", (ggml_tensor *)src0);
    // print_ggml_tensor("ggml_sycl_rms_norm_src1", (ggml_tensor *)src1);
    // int *ptr = NULL;
    // *ptr = 0;
}

static __dpct_inline__ float warp_reduce_sum(float x,
                                             const sycl::nd_item<3> &item_ct1) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        /*
        DPCT1096:98: The right-most dimension of the work-group used in the SYCL
        kernel that calls this function may be less than "32". The function
        "dpct::permute_sub_group_by_xor" may return an unexpected result on the
        CPU device. Modify the size of the work-group to ensure that the value
        of the right-most dimension is a multiple of "32".
        */
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    return x;
}

static __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, const sycl::nd_item<3> &item_ct1) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
                                                mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
                                                mask);
    }
    return a;
}

static __dpct_inline__ float warp_reduce_max(float x,
                                             const sycl::nd_item<3> &item_ct1) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        /*
        DPCT1096:97: The right-most dimension of the work-group used in the SYCL
        kernel that calls this function may be less than "32". The function
        "dpct::permute_sub_group_by_xor" may return an unexpected result on the
        CPU device. Modify the size of the work-group to ensure that the value
        of the right-most dimension is a multiple of "32".
        */
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                              item_ct1.get_sub_group(), x, mask));
    }
    return x;
}

static __dpct_inline__ float op_repeat(const float a, const float b) {
    return b;
}

static __dpct_inline__ float op_add(const float a, const float b) {
    return a + b;
}

static __dpct_inline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __dpct_inline__ float op_div(const float a, const float b) {
    return a / b;
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        /*int s10,*/ int s11, int s12, int s13,
        const sycl::nd_item<3> &item_ct1) {
    const int i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    const int i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    const int i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    const int i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 = i3*s3 + i2*s2 + i1*s1;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  = i_src0;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        const int i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
    }
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static void k_bin_bcast_unravel(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        /*int s10,*/ int s11, int s12, int s13,
        const sycl::nd_item<3> &item_ct1) {

    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    const int i3 = i/(ne2*ne1*ne0);
    const int i2 = (i/(ne1*ne0)) % ne2;
    const int i1 = (i/ne0) % ne1;
    const int i0 = i % ne0;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 = i3*s3 + i2*s2 + i1*s1;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  = i_src0;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    const int i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
}

static void acc_f32(const float * x, const float * y, float * dst, const int ne,
    const int ne10, const int ne11, const int ne12,
    const int nb1, const int nb2, int offset, const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= ne) {
        return;
    }
    int src1_idx = i - offset;
    int oz = src1_idx / nb2;
    int oy = (src1_idx - (oz * nb2)) / nb1;
    int ox = src1_idx % nb1;
    if (src1_idx >= 0 && ox < ne10 && oy < ne11 && oz < ne12) {
        dst[i] = x[i] + y[ox + oy * ne10 + oz * ne10 * ne11];
    } else {
        dst[i] = x[i];
    }
}

static void gelu_f32(const float * x, float * dst, const int k,
                     const sycl::nd_item<3> &item_ct1) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    float xi = x[i];
    dst[i] = 0.5f * xi *
             (1.0f +
              sycl::tanh(SQRT_2_OVER_PI * xi * (1.0f + GELU_COEF_A * xi * xi)));
}

static void silu_f32(const float * x, float * dst, const int k,
                     const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / (1.0f + sycl::native::exp(-x[i]));
}

static void gelu_quick_f32(const float *x, float *dst, int k,
                           const sycl::nd_item<3> &item_ct1) {
    const float GELU_QUICK_COEF = -1.702f;
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= k) {
        return;
    }
    dst[i] = x[i] * (1.0f / (1.0f + sycl::native::exp(GELU_QUICK_COEF * x[i])));
}

static void tanh_f32(const float *x, float *dst, int k,
                     const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= k) {
        return;
    }
    dst[i] = sycl::tanh((float)(x[i]));
}

static void relu_f32(const float * x, float * dst, const int k,
                     const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = sycl::fmax((float)(x[i]), (float)0);
}

static void leaky_relu_f32(const float *x, float *dst, const int k, const float negative_slope,
                           const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    if (i >= k) {
        return;
    }
    dst[i] = sycl::fmax((float)(x[i]), (float)0) +
             sycl::fmin((float)(x[i]), 0.0f) * negative_slope;
}

static void sqr_f32(const float * x, float * dst, const int k,
                    const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * x[i];
}

static void norm_f32(const float * x, float * dst, const int ncols, const float eps,
                     const sycl::nd_item<3> &item_ct1, sycl::float2 *s_sum, int block_size) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    const int tid = item_ct1.get_local_id(2);

    sycl::float2 mean_var = sycl::float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        mean_var.x() += xi;
        mean_var.y() += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        /*
        DPCT1118:0: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        mean_var = s_sum[lane_id];
        mean_var = warp_reduce_sum(mean_var, item_ct1);
    }

    const float mean = mean_var.x() / ncols;
    const float var = mean_var.y() / ncols - mean * mean;
    const float inv_std = sycl::rsqrt(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = (x[row*ncols + col] - mean) * inv_std;
    }
}

static void concat_f32(const float  *x,const float  *y, float *dst, const int ne0, const int ne02,
                       const sycl::nd_item<3> &item_ct1) {
    int nidx = item_ct1.get_local_id(2) +
               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (nidx >= ne0) {
        return;
    }
    // operation
    int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    if (item_ct1.get_group(0) < ne02) { // src0
        int offset_src =
            nidx + item_ct1.get_group(1) * ne0 +
            item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
            dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx + item_ct1.get_group(1) * ne0 +
            (item_ct1.get_group(0) - ne02) * ne0 * item_ct1.get_group_range(1);
            dst[offset_dst] = y[offset_src];
    }
}

static void upscale_f32(const float  *x, float *dst, const int ne00, const int nb02, const int scale_factor,
                        const sycl::nd_item<3> &item_ct1) {
    int ne0 = ne00 * scale_factor;
    int nidx = item_ct1.get_local_id(2) +
               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (nidx >= ne0) {
        return;
    }
    // operation
    int i00 = nidx / scale_factor;
    int i01 = item_ct1.get_group(1) / scale_factor;
    int offset_src = i00 + i01 * ne00 + item_ct1.get_group(0) * nb02;
    int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    dst[offset_dst] = x[offset_src];
}

static void pad_f32(const float  *x, float *dst, const int ne0, const int ne00, const int ne01, const int ne02,
                    const sycl::nd_item<3> &item_ct1) {
    int nidx = item_ct1.get_local_id(2) +
               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (nidx >= ne0) {
        return;
    }

    // operation
    int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    if (nidx < ne00 && item_ct1.get_group(1) < ne01 &&
        item_ct1.get_group(0) < ne02) {
        int offset_src = nidx + item_ct1.get_group(1) * ne00 +
                         item_ct1.get_group(0) * ne00 * ne01;
            dst[offset_dst] = x[offset_src];
    } else {
        dst[offset_dst] = 0.0f;
    }
}

static void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps,
                           const sycl::nd_item<3> &item_ct1, float *s_sum, int block_size) {
    int start = item_ct1.get_group(2) * group_size;
    int end = start + group_size;

    start += item_ct1.get_local_id(2);

    if (end >= ne_elements) {
        end = ne_elements;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        /*
        DPCT1118:1: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        /*
        DPCT1118:2: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:55: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    float variance = tmp / group_size;
    float scale = sycl::rsqrt(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

static void rms_norm_f32(const float * x, float * dst, const int ncols, const float eps,
                         const sycl::nd_item<3> &item_ct1, float *s_sum, int block_size) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    const int tid = item_ct1.get_local_id(2);

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        /*
        DPCT1118:3: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float mean = tmp / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = scale * x[row*ncols + col];
    }
}

static __dpct_inline__ void dequantize_q4_0(const void *vx, const int ib,
                                            const int iqs, dfloat2 &v) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ifdef GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#else
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void dequantize_q4_1(const void *vx, const int ib,
                                            const int iqs, dfloat2 &v) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = x[ib].dm[0];
    const dfloat m = x[ib].dm[1];

    const int vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = (v.s0() * d) + m;
    v.s1() = (v.s1() * d) + m;

#else
    v.x() = (v.x() * d) + m;
    v.y() = (v.y() * d) + m;
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void dequantize_q5_0(const void *vx, const int ib,
                                            const int iqs, dfloat2 &v) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_SYCL_F16
    // v = v - {16.0f, 16.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 16.0f) * d;
    v.s1() = (v.s1() - 16.0f) * d;

#else
    v.x() = (v.x() - 16.0f) * d;
    v.y() = (v.y() - 16.0f) * d;
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void dequantize_q5_1(const void *vx, const int ib,
                                            const int iqs, dfloat2 &v) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = x[ib].dm[0];
    const dfloat m = x[ib].dm[1];

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = (v.s0() * d) + m;
    v.s1() = (v.s1() * d) + m;
#else
    v.x() = (v.x() * d) + m;
    v.y() = (v.y() * d) + m;
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void dequantize_q8_0(const void *vx, const int ib,
                                            const int iqs, dfloat2 &v) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x() = x[ib].qs[iqs + 0];
    v.y() = x[ib].qs[iqs + 1];

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    v.s0() *= d;
    v.s1() *= d;
#else
    v.x() *= d;
    v.y() *= d;
#endif // GGML_SYCL_F16
}

//================================== k-quants

template<typename dst_t>
static void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {

    const int i = item_ct1.get_group(2);
    const block_q2_K * x = (const block_q2_K *) vx;

    const int tid = item_ct1.get_local_id(2);
#if QK_K == 256
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = x[i].dm[0];
    float dmin = x[i].dm[1];
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#else
    const int is = tid/16;  // 0 or 1
    const int il = tid%16;  // 0...15
    const uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;
    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#endif

}

template<typename dst_t>
static void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {

    const int i = item_ct1.get_group(2);
    const block_q3_K * x = (const block_q3_K *) vx;

#if QK_K == 256
    const int r = item_ct1.get_local_id(2) / 4;
    const int tid = r/2;
    const int is0 = r%2;
    const int l0 = 16 * is0 + 4 * (item_ct1.get_local_id(2) % 4);
    const int n = tid / 4;
    const int j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
#else
    const int tid = threadIdx.x;
    const int is  = tid/16;  // 0 or 1
    const int il  = tid%16;  // 0...15
    const int im  = il/8;    // 0...1
    const int in  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    const uint8_t q = x[i].qs[il] >> (2*is);
    const uint8_t h = x[i].hmask[in] >> (2*is + im);
    const float   d = (float)x[i].d;

    if (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } else {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#endif

}

#if QK_K == 256
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#endif

template<typename dst_t>
static void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int i = item_ct1.get_group(2);

#if QK_K == 256
    // assume 32 threads
    const int tid = item_ct1.get_local_id(2);
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = x[i].dm[0];
    const float dmin = x[i].dm[1];

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
#else
    const int tid = threadIdx.x;
    const uint8_t * q = x[i].qs;
    dst_t * y = yy + i*QK_K;
    const float d = (float)x[i].dm[0];
    const float m = (float)x[i].dm[1];
    y[tid+ 0] = d * (x[i].scales[0] & 0xF) * (q[tid] & 0xF) - m * (x[i].scales[0] >> 4);
    y[tid+32] = d * (x[i].scales[1] & 0xF) * (q[tid] >>  4) - m * (x[i].scales[1] >> 4);
#endif
}

template<typename dst_t>
static void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int i = item_ct1.get_group(2);

#if QK_K == 256
    // assume 64 threads - this is very slightly better than the one below
    const int tid = item_ct1.get_local_id(2);
    const int il  = tid/16;   // il is in 0...3
    const int ir  = tid%16;   // ir is in 0...15
    const int is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = x[i].dm[0];
    const float dmin = x[i].dm[1];

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#else
    const int tid = threadIdx.x;
    const uint8_t q = x[i].qs[tid];
    const int im = tid/8;  // 0...3
    const int in = tid%8;  // 0...7
    const int is = tid/16; // 0 or 1
    const uint8_t h = x[i].qh[in] >> im;
    const float d = x[i].d;
    dst_t * y = yy + i*QK_K + tid;
    y[ 0] = d * x[i].scales[is+0] * ((q & 0xF) - ((h >> 0) & 1 ? 0 : 16));
    y[32] = d * x[i].scales[is+2] * ((q >>  4) - ((h >> 4) & 1 ? 0 : 16));
#endif
}

template<typename dst_t>
static void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int i = item_ct1.get_group(2);
#if QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    const int tid = item_ct1.get_local_id(2);
    const int ip  = tid/32;   // ip is 0 or 1
    const int il  = tid - 32*ip; // 0...32
    const int is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#else

    // assume 32 threads
    const int tid = threadIdx.x;
    const int ip  = tid/16;         // 0 or 1
    const int il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    const float d = x[i].d;

    const uint8_t   ql = x[i].ql[16*ip + il];
    const uint8_t   qh = x[i].qh[il] >> (2*ip);
    const int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#endif
}

/*
DPCT1110:4: The total declared local variable size in device function
dequantize_mul_mat_vec_q2_k exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void dequantize_mul_mat_vec_q2_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q2_K * x = (const block_q2_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...31 or 0...15
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2
    const int q_offset = 32*im + l0;
    const int s_offset = 8*im;
    const int y_offset = 128*im + l0;

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + y_offset;
        const uint8_t * q = x[i].qs + q_offset;

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        const uint32_t * a = (const uint32_t *)(x[i].scales + s_offset);
        aux[0] = a[0] & 0x0f0f0f0f;
        aux[1] = a[1] & 0x0f0f0f0f;
        aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
        aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                  + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                  + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                  + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                  + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                  + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                  + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                  +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
            sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                  + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

        }
        tmp += dall * sum1 - dmin * sum2;

    }
#else
    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;

    uint32_t uaux[2];
    const uint8_t * d = (const uint8_t *)uaux;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint32_t * s = (const uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        const float2 dall = __half22float2(x[i].dm);

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x * sum1 - dall.y * sum2;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

/*
DPCT1110:5: The total declared local variable size in device function
dequantize_mul_mat_vec_q3_k exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void dequantize_mul_mat_vec_q3_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q3_K * x = (const block_q3_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...31 or 0...16
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0....15 or 0...7

    const uint8_t m = 1 << (4*im);

    const int l0 = n*in;                                 // 0...15 or 0...14 in steps of 2
    const int q_offset =  32*im + l0;
    const int y_offset = 128*im + l0;

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    const uint16_t s_shift = 4*im;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * q = x[i].qs + q_offset;
        const uint8_t * h = x[i].hmask + l0;

        const uint16_t * a = (const uint16_t *)x[i].scales;
        utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
        utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
        utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
        utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

        const float d = x[i].d;

        float sum = 0;
        for (int l = 0; l < n; ++l) {
            sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                 + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                 + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                 + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
            sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                 + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                 + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
        }
        tmp += d * sum;

    }
#else

    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;         // 0...15 or 0...14
    const int in = offset/8;                                 // 0 or 1
    const int im = offset%8;                                 // 0...7

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint8_t * s = x[i].scales;

        const float dall = (float)x[i].d;

        float sum = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t hl = x[i].hmask[im+l] >> in;
            const uint8_t ql = q[l];
            sum += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

/*
DPCT1110:6: The total declared local variable size in device function
dequantize_mul_mat_vec_q4_k exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void dequantize_mul_mat_vec_q4_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row > nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q4_K * x = (const block_q4_K *)vx + ib0;

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...31 or 0...16
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    const int step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    const int il  = tid/step;                            // 0...3
    const int ir  = tid - step*il;                       // 0...7 or 0...3
    const int n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

#if K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;
#else
    uint16_t q16[4];
    const uint8_t * q4 = (const uint8_t *)q16;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y1 = yy + i*QK_K + y_offset;
        const float   * y2 = y1 + 128;

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#if K_QUANTS_PER_ITERATION == 2
        const uint32_t * q1 = (const uint32_t *)(x[i].qs + q_offset);
        const uint32_t * q2 = q1 + 16;

        q32[0] = q1[0] & 0x0f0f0f0f;
        q32[1] = q1[0] & 0xf0f0f0f0;
        q32[2] = q2[0] & 0x0f0f0f0f;
        q32[3] = q2[0] & 0xf0f0f0f0;

        sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 4; ++l) {
            s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
            s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                       s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
               dmin * smin;
#else
        const uint16_t * q1 = (const uint16_t *)(x[i].qs + q_offset);
        const uint16_t * q2 = q1 + 32;

        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[0] & 0xf0f0;
        q16[2] = q2[0] & 0x0f0f;
        q16[3] = q2[0] & 0xf0f0;

        float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 2; ++l) {
            s.x += y1[l] * q4[l+0]; s.y += y1[l+32] * q4[l+2];
            s.z += y2[l] * q4[l+4]; s.w += y2[l+32] * q4[l+6];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x * sc[0] + s.y * sc[1] * 1.f/16.f + s.z * sc[4] + s.w * sc[5] * 1.f/16.f) - dmin * smin;
#endif

    }
#else
    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);

    const int step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const float   * y = yy + i*QK_K + step;
        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        const float d = (float)x[i].dm[0];
        const float m = (float)x[i].dm[1];
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += sum;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

/*
DPCT1110:7: The total declared local variable size in device function
dequantize_mul_mat_vec_q5_k exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
static void dequantize_mul_mat_vec_q5_k(const void *__restrict__ vx,
                                        const float *__restrict__ yy,
                                        float *__restrict__ dst,
                                        const int ncols,
                                        const sycl::nd_item<3> &item_ct1) {

    const int row = item_ct1.get_group(2);
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q5_K * x = (const block_q5_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = item_ct1.get_local_id(2) / 2; // 0...15
    const int ix = item_ct1.get_local_id(2) % 2;

    const int il  = tid/4;     // 0...3
    const int ir  = tid - 4*il;// 0...3
    const int n   = 2;

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    const uint8_t hm1  = 1 << (2*im);
    const uint8_t hm2  = hm1 << 4;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {

        const uint8_t * ql1 = x[i].qs + q_offset;
        const uint8_t * qh  = x[i].qh + l0;
        const float   * y1  = yy + i*QK_K + y_offset;
        const float   * y2  = y1 + 128;

        const float dall = x[i].dm[0];
        const float dmin = x[i].dm[1];

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

        sycl::float4 sum = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        const uint16_t * q1 = (const uint16_t *)ql1;
        const uint16_t * q2 = q1 + 32;
        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[8] & 0x0f0f;
        q16[2] = (q1[0] >> 4) & 0x0f0f;
        q16[3] = (q1[8] >> 4) & 0x0f0f;
        q16[4] = q2[0] & 0x0f0f;
        q16[5] = q2[8] & 0x0f0f;
        q16[6] = (q2[0] >> 4) & 0x0f0f;
        q16[7] = (q2[8] >> 4) & 0x0f0f;
        for (int l = 0; l < n; ++l) {
            sum.x() +=
                y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
            sum.y() +=
                y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
            sum.z() +=
                y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
            sum.w() +=
                y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
            smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                  + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
        }
        tmp += dall * (sum.x() * sc[0] + sum.y() * sc[1] + sum.z() * sc[4] +
                       sum.w() * sc[5]) -
               dmin * smin;
    }

#else
    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);
    const int step = tid * K_QUANTS_PER_ITERATION;
    const int im = step/8;
    const int in = step%8;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const int8_t  * s = x[i].scales;
        const float   * y = yy + i*QK_K + step;
        const float     d = x[i].d;
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            const uint8_t h = x[i].qh[in+j] >> im;
            sum += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void dequantize_mul_mat_vec_q6_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows,
                                        const sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q6_K * x = (const block_q6_K *)vx + ib0;

#if QK_K == 256

    const int tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...31 or 0...16
    const int ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * in;                               // 0, 4, 8, ..., 28
    const int is = in / 4;
#endif
    const int ql_offset = 64*im + l0;
    const int qh_offset = 32*im + l0;
    const int s_offset  =  8*im + is;
    const int y_offset = 128*im + l0;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * ql = x[i].ql + ql_offset;
        const uint8_t * qh = x[i].qh + qh_offset;
        const int8_t  * s  = x[i].scales + s_offset;

        const float d = x[i].d;

#if K_QUANTS_PER_ITERATION == 1
        float sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                  + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                  + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                  + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                  + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                  + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                  + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                  +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
        tmp += sum;
#else
        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                 + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                 + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                 + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        tmp += sum;
#endif

    }

#else

    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...7
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);  // 0...3

    const int step = tid * K_QUANTS_PER_ITERATION;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + step;
        const uint8_t * ql = x[i].ql + step;
        const uint8_t * qh = x[i].qh + step;
        const int8_t  * s  = x[i].scales;

        const float d = x[i+0].d;

        float sum = 0;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void convert_f16(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const sycl::half *x = (const sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

static void convert_f32(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const float * x = (const float *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

static void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded,
                          const sycl::nd_item<3> &item_ct1) {
    const int ix = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                   item_ct1.get_local_id(2);

    if (ix >= kx_padded) {
        return;
    }

    const int iy = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                   item_ct1.get_local_id(1);

    const int i_padded = iy*kx_padded + ix;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int ib = i_padded / QK8_1; // block index
    const int iqs = i_padded % QK8_1; // quant index

    const float xi = ix < kx ? x[iy*kx + ix] : 0.0f;
    float amax = sycl::fabs((float)xi);
    float sum = xi;

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = sycl::fmax(amax, dpct::permute_sub_group_by_xor(
                                    item_ct1.get_sub_group(), amax, mask));
        sum +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), sum, mask);
    }

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : sycl::round(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<sycl::half &>(y[ib].ds.x()) = d;
    reinterpret_cast<sycl::half &>(y[ib].ds.y()) = sum;
}

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void k_get_rows(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) *
                    2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib = i00/qk; // block index
    const int iqs = (i00%qk)/qr; // quant index
    const int iybs = i00 - i00%qk; // dst block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_row, ib, iqs, v);

    dst_row[iybs + iqs + 0] = v.x();
    dst_row[iybs + iqs + y_offset] = v.y();
}

template<typename src0_t, typename dst_t>
static void k_get_rows_float(
            const src0_t * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const src0_t * src0_row = (const src0_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_row[i00] = src0_row[i00];
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int k,
                             const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  2 * item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_offset] = v.y();
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

template <int vdr>
static __dpct_inline__ float vec_dot_q4_0_q8_1_impl(const int *v, const int *u,
                                                    const float &d4,
                                                    const sycl::half2 &ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    const sycl::float2 ds8f =
        ds8.convert<float, sycl::rounding_mode::automatic>();

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x() - (8 * vdr / QI4_0) * ds8f.y());
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <int vdr>
static __dpct_inline__ float vec_dot_q4_1_q8_1_impl(const int *v, const int *u,
                                                    const sycl::half2 &dm4,
                                                    const sycl::half2 &ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

#ifdef GGML_SYCL_F16
    const sycl::float2 tmp =
        (dm4 * ds8).convert<float, sycl::rounding_mode::automatic>();
    const float d4d8 = tmp.x();
    const float m4s8 = tmp.y();
#else
    const sycl::float2 dm4f =
        dm4.convert<float, sycl::rounding_mode::automatic>();
    const sycl::float2 ds8f =
        ds8.convert<float, sycl::rounding_mode::automatic>();
    const float d4d8 = dm4f.x() * ds8f.x();
    const float m4s8 = dm4f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <int vdr>
static __dpct_inline__ float
vec_dot_q5_0_q8_1_impl(const int *vl, const int *vh, const int *u,
                       const float &d5, const sycl::half2 &ds8) {
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

    const sycl::float2 ds8f =
        ds8.convert<float, sycl::rounding_mode::automatic>();

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x() - (16 * vdr / QI5_0) * ds8f.y());
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <int vdr>
static __dpct_inline__ float
vec_dot_q5_1_q8_1_impl(const int *vl, const int *vh, const int *u,
                       const sycl::half2 &dm5, const sycl::half2 &ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

#ifdef GGML_SYCL_F16
     const sycl::float2 tmp =
        (dm5 * ds8).convert<float, sycl::rounding_mode::automatic>();
    const float d5d8 = tmp.x();
    const float m5s8 = tmp.y();


#else
    const sycl::float2 dm5f =
        dm5.convert<float, sycl::rounding_mode::automatic>();
    const sycl::float2 ds8f =
        ds8.convert<float, sycl::rounding_mode::automatic>();
    const float d5d8 = dm5f.x() * ds8f.x();
    const float m5s8 = dm5f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <int vdr>
static __dpct_inline__ float vec_dot_q8_0_q8_1_impl(const int *v, const int *u,
                                                    const float &d8_0,
                                                    const float &d8_1) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * sumi;
}

template <int vdr>
static __dpct_inline__ float vec_dot_q8_1_q8_1_impl(const int *v, const int *u,
                                                    const sycl::half2 &dm8,
                                                    const sycl::half2 &ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

#ifdef GGML_SYCL_F16
    const sycl::float2 tmp =
        (dm8 * ds8).convert<float, sycl::rounding_mode::automatic>();
    const float d8d8 = tmp.x();
    const float m8s8 = tmp.y();
#else
    const sycl::float2 dm8f =
        dm8.convert<float, sycl::rounding_mode::automatic>();
    const sycl::float2 ds8f =
        ds8.convert<float, sycl::rounding_mode::automatic>();
    const float d8d8 = dm8f.x() * ds8f.x();
    const float m8s8 = dm8f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ  2

// contiguous v/x values
static __dpct_inline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int &v, const int *__restrict__ u, const uint8_t *__restrict__ scales,
    const sycl::half2 &dm2, const float *__restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d +=
            d8[i] * (dpct::dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] *
                  dpct::dp4a(
                      m, u[i],
                      0); // multiply constant q2_K part with sum of q8_1 values
    }

    const sycl::float2 dm2f =
        dm2.convert<float, sycl::rounding_mode::automatic>();

    return dm2f.x() * sumf_d - dm2f.y() * sumf_m;
}

// contiguous u/y values
static __dpct_inline__ float
vec_dot_q2_K_q8_1_impl_mmq(const int *__restrict__ v, const int *__restrict__ u,
                           const uint8_t *__restrict__ scales,
                           const sycl::half2 &dm2, const float &d8) {

    int sumi_d = 0;
    int sumi_m = 0;

#pragma unroll
    for (int i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        int sumi_d_sc = 0;

        const int sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = dpct::dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m = dpct::dp4a(m, u[i],
                                sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    const sycl::float2 dm2f =
        dm2.convert<float, sycl::rounding_mode::automatic>();

    return d8 * (dm2f.x() * sumi_d - dm2f.y() * sumi_m);
}

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ  2

// contiguous v/x values
static __dpct_inline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int &vl, const int &vh, const int *__restrict__ u,
    const uint8_t *__restrict__ scales, const int &scale_offset,
    const float &d3, const float *__restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        const int vi =
            dpct::vectorized_binary<sycl::char4>(vil, vih, dpct::sub_sat());

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d3 * sumf;
}

// contiguous u/y values
static __dpct_inline__ float
vec_dot_q3_K_q8_1_impl_mmq(const int *__restrict__ v, const int *__restrict__ u,
                           const int8_t *__restrict__ scales, const float &d3,
                           const float &d8) {

    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = dpct::dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous v/x values
static __dpct_inline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const sycl::half2 &dm4, const float *__restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 =
            dpct::dp4a(v1i, u[2 * i + 1],
                       dpct::dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
        const int dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 1],
                       dpct::dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const sycl::float2 dm4f =
        dm4.convert<float, sycl::rounding_mode::automatic>();

    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

// contiguous u/y values
static __dpct_inline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const sycl::half2 &dm4, const sycl::half2 *__restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F,
                                u[i * QI8_1 + j], sumi_d); // SIMD dot product
        }

        const sycl::float2 ds8f =
            ds8[i].convert<float, sycl::rounding_mode::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    const sycl::float2 dm4f =
        dm4.convert<float, sycl::rounding_mode::automatic>();

    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous v/x values
static __dpct_inline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int *__restrict__ vl, const int *__restrict__ vh,
    const int *__restrict__ u, const uint8_t *__restrict__ sc,
    const uint8_t *__restrict__ m, const sycl::half2 &dm5,
    const float *__restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 =
            dpct::dp4a(v0i, u[2 * i + 0],
                       dpct::dp4a(v1i, u[2 * i + 1], 0)); // SIMD dot product
        const int dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 0],
                       dpct::dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const sycl::float2 dm5f =
        dm5.convert<float, sycl::rounding_mode::automatic>();

    return dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}

// contiguous u/y values
static __dpct_inline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const sycl::half2 &dm4, const sycl::half2 *__restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = dpct::dp4a(v[i * QI8_1 + j], u[i * QI8_1 + j],
                                sumi_d); // SIMD dot product
        }

        const sycl::float2 ds8f =
            ds8[i].convert<float, sycl::rounding_mode::automatic>();

        sumf_d += ds8f.x() * (sc[i] * sumi_d);
        sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
    }

    const sycl::float2 dm4f =
        dm4.convert<float, sycl::rounding_mode::automatic>();

    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous v/x values
static __dpct_inline__ float
vec_dot_q6_K_q8_1_impl_mmvq(const int &vl, const int &vh,
                            const int *__restrict__ u,
                            const int8_t *__restrict__ scales, const float &d,
                            const float *__restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = dpct::vectorized_binary<sycl::char4>(
            (vil | vih), 0x20202020, dpct::sub_sat()); // vi = (vil | vih) - 32

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}

// contiguous u/y values
static __dpct_inline__ float
vec_dot_q6_K_q8_1_impl_mmq(const int *__restrict__ v, const int *__restrict__ u,
                           const int8_t *__restrict__ sc, const float &d6,
                           const float *__restrict__ d8) {

    float sumf_d = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        sycl::int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x() = dpct::dp4a(v[2 * i + 0], u[2 * i + 0],
                                    sumi_d.x()); // SIMD dot product
            sumi_d.x() = dpct::dp4a(v[2 * i + 1], u[2 * i + 1],
                                    sumi_d.x()); // SIMD dot product

            sumi_d.y() = dpct::dp4a(v[2 * i + 4], u[2 * i + 4],
                                    sumi_d.y()); // SIMD dot product
            sumi_d.y() = dpct::dp4a(v[2 * i + 5], u[2 * i + 5],
                                    sumi_d.y()); // SIMD dot product
        }

        sumf_d += d8[i0 / 4] *
                  (sc[i0 / 2 + 0] * sumi_d.x() + sc[i0 / 2 + 1] * sumi_d.y());
    }

    return d6 * sumf_d;
}

static __dpct_inline__ float
vec_dot_q4_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q4_0(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_qs_q4_0, float *tile_x_d_q4_0) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_qs_q4_0;
    *x_dm = (sycl::half2 *)tile_x_d_q4_0;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q4_0(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;
    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;

    const block_q4_0 * bx0 = (const block_q4_0 *) vx;

    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

static __dpct_inline__ float vec_dot_q4_0_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const float * x_dmf = (const float *) x_dm;

    int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_0) % WARP_SIZE];
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dmf[i * (WARP_SIZE/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

static __dpct_inline__ float
vec_dot_q4_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q4_1(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_qs_q4_1, sycl::half2 *tile_x_dm_q4_1) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_qs_q4_1;
    *x_dm = tile_x_dm_q4_1;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q4_1(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_1;
    const int kqsx = k % QI4_1;

    const block_q4_1 * bx0 = (const block_q4_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + i_offset * QI4_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

static __dpct_inline__ float vec_dot_q4_1_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_1) % WARP_SIZE];
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dm[i * (WARP_SIZE/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

static __dpct_inline__ float
vec_dot_q5_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q5_0(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q5_0, float *tile_x_d_q5_0) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_ql_q5_0;
    *x_dm = (sycl::half2 *)tile_x_d_q5_0;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q5_0(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_0;
    const int kqsx = k % QI5_0;

    const block_q5_0 * bx0 = (const block_q5_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8(bxi->qs, kqsx);
        const int qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0 = dpct::vectorized_binary<sycl::char4>(
            qs0, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1 = dpct::vectorized_binary<sycl::char4>(
            qs1, 0x10101010, dpct::sub_sat()); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

static __dpct_inline__ float vec_dot_q5_0_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_0) + i/QI5_0 + k/QI5_0;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    int u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_0) % WARP_SIZE];
    }

    return vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dmf[index_bx], y_df[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

static __dpct_inline__ float
vec_dot_q5_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q5_1(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q5_1, sycl::half2 *tile_x_dm_q5_1) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_ql_q5_1;
    *x_dm = tile_x_dm_q5_1;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q5_1(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset < nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_1;
    const int kqsx = k % QI5_1;

    const block_q5_1 * bx0 = (const block_q5_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + i_offset * QI5_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

static __dpct_inline__ float vec_dot_q5_1_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_1) + + i/QI5_1 + k/QI5_1;

    int u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_1) % WARP_SIZE];
    }

    return vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dm[index_bx], y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

static __dpct_inline__ float
vec_dot_q8_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                      bq8_1->ds[0]);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q8_0(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_qs_q8_0, float *tile_x_d_q8_0) {
    (void)x_qh; (void)x_sc;

    *x_ql = tile_x_qs_q8_0;
    *x_dm = (sycl::half2 *)tile_x_d_q8_0;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q8_0(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh; (void)x_sc;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;
    float * x_dmf = (float *) x_dm;

    const block_q8_0 * bx0 = (const block_q8_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI8_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

static __dpct_inline__ float vec_dot_q8_0_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh; (void)x_sc;

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[j * WARP_SIZE + k], x_dmf[i * (WARP_SIZE/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_SIZE/QI8_1) + k/QI8_1]);
}

static __dpct_inline__ float
vec_dot_q2_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + i].ds[0];
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q2_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q2_K, sycl::half2 *tile_x_dm_q2_K,
                    int *tile_x_sc_q2_K) {
    (void)x_qh;

    *x_ql = tile_x_ql_q2_K;
    *x_dm = tile_x_dm_q2_K;
    *x_sc = tile_x_sc_q2_K;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q2_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI2_K;
    const int kqsx = k % QI2_K;

    const block_q2_K * bx0 = (const block_q2_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI2_K;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        int i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI2_K/4);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

static __dpct_inline__ float vec_dot_q2_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh;

    const int kbx = k / QI2_K;
    const int ky  = (k % QI2_K) * QR2_K;
    const float * y_df = (const float *) y_ds;

    int v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    const int kqsx = i * (WARP_SIZE + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    const int shift = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#pragma unroll
    for (int l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> shift) & 0x03030303;
    }

    const uint8_t * scales = ((const uint8_t *) &x_sc[i * (WARP_SIZE/4) + i/4 + kbx*4]) + ky/4;

    const int index_y = j * WARP_SIZE + (QR2_K*k) % WARP_SIZE;
    return vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dm[i * (WARP_SIZE/QI2_K) + i/QI2_K + kbx], y_df[index_y/QI8_1]);
}

static __dpct_inline__ float
vec_dot_q3_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + i].ds[0];
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q3_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q3_K, sycl::half2 *tile_x_dm_q3_K,
                    int *tile_x_qh_q3_K, int *tile_x_sc_q3_K) {

    *x_ql = tile_x_ql_q3_K;
    *x_dm = tile_x_dm_q3_K;
    *x_qh = tile_x_qh_q3_K;
    *x_sc = tile_x_sc_q3_K;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q3_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI3_K;
    const int kqsx = k % QI3_K;

    const block_q3_K * bx0 = (const block_q3_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI3_K;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        int i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        int i = i0 + i_offset * 2 + k / (WARP_SIZE/2);

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_SIZE/2) + i / 2 + k % (WARP_SIZE/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI3_K/4);

        const int ksc = k % (QI3_K/4);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = dpct::vectorized_binary<sycl::char4>(
            sc_low | sc_high, 0x20202020, dpct::sub_sat());

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = sc;
    }
}

static __dpct_inline__ float vec_dot_q3_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {

    const int kbx  = k / QI3_K;
    const int ky  = (k % QI3_K) * QR3_K;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * scales = ((const int8_t *) (x_sc + i * (WARP_SIZE/4) + i/4 + kbx*4)) + ky/4;

    int v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        const int kqsx = i * (WARP_SIZE + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        const int shift = 2 * ((ky % 32) / 8);
        const int vll = (x_ql[kqsx + l] >> shift) & 0x03030303;

        const int vh = x_qh[i * (WARP_SIZE/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        const int vlh = (vh << 2) & 0x04040404;

        v[l] = dpct::vectorized_binary<sycl::char4>(vll, vlh, dpct::sub_sat());
    }

    const int index_y = j * WARP_SIZE + (k*QR3_K) % WARP_SIZE;
    return vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dmf[i * (WARP_SIZE/QI3_K) + i/QI3_K + kbx], y_df[index_y/QI8_1]);
}

static __dpct_inline__ float
vec_dot_q4_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

#ifndef GGML_QKK_64
    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->ds[0];

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);

#else

#if __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    const uint16_t * a = (const uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    const float dall = bq4_K->dm[0];
    const float dmin = bq4_K->dm[1];

    const float d8_1 = __low2float(bq8_1[0].ds);
    const float d8_2 = __low2float(bq8_1[1].ds);

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * q4 = (const int *)bq4_K->qs + (iqs/2);
    const int v1 = q4[0];
    const int v2 = q4[4];

    const int dot1 = __dp4a(ui2, v2 & 0x0f0f0f0f, __dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    const int dot2 = __dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, __dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    const int dot3 = __dp4a(0x01010101, ui2, __dp4a(0x01010101, ui1, 0));
    const int dot4 = __dp4a(0x01010101, ui4, __dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    return dall * sumf_d - dmin * sumf_m;

#else
    bad_arch();
#endif // __SYCL_ARCH__ >= VER_4VEC

#endif
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q4_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q4_K, sycl::half2 *tile_x_dm_q4_K,
                    int *tile_x_sc_q4_K) {
    (void)x_qh;

    *x_ql = tile_x_ql_q4_K;
    *x_dm = tile_x_dm_q4_K;
    *x_sc = tile_x_sc_q4_K;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q4_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_K; // == 0 if QK_K == 256
    const int kqsx = k % QI4_K; // == k if QK_K == 256

    const block_q4_K * bx0 = (const block_q4_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#else
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

static __dpct_inline__ float vec_dot_q4_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh;

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    const int index_y = j * WARP_SIZE + (QR4_K*k) % WARP_SIZE;
    return vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI4_K) + i/QI4_K], &y_ds[index_y/QI8_1]);
}

static __dpct_inline__ float
vec_dot_q5_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

#ifndef GGML_QKK_64
    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->ds[0];

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#else

#if __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    const int8_t * s = bq5_K->scales;

    const float d = bq5_K->d;

    const float d8_1 = __low2half(bq8_1[0].ds);
    const float d8_2 = __low2half(bq8_1[1].ds);

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * ql = (const int *)bq5_K->qs + (iqs/2);
    const int vl1 = ql[0];
    const int vl2 = ql[4];

    const int step = 4 * (iqs/2); // 0, 4, 8, 12
    const int im = step/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    const int in = step%8; // 0, 4, 0, 4
    const int vh = (*((const int *)(bq5_K->qh + in))) >> im;

    const int v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    const int v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    const int v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    const int v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    const float sumf_d = d8_1 * (__dp4a(ui1, v1, 0) * s[0] + __dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (__dp4a(ui3, v3, 0) * s[2] + __dp4a(ui4, v4, 0) * s[3]);

    return d * sumf_d;

#else
    bad_arch();
#endif // __SYCL_ARCH__ >= VER_4VEC

#endif
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q5_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql_q5_K, sycl::half2 *tile_x_dm_q5_K,
                    int *tile_x_sc_q5_K) {
    (void)x_qh;

    *x_ql = tile_x_ql_q5_K;
    *x_dm = tile_x_dm_q5_K;
    *x_sc = tile_x_sc_q5_K;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q5_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_K; // == 0 if QK_K == 256
    const int kqsx = k % QI5_K; // == k if QK_K == 256

    const block_q5_K * bx0 = (const block_q5_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR5_K*kqsx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = ql1 | qh1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI5_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

static __dpct_inline__ float vec_dot_q5_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh;

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    const int index_x = i * (QR5_K*WARP_SIZE + 1) +  QR5_K*k;
    const int index_y = j * WARP_SIZE             + (QR5_K*k) % WARP_SIZE;
    return vec_dot_q5_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI5_K) + i/QI5_K], &y_ds[index_y/QI8_1]);
}

static __dpct_inline__ float
vec_dot_q6_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + 2 * i].ds[0];
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

template <int mmq_y>
static __dpct_inline__ void
allocate_tiles_q6_K(int **x_ql, sycl::half2 **x_dm, int **x_qh, int **x_sc,
                    int *tile_x_ql, sycl::half2 *tile_x_dm, int *tile_x_sc) {
    (void)x_qh;

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check>
static __dpct_inline__ void
load_tiles_q6_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
    (void)x_qh;

    GGML_SYCL_ASSUME(i_offset >= 0);
    GGML_SYCL_ASSUME(i_offset <  nwarps);
    GGML_SYCL_ASSUME(k >= 0);
    GGML_SYCL_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    const block_q6_K * bx0 = (const block_q6_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR6_K*kqsx;

        const int ql = get_int_from_uint8(bxi->ql, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        const int qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        const int kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2 * WARP_SIZE + 1) + kq0] =
            dpct::vectorized_binary<sycl::char4>(ql0 | qh0, 0x20202020,
                                                 dpct::sub_sat());
        x_ql[i * (2 * WARP_SIZE + 1) + kq1] =
            dpct::vectorized_binary<sycl::char4>(ql1 | qh1, 0x20202020,
                                                 dpct::sub_sat());
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI6_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = sycl::min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / 4;

        x_sc[i * (WARP_SIZE/8) + i / 8 + k % (WARP_SIZE/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

static __dpct_inline__ float vec_dot_q6_K_q8_1_mul_mat(
    const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
    const int *__restrict__ x_qh, const int *__restrict__ x_sc,
    const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
    const int &i, const int &j, const int &k) {
    (void)x_qh;

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/8]);

    const int index_x = i * (QR6_K*WARP_SIZE + 1) +  QR6_K*k;
    const int index_y = j * WARP_SIZE             + (QR6_K*k) % WARP_SIZE;
    return vec_dot_q6_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, x_dmf[i * (WARP_SIZE/QI6_K) + i/QI6_K], &y_df[index_y/QI8_1]);
}

template <int qk, int qr, int qi, bool need_sum, typename block_q_t, int mmq_x,
          int mmq_y, int nwarps, load_tiles_sycl_t load_tiles, int vdr,
          vec_dot_q_mul_mat_sycl_t vec_dot>
/*
DPCT1110:8: The total declared local variable size in device function mul_mat_q
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
static __dpct_inline__ void
mul_mat_q(const void *__restrict__ vx, const void *__restrict__ vy,
          float *__restrict__ dst, const int ncols_x, const int nrows_x,
          const int ncols_y, const int nrows_y, const int nrows_dst,
          int *tile_x_ql, sycl::half2 *tile_x_dm, int *tile_x_qh,
          int *tile_x_sc, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
          sycl::half2 *tile_y_ds) {

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = WARP_SIZE / qi;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = item_ct1.get_group(2) * mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = item_ct1.get_group(1) * mmq_x;
    const int & col_y_0 = col_dst_0;

    float sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
                   tile_x_qh, tile_x_sc, item_ct1.get_local_id(1),
                   nrows_x - row_x_0 - 1, item_ct1.get_local_id(2),
                   blocks_per_row_x);

#pragma unroll
        for (int ir = 0; ir < qr; ++ir) {
            const int kqs = ir * WARP_SIZE + item_ct1.get_local_id(2);
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = dpct::min(
                    (unsigned int)(col_y_0 + item_ct1.get_local_id(1) + i),
                    ncols_y - 1); // to prevent out-of-bounds memory accesses

                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                const int index_y = (item_ct1.get_local_id(1) + i) * WARP_SIZE +
                                    kqs % WARP_SIZE;
                tile_y_qs[index_y] = get_int_from_int8_aligned(
                    by0->qs, item_ct1.get_local_id(2) % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids =
                    (ids0 + item_ct1.get_local_id(1) * QI8_1 +
                     item_ct1.get_local_id(2) / (WARP_SIZE / QI8_1)) %
                    mmq_x;
                const int kby = item_ct1.get_local_id(2) % (WARP_SIZE / QI8_1);
                const int col_y_eff = sycl::min(col_y_0 + ids, ncols_y - 1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                const sycl::half2 *dsi_src =
                    &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) +
                       ir * (WARP_SIZE / QI8_1) + kby]
                         .ds;
                sycl::half2 *dsi_dst =
                    &tile_y_ds[ids * (WARP_SIZE / QI8_1) + kby];
                if (need_sum) {
                    *dsi_dst = *dsi_src;
                } else {
                    float * dfi_dst = (float *) dsi_dst;
                    *dfi_dst = (*dsi_src)[0];
                }
            }

            /*
            DPCT1118:9: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:56: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k = ir*WARP_SIZE/qr; k < (ir+1)*WARP_SIZE/qr; k += vdr) {
#pragma unroll
                for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    for (int i = 0; i < mmq_y; i += WARP_SIZE) {
                        sum[i / WARP_SIZE][j / nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                            tile_y_qs, tile_y_ds, item_ct1.get_local_id(2) + i,
                            item_ct1.get_local_id(1) + j, k);
                    }
                }
            }

            /*
            DPCT1118:10: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:57: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
    }

#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + item_ct1.get_local_id(1);

        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;

            if (row_dst >= nrows_dst) {
                continue;
            }

            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE][j/nwarps];
        }
    }
}

#define  MMQ_X_Q4_0_RDNA2  64
#define  MMQ_Y_Q4_0_RDNA2  128
#define NWARPS_Q4_0_RDNA2  8
#define  MMQ_X_Q4_0_RDNA1  64
#define  MMQ_Y_Q4_0_RDNA1  64
#define NWARPS_Q4_0_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_0_AMPERE 4
#define  MMQ_Y_Q4_0_AMPERE 32
#define NWARPS_Q4_0_AMPERE 4
#else
#define  MMQ_X_Q4_0_AMPERE 64
#define  MMQ_Y_Q4_0_AMPERE 128
#define NWARPS_Q4_0_AMPERE 4
#endif
#define  MMQ_X_Q4_0_PASCAL 64
#define  MMQ_Y_Q4_0_PASCAL 64
#define NWARPS_Q4_0_PASCAL 8

template <bool need_check> static void
    mul_mat_q4_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q4_0, float *tile_x_d_q4_0,
    int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware

    const int mmq_x  =  MMQ_X_Q4_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    const int nwarps = NWARPS_Q4_0_AMPERE;
    allocate_tiles_q4_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_0, tile_x_d_q4_0);
    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps,
              load_tiles_q4_0<mmq_y, nwarps, need_check>, VDR_Q4_0_Q8_1_MMQ,
              vec_dot_q4_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q4_1_RDNA2  64
#define  MMQ_Y_Q4_1_RDNA2  128
#define NWARPS_Q4_1_RDNA2  8
#define  MMQ_X_Q4_1_RDNA1  64
#define  MMQ_Y_Q4_1_RDNA1  64
#define NWARPS_Q4_1_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_1_AMPERE 4
#define  MMQ_Y_Q4_1_AMPERE 32
#define NWARPS_Q4_1_AMPERE 4
#else
#define  MMQ_X_Q4_1_AMPERE 64
#define  MMQ_Y_Q4_1_AMPERE 128
#define NWARPS_Q4_1_AMPERE 4
#endif
#define  MMQ_X_Q4_1_PASCAL 64
#define  MMQ_Y_Q4_1_PASCAL 64
#define NWARPS_Q4_1_PASCAL 8

template <bool need_check> static void
    mul_mat_q4_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q4_1,
    sycl::half2 *tile_x_dm_q4_1, int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q4_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    const int nwarps = NWARPS_Q4_1_AMPERE;
    allocate_tiles_q4_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q4_1, tile_x_dm_q4_1);
    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps,
              load_tiles_q4_1<mmq_y, nwarps, need_check>, VDR_Q4_1_Q8_1_MMQ,
              vec_dot_q4_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q5_0_RDNA2  64
#define  MMQ_Y_Q5_0_RDNA2  128
#define NWARPS_Q5_0_RDNA2  8
#define  MMQ_X_Q5_0_RDNA1  64
#define  MMQ_Y_Q5_0_RDNA1  64
#define NWARPS_Q5_0_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_0_AMPERE 4
#define  MMQ_Y_Q5_0_AMPERE 32
#define NWARPS_Q5_0_AMPERE 4
#else
#define  MMQ_X_Q5_0_AMPERE 128
#define  MMQ_Y_Q5_0_AMPERE 64
#define NWARPS_Q5_0_AMPERE 4
#endif
#define  MMQ_X_Q5_0_PASCAL 64
#define  MMQ_Y_Q5_0_PASCAL 64
#define NWARPS_Q5_0_PASCAL 8

template <bool need_check> static void
    mul_mat_q5_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q5_0, float *tile_x_d_q5_0,
    int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q5_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    const int nwarps = NWARPS_Q5_0_AMPERE;
    allocate_tiles_q5_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_0, tile_x_d_q5_0);
    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps,
              load_tiles_q5_0<mmq_y, nwarps, need_check>, VDR_Q5_0_Q8_1_MMQ,
              vec_dot_q5_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q5_1_RDNA2  64
#define  MMQ_Y_Q5_1_RDNA2  128
#define NWARPS_Q5_1_RDNA2  8
#define  MMQ_X_Q5_1_RDNA1  64
#define  MMQ_Y_Q5_1_RDNA1  64
#define NWARPS_Q5_1_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_1_AMPERE 4
#define  MMQ_Y_Q5_1_AMPERE 32
#define NWARPS_Q5_1_AMPERE 4
#else
#define  MMQ_X_Q5_1_AMPERE 128
#define  MMQ_Y_Q5_1_AMPERE 64
#define NWARPS_Q5_1_AMPERE 4
#endif
#define  MMQ_X_Q5_1_PASCAL 64
#define  MMQ_Y_Q5_1_PASCAL 64
#define NWARPS_Q5_1_PASCAL 8

template <bool need_check> static void
mul_mat_q5_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q5_1,
    sycl::half2 *tile_x_dm_q5_1, int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q5_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    const int nwarps = NWARPS_Q5_1_AMPERE;
    allocate_tiles_q5_1<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_1, tile_x_dm_q5_1);
    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps,
              load_tiles_q5_1<mmq_y, nwarps, need_check>, VDR_Q5_1_Q8_1_MMQ,
              vec_dot_q5_1_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q8_0_RDNA2  64
#define  MMQ_Y_Q8_0_RDNA2  128
#define NWARPS_Q8_0_RDNA2  8
#define  MMQ_X_Q8_0_RDNA1  64
#define  MMQ_Y_Q8_0_RDNA1  64
#define NWARPS_Q8_0_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q8_0_AMPERE 4
#define  MMQ_Y_Q8_0_AMPERE 32
#define NWARPS_Q8_0_AMPERE 4
#else
#define  MMQ_X_Q8_0_AMPERE 128
#define  MMQ_Y_Q8_0_AMPERE 64
#define NWARPS_Q8_0_AMPERE 4
#endif
#define  MMQ_X_Q8_0_PASCAL 64
#define  MMQ_Y_Q8_0_PASCAL 64
#define NWARPS_Q8_0_PASCAL 8

template <bool need_check> static void
    mul_mat_q8_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_qs_q8_0, float *tile_x_d_q8_0,
    int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q8_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    const int nwarps = NWARPS_Q8_0_AMPERE;
    allocate_tiles_q8_0<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_qs_q8_0, tile_x_d_q8_0);
    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps,
              load_tiles_q8_0<mmq_y, nwarps, need_check>, VDR_Q8_0_Q8_1_MMQ,
              vec_dot_q8_0_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q2_K_RDNA2  64
#define  MMQ_Y_Q2_K_RDNA2  128
#define NWARPS_Q2_K_RDNA2  8
#define  MMQ_X_Q2_K_RDNA1  128
#define  MMQ_Y_Q2_K_RDNA1  32
#define NWARPS_Q2_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q2_K_AMPERE 4
#define  MMQ_Y_Q2_K_AMPERE 32
#define NWARPS_Q2_K_AMPERE 4
#else
#define  MMQ_X_Q2_K_AMPERE 64
#define  MMQ_Y_Q2_K_AMPERE 128
#define NWARPS_Q2_K_AMPERE 4
#endif
#define  MMQ_X_Q2_K_PASCAL 64
#define  MMQ_Y_Q2_K_PASCAL 64
#define NWARPS_Q2_K_PASCAL 8

template <bool need_check> static void
mul_mat_q2_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q2_K,
    sycl::half2 *tile_x_dm_q2_K, int *tile_x_sc_q2_K, int *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q2_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    const int nwarps = NWARPS_Q2_K_AMPERE;
    allocate_tiles_q2_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q2_K, tile_x_dm_q2_K, tile_x_sc_q2_K);
    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps,
              load_tiles_q2_K<mmq_y, nwarps, need_check>, VDR_Q2_K_Q8_1_MMQ,
              vec_dot_q2_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q3_K_RDNA2  128
#define  MMQ_Y_Q3_K_RDNA2  64
#define NWARPS_Q3_K_RDNA2  8
#define  MMQ_X_Q3_K_RDNA1  32
#define  MMQ_Y_Q3_K_RDNA1  128
#define NWARPS_Q3_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q3_K_AMPERE 4
#define  MMQ_Y_Q3_K_AMPERE 32
#define NWARPS_Q3_K_AMPERE 4
#else
#define  MMQ_X_Q3_K_AMPERE 128
#define  MMQ_Y_Q3_K_AMPERE 128
#define NWARPS_Q3_K_AMPERE 4
#endif
#define  MMQ_X_Q3_K_PASCAL 64
#define  MMQ_Y_Q3_K_PASCAL 64
#define NWARPS_Q3_K_PASCAL 8

template <bool need_check> static void
mul_mat_q3_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q3_K,
    sycl::half2 *tile_x_dm_q3_K, int *tile_x_qh_q3_K, int *tile_x_sc_q3_K,
    int *tile_y_qs, sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q3_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    const int nwarps = NWARPS_Q3_K_AMPERE;
    allocate_tiles_q3_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q3_K, tile_x_dm_q3_K, tile_x_qh_q3_K,
                               tile_x_sc_q3_K);
    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps,
              load_tiles_q3_K<mmq_y, nwarps, need_check>, VDR_Q3_K_Q8_1_MMQ,
              vec_dot_q3_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q4_K_RDNA2  64
#define  MMQ_Y_Q4_K_RDNA2  128
#define NWARPS_Q4_K_RDNA2  8
#define  MMQ_X_Q4_K_RDNA1  32
#define  MMQ_Y_Q4_K_RDNA1  64
#define NWARPS_Q4_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q4_K_AMPERE 4
#define  MMQ_Y_Q4_K_AMPERE 32
#define NWARPS_Q4_K_AMPERE 4
#else
#define  MMQ_X_Q4_K_AMPERE 64
#define  MMQ_Y_Q4_K_AMPERE 128
#define NWARPS_Q4_K_AMPERE 4
#endif
#define  MMQ_X_Q4_K_PASCAL 64
#define  MMQ_Y_Q4_K_PASCAL 64
#define NWARPS_Q4_K_PASCAL 8

template <bool need_check> static void
    mul_mat_q4_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q4_K,
    sycl::half2 *tile_x_dm_q4_K, int *tile_x_sc_q4_K, int *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q4_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    const int nwarps = NWARPS_Q4_K_AMPERE;
    allocate_tiles_q4_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q4_K, tile_x_dm_q4_K, tile_x_sc_q4_K);
    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps,
              load_tiles_q4_K<mmq_y, nwarps, need_check>, VDR_Q4_K_Q8_1_MMQ,
              vec_dot_q4_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q5_K_RDNA2  64
#define  MMQ_Y_Q5_K_RDNA2  128
#define NWARPS_Q5_K_RDNA2  8
#define  MMQ_X_Q5_K_RDNA1  32
#define  MMQ_Y_Q5_K_RDNA1  64
#define NWARPS_Q5_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q5_K_AMPERE 4
#define  MMQ_Y_Q5_K_AMPERE 32
#define NWARPS_Q5_K_AMPERE 4
#else
#define  MMQ_X_Q5_K_AMPERE 64
#define  MMQ_Y_Q5_K_AMPERE 128
#define NWARPS_Q5_K_AMPERE 4
#endif
#define  MMQ_X_Q5_K_PASCAL 64
#define  MMQ_Y_Q5_K_PASCAL 64
#define NWARPS_Q5_K_PASCAL 8

template <bool need_check> static void
mul_mat_q5_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql_q5_K,
    sycl::half2 *tile_x_dm_q5_K, int *tile_x_sc_q5_K, int *tile_y_qs,
    sycl::half2 *tile_y_ds) {
    int   * tile_x_ql = nullptr;
    sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q5_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    const int nwarps = NWARPS_Q5_K_AMPERE;
    allocate_tiles_q5_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql_q5_K, tile_x_dm_q5_K, tile_x_sc_q5_K);
    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps,
              load_tiles_q5_K<mmq_y, nwarps, need_check>, VDR_Q5_K_Q8_1_MMQ,
              vec_dot_q5_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

#define  MMQ_X_Q6_K_RDNA2  64
#define  MMQ_Y_Q6_K_RDNA2  128
#define NWARPS_Q6_K_RDNA2  8
#define  MMQ_X_Q6_K_RDNA1  32
#define  MMQ_Y_Q6_K_RDNA1  64
#define NWARPS_Q6_K_RDNA1  8
#if defined(SYCL_USE_XMX)
#define  MMQ_X_Q6_K_AMPERE 4
#define  MMQ_Y_Q6_K_AMPERE 32
#define NWARPS_Q6_K_AMPERE 4
#else
#define  MMQ_X_Q6_K_AMPERE 64
#define  MMQ_Y_Q6_K_AMPERE 64
#define NWARPS_Q6_K_AMPERE 4
#endif
#define  MMQ_X_Q6_K_PASCAL 64
#define  MMQ_Y_Q6_K_PASCAL 64
#define NWARPS_Q6_K_PASCAL 8

template <bool need_check> static void
    mul_mat_q6_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst,
    const sycl::nd_item<3> &item_ct1, int *tile_x_ql, sycl::half2 *tile_x_dm,
    int *tile_x_sc, int *tile_y_qs, sycl::half2 *tile_y_ds) {
    // int   * tile_x_ql = nullptr;
    // sycl::half2 *tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    // int   * tile_x_sc = nullptr;

//sycl_todo: change according to hardware
    const int mmq_x  =  MMQ_X_Q6_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    const int nwarps = NWARPS_Q6_K_AMPERE;
    allocate_tiles_q6_K<mmq_y>(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                               tile_x_ql, tile_x_dm, tile_x_sc);
    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps,
              load_tiles_q6_K<mmq_y, nwarps, need_check>, VDR_Q6_K_Q8_1_MMQ,
              vec_dot_q6_K_q8_1_mul_mat>(
        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, tile_x_ql,
        tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs, tile_y_ds);
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst, const int ncols, const int nrows,
                          const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = 0; i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i +
                        item_ct1.get_local_id(2) / (qi / vdr); // x block index

        const int iby = (i + item_ct1.get_local_id(2) / (qi / vdr)) *
                        (qk / QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static void dequantize_mul_mat_vec(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows,
                                   const sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);

    const int iter_stride = 2*GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#else
    float tmp = 0.0f;
#endif // GGML_SYCL_F16

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#else
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
#ifdef GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#else
        dst[row] = tmp;
#endif // GGML_SYCL_F16
    }
}

static void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        const int row_y = col_x;


        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x, const int channel_x_divisor,
    const sycl::nd_item<3> &item_ct1) {

    const sycl::half *x = (const sycl::half *)vx;

    const int row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    const int channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    const int channel_x = channel / channel_x_divisor;

    const int nrows_y   = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        const int col_x = col_x0 + item_ct1.get_local_id(2);

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel*nrows_y + row_y;

        const float xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<float, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

static void cpy_1_f32_f32(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    float * dsti = (float *) cdsti;

    *dsti = *xi;
}

static void cpy_1_f32_f16(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    sycl::half *dsti = (sycl::half *)cdsti;

    *dsti = sycl::vec<float, 1>(*xi)
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

static void cpy_1_f16_f16(const char * cxi, char * cdsti) {
    const sycl::half *xi = (const sycl::half *)cxi;
    sycl::half *dsti = (sycl::half *)cdsti;

    *dsti = *xi;
}

static void cpy_1_i16_i16(const char * cxi, char * cdsti) {
    const int16_t *xi = (const int16_t *)cxi;
    int16_t *dsti = (int16_t *)cdsti;

    *dsti = *xi;
}

static void cpy_1_i32_i32(const char * cxi, char * cdsti) {
    const int32_t *xi = (const int32_t *)cxi;
    int32_t *dsti = (int32_t *)cdsti;

    *dsti = *xi;
}

template <cpy_kernel_t cpy_1>
static void cpy_f32_f16(const char * cx, char * cdst, const int ne,
                                   const int ne00, const int ne01, const int nb00, const int nb01, const int nb02,
                                   const int ne10, const int ne11, const int nb10, const int nb11, const int nb12,
                                   const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= ne) {
        return;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int i02 = i / (ne00*ne01);
    const int i01 = (i - i02*ne01*ne00) / ne00;
    const int i00 = i - i02*ne01*ne00 - i01*ne00;
    const int x_offset = i00*nb00 + i01*nb01 + i02*nb02;

    const int i12 = i / (ne10*ne11);
    const int i11 = (i - i12*ne10*ne11) / ne10;
    const int i10 = i - i12*ne10*ne11 - i11*ne10;
    const int dst_offset = i10*nb10 + i11*nb11 + i12*nb12;

    cpy_1(cx + x_offset, cdst + dst_offset);
}

static void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q8_0 * dsti = (block_q8_0 *) cdsti;

    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = xi[j];
        amax = sycl::fmax(amax, sycl::fabs((float)v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = xi[j]*id;

        dsti->qs[j] = sycl::round((float)x0);
    }
}

static void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q4_0 * dsti = (block_q4_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = xi[j];
        if (amax < sycl::fabs((float)v)) {
            amax = sycl::fabs((float)v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = xi[0       + j]*id;
        const float x1 = xi[QK4_0/2 + j]*id;

        const uint8_t xi0 = dpct::min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = dpct::min(15, (int8_t)(x1 + 8.5f));

        dsti->qs[j]  = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

static void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_q4_1 * dsti = (block_q4_1 *) cdsti;

    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = xi[j];

        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (xi[0       + j] - vmin)*id;
        const float x1 = (xi[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = dpct::min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = dpct::min(15, (int8_t)(x1 + 0.5f));

        dsti->qs[j]  = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

template <cpy_kernel_t cpy_blck, int qk>
static void cpy_f32_q(const char * cx, char * cdst, const int ne,
                                 const int ne00, const int ne01, const int nb00, const int nb01, const int nb02,
                                 const int ne10, const int ne11, const int nb10, const int nb11, const int nb12,
                                 const sycl::nd_item<3> &item_ct1) {
    const int i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                   item_ct1.get_local_id(2)) *
                  qk;

    if (i >= ne) {
        return;
    }

    const int i02 = i / (ne00*ne01);
    const int i01 = (i - i02*ne01*ne00) / ne00;
    const int i00 = (i - i02*ne01*ne00 - i01*ne00);
    const int x_offset = i00*nb00 + i01*nb01 + i02*nb02;

    const int i12 = i / (ne10*ne11);
    const int i11 = (i - i12*ne10*ne11) / ne10;
    const int i10 = (i - i12*ne10*ne11 - i11*ne10)/qk;
    const int dst_offset = i10*nb10 + i11*nb11 + i12*nb12;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / sycl::max(0.001f, high - low);
    return 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

struct rope_corr_dims {
    float v[4];
};

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, rope_corr_dims corr_dims, int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta
) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    *cos_theta = sycl::cos(theta) * mscale;
    *sin_theta = sycl::sin(theta) * mscale;
}

// rope == RoPE == rotary positional embedding
template<typename T, bool has_pos>
static void rope(
    const T * x, T * dst, int ncols, const int32_t * pos, float freq_scale, int p_delta_rows, float freq_base,
    float ext_factor, float attn_factor, rope_corr_dims corr_dims
,
    const sycl::nd_item<3> &item_ct1) {
    const int col = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                         item_ct1.get_local_id(1));

    if (col >= ncols) {
        return;
    }

    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    const int i = row*ncols + col;
    const int i2 = row/p_delta_rows;

    const int p = has_pos ? pos[i2] : 0;
    const float theta_base = p * dpct::pow(freq_base, -float(col) / ncols);

    float cos_theta, sin_theta;
    rope_yarn(theta_base, freq_scale, corr_dims, col, ext_factor, attn_factor, &cos_theta, &sin_theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0*cos_theta - x1*sin_theta;
    dst[i + 1] = x0*sin_theta + x1*cos_theta;
}

template<typename T, bool has_pos>
static void rope_neox(
    const T * x, T * dst, int ncols, int n_dims, const int32_t * pos, float freq_scale, int p_delta_rows,
    float ext_factor, float attn_factor, rope_corr_dims corr_dims, float theta_scale, float inv_ndims
,
    const sycl::nd_item<3> &item_ct1) {
    const int col = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                         item_ct1.get_local_id(1));

    if (col >= ncols) {
        return;
    }

    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    const int ib = col / n_dims;
    const int ic = col % n_dims;

    if (ib > 0) {
        const int i = row*ncols + ib*n_dims + ic;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int i  = row*ncols + ib*n_dims + ic/2;
    const int i2 = row/p_delta_rows;

    float cur_rot = inv_ndims * ic - ib;

    const int p = has_pos ? pos[i2] : 0;
    const float theta_base =
        p * freq_scale * dpct::pow(theta_scale, col / 2.0f);

    float cos_theta, sin_theta;
    rope_yarn(theta_base, freq_scale, corr_dims, cur_rot, ext_factor, attn_factor, &cos_theta, &sin_theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + n_dims/2];

    dst[i + 0]        = x0*cos_theta - x1*sin_theta;
    dst[i + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

static void rope_glm_f32(
    const float * x, float * dst, int ncols, const int32_t * pos, float freq_scale, int p_delta_rows, float freq_base,
    int n_ctx
, const sycl::nd_item<3> &item_ct1) {
    const int col = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    const int half_n_dims = ncols/4;

    if (col >= half_n_dims) {
        return;
    }

    const int row = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i = row*ncols + col;
    const int i2 = row/p_delta_rows;

    const float col_theta_scale = dpct::pow(freq_base, -2.0f * col / ncols);
     // FIXME: this is likely wrong
    const int p = pos != nullptr ? pos[i2] : 0;

    const float theta = sycl::min(p, n_ctx - 2) * freq_scale * col_theta_scale;
    const float sin_theta = sycl::sin((float)theta);
    const float cos_theta = sycl::cos((float)theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + half_n_dims];

    dst[i + 0]           = x0*cos_theta - x1*sin_theta;
    dst[i + half_n_dims] = x0*sin_theta + x1*cos_theta;

    const float block_theta =
        ((float)sycl::max(p - n_ctx - 2, 0)) * col_theta_scale;
    const float sin_block_theta = sycl::sin((float)block_theta);
    const float cos_block_theta = sycl::cos((float)block_theta);

    const float x2 = x[i + half_n_dims * 2];
    const float x3 = x[i + half_n_dims * 3];

    dst[i + half_n_dims * 2] = x2*cos_block_theta - x3*sin_block_theta;
    dst[i + half_n_dims * 3] = x2*sin_block_theta + x3*cos_block_theta;
}

static void alibi_f32(const float * x, float * dst, const int ncols, const int k_rows,
                                 const int n_heads_log2_floor, const float m0, const float m1,
                                 const sycl::nd_item<3> &item_ct1) {
    const int col = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    if (col >= ncols) {
        return;
    }

    const int row = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i = row*ncols + col;

    const int k = row/k_rows;

    float m_k;
    if (k < n_heads_log2_floor) {
        m_k = dpct::pow(m0, k + 1);
    } else {
        m_k = dpct::pow(m1, 2 * (k - n_heads_log2_floor) + 1);
    }

    dst[i] = col * m_k + x[i];
}

static void k_sum_rows_f32(const float * x, float * dst, const int ncols,
                           const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(1);
    const int col = item_ct1.get_local_id(2);

    float sum = 0.0f;
    for (int i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    if (col == 0) {
        dst[row] = sum;
    }
}

template<typename T>
static inline void swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<ggml_sort_order order>
static void k_argsort_f32_i32(const float * x, int * dst, const int ncols,
                              const sycl::nd_item<3> &item_ct1) {
    // bitonic sort
    int col = item_ct1.get_local_id(2);
    int row = item_ct1.get_group(1);

    if (col >= ncols) return;

    const float * x_row = x + row * ncols;
    int * dst_row = dst + row * ncols;

    // initialize indices
    if (col < ncols) {
        dst_row[col] = col;
    }
    /*
    DPCT1065:58: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (int k = 2; k <= ncols; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (order == GGML_SORT_ASC ? x_row[dst_row[col]] > x_row[dst_row[ixj]] : x_row[dst_row[col]] < x_row[dst_row[ixj]]) {
                        swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (order == GGML_SORT_ASC ? x_row[dst_row[col]] < x_row[dst_row[ixj]] : x_row[dst_row[col]] > x_row[dst_row[ixj]]) {
                        swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            /*
            DPCT1118:11: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:59: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
    }
}

static void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past,
                              const sycl::nd_item<3> &item_ct1) {
    const int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void soft_max_f32(const float * x, const float * y, float * dst, const int ncols, const int nrows_y, const float scale,
                         const sycl::nd_item<3> &item_ct1, float *buf) {
    const int tid = item_ct1.get_local_id(2);
    const int rowx = item_ct1.get_group(2);
    const int rowy = rowx % nrows_y; // broadcast the mask (y) in the row dimension

    const int block_size = item_ct1.get_local_range(2);

    const int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    const int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    float max_val = -INFINITY;

    for (int col = tid; col < ncols; col += block_size) {
        const int ix = rowx*ncols + col;
        const int iy = rowy*ncols + col;
        max_val = sycl::max(max_val, x[ix] * scale + (y ? y[iy] : 0.0f));
    }

    // find the max value in the block
    max_val = warp_reduce_max(max_val, item_ct1);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf[lane_id] = -INFINITY;
        }
        /*
        DPCT1118:12: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:60: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (lane_id == 0) {
            buf[warp_id] = max_val;
        }
        /*
        DPCT1118:13: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:61: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        max_val = buf[lane_id];
        max_val = warp_reduce_max(max_val, item_ct1);
    }

    float tmp = 0.f;

    for (int col = tid; col < ncols; col += block_size) {
        const int ix = rowx*ncols + col;
        const int iy = rowy*ncols + col;
        const float val =
            sycl::native::exp((x[ix] * scale + (y ? y[iy] : 0.0f)) - max_val);
        tmp += val;
        dst[ix] = val;
    }

    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf[lane_id] = 0.f;
        }
        /*
        DPCT1118:14: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:62: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if (lane_id == 0) {
            buf[warp_id] = tmp;
        }
        /*
        DPCT1118:15: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:63: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        tmp = buf[lane_id];
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float inv_tmp = 1.f / tmp;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = rowx*ncols + col;
        dst[i] *= inv_tmp;
    }
}

static void scale_f32(const float * x, float * dst, const float scale, const int k,
                      const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i];
}

static void clamp_f32(const float * x, float * dst, const float min, const float max, const int k,
                      const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = x[i] < min ? min : (x[i] > max ? max : x[i]);
}

static void im2col_f32_f16(const float *x, sycl::half *dst, int offset_delta,
                           int IW, int IH, int OW, int KW, int KH,
                           int pelements, int CHW, int s0, int s1, int p0,
                           int p1, int d0, int d1,
                           const sycl::nd_item<3> &item_ct1) {
    const int i = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (i >= pelements) {
        return;
    }

    const int ksize = OW * (KH > 1 ? KW : 1);
    const int kx = i / ksize;
    const int kd = kx * ksize;
    const int ky = (i - kd) / OW;
    const int ix = i % OW;

    const int64_t iiw = ix * s0 + kx * d0 - p0;
    const int64_t iih = item_ct1.get_group(1) * s1 + ky * d1 - p1;

    const int64_t offset_dst =
        (item_ct1.get_group(1) * OW + ix) * CHW +
        (item_ct1.get_group(0) * (KW * KH) + ky * KW + kx);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        dst[offset_dst] =
            sycl::vec<float, 1>(0.0f)
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    } else {
        const int64_t offset_src = item_ct1.get_group(0) * offset_delta;
        dst[offset_dst] =
            sycl::vec<float, 1>(x[offset_src + iih * IW + iiw])
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    }
}

template <int qk, int qr, dequantize_kernel_t dq>
static void get_rows_sycl(const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          dpct::queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                         });

    (void) dst;
}

template <typename src0_t>
static void get_rows_sycl_float(const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const src0_t *src0_dd, const int32_t *src1_dd,
                                float *dst_dd, dpct::queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + SYCL_GET_ROWS_BLOCK_SIZE - 1) / SYCL_GET_ROWS_BLOCK_SIZE;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                k_get_rows_float(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
            });
    }

    (void) dst;
}

template<float (*bin_op)(const float, const float)>
struct bin_bcast_sycl {
    template <typename src0_t, typename src1_t, typename dst_t>
    void operator()(const struct ggml_tensor *src0,
                    const struct ggml_tensor *src1, struct ggml_tensor *dst,
                    const src0_t *src0_dd, const src1_t *src1_dd, dst_t *dst_dd,
                    dpct::queue_ptr stream) {

        GGML_TENSOR_BINARY_OP_LOCALS

        int nr0 = ne10/ne0;
        int nr1 = ne11/ne1;
        int nr2 = ne12/ne2;
        int nr3 = ne13/ne3;

        int nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne0[] = {ne0, ne1, ne2, ne3};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};
        size_t cnb0[] = {nb0, nb1, nb2, nb3};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};
        auto collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        auto collapse_nb = [](size_t cnb[], int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        for (int i = 0; i < 4; i++) {
            if (nr[i] != 1) {
                break;
            }
            if (i > 0) {
                collapse_nb(cnb0, cne0);
                collapse_nb(cnb1, cne1);
                collapse(cne0);
                collapse(cne1);
            }
        }
        {
            int64_t ne0 = cne0[0];
            int64_t ne1 = cne0[1];
            int64_t ne2 = cne0[2];
            int64_t ne3 = cne0[3];

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb0[0];
            size_t nb1 = cnb0[1];
            size_t nb2 = cnb0[2];
            size_t nb3 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / sizeof(dst_t);
            size_t s2 = nb2 / sizeof(dst_t);
            size_t s3 = nb3 / sizeof(dst_t);

            size_t s10 = nb10 / sizeof(src1_t);
            size_t s11 = nb11 / sizeof(src1_t);
            size_t s12 = nb12 / sizeof(src1_t);
            size_t s13 = nb13 / sizeof(src1_t);

            GGML_ASSERT(s0 == 1);
            GGML_ASSERT(s10 == 1);

            const int block_size = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> block_dims(1, 1, 1);
            block_dims[2] = std::min<unsigned int>(hne0, block_size);
            block_dims[1] = std::min<unsigned int>(
                ne1, block_size / (unsigned int)block_dims[2]);
            block_dims[0] = std::min(
                std::min<unsigned int>(
                    ne2 * ne3, block_size / (unsigned int)block_dims[2] /
                                   (unsigned int)block_dims[1]),
                64U);

            sycl::range<3> block_nums(
                (ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                (ne1 + block_dims[1] - 1) / block_dims[1],
                (hne0 + block_dims[2] - 1) / block_dims[2]);

            if (block_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                int block_num = (ne0*ne1*ne2*ne3 + block_size - 1) / block_size;
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, block_num) *
                                              sycl::range<3>(1, 1, block_size),
                                          sycl::range<3>(1, 1, block_size)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_bin_bcast_unravel<bin_op>(
                                src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13, s1, s2, s3, s11, s12,
                                s13, item_ct1);
                        });
                }
            } else {
                /*
                DPCT1049:16: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_bin_bcast<bin_op>(src0_dd, src1_dd, dst_dd, ne0, ne1,
                                            ne2, ne3, ne10, ne11, ne12, ne13,
                                            s1, s2, s3, s11, s12, s13,
                                            item_ct1);
                    });
            }
        }
    }
};

static void acc_f32_sycl(const float *x, const float *y, float *dst,
                         const int n_elements, const int ne10, const int ne11,
                         const int ne12, const int nb1, const int nb2,
                         const int offset, dpct::queue_ptr stream) {
    int num_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            acc_f32(x, y, dst, n_elements, ne10, ne11, ne12, nb1, nb2, offset,
                    item_ct1);
        });
}

static void gelu_f32_sycl(const float *x, float *dst, const int k,
                          dpct::queue_ptr stream) {
    const int num_blocks = (k + SYCL_GELU_BLOCK_SIZE - 1) / SYCL_GELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_f32(x, dst, k, item_ct1);
        });
}

static void silu_f32_sycl(const float *x, float *dst, const int k,
                          dpct::queue_ptr stream) {
    const int num_blocks = (k + SYCL_SILU_BLOCK_SIZE - 1) / SYCL_SILU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SILU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SILU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            silu_f32(x, dst, k, item_ct1);
        });
}

static void gelu_quick_f32_sycl(const float *x, float *dst, const int k,
                                dpct::queue_ptr stream) {
    const int num_blocks = (k + SYCL_GELU_BLOCK_SIZE - 1) / SYCL_GELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_GELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_quick_f32(x, dst, k, item_ct1);
        });
}

static void tanh_f32_sycl(const float *x, float *dst, const int k,
                          dpct::queue_ptr stream) {
    const int num_blocks = (k + SYCL_TANH_BLOCK_SIZE - 1) / SYCL_TANH_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_TANH_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_TANH_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            tanh_f32(x, dst, k, item_ct1);
        });
}

static void relu_f32_sycl(const float *x, float *dst, const int k,
                          dpct::queue_ptr stream) {
    const int num_blocks = (k + SYCL_RELU_BLOCK_SIZE - 1) / SYCL_RELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            relu_f32(x, dst, k, item_ct1);
        });
}

static void leaky_relu_f32_sycl(const float *x, float *dst, const int k,
                                const float negative_slope,
                                dpct::queue_ptr stream) {
    const int num_blocks = (k + SYCL_RELU_BLOCK_SIZE - 1) / SYCL_RELU_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_RELU_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            leaky_relu_f32(x, dst, k, negative_slope, item_ct1);
        });
}

static void sqr_f32_sycl(const float *x, float *dst, const int k,
                         dpct::queue_ptr stream) {
    const int num_blocks = (k + SYCL_SQR_BLOCK_SIZE - 1) / SYCL_SQR_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SQR_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SQR_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            sqr_f32(x, dst, k, item_ct1);
        });
}

static void norm_f32_sycl(const float *x, float *dst, const int ncols,
                          const int nrows, const float eps,
                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<sycl::float2, 1> s_sum_acc_ct1(
                sycl::range<1>(32), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) * block_dims,
                                  block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                        norm_f32(x, dst, ncols, eps, item_ct1,
                                            s_sum_acc_ct1.get_pointer(), WARP_SIZE);
                    });
        });
    } else {
        const int work_group_size = g_work_group_size;
        const sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<sycl::float2, 1> s_sum_acc_ct1(
                sycl::range<1>(32), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) * block_dims,
                                  block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                        norm_f32(x, dst, ncols, eps, item_ct1,
                                       s_sum_acc_ct1.get_pointer(), work_group_size);
                    });
        });
    }
}

static void group_norm_f32_sycl(const float *x, float *dst,
                                const int num_groups, const int group_size,
                                const int ne_elements, dpct::queue_ptr stream) {
    static const float eps = 1e-6f;
    if (group_size < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(32),
                                                         cgh);

            const float eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_groups) * block_dims,
                                  block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                        group_norm_f32(
                            x, dst, group_size, ne_elements, eps_ct4, item_ct1,
                            s_sum_acc_ct1.get_pointer(), WARP_SIZE);
                    });
        });
    } else {
        const int work_group_size = g_work_group_size;
        const sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(32),
                                                         cgh);

            const float eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_groups) * block_dims,
                                  block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                        group_norm_f32(x, dst, group_size, ne_elements,
                                             eps_ct4, item_ct1,
                                             s_sum_acc_ct1.get_pointer(), work_group_size);
                    });
        });
    }
}

static void concat_f32_sycl(const float *x, const float *y, float *dst,
                            const int ne0, int ne1, int ne2, int ne02,
                            dpct::queue_ptr stream) {
    int num_blocks = (ne0 + SYCL_CONCAT_BLOCK_SIZE - 1) / SYCL_CONCAT_BLOCK_SIZE;
    sycl::range<3> gridDim(ne2, ne1, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(gridDim *
                              sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CONCAT_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            concat_f32(x, y, dst, ne0, ne02, item_ct1);
        });
}

static void upscale_f32_sycl(const float *x, float *dst, const int ne00,
                             const int ne01, const int ne02,
                             const int scale_factor, dpct::queue_ptr stream) {
    int ne0 = (ne00 * scale_factor);
    int num_blocks = (ne0 + SYCL_UPSCALE_BLOCK_SIZE - 1) / SYCL_UPSCALE_BLOCK_SIZE;
    sycl::range<3> gridDim(ne02, (ne01 * scale_factor), num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(gridDim *
                              sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            upscale_f32(x, dst, ne00, ne00 * ne01, scale_factor, item_ct1);
        });
}

static void pad_f32_sycl(const float *x, float *dst, const int ne00,
                         const int ne01, const int ne02, const int ne0,
                         const int ne1, const int ne2, dpct::queue_ptr stream) {
    int num_blocks = (ne0 + SYCL_PAD_BLOCK_SIZE - 1) / SYCL_PAD_BLOCK_SIZE;
    sycl::range<3> gridDim(ne2, ne1, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pad_f32(x, dst, ne0, ne00, ne01, ne02, item_ct1);
        });
}

static void rms_norm_f32_sycl(const float *x, float *dst, const int ncols,
                              const int nrows, const float eps,
                              dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(32),
                                                         cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) * block_dims,
                                  block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                        rms_norm_f32(x, dst, ncols, eps, item_ct1,
                                                s_sum_acc_ct1.get_pointer(), WARP_SIZE);
                    });
        });
    } else {
        const int work_group_size = g_work_group_size;
        const sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(32),
                                                         cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) * block_dims,
                                  block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                        rms_norm_f32(x, dst, ncols, eps, item_ct1,
                                           s_sum_acc_ct1.get_pointer(), work_group_size);
                    });
        });
    }
}

static void quantize_row_q8_1_sycl(const float *x, void *vy, const int kx,
                                   const int ky, const int kx_padded,
                                   dpct::queue_ptr stream) {
    const int block_num_x = (kx_padded + SYCL_QUANTIZE_BLOCK_SIZE - 1) / SYCL_QUANTIZE_BLOCK_SIZE;
    const sycl::range<3> num_blocks(1, ky, block_num_x);
    const sycl::range<3> block_size(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(num_blocks * block_size, block_size),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                quantize_q8_1(x, vy, kx, kx_padded, item_ct1);
            });
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_sycl(const void *__restrict__ vx,
                                  dst_t *__restrict__ y, const int k,
                                  dpct::queue_ptr stream) {
    const int num_blocks = (k + SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / SYCL_DEQUANTIZE_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, num_blocks) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

template <typename dst_t>
static void dequantize_row_q2_K_sycl(const void *vx, dst_t *y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }
#else
    dequantize_block_q2_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template <typename dst_t>
static void dequantize_row_q3_K_sycl(const void *vx, dst_t *y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
#else
    dequantize_block_q3_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template <typename dst_t>
static void dequantize_row_q4_K_sycl(const void *vx, dst_t *y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_K(vx, y, item_ct1);
                             });
    }
}

template <typename dst_t>
static void dequantize_row_q5_K_sycl(const void *vx, dst_t *y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q5_K(vx, y, item_ct1);
                             });
    }
#else
    dequantize_block_q5_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl(const void *vx, dst_t *y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }
#else
    dequantize_block_q6_K<<<nb, 32, 0, stream>>>(vx, y);
#endif
}

static to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_sycl;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_sycl;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_sycl;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_sycl;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_sycl;
        case GGML_TYPE_F32:
            return dequantize_block_sycl<1, 1, convert_f32>;
        default:
            return nullptr;
    }
}

static to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_sycl;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_sycl;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_sycl;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_sycl;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_sycl;
        case GGML_TYPE_F16:
            return dequantize_block_sycl<1, 1, convert_f16>;
        default:
            return nullptr;
    }
}

static void dequantize_mul_mat_vec_q4_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q4_1_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q5_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q5_1_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q8_0_sycl(const void *vx, const dfloat *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

static void dequantize_mul_mat_vec_q2_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, 32);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            dequantize_mul_mat_vec_q2_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q3_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, 32);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            dequantize_mul_mat_vec_q3_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q4_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, 32);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            dequantize_mul_mat_vec_q4_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q5_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const sycl::range<3> block_dims(1, 1, 32);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nrows) * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            dequantize_mul_mat_vec_q5_k(vx, y, dst, ncols, item_ct1);
        });
}

static void dequantize_mul_mat_vec_q6_K_sycl(const void *vx, const float *y,
                                             float *dst, const int ncols,
                                             const int nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int ny = 2 / K_QUANTS_PER_ITERATION;
    const int block_num_y = (nrows + ny - 1) / ny;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, ny, 32);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            dequantize_mul_mat_vec_q6_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

static void convert_mul_mat_vec_f16_sycl(const void *vx, const dfloat *y,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                dequantize_mul_mat_vec<1, 1, convert_f16>(vx, y, dst, ncols,
                                                          nrows, item_ct1);
            });
    }
}

static void mul_mat_vec_q4_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                          vec_dot_q4_0_q8_1>(vx, vy, dst, ncols, nrows,
                                             item_ct1);
        });
}

static void mul_mat_vec_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                          vec_dot_q4_1_q8_1>(vx, vy, dst, ncols, nrows,
                                             item_ct1);
        });
}

static void mul_mat_vec_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                          vec_dot_q5_0_q8_1>(vx, vy, dst, ncols, nrows,
                                             item_ct1);
        });
}

static void mul_mat_vec_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                          vec_dot_q5_1_q8_1>(vx, vy, dst, ncols, nrows,
                                             item_ct1);
        });
}

static void mul_mat_vec_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                          vec_dot_q8_0_q8_1>(vx, vy, dst, ncols, nrows,
                                             item_ct1);
        });
}

static void mul_mat_vec_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                          vec_dot_q2_K_q8_1>(vx, vy, dst, ncols, nrows,
                                             item_ct1);
        });
}

static void mul_mat_vec_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                          vec_dot_q3_K_q8_1>(vx, vy, dst, ncols, nrows,
                                             item_ct1);
        });
}

static void mul_mat_vec_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                          vec_dot_q4_K_q8_1>(vx, vy, dst, ncols, nrows,
                                             item_ct1);
        });
}

static void mul_mat_vec_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                          vec_dot_q5_K_q8_1>(vx, vy, dst, ncols, nrows,
                                             item_ct1);
        });
}

static void mul_mat_vec_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                          vec_dot_q6_K_q8_1>(vx, vy, dst, ncols, nrows,
                                             item_ct1);
        });
}

int get_device_index_by_id(int id){
    int res = g_sycl_device_id2index[id].index;
    // GGML_SYCL_DEBUG("get_device_index_by_id id=%d device_index=%d\n", id, res);
    GGML_ASSERT(res>=0);
    return res;
}

int get_device_id_by_index(int index){
    int res = g_device_caps[index].device_id;
    GGML_ASSERT(res>=0);
    return res;
}


int get_current_device_index(){
    return get_device_index_by_id(dpct::dev_mgr::instance().current_device_id());
}

static void ggml_mul_mat_q4_0_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    const int compute_capability = g_device_caps[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_0_RDNA2;
        mmq_y  =  MMQ_Y_Q4_0_RDNA2;
        nwarps = NWARPS_Q4_0_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_0_RDNA1;
        mmq_y  =  MMQ_Y_Q4_0_RDNA1;
        nwarps = NWARPS_Q4_0_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_0_AMPERE;
        mmq_y  =  MMQ_Y_Q4_0_AMPERE;
        nwarps = NWARPS_Q4_0_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_0_PASCAL;
        mmq_y  =  MMQ_Y_Q4_0_PASCAL;
        nwarps = NWARPS_Q4_0_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q4_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q4_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_0) + mmq_y / QI4_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_qs_q4_0_acc_ct1.get_pointer(),
                            tile_x_d_q4_0_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:21: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q4_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q4_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_0) + mmq_y / QI4_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_qs_q4_0_acc_ct1.get_pointer(),
                            tile_x_d_q4_0_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    const int compute_capability = g_device_caps[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_1_RDNA2;
        mmq_y  =  MMQ_Y_Q4_1_RDNA2;
        nwarps = NWARPS_Q4_1_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_1_RDNA1;
        mmq_y  =  MMQ_Y_Q4_1_RDNA1;
        nwarps = NWARPS_Q4_1_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_1_AMPERE;
        mmq_y  =  MMQ_Y_Q4_1_AMPERE;
        nwarps = NWARPS_Q4_1_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_1_PASCAL;
        mmq_y  =  MMQ_Y_Q4_1_PASCAL;
        nwarps = NWARPS_Q4_1_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q4_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + +mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_1) + mmq_y / QI4_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_qs_q4_1_acc_ct1.get_pointer(),
                            tile_x_dm_q4_1_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:23: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q4_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + +mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_1) + mmq_y / QI4_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_qs_q4_1_acc_ct1.get_pointer(),
                            tile_x_dm_q4_1_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    const int compute_capability = g_device_caps[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_0_RDNA2;
        mmq_y  =  MMQ_Y_Q5_0_RDNA2;
        nwarps = NWARPS_Q5_0_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_0_RDNA1;
        mmq_y  =  MMQ_Y_Q5_0_RDNA1;
        nwarps = NWARPS_Q5_0_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_0_AMPERE;
        mmq_y  =  MMQ_Y_Q5_0_AMPERE;
        nwarps = NWARPS_Q5_0_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_0_PASCAL;
        mmq_y  =  MMQ_Y_Q5_0_PASCAL;
        nwarps = NWARPS_Q5_0_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_0) + mmq_y / QI5_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_q5_0_acc_ct1.get_pointer(),
                            tile_x_d_q5_0_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:25: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q5_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_0) + mmq_y / QI5_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_q5_0_acc_ct1.get_pointer(),
                            tile_x_d_q5_0_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    const int compute_capability = g_device_caps[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_1_RDNA2;
        mmq_y  =  MMQ_Y_Q5_1_RDNA2;
        nwarps = NWARPS_Q5_1_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_1_RDNA1;
        mmq_y  =  MMQ_Y_Q5_1_RDNA1;
        nwarps = NWARPS_Q5_1_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_1_AMPERE;
        mmq_y  =  MMQ_Y_Q5_1_AMPERE;
        nwarps = NWARPS_Q5_1_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_1_PASCAL;
        mmq_y  =  MMQ_Y_Q5_1_PASCAL;
        nwarps = NWARPS_Q5_1_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:26: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_1) + mmq_y / QI5_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_q5_1_acc_ct1.get_pointer(),
                            tile_x_dm_q5_1_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:27: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_1_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_1) + mmq_y / QI5_1),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_1<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_q5_1_acc_ct1.get_pointer(),
                            tile_x_dm_q5_1_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    const int compute_capability = g_device_caps[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q8_0_RDNA2;
        mmq_y  =  MMQ_Y_Q8_0_RDNA2;
        nwarps = NWARPS_Q8_0_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q8_0_RDNA1;
        mmq_y  =  MMQ_Y_Q8_0_RDNA1;
        nwarps = NWARPS_Q8_0_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q8_0_AMPERE;
        mmq_y  =  MMQ_Y_Q8_0_AMPERE;
        nwarps = NWARPS_Q8_0_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q8_0_PASCAL;
        mmq_y  =  MMQ_Y_Q8_0_PASCAL;
        nwarps = NWARPS_Q8_0_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:28: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q8_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q8_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI8_0) + mmq_y / QI8_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q8_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_qs_q8_0_acc_ct1.get_pointer(),
                            tile_x_d_q8_0_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:29: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_qs_q8_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<float, 1> tile_x_d_q8_0_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI8_0) + mmq_y / QI8_0),
                    cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q8_0<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_qs_q8_0_acc_ct1.get_pointer(),
                            tile_x_d_q8_0_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    const int compute_capability = g_device_caps[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q2_K_RDNA2;
        mmq_y  =  MMQ_Y_Q2_K_RDNA2;
        nwarps = NWARPS_Q2_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q2_K_RDNA1;
        mmq_y  =  MMQ_Y_Q2_K_RDNA1;
        nwarps = NWARPS_Q2_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q2_K_AMPERE;
        mmq_y  =  MMQ_Y_Q2_K_AMPERE;
        nwarps = NWARPS_Q2_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q2_K_PASCAL;
        mmq_y  =  MMQ_Y_Q2_K_PASCAL;
        nwarps = NWARPS_Q2_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:30: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI2_K) + mmq_y / QI2_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q2_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_q2_K_acc_ct1.get_pointer(),
                            tile_x_dm_q2_K_acc_ct1.get_pointer(),
                            tile_x_sc_q2_K_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:31: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI2_K) + mmq_y / QI2_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q2_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q2_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_q2_K_acc_ct1.get_pointer(),
                            tile_x_dm_q2_K_acc_ct1.get_pointer(),
                            tile_x_sc_q2_K_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

#if QK_K == 256

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    const int compute_capability = g_device_caps[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q3_K_RDNA2;
        mmq_y  =  MMQ_Y_Q3_K_RDNA2;
        nwarps = NWARPS_Q3_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q3_K_RDNA1;
        mmq_y  =  MMQ_Y_Q3_K_RDNA1;
        nwarps = NWARPS_Q3_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q3_K_AMPERE;
        mmq_y  =  MMQ_Y_Q3_K_AMPERE;
        nwarps = NWARPS_Q3_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q3_K_PASCAL;
        mmq_y  =  MMQ_Y_Q3_K_PASCAL;
        nwarps = NWARPS_Q3_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:32: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI3_K) + mmq_y / QI3_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_qh_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 2) + mmq_y / 2), cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q3_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_q3_K_acc_ct1.get_pointer(),
                            tile_x_dm_q3_K_acc_ct1.get_pointer(),
                            tile_x_qh_q3_K_acc_ct1.get_pointer(),
                            tile_x_sc_q3_K_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:33: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI3_K) + mmq_y / QI3_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_qh_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 2) + mmq_y / 2), cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q3_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 4) + mmq_y / 4), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q3_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_q3_K_acc_ct1.get_pointer(),
                            tile_x_dm_q3_K_acc_ct1.get_pointer(),
                            tile_x_qh_q3_K_acc_ct1.get_pointer(),
                            tile_x_sc_q3_K_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    }
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    const int compute_capability = g_device_caps[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q4_K_RDNA2;
        mmq_y  =  MMQ_Y_Q4_K_RDNA2;
        nwarps = NWARPS_Q4_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q4_K_RDNA1;
        mmq_y  =  MMQ_Y_Q4_K_RDNA1;
        nwarps = NWARPS_Q4_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q4_K_AMPERE;
        mmq_y  =  MMQ_Y_Q4_K_AMPERE;
        nwarps = NWARPS_Q4_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q4_K_PASCAL;
        mmq_y  =  MMQ_Y_Q4_K_PASCAL;
        nwarps = NWARPS_Q4_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_K) + mmq_y / QI4_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_q4_K_acc_ct1.get_pointer(),
                            tile_x_dm_q4_K_acc_ct1.get_pointer(),
                            tile_x_sc_q4_K_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:35: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI4_K) + mmq_y / QI4_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q4_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q4_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_q4_K_acc_ct1.get_pointer(),
                            tile_x_dm_q4_K_acc_ct1.get_pointer(),
                            tile_x_sc_q4_K_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    const int compute_capability = g_device_caps[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q5_K_RDNA2;
        mmq_y  =  MMQ_Y_Q5_K_RDNA2;
        nwarps = NWARPS_Q5_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q5_K_RDNA1;
        mmq_y  =  MMQ_Y_Q5_K_RDNA1;
        nwarps = NWARPS_Q5_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q5_K_AMPERE;
        mmq_y  =  MMQ_Y_Q5_K_AMPERE;
        nwarps = NWARPS_Q5_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q5_K_PASCAL;
        mmq_y  =  MMQ_Y_Q5_K_PASCAL;
        nwarps = NWARPS_Q5_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:36: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_K) + mmq_y / QI5_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_q5_K_acc_ct1.get_pointer(),
                            tile_x_dm_q5_K_acc_ct1.get_pointer(),
                            tile_x_sc_q5_K_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:37: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI5_K) + mmq_y / QI5_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_q5_K_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q5_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_q5_K_acc_ct1.get_pointer(),
                            tile_x_dm_q5_K_acc_ct1.get_pointer(),
                            tile_x_sc_q5_K_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) try {

    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    const int compute_capability = g_device_caps[id].cc;

    int mmq_x, mmq_y, nwarps;
    if (compute_capability >= VER_GEN13) {
        mmq_x  =  MMQ_X_Q6_K_RDNA2;
        mmq_y  =  MMQ_Y_Q6_K_RDNA2;
        nwarps = NWARPS_Q6_K_RDNA2;
    } else if (compute_capability >= VER_GEN12) {
        mmq_x  =  MMQ_X_Q6_K_RDNA1;
        mmq_y  =  MMQ_Y_Q6_K_RDNA1;
        nwarps = NWARPS_Q6_K_RDNA1;
    } else if (compute_capability >= VER_GEN9) {
        mmq_x  =  MMQ_X_Q6_K_AMPERE;
        mmq_y  =  MMQ_Y_Q6_K_AMPERE;
        nwarps = NWARPS_Q6_K_AMPERE;
    } else if (compute_capability >= VER_4VEC) {
        mmq_x  =  MMQ_X_Q6_K_PASCAL;
        mmq_y  =  MMQ_Y_Q6_K_PASCAL;
        nwarps = NWARPS_Q6_K_PASCAL;
    } else {
        GGML_ASSERT(false);
    }

    const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
    const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

    if (nrows_x % mmq_y == 0) {
        const bool need_check = false;
        /*
        DPCT1049:38: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI6_K) + mmq_y / QI6_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q6_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_acc_ct1.get_pointer(),
                            tile_x_dm_acc_ct1.get_pointer(),
                            tile_x_sc_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    } else {
        const bool need_check = true;
        /*
        DPCT1049:39: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<int, 1> tile_x_ql_acc_ct1(
                    sycl::range<1>(mmq_y * (2 * WARP_SIZE) + mmq_y), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_x_dm_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / QI6_K) + mmq_y / QI6_K),
                    cgh);
                sycl::local_accessor<int, 1> tile_x_sc_acc_ct1(
                    sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
                sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE), cgh);
                sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
                    sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        mul_mat_q6_K<need_check>(
                            vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                            nrows_dst, item_ct1,
                            tile_x_ql_acc_ct1.get_pointer(),
                            tile_x_dm_acc_ct1.get_pointer(),
                            tile_x_sc_acc_ct1.get_pointer(),
                            tile_y_qs_acc_ct1.get_pointer(),
                            tile_y_ds_acc_ct1.get_pointer());
                    });
            });
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_mul_mat_p021_f16_f32_sycl(const void *vx, const float *y,
                                           float *dst, const int ncols_x,
                                           const int nrows_x,
                                           const int nchannels_x,
                                           const int nchannels_y,
                                           dpct::queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, item_ct1);
            });
    }
}

static void ggml_mul_mat_vec_nc_f16_f32_sycl(
    const void *vx, const float *y, float *dst, const int ncols_x,
    const int nrows_x, const int row_stride_x, const int nchannels_x,
    const int nchannels_y, const int channel_stride_x, dpct::queue_ptr stream) {

    const sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       row_stride_x, channel_stride_x,
                                       nchannels_y / nchannels_x, item_ct1);
            });
    }
}

static void ggml_cpy_f32_f32_sycl(const char *cx, char *cdst, const int ne,
                                  const int ne00, const int ne01,
                                  const int nb00, const int nb01,
                                  const int nb02, const int ne10,
                                  const int ne11, const int nb10,
                                  const int nb11, const int nb12,
                                  dpct::queue_ptr stream) {

    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, nb00, nb01,
                                           nb02, ne10, ne11, nb10, nb11, nb12,
                                           item_ct1);
            });
    }
}

static void ggml_cpy_f32_f16_sycl(const char *cx, char *cdst, const int ne,
                                  const int ne00, const int ne01,
                                  const int nb00, const int nb01,
                                  const int nb02, const int ne10,
                                  const int ne11, const int nb10,
                                  const int nb11, const int nb12,
                                  dpct::queue_ptr stream) {

    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, nb00, nb01,
                                           nb02, ne10, ne11, nb10, nb11, nb12,
                                           item_ct1);
            });
    }
}

static void ggml_cpy_f32_q8_0_sycl(const char *cx, char *cdst, const int ne,
                                   const int ne00, const int ne01,
                                   const int nb00, const int nb01,
                                   const int nb02, const int ne10,
                                   const int ne11, const int nb10,
                                   const int nb11, const int nb12,
                                   dpct::queue_ptr stream) {

    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ne / QK8_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks),
                                           sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(
                                 cx, cdst, ne, ne00, ne01, nb00, nb01, nb02,
                                 ne10, ne11, nb10, nb11, nb12, item_ct1);
                         });
}

static void ggml_cpy_f32_q4_0_sycl(const char *cx, char *cdst, const int ne,
                                   const int ne00, const int ne01,
                                   const int nb00, const int nb01,
                                   const int nb02, const int ne10,
                                   const int ne11, const int nb10,
                                   const int nb11, const int nb12,
                                   dpct::queue_ptr stream) {

    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ne / QK4_0;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks),
                                           sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(
                                 cx, cdst, ne, ne00, ne01, nb00, nb01, nb02,
                                 ne10, ne11, nb10, nb11, nb12, item_ct1);
                         });
}

static void ggml_cpy_f32_q4_1_sycl(const char *cx, char *cdst, const int ne,
                                   const int ne00, const int ne01,
                                   const int nb00, const int nb01,
                                   const int nb02, const int ne10,
                                   const int ne11, const int nb10,
                                   const int nb11, const int nb12,
                                   dpct::queue_ptr stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ne / QK4_1;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks),
                                           sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(
                                 cx, cdst, ne, ne00, ne01, nb00, nb01, nb02,
                                 ne10, ne11, nb10, nb11, nb12, item_ct1);
                         });
}

static void ggml_cpy_f16_f16_sycl(const char *cx, char *cdst, const int ne,
                                  const int ne00, const int ne01,
                                  const int nb00, const int nb01,
                                  const int nb02, const int ne10,
                                  const int ne11, const int nb10,
                                  const int nb11, const int nb12,
                                  dpct::queue_ptr stream) {

    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, nb00, nb01,
                                           nb02, ne10, ne11, nb10, nb11, nb12,
                                           item_ct1);
            });
    }
}

static void ggml_cpy_i16_i16_sycl(const char *cx, char *cdst, const int ne,
                                  const int ne00, const int ne01,
                                  const int nb00, const int nb01,
                                  const int nb02, const int ne10,
                                  const int ne11, const int nb10,
                                  const int nb11, const int nb12,
                                  dpct::queue_ptr stream) {

    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, nb00, nb01,
                                           nb02, ne10, ne11, nb10, nb11, nb12,
                                           item_ct1);
            });
    }
}

static void ggml_cpy_i32_i32_sycl(const char *cx, char *cdst, const int ne,
                                  const int ne00, const int ne01,
                                  const int nb00, const int nb01,
                                  const int nb02, const int ne10,
                                  const int ne11, const int nb10,
                                  const int nb11, const int nb12,
                                  dpct::queue_ptr stream) {

    const int num_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, nb00, nb01,
                                           nb02, ne10, ne11, nb10, nb11, nb12,
                                           item_ct1);
            });
    }
}

static void scale_f32_sycl(const float *x, float *dst, const float scale,
                           const int k, dpct::queue_ptr stream) {
    const int num_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, k, item_ct1);
        });
}

static void clamp_f32_sycl(const float *x, float *dst, const float min,
                           const float max, const int k,
                           dpct::queue_ptr stream) {
    const int num_blocks = (k + SYCL_CLAMP_BLOCK_SIZE - 1) / SYCL_CLAMP_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, SYCL_CLAMP_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CLAMP_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            clamp_f32(x, dst, min, max, k, item_ct1);
        });
}

template <typename T>
static void rope_sycl(const T *x, T *dst, int ncols, int nrows,
                      const int32_t *pos, float freq_scale, int p_delta_rows,
                      float freq_base, float ext_factor, float attn_factor,
                      rope_corr_dims corr_dims, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % 2 == 0);
    const sycl::range<3> block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int num_blocks_x = (ncols + 2*SYCL_ROPE_BLOCK_SIZE - 1) / (2*SYCL_ROPE_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, num_blocks_x, nrows);
    if (pos == nullptr) {
        /*
        DPCT1049:40: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                rope<T, false>(x, dst, ncols, pos, freq_scale, p_delta_rows,
                               freq_base, ext_factor, attn_factor, corr_dims,
                               item_ct1);
            });
    } else {
        /*
        DPCT1049:41: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                rope<T, true>(x, dst, ncols, pos, freq_scale, p_delta_rows,
                              freq_base, ext_factor, attn_factor, corr_dims,
                              item_ct1);
            });
    }
}

template <typename T>
static void rope_neox_sycl(const T *x, T *dst, int ncols, int n_dims, int nrows,
                           const int32_t *pos, float freq_scale,
                           int p_delta_rows, float freq_base, float ext_factor,
                           float attn_factor, rope_corr_dims corr_dims,
                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % 2 == 0);
    const sycl::range<3> block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int num_blocks_x = (ncols + 2*SYCL_ROPE_BLOCK_SIZE - 1) / (2*SYCL_ROPE_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, num_blocks_x, nrows);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);
    const float inv_ndims = -1.0f / n_dims;

    if (pos == nullptr) {
        /*
        DPCT1049:42: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                rope_neox<T, false>(x, dst, ncols, n_dims, pos, freq_scale,
                                    p_delta_rows, ext_factor, attn_factor,
                                    corr_dims, theta_scale, inv_ndims,
                                    item_ct1);
            });
    } else {
        /*
        DPCT1049:43: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                rope_neox<T, true>(x, dst, ncols, n_dims, pos, freq_scale,
                                   p_delta_rows, ext_factor, attn_factor,
                                   corr_dims, theta_scale, inv_ndims, item_ct1);
            });
    }
}

static void rope_glm_f32_sycl(const float *x, float *dst, int ncols, int nrows,
                              const int32_t *pos, float freq_scale,
                              int p_delta_rows, float freq_base, int n_ctx,
                              dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % 4 == 0);
    const sycl::range<3> block_dims(1, 1, SYCL_ROPE_BLOCK_SIZE / 4);
    const int num_blocks_x = (ncols + SYCL_ROPE_BLOCK_SIZE - 1) / SYCL_ROPE_BLOCK_SIZE;
    const sycl::range<3> block_nums(1, nrows, num_blocks_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             rope_glm_f32(x, dst, ncols, pos, freq_scale,
                                          p_delta_rows, freq_base, n_ctx,
                                          item_ct1);
                         });
}

static void alibi_f32_sycl(const float *x, float *dst, const int ncols,
                           const int nrows, const int k_rows,
                           const int n_heads_log2_floor, const float m0,
                           const float m1, dpct::queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, SYCL_ALIBI_BLOCK_SIZE);
    const int num_blocks_x = (ncols + SYCL_ALIBI_BLOCK_SIZE - 1) / (SYCL_ALIBI_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, nrows, num_blocks_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             alibi_f32(x, dst, ncols, k_rows,
                                       n_heads_log2_floor, m0, m1, item_ct1);
                         });
}

static void sum_rows_f32_sycl(const float *x, float *dst, const int ncols,
                              const int nrows, dpct::queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[intel::reqd_sub_group_size(32)]] {
                                 k_sum_rows_f32(x, dst, ncols, item_ct1);
                             });
}

static void argsort_f32_i32_sycl(const float *x, int *dst, const int ncols,
                                 const int nrows, ggml_sort_order order,
                                 dpct::queue_ptr stream) {
    // bitonic sort requires ncols to be power of 2
    GGML_ASSERT((ncols & (ncols - 1)) == 0);

    const sycl::range<3> block_dims(1, 1, ncols);
    const sycl::range<3> block_nums(1, nrows, 1);
    if (order == GGML_SORT_ASC) {
        /*
        DPCT1049:44: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                k_argsort_f32_i32<GGML_SORT_ASC>(x, dst, ncols, item_ct1);
            });
    } else if (order == GGML_SORT_DESC) {
        /*
        DPCT1049:45: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                k_argsort_f32_i32<GGML_SORT_DESC>(x, dst, ncols, item_ct1);
            });
    } else {
        GGML_ASSERT(false);
    }
}

static void diag_mask_inf_f32_sycl(const float *x, float *dst,
                                   const int ncols_x, const int nrows_x,
                                   const int rows_per_channel, const int n_past,
                                   dpct::queue_ptr stream) {
    const sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int block_num_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    const sycl::range<3> block_nums(1, block_num_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_mask_inf_f32(x, dst, ncols_x,
                                               rows_per_channel, n_past,
                                               item_ct1);
                         });
}

static void soft_max_f32_sycl(const float *x, const float *y, float *dst,
                              const int ncols_x, const int nrows_x,
                              const int nrows_y, const float scale,
                              dpct::queue_ptr stream) {
    int nth = WARP_SIZE;
    while (nth < ncols_x && nth < SYCL_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const sycl::range<3> block_dims(1, 1, nth);
    const sycl::range<3> block_nums(1, 1, nrows_x);
    /*
    DPCT1049:46: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    stream->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:96: 'SYCL_SOFT_MAX_BLOCK_SIZE/WARP_SIZE' expression was
        replaced with a value. Modify the code to use the original expression,
        provided in comments, if it is correct.
        */
        sycl::local_accessor<float, 1> buf_acc_ct1(
            sycl::range<1>(32 /*SYCL_SOFT_MAX_BLOCK_SIZE/WARP_SIZE*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                soft_max_f32(x, y, dst, ncols_x, nrows_y, scale, item_ct1,
                             buf_acc_ct1.get_pointer());
            });
    });
}

static void im2col_f32_f16_sycl(const float *x, sycl::half *dst, int IW, int IH,
                                int OW, int OH, int KW, int KH, int IC,
                                int offset_delta, int s0, int s1, int p0,
                                int p1, int d0, int d1,
                                dpct::queue_ptr stream) {
    const int parallel_elements = OW * KW * KH;
    const int num_blocks = (parallel_elements + SYCL_IM2COL_BLOCK_SIZE - 1) / SYCL_IM2COL_BLOCK_SIZE;
    sycl::range<3> block_nums(IC, OH, num_blocks);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums *
                                  sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_IM2COL_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                im2col_f32_f16(x, dst, offset_delta, IW, IH, OW, KW, KH,
                               parallel_elements, (IC * KH * KW), s0, s1, p0,
                               p1, d0, d1, item_ct1);
            });
    }
}

// buffer pool for sycl
#define MAX_SYCL_BUFFERS 256

struct scoped_spin_lock {
    std::atomic_flag& lock;
    scoped_spin_lock(std::atomic_flag& lock) : lock(lock) {
        while (lock.test_and_set(std::memory_order_acquire)) {
            ; // spin
        }
    }
    ~scoped_spin_lock() {
        lock.clear(std::memory_order_release);
    }
    scoped_spin_lock(const scoped_spin_lock&) = delete;
    scoped_spin_lock& operator=(const scoped_spin_lock&) = delete;
};

static std::atomic_flag g_sycl_pool_lock = ATOMIC_FLAG_INIT;

// #define DEBUG_SYCL_MALLOC
struct sycl_buffer {
    void * ptr = nullptr;
    size_t size = 0;
};

static sycl_buffer g_sycl_buffer_pool[GGML_SYCL_MAX_DEVICES][MAX_SYCL_BUFFERS];
static size_t g_sycl_pool_size[GGML_SYCL_MAX_DEVICES] = {0};

static void *ggml_sycl_pool_malloc_leg(size_t size, size_t *actual_size) try {
    scoped_spin_lock lock(g_sycl_pool_lock);
    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg index %d\n", id);
#ifdef DEBUG_SYCL_MALLOC
    int nnz = 0;
    size_t max_size = 0;
#endif
    size_t best_diff = 1ull << 36;
    int ibest = -1;
    for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
        sycl_buffer& b = g_sycl_buffer_pool[id][i];
        if (b.ptr != nullptr) {
#ifdef DEBUG_SYCL_MALLOC
            ++nnz;
            if (b.size > max_size) max_size = b.size;
#endif
            if (b.size >= size) {
                size_t diff = b.size - size;
                if (diff < best_diff) {
                    best_diff = diff;
                    ibest = i;
                    if (!best_diff) {
                        void * ptr = b.ptr;
                        *actual_size = b.size;
                        b.ptr = nullptr;
                        b.size = 0;
                        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg return 1 %p\n", ptr);
                        return ptr;
                    }
                }
            }
        }
    }
    if (ibest >= 0) {
        sycl_buffer& b = g_sycl_buffer_pool[id][ibest];
        void * ptr = b.ptr;
        *actual_size = b.size;
        b.ptr = nullptr;
        b.size = 0;
        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg return 2 %p\n", ptr);
        return ptr;
    }
    void * ptr;
    size_t look_ahead_size = (size_t) (1.05 * size);
    look_ahead_size = 256 * ((look_ahead_size + 255)/256);

    const dpct::queue_ptr stream = g_syclStreams[id][0];
    SYCL_CHECK(
        CHECK_TRY_ERROR(ptr = (void *)sycl::malloc_device(
                             look_ahead_size, *stream)));
    *actual_size = look_ahead_size;
    g_sycl_pool_size[id] += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
    fprintf(stderr, "%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
            (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif
    // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg return %p\n", ptr);
    return ptr;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_pool_free_leg(void *ptr, size_t size) try {
    scoped_spin_lock lock(g_sycl_pool_lock);
    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));

    const dpct::queue_ptr stream = g_syclStreams[id][0];
    for (int i = 0; i < MAX_SYCL_BUFFERS; ++i) {
        sycl_buffer& b = g_sycl_buffer_pool[id][i];
        if (b.ptr == nullptr) {
            b.ptr = ptr;
            b.size = size;
            return;
        }
    }
    fprintf(stderr, "WARNING: sycl buffer pool full, increase MAX_SYCL_BUFFERS\n");
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, *stream)));
    g_sycl_pool_size[id] -= size;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// pool with virtual memory
/*
DPCT1082:64: Migration of CUmemGenericAllocationHandle type is not supported.
*/
// static std::vector<CUmemGenericAllocationHandle>
//     g_sycl_pool_handles[GGML_SYCL_MAX_DEVICES];
static dpct::device_ptr g_sycl_pool_addr[GGML_SYCL_MAX_DEVICES] = {0};
static size_t g_sycl_pool_used[GGML_SYCL_MAX_DEVICES] = {0};
static const size_t SYCL_POOL_VMM_MAX_SIZE = 1ull << 36; // 64 GB

static void *ggml_sycl_pool_malloc_vmm(size_t size, size_t *actual_size) try {

    return NULL;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_pool_free_vmm(void *ptr, size_t size) try {
    scoped_spin_lock lock(g_sycl_pool_lock);
    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = dpct::dev_mgr::instance().current_device_id()));

#ifdef DEBUG_SYCL_MALLOC
    printf("sycl pool[%d]: freed %llu bytes at %llx\n", id, (unsigned long long) size, ptr);
#endif

    g_sycl_pool_used[id] -= size;

    // all deallocations must be in reverse order of the allocations
    GGML_ASSERT(ptr == (void *) (g_sycl_pool_addr[id] + g_sycl_pool_used[id]));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void *ggml_sycl_pool_malloc(size_t size, size_t *actual_size) try {
    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    if (g_device_caps[id].vmm) {
        return ggml_sycl_pool_malloc_vmm(size, actual_size);
    } else {
        return ggml_sycl_pool_malloc_leg(size, actual_size);
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_pool_free(void *ptr, size_t size) try {
    int id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));
    if (g_device_caps[id].vmm) {
        ggml_sycl_pool_free_vmm(ptr, size);
    } else {
        ggml_sycl_pool_free_leg(ptr, size);
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


template<typename T>
struct sycl_pool_alloc {
    T * ptr = nullptr;
    size_t actual_size = 0;

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) ggml_sycl_pool_malloc(size * sizeof(T), &this->actual_size);
        // GGML_SYCL_DEBUG("alloc %lu return %p actual size=%lu\n", size * sizeof(T), ptr, this->actual_size);
        return ptr;
    }

    sycl_pool_alloc(size_t size) {
        alloc(size);
    }

    ~sycl_pool_alloc() {
        if (ptr != nullptr) {
            ggml_sycl_pool_free(ptr, actual_size);
        }
    }

    T * get() {
        return ptr;
    }

    sycl_pool_alloc() = default;
    sycl_pool_alloc(const sycl_pool_alloc &) = delete;
    sycl_pool_alloc(sycl_pool_alloc &&) = delete;
    sycl_pool_alloc& operator=(const sycl_pool_alloc &) = delete;
    sycl_pool_alloc& operator=(sycl_pool_alloc &&) = delete;
};

static bool g_sycl_loaded = false;

bool ggml_sycl_loaded(void) {
    return g_sycl_loaded;
}

void ggml_backend_sycl_print_sycl_devices(){
    int device_count = dpct::dev_mgr::instance().device_count();
    fprintf(stderr, "found %d SYCL devices:\n", device_count);
    for (int id = 0; id < device_count; ++id) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(id))));
        sycl::device cur_device = dpct::dev_mgr::instance().get_device(id);
        fprintf(stderr, "  Device %d: %s,\tcompute capability %d.%d,\n\tmax compute_units %d,\tmax work group size %d,\tmax sub group size %d,\tglobal mem size %lu\n", id,
                prop.get_name(), prop.get_major_version(),
                prop.get_minor_version(),
                prop.get_max_compute_units(),
                prop.get_max_work_group_size(),
                prop.get_max_sub_group_size(),
                prop.get_global_mem_size()
                );
    }
    // fprintf(stderr, "\n");
}

int get_sycl_env(const char* env_name, int default_val){
    char * user_device_string = getenv(env_name);
    int user_number = default_val;

    unsigned n;
    if (user_device_string != NULL && sscanf(user_device_string, " %u", &n) == 1) {
            user_number = (int)n;
        } else {
            user_number=default_val;
        }
    return user_number;
}

int get_work_group_size(int user_device_id){
    dpct::device_info prop;
    dpct::get_device_info(
        prop,
        dpct::dev_mgr::instance().get_device(user_device_id));
    return prop.get_max_work_group_size();
}

void ggml_init_sycl() try {
    static bool initialized = false;

    if (!initialized) {
        g_ggml_sycl_debug = get_sycl_env("GGML_SYCL_DEBUG", 0);

        printf("GGML_SYCL_DEBUG=%d\n", g_ggml_sycl_debug);

        int user_device_id = get_sycl_env("GGML_SYCL_DEVICE", 0);

        if (CHECK_TRY_ERROR(g_all_sycl_device_count =
                                 dpct::dev_mgr::instance().device_count()) !=
            0) {
            initialized = true;
            g_sycl_loaded = false;
            return;
        }
        GGML_ASSERT(g_all_sycl_device_count <= GGML_SYCL_MAX_DEVICES);
        int64_t total_vram = 0;

#if defined(GGML_SYCL_FP16)
        fprintf(stderr, "%s: GGML_SYCL_FP16:   yes\n", __func__);
#else
        fprintf(stderr, "%s: GGML_SYCL_FP16:   no\n", __func__);
#endif


#if defined(SYCL_USE_XMX)
        fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
#else
        fprintf(stderr, "%s: SYCL_USE_XMX: no\n", __func__);
#endif
        ggml_backend_sycl_print_sycl_devices();
        for (int id = 0; id < GGML_SYCL_MAX_DEVICES; ++id) {
            g_sycl_device_id2index[id].index = -1;
            g_device_caps[id].vmm = 0;
            g_device_caps[id].device_id = -1;
            g_device_caps[id].cc = 0;
            g_tensor_split[id] = 0;
        }

        int device_inx = -1;
        for (int id = 0; id < g_all_sycl_device_count; ++id) {
            if(id!=user_device_id) continue;

            device_inx++;
            int device_vmm = 0;

            g_device_caps[device_inx].vmm = !!device_vmm;
            g_device_caps[device_inx].device_id = id;
            g_sycl_device_id2index[id].index = device_inx;

            dpct::device_info prop;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                prop, dpct::dev_mgr::instance().get_device(id))));

            // fprintf(stderr,
            //         "  Device %d: %s, compute capability %d.%d, VMM: %s\n", id,
            //         prop.get_name(), prop.get_major_version(),
            //         prop.get_minor_version(), device_vmm ? "yes" : "no");

            g_tensor_split[device_inx] = total_vram;
            total_vram += prop.get_global_mem_size();

            g_device_caps[device_inx].cc =
                100 * prop.get_major_version() + 10 * prop.get_minor_version();

            // printf("g_device_caps[%d].cc=%d\n", device_inx, g_device_caps[device_inx].cc);
        }
        device_inx = -1;
        for (int id = 0; id < g_all_sycl_device_count; ++id) {
            if(id!=user_device_id) continue;
            device_inx++;
            g_tensor_split[device_inx] /= total_vram;
        }

        device_inx = -1;
        for (int id = 0; id < g_all_sycl_device_count; ++id) {
            if(id!=user_device_id) continue;
            device_inx++;
            SYCL_CHECK(ggml_sycl_set_device(id));

            // create sycl streams
            for (int is = 0; is < MAX_STREAMS; ++is) {
                /*
                DPCT1025:88: The SYCL queue is created ignoring the flag and
                priority options.
                */
                SYCL_CHECK(CHECK_TRY_ERROR(
                    g_syclStreams[device_inx][is] =
                        dpct::get_current_device().create_queue()));
            }

            const dpct::queue_ptr stream = g_syclStreams[device_inx][0];
            // create sycl handle
            SYCL_CHECK(CHECK_TRY_ERROR(g_sycl_handles[device_inx] =
                                              stream));
            /*
            DPCT1027:89: The call to syclSetMathMode was replaced with 0
            because this functionality is redundant in SYCL.
            */
            SYCL_CHECK(0);
        }

        // configure logging to stdout
        // SYCL_CHECK(syclLoggerConfigure(1, 1, 0, nullptr));

        //hardcode, force set to 1 device
        g_device_count = 1;
        ggml_sycl_set_main_device(user_device_id);
        ggml_sycl_set_device(user_device_id);
        g_work_group_size = get_work_group_size(user_device_id);
        // fprintf(stderr, "Using Device %d\n", user_device_id);

        // for (int id = 0; id < g_all_sycl_device_count; ++id) {
        //     GGML_SYCL_DEBUG("id=%d  g_device_caps[%d].device_id=%d g_sycl_device_id2index[%d].index=%d ", id, id,
        //     g_device_caps[id].device_id, id, g_sycl_device_id2index[id].index);
        // }

        initialized = true;
        g_sycl_loaded = true;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


void ggml_sycl_set_tensor_split(const float * tensor_split) {
    if (tensor_split == nullptr) {
        return;
    }
    bool all_zero = true;
    for (int i = 0; i < g_device_count; ++i) {
        if (tensor_split[i] != 0.0f) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return;
    }
    float split_sum = 0.0f;
    for (int i = 0; i < g_device_count; ++i) {
        g_tensor_split[i] = split_sum;
        split_sum += tensor_split[i];
    }
    for (int i = 0; i < g_device_count; ++i) {
        g_tensor_split[i] /= split_sum;
    }
}

void *ggml_sycl_host_malloc(size_t size) try {
    if (getenv("GGML_SYCL_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    //allow to use dpct::get_in_order_queue() for host malloc
    dpct::err0 err = CHECK_TRY_ERROR(
        ptr = (void *)sycl::malloc_host(size, dpct::get_in_order_queue()));
    /*
    DPCT1000:82: Error handling if-stmt was detected but could not be rewritten.
    */
    if (err != 0) {
        // clear the error
        /*
        DPCT1026:83: The call to syclGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        /*
        DPCT1001:81: The statement could not be removed.
        */
        fprintf(
            stderr,
            "WARNING: failed to allocate %.2f MB of pinned memory: %s\n",
            /*
            DPCT1009:84: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            size / 1024.0 / 1024.0,
            "syclGetErrorString is not supported" /*syclGetErrorString(err)*/);
        return nullptr;
    }

    return ptr;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_host_free(void *ptr) try {
    //allow to use dpct::get_in_order_queue() for host malloc
    SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ptr, dpct::get_in_order_queue())));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static dpct::err0 ggml_sycl_cpy_tensor_2d(void *dst,
                                          const struct ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_low, int64_t i1_high,
                                          dpct::queue_ptr stream) try {

    dpct::memcpy_direction kind;
    char * src_ptr;
    if (src->backend == GGML_BACKEND_CPU) {
        kind = dpct::host_to_device;
        src_ptr = (char *) src->data;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_CPU src_ptr %p\n", src_ptr);
    } else if (src->backend == GGML_BACKEND_GPU || src->backend == GGML_BACKEND_GPU_SPLIT) {
        GGML_ASSERT(src->backend != GGML_BACKEND_GPU_SPLIT || (i1_low == 0 && i1_high == src->ne[1]));
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_index()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (char *) extra->data_device[id];
    } else {
        // GGML_SYCL_DEBUG("GGML_ASSERT(false)\n");
        GGML_ASSERT(false);
    }
    char * dst_ptr = (char *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        return CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } else if (nb0 == ts) {
        return CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, kind, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            if (r != 0) return r;
        }
        return 0;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_op_get_rows(const ggml_tensor *src0,
                                  const ggml_tensor *src1, ggml_tensor *dst,
                                  const float *src0_d, const float *src1_d,
                                  float *dst_d, const dpct::queue_ptr &stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    const int32_t * src1_i32 = (const int32_t *) src1_d;

    switch (src0->type) {
        case GGML_TYPE_F16:
            get_rows_sycl_float(src0, src1, dst, (const sycl::half *)src0_d,
                                src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_sycl_float(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl<QK4_1, QR4_1, dequantize_q4_1>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl<QK5_0, QR5_0, dequantize_q5_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl<QK5_1, QR5_1, dequantize_q5_1>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        default:
            // TODO: k-quants
            fprintf(stderr, "%s: unsupported type: %s\n", __func__, ggml_type_name(src0->type));
            GGML_ASSERT(false);
            break;
    }
}

template <class op>
inline void ggml_sycl_op_bin_bcast(const ggml_tensor *src0,
                                   const ggml_tensor *src1, ggml_tensor *dst,
                                   const float *src0_dd, const float *src1_dd,
                                   float *dst_dd,
                                   const dpct::queue_ptr &main_stream) {

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, src0_dd, src1_dd, dst_dd, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()(src0, src1, dst, (const sycl::half *)src0_dd, src1_dd,
             (sycl::half *)dst_dd, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const sycl::half *)src0_dd, src1_dd, dst_dd,
             main_stream);
    } else if (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()(src0, src1, dst, (const int32_t *)src0_dd, (const int32_t *)src1_dd, (int32_t *)dst_dd,
             main_stream);
    } else if (src0->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()(src0, src1, dst, (const int16_t *)src0_dd, (const int16_t *)src1_dd, (int16_t *)dst_dd,
             main_stream);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ASSERT(false);
    }
}

static void ggml_sycl_op_repeat(const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const float *src0_d, const float *src1_d,
                                float *dst_d,
                                const dpct::queue_ptr &main_stream) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(dst, src0, dst, nullptr, src0_d, dst_d, main_stream);

    (void) src1;
    (void) src1_d;
}

inline void ggml_sycl_op_add(const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const dpct::queue_ptr &main_stream) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_add>>(src0, src1, dst, src0_dd, src1_dd, dst_dd, main_stream);
}

inline void ggml_sycl_op_acc(const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[3] == 1); // just 3D tensors supported

    int nb1 = dst->op_params[0] / 4; // 4 bytes of float32
    int nb2 = dst->op_params[1] / 4; // 4 bytes of float32
    // int nb3 = dst->op_params[2] / 4; // 4 bytes of float32 - unused
    int offset = dst->op_params[3] / 4; // offset in bytes

    acc_f32_sycl(src0_dd, src1_dd, dst_dd, ggml_nelements(dst), src1->ne[0], src1->ne[1], src1->ne[2], nb1, nb2, offset, main_stream);

    (void) dst;
}

inline void ggml_sycl_op_mul(const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const dpct::queue_ptr &main_stream) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(src0, src1, dst, src0_dd, src1_dd, dst_dd, main_stream);
}

inline void ggml_sycl_op_div(const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const dpct::queue_ptr &main_stream) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(src0, src1, dst, src0_dd, src1_dd, dst_dd, main_stream);
}

inline void ggml_sycl_op_gelu(const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_silu(const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    silu_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_gelu_quick(const ggml_tensor *src0,
                                    const ggml_tensor *src1, ggml_tensor *dst,
                                    const float *src0_dd, const float *src1_dd,
                                    float *dst_dd,
                                    const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_quick_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_tanh(const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    tanh_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_relu(const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    relu_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_leaky_relu(const ggml_tensor *src0,
                                    const ggml_tensor *src1, ggml_tensor *dst,
                                    const float *src0_dd, const float *src1_dd,
                                    float *dst_dd,
                                    const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    leaky_relu_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), negative_slope, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_sqr(const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sqr_f32_sycl(src0_dd, dst_dd, ggml_nelements(src0), main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_norm(const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    norm_f32_sycl(src0_dd, dst_dd, ne00, nrows, eps, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_group_norm(const ggml_tensor *src0,
                                    const ggml_tensor *src1, ggml_tensor *dst,
                                    const float *src0_dd, const float *src1_dd,
                                    float *dst_dd,
                                    const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];
    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_sycl(src0_dd, dst_dd, num_groups, group_size, src0->ne[0] * src0->ne[1] * src0->ne[2], main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_concat(const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const float *src0_dd, const float *src1_dd,
                                float *dst_dd,
                                const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    for (int i3 = 0; i3 < dst->ne[3]; i3++) {
        concat_f32_sycl(src0_dd + i3 * (src0->nb[3] / 4), src1_dd + i3 * (src1->nb[3] / 4), dst_dd + i3 * (dst->nb[3] / 4), dst->ne[0], dst->ne[1], dst->ne[2], src0->ne[2], main_stream);
    }

    (void) src1;
    (void) dst;
}

inline void ggml_sycl_op_upscale(const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 const float *src0_dd, const float *src1_dd,
                                 float *dst_dd,
                                 const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

    const int scale_factor = dst->op_params[0];

    upscale_f32_sycl(src0_dd, dst_dd, src0->ne[0], src0->ne[1], src0->ne[2], scale_factor, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_pad(const ggml_tensor *src0, const ggml_tensor *src1,
                             ggml_tensor *dst, const float *src0_dd,
                             const float *src1_dd, float *dst_dd,
                             const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

    pad_f32_sycl(src0_dd, dst_dd,
        src0->ne[0], src0->ne[1], src0->ne[2],
        dst->ne[0], dst->ne[1], dst->ne[2], main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_rms_norm(const ggml_tensor *src0,
                                  const ggml_tensor *src1, ggml_tensor *dst,
                                  const float *src0_dd, const float *src1_dd,
                                  float *dst_dd,
                                  const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, nrows, eps, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_mul_mat_q(
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream) try {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int device_id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(device_id = dpct::dev_mgr::instance().current_device_id()));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the dequantize_mul_mat kernel writes into
    const int64_t nrows_dst = dst->backend == GGML_BACKEND_GPU && device_id == g_main_device ? ne0 : row_diff;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            ggml_mul_mat_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_1:
            ggml_mul_mat_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_0:
            ggml_mul_mat_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_1:
            ggml_mul_mat_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q2_K:
            ggml_mul_mat_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q3_K:
            ggml_mul_mat_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_K:
            ggml_mul_mat_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_K:
            ggml_mul_mat_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    (void) src1;
    (void) dst;
    (void) src1_ddf_i;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static int64_t get_row_rounding(ggml_type type) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    for (int64_t id = 0; id < g_device_count; ++id) {
        if (g_tensor_split[id] < (id + 1 < g_device_count ? g_tensor_split[id + 1] : 1.0f)) {
            if (min_compute_capability > g_device_caps[id].cc) {
                min_compute_capability = g_device_caps[id].cc;
            }
            if (max_compute_capability < g_device_caps[id].cc) {
                max_compute_capability = g_device_caps[id].cc;
            }
        }
    }

    switch(type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ASSERT(false);
    }
}

inline void ggml_sycl_op_mul_mat_vec_q(
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream) {

    GGML_ASSERT(ggml_nrows(src1) == 1);

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    (void) src1;
    (void) dst;
    (void) src1_ddf_i;
    (void) src1_ncols;
    (void) src1_padded_row_size;
}

inline void ggml_sycl_op_dequantize_mul_mat_vec(
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ifdef GGML_SYCL_F16
    sycl_pool_alloc<sycl::half> src1_dfloat_a;
    sycl::half *src1_dfloat = nullptr; // dfloat == half

    bool src1_convert_f16 =
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_F16;

    if (src1_convert_f16) {
        if (src1->type == GGML_TYPE_F16) {
            src1_dfloat = (sycl::half *)src1->data + src1_padded_row_size;
        } else {
            src1_dfloat = src1_dfloat_a.alloc(ne00);
            ggml_cpy_f32_f16_sycl((const char *)src1_ddf_i, (char *)src1_dfloat,
                                  ne00, ne00, 1, sizeof(float), 0, 0, ne00, 1,
                                  sizeof(sycl::half), 0, 0, stream);
        }
    }
#else
    const dfloat * src1_dfloat = (const dfloat *) src1_ddf_i; // dfloat == float, no conversion
#endif // GGML_SYCL_F16

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            dequantize_mul_mat_vec_q4_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_mul_mat_vec_q8_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_q2_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_q3_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_q4_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_q5_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_q6_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            break;
        case GGML_TYPE_F16:
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }

    (void) src1;
    (void) dst;
    (void) src1_ddq_i;
    (void) src1_ncols;
    (void) src1_padded_row_size;
}

inline void ggml_sycl_op_mul_mat_sycl(
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream) try {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id;
    int device_id = dpct::dev_mgr::instance().current_device_id();
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_index()));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = dst->backend == GGML_BACKEND_GPU && device_id == g_main_device ? ne0 : row_diff;

    const int compute_capability = g_device_caps[id].cc;
#ifdef GGML_SYCL_F16
    bool use_fp16 = true;  // TODO(Yu) SYCL capability check
#else
    bool use_fp16 = false;
#endif
    // if (compute_capability >= VER_GEN9 && (src0->type == GGML_TYPE_F16 ||
    // ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff ==
    // src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
        use_fp16 && ggml_is_contiguous(src0) && row_diff == src0->ne[1] &&
        dst->op_params[0] == GGML_PREC_DEFAULT) {

        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        // GGML_SYCL_DEBUG("ggml_sycl_op_mul_mat_sycl - fp16 path\n");
        sycl_pool_alloc<sycl::half> src0_as_f16;
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (const sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

        sycl_pool_alloc<sycl::half> src1_as_f16;
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (const sycl::half *)src1->data + src1_padded_row_size
                                         : src1_as_f16.get();
        sycl_pool_alloc<sycl::half> dst_f16(row_diff * src1_ncols);

        const sycl::half alpha_f16 = 1.0f;
        const sycl::half beta_f16 = 0.0f;

        SYCL_CHECK(CHECK_TRY_ERROR(g_sycl_handles[id] = stream));
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
            *g_sycl_handles[id], oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
            &alpha_f16, src0_ptr, dpct::library_data_t::real_half, ne00,
            src1_ptr, dpct::library_data_t::real_half, ne10, &beta_f16,
            dst_f16.get(), dpct::library_data_t::real_half, ldc,
            dpct::library_data_t::real_half)));

        const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(GGML_TYPE_F16);
        to_fp32_sycl(dst_f16.get(), dst_dd_i, row_diff*src1_ncols, stream);
    }
    else {
        // GGML_SYCL_DEBUG("ggml_sycl_op_mul_mat_sycl - fp32 path\n");
        sycl_pool_alloc<float> src0_ddq_as_f32;

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        SYCL_CHECK(CHECK_TRY_ERROR(g_sycl_handles[id] = stream));
        SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
            *g_sycl_handles[id], oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
            dpct::get_value(&alpha, *g_sycl_handles[id]), src0_ddf_i, ne00,
            src1_ddf_i, ne10, dpct::get_value(&beta, *g_sycl_handles[id]),
            dst_dd_i, ldc)));
    }

    (void) dst;
    (void) src1_ddq_i;
    (void) src1_padded_row_size;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

inline void ggml_sycl_op_rope(const ggml_tensor *src0, const ggml_tensor *src1,
                              ggml_tensor *dst, const float *src0_dd,
                              const float *src1_dd, float *dst_dd,
                              const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t nrows = ggml_nrows(src0);

    //const int n_past      = ((int32_t *) dst->op_params)[0];
    const int n_dims      = ((int32_t *) dst->op_params)[1];
    const int mode        = ((int32_t *) dst->op_params)[2];
    const int n_ctx       = ((int32_t *) dst->op_params)[3];
    const int n_orig_ctx  = ((int32_t *) dst->op_params)[4];

    // RoPE alteration for extended context
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    const int32_t * pos = nullptr;
    if ((mode & 1) == 0) {
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        GGML_ASSERT(src1->ne[0] == ne2);
        pos = (const int32_t *) src1_dd;
    }

    const bool is_neox = mode & 2;
    const bool is_glm  = mode & 4;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_orig_ctx, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_glm) {
        GGML_ASSERT(false);
        rope_glm_f32_sycl(src0_dd, dst_dd, ne00, nrows, pos, freq_scale, ne01, freq_base, n_ctx, main_stream);
    } else if (is_neox) {
        if (src0->type == GGML_TYPE_F32) {
            rope_neox_sycl(
                (const float *)src0_dd, (float *)dst_dd, ne00, n_dims, nrows, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, main_stream
            );
        } else if (src0->type == GGML_TYPE_F16) {
            rope_neox_sycl((const sycl::half *)src0_dd, (sycl::half *)dst_dd,
                           ne00, n_dims, nrows, pos, freq_scale, ne01,
                           freq_base, ext_factor, attn_factor, corr_dims,
                           main_stream);
        } else {
            GGML_ASSERT(false);
        }
    } else {
        if (src0->type == GGML_TYPE_F32) {
            rope_sycl(
                (const float *)src0_dd, (float *)dst_dd, ne00, nrows, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, main_stream
            );
        } else if (src0->type == GGML_TYPE_F16) {
            rope_sycl((const sycl::half *)src0_dd, (sycl::half *)dst_dd, ne00,
                      nrows, pos, freq_scale, ne01, freq_base, ext_factor,
                      attn_factor, corr_dims, main_stream);
        } else {
            GGML_ASSERT(false);
        }
    }

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_alibi(const ggml_tensor *src0, const ggml_tensor *src1,
                               ggml_tensor *dst, const float *src0_dd,
                               const float *src1_dd, float *dst_dd,
                               const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t nrows = ggml_nrows(src0);

    //const int n_past = ((int32_t *) dst->op_params)[0];
    const int n_head = ((int32_t *) dst->op_params)[1];
    float max_bias;
    memcpy(&max_bias, (int32_t *) dst->op_params + 2, sizeof(float));

    //GGML_ASSERT(ne01 + n_past == ne00);
    GGML_ASSERT(n_head == ne02);

    const int n_heads_log2_floor = 1 << (int) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

    alibi_f32_sycl(src0_dd, dst_dd, ne00, nrows, ne01, n_heads_log2_floor, m0, m1, main_stream);

    (void) src1;
    (void) src1_dd;
}

inline void ggml_sycl_op_im2col(const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const float *src0_dd, const float *src1_dd,
                                float *dst_dd,
                                const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const int64_t IC = src1->ne[is_2D ? 2 : 1];
    const int64_t IH = is_2D ? src1->ne[1] : 1;
    const int64_t IW =         src1->ne[0];

    const int64_t KH = is_2D ? src0->ne[1] : 1;
    const int64_t KW =         src0->ne[0];

    const int64_t OH = is_2D ? dst->ne[2] : 1;
    const int64_t OW =         dst->ne[1];

    const size_t delta_offset = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32

    im2col_f32_f16_sycl(src1_dd, (sycl::half *)dst_dd, IW, IH, OW, OH, KW, KH,
                        IC, delta_offset, s0, s1, p0, p1, d0, d1, main_stream);

    (void) src0;
    (void) src0_dd;
}

inline void ggml_sycl_op_sum_rows(const ggml_tensor *src0,
                                  const ggml_tensor *src1, ggml_tensor *dst,
                                  const float *src0_dd, const float *src1_dd,
                                  float *dst_dd,
                                  const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_argsort(const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 const float *src0_dd, const float *src1_dd,
                                 float *dst_dd,
                                 const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (int *)dst_dd, ncols, nrows, order, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_diag_mask_inf(const ggml_tensor *src0,
                                       const ggml_tensor *src1,
                                       ggml_tensor *dst, const float *src0_dd,
                                       const float *src1_dd, float *dst_dd,
                                       const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int nrows0 = ggml_nrows(src0);

    const int n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_soft_max(const ggml_tensor *src0,
                                  const ggml_tensor *src1, ggml_tensor *dst,
                                  const float *src0_dd, const float *src1_dd,
                                  float *dst_dd,
                                  const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src1 ? ggml_nrows(src1) : 1;

    float scale = 1.0f;
    memcpy(&scale, dst->op_params, sizeof(float));

    soft_max_f32_sycl(src0_dd, src1 ? src1_dd : nullptr, dst_dd, ne00, nrows_x, nrows_y, scale, main_stream);

    (void) dst;
}

inline void ggml_sycl_op_scale(const ggml_tensor *src0, const ggml_tensor *src1,
                               ggml_tensor *dst, const float *src0_dd,
                               const float *src1_dd, float *dst_dd,
                               const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    scale_f32_sycl(src0_dd, dst_dd, scale, ggml_nelements(src0), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

inline void ggml_sycl_op_clamp(const ggml_tensor *src0, const ggml_tensor *src1,
                               ggml_tensor *dst, const float *src0_dd,
                               const float *src1_dd, float *dst_dd,
                               const dpct::queue_ptr &main_stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    clamp_f32_sycl(src0_dd, dst_dd, min, max, ggml_nelements(src0), main_stream);
    /*
    DPCT1010:88: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);

    (void) src1;
    (void) dst;
    (void) src1_dd;
}

static void ggml_sycl_op_flatten(const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 const ggml_sycl_op_flatten_t op) try {
    const int64_t nrows0 = ggml_nrows(src0);

    const bool use_src1 = src1 != nullptr;
    const int64_t nrows1 = use_src1 ? ggml_nrows(src1) : 1;

    GGML_ASSERT(!use_src1 || src1->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(              dst->backend != GGML_BACKEND_GPU_SPLIT);

    ggml_tensor_extra_gpu * src0_extra =            (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * src1_extra = use_src1 ? (ggml_tensor_extra_gpu *) src1->extra : nullptr;
    ggml_tensor_extra_gpu * dst_extra  =            (ggml_tensor_extra_gpu *)  dst->extra;

    const bool src0_on_device =             src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT;
    const bool src1_on_device = use_src1 && src1->backend == GGML_BACKEND_GPU;
    const bool  dst_on_device =              dst->backend == GGML_BACKEND_GPU;

    // dd = data device
    float * src0_ddf = nullptr;
    float * src1_ddf = nullptr;
    float *  dst_ddf = nullptr;

    sycl_pool_alloc<float> src0_f;
    sycl_pool_alloc<float> src1_f;
    sycl_pool_alloc<float>  dst_f;

    ggml_sycl_set_device(g_main_device);
    dpct::queue_ptr main_stream = g_syclStreams[g_main_device_index][0];
    // GGML_SYCL_DEBUG("g_main_device_index=%d, main_stream=%p src0_on_device=%d, src1_on_device=%d, dst_on_device=%d\n",
        // g_main_device_index, main_stream, src0_on_device, src1_on_device, dst_on_device);

    if (src0_on_device) {
        src0_ddf = (float *) src0_extra->data_device[g_main_device_index];
    } else {
        src0_ddf = src0_f.alloc(ggml_nelements(src0));
        // GGML_SYCL_DEBUG("before ggml_sycl_cpy_tensor_2d src0_ddf=%p, src0=%p\n", src0_ddf, src0);
        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_ddf, src0, 0, 0, 0, nrows0, main_stream));
    }

    if (use_src1) {
        if (src1_on_device) {
            src1_ddf = (float *) src1_extra->data_device[g_main_device_index];
        } else {
            src1_ddf = src1_f.alloc(ggml_nelements(src1));
            SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf, src1, 0, 0, 0, nrows1, main_stream));
        }
    }
    if (dst_on_device) {
        dst_ddf = (float *) dst_extra->data_device[g_main_device_index];
        // printf("zjy dst_ddf=%p main_stream=%p g_main_device_index=%d\n", dst_ddf, main_stream, g_main_device_index);
    } else {
        dst_ddf = dst_f.alloc(ggml_nelements(dst));
    }

    // GGML_SYCL_DEBUG("op src0=%p, src1=%p, dst=%p, src0_ddf=%p, src1_ddf=%p, dst_ddf=%p, main_stream=%p\n",
        // src0, src1, dst, src0_ddf, src1_ddf, dst_ddf, main_stream);
    // do the computation
    op(src0, src1, dst, src0_ddf, src1_ddf, dst_ddf, main_stream);
    /*
    DPCT1010:89: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);

    // copy dst to host if necessary
    if (!dst_on_device) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            main_stream->memcpy(dst->data, dst_ddf, ggml_nbytes(dst))));
    }

    if (dst->backend == GGML_BACKEND_CPU) {
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
    }
    // print_ggml_tensor("tensor", dst);
}
catch (sycl::exception const &exc) {

  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_set_peer_access(const int n_tokens) {
    static bool peer_access_enabled = false;

    const bool enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_SIZE;

    if (peer_access_enabled == enable_peer_access) {
        return;
    }

#ifdef NDEBUG
    for (int id = 0; id < g_device_count; ++id) {
        SYCL_CHECK(ggml_sycl_set_device(get_device_id_by_index(id)));
        // SYCL_CHECK(syclDeviceSynchronize());
    }

    for (int id = 0; id < g_device_count; ++id) {
        SYCL_CHECK(ggml_sycl_set_device(get_device_id_by_index(id)));
        int device_id = g_device_caps[id].device_id;

        for (int id_other = 0; id_other < g_device_count; ++id_other) {
            int device_id_other = g_device_caps[id_other].device_id;
            if (device_id == id_other) {
                continue;
            }
            if (device_id != g_main_device && device_id_other != g_main_device) {
                continue;
            }

            int can_access_peer;
            // SYCL_CHECK(syclDeviceCanAccessPeer(&can_access_peer, id, id_other));
            // if (can_access_peer) {
            //     if (enable_peer_access) {
            //         SYCL_CHECK(syclDeviceEnablePeerAccess(id_other, 0));
            //     } else {
            //         SYCL_CHECK(syclDeviceDisablePeerAccess(id_other));
            //     }
            // }
        }
    }
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;
}

static void ggml_sycl_op_mul_mat(const ggml_tensor *src0,
                                 const ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op,
                                 const bool convert_src1_to_q8_1) try {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t nrows0 = ggml_nrows(src0);

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    GGML_ASSERT(dst->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(src1->backend != GGML_BACKEND_GPU_SPLIT);

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    ggml_tensor_extra_gpu *  dst_extra = (ggml_tensor_extra_gpu *)  dst->extra;

    const bool src0_on_device = src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT;
    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = src0->backend == GGML_BACKEND_GPU_SPLIT;
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    // dd = data device
    char  *  src0_dd[GGML_SYCL_MAX_DEVICES] = {nullptr};
    float * src1_ddf[GGML_SYCL_MAX_DEVICES] = {nullptr}; // float
    char  * src1_ddq[GGML_SYCL_MAX_DEVICES] = {nullptr}; // q8_1
    float *   dst_dd[GGML_SYCL_MAX_DEVICES] = {nullptr};

    // as = actual size
    size_t  src0_as[GGML_SYCL_MAX_DEVICES] = {0};
    size_t src1_asf[GGML_SYCL_MAX_DEVICES] = {0};
    size_t src1_asq[GGML_SYCL_MAX_DEVICES] = {0};
    size_t   dst_as[GGML_SYCL_MAX_DEVICES] = {0};

    int64_t  row_low[GGML_SYCL_MAX_DEVICES];
    int64_t row_high[GGML_SYCL_MAX_DEVICES];

    int used_devices = 0;

    for (int64_t id = 0; id < g_device_count; ++id) {
        // by default, use all rows
        row_low[id]  = 0;
        row_high[id] = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type);

            if (id != 0) {
                row_low[id]  = ne01*g_tensor_split[id];
                if (row_low[id] < ne01) {
                    row_low[id] -= row_low[id] % rounding;
                }
            }

            if (id != g_device_count - 1) {
                row_high[id]  = ne01*g_tensor_split[id + 1];
                if (row_high[id] < ne01) {
                    row_high[id] -= row_high[id] % rounding;
                }
            }
        }
    }
    for (int64_t id = 0; id < g_device_count; ++id) {

        if ((!split && id != g_main_device_index) || row_low[id] == row_high[id]) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = src1->backend == GGML_BACKEND_GPU && id == g_main_device_index;
        const bool  dst_on_device =  dst->backend == GGML_BACKEND_GPU && id == g_main_device_index;

        ggml_sycl_set_device(get_device_id_by_index(id));
        const dpct::queue_ptr stream = g_syclStreams[id][0];

        if (src0_on_device && src0_is_contiguous) {
            src0_dd[id] = (char *) src0_extra->data_device[id];
        } else {
            // const size_t size_src0_ddq = split ? (row_high[id]-row_low[id])*ne00 * src0_ts/src0_bs : ggml_nbytes(src0);
            src0_dd[id] = (char *) ggml_sycl_pool_malloc(ggml_nbytes(src0), &src0_as[id]);
        }

        if (src1_on_device && src1_is_contiguous) {
            src1_ddf[id] = (float *) src1_extra->data_device[id];
        } else {
            src1_ddf[id] = (float *) ggml_sycl_pool_malloc(ggml_nbytes(src1), &src1_asf[id]);
        }

        if (convert_src1_to_q8_1) {
            src1_ddq[id] = (char *) ggml_sycl_pool_malloc(nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs, &src1_asq[id]);

            if (src1_on_device && src1_is_contiguous) {
                quantize_row_q8_1_sycl(src1_ddf[id], src1_ddq[id], ne10, nrows1, src1_padded_col_size, stream);
                /*
                DPCT1010:90: SYCL uses exceptions to report errors and does not
                use the error codes. The call was replaced with 0. You need to
                rewrite this code.
                */
                SYCL_CHECK(0);
            }
        }

        if (dst_on_device) {
            dst_dd[id] = (float *) dst_extra->data_device[id];
        } else {
            const size_t size_dst_ddf = split ? (row_high[id]-row_low[id])*ne1*sizeof(float) : ggml_nbytes(dst);
            dst_dd[id] = (float *) ggml_sycl_pool_malloc(size_dst_ddf, &dst_as[id]);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        SYCL_CHECK(ggml_sycl_set_device(g_main_device));
        /*
        DPCT1024:91: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[g_main_device_index][0] =
                g_syclStreams[g_main_device_index][0]->ext_oneapi_submit_barrier()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = split ? (src1_col_0/src1_col_stride) % MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int64_t id = 0; id < g_device_count; ++id) {
            if ((!split && id != g_main_device_index) || row_low[id] == row_high[id]) {
                continue;
            }

            const bool src1_on_device = src1->backend == GGML_BACKEND_GPU && id == g_main_device_index;
            const bool  dst_on_device =  dst->backend == GGML_BACKEND_GPU && id == g_main_device_index;
            const int64_t row_diff = row_high[id] - row_low[id];

            ggml_sycl_set_device(get_device_id_by_index(id));
            const dpct::queue_ptr stream = g_syclStreams[id][is];

            // wait for main GPU data if necessary
            if (split && (id != g_main_device_index || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[g_main_device_index][0]})));
            }

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  src0_dd[id] + (i0/i02_divisor) * (ne01*ne00*src0_ts)/src0_bs;
                float * src1_ddf_i = src1_ddf[id] + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = src1_ddq[id] +  src1_ddq_i_offset;
                float *   dst_dd_i =   dst_dd[id] + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (dst->backend == GGML_BACKEND_GPU && id == g_main_device_index) {
                    dst_dd_i += row_low[id]; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1->backend == GGML_BACKEND_GPU && src1_is_contiguous) {
                    if (id != g_main_device_index) {
                        if (convert_src1_to_q8_1) {
                            char * src1_ddq_i_source = src1_ddq[g_main_device_index] + src1_ddq_i_offset;
                            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
                                src1_ddq_i, src1_ddq_i_source,
                                src1_ncols * src1_padded_col_size * q8_1_ts /
                                    q8_1_bs)));
                        } else {
                            float * src1_ddf_i_source = (float *) src1_extra->data_device[g_main_device_index];
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
                                src1_ddf_i, src1_ddf_i_source,
                                src1_ncols * ne10 * sizeof(float))));
                        }
                    }
                } else if (src1->backend == GGML_BACKEND_CPU || (src1_on_device && !src1_is_contiguous)) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(
                                   src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                } else {
                    GGML_ASSERT(false);
                }

                if (convert_src1_to_q8_1 && (src1->backend == GGML_BACKEND_CPU || !src1_is_contiguous)) {
                    quantize_row_q8_1_sycl(src1_ddf_i, src1_ddq_i, ne10, src1_ncols, src1_padded_col_size, stream);
                    /*
                    DPCT1010:92: SYCL uses exceptions to report errors and does
                    not use the error codes. The call was replaced with 0. You
                    need to rewrite this code.
                    */
                    SYCL_CHECK(0);
                }

                if (src1_col_0 == 0 && (!src0_on_device || !src0_is_contiguous) && i02 % i02_divisor == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, row_low[id], row_high[id], stream));
                }
                if (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                op(src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                   row_low[id], row_high[id], src1_ncols, src1_padded_col_size, stream);
                /*
                DPCT1010:93: SYCL uses exceptions to report errors and does not
                use the error codes. The call was replaced with 0. You need to
                rewrite this code.
                */
                SYCL_CHECK(0);

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device;
                    dpct::memcpy_direction kind;
                    if (dst->backend == GGML_BACKEND_CPU) {
                        dst_off_device = dst->data;
                        kind = dpct::device_to_host;
                    } else if (dst->backend == GGML_BACKEND_GPU) {
                        dst_off_device = dst_extra->data_device[g_main_device_index];
                        kind = dpct::device_to_device;
                    } else {
                        GGML_ASSERT(false);
                    }
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0 + row_low[id];
                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * sizeof(float), dst_dd_i,
                            row_diff * sizeof(float), row_diff * sizeof(float),
                            src1_ncols, kind, *stream)));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * sizeof(float))));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (id != g_main_device_index || is != 0)) {
                    /*
                    DPCT1024:94: The original code returned the error code that
                    was further consumed by the program logic. This original
                    code was replaced with 0. You may need to rewrite the
                    program logic consuming the error code.
                    */
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[id][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    for (int64_t id = 0; id < g_device_count; ++id) {
        if ((!split && id != g_main_device_index) || row_low[id] == row_high[id]) {
            continue;
        }
        SYCL_CHECK(ggml_sycl_set_device(get_device_id_by_index(id)));

        // free buffers again when done
        if (dst_as[id] > 0) {
            ggml_sycl_pool_free(dst_dd[id], dst_as[id]);
        }
        if (src1_asq[id] > 0) {
            ggml_sycl_pool_free(src1_ddq[id], src1_asq[id]);
        }
        if (src1_asf[id] > 0) {
            ggml_sycl_pool_free(src1_ddf[id], src1_asf[id]);
        }
        if (src0_as[id] > 0) {
            ggml_sycl_pool_free(src0_dd[id], src0_as[id]);
        }
    }

    // main device waits for all other devices to be finished
    if (split && g_device_count > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= MAX_STREAMS ? is_max : MAX_STREAMS;

        SYCL_CHECK(ggml_sycl_set_device(g_main_device));
        for (int64_t id = 0; id < g_device_count; ++id) {
            if (row_low[id] == row_high[id]) {
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    g_syclStreams[g_main_device_index][0]->ext_oneapi_submit_barrier(
                        {*src0_extra->events[id][is]})));
            }
        }
    }

    if (dst->backend == GGML_BACKEND_CPU) {
        SYCL_CHECK(ggml_sycl_set_device(g_main_device));
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_repeat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_repeat);
}

static void ggml_sycl_get_rows(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_get_rows);
}

static void ggml_sycl_add(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_add);
    // log_tensor_with_cnt("log_ggml_sycl_add_src0", (struct ggml_tensor *) src0, 6);
    // log_tensor_with_cnt("log_ggml_sycl_add_src1", (struct ggml_tensor *)src1, 6);
    // log_tensor_with_cnt("log_ggml_sycl_add_dst", dst, 6);
}

static void ggml_sycl_acc(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_acc);
}

static void ggml_sycl_mul(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_mul);
    // log_tensor_with_cnt("log_ggml_sycl_mul_src0", (struct ggml_tensor *)src0, 6);
    // log_tensor_with_cnt("log_ggml_sycl_mul_src1", (struct ggml_tensor *)src1, 6);
    // log_tensor_with_cnt("log_ggml_sycl_mul_dst", dst, 6);

}

static void ggml_sycl_div(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_div);
}

static void ggml_sycl_gelu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_gelu);
}

static void ggml_sycl_silu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_silu);
}

static void ggml_sycl_gelu_quick(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_gelu_quick);
}

static void ggml_sycl_tanh(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_tanh);
}

static void ggml_sycl_relu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_relu);
}

static void ggml_sycl_leaky_relu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_leaky_relu);
}

static void ggml_sycl_sqr(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_sqr);
}

static void ggml_sycl_norm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_norm);
}

static void ggml_sycl_group_norm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_group_norm);
}

static void ggml_sycl_concat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_concat);
}

static void ggml_sycl_upscale(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_upscale);
}

static void ggml_sycl_pad(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_pad);
}


static void ggml_sycl_rms_norm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_rms_norm);
    // log_tensor_with_cnt("log_ggml_sycl_rms_norm_src0", (struct ggml_tensor *)src0, 6);
    // log_tensor_with_cnt("log_ggml_sycl_rms_norm_src1", (struct ggml_tensor *)src1, 6);
    // log_tensor_with_cnt("log_ggml_sycl_rms_norm_dst", dst, 6);
}

bool ggml_sycl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    if (!g_sycl_loaded) return false;

    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
            src1->type == GGML_TYPE_F32 &&
             dst->type == GGML_TYPE_F32 &&
            (ne0 >= 32 && ne1 >= 32 && ne10 >= 32);
}

static void ggml_sycl_mul_mat_vec_p021(const ggml_tensor *src0,
                                       const ggml_tensor *src1,
                                       ggml_tensor *dst) try {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(g_main_device));
    dpct::queue_ptr main_stream = g_syclStreams[g_main_device_index][0];

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    void * src0_ddq = src0_extra->data_device[g_main_device_index];

    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    float * src1_ddf = (float *) src1_extra->data_device[g_main_device_index];

    ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;
    float * dst_ddf = (float *) dst_extra->data_device[g_main_device_index];

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_mul_mat_vec_nc(const ggml_tensor *src0,
                                     const ggml_tensor *src1,
                                     ggml_tensor *dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(g_main_device));
    dpct::queue_ptr main_stream = g_syclStreams[g_main_device_index][0];

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    void * src0_ddq = src0_extra->data_device[g_main_device_index];

    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    float * src1_ddf = (float *) src1_extra->data_device[g_main_device_index];

    ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;
    float * dst_ddf = (float *) dst_extra->data_device[g_main_device_index];

    const int64_t row_stride_x = nb01 / sizeof(sycl::half);
    const int64_t channel_stride_x = nb02 / sizeof(sycl::half);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x, main_stream);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void k_compute_batched_ptrs(const sycl::half *src0_as_f16,
                                   const sycl::half *src1_as_f16, char *dst,
                                   const void **ptrs_src, void **ptrs_dst,
                                   int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12,
                                   size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3,
                                   const sycl::nd_item<3> &item_ct1) {
    int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                  item_ct1.get_local_id(1);

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    int64_t i03 = i13 / r3;
    int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02   + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12/2 + i13*nb13/2;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2   + i13*nbd3;
}

static void ggml_sycl_mul_mat_mat_batched_sycl(const ggml_tensor *src0,
                                                 const ggml_tensor *src1,
                                                 ggml_tensor *dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));

    GGML_ASSERT(src0->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0]; GGML_UNUSED(ne00);
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2]; GGML_UNUSED(nb02);
    const int64_t nb03 = src0->nb[3]; GGML_UNUSED(nb03);

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2]; GGML_UNUSED(nb12);
    const int64_t nb13 = src1->nb[3]; GGML_UNUSED(nb13);

    const int64_t ne1 = ggml_nelements(src1);
    const int64_t ne  = ggml_nelements(dst);

    SYCL_CHECK(ggml_sycl_set_device(g_main_device));
    dpct::queue_ptr main_stream = g_syclStreams[g_main_device_index][0];

    SYCL_CHECK(
        CHECK_TRY_ERROR(g_sycl_handles[g_main_device_index] = main_stream));

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    void * src0_ddq = src0_extra->data_device[g_main_device_index];
    sycl::half *src0_as_f16 = (sycl::half *)src0_ddq;

    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    float * src1_ddf = (float *) src1_extra->data_device[g_main_device_index];

    ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;
    float * dst_ddf = (float *) dst_extra->data_device[g_main_device_index];

    // convert src1 to fp16
    const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type);
    GGML_ASSERT(to_fp16_sycl != nullptr);

    sycl_pool_alloc<sycl::half> src1_as_f16(ne1);
    to_fp16_sycl(src1_ddf, src1_as_f16.get(), ne1, main_stream);

    sycl_pool_alloc<sycl::half> dst_f16;
    char * dst_t;

    dpct::library_data_t cu_compute_type = dpct::library_data_t::real_half;
    dpct::library_data_t cu_data_type = dpct::library_data_t::real_half;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const sycl::half alpha_f16 = 1.0f;
    const sycl::half beta_f16 = 0.0f;

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    const void * alpha = &alpha_f16;
    const void * beta  = &beta_f16;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        dst_t = (char *) dst_f16.alloc(ne);

        nbd2 /= sizeof(float) / sizeof(sycl::half);
        nbd3 /= sizeof(float) / sizeof(sycl::half);
    } else {
        dst_t = (char *) dst_ddf;

        cu_compute_type = dpct::library_data_t::real_float;
        cu_data_type = dpct::library_data_t::real_float;

        alpha = &alpha_f32;
        beta  = &beta_f32;
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

#if 0
    // use syclGemmEx
    {
        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                int i03 = i13 / r3;
                int i02 = i12 / r2;

                SYCL_CHECK(
                        syclGemmEx(g_sycl_handles[g_main_device_index], CUBLAS_OP_T, CUBLAS_OP_N,
                            ne01, ne11, ne10,
                            alpha, (const char *) src0_as_f16 + i02*src0->nb[2]   + i03*src0->nb[3]  , SYCL_R_16F,   nb01/sizeof(half),
                                   (const char *) src1_as_f16 + i12*src1->nb[2]/2 + i13*src1->nb[3]/2, SYCL_R_16F,   nb11/sizeof(float),
                            beta,  (      char *)       dst_t + i12*nbd2          + i13*nbd3,          cu_data_type, ne01,
                            cu_compute_type,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
    }
#else
    if (r2 == 1 && r3 == 1 && src0->nb[2]*src0->ne[2] == src0->nb[3] && src1->nb[2]*src1->ne[2] == src1->nb[3]) {
        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use syclGemmStridedBatchedEx
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
            *g_sycl_handles[g_main_device_index], oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
            (const char *)src0_as_f16, dpct::library_data_t::real_half,
            nb01 / sizeof(sycl::half), src0->nb[2] / sizeof(sycl::half),
            (const char *)src1_as_f16.get(), dpct::library_data_t::real_half,
            nb11 / sizeof(float), src1->nb[2] / sizeof(float), beta,
            (char *)dst_t, cu_data_type, ne01, dst->nb[2] / sizeof(float),
            ne12 * ne13, cu_compute_type)));
    } else {
        // use syclGemmBatchedEx
        const int ne23 = ne12*ne13;

        sycl_pool_alloc<const void *> ptrs_src(2*ne23);
        sycl_pool_alloc<      void *> ptrs_dst(1*ne23);

        sycl::range<3> block_dims(1, ne12, ne13);
        /*
        DPCT1049:47: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(main_stream->get_device(),
                                         {sycl::aspect::fp16});

            main_stream->submit([&](sycl::handler &cgh) {
                const sycl::half *src1_as_f16_get_ct1 = src1_as_f16.get();
                const void **ptrs_src_get_ct3 = ptrs_src.get();
                void **ptrs_dst_get_ct4 = ptrs_dst.get();

                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     k_compute_batched_ptrs(
                                         src0_as_f16, src1_as_f16_get_ct1,
                                         dst_t, ptrs_src_get_ct3,
                                         ptrs_dst_get_ct4, ne12, ne13, ne23,
                                         nb02, nb03, nb12, nb13, nbd2, nbd3, r2,
                                         r3, item_ct1);
                                 });
            });
        }
        /*
        DPCT1010:95: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        SYCL_CHECK(0);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
            *g_sycl_handles[g_main_device_index], oneapi::mkl::transpose::trans,
            oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
            (const void **)(ptrs_src.get() + 0 * ne23),
            dpct::library_data_t::real_half, nb01 / sizeof(sycl::half),
            (const void **)(ptrs_src.get() + 1 * ne23),
            dpct::library_data_t::real_half, nb11 / sizeof(float), beta,
            (void **)(ptrs_dst.get() + 0 * ne23), cu_data_type, ne01, ne23,
            cu_compute_type)));
    }
#endif

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(GGML_TYPE_F16);
        to_fp32_sycl(dst_f16.get(), dst_ddf, ne, main_stream);
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_mul_mat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const bool all_on_device =
        (src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT) &&
        (src1->backend == GGML_BACKEND_GPU) &&
        ( dst->backend == GGML_BACKEND_GPU);

    const bool split = src0->backend == GGML_BACKEND_GPU_SPLIT;

    int64_t min_compute_capability = INT_MAX;
    for (int64_t id = 0; id < g_device_count; ++id) {
        if (min_compute_capability > g_device_caps[id].cc && g_tensor_split[id] < (id + 1 < g_device_count ? g_tensor_split[id + 1] : 1.0f)) {
            min_compute_capability = g_device_caps[id].cc;
        }
    }

#ifdef SYCL_USE_XMX
    const bool use_xmx = true;
#else
    const bool use_xmx = false;
#endif

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (!split && all_on_device && !use_xmx && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // KQ single-batch
        // GGML_SYCL_DEBUG("ggml_sycl_mul_mat_vec_p021\n");
        ggml_sycl_mul_mat_vec_p021(src0, src1, dst);
    } else if (!split && all_on_device && !use_xmx && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1) {
        // KQV single-batch
        // GGML_SYCL_DEBUG("ggml_sycl_mul_mat_vec_nc\n");
        ggml_sycl_mul_mat_vec_nc(src0, src1, dst);
    } else if (!split && all_on_device && use_xmx && src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1)) {
        // KQ + KQV multi-batch
        // GGML_SYCL_DEBUG("ggml_sycl_mul_mat_mat_batched_sycl\n");
        ggml_sycl_mul_mat_mat_batched_sycl(src0, src1, dst);
    } else if (src0->type == GGML_TYPE_F32) {
        // GGML_SYCL_DEBUG("ggml_sycl_op_mul_mat\n");
        ggml_sycl_op_mul_mat(src0, src1, dst, ggml_sycl_op_mul_mat_sycl, false);
    } else if (ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_F16) {
        // GGML_SYCL_DEBUG("ggml_is_quantized or GGML_TYPE_F16\n");
        if (src1->ne[1] == 1 && src0->ne[0] % GGML_SYCL_DMMV_X == 0) {
#ifdef GGML_SYCL_FORCE_DMMV
            const bool use_mul_mat_vec_q = false;
#else
            const bool use_mul_mat_vec_q = min_compute_capability >= VER_4VEC && ggml_is_quantized(src0->type) && ggml_nrows(src1) == 1;
#endif // GGML_SYCL_FORCE_DMMV

            if (use_mul_mat_vec_q) {
                // NOTE: this kernel does not support ggml_nrows(src1) > 1
                // GGML_SYCL_DEBUG("ggml_sycl_mul_mat ggml_sycl_op_mul_mat_vec_q path\n");
                ggml_sycl_op_mul_mat(src0, src1, dst, ggml_sycl_op_mul_mat_vec_q, true);
            } else {
                // GGML_SYCL_DEBUG("ggml_sycl_mul_mat ggml_sycl_op_dequantize_mul_mat_vec path\n");
                ggml_sycl_op_mul_mat(src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec, false);
            }
        } else {
            bool use_mul_mat_q = min_compute_capability >= VER_4VEC && ggml_is_quantized(src0->type);

            if (use_xmx && min_compute_capability >= VER_GEN9 && src1->ne[1] > XMX_MAX_BATCH_SIZE) {
                use_mul_mat_q = false;
            }

            if (use_mul_mat_q) {
                // GGML_SYCL_DEBUG("ggml_sycl_mul_mat ggml_sycl_op_mul_mat_q path\n");
                ggml_sycl_op_mul_mat(src0, src1, dst, ggml_sycl_op_mul_mat_q, true);
            } else {
                // GGML_SYCL_DEBUG("ggml_sycl_mul_mat ggml_sycl_op_mul_mat_sycl path\n");
                ggml_sycl_op_mul_mat(src0, src1, dst, ggml_sycl_op_mul_mat_sycl, false);
            }
        }
    } else {
        GGML_ASSERT(false);
    }
}

#if 0
template<typename ... Srcs>
static __global__ void k_compute_batched_ptrs_id(
        const void ** ptrs_src, void ** ptrs_dst,
        int ne12, int ne13,
        int ne23,
        int nb02, int nb03,
        int nb12, int nb13,
        int nb2, int nb3,
        int r2, int r3,
        ggml_type src0_type, half * src0_as_f16, int64_t src0_ne,
        const half * src1_f16, half * dst_f16,
        const int32_t * ids, const int id,
        Srcs... src0s) {

    int i = ids[id];

    half * src0_f16;
    const void * srcs_ar[] = { (const half *) src0s... };
    if (src0_type == GGML_TYPE_F16) {
        src0_f16 = (half *) srcs_ar[i];
    } else {
        src0_f16 = src0_as_f16;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            const to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(src0_type);
            to_fp16(srcs_ar[i], src0_f16, src0_ne, syclStreamFireAndForget);
        }
    }

    int i13 = blockIdx.x * blockDim.x + threadIdx.x;
    int i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    int i03 = i13 / r3;
    int i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_f16 + i02*nb02   + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_f16 + i12*nb12/2 + i13*nb13/2;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)  dst_f16 + i12* nb2/2 + i13* nb3/2;
}

static void ggml_sycl_mul_mat_id_sycl(ggml_tensor * dst) {
    const struct ggml_tensor * ids = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src00 = dst->src[2];

    const int id = dst->op_params[0];

    GGML_ASSERT(!ggml_is_transposed(src00));
    GGML_ASSERT(!ggml_is_transposed(src1));

    GGML_ASSERT(src00->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src00->ne[0]; GGML_UNUSED(ne00);
    const int64_t ne01 = src00->ne[1];
    const int64_t ne02 = src00->ne[2];
    const int64_t ne03 = src00->ne[3];

    //const int64_t nb01 = src00->nb[1];
    const int64_t nb02 = src00->nb[2]; GGML_UNUSED(nb02);
    const int64_t nb03 = src00->nb[3]; GGML_UNUSED(nb03);

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    //const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2]; GGML_UNUSED(nb12);
    const int64_t nb13 = src1->nb[3]; GGML_UNUSED(nb13);

    const int64_t ne1 = ggml_nelements(src1);
    const int64_t ne  = ggml_nelements(dst);

    SYCL_CHECK(ggml_sycl_set_device(g_main_device));
    syclStream_t main_stream = g_syclStreams[g_main_device_index][0];

    SYCL_CHECK(syclSetStream(g_sycl_handles[g_main_device_index], main_stream));

    //ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    //void * src0_ddq = src0_extra->data_device[g_main_device_index];
    //half * src0_as_f16 = (half *) src0_ddq;

    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    float * src1_ddf = (float *) src1_extra->data_device[g_main_device_index];

    ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;
    float * dst_ddf = (float *) dst_extra->data_device[g_main_device_index];

    // convert src1 to fp16
    const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type);
    GGML_ASSERT(to_fp16_sycl != nullptr);

    size_t src1_as = 0;
    half * src1_as_f16 = (half *) ggml_sycl_pool_malloc(ne1 * sizeof(half), &src1_as);
    to_fp16_sycl(src1_ddf, src1_as_f16, ne1, main_stream);

    size_t dst_as = 0;
    half * dst_f16 = (half *) ggml_sycl_pool_malloc(ne * sizeof(half), &dst_as);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    const half alpha_f16 = 1.0f;
    const half beta_f16  = 0.0f;

    // use syclGemmBatchedEx
    const int ne23 = ne12*ne13;

    const void ** ptrs_src = nullptr;
          void ** ptrs_dst = nullptr;

    size_t ptrs_src_s = 0;
    size_t ptrs_dst_s = 0;

    ptrs_src = (const void **) ggml_sycl_pool_malloc(2*ne23*sizeof(void *), &ptrs_src_s);
    ptrs_dst = (      void **) ggml_sycl_pool_malloc(1*ne23*sizeof(void *), &ptrs_dst_s);

    int64_t src0_ne = ggml_nelements(src00);
    half * src0_as_f16 = nullptr;
    size_t src0_as = 0;
    if (src00->type != GGML_TYPE_F16) {
        src0_as_f16 = (half *) ggml_sycl_pool_malloc(src0_ne * sizeof(half), &src0_as);
    }

    static_assert(GGML_MAX_SRC == 6, "GGML_MAX_SRC == 6");
    dim3 block_dims(ne13, ne12);
    k_compute_batched_ptrs_id<<<1, block_dims, 0, main_stream>>>(
            ptrs_src, ptrs_dst,
            ne12, ne13,
            ne23,
            ne00*ne01*sizeof(half), ne00*ne01*ne02*sizeof(half),
            nb12, nb13,
            dst->nb[2], dst->nb[3],
            r2, r3,
            src00->type, src0_as_f16, src0_ne,
            src1_as_f16, dst_f16,
            (const int *)((ggml_tensor_extra_gpu *)ids->extra)->data_device[g_main_device_index], id,
            dst->src[2] ? (const half *)((ggml_tensor_extra_gpu *)dst->src[2]->extra)->data_device[g_main_device_index] : nullptr,
            dst->src[3] ? (const half *)((ggml_tensor_extra_gpu *)dst->src[3]->extra)->data_device[g_main_device_index] : nullptr,
            dst->src[4] ? (const half *)((ggml_tensor_extra_gpu *)dst->src[4]->extra)->data_device[g_main_device_index] : nullptr,
            dst->src[5] ? (const half *)((ggml_tensor_extra_gpu *)dst->src[5]->extra)->data_device[g_main_device_index] : nullptr
    );
    SYCL_CHECK(syclGetLastError());

    SYCL_CHECK(
    syclGemmBatchedEx(g_sycl_handles[g_main_device_index], CUBLAS_OP_T, CUBLAS_OP_N,
            ne01, ne11, ne10,
            &alpha_f16, (const void **) (ptrs_src + 0*ne23), SYCL_R_16F, ne00,
                        (const void **) (ptrs_src + 1*ne23), SYCL_R_16F, ne10,
            &beta_f16,  (      void **) (ptrs_dst + 0*ne23), SYCL_R_16F, ne01,
            ne23,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (src0_as != 0) {
        ggml_sycl_pool_free(src0_as_f16, src0_as);
    }
    if (ptrs_src_s != 0) {
        ggml_sycl_pool_free(ptrs_src, ptrs_src_s);
    }
    if (ptrs_dst_s != 0) {
        ggml_sycl_pool_free(ptrs_dst, ptrs_dst_s);
    }

    const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(GGML_TYPE_F16);
    to_fp32_sycl(dst_f16, dst_ddf, ne, main_stream);

    ggml_sycl_pool_free(src1_as_f16, src1_as);
    ggml_sycl_pool_free(dst_f16, dst_as);
}
#endif

static void ggml_sycl_mul_mat_id(const ggml_tensor *src0,
                                 const ggml_tensor *src1,
                                 ggml_tensor *dst) try {
#if 0
    ggml_sycl_mul_mat_id_sycl(dst);
    // TODO: mmq/mmv support
#endif

    const int64_t nb11 = src1->nb[1];
    const int64_t nb1  =  dst->nb[1];

    const struct ggml_tensor * ids = src0;
    const int32_t id = ((int32_t *) dst->op_params)[0];
    const int32_t n_as = ((int32_t *) dst->op_params)[1];

    std::vector<char> ids_host(ggml_nbytes(ids));

    const dpct::queue_ptr stream = g_syclStreams[g_main_device_index][0];

    if (ids->backend == GGML_BACKEND_GPU) {
        const char * ids_dev = (const char *)((const ggml_tensor_extra_gpu *)ids->extra)->data_device[g_main_device_index];
        SYCL_CHECK(CHECK_TRY_ERROR(
            stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
    } else {
        memcpy(ids_host.data(), ids->data, ggml_nbytes(ids));
    }

    const ggml_tensor_extra_gpu * src1_extra = (const ggml_tensor_extra_gpu *) src1->extra;
    const ggml_tensor_extra_gpu * dst_extra = (const ggml_tensor_extra_gpu *) dst->extra;

    ggml_tensor_extra_gpu src1_row_extra;
    ggml_tensor_extra_gpu dst_row_extra;

    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    src1_row.backend = GGML_BACKEND_GPU;
    dst_row.backend  = GGML_BACKEND_GPU;

    src1_row.extra = &src1_row_extra;
    dst_row.extra = &dst_row_extra;

    char * src1_original = src1->backend == GGML_BACKEND_CPU ?
        (char *) src1->data : (char *) src1_extra->data_device[g_main_device_index];
    char * dst_original  =  dst->backend == GGML_BACKEND_CPU ?
        (char *)  dst->data : (char *)  dst_extra->data_device[g_main_device_index];

    if (src1->ne[1] == 1) {
        GGML_ASSERT(src1->backend == GGML_BACKEND_GPU);
        GGML_ASSERT(dst->backend  == GGML_BACKEND_GPU);

        for (int64_t i01 = 0; i01 < ids->ne[1]; i01++) {
            //int32_t row_id;
            //SYCL_CHECK(syclMemcpyAsync(&row_id, ids_dev + i01*ids->nb[1] + id*ids->nb[0], sizeof(int32_t), syclMemcpyDeviceToHost, g_syclStreams[g_main_device][0]));
            //SYCL_CHECK(syclStreamSynchronize(g_syclStreams[g_main_device][0]));

            const int32_t row_id = *(const int32_t *) (ids_host.data() + i01*ids->nb[1] + id*ids->nb[0]);

            GGML_ASSERT(row_id >= 0 && row_id < n_as);

            const struct ggml_tensor * src0_row = dst->src[row_id + 2];

            src1_row_extra.data_device[g_main_device_index] = src1_original + i01*src1->nb[1];
            src1_row.data = (char *) src1->data + i01*src1->nb[1]; // TODO why is this set?

            dst_row_extra.data_device[g_main_device_index] = dst_original + i01*dst->nb[1];
            dst_row.data = (char *) dst->data + i01*dst->nb[1]; // TODO why is this set?

            ggml_sycl_mul_mat(src0_row, &src1_row, &dst_row);
        }
    } else {
        sycl_pool_alloc<char> src1_contiguous(sizeof(float)*ggml_nelements(src1));
        sycl_pool_alloc<char>  dst_contiguous(sizeof(float)*ggml_nelements(dst));

        src1_row_extra.data_device[g_main_device_index] = src1_contiguous.get();
        dst_row_extra.data_device[g_main_device_index]  =  dst_contiguous.get();

        const dpct::memcpy_direction src1_kind =
            src1->backend == GGML_BACKEND_CPU ? dpct::host_to_device
                                              : dpct::device_to_device;
        const dpct::memcpy_direction dst_kind = dst->backend == GGML_BACKEND_CPU
                                                    ? dpct::device_to_host
                                                    : dpct::device_to_device;

        for (int32_t row_id = 0; row_id < n_as; ++row_id) {
            const struct ggml_tensor * src0_row = dst->src[row_id + 2];

            int64_t num_src1_rows = 0;
            for (int64_t i01 = 0; i01 < ids->ne[1]; i01++) {
                const int32_t row_id_i = *(const int32_t *) (ids_host.data() + i01*ids->nb[1] + id*ids->nb[0]);

                if (row_id_i != row_id) {
                    continue;
                }

                GGML_ASSERT(row_id >= 0 && row_id < n_as);

                SYCL_CHECK(CHECK_TRY_ERROR(
                    stream->memcpy(src1_contiguous.get() + num_src1_rows * nb11,
                                   src1_original + i01 * nb11, nb11)));
                num_src1_rows++;
            }

            if (num_src1_rows == 0) {
                continue;
            }

            src1_row.ne[1] = num_src1_rows;
            dst_row.ne[1] = num_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows*nb11;
            src1_row.nb[3] = num_src1_rows*nb11;

            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows*nb1;
            dst_row.nb[3] = num_src1_rows*nb1;

            ggml_sycl_mul_mat(src0_row, &src1_row, &dst_row);

            num_src1_rows = 0;
            for (int64_t i01 = 0; i01 < ids->ne[1]; i01++) {
                const int32_t row_id_i = *(const int32_t *) (ids_host.data() + i01*ids->nb[1] + id*ids->nb[0]);

                if (row_id_i != row_id) {
                    continue;
                }

                GGML_ASSERT(row_id >= 0 && row_id < n_as);

                SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(
                    dst_original + i01 * nb1,
                    dst_contiguous.get() + num_src1_rows * nb1, nb1)));
                num_src1_rows++;
            }
        }
    }

    if (dst->backend == GGML_BACKEND_CPU) {
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_scale(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_scale);
}

static void ggml_sycl_clamp(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_clamp);
}

static void ggml_sycl_cpy(const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst) try {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    GGML_ASSERT(src0->backend == GGML_BACKEND_GPU);
    GGML_ASSERT(src1->backend == GGML_BACKEND_GPU);

    GGML_ASSERT(ggml_nbytes(src0) <= INT_MAX);
    GGML_ASSERT(ggml_nbytes(src1) <= INT_MAX);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    GGML_ASSERT(src0->ne[3] == 1);

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(src1->ne[3] == 1);

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];

    SYCL_CHECK(ggml_sycl_set_device(g_main_device));
    dpct::queue_ptr main_stream = g_syclStreams[g_main_device_index][0];

    const ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    const ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;

    char * src0_ddc = (char *) src0_extra->data_device[g_main_device_index];
    char * src1_ddc = (char *) src1_extra->data_device[g_main_device_index];

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl (src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl (src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl (src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12, main_stream);
    } else if (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl (src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12, main_stream);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl (src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12, main_stream);
    } else {
        fprintf(stderr, "%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ASSERT(false);
    }

    (void) dst;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_sycl_dup(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // TODO: why do we pass dst as src1 here?
    ggml_sycl_cpy(src0, dst, nullptr);
    (void) src1;
}

static void ggml_sycl_diag_mask_inf(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_diag_mask_inf);
}

static void ggml_sycl_soft_max(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_soft_max);
}

static void ggml_sycl_rope(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0)); // TODO: this restriction is temporary until non-cont support is implemented
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_rope);
}

static void ggml_sycl_alibi(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_alibi);
}

static void ggml_sycl_im2col(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_im2col);
}

static void ggml_sycl_sum_rows(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_sum_rows);
}

static void ggml_sycl_argsort(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    ggml_sycl_op_flatten(src0, src1, dst, ggml_sycl_op_argsort);
}

static void ggml_sycl_nop(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    (void) src0;
    (void) src1;
    (void) dst;
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

void ggml_sycl_transform_tensor(void *data, struct ggml_tensor *tensor) try {
    const int64_t nrows = ggml_nrows(tensor);

    const int64_t ne0 = tensor->ne[0];

    const size_t nb1 = tensor->nb[1];

    ggml_backend_type backend = tensor->backend;
    ggml_tensor_extra_gpu * extra = new struct ggml_tensor_extra_gpu;
    memset(extra, 0, sizeof(*extra));

    for (int64_t id = 0; id < g_device_count; ++id) {
        if (backend == GGML_BACKEND_GPU && id != g_main_device_index) {
            continue;
        }
        ggml_sycl_set_device(get_device_id_by_index(id));
        const dpct::queue_ptr stream = g_syclStreams[id][0];

        int64_t row_low, row_high;
        if (backend == GGML_BACKEND_GPU) {
            row_low = 0;
            row_high = nrows;
        } else if (backend == GGML_BACKEND_GPU_SPLIT) {
            const int64_t rounding = get_row_rounding(tensor->type);

            row_low = id == 0 ? 0 : nrows*g_tensor_split[id];
            row_low -= row_low % rounding;

            if (id == g_device_count - 1) {
                row_high = nrows;
            } else {
                row_high = nrows*g_tensor_split[id + 1];
                row_high -= row_high % rounding;
            }
        } else {
            GGML_ASSERT(false);
        }
        if (row_low == row_high) {
            continue;
        }

        int64_t nrows_split = row_high - row_low;

        const size_t offset_split = row_low*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (char *)sycl::malloc_device(
                                        size, *stream)));
        char * buf_host = (char *)data + offset_split;

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                .memset(buf + original_size, 0, size - original_size)
                .wait()));
        }

        SYCL_CHECK(CHECK_TRY_ERROR((*stream)
                                    .memcpy(buf, buf_host, original_size)
                                    .wait()));

        extra->data_device[id] = buf;

        if (backend == GGML_BACKEND_GPU_SPLIT) {
            for (int64_t is = 0; is < MAX_STREAMS; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(extra->events[id][is] =
                                                new sycl::event()));
            }
        }
    }

    tensor->extra = extra;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_free_data(struct ggml_tensor *tensor) try {
    if (!tensor || !tensor->extra || (tensor->backend != GGML_BACKEND_GPU && tensor->backend != GGML_BACKEND_GPU_SPLIT) ) {
        return;
    }

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    for (int64_t id = 0; id < g_device_count; ++id) {
        const dpct::queue_ptr stream = g_syclStreams[id][0];
        if (extra->data_device[id] != nullptr) {
            SYCL_CHECK(ggml_sycl_set_device(get_device_id_by_index(id)));
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(extra->data_device[id], *stream)));
        }

        for (int64_t is = 0; is < MAX_STREAMS; ++is) {
            if (extra->events[id][is] != nullptr) {
                SYCL_CHECK(ggml_sycl_set_device(get_device_id_by_index(id)));
                SYCL_CHECK(CHECK_TRY_ERROR(
                    dpct::destroy_event(extra->events[id][is])));
            }
        }
    }

    delete extra;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static ggml_tensor_extra_gpu * g_temp_tensor_extras = nullptr;
static size_t g_temp_tensor_extra_index = 0;

static ggml_tensor_extra_gpu * ggml_sycl_alloc_temp_tensor_extra() {
    if (g_temp_tensor_extras == nullptr) {
        g_temp_tensor_extras = new ggml_tensor_extra_gpu[GGML_SYCL_MAX_NODES];
    }

    size_t alloc_index = g_temp_tensor_extra_index;
    g_temp_tensor_extra_index = (g_temp_tensor_extra_index + 1) % GGML_SYCL_MAX_NODES;
    ggml_tensor_extra_gpu * extra = &g_temp_tensor_extras[alloc_index];
    memset(extra, 0, sizeof(*extra));

    return extra;
}

static void ggml_sycl_assign_buffers_impl(struct ggml_tensor *tensor,
                                          bool scratch, bool force_inplace,
                                          bool no_alloc) try {
    if (scratch && g_scratch_size == 0) {
        return;
    }

    tensor->backend = GGML_BACKEND_GPU;

    if (tensor->src[0] != nullptr && tensor->src[0]->backend == GGML_BACKEND_CPU) {
        const ggml_op src0_op = tensor->src[0]->op;
        if (src0_op == GGML_OP_RESHAPE || src0_op == GGML_OP_TRANSPOSE || src0_op == GGML_OP_VIEW || src0_op == GGML_OP_PERMUTE) {
            ggml_sycl_assign_buffers_impl(tensor->src[0], scratch, force_inplace, no_alloc);
        }
    }
    if (tensor->op == GGML_OP_CPY && tensor->src[1]->backend == GGML_BACKEND_CPU) {
        ggml_sycl_assign_buffers_impl(tensor->src[1], scratch, force_inplace, no_alloc);
    }

    if (scratch && no_alloc) {
        return;
    }

    ggml_tensor_extra_gpu * extra;

    const bool inplace = (tensor->src[0] != nullptr && tensor->src[0]->data == tensor->data) ||
        tensor->op == GGML_OP_VIEW ||
        force_inplace;
    const size_t size = ggml_nbytes(tensor);

    SYCL_CHECK(ggml_sycl_set_device(g_main_device));
    const dpct::queue_ptr stream = g_syclStreams[g_main_device_index][0];

    if (inplace && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT)) {
        ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu * ) tensor->src[0]->extra;
        char * src0_ddc = (char *) src0_extra->data_device[g_main_device_index];
        size_t offset = 0;
        if (tensor->op == GGML_OP_VIEW) {
            memcpy(&offset, tensor->op_params, sizeof(size_t));
        }
        extra = ggml_sycl_alloc_temp_tensor_extra();
        extra->data_device[g_main_device_index] = src0_ddc + offset;
    } else if (tensor->op == GGML_OP_CPY) {
        ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu * ) tensor->src[1]->extra;
        void * src1_ddv = src1_extra->data_device[g_main_device_index];
        extra = ggml_sycl_alloc_temp_tensor_extra();
        extra->data_device[g_main_device_index] = src1_ddv;
    } else if (scratch) {
        GGML_ASSERT(size <= g_scratch_size);
        if (g_scratch_offset + size > g_scratch_size) {
            g_scratch_offset = 0;
        }

        char * data = (char *) g_scratch_buffer;
        if (data == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(
                data = (char *)sycl::malloc_device(
                    g_scratch_size, *stream)));
            g_scratch_buffer = data;
        }
        extra = ggml_sycl_alloc_temp_tensor_extra();
        extra->data_device[g_main_device_index] = data + g_scratch_offset;

        g_scratch_offset += size;

        GGML_ASSERT(g_scratch_offset <= g_scratch_size);
    } else { // allocate new buffers outside of scratch
        void * data;
        SYCL_CHECK(CHECK_TRY_ERROR(data = (void *)sycl::malloc_device(
                                        size, *stream)));
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream).memset(data, 0, size).wait()));
        extra = new ggml_tensor_extra_gpu;
        memset(extra, 0, sizeof(*extra));
        extra->data_device[g_main_device_index] = data;
    }

    tensor->extra = extra;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_assign_scratch_offset(struct ggml_tensor *tensor,
                                     size_t offset) try {
    if (g_scratch_size == 0) {
        return;
    }
    if (g_scratch_buffer == nullptr) {
        ggml_sycl_set_device(g_main_device);
        const dpct::queue_ptr stream = g_syclStreams[g_main_device_index][0];
        SYCL_CHECK(
            CHECK_TRY_ERROR(g_scratch_buffer = (void *)sycl::malloc_device(
                                 g_scratch_size, *stream)));
    }

    ggml_tensor_extra_gpu * extra = ggml_sycl_alloc_temp_tensor_extra();

    const bool inplace = tensor->view_src != nullptr;

    if (inplace && (tensor->view_src->backend == GGML_BACKEND_GPU || tensor->view_src->backend == GGML_BACKEND_GPU_SPLIT)) {
        ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu * ) tensor->view_src->extra;
        char * src0_ddc = (char *) src0_extra->data_device[g_main_device_index];
        size_t view_offset = 0;
        if (tensor->op == GGML_OP_VIEW) {
            memcpy(&view_offset, tensor->op_params, sizeof(size_t));
        }
        extra->data_device[g_main_device_index] = src0_ddc + view_offset;
    } else {
        extra->data_device[g_main_device_index] = (char *) g_scratch_buffer + offset;
    }

    tensor->extra = extra;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_copy_to_device(struct ggml_tensor *tensor) try {
    GGML_ASSERT(tensor->backend == GGML_BACKEND_GPU);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;
    SYCL_CHECK(ggml_sycl_set_device(g_main_device));
    const dpct::queue_ptr stream = g_syclStreams[g_main_device_index][0];
    SYCL_CHECK(CHECK_TRY_ERROR((*stream)
                                    .memcpy(extra->data_device[g_main_device_index],
                                            tensor->data, ggml_nbytes(tensor))
                                    .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_assign_buffers(struct ggml_tensor * tensor) {
    ggml_sycl_assign_buffers_impl(tensor, true, false, false);
}

void ggml_sycl_assign_buffers_no_alloc(struct ggml_tensor * tensor) {
    ggml_sycl_assign_buffers_impl(tensor, true, false, true);
}

void ggml_sycl_assign_buffers_no_scratch(struct ggml_tensor * tensor) {
    ggml_sycl_assign_buffers_impl(tensor, false, false, false);
}

void ggml_sycl_assign_buffers_force_inplace(struct ggml_tensor * tensor) {
    ggml_sycl_assign_buffers_impl(tensor, false, true, false);
}

void ggml_sycl_set_main_device(const int main_device) try {

    if (main_device >= g_all_sycl_device_count) {
        fprintf(stderr, "warning: cannot set main_device=%d because there are only %d devices. Using device %d instead.\n",
                main_device, g_all_sycl_device_count, g_main_device);
        return;
    }

    if (g_main_device != main_device && g_device_count >= 1) {
        g_main_device = main_device;
        g_main_device_index = get_device_index_by_id(g_main_device);
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(g_main_device))));
        fprintf(stderr, "Using device %d (%s) as main device\n",
                g_main_device, prop.get_name());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_set_scratch_size(const size_t scratch_size) {
    // this is a hack to not completely break llama.cpp when using multiple models or contexts simultaneously
    // it still won't always work as expected, but it's better than nothing
    if (scratch_size > g_scratch_size) {
        ggml_sycl_free_scratch();
    }
    g_scratch_size = std::max(g_scratch_size, scratch_size);
}

void ggml_sycl_free_scratch() try {
    if (g_scratch_buffer == nullptr) {
        return;
    }
    ggml_sycl_set_device(g_main_device);
    const dpct::queue_ptr stream = g_syclStreams[g_main_device_index][0];

    SYCL_CHECK(CHECK_TRY_ERROR(
        sycl::free(g_scratch_buffer, *stream)));
    g_scratch_buffer = nullptr;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool ggml_sycl_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    if (!g_sycl_loaded) return false;

    ggml_sycl_func_t func;
    const bool any_on_device = tensor->backend == GGML_BACKEND_GPU
        || (tensor->src[0] != nullptr && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT))
        || (tensor->src[1] != nullptr && tensor->src[1]->backend == GGML_BACKEND_GPU);

    if (!any_on_device && tensor->op != GGML_OP_MUL_MAT && tensor->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT) {
        if (tensor->src[0]->ne[3] != tensor->src[1]->ne[3]) {
#ifndef NDEBUG
            fprintf(stderr, "%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, tensor->name, tensor->src[0]->ne[3], tensor->src[1]->ne[3]);
#endif
            return false;
        }
    }

    switch (tensor->op) {
        case GGML_OP_REPEAT:
            func = ggml_sycl_repeat;
            break;
        case GGML_OP_GET_ROWS:
            func = ggml_sycl_get_rows;
            break;
        case GGML_OP_DUP:
            func = ggml_sycl_dup;
            break;
        case GGML_OP_ADD:
            func = ggml_sycl_add;
            break;
        case GGML_OP_ACC:
            func = ggml_sycl_acc;
            break;
        case GGML_OP_MUL:
            func = ggml_sycl_mul;
            break;
        case GGML_OP_DIV:
            func = ggml_sycl_div;
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(tensor)) {
                case GGML_UNARY_OP_GELU:
                    func = ggml_sycl_gelu;
                    break;
                case GGML_UNARY_OP_SILU:
                    func = ggml_sycl_silu;
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    func = ggml_sycl_gelu_quick;
                    break;
                case GGML_UNARY_OP_TANH:
                    func = ggml_sycl_tanh;
                    break;
                case GGML_UNARY_OP_RELU:
                    func = ggml_sycl_relu;
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            func = ggml_sycl_norm;
            break;
        case GGML_OP_GROUP_NORM:
            func = ggml_sycl_group_norm;
            break;
        case GGML_OP_CONCAT:
            func = ggml_sycl_concat;
            break;
        case GGML_OP_UPSCALE:
            func = ggml_sycl_upscale;
            break;
        case GGML_OP_PAD:
            func = ggml_sycl_pad;
            break;
        case GGML_OP_LEAKY_RELU:
            func = ggml_sycl_leaky_relu;
            break;
        case GGML_OP_RMS_NORM:
            func = ggml_sycl_rms_norm;
            break;
        case GGML_OP_MUL_MAT:
            if (!any_on_device && !ggml_sycl_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
                return false;
            }
            func = ggml_sycl_mul_mat;
            break;
        case GGML_OP_MUL_MAT_ID:
            if (!any_on_device && !ggml_sycl_can_mul_mat(tensor->src[2], tensor->src[1], tensor)) {
                return false;
            }
            func = ggml_sycl_mul_mat_id;
            break;
        case GGML_OP_SCALE:
            func = ggml_sycl_scale;
            break;
        case GGML_OP_SQR:
            func = ggml_sycl_sqr;
            break;
        case GGML_OP_CLAMP:
            func = ggml_sycl_clamp;
            break;
        case GGML_OP_CPY:
            func = ggml_sycl_cpy;
            break;
        case GGML_OP_CONT:
            func = ggml_sycl_dup;
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            func = ggml_sycl_nop;
            break;
        case GGML_OP_DIAG_MASK_INF:
            func = ggml_sycl_diag_mask_inf;
            break;
        case GGML_OP_SOFT_MAX:
            func = ggml_sycl_soft_max;
            break;
        case GGML_OP_ROPE:
            func = ggml_sycl_rope;
            break;
        case GGML_OP_ALIBI:
            func = ggml_sycl_alibi;
            break;
        case GGML_OP_IM2COL:
            func = ggml_sycl_im2col;
            break;
        case GGML_OP_SUM_ROWS:
            func = ggml_sycl_sum_rows;
            break;
        case GGML_OP_ARGSORT:
            func = ggml_sycl_argsort;
            break;
        default:
            return false;
    }

    if (tensor->src[0] != nullptr && tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT) {
        ggml_sycl_set_peer_access(tensor->src[1]->ne[1]);
    }

    if (params->ith != 0) {
        return true;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return true;
    }
    func(tensor->src[0], tensor->src[1], tensor);
    return true;
}

int ggml_sycl_get_device_count() try {
    int device_count;
    if (CHECK_TRY_ERROR(device_count =
                             dpct::dev_mgr::instance().device_count()) != 0) {
        return 0;
    }
    return device_count;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_sycl_get_device_description(int device, char *description,
                                      size_t description_size) try {
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend interface

#define UNUSED GGML_UNUSED

struct ggml_backend_sycl_context {
    int device;
    std::string name;
};

// sycl buffer

struct ggml_backend_sycl_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    ggml_tensor_extra_gpu * temp_tensor_extras = nullptr;
    size_t temp_tensor_extra_index = 0;
    std::string name;

     ggml_backend_sycl_buffer_context(int device, void * dev_ptr) : device(device), dev_ptr(dev_ptr) {}

    ~ ggml_backend_sycl_buffer_context() {
        delete[] temp_tensor_extras;
    }

    ggml_tensor_extra_gpu * ggml_sycl_alloc_temp_tensor_extra() {
        if (temp_tensor_extras == nullptr) {
            temp_tensor_extras = new ggml_tensor_extra_gpu[GGML_SYCL_MAX_NODES];
        }

        size_t alloc_index = temp_tensor_extra_index;
        temp_tensor_extra_index = (temp_tensor_extra_index + 1) % GGML_SYCL_MAX_NODES;
        ggml_tensor_extra_gpu * extra = &temp_tensor_extras[alloc_index];
        memset(extra, 0, sizeof(*extra));

        return extra;
    }
};

GGML_CALL static const char * ggml_backend_sycl_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_sycl_buffer_get_name;
}

static void
ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) try {
     ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);
    int device_index = get_device_index_by_id(ctx->device);
    const dpct::queue_ptr stream = g_syclStreams[device_index][0];

    SYCL_CHECK(
        CHECK_TRY_ERROR(sycl::free(ctx->dev_ptr, *stream)));
    delete ctx;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
     ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static void ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                 ggml_tensor *tensor) try {
     ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        tensor->backend = tensor->view_src->backend;
        tensor->extra = tensor->view_src->extra;
        return;
    }

    ggml_tensor_extra_gpu * extra = ctx->ggml_sycl_alloc_temp_tensor_extra();

    extra->data_device[ctx->device] = tensor->data;

    tensor->backend = GGML_BACKEND_GPU;
    tensor->extra = extra;

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        int64_t row_low = 0;
        int64_t row_high = ggml_nrows(tensor);
        int64_t nrows_split = row_high - row_low;

        size_t original_size = ggml_nbytes_split(tensor, nrows_split);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(g_syclStreams[ctx->device][0]->memset(
                (char *)tensor->data + original_size, 0,
                padded_size - original_size)));
        }
    }

    UNUSED(buffer);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                const void *data, size_t offset,
                                                size_t size) try {
    GGML_ASSERT(tensor->backend == GGML_BACKEND_GPU);

     ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    int device_index = get_device_index_by_id(ctx->device);
    const dpct::queue_ptr stream = g_syclStreams[device_index][0];
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    SYCL_CHECK(
        CHECK_TRY_ERROR((*stream)
                             .memcpy((char *)tensor->data + offset, data, size)
                             .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size) try {
    GGML_ASSERT(tensor->backend == GGML_BACKEND_GPU);

     ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    int device_index = get_device_index_by_id(ctx->device);
    const dpct::queue_ptr stream = g_syclStreams[device_index][0];

    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    SYCL_CHECK(CHECK_TRY_ERROR(
        (*stream)
            .memcpy(data, (const char *)tensor->data + offset, size)
            .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) try {
     ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    int device_index = get_device_index_by_id(ctx->device);
    const dpct::queue_ptr stream = g_syclStreams[device_index][0];
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    SYCL_CHECK(CHECK_TRY_ERROR((*stream)
                                    .memset(ctx->dev_ptr, value, buffer->size)
                                    .wait()));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static struct ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .get_name        = */ ggml_backend_sycl_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ NULL,
};

// sycl buffer type
struct ggml_backend_sycl_buffer_type_context {
    int device;
    std::string name;
};

GGML_CALL static const char * ggml_backend_sycl_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) try {
    int device = (int) (intptr_t) buft->context;

    ggml_sycl_set_device(device);
    int device_index = get_device_index_by_id(device);
    const dpct::queue_ptr stream = g_syclStreams[device_index][0];
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0

    void * dev_ptr;
    SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (void *)sycl::malloc_device(
                                    size, *stream)));

     ggml_backend_sycl_buffer_context * ctx = new  ggml_backend_sycl_buffer_context(device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    UNUSED(buft);
}

static size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    int64_t row_low = 0;
    int64_t row_high = ggml_nrows(tensor);
    int64_t nrows_split = row_high - row_low;

    size_t size = ggml_nbytes_split(tensor, nrows_split);

    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    UNUSED(buft);
}

static bool ggml_backend_sycl_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_sycl(backend);

    UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_sycl_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // TODO: return device.maxBufferLength
    /* .get_alloc_size   = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .supports_backend = */ ggml_backend_sycl_buffer_type_supports_backend,
    /* .is_host          = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device) {
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    static bool ggml_backend_sycl_buffer_type_initialized = false;

    if (!ggml_backend_sycl_buffer_type_initialized) {
        for (int i = 0; i < GGML_SYCL_MAX_DEVICES; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .context  = */ (ggml_backend_buffer_type_context_t) (intptr_t) i,
            };
        }
        ggml_backend_sycl_buffer_type_initialized = true;
    }

    return &ggml_backend_sycl_buffer_types[device];
}

// host buffer type

GGML_CALL static const char * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_SYCL_NAME "_Host";

    UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_sycl_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_SYCL_NAME "_Host";

    UNUSED(buffer);
}

static void ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_sycl_host_free(buffer->context);
}

static ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_sycl_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_sycl_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_sycl_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_sycl_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // TODO: return device.maxBufferLength
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .supports_backend = */ ggml_backend_cpu_buffer_type()->iface.supports_backend,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .context  = */ nullptr,
    };

    return &ggml_backend_sycl_buffer_type_host;
}

// backend

struct ggml_backend_context_sycl {
    int device;
};

static const char * ggml_backend_sycl_name(ggml_backend_t backend) {
    return GGML_SYCL_NAME;

    UNUSED(backend);
}

static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_context_sycl * sycl_ctx = (ggml_backend_context_sycl *)backend->context;

    delete sycl_ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_sycl_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_context_sycl * sycl_ctx = (ggml_backend_context_sycl *)backend->context;

    return ggml_backend_sycl_buffer_type(sycl_ctx->device);
}

static void ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               const void *data, size_t offset,
                                               size_t size) try {
    ggml_backend_context_sycl * sycl_ctx = (ggml_backend_context_sycl *)backend->context;

    GGML_ASSERT(tensor->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(tensor->backend == GGML_BACKEND_GPU);

    SYCL_CHECK(CHECK_TRY_ERROR(g_syclStreams[sycl_ctx->device][0]->memcpy(
        (char *)tensor->data + offset, data, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *tensor,
                                               void *data, size_t offset,
                                               size_t size) try {
    ggml_backend_context_sycl * sycl_ctx = (ggml_backend_context_sycl *)backend->context;

    GGML_ASSERT(tensor->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(tensor->backend == GGML_BACKEND_GPU);

    SYCL_CHECK(CHECK_TRY_ERROR(g_syclStreams[sycl_ctx->device][0]->memcpy(
        data, (const char *)tensor->data + offset, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {
    ggml_backend_context_sycl * sycl_ctx = (ggml_backend_context_sycl *)backend->context;

    SYCL_CHECK(CHECK_TRY_ERROR(g_syclStreams[sycl_ctx->device][0]->wait()));

    UNUSED(backend);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static ggml_backend_graph_plan_t ggml_backend_sycl_graph_plan_create(ggml_backend_t backend, const ggml_cgraph * cgraph) {
    GGML_ASSERT(!"not implemented");

    return nullptr;

    UNUSED(backend);
    UNUSED(cgraph);
}

static void ggml_backend_sycl_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(!"not implemented");

    UNUSED(backend);
    UNUSED(plan);
}

static void ggml_backend_sycl_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(!"not implemented");

    UNUSED(backend);
    UNUSED(plan);
}

static bool ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_context_sycl * sycl_ctx = (ggml_backend_context_sycl *)backend->context;

    ggml_sycl_set_main_device(sycl_ctx->device);

    ggml_compute_params params = {};
    params.type = GGML_TASK_COMPUTE;
    params.ith = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE)
            continue;

        assert(node->backend == GGML_BACKEND_GPU);
        assert(node->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        assert(node->extra != nullptr);

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->backend == GGML_BACKEND_GPU);
                assert(node->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
                assert(node->src[j]->extra != nullptr);
            }
        }

        bool ok = ggml_sycl_compute_forward(&params, node);
        if (!ok) {
            fprintf(stderr, "%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);

#if 0
        if (node->type == GGML_TYPE_F32) {
            syclDeviceSynchronize();
            std::vector<float> tmp(ggml_nelements(node), 0.0f);
            syclMemcpy(tmp.data(), node->data, ggml_nelements(node)*sizeof(float), syclMemcpyDeviceToHost);
            printf("\n%s (%s) (%s %s) (%s %s): ", node->name, ggml_op_name(node->op),
                ggml_type_name(node->src[0]->type),
                node->src[1] ? ggml_type_name(node->src[1]->type) : "none",
                node->src[0]->name,
                node->src[1] ? node->src[1]->name : "none");
            double sum = 0.0;
            double sq_sum = 0.0;
            for (int i = 0; i < ggml_nelements(node); i++) {
                printf("%f ", tmp[i]);
                sum += tmp[i];
                sq_sum += tmp[i]*tmp[i];
            }
            printf("\n");
            printf("sum: %f, ", sum);
            printf("sq_sum: %f\n", sq_sum);
        }
#endif
    }

    UNUSED(backend);
    return true;
}

static bool ggml_backend_sycl_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                    return true;
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a;
                struct ggml_tensor * b;
                if (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } else {
                    a = op->src[2];
                    b = op->src[1];
                }
                if (a->ne[3] != b->ne[3]) {
                    return false;
                }

                if (a->type == GGML_TYPE_IQ2_XXS) {
                    return false;
                }
                if (a->type == GGML_TYPE_IQ2_XS) {
                    return false;
                }

                return true;
            } break;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                if (src0_type == GGML_TYPE_F32) {
                    return true;
                } else {
                    return false;
                }
            } break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_REPEAT:
        case GGML_OP_DUP:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ALIBI:
        case GGML_OP_IM2COL:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_LEAKY_RELU:
            return true;
        default:
            return false;
    }

    UNUSED(backend);
}

static ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .get_default_buffer_type = */ ggml_backend_sycl_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ ggml_backend_sycl_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_sycl_graph_plan_free,
    /* .graph_plan_compute      = */ ggml_backend_sycl_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .supports_op             = */ ggml_backend_sycl_supports_op,
};

ggml_backend_t ggml_backend_sycl_init(int device) {
    ggml_init_sycl(); // TODO: remove from ggml.c

    if (device < 0 || device >= ggml_sycl_get_device_count()) {
        fprintf(stderr, "%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    // not strictly necessary, but it may reduce the overhead of the first graph_compute
    ggml_sycl_set_main_device(device);

    ggml_backend_context_sycl * ctx = new ggml_backend_context_sycl {
        /* .device = */ device
    };

    ggml_backend_t sycl_backend = new ggml_backend {
        /* .interface = */ ggml_backend_sycl_interface,
        /* .context   = */ ctx
    };

    return sycl_backend;
}

bool ggml_backend_is_sycl(ggml_backend_t backend) {
    return backend->iface.get_name == ggml_backend_sycl_name;
}

static ggml_backend_t ggml_backend_reg_sycl_init(const char * params, void * user_data) {
    ggml_backend_t sycl_backend = ggml_backend_sycl_init((int) (intptr_t) user_data);
    return sycl_backend;

    UNUSED(params);
}

extern "C" int ggml_backend_sycl_reg_devices();

int ggml_backend_sycl_reg_devices() {
    int device_count = ggml_sycl_get_device_count();

    for (int i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%d", GGML_SYCL_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_sycl_init, ggml_backend_sycl_buffer_type(i), (void *) (intptr_t) i);
    }
    return device_count;
}
