/*
 * Copyright (C) 2019 by AutoSense Organization. All rights reserved.
 * Gary Chan <chenshj35@mail2.sysu.edu.cn>
 */
#ifndef COMMON_LIBS_INCLUDE_COMMON_THREAD_POOL_HPP_
#define COMMON_LIBS_INCLUDE_COMMON_THREAD_POOL_HPP_

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

namespace autosense {
namespace common {

class ThreadPool {
 public:
    explicit ThreadPool(size_t size) : work_(io_service_) {
        for (size_t i = 0; i < size; ++i) {
            workers_.create_thread(
                boost::bind(&boost::asio::io_service::run, &io_service_));
        }
    }

    ~ThreadPool() {
        io_service_.stop();
        workers_.join_all();
    }

    // Add new work item to the pool.
    template <class F>
    void Enqueue(F f) {
        io_service_.post(f);
    }

 private:
    boost::thread_group workers_;
    boost::asio::io_service io_service_;
    boost::asio::io_service::work work_;
};

}  // namespace common
}  // namespace autosense

#endif  // COMMON_LIBS_INCLUDE_COMMON_THREAD_POOL_HPP_
