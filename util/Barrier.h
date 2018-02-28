#ifndef __BARRIER_H__
#define __BARRIER_H__

#include <cassert>
#if defined(PTHREAD_BARRIER) || defined(PTHREADCV_BARRIER)
#include <pthread.h>
#else
#include <boost/thread.hpp>
#endif

#if defined(PTHREAD_BARRIER)
class Barrier{
public:
	Barrier(int thread_count){
        thread_num_ = thread_count;
        pthread_barrier_init(&barrier, NULL, thread_num_);
    }
	~Barrier(){
        pthread_barrier_destroy(&barrier);
	}

	void Arrive(){
        int ret = pthread_barrier_wait(&barrier);
	    assert(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD);
    }

private:
	int thread_num_;
    pthread_barrier_t barrier;
};
#elif defined(PTHREADCV_BARRIER)
class Barrier{
public:
	Barrier(int thread_count){
	    int ret = pthread_mutex_init(&sync_lock_, NULL);
		assert(ret == 0);
		ret = pthread_cond_init(&sync_cv_, NULL);
		assert(ret == 0);
		arrive_count_ = 0;
		thread_num_ = thread_count;
    }
	~Barrier(){
		pthread_mutex_destroy(&sync_lock_);
		pthread_cond_destroy(&sync_cv_);
	}

	void Arrive(){
		assert(thread_num_ > 0);
		pthread_mutex_lock(&sync_lock_);
		++arrive_count_;
		if (arrive_count_ == thread_num_){
			arrive_count_ = 0;
			pthread_cond_broadcast(&sync_cv_);
		}
		else{
			pthread_cond_wait(&sync_cv_, &sync_lock_);
		}
		pthread_mutex_unlock(&sync_lock_);
    }

private:
	int thread_num_;
	pthread_mutex_t sync_lock_;
	pthread_cond_t sync_cv_;
	volatile int arrive_count_;
};
#else
class Barrier{
public:
	Barrier(const size_t& n) : thread_num_(n), called_(0){

	}
	
	void Arrive(){
		boost::mutex::scoped_lock lock(mtx_);
		++called_;
		if (called_ == thread_num_){
			called_ = 0;
			cv_.notify_all();
		}
		else{
			cv_.wait(lock);
		}
	}

private:
	boost::mutex mtx_;
	boost::condition_variable cv_;
	size_t thread_num_;
	size_t called_;
};
#endif

#endif
