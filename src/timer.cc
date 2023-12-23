#include "timer.h"

#include <assert.h>
#include <sys/time.h>

#include "dwarves_debug.h"

std::vector<std::pair<std::string, std::pair<double, int>>> Timer::timers_;

double get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (tv.tv_usec / 1e6);
}

void Timer::timer_start(std::string timer_name) {
	for (int i = 0, j = timers_.size(); i < j; ++ i) {
		if (timers_[i].first == timer_name) {
			timers_[i].second.first -= get_time();
			timers_[i].second.second += 1;
			return ;
		}
	}
	timers_.push_back(std::make_pair(timer_name, std::make_pair(-get_time(), 1)));
}

void Timer::timer_stop(std::string timer_name) {
	for (int i = 0, j = timers_.size(); i < j; ++ i) {
		if (timers_[i].first == timer_name) {
			timers_[i].second.first += get_time();
			return ;
		}
	}
	assert(false);
}

void Timer::report_timers() {
	Debug::get_instance()->print("timer_name,		time(ms)");
	for (std::pair<std::string, std::pair<double, int>> i: timers_) {
		if (i.second.first >= 0.01) {
			Debug::get_instance()->print(i.first, ",		", int(i.second.first * 1000 / i.second.second));
		} else {
			Debug::get_instance()->print(i.first, ",		", int(i.second.first * 1000 / i.second.second), ".", int(i.second.first * 1000000 / i.second.second) % 1000);
		}
	}
}

