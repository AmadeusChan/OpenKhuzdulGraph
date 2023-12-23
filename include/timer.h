#ifndef TIMER_H
#define TIMER_H

#include <string>
#include <utility>
#include <string>
#include <vector>

double get_time();

class Timer {
	private:
		static std::vector<std::pair<std::string, std::pair<double, int>>> timers_;
	public:
		static void timer_start(std::string timer_name);
		static void timer_stop(std::string timer_name);
		static void report_timers();
};

#endif
