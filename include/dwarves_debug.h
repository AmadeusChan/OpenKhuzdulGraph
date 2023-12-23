#ifndef DWARVES_DEBUG_H
#define DWARVES_DEBUG_H

#include <iostream>
#include <string>

#define DEBUG_MODE

class Debug {
	private:
		static Debug * instance;

		Debug() {
		}
		template<typename T>
			void rec_log(T t) {
				std::cout << t << std::endl;
			}
		template<typename T, typename... Args>
			void rec_log(T t, Args... args) {
				std::cout << t;
				rec_log(args...);
			}

	public:		
		static Debug * get_instance();

		template<typename... Args>
			void log(Args... args) {
#ifdef DEBUG_MODE
				std::cout << "[log] ";
				rec_log(args...);
#endif
			}
		template<typename ...Args>
			void print(Args... args) {
				rec_log(args...);
			}
		void enter_function(std::string func_name);
		void leave_function(std::string func_name);
};

#endif
