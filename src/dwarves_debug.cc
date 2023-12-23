#include "dwarves_debug.h"

Debug * Debug::instance = nullptr;

Debug* Debug::get_instance() {
	if (instance == nullptr) {
		instance = new Debug();
		return instance;
	}
	return instance;
}

void Debug::enter_function(std::string func_name) {
	log(">>> entering function ", func_name, "...");
}

void Debug::leave_function(std::string func_name) {
	log("<<< leaving function ", func_name, "...");
}
