// (c) 2020 Ruslan Guseinov, IST Austria
// This code is licensed under MIT license (see LICENSE for details)

#ifndef	CGB_CEREAL_H
#define CGB_CEREAL_H

#include <fstream>

#include "cereal_eigen.h"
#include "cereal/cereal.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal/types/map.hpp"
#include "cereal/types/memory.hpp"

namespace cereal {

	template<typename T>
	class CerealRW {
	public:

		CerealRW(std::string data_name) {
			mDataName = data_name;
		}

		std::string writeJSONs() {
			std::stringstream ss; {
				cereal::JSONOutputArchive archive(ss);
				try {
					archive(cereal::make_nvp(mDataName, child()));
				}
				catch (std::exception& e) {
					return std::string();
				}
			}
			return ss.str();
		}

		bool readJSON(std::string file_name) {
			std::ifstream ifs(file_name);
			if (ifs.is_open()) {
				cereal::JSONInputArchive archive(ifs);
				try {
					archive(cereal::make_nvp(mDataName, child()));
					return true;
				}
				catch (std::exception& e) {
					return false;
				}
			}
			return false;
		}

		bool writeJSON(std::string file_name) {
			std::ofstream ofs(file_name);
			if (ofs.is_open()) {
				cereal::JSONOutputArchive archive(ofs);
				try {
					archive(cereal::make_nvp(mDataName, child()));
					return true;
				}
				catch (std::exception& e) {
					return false;
				}
			}
			return false;
		}

		bool readBIN(std::string file_name)
		{
			std::ifstream ifs(file_name, std::ios::binary);
			if (ifs.is_open()) {
				cereal::BinaryInputArchive archive(ifs);
				try {
					archive(cereal::make_nvp(mDataName, child()));
					return true;
				}
				catch (std::exception& e) {
					return false;
				}
			}
			return false;
		}

		bool writeBIN(std::string file_name)
		{
			std::ofstream ofs(file_name, std::ios::binary);
			if (ofs.is_open()) {
				cereal::BinaryOutputArchive archive(ofs);
				try {
					archive(cereal::make_nvp(mDataName, child()));
					return true;
				}
				catch (std::exception& e) {
					return false;
				}
			}
			return false;
		}

	private:
		std::string mDataName;

		inline T& child() { return *static_cast<T*>(this); }
	};
}

#endif