// (c) 2020 Ruslan Guseinov, IST Austria
// This code is licensed under MIT license (see LICENSE for details)

#ifndef CEREAL_EIGEN_H
#define CEREAL_EIGEN_H

#include "Eigen/Eigen"
#include "cereal/types/vector.hpp"

namespace cereal {
	
	template<class Archive, typename Scalar>
	void save(Archive& ar, Eigen::Matrix<Scalar, -1, -1> const & matrix) {
		ar(cereal::make_nvp("rows", matrix.rows()));
		ar(cereal::make_nvp("cols", matrix.cols()));
		ar(cereal::make_nvp("data", std::vector<Scalar>(matrix.data(), matrix.data() + matrix.size())));
	}

	template<class Archive, typename Scalar>
	void load(Archive& ar, Eigen::Matrix<Scalar, -1, -1>& matrix) {
		Eigen::Index rows;
		Eigen::Index cols;
		ar(cereal::make_nvp("rows", rows));
		ar(cereal::make_nvp("cols", cols));

		if (rows == 0 || cols == 0) return;

		if (rows != matrix.rows() || cols != matrix.cols())
			matrix.resize(rows, cols);

		std::vector<Scalar> vec;
		ar(cereal::make_nvp("data", vec));
		memcpy(matrix.data(), vec.data(), vec.size() * sizeof(Scalar));
	}

	template<class Archive, typename Scalar>
	void save(Archive& ar, Eigen::Matrix<Scalar, -1, 1> const & vector) {
		ar(cereal::make_nvp("rows", vector.rows()));
		ar(cereal::make_nvp("data", std::vector<Scalar>(vector.data(), vector.data() + vector.size())));
	}

	template<class Archive, typename Scalar>
	void load(Archive& ar, Eigen::Matrix<Scalar, -1, 1>& vector) {
		Eigen::Index rows;
		ar(cereal::make_nvp("rows", rows));

		if (rows == 0) return;

		if (rows != vector.rows())
			vector.resize(rows);

		std::vector<Scalar> vec;
		ar(cereal::make_nvp("data", vec));
		memcpy(vector.data(), vec.data(), vec.size() * sizeof(Scalar));
	}
	
}

#endif