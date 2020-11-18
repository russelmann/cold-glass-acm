// (c) 2020 Ruslan Guseinov, IST Austria
// This code is licensed under MIT license (see LICENSE for details)

#ifndef CGB_SHELL_MODEL_H
#define CGB_SHELL_MODEL_H

#include "cgb_cereal.h"
#include "shell_data.h"
#include "cereal_eigen.h"

struct ShellModel : public cereal::CerealRW<ShellModel> {
	Eigen::MatrixXd V0;  // rest state
	Eigen::MatrixXd VX;  // deformed state
	Eigen::VectorXd DX;  // distance along normal
	Eigen::VectorXd phi; // extra angle dof per edge
	std::shared_ptr<ShellData> mShellData;

	ShellModel() : CerealRW("shell model")
	{ }

	template<class Archive>
	void serialize(Archive& archive) {
		archive(cereal::make_nvp("V0", V0));
		archive(cereal::make_nvp("VX", VX));
		archive(cereal::make_nvp("DX", DX));
		archive(cereal::make_nvp("phi", phi));
		archive(cereal::make_nvp("shell data", mShellData));
	}
};

#endif