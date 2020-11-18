// (c) 2020 Ruslan Guseinov, IST Austria
// This code is licensed under MIT license (see LICENSE for details)

#ifndef CGB_SHELL_DATA_H
#define CGB_SHELL_DATA_H

#include "shell_options.h"
#include "cgb_cereal.h"

struct ShellData : public ShellOptions {
	Eigen::MatrixXd VB0; // initial flat boundary
	Eigen::MatrixXd VX0; // target boundary
	Eigen::MatrixXd NX0; // target vertex normals
	Eigen::MatrixXd NB0; // target boundary edge normals
	Eigen::MatrixXi F;

	Eigen::MatrixXi TT;
	Eigen::MatrixXi TTi;
	Eigen::MatrixXi vind;
	Eigen::MatrixXi E;
	Eigen::MatrixXi FE;

	Eigen::VectorXi B;   // Boundary vertex index list
	Eigen::VectorXi I;   // Internal vertex index list

	Eigen::VectorXi BVind; // for each boundary edge, inner vertex index forming a triangle with the edge

	// Express internal rest shape vertices via boundary and constant
	Eigen::MatrixXd BAX; // Flitered matrix VI = BAX_unflitered * VB, s.t. distant boundary vertices are ignored in computation of internal vertices
	Eigen::MatrixXd V0c; // VI = BAX * VB + V0c

	template <class Archive>
	void serialize(Archive& archive) {
		archive(cereal::make_nvp("shell options", cereal::base_class<ShellOptions>(this)));
		archive(cereal::make_nvp("VB0", VB0));
		archive(cereal::make_nvp("VX0", VX0));
		archive(cereal::make_nvp("NX0", NX0));
		archive(cereal::make_nvp("F", F));
	}
};

#endif