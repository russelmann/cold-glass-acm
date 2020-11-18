// (c) 2020 Ruslan Guseinov, IST Austria
// This code is licensed under MIT license (see LICENSE for details)

#ifndef CGB_SHELL_OPTIONS_H
#define CGB_SHELL_OPTIONS_H

#include "cgb_cereal.h"

struct ShellOptions : public cereal::CerealRW<ShellOptions> {
	bool soft_boundary;
	bool preserve_boundary_normal;
	bool free_rest;
	bool quality_term;
	bool free_edges;
	bool exact_shape_operator;
	bool parameter_interior; // project interior vertices as a solver iteration post-processing step

	double lame1;
	double lame2;

	double h; // thickness

	int n_boundary = 20; // number of boundary edges per side

	double spring_strength;
	double boundary_normal_strength;
	double quality_coeff;

	bool suppress_stdout;

	ShellOptions() : CerealRW("shell options") {
		soft_boundary = true;
		preserve_boundary_normal = false;
		free_rest = false;
		quality_term = false;
		free_edges = true;
		exact_shape_operator = false;
		parameter_interior = true;

		setYoungPoisson(1.0, 0.208);
		h = 0.1;

		spring_strength = 1000.0;
		boundary_normal_strength = 1e9;
		quality_coeff = 10000.0;

		suppress_stdout = true;
	}

	void setYoungPoisson(double Y, double nu) {
		lame1 = Y * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
		lame2 = Y / (2.0 * (1.0 + nu));
	}

	template<class Archive>
	void serialize(Archive& archive) {
		archive(cereal::make_nvp("soft boundary", soft_boundary));
		archive(cereal::make_nvp("preserve boundary normal", preserve_boundary_normal));
		archive(cereal::make_nvp("optimize rest configuration", free_rest));
		archive(cereal::make_nvp("energy quality term", quality_term));
		archive(cereal::make_nvp("edge extra dofs", free_edges));
		archive(cereal::make_nvp("exact shape operator", exact_shape_operator));
		archive(cereal::make_nvp("parameterize interior", parameter_interior));

		archive(cereal::make_nvp("lame1", lame1));
		archive(cereal::make_nvp("lame2", lame2));

		archive(cereal::make_nvp("thickness", h));

		archive(cereal::make_nvp("number of boundary edges", n_boundary));

		archive(cereal::make_nvp("energy spring strength", spring_strength));
		archive(cereal::make_nvp("energy boundary normal strength", boundary_normal_strength));
		archive(cereal::make_nvp("energy quality factor", quality_coeff));

		archive(cereal::make_nvp("suppress stdout", suppress_stdout));
	}
};

#endif