// (c) 2020 Ruslan Guseinov, IST Austria
// This code is licensed under MIT license (see LICENSE for details)

#include <fstream>
#include "shell_model.h"

void write_obj(std::ostream& output, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
	static Eigen::IOFormat vertex_format(Eigen::FullPrecision, Eigen::DontAlignCols, " ", "\n", "v ", "", "", "\n");
	static Eigen::IOFormat face_format(Eigen::FullPrecision, Eigen::DontAlignCols, " ", "\n", "f ", "", "", "\n");
	output << V.format(vertex_format);
	output << (F.array() + 1).format(face_format);
}

int main(int argc, char* argv[]) {
	if (argc == 1 || argc == 3 || argc > 4) {
		std::cout << "CgbSimReader <shell.bin> [-o <path>] [-j <shell.json>]\n";
		std::cout << "Read data from <shell.bin>\n";
		std::cout << "-o : output deformed and undeformed shell as OBJ files to folder <path> (edge angle deviations are lost)\n";
		std::cout << "-j : unpack binary data and save in JSON format as <shell.json>\n";
		return 0;
	}

	char* shell_bin_filename = argv[1];

	ShellModel sm;

	if (!sm.readBIN(shell_bin_filename)) {
		std::cout << "Error reading binary file <" << shell_bin_filename << ">\n";
		return 1;
	}

	char* option = argv[2];
	
	if (!strcmp(option, "-o")) {
		char* output_path = argv[3];
		std::string output_filename;
		std::ofstream ofs;

		output_filename = std::string(output_path) + "/deformed.obj";
		ofs = std::ofstream(output_filename);
		if (ofs.good()) {
			write_obj(ofs, sm.VX, sm.mShellData->F);
		}
		else {
			std::cout << "Error: could not write file <" << output_filename << ">\n";
		}

		output_filename = std::string(output_path) + "/undeformed.obj";
		ofs = std::ofstream(output_filename);
		if (ofs.good()) {
			Eigen::MatrixXd V = Eigen::MatrixXd::Zero(sm.V0.rows(), 3);
			V.leftCols(2) = sm.V0;
			write_obj(ofs, V, sm.mShellData->F);
		}
		else {
			std::cout << "Error: could not write file <" << output_filename << ">\n";
		}
	}
	else if (!strcmp(option, "-j")) {
		char* shell_json_filename = argv[3];
		if (sm.writeJSON(shell_json_filename))
			std::cout << "File successfully written\n";
		else
			std::cout << "Error writing file <" << shell_json_filename << ">\n";
	}
	else {
		std::cout << "Error: unrecognized option '" << option << "'\n";
		return 1;
	}

	return 0;
}