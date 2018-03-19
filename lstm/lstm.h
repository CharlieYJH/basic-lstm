#ifndef LSTM_H
#define LSTM_H

#include "Eigen/Dense"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

class LSTM
{
	private:

	size_t m_input_size;
	size_t m_output_size;

	// Input Weight Matrices
	Eigen::MatrixXd m_Wa;
	Eigen::MatrixXd m_Wi;
	Eigen::MatrixXd m_Wf;
	Eigen::MatrixXd m_Wo;

	// Recurrent Weight Matrices
	Eigen::MatrixXd m_Ra;
	Eigen::MatrixXd m_Ri;
	Eigen::MatrixXd m_Rf;
	Eigen::MatrixXd m_Ro;
	
	// Bias Vectors
	Eigen::ArrayXd m_ba;
	Eigen::ArrayXd m_bi;
	Eigen::ArrayXd m_bf;
	Eigen::ArrayXd m_bo;

	// Internal State Vector
	Eigen::ArrayXd m_state;

	// Output Vector
	Eigen::ArrayXd m_output;

	// Activations
	Eigen::ArrayXd m_a_t;
	Eigen::ArrayXd m_i_t;
	Eigen::ArrayXd m_f_t;
	Eigen::ArrayXd m_o_t;

	// Learning rate
	float m_rate;

	// File containing training samples
	std::ifstream m_infile;

	// File where state is saved to
	std::string m_state_file;

	void reset(void);

	void feedforward(Eigen::ArrayXd &input);

	void backpropogate(
			std::vector<Eigen::ArrayXd> &a_t_cache,
			std::vector<Eigen::ArrayXd> &i_t_cache,
			std::vector<Eigen::ArrayXd> &f_t_cache,
			std::vector<Eigen::ArrayXd> &o_t_cache,
			std::vector<Eigen::ArrayXd> &state_cache,
			std::vector<Eigen::ArrayXd> &input_cache,
			std::vector<Eigen::ArrayXd> &output_cache,
			std::vector<Eigen::ArrayXd> &loss_cache
	);

	void saveState(void);

	template<typename T>
	void writeData(const T &data, const std::string &id, std::ofstream &outfile);

	template<typename T>
	void loadData(T &parameter, std::istringstream &data_stream);

	// Helper functions
	static double sigmoid(double num);

	Eigen::ArrayXd charToVector(const char &c);

	char vectorToChar(const Eigen::ArrayXd &v);

	public:

	LSTM(size_t input_size, size_t output_size, float learning_rate);

	void load(const std::string &filename);

	void train(const size_t epochs, const size_t num_steps);

	void saveTo(const std::string &filename);

	void loadState(const std::string &filename);

	void output(const size_t iterations);
};

#endif
