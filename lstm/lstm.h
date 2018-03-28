#ifndef LSTM_H
#define LSTM_H

#include "Eigen/Dense"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <unordered_map>

class LSTM
{
	private:

	struct Candidate {
		int candidate_num;
		std::string sequence;
		double probability;
		Eigen::ArrayXd state;
		Eigen::ArrayXd hidden;
	};

	size_t m_input_size;
	size_t m_hidden_size;
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

	// Hidden Output Vector (LSTM Output)
	Eigen::ArrayXd m_h_t;

	// Activations
	Eigen::ArrayXd m_a_t;
	Eigen::ArrayXd m_i_t;
	Eigen::ArrayXd m_f_t;
	Eigen::ArrayXd m_o_t;

	// Fully Connected Layer Layer Weight and Bias
	Eigen::MatrixXd m_Wy;
	Eigen::ArrayXd m_by;

	// Fully Connected Layer Output
	Eigen::ArrayXd m_y_t;
	
	// Softmax Probabilities Vector
	Eigen::ArrayXd m_output;

	// Learning rate
	float m_rate;

	// Softmax temperature
	float m_temperature;

	// File containing training samples
	std::ifstream m_infile;

	// Vocab list containing all possible characters
	std::unordered_map<char, int> m_vocabs;
	std::unordered_map<int, char> m_vocabs_indices;

	// File where state is saved to
	std::string m_state_file;
	std::string m_sample_file;

	void reset(void);

	void feedforward(Eigen::ArrayXd &input);

	void backpropogate(
			std::vector<Eigen::ArrayXd> &a_t_cache,
			std::vector<Eigen::ArrayXd> &i_t_cache,
			std::vector<Eigen::ArrayXd> &f_t_cache,
			std::vector<Eigen::ArrayXd> &o_t_cache,
			std::vector<Eigen::ArrayXd> &h_t_cache,
			std::vector<Eigen::ArrayXd> &state_cache,
			std::vector<Eigen::ArrayXd> &input_cache,
			std::vector<Eigen::ArrayXd> &prob_cache,
			std::vector<char> &label_cache,
			const int lookback
	);

	void saveState(void);

	template<typename T>
	void writeData(const T &data, const std::string &id, std::ofstream &outfile);

	template<typename T>
	void loadData(T &parameter, std::istringstream &data_stream);

	template<typename T>
	void clipGradients(T &parameter);

	// Helper functions
	static double sigmoid(double num);

	Eigen::ArrayXd softmax(const Eigen::ArrayXd &input);

	double crossEntropy(const Eigen::ArrayXd &output, const char &label);

	Eigen::ArrayXd charToVector(const char &c);

	char vectorToChar(const Eigen::ArrayXd &v);

	void fillVocabList(std::unordered_map<char, int> &vocabs, std::unordered_map<int, char> &indices, std::ifstream &infile);

	void initiateMatrices(void);

	public:

	LSTM(size_t hidden_size, float learning_rate);

	void load(const std::string &filename);

	void train(const size_t epochs, const size_t num_steps, const size_t lookback, const int reset_num);

	void saveStateTo(const std::string &filename);

	void loadState(const std::string &filename);

	void setSoftmaxTemperature(const float temp);

	void output(const size_t iterations);

	void beamSearchOutput(const size_t beams, const size_t iterations);
};

#endif
