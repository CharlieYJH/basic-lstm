#include "lstm.h"

LSTM::LSTM(size_t input_size, size_t hidden_size, size_t output_size, float learning_rate)
{
	m_input_size = input_size;
	m_hidden_size = hidden_size;
	m_output_size = output_size;

	// Initialize input weight matrices
	m_Wa = Eigen::MatrixXd::Random(hidden_size, input_size);
	m_Wi = Eigen::MatrixXd::Random(hidden_size, input_size);
	m_Wf = Eigen::MatrixXd::Random(hidden_size, input_size);
	m_Wo = Eigen::MatrixXd::Random(hidden_size, input_size);

	// Initialize recurrent weight matrices
	m_Ra = Eigen::MatrixXd::Random(hidden_size, hidden_size);
	m_Ri = Eigen::MatrixXd::Random(hidden_size, hidden_size);
	m_Rf = Eigen::MatrixXd::Random(hidden_size, hidden_size);
	m_Ro = Eigen::MatrixXd::Random(hidden_size, hidden_size);

	// Initialize bias vectors
	m_ba = Eigen::ArrayXd::Random(hidden_size);
	m_bi = Eigen::ArrayXd::Random(hidden_size);
	m_bf = Eigen::ArrayXd::Random(hidden_size);
	m_bo = Eigen::ArrayXd::Random(hidden_size);

	// Initialize output vector
	m_state = Eigen::ArrayXd::Zero(hidden_size);
	m_h_t = Eigen::ArrayXd::Zero(hidden_size);

	// Initialize fully connected layer weights and biases
	m_Wy = Eigen::MatrixXd::Random(output_size, hidden_size);
	m_by = Eigen::ArrayXd::Random(output_size);

	// Initialize fully connected layer output and softmax probabilities vector
	m_y_t = Eigen::ArrayXd::Zero(output_size);
	m_output = Eigen::ArrayXd::Zero(output_size);

	// Set learning rate
	m_rate = learning_rate;

	// Setting default file where weights and biases are saved
	m_state_file = "./weights.txt";
}

void LSTM::load(const std::string &filename)
{
	m_infile.open(filename, std::ifstream::in);

	if (!m_infile)
		throw std::runtime_error(filename + " not found");

	return;
}

void LSTM::reset(void)
{
	m_state.setZero();
	m_h_t.setZero();
	m_y_t.setZero();
	m_output.setZero();

	return;
}

void LSTM::feedforward(Eigen::ArrayXd &input)
{
	// Input activation
	m_a_t = (m_Wa * input.matrix() + m_Ra * m_h_t.matrix()).array() + m_ba;
	m_a_t = m_a_t.tanh();

	// Input gate
	m_i_t = (m_Wi * input.matrix() + m_Ri * m_h_t.matrix()).array() + m_bi;
	m_i_t = m_i_t.unaryExpr(&LSTM::sigmoid);

	// Forget gate
	m_f_t = (m_Wf * input.matrix() + m_Rf * m_h_t.matrix()).array() + m_bf;
	m_f_t = m_f_t.unaryExpr(&LSTM::sigmoid);

	// Output gate
	m_o_t = (m_Wo * input.matrix() + m_Ro * m_h_t.matrix()).array() + m_bo;
	m_o_t = m_o_t.unaryExpr(&LSTM::sigmoid);
	
	// State update
	m_state = m_a_t * m_i_t + m_f_t * m_state;

	// LSTM output
	m_h_t = m_state.tanh() * m_o_t;

	// Fully connected layer output
	m_y_t = (m_Wy * m_h_t.matrix()).array() + m_by;

	// Apply softmax classifier on fully connected layer output to get a vector of probabilities
	m_output = softmax(m_y_t);

	return;
}

void LSTM::backpropogate(
		std::vector<Eigen::ArrayXd> &a_t_cache,
		std::vector<Eigen::ArrayXd> &i_t_cache,
		std::vector<Eigen::ArrayXd> &f_t_cache,
		std::vector<Eigen::ArrayXd> &o_t_cache,
		std::vector<Eigen::ArrayXd> &h_t_cache,
		std::vector<Eigen::ArrayXd> &state_cache,
		std::vector<Eigen::ArrayXd> &input_cache,
		std::vector<Eigen::ArrayXd> &prob_cache,
		std::vector<char> &label_cache
) {
	// LSTM output, output delta, and state differentials
	Eigen::ArrayXd d_h_t = Eigen::ArrayXd::Zero(m_hidden_size);
	Eigen::ArrayXd d_delta_t = Eigen::ArrayXd::Zero(m_hidden_size);
	Eigen::ArrayXd d_state_t = Eigen::ArrayXd::Zero(m_hidden_size);

	// Softmax gradient
	Eigen::ArrayXd d_y_t = Eigen::ArrayXd::Zero(m_output_size);

	// Input weight adjustment matrices
	Eigen::MatrixXd d_Wa = Eigen::MatrixXd::Zero(m_Wa.rows(), m_Wa.cols());
	Eigen::MatrixXd d_Wi = Eigen::MatrixXd::Zero(m_Wi.rows(), m_Wi.cols());
	Eigen::MatrixXd d_Wf = Eigen::MatrixXd::Zero(m_Wf.rows(), m_Wf.cols());
	Eigen::MatrixXd d_Wo = Eigen::MatrixXd::Zero(m_Wo.rows(), m_Wo.cols());

	// Recurrent weight adjustment matrices
	Eigen::MatrixXd d_Ra = Eigen::MatrixXd::Zero(m_Ra.rows(), m_Ra.cols());
	Eigen::MatrixXd d_Ri = Eigen::MatrixXd::Zero(m_Ri.rows(), m_Ri.cols());
	Eigen::MatrixXd d_Rf = Eigen::MatrixXd::Zero(m_Rf.rows(), m_Rf.cols());
	Eigen::MatrixXd d_Ro = Eigen::MatrixXd::Zero(m_Ro.rows(), m_Ro.cols());

	// Bias adjustment matrices
	Eigen::ArrayXd d_ba = Eigen::ArrayXd::Zero(m_ba.size());
	Eigen::ArrayXd d_bi = Eigen::ArrayXd::Zero(m_bi.size());
	Eigen::ArrayXd d_bf = Eigen::ArrayXd::Zero(m_bf.size());
	Eigen::ArrayXd d_bo = Eigen::ArrayXd::Zero(m_bo.size());

	// Fully connected layer weight and bias adjustments
	Eigen::MatrixXd d_Wy = Eigen::MatrixXd::Zero(m_Wy.rows(), m_Wy.cols());
	Eigen::ArrayXd d_by = Eigen::ArrayXd::Zero(m_by.size());

	for (int t = input_cache.size() - 1; t >= 0; t--) {

		// Softmax gradient
		d_y_t = prob_cache[t];
		d_y_t((int)label_cache[t]) -= 1;

		// Accumulate fully connected layer weight and bias adjustments
		d_Wy += d_y_t.matrix() * h_t_cache[t].matrix().transpose();
		d_by += d_y_t;

		// Output delta
		d_h_t = (m_Wy.transpose() * d_y_t.matrix()).array() + d_delta_t;

		// State delta
		if (t + 1 < input_cache.size())
			d_state_t = d_h_t * o_t_cache[t] * (1 - state_cache[t].tanh().square()) + d_state_t * f_t_cache[t + 1];
		else
			d_state_t = d_h_t * o_t_cache[t] * (1 - state_cache[t].tanh().square());

		// Input activation delta
		Eigen::ArrayXd d_a_t = d_state_t * i_t_cache[t] * (1 - a_t_cache[t].square());

		// Input gate delta
		Eigen::ArrayXd d_i_t = d_state_t * a_t_cache[t] * i_t_cache[t] * (1 - i_t_cache[t]);

		// Forget gate delta
		Eigen::ArrayXd d_f_t;

		if (t == 0)
			d_f_t = Eigen::ArrayXd::Zero(m_hidden_size);
		else
			d_f_t = d_state_t * state_cache[t - 1] * f_t_cache[t] * (1 - f_t_cache[t]);

		// Output gate delta
		Eigen::ArrayXd d_o_t = d_h_t * state_cache[t].tanh() * o_t_cache[t] * (1 - o_t_cache[t]);

		d_delta_t = m_Ra.transpose() * d_a_t.matrix() + m_Ri.transpose() * d_i_t.matrix() + m_Rf.transpose() * d_f_t.matrix() + m_Ro * d_o_t.matrix();

		// Accumulate the adjustment for the input weights
		d_Wa += d_a_t.matrix() * input_cache[t].matrix().transpose();
		d_Wi += d_i_t.matrix() * input_cache[t].matrix().transpose();
		d_Wf += d_f_t.matrix() * input_cache[t].matrix().transpose();
		d_Wo += d_o_t.matrix() * input_cache[t].matrix().transpose();

		// Accmuluate the djustment for the recurrent weights
		if (t > 0) {
			d_Ra += d_a_t.matrix() * h_t_cache[t - 1].matrix().transpose();
			d_Ri += d_i_t.matrix() * h_t_cache[t - 1].matrix().transpose();
			d_Rf += d_f_t.matrix() * h_t_cache[t - 1].matrix().transpose();
			d_Ro += d_o_t.matrix() * h_t_cache[t - 1].matrix().transpose();
		}

		// Accmuluate the adjustments for the biases
		d_ba += d_a_t;
		d_bi += d_i_t;
		d_bf += d_f_t;
		d_bo += d_o_t;
	}

	// Update weights and biases
	m_Wa -= (m_rate * d_Wa.array()).matrix();
	m_Wi -= (m_rate * d_Wi.array()).matrix();
	m_Wf -= (m_rate * d_Wf.array()).matrix();
	m_Wo -= (m_rate * d_Wo.array()).matrix();

	m_Ra -= (m_rate * d_Ra.array()).matrix();
	m_Ri -= (m_rate * d_Ri.array()).matrix();
	m_Rf -= (m_rate * d_Rf.array()).matrix();
	m_Ro -= (m_rate * d_Ro.array()).matrix();

	m_ba -= m_rate * d_ba;
	m_bi -= m_rate * d_bi;
	m_bf -= m_rate * d_bf;
	m_bo -= m_rate * d_bo;

	m_Wy -= (m_rate * d_Wy.array()).matrix();
	m_by -= m_rate * d_by;

	return;
}

void LSTM::train(size_t epochs, size_t num_steps)
{
	if (!m_infile)
		throw std::runtime_error("No training samples currently open");

	size_t iteration = 0;

	for (int i = 0; i < epochs; i++) {

		char curr_char = ' ';
		char next_char = ' ';

		// Iterate through entire training sample
		while (next_char != std::ifstream::traits_type::eof()) {

			// Reset hidden state and output at the start of each batch
			reset();

			std::vector<Eigen::ArrayXd> a_t_cache;
			std::vector<Eigen::ArrayXd> i_t_cache;
			std::vector<Eigen::ArrayXd> f_t_cache;
			std::vector<Eigen::ArrayXd> o_t_cache;
			std::vector<Eigen::ArrayXd> h_t_cache;

			std::vector<Eigen::ArrayXd> state_cache;
			std::vector<Eigen::ArrayXd> input_cache;
			std::vector<Eigen::ArrayXd> prob_cache;
			std::vector<char> label_cache;

			a_t_cache.reserve(num_steps);
			i_t_cache.reserve(num_steps);
			f_t_cache.reserve(num_steps);
			o_t_cache.reserve(num_steps);
			h_t_cache.reserve(num_steps);

			state_cache.reserve(num_steps);
			input_cache.reserve(num_steps);
			prob_cache.reserve(num_steps);
			label_cache.reserve(num_steps);

			double loss = 0;

			// Iterate through each batch
			for (int j = 0; j < num_steps; j++) {

				// Get current character and the next one to use as input and label
				m_infile.get(curr_char);
				next_char = m_infile.peek();

				// If we've reached the end of the training sample, end the batch early
				if (next_char == std::ifstream::traits_type::eof()) {
					m_infile.clear();
					m_infile.seekg(0);
					break;
				}

				Eigen::ArrayXd input = charToVector(curr_char);
				// Eigen::ArrayXd label = charToVector(next_char);

				// Forward pass of the network
				feedforward(input);

				// Calculate cross entropy loss for this time step
				loss += crossEntropy(m_output, next_char);

				// Pushback various network variables to use for backpropogation through time
				a_t_cache.push_back(m_a_t);
				i_t_cache.push_back(m_i_t);
				f_t_cache.push_back(m_f_t);
				o_t_cache.push_back(m_o_t);
				h_t_cache.push_back(m_h_t);

				state_cache.push_back(m_state);
				input_cache.push_back(input);
				prob_cache.push_back(m_output);
				label_cache.push_back(next_char);
			}

			// Calculate average loss for this batch and backpropogate
			loss /= num_steps;
			backpropogate(a_t_cache, i_t_cache, f_t_cache, o_t_cache, h_t_cache, state_cache, input_cache, prob_cache, label_cache);
			
			// Display the current iteration and loss
			if (iteration % 500 == 0) {
				std::cout << "Iter: " << iteration << " " << "Loss: " << loss << std::endl;
			}

			iteration++;
		}

		std::cout << "-------------------------------------------------------------------------" << std::endl;
		std::cout << "Epoch " << i + 1 << "/" << epochs << ". State saved to " << m_state_file << std::endl;
		std::cout << "-------------------------------------------------------------------------" << std::endl;
		saveState();
	}

	return;
}

void LSTM::output(const size_t iterations)
{
	reset();
	Eigen::ArrayXd input = charToVector('a' + std::rand() % 26);

	for (int i = 0; i < iterations; i++) {
		feedforward(input);
		char output = vectorToChar(m_output);
		input = charToVector(output);
		std::cout << output;
	}

	return;
}

void LSTM::saveTo(const std::string &filename)
{
	m_state_file = filename;
	return;
}

void LSTM::saveState(void)
{
	std::ofstream outfile(m_state_file);

	if (outfile.is_open()) {
		writeData(m_Wa, "Wa", outfile);
		writeData(m_Wi, "Wi", outfile);
		writeData(m_Wf, "Wf", outfile);
		writeData(m_Wo, "Wo", outfile);
		writeData(m_Ra, "Ra", outfile);
		writeData(m_Ri, "Ri", outfile);
		writeData(m_Rf, "Rf", outfile);
		writeData(m_Ro, "Ro", outfile);
		writeData(m_ba, "ba", outfile);
		writeData(m_bi, "bi", outfile);
		writeData(m_bf, "bf", outfile);
		writeData(m_bo, "bo", outfile);
		writeData(m_Wy, "Wy", outfile);
		outfile.close();
	} else {
		throw std::runtime_error("Unable to open " + m_state_file);
	}

	return;
}

template<typename T>
void LSTM::writeData(const T &data, const std::string &id, std::ofstream &outfile)
{
	outfile << id << '\t';

	for (int i = 0; i < data.rows(); i++) {
		for (int j = 0; j < data.cols(); j++) {
			outfile << data(i, j) << '\t';
		}
	}

	outfile << "\r\n";
	return;
}

void LSTM::loadState(const std::string &filename)
{
	std::ifstream infile(filename);

	if (infile.is_open()) {
		std::string line;
		std::string id;

		while (getline(infile, line)) {
			std::istringstream data(line);
			data >> id;

			if (id == "Wa")
				loadData(m_Wa, data);
			else if (id == "Wi")
				loadData(m_Wi, data);
			else if (id == "Wf")
				loadData(m_Wf, data);
			else if (id == "Wo")
				loadData(m_Wo, data);
			else if (id == "Ra")
				loadData(m_Ra, data);
			else if (id == "Ri")
				loadData(m_Ri, data);
			else if (id == "Rf")
				loadData(m_Rf, data);
			else if (id == "Ro")
				loadData(m_Ro, data);
			else if (id == "ba")
				loadData(m_ba, data);
			else if (id == "bi")
				loadData(m_bi, data);
			else if (id == "bf")
				loadData(m_bf, data);
			else if (id == "bo")
				loadData(m_bo, data);
			else if (id == "Wy")
				loadData(m_Wy, data);
			else if (id == "by")
				loadData(m_by, data);
		}

		infile.close();
	} else {
		throw std::runtime_error("Unable to open file " + filename);
	}
}

template<typename T>
void LSTM::loadData(T &parameter, std::istringstream &data_stream)
{
	for (int i = 0; i < parameter.rows(); i++) {
		for (int j = 0; j < parameter.cols(); j++) {
			data_stream >> parameter(i, j);
		}
	}

	return;
}

Eigen::ArrayXd LSTM::softmax(const Eigen::ArrayXd &input)
{
	Eigen::ArrayXd probabilities = input.exp();
	double sum = probabilities.sum();

	return probabilities / sum;
}

double LSTM::crossEntropy(const Eigen::ArrayXd &output, const char &label)
{
	return -std::log(output(label));
}

double LSTM::sigmoid(double num)
{
	return 1 / (1 + std::exp(-num));
}

Eigen::ArrayXd LSTM::charToVector(const char &c)
{
	Eigen::ArrayXd one_hot_vector = Eigen::ArrayXd::Zero(m_input_size);
	one_hot_vector((int)c) = 1;

	return one_hot_vector;
}

char LSTM::vectorToChar(const Eigen::ArrayXd &v)
{
	int max_index;
	double max_num = v(0);

	for (int i = 1; i < v.size(); i++) {
		if (v(i) > max_num) {
			max_num = v(i);
			max_index = i;
		}
	}

	return (char)max_index;
}
