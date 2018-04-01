#include "lstm.h"

LSTM::LSTM(size_t hidden_size, float learning_rate)
	: m_hidden_size(hidden_size),
	  m_rate(learning_rate),
	  m_temperature(1.0),
	  m_beta1(0.9),
	  m_beta2(0.999),
	  m_epsilon(0.00000001),
	  m_update_iteration(0),
	  m_state_file("./weights.txt"),
	  m_sample_loaded(false)
{
}

void LSTM::load(const std::string &filename)
{
	m_infile.open(filename, std::ifstream::in);
	m_sample_file = filename;

	if (!m_infile)
		throw std::runtime_error(filename + " not found");

	// Go through the file and fill the vocab list
	fillVocabList(m_vocabs, m_vocabs_indices, m_infile);

	// Initiate all weights and biases according to the sizes given
	initiateMatrices();

	m_infile.close();

	m_sample_loaded = true;
}

void LSTM::reset(void)
{
	m_state.setZero();
	m_h_t.setZero();
	m_y_t.setZero();
	m_output.setZero();
}

void LSTM::resetMomentums(void)
{
	m_m_Wa.setZero();
	m_m_Wi.setZero();
	m_m_Wf.setZero();
	m_m_Wo.setZero();

	m_m_Ra.setZero();
	m_m_Ri.setZero();
	m_m_Rf.setZero();
	m_m_Ro.setZero();

	m_m_ba.setZero();
	m_m_bi.setZero();
	m_m_bf.setZero();
	m_m_bo.setZero();

	m_v_Wa.setZero();
	m_v_Wi.setZero();
	m_v_Wf.setZero();
	m_v_Wo.setZero();

	m_v_Ra.setZero();
	m_v_Ri.setZero();
	m_v_Rf.setZero();
	m_v_Ro.setZero();

	m_v_ba.setZero();
	m_v_bi.setZero();
	m_v_bf.setZero();
	m_v_bo.setZero();

	m_m_Wy.setZero();
	m_m_by.setZero();

	m_v_Wy.setZero();
	m_v_by.setZero();
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
		std::vector<char> &label_cache,
		const int lookback
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

	int timesteps = input_cache.size();
	int window = (timesteps - lookback >= 0) ? timesteps - lookback : 0;

	for (int t = timesteps - 1; t >= window; t--) {

		// Softmax gradient
		d_y_t = prob_cache[t];
		d_y_t(m_vocabs[label_cache[t]]) -= 1;

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

	// Clip all gradients to prevent gradient explosions
	clipGradients(d_Wa);
	clipGradients(d_Wi);
	clipGradients(d_Wf);
	clipGradients(d_Wo);
	clipGradients(d_Ra);
	clipGradients(d_Ri);
	clipGradients(d_Rf);
	clipGradients(d_Ro);
	clipGradients(d_ba);
	clipGradients(d_bi);
	clipGradients(d_bf);
	clipGradients(d_bo);
	clipGradients(d_Wy);
	clipGradients(d_by);

	m_update_iteration++;

	adamUpdate(d_Wa, m_m_Wa, m_v_Wa, m_Wa);
	adamUpdate(d_Wi, m_m_Wi, m_v_Wi, m_Wi);
	adamUpdate(d_Wf, m_m_Wf, m_v_Wf, m_Wf);
	adamUpdate(d_Wo, m_m_Wo, m_v_Wo, m_Wo);

	adamUpdate(d_Ra, m_m_Ra, m_v_Ra, m_Ra);
	adamUpdate(d_Ri, m_m_Ri, m_v_Ri, m_Ri);
	adamUpdate(d_Rf, m_m_Rf, m_v_Rf, m_Rf);
	adamUpdate(d_Ro, m_m_Ro, m_v_Ro, m_Ro);

	adamUpdate(d_ba, m_m_ba, m_v_ba, m_ba);
	adamUpdate(d_bi, m_m_bi, m_v_bi, m_bi);
	adamUpdate(d_bf, m_m_bf, m_v_bf, m_bf);
	adamUpdate(d_bo, m_m_bo, m_v_bo, m_bo);

	adamUpdate(d_Wy, m_m_Wy, m_v_Wy, m_Wy);
	adamUpdate(d_by, m_m_by, m_v_by, m_by);

	// Update weights and biases
	// m_Wa -= (m_rate * d_Wa.array()).matrix();
	// m_Wi -= (m_rate * d_Wi.array()).matrix();
	// m_Wf -= (m_rate * d_Wf.array()).matrix();
	// m_Wo -= (m_rate * d_Wo.array()).matrix();

	// m_Ra -= (m_rate * d_Ra.array()).matrix();
	// m_Ri -= (m_rate * d_Ri.array()).matrix();
	// m_Rf -= (m_rate * d_Rf.array()).matrix();
	// m_Ro -= (m_rate * d_Ro.array()).matrix();

	// m_ba -= m_rate * d_ba;
	// m_bi -= m_rate * d_bi;
	// m_bf -= m_rate * d_bf;
	// m_bo -= m_rate * d_bo;

	// m_Wy -= (m_rate * d_Wy.array()).matrix();
	// m_by -= m_rate * d_by;
}

void LSTM::adamUpdate(Eigen::MatrixXd &gradient, Eigen::MatrixXd &m_t, Eigen::MatrixXd &v_t, Eigen::MatrixXd &weight)
{
	double iteration = (double)m_update_iteration;
	double m_t_correct = 1 - std::pow(m_beta1, iteration);
	double v_t_correct = 1 - std::pow(m_beta2, iteration);
	double rate = m_rate * v_t_correct / m_t_correct;
	
	// Calculate moving average for the momentums and do bias correction
	m_t = (m_beta1 * m_t.array() + (1 - m_beta1) * gradient.array());
	v_t = (m_beta2 * v_t.array() + (1 - m_beta2) * gradient.array().square());

	// Perform the update
	weight -= (rate * (m_t.array() / (v_t.array().sqrt() + m_epsilon))).matrix();
}

void LSTM::adamUpdate(Eigen::ArrayXd &gradient, Eigen::ArrayXd &m_t, Eigen::ArrayXd &v_t, Eigen::ArrayXd &bias)
{
	double iteration = (double)m_update_iteration;
	double m_t_correct = 1 - std::pow(m_beta1, iteration);
	double v_t_correct = 1 - std::pow(m_beta2, iteration);
	double rate = m_rate * v_t_correct / m_t_correct;

	// Calculate moving average for the momentums and do bias correction
	m_t = (m_beta1 * m_t + (1 - m_beta1) * gradient);
	v_t = (m_beta2 * v_t + (1 - m_beta2) * gradient.square());

	// Perform the update
	bias -= rate * (m_t / (v_t.sqrt() + m_epsilon));
}

void LSTM::train(const size_t epochs, const size_t num_steps, const size_t lookback, const int reset_num)
{
	m_infile.open(m_sample_file, std::ifstream::in);

	if (!m_infile)
		throw std::runtime_error("No training samples currently open");

	size_t iteration = 0;

	for (int i = 0; i < epochs; i++) {

		// Reset hidden state and output at the start of each epoch
		reset();

		float temperature = m_temperature;
		m_temperature = 1.0;

		char curr_char = ' ';
		char next_char = ' ';
		double loss = 0;
		double last_loss = 0;
		int batch_num = 0;
		std::streampos start_pos = m_infile.tellg();

		// Iterate through entire training sample
		while (next_char != std::ifstream::traits_type::eof()) {

			// m_infile.seekg(start_pos);
			// start_pos += 1;

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

			last_loss = loss;
			loss = 0;

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
			backpropogate(a_t_cache, i_t_cache, f_t_cache, o_t_cache, h_t_cache, state_cache, input_cache, prob_cache, label_cache, lookback);
			
			// Display the current iteration and loss
			// if (iteration % 1000 == 0) {
				// std::cout << "Iter: " << iteration << " " << "Loss: " << loss << std::endl;
			// }

			iteration++;
			batch_num++;

			// Reset the internal states after reset_num numbers of mini batches are done
			// Doesn't reset between batches if reset_num is 0 or less
			if (reset_num > 0 && batch_num % reset_num == 0)
				reset();
		}

		m_temperature = temperature;

		saveState();
		std::cout << "-------------------------------------------------------------------------" << std::endl;
		std::cout << "Epoch " << i + 1 << "/" << epochs << ". State saved to " << m_state_file << ". Loss: " << last_loss << ". LR: " << m_rate << std::endl;
		std::cout << "-------------------------------------------------------------------------" << std::endl;
		std::cout << "Sample: " << beamSearchOutput(4, 200) << std::endl;
		std::cout << "-------------------------------------------------------------------------" << std::endl;
	}

	m_infile.close();
}

void LSTM::setSoftmaxTemperature(const float temp)
{
	m_temperature = temp;
}

void LSTM::setAdamParams(const double beta1, const double beta2, const double epsilon)
{
	m_beta1 = beta1;
	m_beta2 = beta2;
	m_epsilon = epsilon;
}

std::string LSTM::output(const size_t iterations)
{
	// if (!m_sample_loaded)
		// throw std::runtime_error("No sample file loaded. Unable to generate input and output vectors.");

	std::string output = "";

	Eigen::ArrayXd input = charToVector(m_vocabs_indices[std::rand() % m_input_size]);

	for (int i = 0; i < iterations; i++) {
		feedforward(input);
		char outchar = vectorToChar(m_output);
		input = charToVector(outchar);
		output += outchar;
	}

	return output;
}

std::string LSTM::beamSearchOutput(const size_t beams, const size_t iterations)
{
	// if (!m_sample_loaded)
		// throw std::runtime_error("No sample file loaded. Unable to generate input and output vectors.");

	const char seed = m_vocabs_indices[std::rand() % m_input_size];
	std::vector<Candidate> top_candidates(beams);
	std::vector<Eigen::ArrayXd> cell_states(beams);
	std::vector<Eigen::ArrayXd> hidden_states(beams);

	// Initiate the top_candidates vector
	for (int i = 0; i < top_candidates.size(); i++) {
		Candidate &candidate = top_candidates[i];
		candidate.candidate_num = i;
		candidate.sequence += seed;
		candidate.probability = 0;
		candidate.state = m_state;
		candidate.hidden = m_h_t;
	}

	for (int i = 0; i < iterations; i++) {

		std::vector<Candidate> all_candidates;

		for (int j = 0; j < top_candidates.size(); j++) {

			// Iterate through each top candidate
			Candidate &candidate = top_candidates[j];

			// Input to network is the last character of the candidate
			Eigen::ArrayXd input = charToVector(candidate.sequence[candidate.sequence.length() - 1]);

			// Load saved states from that candidate
			m_state = candidate.state;
			m_h_t = candidate.hidden;

			// Feedforward and get the output probabilities
			feedforward(input);

			// Save the state from this candidate
			cell_states[j] = m_state;
			hidden_states[j] = m_h_t;

			for (int k = 0; k < m_output.size(); k++) {
				Candidate curr_candidate;
				curr_candidate.candidate_num = j;
				curr_candidate.sequence = candidate.sequence + m_vocabs_indices[k];
				curr_candidate.probability = candidate.probability - std::log(m_output(k));
				all_candidates.push_back(curr_candidate);
			}

			// If this is the first iteration, all candidates generate the same outputs, so just do it for the first candidate
			if (i == 0)
				break;
		}

		// Sort all candidates by the output probabilities
		std::sort(all_candidates.begin(), all_candidates.end(), 
				[] (const Candidate &a, const Candidate &b) {
					return a.probability < b.probability;
		});

		// Update the top candidates with the info from the sorted list of all candidates
		for (int j = 0; j < top_candidates.size(); j++) {

			Candidate &old_candidate = top_candidates[j];
			Candidate &new_candidate = all_candidates[j];

			old_candidate.sequence = new_candidate.sequence;
			old_candidate.probability = new_candidate.probability;
			
			old_candidate.state = cell_states[new_candidate.candidate_num];
			old_candidate.hidden = hidden_states[new_candidate.candidate_num];
		}
	}

	// Output sequence with highest probability
	return top_candidates[0].sequence;
}

void LSTM::saveStateTo(const std::string &filename)
{
	m_state_file = filename;
}

void LSTM::saveState(void)
{
	std::ofstream outfile(m_state_file);

	if (outfile.is_open()) {

		// Write layer dimensions
		outfile << "Size" << '\t' << m_input_size << '\t' << m_hidden_size << '\t' << m_output_size << "\r\n";

		// Write vocab list
		outfile << "Vocab" << '\t';
		for (std::pair<char, int> element : m_vocabs) {
			outfile << (int)element.first << ':' << element.second << '\t';
		}
		outfile << "\r\n";

		// Write weights
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
		writeData(m_by, "by", outfile);
		outfile.close();

	} else {
		throw std::runtime_error("Unable to open " + m_state_file);
	}
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
}

void LSTM::loadState(const std::string &filename)
{
	std::ifstream infile(filename);

	if (infile.is_open()) {
		std::string line;
		std::string id;
		std::istringstream data;

		// Get matrix and vector sizes;
		getline(infile, line);
		data.str(line);
		data >> id;

		// Initiate saved matrix sizes
		if (id == "Size") {

			size_t input_size;
			size_t hidden_size;
			size_t output_size;

			data >> input_size;
			data >> hidden_size;
			data >> output_size;

			// If a sample was loaded, layer sizes are already configured, so make sure they match before loading in the saved settings
			if (m_sample_loaded && (m_input_size != input_size || m_hidden_size != hidden_size || m_output_size != output_size)) {
				std::cerr << "Layer dimensions set from the sample doesn't match the dimensions on the file. Aborting load." << std::endl;
				return;
			}

			m_input_size = input_size;
			m_hidden_size = hidden_size;
			m_output_size = output_size;

			initiateMatrices();

		} else {
			throw std::runtime_error("Invalid input file format.");
		}

		// Get line containing the vocabs list
		getline(infile, line);
		data.str(line);
		data.clear();
		data >> id;

		// Initiate saved vocab list
		if (id == "Vocab") {

				std::string vocab;

				while (getline(data, vocab, '\t')) {

					int pos = vocab.find(":");
					if (pos == std::string::npos) continue;

					char c = (char)stoi(vocab.substr(0, pos));
					int index = stoi(vocab.substr(pos + 1));

					m_vocabs[c] = index;
					m_vocabs_indices[index] = c;
				}

		} else {
			throw std::runtime_error("Invalid input file format.");
		}

		// Get all the saved weights and biases
		while (getline(infile, line)) {
			data.str(line);
			data.clear();
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
			else
				throw std::runtime_error("Invalid input file format.");
		}

		infile.close();

	} else {
		if (!m_sample_loaded)
			throw std::runtime_error("Could not open file " + filename + ". Please load in a sample to initialize parameters.");
		else
			std::cerr << "Could not open file " << filename << ". Parameters initialized randomly." << std::endl;
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
}

template<typename T>
void LSTM::clipGradients(T &parameter)
{
	const double threshold = 7.0;
	double param_norm = parameter.matrix().norm();

	if (param_norm > threshold) {
		for (int i = 0; i < parameter.rows(); i++) {
			for (int j = 0; j < parameter.cols(); j++) {
				parameter(i, j) *= (threshold / param_norm);
			}
		}
	}
}

Eigen::ArrayXd LSTM::softmax(const Eigen::ArrayXd &input)
{
	Eigen::ArrayXd probabilities = (input / m_temperature).exp();
	double sum = probabilities.sum();

	return probabilities / sum;
}

double LSTM::crossEntropy(const Eigen::ArrayXd &output, const char &label)
{
	return -std::log(output(m_vocabs[label]));
}

double LSTM::sigmoid(double num)
{
	return 1 / (1 + std::exp(-num));
}

Eigen::ArrayXd LSTM::charToVector(const char &c)
{
	Eigen::ArrayXd one_hot_vector = Eigen::ArrayXd::Zero(m_input_size);
	one_hot_vector(m_vocabs[c]) = 1;

	return one_hot_vector;
}

char LSTM::vectorToChar(const Eigen::ArrayXd &v)
{
	int max_index = 0;
	double max_num = v(0);

	for (int i = 1; i < v.size(); i++) {
		if (v(i) > max_num) {
			max_num = v(i);
			max_index = i;
		}
	}

	return m_vocabs_indices[max_index];
}

void LSTM::fillVocabList(std::unordered_map<char, int> &vocabs, std::unordered_map<int, char> &indices, std::ifstream &infile)
{
	int counter = 0;
	char c = ' ';

	// Fill vocab list and append an index number to it
	while (infile.get(c)) {
		if (vocabs.find(c) == vocabs.end()) {
			vocabs[c] = counter;
			indices[counter++] = c;
		}
	}

	// Initiate the input and output vector sizes
	m_input_size = vocabs.size();
	m_output_size = vocabs.size();

	// Reset file stream pointer location
	infile.clear();
	infile.seekg(0);
}

void LSTM::initiateMatrices(void)
{
	// Initialize input weight matrices
	m_Wa = Eigen::MatrixXd::Random(m_hidden_size, m_input_size);
	m_Wi = Eigen::MatrixXd::Random(m_hidden_size, m_input_size);
	m_Wf = Eigen::MatrixXd::Random(m_hidden_size, m_input_size);
	m_Wo = Eigen::MatrixXd::Random(m_hidden_size, m_input_size);

	// Initialize recurrent weight matrices
	m_Ra = Eigen::MatrixXd::Random(m_hidden_size, m_hidden_size);
	m_Ri = Eigen::MatrixXd::Random(m_hidden_size, m_hidden_size);
	m_Rf = Eigen::MatrixXd::Random(m_hidden_size, m_hidden_size);
	m_Ro = Eigen::MatrixXd::Random(m_hidden_size, m_hidden_size);

	// Initialize bias vectors
	m_ba = Eigen::ArrayXd::Random(m_hidden_size);
	m_bi = Eigen::ArrayXd::Random(m_hidden_size);
	m_bf = Eigen::ArrayXd::Random(m_hidden_size);
	m_bo = Eigen::ArrayXd::Random(m_hidden_size);

	// Initialize output vector
	m_state = Eigen::ArrayXd::Zero(m_hidden_size);
	m_h_t = Eigen::ArrayXd::Zero(m_hidden_size);

	// Initialize fully connected layer weights and biases
	m_Wy = Eigen::MatrixXd::Random(m_output_size, m_hidden_size);
	m_by = Eigen::ArrayXd::Random(m_output_size);

	// Initialize fully connected layer output and softmax probabilities vector
	m_y_t = Eigen::ArrayXd::Zero(m_output_size);
	m_output = Eigen::ArrayXd::Zero(m_output_size);

	// Initalize input weight 1st momentums
	m_m_Wa = Eigen::MatrixXd::Zero(m_hidden_size, m_input_size);
	m_m_Wi = Eigen::MatrixXd::Zero(m_hidden_size, m_input_size);
	m_m_Wf = Eigen::MatrixXd::Zero(m_hidden_size, m_input_size);
	m_m_Wo = Eigen::MatrixXd::Zero(m_hidden_size, m_input_size);

	// Initialize recurrent weight 1st momentums
	m_m_Ra = Eigen::MatrixXd::Zero(m_hidden_size, m_hidden_size);
	m_m_Ri = Eigen::MatrixXd::Zero(m_hidden_size, m_hidden_size);
	m_m_Rf = Eigen::MatrixXd::Zero(m_hidden_size, m_hidden_size);
	m_m_Ro = Eigen::MatrixXd::Zero(m_hidden_size, m_hidden_size);

	// Initialize bias 1st momentums
	m_m_ba = Eigen::ArrayXd::Zero(m_hidden_size);
	m_m_bi = Eigen::ArrayXd::Zero(m_hidden_size);
	m_m_bf = Eigen::ArrayXd::Zero(m_hidden_size);
	m_m_bo = Eigen::ArrayXd::Zero(m_hidden_size);

	// Initialize input weight 2nd momentums
	m_v_Wa = Eigen::MatrixXd::Zero(m_hidden_size, m_input_size);
	m_v_Wi = Eigen::MatrixXd::Zero(m_hidden_size, m_input_size);
	m_v_Wf = Eigen::MatrixXd::Zero(m_hidden_size, m_input_size);
	m_v_Wo = Eigen::MatrixXd::Zero(m_hidden_size, m_input_size);

	// Initialize recurrent weight 2nd momentums
	m_v_Ra = Eigen::MatrixXd::Zero(m_hidden_size, m_hidden_size);
	m_v_Ri = Eigen::MatrixXd::Zero(m_hidden_size, m_hidden_size);
	m_v_Rf = Eigen::MatrixXd::Zero(m_hidden_size, m_hidden_size);
	m_v_Ro = Eigen::MatrixXd::Zero(m_hidden_size, m_hidden_size);

	// Initialize bias 2nd momentums
	m_v_ba = Eigen::ArrayXd::Zero(m_hidden_size);
	m_v_bi = Eigen::ArrayXd::Zero(m_hidden_size);
	m_v_bf = Eigen::ArrayXd::Zero(m_hidden_size);
	m_v_bo = Eigen::ArrayXd::Zero(m_hidden_size);

	// Initialize fully connected layer weight and bias 1st momentums
	m_m_Wy = Eigen::MatrixXd(m_output_size, m_hidden_size);
	m_m_by = Eigen::ArrayXd(m_output_size);

	// Initialize fully connected layer weight and bias 2nd momentums
	m_v_Wy = Eigen::MatrixXd(m_output_size, m_hidden_size);
	m_v_by = Eigen::ArrayXd(m_output_size);
}
