#define NDEBUG
#define EIGEN_NO_DEBUG
#include <iostream>
#include "lstm.h"

int main(int argc, char const* argv[])
{
	std::srand(std::time(0));
	LSTM network(128, 256, 128, 0.001);
	network.load("./samples/japan.txt");
	network.loadState("./data/weights.txt");
	// network.saveTo("./data/weights.txt");
	// network.train(300, 100, 100);
	network.output(1000);
	return 0;
}
