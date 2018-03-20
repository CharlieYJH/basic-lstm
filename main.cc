#include <iostream>
#include "lstm.h"

int main(int argc, char const* argv[])
{
	std::srand(std::time(0));
	LSTM network(128, 128, 128, 0.01);
	network.load("./samples/shakespear.txt");
	network.loadState("./data/weights.txt");
	// network.saveTo("./data/weights.txt");
	// network.train(5, 300);
	network.output(300);
	std::cout << std::endl;
	return 0;
}
