#pragma once
#include <fstream>

class FileIO
{
public:
	FileIO(const char* filename);
	~FileIO();
	void update(const char* file);
private:
	std::fstream file;
};

