#include "FileIO.h"

FileIO::FileIO(const char* filename)
{
	file.open(filename);
	if (!file) {

	}
}

FileIO::~FileIO()
{
}

void FileIO::update(const char* file)
{

}
