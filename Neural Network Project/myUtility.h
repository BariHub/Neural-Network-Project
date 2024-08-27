#pragma once

#define SIGN(a) ((a) >= 0.0f ? 1.0f : -1.0f)

/*****************************************************************
						OTHER FUNCTIONS
*****************************************************************/

class utilities
{
public:
	// Calculate rand values within a specific range - DONE
	// this has an error in int, assume it strictly does not include max, or int will give 0 always
	template<typename T>
	static double myRand(T min, T max)
	{
		if (max > min) return ((T)rand() / (T)RAND_MAX) * (max - min) + min;
		else return ((T)rand() / (T)RAND_MAX) * (min - max) + max;
	}
};
