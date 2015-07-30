#pragma once

#include "vector.cuh"

class string {
private:
	vector<char> str;

public:
	__host__ __device__
	string() {}

	__host__ __device__
	string(const char* str) {
		for (int i = 0; ; ++i) {
			if (str[i] == 0) break;
			this->str.push_back(str[i]);
		}
	}

	__host__ __device__
	string& operator=(const char* str) {
		for (int i = 0; ; ++i) {
			if (str[i] == 0) break;
			this->str.push_back(str[i]);
		}
		return *this;
	}

	__host__ __device__
	int length() const {
		return str.size();
	}

	__host__ __device__
	char operator[](int index) const {
		return str[index];
	}

	__host__ __device__
	char& operator[](int index) {
		return str[index];
	}

	__host__ __device__
	bool operator==(const string& str) const {
		if (this->str.size() != str.length()) return false;

		for (int i = 0; i < this->str.size(); ++i) {
			if (this->str[i] != str[i]) return false;
		}
		return true;
	}

	__host__ __device__
	void operator+=(const string& str) {
		for (int i = 0; i < str.length(); ++i) {
			this->str.push_back(str[i]);
		}
	}
};

