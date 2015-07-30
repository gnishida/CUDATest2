#pragma once

template<class T>
class vector_iterator {
public:
	int index;
	T* ptr;

	__host__ __device__
	vector_iterator(int index, T* ptr) {
		this->index = index;
		this->ptr = ptr;
	}

	__host__ __device__
	void operator++() {
		ptr += sizeof(T);
	}

	__host__ __device__
	vector_iterator operator+(int val) {
		return vector_iterator(this->index + val, this->ptr + sizeof(T) * val);
	}
};

template<class T>
class vector {
private:
	int _size;
	int capacity;
	T* data;

public:
	static vector_iterator<T> iterator;

public:
	__host__ __device__
	vector() {
		_size = 0;
		capacity = 100;
		data = (T*)malloc(sizeof(T) * capacity);
	}

	__host__ __device__ 
	int size() const {
		return _size;
	}

	__host__ __device__ 
	T operator[](int index) const {
		return data[index];
	}

	__host__ __device__ 
	T& operator[](int index) {
		return data[index];
	}

	__host__ __device__
	void clear() {
		_size = 0;
	}

	__host__ __device__ 
	void push_back(const T& value) {
		data[_size++] = value;
	}

	__host__ __device__
	vector_iterator<T> begin() const {
		return vector_iterator<T>(0, &data[0]);
	}

	__host__ __device__
	vector_iterator<T> end() const {
		return vector_iterator<T>(_size, &data[_size]);
	}

	__host__ __device__
	void erase(const vector_iterator<T>& it) {
		T* tmp = (T*)malloc(sizeof(T) * _size);
		for (int i = 0; i < it.index; ++i) {
			tmp[i] = data[i];
		}
		for (int i = it.index + 1; i < _size; ++i) {
			tmp[i - 1] = data[i];
		}

		free(data);
		data = tmp;
		_size--;
	}

	__host__ __device__
	void insert(const vector_iterator<T>& it, const vector_iterator<T>& begin, const vector_iterator<T>& end) {
		T* tmp = (T*)malloc(sizeof(T) * (_size + end.index - begin.index));
		for (int i = 0; i < it.index; ++i) {
			tmp[i] = data[i];
		}
		T* ptr = begin.ptr;
		for (int i = begin.index; i < end.index; ++i) {
			tmp[it.index + i] = *ptr;
			ptr++;
		}
		for (int i = it.index; i < _size; ++i) {
			tmp[it.index + end.index - begin.index + i] = data[i];
		}

		free(data);
		data = tmp;
		_size += end.index - begin.index;
	}
};