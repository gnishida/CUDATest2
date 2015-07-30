#pragma once

class iterator {
private:
	int _size;
	int capacity;
	T* data;

public:
	__device__
	vector() {