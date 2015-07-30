#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class Hoge;

class Literal {
public:
	int depth;

public:
	CUDA_CALLABLE_MEMBER Literal() {}

	CUDA_CALLABLE_MEMBER Literal(int depth);

	CUDA_CALLABLE_MEMBER Hoge getHoge();
};

class Hoge {
public:
	Literal lit;

public:
	CUDA_CALLABLE_MEMBER Hoge() {}
};

CUDA_CALLABLE_MEMBER
Literal::Literal(int depth) {
	this->depth = depth;
}

CUDA_CALLABLE_MEMBER
Hoge Literal::getHoge() {
	return Hoge();
}

