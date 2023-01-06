#pragma once

#include<Windows.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <random>
#include <utility>


#ifndef CUDACC
#define CUDACC
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

//////// helper_cuda.h check error macro ///////

#ifdef __DRIVER_TYPES_H__
static const char* _cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}
#endif

template <typename T>
void check(T result, char const* const func, const char* const file,
    int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

//////////////////////////////////


struct float_pair
{
	float first, second;
};

__device__ __host__ inline double random_float()
{
	return rand() / (RAND_MAX + 1.0f);
}

__device__ __host__ inline float random_float(float min, float max) {

    return min + (max - min) * random_float();
}

__device__ __host__ inline float clamp(float x, float minimum, float maximum)
{
	return min(maximum, max(x, minimum));
}


__device__ __host__ float fastPow(float num, int exp)
{
	float result = 1.0f;
	while (exp > 0)
	{
		if (exp % 2 == 1)
			result *= num;
		exp >>= 1;
		num *= num;
	}

	return result;
}

__device__ float_pair make_pair(float l, float r)
{
	float_pair p;
	p.first = l;
	p.second = r;
}


inline float degrees_to_radians(double degrees) {
	return degrees * 3.1415926535f / 180.0f;
}

__host__ __device__ float3 operator*(float a, float3 v)
{
	return make_float3(a * v.x, a * v.y, a * v.z);
}

__host__ __device__ float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator/(float3 v, float a)
{
	return make_float3(v.x / a, v.y / a, v.z / a);
}

__host__ __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 cross(float3 u, float3 v)
{
	return make_float3(u.y * v.z - u.z * v.y,
		u.z * v.x - u.x * v.z,
		u.x * v.y - u.y * v.x);
}

__host__ __device__ float length(float3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ float length_squared(float3 v) // same as dot(v,v) xd
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ float3 normalize(float3 v) // versor
{
	return v/sqrtf(dot(v, v));
}


