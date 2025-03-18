#pragma once

// #include <vector_types.h>
#include <cuda_runtime.h>

// namespace LiteMath 
// {
typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef __CUDACC__
#include <math.h>


// ////////////////////////////////////////////////////////////////////////////////
// // host implementations of CUDA functions
// ////////////////////////////////////////////////////////////////////////////////

// inline float fminf(float a, float b)
// {
//     return a < b ? a : b;
// }

// inline float fmaxf(float a, float b)
// {
//     return a > b ? a : b;
// }

// inline double fmin(double a, double b)
// {
//     return a < b ? a : b;
// }

// inline double fmax(double a, double b)
// {
//     return a > b ? a : b;
// }

// inline int max(int a, int b)
// {
//     return a > b ? a : b;
// }

// inline int min(int a, int b)
// {
//     return a < b ? a : b;
// }

// inline uint max(uint a, uint b)
// {
//     return a > b ? a : b;
// }

// inline uint min(uint a, uint b)
// {
//     return a < b ? a : b;
// }

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}

inline double rsqrt(double x)
{
    return 1.0 / sqrt(x);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}
inline __host__ __device__ double2 operator-(double2 &a)
{
    return make_double2(-a.x, -a.y);
}
inline __host__ __device__ int2 operator-(int2 &a)
{
    return make_int2(-a.x, -a.y);
}
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ double3 operator-(double3 &a)
{
    return make_double3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ int3 operator-(int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ double4 operator-(double4 &a)
{
    return make_double4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ int4 operator-(int4 &a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ double2 operator+(double2 a, double2 b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(double2 &a, double2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ double2 operator+(double2 a, double b)
{
    return make_double2(a.x + b, a.y + b);
}
inline __host__ __device__ double2 operator+(double b, double2 a)
{
    return make_double2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(double2 &a, double b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ int2 operator+(int2 a, int b)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ int2 operator+(int b, int2 a)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(int2 &a, int b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ uint2 operator+(uint2 a, uint b)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ uint2 operator+(uint b, uint2 a)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(uint2 &a, uint b)
{
    a.x += b;
    a.y += b;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(double3 &a, double3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ double3 operator+(double3 a, double b)
{
    return make_double3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(double3 &a, double b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(int3 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(uint3 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int b, int3 a)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ uint3 operator+(uint b, uint3 a)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ double3 operator+(double b, double3 a)
{
    return make_double3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ double4 operator+(double4 a, double4 b)
{
    return make_double4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(double4 &a, double4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ double4 operator+(double4 a, double b)
{
    return make_double4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ double4 operator+(double b, double4 a)
{
    return make_double4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(double4 &a, double b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(int4 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(uint4 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ double2 operator-(double2 a, double2 b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(double2 &a, double2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ double2 operator-(double2 a, double b)
{
    return make_double2(a.x - b, a.y - b);
}
inline __host__ __device__ double2 operator-(double b, double2 a)
{
    return make_double2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(double2 &a, double b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}
inline __host__ __device__ int2 operator-(int b, int2 a)
{
    return make_int2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(int2 &a, int b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ uint2 operator-(uint2 a, uint b)
{
    return make_uint2(a.x - b, a.y - b);
}
inline __host__ __device__ uint2 operator-(uint b, uint2 a)
{
    return make_uint2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(double3 &a, double3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ double3 operator-(double3 a, double b)
{
    return make_double3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ double3 operator-(double b, double3 a)
{
    return make_double3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(double3 &a, double b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ int3 operator-(int3 a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ int3 operator-(int b, int3 a)
{
    return make_int3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(int3 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ uint3 operator-(uint3 a, uint b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ uint3 operator-(uint b, uint3 a)
{
    return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ double4 operator-(double4 a, double4 b)
{
    return make_double4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(double4 &a, double4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ double4 operator-(double4 a, double b)
{
    return make_double4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(double4 &a, double b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(int4 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ double2 operator*(double2 a, double2 b)
{
    return make_double2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(double2 &a, double2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ double2 operator*(double2 a, double b)
{
    return make_double2(a.x * b, a.y * b);
}
inline __host__ __device__ double2 operator*(double b, double2 a)
{
    return make_double2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(double2 &a, double b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ int2 operator*(int2 a, int b)
{
    return make_int2(a.x * b, a.y * b);
}
inline __host__ __device__ int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(int2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ uint2 operator*(uint2 a, uint b)
{
    return make_uint2(a.x * b, a.y * b);
}
inline __host__ __device__ uint2 operator*(uint b, uint2 a)
{
    return make_uint2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(double3 &a, double3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ double3 operator*(double3 a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ double3 operator*(double b, double3 a)
{
    return make_double3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(double3 &a, double b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ int3 operator*(int3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ int3 operator*(int b, int3 a)
{
    return make_int3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(int3 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ uint3 operator*(uint3 a, uint b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ uint3 operator*(uint b, uint3 a)
{
    return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ double4 operator*(double4 a, double4 b)
{
    return make_double4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(double4 &a, double4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ double4 operator*(double4 a, double b)
{
    return make_double4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ double4 operator*(double b, double4 a)
{
    return make_double4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(double4 &a, double b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(int4 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

  template<typename T> inline T SQR(T x) { return x * x; }

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}

inline __host__ __device__ void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}

inline __host__ __device__ float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}

inline __host__ __device__ void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ float4 operator/(float b, float4 a)
{
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}


inline __host__ __device__ double2 operator/(double2 a, double2 b)
{
    return make_double2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(double2 &a, double2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ double2 operator/(double2 a, double b)
{
    return make_double2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(double2 &a, double b)
{
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ double2 operator/(double b, double2 a)
{
    return make_double2(b / a.x, b / a.y);
}

inline __host__ __device__ double3 operator/(double3 a, double3 b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(double3 &a, double3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ double3 operator/(double3 a, double b)
{
    return make_double3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(double3 &a, double b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ double3 operator/(double b, double3 a)
{
    return make_double3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ double4 operator/(double4 a, double4 b)
{
    return make_double4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
inline __host__ __device__ void operator/=(double4 &a, double4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ double4 operator/(double4 a, double b)
{
    return make_double4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline __host__ __device__ void operator/=(double4 &a, double b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ double4 operator/(double b, double4 a)
{
    return make_double4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

// inline  __host__ __device__ float2 fminf(float2 a, float2 b)
// {
//     return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
// }
// inline __host__ __device__ float3 fminf(float3 a, float3 b)
// {
//     return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
// }
// inline  __host__ __device__ float4 fminf(float4 a, float4 b)
// {
//     return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
// }

// inline  __host__ __device__ double2 fmin(double2 a, double2 b)
// {
//     return make_double2(fmin(a.x,b.x), fmin(a.y,b.y));
// }
// inline __host__ __device__ double3 fmin(double3 a, double3 b)
// {
//     return make_double3(fmin(a.x,b.x), fmin(a.y,b.y), fmin(a.z,b.z));
// }
// inline  __host__ __device__ double4 fmin(double4 a, double4 b)
// {
//     return make_double4(fmin(a.x,b.x), fmin(a.y,b.y), fmin(a.z,b.z), fmin(a.w,b.w));
// }

// inline __host__ __device__ int2 min(int2 a, int2 b)
// {
//     return make_int2(min(a.x,b.x), min(a.y,b.y));
// }
// inline __host__ __device__ int3 min(int3 a, int3 b)
// {
//     return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
// }
// inline __host__ __device__ int4 min(int4 a, int4 b)
// {
//     return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
// }

// inline __host__ __device__ uint2 min(uint2 a, uint2 b)
// {
//     return make_uint2(min(a.x,b.x), min(a.y,b.y));
// }
// inline __host__ __device__ uint3 min(uint3 a, uint3 b)
// {
//     return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
// }
// inline __host__ __device__ uint4 min(uint4 a, uint4 b)
// {
//     return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
// }

// ////////////////////////////////////////////////////////////////////////////////
// // max
// ////////////////////////////////////////////////////////////////////////////////

// inline __host__ __device__ float2 fmaxf(float2 a, float2 b)
// {
//     return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
// }
// inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
// {
//     return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
// }
// inline __host__ __device__ float4 fmaxf(float4 a, float4 b)
// {
//     return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
// }

// inline __host__ __device__ double2 fmax(double2 a, double2 b)
// {
//     return make_double2(fmax(a.x,b.x), fmax(a.y,b.y));
// }
// inline __host__ __device__ double3 fmax(double3 a, double3 b)
// {
//     return make_double3(fmax(a.x,b.x), fmax(a.y,b.y), fmax(a.z,b.z));
// }
// inline __host__ __device__ double4 fmax(double4 a, double4 b)
// {
//     return make_double4(fmax(a.x,b.x), fmax(a.y,b.y), fmax(a.z,b.z), fmax(a.w,b.w));
// }

// inline __host__ __device__ int2 max(int2 a, int2 b)
// {
//     return make_int2(max(a.x,b.x), max(a.y,b.y));
// }
// inline __host__ __device__ int3 max(int3 a, int3 b)
// {
//     return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
// }
// inline __host__ __device__ int4 max(int4 a, int4 b)
// {
//     return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
// }

// inline __host__ __device__ uint2 max(uint2 a, uint2 b)
// {
//     return make_uint2(max(a.x,b.x), max(a.y,b.y));
// }
// inline __host__ __device__ uint3 max(uint3 a, uint3 b)
// {
//     return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
// }
// inline __host__ __device__ uint4 max(uint4 a, uint4 b)
// {
//     return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
// }

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}


inline __device__ __host__ double lerp(double a, double b, double t)
{
    return a + t*(b-a);
}
inline __device__ __host__ double2 lerp(double2 a, double2 b, double t)
{
    return a + t*(b-a);
}
inline __device__ __host__ double3 lerp(double3 a, double3 b, double t)
{
    return a + t*(b-a);
}
inline __device__ __host__ double4 lerp(double4 a, double4 b, double t)
{
    return a + t*(b-a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmax(a, fmin(f, b));
}
inline __device__ __host__ double clamp(double f, double a, double b)
{
    return fmax(a, fmin(f, b));
}
inline __device__ __host__ int clamp(int f, int a, int b)
{
    const int tmp = (f < b ? f : b) ;
    return a > tmp ? a : tmp;
}
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
    const uint tmp = (f < b ? f : b);
    return a > tmp ? a : tmp;
}

inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ double2 clamp(double2 v, double a, double b)
{
    return make_double2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ double2 clamp(double2 v, double2 a, double2 b)
{
    return make_double2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ double3 clamp(double3 v, double a, double b)
{
    return make_double3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ double3 clamp(double3 v, double3 a, double3 b)
{
    return make_double3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ double4 clamp(double4 v, double a, double b)
{
    return make_double4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ double4 clamp(double4 v, double4 a, double4 b)
{
    return make_double4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ int2 clamp(int2 v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ int4 clamp(int4 v, int a, int b)
{
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b)
{
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)
{
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b)
{
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)
{
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b)
{
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ double dot(double2 a, double2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ double dot(double4 a, double4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}


inline __host__ __device__ int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint dot(uint2 a, uint2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ uint dot(uint3 a, uint3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ uint dot(uint4 a, uint4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float4 v)
{
    return sqrtf(dot(v, v));
}


inline __host__ __device__ double length(double2 v)
{
    return sqrt(dot(v, v));
}
inline __host__ __device__ double length(double3 v)
{
    return sqrt(dot(v, v));
}
inline __host__ __device__ double length(double4 v)
{
    return sqrt(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}


inline __host__ __device__ double2 normalize(double2 v)
{
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ double3 normalize(double3 v)
{
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ double4 normalize(double4 v)
{
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}

// ////////////////////////////////////////////////////////////////////////////////
// // floor
// ////////////////////////////////////////////////////////////////////////////////

// inline __host__ __device__ float2 floorf(float2 v)
// {
//     return make_float2(std::floor(v.x), std::floor(v.y));
// }
// inline __host__ __device__ float3 floorf(float3 v)
// {
//     return make_float3(std::floor(v.x), std::floor(v.y), std::floor(v.z));
// }
// inline __host__ __device__ float4 floorf(float4 v)
// {
//     return make_float4(std::floor(v.x), std::floor(v.y), std::floor(v.z), std::floor(v.w));
// }


// inline __host__ __device__ double2 floor(double2 v)
// {
//     return make_double2(floor(v.x), floor(v.y));
// }
// inline __host__ __device__ double3 floor(double3 v)
// {
//     return make_double3(floor(v.x), floor(v.y), floor(v.z));
// }
// inline __host__ __device__ double4 floor(double4 v)
// {
//     return make_double4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
// }

// ////////////////////////////////////////////////////////////////////////////////
// // frac - returns the fractional portion of a scalar or each vector component
// ////////////////////////////////////////////////////////////////////////////////

// inline __host__ __device__ float fracf(float v)
// {
//     return v - floor(v);
// }
// inline __host__ __device__ float2 fracf(float2 v)
// {
//     return make_float2(fracf(v.x), fracf(v.y));
// }
// inline __host__ __device__ float3 fracf(float3 v)
// {
//     return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
// }
// inline __host__ __device__ float4 fracf(float4 v)
// {
//     return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
// }


// inline __host__ __device__ double frac(double v)
// {
//     return v - floor(v);
// }
// inline __host__ __device__ double2 frac(double2 v)
// {
//     return make_double2(frac(v.x), frac(v.y));
// }
// inline __host__ __device__ double3 frac(double3 v)
// {
//     return make_double3(frac(v.x), frac(v.y), frac(v.z));
// }
// inline __host__ __device__ double4 frac(double4 v)
// {
//     return make_double4(frac(v.x), frac(v.y), frac(v.z), frac(v.w));
// }

// ////////////////////////////////////////////////////////////////////////////////
// // fmod
// ////////////////////////////////////////////////////////////////////////////////

// inline __host__ __device__ float2 fmodf(float2 a, float2 b)
// {
//     return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
// }
// inline __host__ __device__ float3 fmodf(float3 a, float3 b)
// {
//     return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
// }
// inline __host__ __device__ float4 fmodf(float4 a, float4 b)
// {
//     return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
// }


// inline __host__ __device__ double2 fmod(double2 a, double2 b)
// {
//     return make_double2(fmod(a.x, b.x), fmod(a.y, b.y));
// }
// inline __host__ __device__ double3 fmod(double3 a, double3 b)
// {
//     return make_double3(fmod(a.x, b.x), fmod(a.y, b.y), fmod(a.z, b.z));
// }
// inline __host__ __device__ double4 fmod(double4 a, double4 b)
// {
//     return make_double4(fmod(a.x, b.x), fmod(a.y, b.y), fmod(a.z, b.z), fmod(a.w, b.w));
// }

// ////////////////////////////////////////////////////////////////////////////////
// // absolute value
// ////////////////////////////////////////////////////////////////////////////////

// inline __host__ __device__ float2 fabs(float2 v)
// {
//     return make_float2(fabs(v.x), fabs(v.y));
// }
// inline __host__ __device__ float3 fabs(float3 v)
// {
//     return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
// }
// inline __host__ __device__ float4 fabs(float4 v)
// {
//     return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
// }

// inline __host__ __device__ double2 fab(double2 v)
// {
//     return make_double2(fabs(v.x), fabs(v.y));
// }
// inline __host__ __device__ double3 fabs(double3 v)
// {
//     return make_double3(fabs(v.x), fabs(v.y), fabs(v.z));
// }
// inline __host__ __device__ double4 fabs(double4 v)
// {
//     return make_double4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
// }

// inline __host__ __device__ int2 abs(int2 v)
// {
//     return make_int2(abs(v.x), abs(v.y));
// }
// inline __host__ __device__ int3 abs(int3 v)
// {
//     return make_int3(abs(v.x), abs(v.y), abs(v.z));
// }
// inline __host__ __device__ int4 abs(int4 v)
// {
//     return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
// }

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

inline __host__ __device__ double3 reflect(double3 i, double3 n)
{
    return i - 2.0 * n * dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __host__ __device__ double3 cross(double3 a, double3 b)
{
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}


struct double3x3
{
  __host__ __device__ double3x3()  { identity(); }
  
  __host__ __device__ double3x3(const double rhs)
  {
    m_col[0] = make_double3(rhs, rhs, rhs);
    m_col[1] = make_double3(rhs, rhs, rhs);
    m_col[2] = make_double3(rhs, rhs, rhs); 
  } 

  __host__ __device__ double3x3(const double3x3& rhs) 
  { 
    m_col[0] = rhs.m_col[0];
    m_col[1] = rhs.m_col[1];
    m_col[2] = rhs.m_col[2]; 
  }

  __host__ __device__ double3x3& operator=(const double3x3& rhs)
  {
    m_col[0] = rhs.m_col[0];
    m_col[1] = rhs.m_col[1];
    m_col[2] = rhs.m_col[2]; 
    return *this;
  }

  // col-major matrix from row-major array
  __host__ __device__ double3x3(const double A[9])
  {
    m_col[0] = double3{ A[0], A[3], A[6] };
    m_col[1] = double3{ A[1], A[4], A[7] };
    m_col[2] = double3{ A[2], A[5], A[8] };
  }

  __host__ __device__ double3x3(double A0, double A1, double A2, double A3, double A4, 
                                double A5, double A6, double A7, double A8)
  {
    m_col[0] = double3{ A0, A3, A6 };
    m_col[1] = double3{ A1, A4, A7 };
    m_col[2] = double3{ A2, A5, A8 };
  }

  __host__ __device__ void identity()
  {
    m_col[0] = double3{ 1.0, 0.0, 0.0 };
    m_col[1] = double3{ 0.0, 1.0, 0.0 };
    m_col[2] = double3{ 0.0, 0.0, 1.0 };
  }

  __host__ __device__ void zero()
    {
    m_col[0] = double3{ 0.0, 0.0, 0.0 };
    m_col[1] = double3{ 0.0, 0.0, 0.0 };
    m_col[2] = double3{ 0.0, 0.0, 0.0 };
    }

  __host__ __device__ double3 get_col(int i) const                 { return m_col[i]; }
  __host__ __device__ void    set_col(int i, const double3& a_col) { m_col[i] = a_col; }


  __host__ __device__ double3& col(int i)       { return m_col[i]; }
  __host__ __device__ double3  col(int i) const { return m_col[i]; }


  double3 m_col[3];
};

typedef struct double3x3 double3x3;

inline __host__ __device__ double3x3 make_double3x3_from_cols(double3 a, double3 b, double3 c)
{
  double3x3 m;
  m.set_col(0, a);
  m.set_col(1, b);
  m.set_col(2, c);
  return m;
}

inline __host__ __device__ void mat3_colmajor_mul_vec3_double(double* __restrict RES, const double* __restrict B, const double* __restrict V) 
{
  RES[0] = V[0] * B[0] + V[1] * B[3] + V[2] * B[6];
  RES[1] = V[0] * B[1] + V[1] * B[4] + V[2] * B[7];
  RES[2] = V[0] * B[2] + V[1] * B[5] + V[2] * B[8];
}

inline __host__ __device__ double3 operator*(const double3x3& m, const double3& v)
{
  double3 res;
  mat3_colmajor_mul_vec3_double((double*)&res, (const double*)&m, (const double*)&v);
  return res;
}

inline __host__ __device__ double3 mul(const double3x3& m, const double3& v)
{
  double3 res;                             
  mat3_colmajor_mul_vec3_double((double*)&res, (const double*)&m, (const double*)&v);
  return res;
}

inline __host__ __device__ double3x3 transpose(const double3x3& rhs)
{
  double3x3 res;

  res.m_col[0].x = rhs.m_col[0].x;
  res.m_col[0].y = rhs.m_col[1].x;
  res.m_col[0].z = rhs.m_col[2].x;

  res.m_col[1].x = rhs.m_col[0].y;
  res.m_col[1].y = rhs.m_col[1].y;
  res.m_col[1].z = rhs.m_col[2].y;

  res.m_col[2].x = rhs.m_col[0].z;
  res.m_col[2].y = rhs.m_col[1].z;
  res.m_col[2].z = rhs.m_col[2].z;
  
  return res;
}

inline __host__ __device__ double determinant(const double3x3& mat)
{
  const double a = mat.m_col[0].x;
  const double b = mat.m_col[1].x;
  const double c = mat.m_col[2].x;
  const double d = mat.m_col[0].y;
  const double e = mat.m_col[1].y;
  const double f = mat.m_col[2].y;
  const double g = mat.m_col[0].z;
  const double h = mat.m_col[1].z;
  const double i = mat.m_col[2].z;

  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}


inline __host__ __device__ double3x3 inverse3x3(const double3x3& mat)
{
  double det = determinant(mat);

  double inv_det = 1.0 / det;

  double a = mat.m_col[0].x;
  double b = mat.m_col[1].x;
  double c = mat.m_col[2].x;
  double d = mat.m_col[0].y;
  double e = mat.m_col[1].y;
  double f = mat.m_col[2].y;
  double g = mat.m_col[0].z;
  double h = mat.m_col[1].z;
  double i = mat.m_col[2].z;

  double3x3 inv;
  inv.m_col[0].x = (e * i - f * h) * inv_det;
  inv.m_col[1].x = (c * h - b * i) * inv_det;
  inv.m_col[2].x = (b * f - c * e) * inv_det;
  inv.m_col[0].y = (f * g - d * i) * inv_det;
  inv.m_col[1].y = (a * i - c * g) * inv_det;
  inv.m_col[2].y = (c * d - a * f) * inv_det;
  inv.m_col[0].z = (d * h - e * g) * inv_det;
  inv.m_col[1].z = (b * g - a * h) * inv_det;
  inv.m_col[2].z = (a * e - b * d) * inv_det;

  return inv;
} 


inline __host__ __device__ double3x3 scale3x3(double3 t)
{
  double3x3 res;
  res.set_col(0, double3{t.x, 0.0, 0.0});
  res.set_col(1, double3{0.0, t.y, 0.0});
  res.set_col(2, double3{0.0, 0.0,  t.z});
  return res;
}

inline __host__ __device__ double3x3 rotate3x3X(double phi)
{
  double3x3 res;
  res.set_col(0, double3{1.0,      0.0,       0.0});
  res.set_col(1, double3{0.0, +cos(phi),  +sin(phi)});
  res.set_col(2, double3{0.0, -sin(phi),  +cos(phi)});
  return res;
}

inline __host__ __device__ double3x3 rotate3x3Y(double phi)
{
  double3x3 res;
  res.set_col(0, double3{+cos(phi), 0.0, -sin(phi)});
  res.set_col(1, double3{     0.0, 1.0,      0.0});
  res.set_col(2, double3{+sin(phi), 0.0, +cos(phi)});
  return res;
}

inline __host__ __device__ double3x3 rotate3x3Z(double phi)
{
  double3x3 res;
  res.set_col(0, double3{+cos(phi), sin(phi), 0.0});
  res.set_col(1, double3{-sin(phi), cos(phi), 0.0});
  res.set_col(2, double3{     0.0,     0.0, 1.0});
  return res;
}

inline __host__ __device__ double3x3 mul(double3x3 m1, double3x3 m2)
{
  const double3 column1 = mul(m1, m2.col(0));
  const double3 column2 = mul(m1, m2.col(1));
  const double3 column3 = mul(m1, m2.col(2));
  double3x3 res;
  res.set_col(0, column1);
  res.set_col(1, column2);
  res.set_col(2, column3);

  return res;
}

inline __host__ __device__ double3x3 operator*(double3x3 m1, double3x3 m2)
{
  const double3 column1 = mul(m1, m2.col(0));
  const double3 column2 = mul(m1, m2.col(1));
  const double3 column3 = mul(m1, m2.col(2));

  double3x3 res;
  res.set_col(0, column1);
  res.set_col(1, column2);
  res.set_col(2, column3);
  return res;
}

inline __host__ __device__ double3x3 operator*(double scale, double3x3 m)
{
  double3x3 res;
  res.m_col[0] = m.m_col[0] * scale;
  res.m_col[1] = m.m_col[1] * scale;
  res.m_col[2] = m.m_col[2] * scale;
  return res;
}

inline __host__ __device__ double3x3 operator*(double3x3 m, double scale)
{
  double3x3 res;
  res.m_col[0] = m.m_col[0] * scale;
  res.m_col[1] = m.m_col[1] * scale;
  res.m_col[2] = m.m_col[2] * scale;
  return res;
}

inline __host__ __device__ double3x3 operator+(double3x3 m1, double3x3 m2)
{
  double3x3 res;
  res.m_col[0] = m1.m_col[0] + m2.m_col[0];
  res.m_col[1] = m1.m_col[1] + m2.m_col[1];
  res.m_col[2] = m1.m_col[2] + m2.m_col[2];
  return res;
}

inline __host__ __device__ double3x3 operator-(double3x3 m1, double3x3 m2)
{
  double3x3 res;
  res.m_col[0] = m1.m_col[0] - m2.m_col[0];
  res.m_col[1] = m1.m_col[1] - m2.m_col[1];
  res.m_col[2] = m1.m_col[2] - m2.m_col[2];
  return res;
}

#ifdef __CUDACC__
  inline __host__ __device__ void InterlockedAdd(float& mem, float data)                  {         atomicAdd(&mem, data); }
  inline __host__ __device__ void InterlockedAdd(float& mem, float data, float& a_res)    { a_res = atomicAdd(&mem, data); }

  #if __CUDA_ARCH__ >= 600
  inline __host__ __device__ void InterlockedAdd(double& mem, double data)                {         atomicAdd(&mem, data); }
  inline __host__ __device__ void InterlockedAdd(double& mem, double data, double& a_res) { a_res = atomicAdd(&mem, data); }
  #endif

  inline __host__ __device__ void InterlockedAdd(int& mem, int data)                {         atomicAdd(&mem, data); }
  inline __host__ __device__ void InterlockedAdd(int& mem, int data, int& a_res)    { a_res = atomicAdd(&mem, data); }
  inline __host__ __device__ void InterlockedAdd(uint& mem, uint data)              {         atomicAdd(&mem, data); }
  inline __host__ __device__ void InterlockedAdd(uint& mem, uint data, uint& a_res) { a_res = atomicAdd(&mem, data); }
#else
  static inline void InterlockedAdd(float& mem, float data) 
  { 
    #pragma omp atomic
    mem += data;
  }

  static inline void InterlockedAdd(float& mem, float data, float& a_res) 
  { 
    a_res = mem;
    #pragma omp atomic
    mem += data;
  }

  static inline void InterlockedAdd(double& mem, double data) 
  { 
    #pragma omp atomic
    mem += data;
  }

  static inline void InterlockedAdd(double& mem, double data, double& a_res) 
  { 
    a_res = mem;
    #pragma omp atomic
    mem += data;
  }

  static inline void InterlockedAdd(int& mem, int data) 
  { 
    #pragma omp atomic
    mem += data;
  }

  static inline void InterlockedAdd(int& mem, int data, int& a_res) 
  { 
    a_res = mem;
    #pragma omp atomic
    mem += data;
  }

  static inline void InterlockedAdd(uint& mem, uint data) 
  { 
    #pragma omp atomic
    mem += data;
  }

  static inline void InterlockedAdd(uint& mem, uint data, uint& a_res) 
  { 
    a_res = mem;
    #pragma omp atomic
    mem += data;
  }
#endif

// }