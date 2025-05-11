#pragma once

// #include <vector_types.h>
#include <cuda_runtime.h>

typedef unsigned int uint;
typedef unsigned short ushort;

#include <vector>
#ifndef __CUDACC__
#include <math.h>
#include <omp.h>

#define __global

// ////////////////////////////////////////////////////////////////////////////////
// // host implementations of CUDA functions
// ////////////////////////////////////////////////////////////////////////////////

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}

inline double rsqrt(double x)
{
    return 1.0 / sqrt(x);
}
#endif

//#if defined(__CUDACC__)
//  #include <vector_types.h>
//  #include <cuda_runtime.h>
//#elif defined(__HIPCC__)
//  #include <hip/hip_runtime.h>
//#else
//  #include <math.h>
//  #define __global
//  #define __host__
//  #define __device__
//  inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
//  inline double rsqrt(double x) { return 1.0 / sqrt(x); }
//#endif
//
//#include <vector>
//#include <omp.h>
//
//// namespace LiteMath 
//// {
//typedef unsigned int uint;
//typedef unsigned short ushort;

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
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(double2 &a, double2 b)
{
    a.x += b.x;
    a.y += b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(double3 &a, double3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(double4 &a, double4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator-=(double2 &a, double2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator-=(double3 &a, double3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator-=(double4 &a, double4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator*=(double2 &a, double2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator*=(double3 &a, double3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator*=(double4 &a, double4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
#endif
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

template<typename T> __host__ __device__ inline T SQR(T x) { return x * x; }

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
#ifndef __HIPCC__
inline __host__ __device__ void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator/=(double2 &a, double2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator/=(double3 &a, double3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
#endif
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
#ifndef __HIPCC__
inline __host__ __device__ void operator/=(double4 &a, double4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
#endif
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

inline __host__ __device__ double3 operator*(const double3x3& m, const double3& v)
{
  double3 res;
  res.x = v.x * m.m_col[0].x + v.y * m.m_col[1].x + v.z * m.m_col[2].x;
  res.y = v.x * m.m_col[0].y + v.y * m.m_col[1].y + v.z * m.m_col[2].y;
  res.z = v.x * m.m_col[0].z + v.y * m.m_col[1].z + v.z * m.m_col[2].z;
  return res;
}

inline __host__ __device__ double3 mul(const double3x3& m, const double3& v)
{
  double3 res;                             
  res.x = v.x * m.m_col[0].x + v.y * m.m_col[1].x + v.z * m.m_col[2].x;
  res.y = v.x * m.m_col[0].y + v.y * m.m_col[1].y + v.z * m.m_col[2].y;
  res.z = v.x * m.m_col[0].z + v.y * m.m_col[1].z + v.z * m.m_col[2].z;
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

/// Outer product of two 3-dimensional vectors |a><b|
static inline __host__ __device__ double3x3 outerProduct(const double3& a, const double3& b)
{
  double3x3 m;
  
  m.m_col[0].x = a.x * b.x;
  m.m_col[1].x = a.x * b.y;
  m.m_col[2].x = a.x * b.z;

  m.m_col[0].y = a.y * b.x;
  m.m_col[1].y = a.y * b.y;
  m.m_col[2].y = a.y * b.z;

  m.m_col[0].z = a.z * b.x;
  m.m_col[1].z = a.z * b.y;
  m.m_col[2].z = a.z * b.z;
      
  return m;
}

struct float3x3
{
  __host__ __device__ float3x3()  { identity(); }
  
  __host__ __device__ float3x3(const float rhs)
  {
    m_col[0] = make_float3(rhs, rhs, rhs);
    m_col[1] = make_float3(rhs, rhs, rhs);
    m_col[2] = make_float3(rhs, rhs, rhs); 
  } 

  __host__ __device__ float3x3(const float3x3& rhs) 
  { 
    m_col[0] = rhs.m_col[0];
    m_col[1] = rhs.m_col[1];
    m_col[2] = rhs.m_col[2]; 
  }

  __host__ __device__ float3x3& operator=(const float3x3& rhs)
  {
    m_col[0] = rhs.m_col[0];
    m_col[1] = rhs.m_col[1];
    m_col[2] = rhs.m_col[2]; 
    return *this;
  }

  // col-major matrix from row-major array
  __host__ __device__ float3x3(const float A[9])
  {
    m_col[0] = float3{ A[0], A[3], A[6] };
    m_col[1] = float3{ A[1], A[4], A[7] };
    m_col[2] = float3{ A[2], A[5], A[8] };
  }

  __host__ __device__ float3x3(float A0, float A1, float A2, float A3, float A4, 
                               float A5, float A6, float A7, float A8)
  {
    m_col[0] = float3{ A0, A3, A6 };
    m_col[1] = float3{ A1, A4, A7 };
    m_col[2] = float3{ A2, A5, A8 };
  }

  __host__ __device__ void identity()
  {
    m_col[0] = float3{ 1.0, 0.0, 0.0 };
    m_col[1] = float3{ 0.0, 1.0, 0.0 };
    m_col[2] = float3{ 0.0, 0.0, 1.0 };
  }

  __host__ __device__ void zero()
  {
    m_col[0] = float3{ 0.0, 0.0, 0.0 };
    m_col[1] = float3{ 0.0, 0.0, 0.0 };
    m_col[2] = float3{ 0.0, 0.0, 0.0 };
  }

  __host__ __device__ float3 get_col(int i) const                { return m_col[i]; }
  __host__ __device__ void   set_col(int i, const float3& a_col) { m_col[i] = a_col; }


  __host__ __device__ float3& col(int i)       { return m_col[i]; }
  __host__ __device__ float3  col(int i) const { return m_col[i]; }

  float3 m_col[3];
};

typedef struct float3x3 float3x3;

inline __host__ __device__ float3x3 make_float3x3_from_cols(float3 a, float3 b, float3 c)
{
  float3x3 m;
  m.set_col(0, a);
  m.set_col(1, b);
  m.set_col(2, c);
  return m;
}

inline __host__ __device__ float3 operator*(const float3x3& m, const float3& v)
{
  float3 res;
  res.x = v.x * m.m_col[0].x + v.y * m.m_col[1].x + v.z * m.m_col[2].x;
  res.y = v.x * m.m_col[0].y + v.y * m.m_col[1].y + v.z * m.m_col[2].y;
  res.z = v.x * m.m_col[0].z + v.y * m.m_col[1].z + v.z * m.m_col[2].z;
  return res;
}

inline __host__ __device__ float3 mul(const float3x3& m, const float3& v)
{
  float3 res;                             
  res.x = v.x * m.m_col[0].x + v.y * m.m_col[1].x + v.z * m.m_col[2].x;
  res.y = v.x * m.m_col[0].y + v.y * m.m_col[1].y + v.z * m.m_col[2].y;
  res.z = v.x * m.m_col[0].z + v.y * m.m_col[1].z + v.z * m.m_col[2].z;
  return res;
}

inline __host__ __device__ float3x3 transpose(const float3x3& rhs)
{
  float3x3 res;

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

inline __host__ __device__ float determinant(const float3x3& mat)
{
  const float a = mat.m_col[0].x;
  const float b = mat.m_col[1].x;
  const float c = mat.m_col[2].x;
  const float d = mat.m_col[0].y;
  const float e = mat.m_col[1].y;
  const float f = mat.m_col[2].y;
  const float g = mat.m_col[0].z;
  const float h = mat.m_col[1].z;
  const float i = mat.m_col[2].z;
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

inline __host__ __device__ float3x3 inverse3x3(const float3x3& mat)
{
  float det = determinant(mat);
  float inv_det = 1.0f / det;

  float a = mat.m_col[0].x;
  float b = mat.m_col[1].x;
  float c = mat.m_col[2].x;
  float d = mat.m_col[0].y;
  float e = mat.m_col[1].y;
  float f = mat.m_col[2].y;
  float g = mat.m_col[0].z;
  float h = mat.m_col[1].z;
  float i = mat.m_col[2].z;

  float3x3 inv;
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

inline __host__ __device__ float3x3 scale3x3(float3 t)
{
  float3x3 res;
  res.set_col(0, float3{t.x, 0.0, 0.0});
  res.set_col(1, float3{0.0, t.y, 0.0});
  res.set_col(2, float3{0.0, 0.0,  t.z});
  return res;
}

inline __host__ __device__ float3x3 rotate3x3X(float phi)
{
  float3x3 res;
  res.set_col(0, float3{1.0,      0.0,       0.0});
  res.set_col(1, float3{0.0, +cos(phi),  +sin(phi)});
  res.set_col(2, float3{0.0, -sin(phi),  +cos(phi)});
  return res;
}

inline __host__ __device__ float3x3 rotate3x3Y(float phi)
{
    float3x3 res;
  res.set_col(0, float3{+cos(phi), 0.0, -sin(phi)});
  res.set_col(1, float3{     0.0, 1.0,      0.0});
  res.set_col(2, float3{+sin(phi), 0.0, +cos(phi)});
  return res;
}

inline __host__ __device__ float3x3 rotate3x3Z(float phi)
{
  float3x3 res;
  res.set_col(0, float3{+cos(phi), sin(phi), 0.0});
  res.set_col(1, float3{-sin(phi), cos(phi), 0.0});
  res.set_col(2, float3{     0.0,     0.0, 1.0});
  return res;
}

inline __host__ __device__ float3x3 mul(float3x3 m1, float3x3 m2)
{
  const float3 column1 = mul(m1, m2.col(0));
  const float3 column2 = mul(m1, m2.col(1));
  const float3 column3 = mul(m1, m2.col(2));

  float3x3 res;
  res.set_col(0, column1);
  res.set_col(1, column2);
  res.set_col(2, column3);

  return res;
}

inline __host__ __device__ float3x3 operator*(float3x3 m1, float3x3 m2)
{
  const float3 column1 = mul(m1, m2.col(0));
  const float3 column2 = mul(m1, m2.col(1));
  const float3 column3 = mul(m1, m2.col(2));

  float3x3 res;
  res.set_col(0, column1);
  res.set_col(1, column2);
  res.set_col(2, column3);
  return res;
}

inline __host__ __device__ float3x3 operator*(float scale, float3x3 m)
{
  float3x3 res;
  res.m_col[0] = m.m_col[0] * scale;
  res.m_col[1] = m.m_col[1] * scale;
  res.m_col[2] = m.m_col[2] * scale;
  return res;
}

inline __host__ __device__ float3x3 operator*(float3x3 m, float scale)
{
  float3x3 res;
  res.m_col[0] = m.m_col[0] * scale;
  res.m_col[1] = m.m_col[1] * scale;
  res.m_col[2] = m.m_col[2] * scale;
  return res;
}

inline __host__ __device__ float3x3 operator+(float3x3 m1, float3x3 m2)
{
  float3x3 res;
  res.m_col[0] = m1.m_col[0] + m2.m_col[0];
  res.m_col[1] = m1.m_col[1] + m2.m_col[1];
  res.m_col[2] = m1.m_col[2] + m2.m_col[2];
  return res;
}

inline __host__ __device__ float3x3 operator-(float3x3 m1, float3x3 m2)
{
  float3x3 res;
  res.m_col[0] = m1.m_col[0] - m2.m_col[0];
  res.m_col[1] = m1.m_col[1] - m2.m_col[1];
  res.m_col[2] = m1.m_col[2] - m2.m_col[2];
  return res;
}

/// Outer product of two 3-dimensional vectors |a><b|
static inline __host__ __device__ float3x3 outerProduct(const float3& a, const float3& b)
{
  float3x3 m;
  
  m.m_col[0].x = a.x * b.x;
  m.m_col[1].x = a.x * b.y;
  m.m_col[2].x = a.x * b.z;

  m.m_col[0].y = a.y * b.x;
  m.m_col[1].y = a.y * b.y;
  m.m_col[2].y = a.y * b.z;

  m.m_col[0].z = a.z * b.x;
  m.m_col[1].z = a.z * b.y;
  m.m_col[2].z = a.z * b.z;
      
  return m;
}

#if defined(__CUDA_ARCH__) 
  inline __device__ void InterlockedAdd(float& mem, float data)                  {         atomicAdd(&mem, data); }
  inline __device__ void InterlockedAdd(float& mem, float data, float& a_res)    { a_res = atomicAdd(&mem, data); }

  //#if __CUDA_ARCH__ >= 600
  //inline __device__ void InterlockedAdd(double& mem, double data)                {         atomicAdd(&mem, data); }
  //inline __device__ void InterlockedAdd(double& mem, double data, double& a_res) { a_res = atomicAdd(&mem, data); }
  //#endif

  inline __device__ void InterlockedAdd(int& mem, int data)                {         atomicAdd(&mem, data); }
  inline __device__ void InterlockedAdd(int& mem, int data, int& a_res)    { a_res = atomicAdd(&mem, data); }
  inline __device__ void InterlockedAdd(uint& mem, uint data)              {         atomicAdd(&mem, data); }
  inline __device__ void InterlockedAdd(uint& mem, uint data, uint& a_res) { a_res = atomicAdd(&mem, data); }
#elif defined(__HIPCC__)
  inline __device__ void InterlockedAdd(float& mem, float data)                  {         atomicAdd(&mem, data); }
  inline __device__ void InterlockedAdd(float& mem, float data, float& a_res)    { a_res = atomicAdd(&mem, data); }

  //#if __CUDA_ARCH__ >= 600
  //inline __device__ void InterlockedAdd(double& mem, double data)                {         atomicAdd(&mem, data); }
  //inline __device__ void InterlockedAdd(double& mem, double data, double& a_res) { a_res = atomicAdd(&mem, data); }
  //#endif

  inline __device__ void InterlockedAdd(int& mem, int data)                {         atomicAdd(&mem, data); }
  inline __device__ void InterlockedAdd(int& mem, int data, int& a_res)    { a_res = atomicAdd(&mem, data); }
  inline __device__ void InterlockedAdd(uint& mem, uint data)              {         atomicAdd(&mem, data); }
  inline __device__ void InterlockedAdd(uint& mem, uint data, uint& a_res) { a_res = atomicAdd(&mem, data); }

  template<typename IndexType>
  static IndexType align(IndexType a_size, IndexType a_alignment)
  {
    if (a_size % a_alignment == 0)
      return a_size;
    else
    {
      IndexType sizeCut = a_size - (a_size % a_alignment);
      return sizeCut + a_alignment;
    }
  }

  template<typename T>
  inline size_t ReduceAddInit(std::vector<T>& a_vec, size_t a_targetSize)
  {
    const size_t cacheLineSize = 128; 
    const size_t alignSize     = cacheLineSize/sizeof(double);
    const size_t vecSizeAlign  = align(a_targetSize, alignSize);
    const size_t maxThreads    = omp_get_max_threads(); 
    a_vec.reserve(vecSizeAlign*maxThreads);
    a_vec.resize(a_targetSize);
    for(size_t i=0;i<a_vec.capacity();i++)
      a_vec.data()[i] = 0;  
    return vecSizeAlign; // can use later with 'ReduceAdd' 
  }

  template<typename T>
  inline void ReduceAddComplete(std::vector<T>& a_vec)
  {
    const size_t maxThreads = omp_get_max_threads();
    const size_t szAligned  = a_vec.capacity() / maxThreads;
    for(size_t threadId = 1; threadId < maxThreads; threadId++) 
    {
      T* threadData = a_vec.data() + threadId*szAligned;
      for(size_t i=0; i<a_vec.size(); i++)
        a_vec[i] += threadData[i];
    }
  }

  template<typename T, typename IndexType> 
  inline void ReduceAdd(std::vector<T>& a_vec, IndexType offset, T val)
  {
    if(!std::isfinite(val))
      return;
    const size_t maxThreads = omp_get_num_threads(); // here not max_threads
    const size_t szAligned  = a_vec.capacity() / maxThreads; // todo replace div by shift if number of threads is predefined
    const size_t threadId   = omp_get_thread_num();
    const size_t threadOffs = szAligned*threadId;
    a_vec.data()[threadOffs + offset] += val;
  }

  template<typename T, typename IndexType> 
  inline void ReduceAdd(std::vector<T>& a_vec, IndexType offset, IndexType a_sizeAligned, T val) // more optimized version
  {
    if(!std::isfinite(val))
      return;
    a_vec.data()[a_sizeAligned*omp_get_thread_num() + offset] += val;
  }

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

  template<typename IndexType>
  static IndexType align(IndexType a_size, IndexType a_alignment)
  {
    if (a_size % a_alignment == 0)
      return a_size;
    else
    {
      IndexType sizeCut = a_size - (a_size % a_alignment);
      return sizeCut + a_alignment;
    }
  }

  //template<typename T>
  //inline size_t ReduceAddInit(std::vector<T>& a_vec, size_t a_targetSize)
  //{
  //  const size_t cacheLineSize = 128; 
  //  const size_t alignSize     = cacheLineSize/sizeof(double);
  //  const size_t vecSizeAlign  = align(a_targetSize, alignSize);
  //  const size_t maxThreads    = omp_get_max_threads(); 
  //  a_vec.reserve(vecSizeAlign*maxThreads);
  //  a_vec.resize(a_targetSize);
  //  for(size_t i=0;i<a_vec.capacity();i++)
  //    a_vec.data()[i] = 0;  
  //  return vecSizeAlign; // can use later with 'ReduceAdd' 
  //}
  //
  //template<typename T>
  //inline void ReduceAddComplete(std::vector<T>& a_vec)
  //{
  //  const size_t maxThreads = omp_get_max_threads();
  //  const size_t szAligned  = a_vec.capacity() / maxThreads;
  //  for(size_t threadId = 1; threadId < maxThreads; threadId++) 
  //  {
  //    T* threadData = a_vec.data() + threadId*szAligned;
  //    for(size_t i=0; i<a_vec.size(); i++)
  //      a_vec[i] += threadData[i];
  //  }
  //}

  //template<typename T, typename IndexType> 
  //inline void ReduceAdd(std::vector<T>& a_vec, IndexType offset, T val)
  //{
  //  if(!std::isfinite(val))
  //    return;
  //  const size_t maxThreads = omp_get_num_threads(); // here not max_threads
  //  const size_t szAligned  = a_vec.capacity() / maxThreads; // todo replace div by shift if number of threads is predefined
  //  const size_t threadId   = omp_get_thread_num();
  //  const size_t threadOffs = szAligned*threadId;
  //  a_vec.data()[threadOffs + offset] += val;
  //}
  //
  //template<typename T, typename IndexType> 
  //inline void ReduceAdd(std::vector<T>& a_vec, IndexType offset, IndexType a_sizeAligned, T val) // more optimized version
  //{
  //  if(!std::isfinite(val))
  //    return;
  //  a_vec.data()[a_sizeAligned*omp_get_thread_num() + offset] += val;
  //}

#endif

// }