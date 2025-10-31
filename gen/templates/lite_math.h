#pragma once

#ifdef __OPENCL_VERSION__
  #include "extended/LiteMathCL.h"  // if this file is included in OpenCL shaders 
#else
#ifdef ISPC
  #include "extended/LiteMathISPC.h"
#else  
#ifdef CUDA_MATH
  #include "extended/LiteMathCU.h"
#else  

#include <cstdint>
#include <cmath>
#include <limits>           // for std::numeric_limits
#include <cstring>          // for memcpy
#include <algorithm>        // for std::min/std::max 
#include <initializer_list> //
#include <vector>

#ifndef MAXFLOAT
  #include <cfloat>
  #define MAXFLOAT FLT_MAX
#endif

#ifdef M_PI
#undef M_PI // same if we have such macro some-where else ...
#endif

#ifdef min 
#undef min // if we on windows, need thid due to macro definitions in windows.h; same if we have such macro some-where else.
#endif

#ifdef max
#undef max // if we on windows, need thid due to macro definitions in windows.h; same if we have such macro some-where else.
#endif

#if defined(_MSC_VER)
#define CVEX_ALIGNED(x) __declspec(align(x))
#else
//#if defined(__GNUC__)
#define CVEX_ALIGNED(x) __attribute__ ((aligned(x)))
//#endif
#endif

#ifndef __APPLE__
#define __global
#endif

#ifdef KERNEL_SLICER
#define KSLICER_DATA_SIZE(x) __attribute__((size(#x)))
#else
#define KSLICER_DATA_SIZE(x) 
#endif

namespace LiteMath
{ 
  typedef unsigned int   uint;
  typedef unsigned short ushort;
  typedef unsigned char  uchar;

  constexpr float EPSILON      = 1e-6f;
  constexpr float INF_POSITIVE = +std::numeric_limits<float>::infinity();
  constexpr float INF_NEGATIVE = -std::numeric_limits<float>::infinity();
  
  constexpr float DEG_TO_RAD   = float(3.14159265358979323846f) / 180.0f;
  constexpr float M_PI         = float(3.14159265358979323846f);
  constexpr float M_TWOPI      = M_PI*2.0f;
  constexpr float INV_PI       = 1.0f/M_PI;
  constexpr float INV_TWOPI    = 1.0f/(2.0f*M_PI);

  using std::min;
  using std::max;
  using std::sqrt;
  using std::abs;

  static inline int as_int(float x) 
  {
    int res; 
    memcpy((void*)&res, (void*)&x, sizeof(float)); // modern C++ allow only this way, speed ik ok, check assembly with godbolt
    return res; 
  }

  static inline uint as_uint(float x) 
  {
    uint res; 
    memcpy((void*)&res, (void*)&x, sizeof(float)); // modern C++ allow only this way, speed ik ok, check assembly with godbolt
    return res; 
  }

  static inline float as_float(int x)
  {
    float res; 
    memcpy((void*)&res, (void*)&x, sizeof(float)); // modern C++ allow only this way, speed ik ok, check assembly with godbolt
    return res; 
  }

  static inline int   as_int32(float x)  { return as_int(x);    }
  static inline uint  as_uint32(float x) { return as_uint(x); }
  static inline float as_float32(int x)  { return as_float(x);  }

  #ifdef _MSC_VER
  static inline unsigned short     bitCount16(unsigned short x)     { return __popcnt16(x); }
  static inline unsigned int       bitCount32(unsigned int x)       { return __popcnt(x); }
  static inline unsigned int       bitCount  (unsigned int x)       { return __popcnt(x); }
  static inline unsigned long long bitCount64(unsigned long long x) { return __popcnt64(x); }
  #else
  static inline unsigned short     bitCount16(unsigned short x)     { return __builtin_popcount((unsigned int)(x)); }
  static inline unsigned int       bitCount32(unsigned int x)       { return __builtin_popcount(x); }
  static inline unsigned int       bitCount  (unsigned int x)       { return __builtin_popcount(x); }
  static inline unsigned long long bitCount64(unsigned long long x) { return __builtin_popcountll(x); }
  #endif

  static inline double clamp(double u, double a, double b) { return std::min(std::max(a, u), b); }
  static inline float clamp(float u, float a, float b) { return std::min(std::max(a, u), b); }
  static inline uint  clamp(uint u,  uint a,  uint b)  { return std::min(std::max(a, u), b); }
  static inline int   clamp(int u,   int a,   int b)   { return std::min(std::max(a, u), b); }

  inline float rnd(float s, float e)
  {
    const float t = (float)(rand()) / (float)RAND_MAX;
    return s + t*(e - s);
  }
  
  template<typename T> inline T SQR(T x) { return x * x; }

  static inline float  lerp(float u, float v, float t) { return u + t * (v - u);  } 
  static inline float  mix (float u, float v, float t) { return u + t * (v - u);  } 
  static inline float  dot (float a, float b)          { return a*b;  } 

  static inline float smoothstep(float edge0, float edge1, float x)
  {
    float  tVal = (x - edge0) / (edge1 - edge0);
    float  t    = std::min(std::max(tVal, 0.0f), 1.0f); 
    return t * t * (3.0f - 2.0f * t);
  }

  static inline float fract(float x)        { return x - std::floor(x); }
  static inline float mod(float x, float y) { return x - y * std::floor(x/y); }
  static inline float sign(float x) // TODO: on some architectures we can try to effitiently check sign bit       
  { 
    if(x == 0.0f)     return 0.0f;
    else if(x < 0.0f) return -1.0f;
    else              return +1.0f;
  } 
  static inline double sign(double x) // TODO: on some architectures we can try to effitiently check sign bit       
  { 
    if(x == 0.0)     return 0.0;
    else if(x < 0.0) return -1.0;
    else             return +1.0;
  } 
  
  static inline float inversesqrt(float x) { return 1.0f/std::sqrt(x); }
  static inline float rcp(float x)         { return 1.0f/x; }

  static inline int sign(int x) // TODO: on some architectures we can try to effitiently check sign bit       
  { 
    if(x == 0)     return 0;
    else if(x < 0) return -1;
    else           return +1;
  }

## for Tests in AllTests
## for Test  in Tests.Tests
  struct {{Test.Type}};
## endfor
## endfor

## for Tests in AllTests
## for Test  in Tests.Tests
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct {{Test.Type}}
  {
    inline {{Test.Type}}() :{% for Coord in Test.XYZW %} {{Coord}}(0){% if loop.index1 != Test.VecLen %},{% endif %}{% endfor %} {}
    inline {{Test.Type}}({% for Coord in Test.XYZW %}{{Test.TypeS}} a_{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}) :{% for Coord in Test.XYZW %} {{Coord}}(a_{{Coord}}){% if loop.index1 != Test.VecLen %},{% endif %}{% endfor %} {}
    inline explicit {{Test.Type}}({{Test.TypeS}} a_val) :{% for Coord in Test.XYZW %} {{Coord}}(a_val){% if loop.index1 != Test.VecLen %},{% endif %}{% endfor %} {}
    inline explicit {{Test.Type}}(const {{Test.TypeS}} a[{{Test.VecLen}}]) :{% for Coord in Test.XYZW %} {{Coord}}(a[{{loop.index}}]){% if loop.index1 != Test.VecLen %},{% endif %}{% endfor %} {}
    {% for TypeFrom in Test.TypesCV %}
    inline explicit {{Test.Type}}({{TypeFrom}}{{Test.VecLen}} a); {% endfor %}
    
    inline {{Test.TypeS}}& operator[](int i)       { return M[i]; }
    inline {{Test.TypeS}}  operator[](int i) const { return M[i]; }

    union
    {
      struct { {{Test.TypeS}}{% for Coord in Test.XYZW %} {{Coord}}{% if loop.index1 != Test.VecLen %},{% endif %}{% endfor %}; };{% if Test.VecLen == 3 %}
      #ifdef LAYOUT_STD140
      {{Test.TypeS}} M[4];
      #else
      {{Test.TypeS}} M[3];
      #endif{% else %}
      {{Test.TypeS}} M[{{Test.VecLen}}];{% endif %}
    };
  };

  static inline {{Test.Type}} operator+({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} + b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator-({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} - b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator*({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} * b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator/({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} / b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }

  static inline {{Test.Type}} operator * ({{Test.TypeC}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} * b{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator / ({{Test.TypeC}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} / b{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator * ({{Test.TypeS}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a * b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator / ({{Test.TypeS}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a / b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }

  static inline {{Test.Type}} operator + ({{Test.TypeC}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} + b{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator - ({{Test.TypeC}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} - b{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator + ({{Test.TypeS}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a + b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator - ({{Test.TypeS}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a - b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator - ({{Test.TypeC}} a)                   { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}-a.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }

  static inline {{Test.Type}}& operator *= ({{Test.Type}}& a, {{Test.TypeC}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} *= b.{{Coord}}; {% endfor %} return a; }
  static inline {{Test.Type}}& operator /= ({{Test.Type}}& a, {{Test.TypeC}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} /= b.{{Coord}}; {% endfor %} return a; }
  static inline {{Test.Type}}& operator *= ({{Test.Type}}& a, {{Test.TypeS}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} *= b; {% endfor %} return a; }
  static inline {{Test.Type}}& operator /= ({{Test.Type}}& a, {{Test.TypeS}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} /= b; {% endfor %} return a; }

  static inline {{Test.Type}}& operator += ({{Test.Type}}& a, {{Test.TypeC}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} += b.{{Coord}}; {% endfor %} return a; }
  static inline {{Test.Type}}& operator -= ({{Test.Type}}& a, {{Test.TypeC}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} -= b.{{Coord}}; {% endfor %} return a; }
  static inline {{Test.Type}}& operator += ({{Test.Type}}& a, {{Test.TypeS}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} += b; {% endfor %} return a; }
  static inline {{Test.Type}}& operator -= ({{Test.Type}}& a, {{Test.TypeS}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} -= b; {% endfor %} return a; }

  static inline uint{{Test.VecLen}} operator> ({{Test.TypeC}} a, {{Test.TypeC}} b) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} >  b.{{Coord}} ? 0xFFFFFFFF : 0{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline uint{{Test.VecLen}} operator< ({{Test.TypeC}} a, {{Test.TypeC}} b) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} <  b.{{Coord}} ? 0xFFFFFFFF : 0{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline uint{{Test.VecLen}} operator>=({{Test.TypeC}} a, {{Test.TypeC}} b) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} >= b.{{Coord}} ? 0xFFFFFFFF : 0{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline uint{{Test.VecLen}} operator<=({{Test.TypeC}} a, {{Test.TypeC}} b) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} <= b.{{Coord}} ? 0xFFFFFFFF : 0{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline uint{{Test.VecLen}} operator==({{Test.TypeC}} a, {{Test.TypeC}} b) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} == b.{{Coord}} ? 0xFFFFFFFF : 0{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline uint{{Test.VecLen}} operator!=({{Test.TypeC}} a, {{Test.TypeC}} b) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} != b.{{Coord}} ? 0xFFFFFFFF : 0{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  {% if not Test.IsFloat %} 
  static inline {{Test.Type}} operator& ({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} & b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator| ({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} | b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator~ ({{Test.TypeC}} a)                { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}~a.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator>>({{Test.TypeC}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} >> b{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator<<({{Test.TypeC}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} << b{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
 
  static inline bool all_of({{Test.TypeC}} a) { return ({% for Coord in Test.XYZW %}a.{{Coord}} != 0{% if loop.index1 != Test.VecLen %} && {% endif %}{% endfor %}); } 
  static inline bool any_of({{Test.TypeC}} a) { return ({% for Coord in Test.XYZW %}a.{{Coord}} != 0{% if loop.index1 != Test.VecLen %} || {% endif %}{% endfor %}); } 
 
  {% endif %}
  static inline void store  ({{Test.TypeS}}* p, {{Test.TypeC}} a_val) { memcpy((void*)p, (void*)&a_val, sizeof({{Test.TypeS}})*{{Test.VecLen}}); }
  static inline void store_u({{Test.TypeS}}* p, {{Test.TypeC}} a_val) { memcpy((void*)p, (void*)&a_val, sizeof({{Test.TypeS}})*{{Test.VecLen}}); }  


  static inline {{Test.Type}} min  ({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::min(a.{{Coord}}, b.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} max  ({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::max(a.{{Coord}}, b.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} clamp({{Test.TypeC}} u, {{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}clamp(u.{{Coord}}, a.{{Coord}}, b.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} clamp({{Test.TypeC}} u, {{Test.TypeS}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}clamp(u.{{Coord}}, a, b){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  {% if Test.IsSigned %}
  static inline {{Test.Type}} abs ({{Test.TypeC}} a) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::abs(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; } 
  static inline {{Test.Type}} sign({{Test.TypeC}} a) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}sign(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  {% endif %}{% if Test.IsFloat %}
  static inline {{Test.Type}} lerp({{Test.TypeC}} a, {{Test.TypeC}} b, {{Test.TypeS}} t) { return a + t * (b - a); }
  static inline {{Test.Type}} mix ({{Test.TypeC}} a, {{Test.TypeC}} b, {{Test.TypeS}} t) { return a + t * (b - a); }
  static inline {{Test.Type}} floor({{Test.TypeC}} a)                { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::floor(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} ceil({{Test.TypeC}} a)                 { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::ceil(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} rcp ({{Test.TypeC}} a)                 { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}1.0f/a.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} mod ({{Test.TypeC}} x, {{Test.TypeC}} y) { return x - y * floor(x/y); }
  static inline {{Test.Type}} fract({{Test.TypeC}} x)                { return x - floor(x); }
  static inline {{Test.Type}} sqrt({{Test.TypeC}} a)                 { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::sqrt(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} inversesqrt({{Test.TypeC}} a)          { return 1.0f/sqrt(a); }
  {% endif %}  
  static inline  {{Test.TypeS}} dot({{Test.TypeC}} a, {{Test.TypeC}} b)  { return {% for Coord in Test.XYZW %}a.{{Coord}}*b.{{Coord}}{% if loop.index1 != Test.VecLen %} + {% endif %}{% endfor %}; }
  {% if Test.IsFloat %}
  static inline  {{Test.TypeS}} length({{Test.TypeC}} a) { return std::sqrt(dot(a,a)); }
  static inline  {{Test.Type}} normalize({{Test.TypeC}} a) { {{Test.TypeS}} lenInv = {{Test.TypeS}}(1)/length(a); return a*lenInv; }
  {% endif %}
  {% if Test.VecLen == 4 %}
  static inline {{Test.TypeS}}  dot3({{Test.TypeC}} a, {{Test.TypeC}} b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
  static inline {{Test.TypeS}}  dot4({{Test.TypeC}} a, {{Test.TypeC}} b)  { return dot(a,b); } {% if Test.IsFloat %}
  static inline {{Test.TypeS}}  dot3f({{Test.TypeC}} a, {{Test.TypeC}} b) { return dot3(a,b); }
  static inline {{Test.TypeS}}  dot4f({{Test.TypeC}} a, {{Test.TypeC}} b) { return dot(a,b); }
  static inline {{Test.Type}} dot3v({{Test.TypeC}} a, {{Test.TypeC}} b) { {{Test.TypeS}} res = dot3(a,b); return {{Test.Type}}(res); }
  static inline {{Test.Type}} dot4v({{Test.TypeC}} a, {{Test.TypeC}} b) { {{Test.TypeS}} res = dot(a,b);  return {{Test.Type}}(res); }
  {% if Test.IsFloat %}
  static inline {{Test.TypeS}} length3({{Test.TypeC}} a)  { return std::sqrt(dot3(a,a)); }
  static inline {{Test.TypeS}} length3f({{Test.TypeC}} a) { return std::sqrt(dot3(a,a)); }
  static inline {{Test.TypeS}} length4({{Test.TypeC}} a)  { return std::sqrt(dot4(a,a)); }
  static inline {{Test.TypeS}} length4f({{Test.TypeC}} a) { return std::sqrt(dot4(a,a)); }
  static inline {{Test.Type}} length3v({{Test.TypeC}} a) { {{Test.TypeS}} res = std::sqrt(dot3(a,a)); return {{Test.Type}}(res); }
  static inline {{Test.Type}} length4v({{Test.TypeC}} a) { {{Test.TypeS}} res = std::sqrt(dot4(a,a)); return {{Test.Type}}(res); }
  static inline {{Test.Type}} normalize3({{Test.TypeC}} a) { {{Test.TypeS}} lenInv = {{Test.TypeS}}(1)/length3(a); return a*lenInv; }
  {% endif %} {% endif %} {% endif %}
  static inline {{Test.Type}} blend({{Test.TypeC}} a, {{Test.TypeC}} b, const uint{{Test.VecLen}} mask) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}(mask.{{Coord}} == 0) ? b.{{Coord}} : a.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  {% for Coord in Test.XYZW %}
  static inline {{Test.TypeS}} extract_{{loop.index}}({{Test.TypeC}} a) { return a.{{Coord}}; } {% endfor %}
  {% for Coord2 in Test.XYZW %}
  static inline {{Test.Type}} splat_{{loop.index}}({{Test.TypeC}} a) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord2}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; } {% endfor %}
  {% if Test.VecLen == 4 %}
  static inline {{Test.TypeS}} hmin({{Test.TypeC}} a)  { return std::min(std::min(a.x, a.y), std::min(a.z, a.w) ); }
  static inline {{Test.TypeS}} hmax({{Test.TypeC}} a)  { return std::max(std::max(a.x, a.y), std::max(a.z, a.w) ); }
  static inline {{Test.TypeS}} hmin3({{Test.TypeC}} a) { return std::min(a.x, std::min(a.y, a.z) ); }
  static inline {{Test.TypeS}} hmax3({{Test.TypeC}} a) { return std::max(a.x, std::max(a.y, a.z) ); }

  static inline {{Test.Type}} shuffle_xzyw({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.x, a.z, a.y, a.w{{CLS}}; }
  static inline {{Test.Type}} shuffle_yxzw({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.y, a.x, a.z, a.w{{CLS}}; }
  static inline {{Test.Type}} shuffle_yzxw({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.y, a.z, a.x, a.w{{CLS}}; }
  static inline {{Test.Type}} shuffle_zxyw({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.z, a.x, a.y, a.w{{CLS}}; }
  static inline {{Test.Type}} shuffle_zyxw({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.z, a.y, a.x, a.w{{CLS}}; }
  static inline {{Test.Type}} shuffle_xyxy({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.x, a.y, a.x, a.y{{CLS}}; }
  static inline {{Test.Type}} shuffle_zwzw({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.z, a.w, a.z, a.w{{CLS}}; }
  static inline {{Test.Type}} cross3({{Test.TypeC}} a, {{Test.TypeC}} b) 
  {
    const {{Test.Type}} a_yzx = shuffle_yzxw(a);
    const {{Test.Type}} b_yzx = shuffle_yzxw(b);
    return shuffle_yzxw(a*b_yzx - a_yzx*b);
  }
  static inline {{Test.Type}} cross({{Test.TypeC}} a, {{Test.TypeC}} b) { return cross3(a,b); }
  {% else if Test.VecLen == 3 %}
  static inline {{Test.TypeS}} hmin({{Test.TypeC}} a) { return std::min(a.x, std::min(a.y, a.z) ); }
  static inline {{Test.TypeS}} hmax({{Test.TypeC}} a) { return std::max(a.x, std::max(a.y, a.z) ); }

  static inline {{Test.Type}} shuffle_xzy({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.x, a.z, a.y{{CLS}}; }
  static inline {{Test.Type}} shuffle_yxz({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.y, a.x, a.z{{CLS}}; }
  static inline {{Test.Type}} shuffle_yzx({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.y, a.z, a.x{{CLS}}; }
  static inline {{Test.Type}} shuffle_zxy({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.z, a.x, a.y{{CLS}}; }
  static inline {{Test.Type}} shuffle_zyx({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.z, a.y, a.x{{CLS}}; }
  static inline {{Test.Type}} cross({{Test.TypeC}} a, {{Test.TypeC}} b) 
  {
    const {{Test.Type}} a_yzx = shuffle_yzx(a);
    const {{Test.Type}} b_yzx = shuffle_yzx(b);
    return shuffle_yzx(a*b_yzx - a_yzx*b);
  }
  {% else if Test.VecLen == 2 %}
  static inline {{Test.TypeS}} hmin({{Test.TypeC}} a) { return std::min(a.x, a.y); }
  static inline {{Test.TypeS}} hmax({{Test.TypeC}} a) { return std::max(a.x, a.y); }

  static inline {{Test.Type}} shuffle_yx({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.y, a.x{{CLS}}; }
  {% endif %} 

## endfor
## endfor
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## for Tests in AllTests
## for Test  in Tests.Tests
  {% for TypeFrom in Test.TypesCV %}
  inline {{Test.Type}}::{{Test.Type}}({{TypeFrom}}{{Test.VecLen}} a) :{% for Coord in Test.XYZW %} {{Coord}}({{Test.TypeS}}(a[{{loop.index}}])){% if loop.index1 != Test.VecLen %},{% endif %}{% endfor %} {} {% endfor %}
  
  {% if Test.IsFloat %}
  static inline int{{Test.VecLen}}  to_int32 ({{Test.TypeC}} a) { return int{{Test.VecLen}} {{OPN}}{% for Coord in Test.XYZW %}int (a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline uint{{Test.VecLen}} to_uint32({{Test.TypeC}} a) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}uint(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline int{{Test.VecLen}}  as_int32 ({{Test.TypeC}} a) { int{{Test.VecLen}}  res; memcpy((void*)&res, (void*)&a, sizeof(int)*{{Test.VecLen}});  return res; }
  static inline uint{{Test.VecLen}} as_uint32({{Test.TypeC}} a) { uint{{Test.VecLen}} res; memcpy((void*)&res, (void*)&a, sizeof(uint)*{{Test.VecLen}}); return res; } 

  static inline {{Test.Type}} reflect({{Test.TypeC}} dir, {{Test.TypeC}} normal) { return normal * dot(dir, normal) * (-2.0f) + dir; }
  static inline {{Test.Type}} refract({{Test.TypeC}} incidentVec, {{Test.TypeC}} normal, {{Test.TypeS}} eta)
  {
    {{Test.TypeS}} N_dot_I = dot(normal, incidentVec);
    {{Test.TypeS}} k = {{Test.TypeS}}(1.f) - eta * eta * ({{Test.TypeS}}(1.f) - N_dot_I * N_dot_I);
    if (k < {{Test.TypeS}}(0.f))
      return {{Test.Type}}(0.f);
    else
      return eta * incidentVec - (eta * N_dot_I + std::sqrt(k)) * normal;
  }
  // A floating-point, surface normal vector that is facing the view direction
  static inline {{Test.Type}} faceforward({{Test.TypeC}} N, {{Test.TypeC}} I, {{Test.TypeC}} Ng) { return dot(I, Ng) < {{Test.TypeS}}(0) ? N : {{Test.TypeS}}(-1)*N; }
  {% else %}
  static inline float{{Test.VecLen}} to_float32({{Test.TypeC}} a) { return float{{Test.VecLen}} {{OPN}}{% for Coord in Test.XYZW %}float(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline float{{Test.VecLen}} as_float32({{Test.TypeC}} a) { float{{Test.VecLen}} res; memcpy((void*)&res, (void*)&a, sizeof(uint)*{{Test.VecLen}}); return res; }
  {% endif %} 
## endfor
## endfor
## for Tests in AllTests
## for Test  in Tests.Tests
  static inline {{Test.Type}} make_{{Test.Type}}({% for Coord in Test.XYZW %}{{Test.TypeS}} {{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
## endfor
## endfor
  static inline float3  to_float3(float4 f4)           { return float3(f4.x, f4.y, f4.z); }
  static inline float4  to_float4(float3 v, float w)   { return float4(v.x, v.y, v.z, w); }
  static inline double3 to_double3(double4 f4)         { return double3(f4.x, f4.y, f4.z); }
  static inline double4 to_double4(double3 v, float w) { return double4(v.x, v.y, v.z, w); }
  static inline uint3   to_uint3 (uint4 f4)            { return uint3(f4.x, f4.y, f4.z);  }
  static inline uint4   to_uint4 (uint3 v, uint w)     { return uint4(v.x, v.y, v.z, w);  }
  static inline int3    to_int3  (int4 f4)             { return int3(f4.x, f4.y, f4.z);   }
  static inline int4    to_int4  (int3 v, int w)       { return int4(v.x, v.y, v.z, w);   }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct ushort4
  {
    inline ushort4() : x(0), y(0), z(0), w(0) {}
    inline ushort4(ushort a_x, ushort a_y, ushort a_z, ushort a_w) : x(a_x), y(a_y), z(a_z), w(a_w) {}
    inline explicit ushort4(ushort a_val) : x(a_val), y(a_val), z(a_val), w(a_val) {}
    inline explicit ushort4(const ushort a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

    inline ushort& operator[](int i)       { return M[i]; }
    inline ushort  operator[](int i) const { return M[i]; }

    union
    {
      struct { ushort x, y, z, w; };
      ushort   M[4];
    };
  };

  struct ushort2
  {
    inline ushort2() : x(0), y(0) {}
    inline ushort2(ushort a_x, ushort a_y) : x(a_x), y(a_y) {}
    inline explicit ushort2(ushort a_val) : x(a_val), y(a_val){}
    inline explicit ushort2(const ushort a[2]) : x(a[0]), y(a[1]) {}

    inline ushort& operator[](int i)       { return M[i]; }
    inline ushort  operator[](int i) const { return M[i]; }

    union
    {
      struct { ushort x, y; };
      ushort   M[2];
      uint64_t u64;
    };
  };

  struct uchar4
  {
    inline uchar4() : x(0), y(0), z(0), w(0) {}
    inline uchar4(uchar a_x, uchar a_y, uchar a_z, uchar a_w) : x(a_x), y(a_y), z(a_z), w(a_w) {}
    inline explicit uchar4(uchar a_val) : x(a_val), y(a_val), z(a_val), w(a_val) {}
    inline explicit uchar4(const uchar a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

    inline uchar& operator[](int i)       { return M[i]; }
    inline uchar  operator[](int i) const { return M[i]; }

    union
    {
      struct { uchar x, y, z, w; };
      uchar M[4];
      uint32_t u32;
    };
  };

  static inline uchar4 operator * (const uchar4 & u, float v) { return uchar4(uchar(float(u.x) * v), uchar(float(u.y) * v),
                                                                              uchar(float(u.z) * v), uchar(float(u.w) * v)); }
  static inline uchar4 operator / (const uchar4 & u, float v) { return uchar4(uchar(float(u.x) / v), uchar(float(u.y) / v),
                                                                              uchar(float(u.z) / v), uchar(float(u.w) / v)); }
  static inline uchar4 operator + (const uchar4 & u, float v) { return uchar4(uchar(float(u.x) + v), uchar(float(u.y) + v),
                                                                              uchar(float(u.z) + v), uchar(float(u.w) + v)); }
  static inline uchar4 operator - (const uchar4 & u, float v) { return uchar4(uchar(float(u.x) - v), uchar(float(u.y) - v),
                                                                              uchar(float(u.z) - v), uchar(float(u.w) - v)); }
  static inline uchar4 operator * (float v, const uchar4 & u) { return uchar4(uchar(v * float(u.x)), uchar(v * float(u.y)),
                                                                              uchar(v * float(u.z)), uchar(v * float(u.w))); }
  static inline uchar4 operator / (float v, const uchar4 & u) { return uchar4(uchar(v / float(u.x)), uchar(v / float(u.y)),
                                                                              uchar(v / float(u.z)), uchar(v / float(u.w))); }
  static inline uchar4 operator + (float v, const uchar4 & u) { return uchar4(uchar(float(u.x) + v), uchar(float(u.y) + v),
                                                                              uchar(float(u.z) + v), uchar(float(u.w) + v)); }
  static inline uchar4 operator - (float v, const uchar4 & u) { return uchar4(uchar(float(u.x) - v), uchar(float(u.y) - v),
                                                                              uchar(float(u.z) - v), uchar(float(u.w) - v)); }

  static inline uchar4 operator + (const uchar4 & u, const uchar4 & v) { return uchar4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w); }
  static inline uchar4 operator - (const uchar4 & u, const uchar4 & v) { return uchar4(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w); }
  static inline uchar4 operator * (const uchar4 & u, const uchar4 & v) { return uchar4(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w); }
  static inline uchar4 operator / (const uchar4 & u, const uchar4 & v) { return uchar4(u.x / v.x, u.y / v.y, u.z / v.z, u.w / v.w); }
  
  static inline uchar4 lerp(const uchar4 & u, const uchar4 & v, float t) { return u + t * (v - u); }
  static inline int    dot(uchar4 a, uchar4 b) { return int(a.x)*int(b.x) + int(a.y)*int(b.y) + int(a.z)*int(b.z); }
 
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) { return ushort4{x, y, z, w}; } //
  static inline uchar4  make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)      { return uchar4{x, y, z, w};  } //

  {% for FType in MatTypes %}

  static inline void mat4_rowmajor_mul_mat4({{FType.Name}}* __restrict M, const {{FType.Name}}* __restrict A, const {{FType.Name}}* __restrict B) // modern gcc compiler succesfuly vectorize such implementation!
  {
  	M[ 0] = A[ 0] * B[ 0] + A[ 1] * B[ 4] + A[ 2] * B[ 8] + A[ 3] * B[12];
  	M[ 1] = A[ 0] * B[ 1] + A[ 1] * B[ 5] + A[ 2] * B[ 9] + A[ 3] * B[13];
  	M[ 2] = A[ 0] * B[ 2] + A[ 1] * B[ 6] + A[ 2] * B[10] + A[ 3] * B[14];
  	M[ 3] = A[ 0] * B[ 3] + A[ 1] * B[ 7] + A[ 2] * B[11] + A[ 3] * B[15];
  	M[ 4] = A[ 4] * B[ 0] + A[ 5] * B[ 4] + A[ 6] * B[ 8] + A[ 7] * B[12];
  	M[ 5] = A[ 4] * B[ 1] + A[ 5] * B[ 5] + A[ 6] * B[ 9] + A[ 7] * B[13];
  	M[ 6] = A[ 4] * B[ 2] + A[ 5] * B[ 6] + A[ 6] * B[10] + A[ 7] * B[14];
  	M[ 7] = A[ 4] * B[ 3] + A[ 5] * B[ 7] + A[ 6] * B[11] + A[ 7] * B[15];
  	M[ 8] = A[ 8] * B[ 0] + A[ 9] * B[ 4] + A[10] * B[ 8] + A[11] * B[12];
  	M[ 9] = A[ 8] * B[ 1] + A[ 9] * B[ 5] + A[10] * B[ 9] + A[11] * B[13];
  	M[10] = A[ 8] * B[ 2] + A[ 9] * B[ 6] + A[10] * B[10] + A[11] * B[14];
  	M[11] = A[ 8] * B[ 3] + A[ 9] * B[ 7] + A[10] * B[11] + A[11] * B[15];
  	M[12] = A[12] * B[ 0] + A[13] * B[ 4] + A[14] * B[ 8] + A[15] * B[12];
  	M[13] = A[12] * B[ 1] + A[13] * B[ 5] + A[14] * B[ 9] + A[15] * B[13];
  	M[14] = A[12] * B[ 2] + A[13] * B[ 6] + A[14] * B[10] + A[15] * B[14];
  	M[15] = A[12] * B[ 3] + A[13] * B[ 7] + A[14] * B[11] + A[15] * B[15];
  }

  static inline void mat4_colmajor_mul_vec4({{FType.Name}}* __restrict RES, const {{FType.Name}}* __restrict B, const {{FType.Name}}* __restrict V) // modern gcc compiler succesfuly vectorize such implementation!
  {
  	RES[0] = V[0] * B[0] + V[1] * B[4] + V[2] * B[ 8] + V[3] * B[12];
  	RES[1] = V[0] * B[1] + V[1] * B[5] + V[2] * B[ 9] + V[3] * B[13];
  	RES[2] = V[0] * B[2] + V[1] * B[6] + V[2] * B[10] + V[3] * B[14];
  	RES[3] = V[0] * B[3] + V[1] * B[7] + V[2] * B[11] + V[3] * B[15];
  }

  static inline void mat3_colmajor_mul_vec3({{FType.Name}}* __restrict RES, const {{FType.Name}}* __restrict B, const {{FType.Name}}* __restrict V) 
  {
  	RES[0] = V[0] * B[0] + V[1] * B[3] + V[2] * B[6];
  	RES[1] = V[0] * B[1] + V[1] * B[4] + V[2] * B[7];
  	RES[2] = V[0] * B[2] + V[1] * B[5] + V[2] * B[8];
  }

  static inline void transpose4(const {{FType.Name}}4 in_rows[4], {{FType.Name}}4 out_rows[4])
  {
    CVEX_ALIGNED({{FType.Align}}) {{FType.Name}} rows[16];
    store(rows+0,  in_rows[0]);
    store(rows+4,  in_rows[1]);
    store(rows+8,  in_rows[2]);
    store(rows+12, in_rows[3]);
  
    out_rows[0] = {{FType.Name}}4{rows[0], rows[4], rows[8],  rows[12]};
    out_rows[1] = {{FType.Name}}4{rows[1], rows[5], rows[9],  rows[13]};
    out_rows[2] = {{FType.Name}}4{rows[2], rows[6], rows[10], rows[14]};
    out_rows[3] = {{FType.Name}}4{rows[3], rows[7], rows[11], rows[15]};
  }

  static inline void transpose3(const {{FType.Name}}3 in_rows[3], {{FType.Name}}3 out_rows[3])
  {
    {{FType.Name}} rows[9];
    store(rows+0,  in_rows[0]);
    store(rows+3,  in_rows[1]);
    store(rows+6,  in_rows[2]);
  
    out_rows[0] = {{FType.Name}}3{rows[0], rows[3], rows[6]};
    out_rows[1] = {{FType.Name}}3{rows[1], rows[4], rows[7]};
    out_rows[2] = {{FType.Name}}3{rows[2], rows[5], rows[8]};
  }

  /**
  \brief this class use colmajor memory layout for effitient vector-matrix operations
  */
  struct  {{FType.Name}}4x4
  {
    inline {{FType.Name}}4x4()  { identity(); }

    inline explicit {{FType.Name}}4x4(const {{FType.Name}} A[16])
    {
      m_col[0] =  {{FType.Name}}4{ A[0], A[4], A[8],  A[12] };
      m_col[1] =  {{FType.Name}}4{ A[1], A[5], A[9],  A[13] };
      m_col[2] =  {{FType.Name}}4{ A[2], A[6], A[10], A[14] };
      m_col[3] =  {{FType.Name}}4{ A[3], A[7], A[11], A[15] };
    }

    inline explicit {{FType.Name}}4x4({{FType.Name}} A0,  {{FType.Name}} A1,  {{FType.Name}} A2, {{FType.Name}} A3,
                                      {{FType.Name}} A4,  {{FType.Name}} A5,  {{FType.Name}} A6,  {{FType.Name}} A7,
                                      {{FType.Name}} A8,  {{FType.Name}} A9,  {{FType.Name}} A10, {{FType.Name}} A11,
                                      {{FType.Name}} A12, {{FType.Name}} A13, {{FType.Name}} A14, {{FType.Name}} A15)
    {
      m_col[0] = {{FType.Name}}4{ A0, A4, A8,  A12 };
      m_col[1] = {{FType.Name}}4{ A1, A5, A9,  A13 };
      m_col[2] = {{FType.Name}}4{ A2, A6, A10, A14 };
      m_col[3] = {{FType.Name}}4{ A3, A7, A11, A15 };
    }

    inline {{FType.Name}}4x4& operator=(const {{FType.Name}}4x4& rhs)
    {
      m_col[0] = rhs.m_col[0];
      m_col[1] = rhs.m_col[1];
      m_col[2] = rhs.m_col[2]; 
      m_col[3] = rhs.m_col[3]; 
      return *this;
    }

    inline void identity()
    {
      m_col[0] = {{FType.Name}}4{ 1.0f, 0.0f, 0.0f, 0.0f };
      m_col[1] = {{FType.Name}}4{ 0.0f, 1.0f, 0.0f, 0.0f };
      m_col[2] = {{FType.Name}}4{ 0.0f, 0.0f, 1.0f, 0.0f };
      m_col[3] = {{FType.Name}}4{ 0.0f, 0.0f, 0.0f, 1.0f };
    }

    inline  {{FType.Name}}4 get_col(int i) const { return m_col[i]; }
    inline void set_col(int i, const  {{FType.Name}}4& a_col) { m_col[i] = a_col; }

    inline  {{FType.Name}}4 get_row(int i) const { return {{FType.Name}}4{ m_col[0][i], m_col[1][i], m_col[2][i], m_col[3][i] }; }
    inline void set_row(int i, const {{FType.Name}}4& a_col)
    {
      m_col[0][i] = a_col[0];
      m_col[1][i] = a_col[1];
      m_col[2][i] = a_col[2];
      m_col[3][i] = a_col[3];
    }

    inline {{FType.Name}}4& col(int i)       { return m_col[i]; }
    inline {{FType.Name}}4  col(int i) const { return m_col[i]; }

    inline  {{FType.Name}}& operator()(int row, int col)       { return m_col[col][row]; }
    inline  {{FType.Name}}  operator()(int row, int col) const { return m_col[col][row]; }

    struct RowTmp 
    {
      {{FType.Name}}4x4* self;
      int       row;
      inline  {{FType.Name}}& operator[](int col)       { return self->m_col[col][row]; }
      inline  {{FType.Name}}  operator[](int col) const { return self->m_col[col][row]; }
    };

    inline RowTmp operator[](int a_row) 
    {
      RowTmp row;
      row.self = this;
      row.row  = a_row;
      return row;
    }

    {{FType.Name}}4 m_col[4];
  };

  static inline {{FType.Name}}4x4 make_{{FType.Name}}4x4_from_rows({{FType.Name}}4 a, {{FType.Name}}4 b, {{FType.Name}}4 c, {{FType.Name}}4 d)
  {
    {{FType.Name}}4x4 m;
    m.set_row(0, a);
    m.set_row(1, b);
    m.set_row(2, c);
    m.set_row(3, d);
    return m;
  }

  static inline {{FType.Name}}4x4 make_{{FType.Name}}4x4_from_cols({{FType.Name}}4 a, {{FType.Name}}4 b, {{FType.Name}}4 c, {{FType.Name}}4 d)
  {
    {{FType.Name}}4x4 m;
    m.set_col(0, a);
    m.set_col(1, b);
    m.set_col(2, c);
    m.set_col(3, d);
    return m;
  }

  static inline {{FType.Name}}4 operator*(const  {{FType.Name}}4x4& m, const {{FType.Name}}4& v)
  {
    {{FType.Name}}4 res;
    mat4_colmajor_mul_vec4(({{FType.Name}}*)&res, (const {{FType.Name}}*)&m, (const {{FType.Name}}*)&v);
    return res;
  }

  static inline {{FType.Name}}4 mul4x4x4(const {{FType.Name}}4x4& m, const {{FType.Name}}4& v) { return m*v; }

  static inline {{FType.Name}}4 mul(const {{FType.Name}}4x4& m, const {{FType.Name}}4& v)
  {
    {{FType.Name}}4 res;
    mat4_colmajor_mul_vec4(({{FType.Name}}*)&res, (const {{FType.Name}}*)&m, (const {{FType.Name}}*)&v);
    return res;
  }

  static inline {{FType.Name}}3 operator*(const {{FType.Name}}4x4& m, const {{FType.Name}}3& v)
  {
    {{FType.Name}}4 v2 = {{FType.Name}}4{v.x, v.y, v.z, 1.0f}; 
    {{FType.Name}}4 res;                             
    mat4_colmajor_mul_vec4(({{FType.Name}}*)&res, (const {{FType.Name}}*)&m, (const {{FType.Name}}*)&v2);
    return to_{{FType.Name}}3(res);
  }

  static inline {{FType.Name}}3 mul(const {{FType.Name}}4x4& m, const {{FType.Name}}3& v)
  {
    {{FType.Name}}4 v2 = {{FType.Name}}4{v.x, v.y, v.z, 1.0f}; 
    {{FType.Name}}4 res;                             
    mat4_colmajor_mul_vec4(({{FType.Name}}*)&res, (const {{FType.Name}}*)&m, (const {{FType.Name}}*)&v2);
    return to_{{FType.Name}}3(res);
  }

  static inline {{FType.Name}}4x4 transpose(const {{FType.Name}}4x4& rhs)
  {
    {{FType.Name}}4x4 res;
    transpose4(rhs.m_col, res.m_col);
    return res;
  }

  static inline {{FType.Name}}4x4 translate4x4({{FType.Name}}3 t)
  {
    {{FType.Name}}4x4 res;
    res.set_col(3, {{FType.Name}}4{t.x,  t.y,  t.z, 1.0f });
    return res;
  }

  static inline {{FType.Name}}4x4 scale4x4({{FType.Name}}3 t)
  {
    {{FType.Name}}4x4 res;
    res.set_col(0, {{FType.Name}}4{t.x, 0.0f, 0.0f,  0.0f});
    res.set_col(1, {{FType.Name}}4{0.0f, t.y, 0.0f,  0.0f});
    res.set_col(2, {{FType.Name}}4{0.0f, 0.0f,  t.z, 0.0f});
    res.set_col(3, {{FType.Name}}4{0.0f, 0.0f, 0.0f, 1.0f});
    return res;
  }

  static inline {{FType.Name}}4x4 rotate4x4X({{FType.Name}} phi)
  {
    {{FType.Name}}4x4 res;
    res.set_col(0, {{FType.Name}}4{1.0f,      0.0f,       0.0f, 0.0f  });
    res.set_col(1, {{FType.Name}}4{0.0f, +cosf(phi),  +sinf(phi), 0.0f});
    res.set_col(2, {{FType.Name}}4{0.0f, -sinf(phi),  +cosf(phi), 0.0f});
    res.set_col(3, {{FType.Name}}4{0.0f,      0.0f,       0.0f, 1.0f  });
    return res;
  }

  static inline {{FType.Name}}4x4 rotate4x4Y({{FType.Name}} phi)
  {
    {{FType.Name}}4x4 res;
    res.set_col(0, {{FType.Name}}4{+cosf(phi), 0.0f, -sinf(phi), 0.0f});
    res.set_col(1, {{FType.Name}}4{     0.0f, 1.0f,      0.0f, 0.0f  });
    res.set_col(2, {{FType.Name}}4{+sinf(phi), 0.0f, +cosf(phi), 0.0f});
    res.set_col(3, {{FType.Name}}4{     0.0f, 0.0f,      0.0f, 1.0f  });
    return res;
  }

  static inline {{FType.Name}}4x4 rotate4x4Z({{FType.Name}} phi)
  {
    {{FType.Name}}4x4 res;
    res.set_col(0, {{FType.Name}}4{+cosf(phi), sinf(phi), 0.0f, 0.0f});
    res.set_col(1, {{FType.Name}}4{-sinf(phi), cosf(phi), 0.0f, 0.0f});
    res.set_col(2, {{FType.Name}}4{     0.0f,     0.0f, 1.0f, 0.0f  });
    res.set_col(3, {{FType.Name}}4{     0.0f,     0.0f, 0.0f, 1.0f  });
    return res;
  }
  
  static inline {{FType.Name}}4x4 mul({{FType.Name}}4x4 m1, {{FType.Name}}4x4 m2)
  {
    const {{FType.Name}}4 column1 = mul(m1, m2.col(0));
    const {{FType.Name}}4 column2 = mul(m1, m2.col(1));
    const {{FType.Name}}4 column3 = mul(m1, m2.col(2));
    const {{FType.Name}}4 column4 = mul(m1, m2.col(3));
    
    {{FType.Name}}4x4 res;
    res.set_col(0, column1);
    res.set_col(1, column2);
    res.set_col(2, column3);
    res.set_col(3, column4);
    return res;
  }

  static inline {{FType.Name}}4x4 operator*({{FType.Name}}4x4 m1, {{FType.Name}}4x4 m2)
  {
    const {{FType.Name}}4 column1 = mul(m1, m2.col(0));
    const {{FType.Name}}4 column2 = mul(m1, m2.col(1));
    const {{FType.Name}}4 column3 = mul(m1, m2.col(2));
    const {{FType.Name}}4 column4 = mul(m1, m2.col(3));

    {{FType.Name}}4x4 res;
    res.set_col(0, column1);
    res.set_col(1, column2);
    res.set_col(2, column3);
    res.set_col(3, column4);
    return res;
  }
  
  static inline {{FType.Name}}4x4 inverse4x4({{FType.Name}}4x4 m1)
  {
    CVEX_ALIGNED({{FType.Align}}) {{FType.Name}} tmp[12]; // temp array for pairs
    {{FType.Name}}4x4 m;

    // calculate pairs for first 8 elements (cofactors)
    //
    tmp[0]  = m1(2,2) * m1(3,3);
    tmp[1]  = m1(2,3) * m1(3,2);
    tmp[2]  = m1(2,1) * m1(3,3);
    tmp[3]  = m1(2,3) * m1(3,1);
    tmp[4]  = m1(2,1) * m1(3,2);
    tmp[5]  = m1(2,2) * m1(3,1);
    tmp[6]  = m1(2,0) * m1(3,3);
    tmp[7]  = m1(2,3) * m1(3,0);
    tmp[8]  = m1(2,0) * m1(3,2);
    tmp[9]  = m1(2,2) * m1(3,0);
    tmp[10] = m1(2,0) * m1(3,1);
    tmp[11] = m1(2,1) * m1(3,0);

    // calculate first 8 m1.rowents (cofactors)
    //
    m(0,0) = tmp[0]  * m1(1,1) + tmp[3] * m1(1,2) + tmp[4]  * m1(1,3);
    m(0,0) -= tmp[1] * m1(1,1) + tmp[2] * m1(1,2) + tmp[5]  * m1(1,3);
    m(1,0) = tmp[1]  * m1(1,0) + tmp[6] * m1(1,2) + tmp[9]  * m1(1,3);
    m(1,0) -= tmp[0] * m1(1,0) + tmp[7] * m1(1,2) + tmp[8]  * m1(1,3);
    m(2,0) = tmp[2]  * m1(1,0) + tmp[7] * m1(1,1) + tmp[10] * m1(1,3);
    m(2,0) -= tmp[3] * m1(1,0) + tmp[6] * m1(1,1) + tmp[11] * m1(1,3);
    m(3,0) = tmp[5]  * m1(1,0) + tmp[8] * m1(1,1) + tmp[11] * m1(1,2);
    m(3,0) -= tmp[4] * m1(1,0) + tmp[9] * m1(1,1) + tmp[10] * m1(1,2);
    m(0,1) = tmp[1]  * m1(0,1) + tmp[2] * m1(0,2) + tmp[5]  * m1(0,3);
    m(0,1) -= tmp[0] * m1(0,1) + tmp[3] * m1(0,2) + tmp[4]  * m1(0,3);
    m(1,1) = tmp[0]  * m1(0,0) + tmp[7] * m1(0,2) + tmp[8]  * m1(0,3);
    m(1,1) -= tmp[1] * m1(0,0) + tmp[6] * m1(0,2) + tmp[9]  * m1(0,3);
    m(2,1) = tmp[3]  * m1(0,0) + tmp[6] * m1(0,1) + tmp[11] * m1(0,3);
    m(2,1) -= tmp[2] * m1(0,0) + tmp[7] * m1(0,1) + tmp[10] * m1(0,3);
    m(3,1) = tmp[4]  * m1(0,0) + tmp[9] * m1(0,1) + tmp[10] * m1(0,2);
    m(3,1) -= tmp[5] * m1(0,0) + tmp[8] * m1(0,1) + tmp[11] * m1(0,2);

    // calculate pairs for second 8 m1.rowents (cofactors)
    //
    tmp[0]  = m1(0,2) * m1(1,3);
    tmp[1]  = m1(0,3) * m1(1,2);
    tmp[2]  = m1(0,1) * m1(1,3);
    tmp[3]  = m1(0,3) * m1(1,1);
    tmp[4]  = m1(0,1) * m1(1,2);
    tmp[5]  = m1(0,2) * m1(1,1);
    tmp[6]  = m1(0,0) * m1(1,3);
    tmp[7]  = m1(0,3) * m1(1,0);
    tmp[8]  = m1(0,0) * m1(1,2);
    tmp[9]  = m1(0,2) * m1(1,0);
    tmp[10] = m1(0,0) * m1(1,1);
    tmp[11] = m1(0,1) * m1(1,0);

    // calculate second 8 m1 (cofactors)
    //
    m(0,2) = tmp[0]   * m1(3,1) + tmp[3]  * m1(3,2) + tmp[4]  * m1(3,3);
    m(0,2) -= tmp[1]  * m1(3,1) + tmp[2]  * m1(3,2) + tmp[5]  * m1(3,3);
    m(1,2) = tmp[1]   * m1(3,0) + tmp[6]  * m1(3,2) + tmp[9]  * m1(3,3);
    m(1,2) -= tmp[0]  * m1(3,0) + tmp[7]  * m1(3,2) + tmp[8]  * m1(3,3);
    m(2,2) = tmp[2]   * m1(3,0) + tmp[7]  * m1(3,1) + tmp[10] * m1(3,3);
    m(2,2) -= tmp[3]  * m1(3,0) + tmp[6]  * m1(3,1) + tmp[11] * m1(3,3);
    m(3,2) = tmp[5]   * m1(3,0) + tmp[8]  * m1(3,1) + tmp[11] * m1(3,2);
    m(3,2) -= tmp[4]  * m1(3,0) + tmp[9]  * m1(3,1) + tmp[10] * m1(3,2);
    m(0,3) = tmp[2]   * m1(2,2) + tmp[5]  * m1(2,3) + tmp[1]  * m1(2,1);
    m(0,3) -= tmp[4]  * m1(2,3) + tmp[0]  * m1(2,1) + tmp[3]  * m1(2,2);
    m(1,3) = tmp[8]   * m1(2,3) + tmp[0]  * m1(2,0) + tmp[7]  * m1(2,2);
    m(1,3) -= tmp[6]  * m1(2,2) + tmp[9]  * m1(2,3) + tmp[1]  * m1(2,0);
    m(2,3) = tmp[6]   * m1(2,1) + tmp[11] * m1(2,3) + tmp[3]  * m1(2,0);
    m(2,3) -= tmp[10] * m1(2,3) + tmp[2]  * m1(2,0) + tmp[7]  * m1(2,1);
    m(3,3) = tmp[10]  * m1(2,2) + tmp[4]  * m1(2,0) + tmp[9]  * m1(2,1);
    m(3,3) -= tmp[8]  * m1(2,1) + tmp[11] * m1(2,2) + tmp[5]  * m1(2,0);

    // calculate matrix inverse
    //
    const {{FType.Name}} k = 1.0f / (m1(0,0) * m(0,0) + m1(0,1) * m(1,0) + m1(0,2) * m(2,0) + m1(0,3) * m(3,0));
    const {{FType.Name}}4 vK{k,k,k,k};

    m.set_col(0, m.get_col(0)*vK);
    m.set_col(1, m.get_col(1)*vK);
    m.set_col(2, m.get_col(2)*vK);
    m.set_col(3, m.get_col(3)*vK);

    return m;
  }

  static inline {{FType.Name}}4x4 operator+({{FType.Name}}4x4 m1, {{FType.Name}}4x4 m2)
  {
    {{FType.Name}}4x4 res;
    res.m_col[0] = m1.m_col[0] + m2.m_col[0];
    res.m_col[1] = m1.m_col[1] + m2.m_col[1];
    res.m_col[2] = m1.m_col[2] + m2.m_col[2];
    res.m_col[3] = m1.m_col[3] + m2.m_col[3];
    return res;
  }

  static inline {{FType.Name}}4x4 operator-({{FType.Name}}4x4 m1, {{FType.Name}}4x4 m2)
  {
    {{FType.Name}}4x4 res;
    res.m_col[0] = m1.m_col[0] - m2.m_col[0];
    res.m_col[1] = m1.m_col[1] - m2.m_col[1];
    res.m_col[2] = m1.m_col[2] - m2.m_col[2];
    res.m_col[3] = m1.m_col[3] - m2.m_col[3];
    return res;
  }

  static inline {{FType.Name}}4x4 outerProduct({{FType.Name}}4 a, {{FType.Name}}4 b) 
  {
    {{FType.Name}}4x4 m;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        m[i][j] = a[i] * b[j];
    return m;
  }

  /**
  \brief this class use colmajor memory layout for effitient vector-matrix operations
  */
  struct {{FType.Name}}3x3
  {
    inline {{FType.Name}}3x3()  { identity(); }
    
    inline explicit {{FType.Name}}3x3(const {{FType.Name}} rhs)
    {
      m_col[0] = {{FType.Name}}3(rhs);
      m_col[1] = {{FType.Name}}3(rhs);
      m_col[2] = {{FType.Name}}3(rhs); 
    } 

    inline {{FType.Name}}3x3(const {{FType.Name}}3x3& rhs) 
    { 
      m_col[0] = rhs.m_col[0];
      m_col[1] = rhs.m_col[1];
      m_col[2] = rhs.m_col[2]; 
    }

    inline {{FType.Name}}3x3& operator=(const {{FType.Name}}3x3& rhs)
    {
      m_col[0] = rhs.m_col[0];
      m_col[1] = rhs.m_col[1];
      m_col[2] = rhs.m_col[2]; 
      return *this;
    }

    // col-major matrix from row-major array
    inline explicit {{FType.Name}}3x3(const {{FType.Name}} A[9])
    {
      m_col[0] = {{FType.Name}}3{ A[0], A[3], A[6] };
      m_col[1] = {{FType.Name}}3{ A[1], A[4], A[7] };
      m_col[2] = {{FType.Name}}3{ A[2], A[5], A[8] };
    }

    inline explicit {{FType.Name}}3x3({{FType.Name}} A0, {{FType.Name}} A1, {{FType.Name}} A2, {{FType.Name}} A3, {{FType.Name}} A4, 
                                      {{FType.Name}} A5, {{FType.Name}} A6, {{FType.Name}} A7, {{FType.Name}} A8)
    {
      m_col[0] = {{FType.Name}}3{ A0, A3, A6 };
      m_col[1] = {{FType.Name}}3{ A1, A4, A7 };
      m_col[2] = {{FType.Name}}3{ A2, A5, A8 };
    }

    inline void identity()
    {
      m_col[0] = {{FType.Name}}3{ 1.0, 0.0, 0.0 };
      m_col[1] = {{FType.Name}}3{ 0.0, 1.0, 0.0 };
      m_col[2] = {{FType.Name}}3{ 0.0, 0.0, 1.0 };
    }

    inline void zero()
    {
      m_col[0] = {{FType.Name}}3{ 0.0, 0.0, 0.0 };
      m_col[1] = {{FType.Name}}3{ 0.0, 0.0, 0.0 };
      m_col[2] = {{FType.Name}}3{ 0.0, 0.0, 0.0 };
    }

    inline {{FType.Name}}3 get_col(int i) const { return m_col[i]; }
    inline void set_col(int i, const {{FType.Name}}3& a_col) { m_col[i] = a_col; }

    inline {{FType.Name}}3 get_row(int i) const { return {{FType.Name}}3{ m_col[0][i], m_col[1][i], m_col[2][i] }; }
    inline void set_row(int i, const {{FType.Name}}3& a_col)
    {
      m_col[0][i] = a_col[0];
      m_col[1][i] = a_col[1];
      m_col[2][i] = a_col[2];
    }

    inline {{FType.Name}}3& col(int i)       { return m_col[i]; }
    inline {{FType.Name}}3  col(int i) const { return m_col[i]; }

    inline {{FType.Name}}& operator()(int row, int col)       { return m_col[col][row]; }
    inline {{FType.Name}}  operator()(int row, int col) const { return m_col[col][row]; }

    struct RowTmp 
    {
      {{FType.Name}}3x3* self;
      int row;
      inline {{FType.Name}}& operator[](int col)       { return self->m_col[col][row]; }
      inline {{FType.Name}}  operator[](int col) const { return self->m_col[col][row]; }
    };

    inline RowTmp operator[](int a_row) 
    {
      RowTmp row;
      row.self = this;
      row.row  = a_row;
      return row;
    }

    {{FType.Name}}3 m_col[3];
  };
  
  static inline {{FType.Name}}3x3 make_{{FType.Name}}3x3_from_rows({{FType.Name}}3 a, {{FType.Name}}3 b, {{FType.Name}}3 c)
  {
    {{FType.Name}}3x3 m;
    m.set_row(0, a);
    m.set_row(1, b);
    m.set_row(2, c);
    return m;
  }

  static inline {{FType.Name}}3x3 make_{{FType.Name}}3x3_from_cols({{FType.Name}}3 a, {{FType.Name}}3 b, {{FType.Name}}3 c)
  {
    {{FType.Name}}3x3 m;
    m.set_col(0, a);
    m.set_col(1, b);
    m.set_col(2, c);
    return m;
  }

  static inline {{FType.Name}}3x3 make_{{FType.Name}}3x3({{FType.Name}}3 a, {{FType.Name}}3 b, {{FType.Name}}3 c)            // deprecated
  {
    return make_{{FType.Name}}3x3_from_rows(a,b,c);
  }

  static inline {{FType.Name}}3x3 make_{{FType.Name}}3x3_by_columns({{FType.Name}}3 a, {{FType.Name}}3 b, {{FType.Name}}3 c) // deprecated
  {
    return make_{{FType.Name}}3x3_from_cols(a,b,c);
  }

  static inline {{FType.Name}}3 operator*(const {{FType.Name}}3x3& m, const {{FType.Name}}3& v)
  {
    {{FType.Name}}3 res;
    mat3_colmajor_mul_vec3(({{FType.Name}}*)&res, (const {{FType.Name}}*)&m, (const {{FType.Name}}*)&v);
    return res;
  }

  static inline {{FType.Name}}3 mul(const {{FType.Name}}3x3& m, const {{FType.Name}}3& v)
  {
    {{FType.Name}}3 res;                             
    mat3_colmajor_mul_vec3(({{FType.Name}}*)&res, (const {{FType.Name}}*)&m, (const {{FType.Name}}*)&v);
    return res;
  }

  static inline {{FType.Name}}3x3 transpose(const {{FType.Name}}3x3& rhs)
  {
    {{FType.Name}}3x3 res;
    transpose3(rhs.m_col, res.m_col);
    return res;
  }

  static inline {{FType.Name}} determinant(const {{FType.Name}}3x3& mat)
  {
    const {{FType.Name}} a = mat.m_col[0].x;
    const {{FType.Name}} b = mat.m_col[1].x;
    const {{FType.Name}} c = mat.m_col[2].x;
    const {{FType.Name}} d = mat.m_col[0].y;
    const {{FType.Name}} e = mat.m_col[1].y;
    const {{FType.Name}} f = mat.m_col[2].y;
    const {{FType.Name}} g = mat.m_col[0].z;
    const {{FType.Name}} h = mat.m_col[1].z;
    const {{FType.Name}} i = mat.m_col[2].z;
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  }

  static inline {{FType.Name}}3x3 inverse3x3(const {{FType.Name}}3x3& mat)
  {
    {{FType.Name}} det = determinant(mat);
    {{FType.Name}} inv_det = 1.0 / det;
    {{FType.Name}} a = mat.m_col[0].x;
    {{FType.Name}} b = mat.m_col[1].x;
    {{FType.Name}} c = mat.m_col[2].x;
    {{FType.Name}} d = mat.m_col[0].y;
    {{FType.Name}} e = mat.m_col[1].y;
    {{FType.Name}} f = mat.m_col[2].y;
    {{FType.Name}} g = mat.m_col[0].z;
    {{FType.Name}} h = mat.m_col[1].z;
    {{FType.Name}} i = mat.m_col[2].z;

    {{FType.Name}}3x3 inv;
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

  static inline {{FType.Name}}3x3 scale3x3({{FType.Name}}3 t)
  {
    {{FType.Name}}3x3 res;
    res.set_col(0, {{FType.Name}}3{t.x, 0.0, 0.0});
    res.set_col(1, {{FType.Name}}3{0.0, t.y, 0.0});
    res.set_col(2, {{FType.Name}}3{0.0, 0.0,  t.z});
    return res;
  }

  static inline {{FType.Name}}3x3 rotate3x3X({{FType.Name}} phi)
  {
    {{FType.Name}}3x3 res;
    res.set_col(0, {{FType.Name}}3{1.0,      0.0,       0.0});
    res.set_col(1, {{FType.Name}}3{0.0, +std::cos(phi),  +std::sin(phi)});
    res.set_col(2, {{FType.Name}}3{0.0, -std::sin(phi),  +std::cos(phi)});
    return res;
  }

  static inline {{FType.Name}}3x3 rotate3x3Y({{FType.Name}} phi)
  {
    {{FType.Name}}3x3 res;
    res.set_col(0, {{FType.Name}}3{+std::cos(phi), 0.0, -std::sin(phi)});
    res.set_col(1, {{FType.Name}}3{     0.0, 1.0,      0.0});
    res.set_col(2, {{FType.Name}}3{+std::sin(phi), 0.0, +std::cos(phi)});
    return res;
  }

  static inline {{FType.Name}}3x3 rotate3x3Z({{FType.Name}} phi)
  {
    {{FType.Name}}3x3 res;
    res.set_col(0, {{FType.Name}}3{+std::cos(phi), std::sin(phi), 0.0});
    res.set_col(1, {{FType.Name}}3{-std::sin(phi), std::cos(phi), 0.0});
    res.set_col(2, {{FType.Name}}3{     0.0,     0.0, 1.0});
    return res;
  }
  
  static inline {{FType.Name}}3x3 mul({{FType.Name}}3x3 m1, {{FType.Name}}3x3 m2)
  {
    const {{FType.Name}}3 column1 = mul(m1, m2.col(0));
    const {{FType.Name}}3 column2 = mul(m1, m2.col(1));
    const {{FType.Name}}3 column3 = mul(m1, m2.col(2));
    {{FType.Name}}3x3 res;
    res.set_col(0, column1);
    res.set_col(1, column2);
    res.set_col(2, column3);

    return res;
  }

  static inline {{FType.Name}}3x3 operator*({{FType.Name}}3x3 m1, {{FType.Name}}3x3 m2)
  {
    const {{FType.Name}}3 column1 = mul(m1, m2.col(0));
    const {{FType.Name}}3 column2 = mul(m1, m2.col(1));
    const {{FType.Name}}3 column3 = mul(m1, m2.col(2));

    {{FType.Name}}3x3 res;
    res.set_col(0, column1);
    res.set_col(1, column2);
    res.set_col(2, column3);
    return res;
  }

  static inline {{FType.Name}}3x3 operator*({{FType.Name}} scale, {{FType.Name}}3x3 m)
  {
    {{FType.Name}}3x3 res;
    res.m_col[0] = m.m_col[0] * scale;
    res.m_col[1] = m.m_col[1] * scale;
    res.m_col[2] = m.m_col[2] * scale;
    return res;
  }

  static inline {{FType.Name}}3x3 operator*({{FType.Name}}3x3 m, {{FType.Name}} scale)
  {
    {{FType.Name}}3x3 res;
    res.m_col[0] = m.m_col[0] * scale;
    res.m_col[1] = m.m_col[1] * scale;
    res.m_col[2] = m.m_col[2] * scale;
    return res;
  }

  static inline {{FType.Name}}3x3 operator+({{FType.Name}}3x3 m1, {{FType.Name}}3x3 m2)
  {
    {{FType.Name}}3x3 res;
    res.m_col[0] = m1.m_col[0] + m2.m_col[0];
    res.m_col[1] = m1.m_col[1] + m2.m_col[1];
    res.m_col[2] = m1.m_col[2] + m2.m_col[2];
    return res;
  }

  static inline {{FType.Name}}3x3 operator-({{FType.Name}}3x3 m1, {{FType.Name}}3x3 m2)
  {
    {{FType.Name}}3x3 res;
    res.m_col[0] = m1.m_col[0] - m2.m_col[0];
    res.m_col[1] = m1.m_col[1] - m2.m_col[1];
    res.m_col[2] = m1.m_col[2] - m2.m_col[2];
    return res;
  }

  static inline {{FType.Name}}3x3 outerProduct({{FType.Name}}3 a, {{FType.Name}}3 b) 
  {
    {{FType.Name}}3x3 m;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        m[i][j] = a[i] * b[j];
    return m;
  }

  static inline {{FType.Name}}3 mul3x3({{FType.Name}}4x4 m, {{FType.Name}}3 v) { return to_{{FType.Name}}3(m*to_{{FType.Name}}4(v, 0.0f)); }
  static inline {{FType.Name}}3 mul4x3({{FType.Name}}4x4 m, {{FType.Name}}3 v) { return to_{{FType.Name}}3(m*to_{{FType.Name}}4(v, 1.0f)); }
  
  ///////////////////////////////////////////////////////////////////
  //////////////////// complex {{FType.Name}} ///////////////////////
  ///////////////////////////////////////////////////////////////////

  // complex numbers adapted from PBRT-v4
  struct complex{{FType.Suffix}} 
  {
    inline complex{{FType.Suffix}}() : re(0), im(0) {}
    inline complex{{FType.Suffix}}({{FType.Name}} re_) : re(re_), im(0) {}
    inline complex{{FType.Suffix}}({{FType.Name}} re_, {{FType.Name}} im_) : re(re_), im(im_) {}

    inline complex{{FType.Suffix}} operator-()          const { return {-re, -im}; }
    inline complex{{FType.Suffix}} operator+(complex{{FType.Suffix}} z) const { return {re + z.re, im + z.im}; }
    inline complex{{FType.Suffix}} operator-(complex{{FType.Suffix}} z) const { return {re - z.re, im - z.im}; }
    inline complex{{FType.Suffix}} operator*(complex{{FType.Suffix}} z) const { return {re * z.re - im * z.im, re * z.im + im * z.re}; }

    inline complex{{FType.Suffix}} operator/(complex{{FType.Suffix}} z) const 
    {
      {{FType.Name}} scale = 1 / (z.re * z.re + z.im * z.im);
      return {scale * (re * z.re + im * z.im), scale * (im * z.re - re * z.im)};
    }

    inline friend complex{{FType.Suffix}} operator+({{FType.Name}} value, complex{{FType.Suffix}} z) { return complex{{FType.Suffix}}(value) + z; }
    inline friend complex{{FType.Suffix}} operator-({{FType.Name}} value, complex{{FType.Suffix}} z) { return complex{{FType.Suffix}}(value) - z; }
    inline friend complex{{FType.Suffix}} operator*({{FType.Name}} value, complex{{FType.Suffix}} z) { return complex{{FType.Suffix}}(value) * z; }
    inline friend complex{{FType.Suffix}} operator/({{FType.Name}} value, complex{{FType.Suffix}} z) { return complex{{FType.Suffix}}(value) / z; }

    {{FType.Name}} re, im;
  };

  inline static {{FType.Name}} real(const complex{{FType.Suffix}} &z) { return z.re; }
  inline static {{FType.Name}} imag(const complex{{FType.Suffix}} &z) { return z.im; }

  inline static {{FType.Name}} complex_norm(const complex{{FType.Suffix}} &z) { return z.re * z.re + z.im * z.im; }
  inline static {{FType.Name}} complex_abs (const complex{{FType.Suffix}} &z) { return std::sqrt(complex_norm(z)); }
  inline static complex{{FType.Suffix}} complex_sqrt(const complex{{FType.Suffix}} &z) 
  {
    {{FType.Name}} n = complex_abs(z);
    {{FType.Name}} t1 = std::sqrt(0.5f * (n + std::abs(z.re)));
    {{FType.Name}} t2 = 0.5f * z.im / t1;

    if (n == 0)
      return 0;

    if (z.re >= 0)
      return {t1, t2};
    else
      return {std::abs(t2), std::copysign(t1, z.im)};
  }

  {% endfor %}

  ///////////////////////////////////////////////////////////////////
  ///// Auxilary functions which are not in the core of library /////
  ///////////////////////////////////////////////////////////////////
  
  inline static uint color_pack_rgba(const float4 rel_col)
  {
    static const float4 const_255(255.0f);
    const uint4 rgba = to_uint32(rel_col*const_255);
    return (rgba[3] << 24) | (rgba[2] << 16) | (rgba[1] << 8) | rgba[0];
  }

  inline static uint color_pack_bgra(const float4 rel_col)
  {
    static const float4 const_255(255.0f);
    const uint4 rgba = to_uint32(shuffle_zyxw(rel_col)*const_255);
    return (rgba[3] << 24) | (rgba[2] << 16) | (rgba[1] << 8) | rgba[0];
  }
  
  inline static float4 color_unpack_bgra(int packedColor)
  {
    const int red   = (packedColor & 0x00FF0000) >> 16;
    const int green = (packedColor & 0x0000FF00) >> 8;
    const int blue  = (packedColor & 0x000000FF) >> 0;
    const int alpha = (packedColor & 0xFF000000) >> 24;
    return float4((float)red, (float)green, (float)blue, (float)alpha)*(1.0f / 255.0f);
  }

  inline static float4 color_unpack_rgba(int packedColor)
  {
    const int blue  = (packedColor & 0x00FF0000) >> 16;
    const int green = (packedColor & 0x0000FF00) >> 8;
    const int red   = (packedColor & 0x000000FF) >> 0;
    const int alpha = (packedColor & 0xFF000000) >> 24;
    return float4((float)red, (float)green, (float)blue, (float)alpha)*(1.0f / 255.0f);
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  // Look At matrix creation
  // return the inverse view matrix
  //
  static inline float4x4 lookAt(float3 eye, float3 center, float3 up)
  {
    float3 x, y, z; // basis; will make a rotation matrix
    z.x = eye.x - center.x;
    z.y = eye.y - center.y;
    z.z = eye.z - center.z;
    z = normalize(z);
    y.x = up.x;
    y.y = up.y;
    y.z = up.z;
    x = cross(y, z); // X vector = Y cross Z
    y = cross(z, x); // Recompute Y = Z cross X
    // cross product gives area of parallelogram, which is < 1.0 for
    // non-perpendicular unit-length vectors; so normalize x, y here
    x = normalize(x);
    y = normalize(y);
    float4x4 M;
    M.set_col(0, float4{ x.x, y.x, z.x, 0.0f });
    M.set_col(1, float4{ x.y, y.y, z.y, 0.0f });
    M.set_col(2, float4{ x.z, y.z, z.z, 0.0f });
    M.set_col(3, float4{ -x.x * eye.x - x.y * eye.y - x.z*eye.z,
                         -y.x * eye.x - y.y * eye.y - y.z*eye.z,
                         -z.x * eye.x - z.y * eye.y - z.z*eye.z,
                         1.0f });
    return M;
  }

  static inline float4x4 perspectiveMatrix(float fovy, float aspect, float zNear, float zFar)
  {
    const float ymax = zNear * tanf(fovy * 3.14159265358979323846f / 360.0f);
    const float xmax = ymax * aspect;
    const float left = -xmax;
    const float right = +xmax;
    const float bottom = -ymax;
    const float top = +ymax;
    const float temp = 2.0f * zNear;
    const float temp2 = right - left;
    const float temp3 = top - bottom;
    const float temp4 = zFar - zNear;
    float4x4 res;
    res.m_col[0] = float4{ temp / temp2, 0.0f, 0.0f, 0.0f };
    res.m_col[1] = float4{ 0.0f, temp / temp3, 0.0f, 0.0f };
    res.m_col[2] = float4{ (right + left) / temp2,  (top + bottom) / temp3, (-zFar - zNear) / temp4, -1.0 };
    res.m_col[3] = float4{ 0.0f, 0.0f, (-temp * zFar) / temp4, 0.0f };
    return res;
  }

  static inline float4x4 ortoMatrix(const float l, const float r, const float b, const float t, const float n, const float f)
  {
    float4x4 res;
    res(0,0) = 2.0f / (r - l);
    res(0,1) = 0;
    res(0,2) = 0;
    res(0,3) = -(r + l) / (r - l);
    res(1,0) = 0;
    res(1,1) = 2.0f / (t - b);
    res(1,2) = 0;
    res(1,3) = -(t + b) / (t - b);
    res(2,0) = 0;
    res(2,1) = 0;
    res(2,2) = -2.0f / (f - n);
    res(2,3) = -(f + n) / (f - n);
    res(3,0) = 0.0f;
    res(3,1) = 0.0f;
    res(3,2) = 0.0f;
    res(3,3) = 1.0f;
    return res;
  }

  // http://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
  //
  static inline float4x4 OpenglToVulkanProjectionMatrixFix()
  {
    float4x4 res;
    res(1,1) = -1.0f;
    res(2,2) = 0.5f;
    res(2,3) = 0.5f;
    return res;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  static inline float4 packFloatW(const float4& a, float data) { return blend(a, float4(data),            uint4{0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0}); }
  static inline float4 packIntW(const float4& a, int data)     { return blend(a, as_float32(int4(data)),  uint4{0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0}); }
  static inline float4 packUIntW(const float4& a, uint data)   { return blend(a, as_float32(uint4(data)), uint4{0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0}); }
  static inline int    extractIntW(const float4& a)            { return as_int(a.w);  }
  static inline uint   extractUIntW(const float4& a)           { return as_uint(a.w); }

  /////////////////////////////////////////
  /////////////// Boxes stuff /////////////
  /////////////////////////////////////////
  
  struct Box4f 
  { 
    inline Box4f()
    {
      boxMin = float4( +std::numeric_limits<float>::infinity() );
      boxMax = float4( -std::numeric_limits<float>::infinity() );   
    }
  
    inline Box4f(const float4 a_bMin, const float4 a_bMax)
    {
      boxMin = a_bMin;
      boxMax = a_bMax;   
    }
  
    inline void include(const LiteMath::float4 p) // please note that this function may override Start/Count pair, so use it carefully
    {                                           
      boxMin = LiteMath::min(boxMin, p);
      boxMax = LiteMath::max(boxMax, p);
    }
  
    inline void include(const Box4f& b) // please note that this function may override Start/Count pair, so use it carefully
    {                                     
      boxMin = LiteMath::min(boxMin, b.boxMin);
      boxMax = LiteMath::max(boxMax, b.boxMax);
    } 
  
    inline void intersect(const Box4f& a_box) 
    {
      boxMin = LiteMath::max(boxMin, a_box.boxMin);
      boxMax = LiteMath::min(boxMax, a_box.boxMax);
    }
  
    inline float surfaceArea() const
    {
      const float4 abc = boxMax - boxMin;
      return 2.0f*(abc[0]*abc[1] + abc[0]*abc[2] + abc[1]*abc[2]);
    }
  
    inline float volume() const 
    {
      const float4 abc = boxMax - boxMin;
      return abc[0]*abc[1]*abc[2];       // #TODO: hmul3
    }
  
    inline void setStart(uint i) { boxMin = packUIntW(boxMin, uint(i)); }
    inline void setCount(uint i) { boxMax = packUIntW(boxMax, uint(i)); }
    inline uint getStart() const { return extractUIntW(boxMin); }
    inline uint getCount() const { return extractUIntW(boxMax); }
    inline bool isAxisAligned(int axis, float split) const { return (boxMin[axis] == boxMax[axis]) && (boxMin[axis]==split); }
  
    float4 boxMin; // as_int(boxMin4f.w) may store index of the object inside the box (or start index of the object sequence)
    float4 boxMax; // as_int(boxMax4f.w) may store size (count) of objects inside the box
  };
  
  struct Ray4f 
  {
    inline Ray4f(){}
    inline Ray4f(const float4& pos, const float4& dir) : posAndNear(pos), dirAndFar(dir) { }
    inline Ray4f(const float4& pos, const float4& dir, float tNear, float tFar) : posAndNear(pos), dirAndFar(dir) 
    { 
      posAndNear = packFloatW(posAndNear, tNear);
      dirAndFar  = packFloatW(dirAndFar,  tFar);
    }
  
    inline Ray4f(const float3& pos, const float3& dir, float tNear, float tFar) : posAndNear(to_float4(pos,tNear)), dirAndFar(to_float4(dir, tFar)) { }
  
    inline float getNear() const { return extract_3(posAndNear); }
    inline float getFar()  const { return extract_3(dirAndFar); }

    inline void  setNear(float tNear) { posAndNear = packFloatW(posAndNear, tNear); } 
    inline void  setFar (float tFar)  { dirAndFar  = packFloatW(dirAndFar,  tFar); } 
  
    float4 posAndNear;
    float4 dirAndFar;
  };
  
  /////////////////////////////////////////
  /////////////// rays stuff //////////////
  /////////////////////////////////////////
  
  /**
  \brief Computes near and far intersection of ray and box
  \param  rayPos     - input ray origin
  \param  rayDirInv  - input inverse ray dir (1.0f/rayDirection)
  \param  boxMin     - input box min
  \param  boxMax     - input box max
  \return (tnear, tfar); if tnear > tfar, no interection is found. 
  */
  static inline float2 Ray4fBox4fIntersection(float4 rayPos, float4 rayDirInv, float4 boxMin, float4 boxMax)
  {
    const float4 lo   = rayDirInv*(boxMin - rayPos);
    const float4 hi   = rayDirInv*(boxMax - rayPos);
    const float4 vmin = min(lo, hi);
    const float4 vmax = max(lo, hi);
    return float2(hmax3(vmin), hmin3(vmax));
  }
    
  /**
  \brief Create eye ray for target x,y and Proj matrix
  \param  x - input x coordinate of pixel
  \param  y - input y coordinate of pixel
  \param  w - input framebuffer image width  
  \param  h - input framebuffer image height
  \param  a_mProjInv - input inverse projection matrix
  \return Eye ray direction; the fourth component will contain +INF as tfar according to Ray4f tnear/tfar storage agreement 
  */
  static inline float4 EyeRayDir4f(float x, float y, float w, float h, float4x4 a_mProjInv) // g_mViewProjInv
  {
    float4 pos = float4( 2.0f * (x + 0.5f) / w - 1.0f, 
                        -2.0f * (y + 0.5f) / h + 1.0f, 
                         0.0f, 
                         1.0f );
  
    pos = a_mProjInv*pos;
    pos = pos/pos.w;
    pos.y *= (-1.0f);      // TODO: do we need remove this (???)
    pos = normalize3(pos);
    pos.w = INF_POSITIVE;
    return pos;
  }
  
  /**
  \brief  calculate overlapping area (volume) of 2 bounding boxes and return it if form of bounding box
  \param  box1 - input first box 
  \param  box2 - input second box
  \return overlaped volume bounding box. If no intersection found, return zero sized bounding box 
  */
  inline Box4f BoxBoxOverlap(const Box4f& box1, const Box4f& box2)
  {
    Box4f tempBox1 = box1;
    Box4f tempBox2 = box2;
    float4 res_min;
    float4 res_max;
    
    for(int axis = 0; axis < 3; ++axis){ // #TODO: unroll loop and vectorize code
      // sort boxes by min
      if(tempBox2.boxMin[axis] < tempBox1.boxMin[axis]){
        float tempSwap = tempBox1.boxMin[axis];
        tempBox1.boxMin[axis] = tempBox2.boxMin[axis];
        tempBox2.boxMin[axis] = tempSwap;
  
        tempSwap = tempBox1.boxMax[axis];
        tempBox1.boxMax[axis] = tempBox2.boxMax[axis];
        tempBox2.boxMax[axis] = tempSwap;
      }
      // check the intersection
      if(tempBox1.boxMax[axis] < tempBox2.boxMin[axis])
        return Box4f(box1.boxMax, box1.boxMax);
  
      // Intersected box
      res_min[axis] = tempBox2.boxMin[axis];
      res_max[axis] = std::min(tempBox1.boxMax[axis], tempBox2.boxMax[axis]);
    }
    return Box4f(res_min, res_max);
  }
  
   /// 3D bounding box, parallel to the coordinate planes
  struct BBox3f
  {
    /// Result of ray intersection test
    struct HitTestRes
      {
      /// Natural coordinates (along the ray, 0 at the origin) of ray segment
      float t1, t2;
      /// The wall the ray enters the box (should the origin be at -$\infty$),
      /// 0,1 or 2 if its normal is along X,Y or Z, respectively
      int face;
      };

    /// Intersection of the ray with the box. Inputs ray origin and inverse direction, outputs natural coordinate for hits and the index of side hit
    inline HitTestRes Intersection(const float3& origin, const float3& inv_dir, float min_t, float max_t) const;

    /// Diagonal points
    float3 boxMin;
    float3 boxMax;
  };

  /// Intersection of the ray with the box. 
  /// @param[in] origin  - ray origin
  /// @param[in] inv_dir - inverse ray direction (=1/dir)
  /// @param[in] min_t   - low bound of allowed range of the natural coordinate 
  ///                      (along the ray, 0 at the origin)
  /// @param[in] max_t   - upper bound of allowed range of the natural coordinate 
  ///                      (along the ray, 0 at the origin)
  /// natural coordinate t.
  /// @return Triplet (t1,t2,i) where [t1,t2] are the natural coordinates of ray 
  ///         segment inside the box, clipped by allowed range, and 'i' is the index
  ///         of the "entry wall": 0,1,2 if its normal is along X,Y,Z or -1 if the
  ///         ray did NOT hit the box.
  BBox3f::HitTestRes BBox3f::Intersection(const float3& origin, const float3& inv_dir, float min_t, float max_t) const
  {
    float tmin, tmax;

    const float min_x = inv_dir[0] < 0 ? boxMax[0] : boxMin[0];
    const float min_y = inv_dir[1] < 0 ? boxMax[1] : boxMin[1];
    const float min_z = inv_dir[2] < 0 ? boxMax[2] : boxMin[2];
    const float max_x = inv_dir[0] < 0 ? boxMin[0] : boxMax[0];
    const float max_y = inv_dir[1] < 0 ? boxMin[1] : boxMax[1];
    const float max_z = inv_dir[2] < 0 ? boxMin[2] : boxMax[2];

    // X
    const float tmin_x = (min_x - origin[0]) * inv_dir[0];
    // MaxMult robust BVH traversal(up to 4 ulp).
    // 1.0000000000000004 for double precision.
    const float tmax_x = (max_x - origin[0]) * inv_dir[0] * 1.00000024f;

    // Y
    const float tmin_y = (min_y - origin[1]) * inv_dir[1];
    const float tmax_y = (max_y - origin[1]) * inv_dir[1] * 1.00000024f;

    // Z
    const float tmin_z = (min_z - origin[2]) * inv_dir[2];
    const float tmax_z = (max_z - origin[2]) * inv_dir[2] * 1.00000024f;

    tmax = std::min(tmax_z, std::min(tmax_y, std::min(tmax_x, max_t)));


    //  tmin = std::max(tmin_z, std::max(tmin_y, std::max(tmin_x, min_t)));
    int idx = -1;
    if (tmin_x > tmin_y)
      {
      if (tmin_z > tmin_x)
        {
        idx = 2;
        tmin = std::max(tmin_z, min_t);
        }
      else
        {
        idx = 0;
        tmin = std::max(tmin_x, min_t);
        }
      }
    else
      {
      if (tmin_z > tmin_y)
        {
        idx = 2;
        tmin = std::max(tmin_z, min_t);
        }
      else
        {
        idx = 1;
        tmin = std::max(tmin_y, min_t);
        }
      }

    HitTestRes res;
    res.t1 = tmin;
    res.t2 = tmax;
    res.face = idx;
    return res;
  }
  
};

#ifdef HALFFLOAT
#include "half.hpp"

namespace LiteMath
{ 
  using half_float::half;

  struct half4
  {
    inline half4() : x(0), y(0), z(0), w(0) {}
    inline half4(float a_x, float a_y, float a_z, float a_w) : x(a_x), y(a_y), z(a_z), w(a_w) {}

    inline explicit half4(float a_val) : x(a_val), y(a_val), z(a_val), w(a_val) {}
    inline explicit half4(const float a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}
    inline explicit half4(float4 f4)   : x(f4.x), y(f4.y), z(f4.z), w(f4.w) {}
    
    inline half4(half a_x, half a_y, half a_z, half a_w) : x(a_x), y(a_y), z(a_z), w(a_w) {}
    inline explicit half4(half a_val) : x(a_val), y(a_val), z(a_val), w(a_val) {}
    inline explicit half4(const half a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

    inline half& operator[](int i)  
    {
      switch(i)
      {
        case 0:  return x;
        case 1:  return y;
        case 2:  return z;
        default: return w;
      };
    }

    inline half  operator[](int i) const 
    {
      switch(i)
      {
        case 0:  return x;
        case 1:  return y;
        case 2:  return z;
        default: return w;
      };
    }

    half x, y, z, w;
  };

  struct half2
  {
    inline half2() : x(0), y(0) {}
    inline half2(float a_x, float a_y) : x(a_x), y(a_y) {}

    inline explicit half2(float a_val) : x(a_val), y(a_val) {}
    inline explicit half2(const float a[4]) : x(a[0]), y(a[1]) {}
    inline explicit half2(float4 f4)   : x(f4.x), y(f4.y) {}
    
    inline half2(half a_x, half a_y) : x(a_x), y(a_y) {}
    inline explicit half2(half a_val) : x(a_val), y(a_val) {}
    inline explicit half2(const half a[4]) : x(a[0]), y(a[1]) {}

    inline half& operator[](int i)  
    {
      switch(i)
      {
        case 0:  return x;
        default: return y;
      };
    }

    inline half  operator[](int i) const 
    {
      switch(i)
      {
        case 0:  return x;
        default: return y;
      };
    }

    half x, y;
  };

  static inline float2 to_float2(half2 v) { return float2(v.x, v.y); }
  static inline float4 to_float4(half4 v) { return float4(v.x, v.y, v.z, v.w); }
};
#endif

#include <omp.h>
//#ifndef _OPENMP
//static int omp_get_num_threads() { return 1; }
//static int omp_get_max_threads() { return 1; }
//static int omp_get_thread_num()  { return 0; }
//#endif

namespace LiteMath
{ 
  
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
    const size_t maxThreads = size_t(omp_get_num_threads()); // here not max_threads
    const size_t szAligned  = a_vec.capacity() / maxThreads; // todo replace div by shift if number of threads is predefined
    const size_t threadId   = size_t(omp_get_thread_num());
    a_vec.data()[szAligned*threadId + size_t(offset)] += val;
  }

  template<typename T, typename IndexType> 
  inline void ReduceAdd(std::vector<T>& a_vec, IndexType offset, IndexType a_sizeAligned, T val) // more optimized version
  {
    if(!std::isfinite(val))
      return;
    a_vec.data()[a_sizeAligned*omp_get_thread_num() + offset] += val;
  }
};

#endif
#endif
#endif
