#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include "LiteMath.h"
using namespace LiteMath;

bool test000_scalar_funcs()
{
  volatile float x = 3.75f;
  volatile float z = -1.25f;

  float y01 = mod(x,0.5); // mod returns the value of x modulo y. This is computed as x - y * floor(x/y). 
  float y02 = mod(z,0.5); // mod returns the value of x modulo y. This is computed as x - y * floor(x/y). 
  
  float y03 = fract(x);   // return only the fraction part of a number; This is calculated as x - floor(x). 
  float y04 = fract(z);   // return only the fraction part of a number; This is calculated as x - floor(x).
  
  float y05 = ceil(x);          // nearest integer that is greater than or equal to x
  float y06 = ceil(z);          // nearest integer that is greater than or equal to x

  float y07 = floor(x);         // nearest integer less than or equal to x
  float y08 = floor(z);         // nearest integer less than or equal to x

  float y09 = sign(x);          // sign returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x is greater than 0.0. 
  float y10 = sign(z);          // sign returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x is greater than 0.0. 
  float y11 = sign(0.0f); 

  float y12 = abs(x);           // return the absolute value of x
  float y13 = abs(z);           // return the absolute value of x

  float y14 = clamp(x,0.0f,1.0f); // constrain x to lie between 0.0 and 1.0
  float y15 = clamp(z,0.0f,1.0f); // constrain x to lie between 0.0 and 1.0
  
  float y16 = min(x,z);    // return the lesser of x and z
  float y17 = max(x,z);    // return the greater of x and z 
  
  float y18 = mix (2.0f, 4.0f, 0.5f); // linear interpolate, same as lerp
  float y19 = lerp(2.0f, 4.0f, 0.5f); // linear interpolate
  
  float y20 = smoothstep(2.0f, 4.0f, 3.25f);
  float y21 = smoothstep(2.0f, 4.0f, 3.5f);
  float y22 = smoothstep(2.0f, 4.0f, 3.75f);

  float y23 = sqrt(x);

  float y26 = inversesqrt(x);   // 1.0f / sqrt(x)
  float y27 = rcp(x);           // fast reciprocal

  int32_t  y28  = as_int(x);
  int32_t  y29  = as_int32(x);
  uint32_t y30  = as_uint(x);
  uint32_t y31  = as_uint32(x);
  float    y32  = as_float(y28); 
  float    y33  = as_float(y29); 
  float    y34  = as_float32(y30); 
  float    y35  = as_float32(y31); 
  
  float randoms[100];
  for(int i=0;i<100;i++)
    randoms[i] = rnd(2.0f, 4.0f);

  // check 
  //
  bool passed = true;
  passed = passed && (y01 == y02)  && (y01 == 0.25f);
  passed = passed && (y03 == y04)  && (y03 == 0.75f);
  passed = passed && (y05 == 4.0f) && (y06 == -1.0f);
  passed = passed && (y07 == 3.0f) && (y08 == -2.0f);
  passed = passed && (y09 == 1.0f) && (y10 == -1.0f) && (y11 == 0.0f);
  passed = passed && (y12 == 3.75f) && (y13 == 1.25f);
  passed = passed && (y14 == 1.0f)  && (y15 == 0.0f);
  passed = passed && (y16 == -1.25f) && (y17 == 3.75f);
  passed = passed && (y18 == 3.0f) && (y19 == 3.0f);
  passed = passed && (y20 > 0.0f) && (y20 < 1.0f) && (y21 > 0.0f) && (y21 < 1.0f) && (y22 > 0.0f) && (y22 < 1.0f);
  passed = passed && abs(y23*y23 - x) < 1e-6f && abs((1.0f/y26)*(1.0f/y26) - x) < 1e-5f;
  passed = passed && fabs(1.0f/y27 - x) < 1e-4f; 
  passed = passed && (y32 == x) && (y33 == x) && (y34 == x) && (y35 == x);
  
  for(int i=0;i<100;i++)
    passed = passed && (randoms[i] >= 2.0f && randoms[i] <= 4.0f);

  for(int i=0;i<100;i++)
  {
    for(int j=i+1;j<100;j++)
    {
      if(i!=j)
        passed = passed && (randoms[i] != randoms[j]);
    }
  }

  return passed;
}

bool test001_dot_cross_f4()
{
  const float4 Cx1 = { 1.0f, 2.0f, 3.0f, 4.0f };
  const float4 Cx2 = { 5.0f, 6.0f, 7.0f, 8.0f };

  const float  dot1 = dot3f(Cx1, Cx2);
  const float4 dot2 = dot3v(Cx1, Cx2);
  const float  dot3 = dot4f(Cx1, Cx2);
  const float4 dot4 = dot4v(Cx1, Cx2);
  const float  dot5 = dot  (Cx1, Cx2);
  const float4 crs3 = cross(Cx1, Cx2);

  CVEX_ALIGNED(16) float result1[4];
  CVEX_ALIGNED(16) float result2[4];
  CVEX_ALIGNED(16) float result3[4];
  store(result1, dot2);
  store(result2, dot4);
  store(result3, crs3);

  const float ref_dp3 = 1.0f*5.0f + 2.0f*6.0f + 3.0f*7.0f;
  const float ref_dp4 = 1.0f*5.0f + 2.0f*6.0f + 3.0f*7.0f + 4.0f*8.0f;

  const float crs_ref[3] = { Cx1[1]*Cx2[2] - Cx1[2]*Cx2[1], 
                             Cx1[2]*Cx2[0] - Cx1[0]*Cx2[2], 
                             Cx1[0]*Cx2[1] - Cx1[1]*Cx2[0] };

  const bool b1 = fabs(dot1 - ref_dp3) < 1e-6f;
  const bool b2 = fabs(result1[0] - ref_dp3) < 1e-6f && 
                  fabs(result1[1] - ref_dp3) < 1e-6f && 
                  fabs(result1[2] - ref_dp3) < 1e-6f &&
                  fabs(result1[3] - ref_dp3) < 1e-6f;

  const bool b3 = fabs(dot3 - ref_dp4) < 1e-6f;
  const bool b4 = fabs(result2[0] - ref_dp4) < 1e-6f &&
                  fabs(result2[1] - ref_dp4) < 1e-6f &&
                  fabs(result2[2] - ref_dp4) < 1e-6f &&
                  fabs(result2[3] - ref_dp4) < 1e-6f;

  const bool b5 = fabs(result3[0] - crs_ref[0]) < 1e-6f && 
                  fabs(result3[1] - crs_ref[1]) < 1e-6f &&
                  fabs(result3[2] - crs_ref[2]) < 1e-6f;

  const bool b6 = fabs(dot5 - dot3) < 1e-10f;

  return b1 && b2 && b3 && b4 && b5 && b6;
}

bool test002_dot_cross_f3()
{
  const float3 Cx1(1.0f, 2.0f, 3.0f);
  const float3 Cx2(5.0f, 6.0f, 7.0f);

  const float   dot1 = dot(Cx1, Cx2);
  const float3  crs3 = cross(Cx1, Cx2);

  CVEX_ALIGNED(16) float result3[4];
  store(result3, crs3);

  const float ref_dp3 = 1.0f*5.0f + 2.0f*6.0f + 3.0f*7.0f;
  const float crs_ref[3] = { Cx1[1]*Cx2[2] - Cx1[2]*Cx2[1], 
                             Cx1[2]*Cx2[0] - Cx1[0]*Cx2[2], 
                             Cx1[0]*Cx2[1] - Cx1[1]*Cx2[0] };

  const bool b1 = fabs(dot1 - ref_dp3) < 1e-6f;
  const bool b5 = fabs(result3[0] - crs_ref[0]) < 1e-6f && 
                  fabs(result3[1] - crs_ref[1]) < 1e-6f &&
                  fabs(result3[2] - crs_ref[2]) < 1e-6f;

  return b1 && b5;
}

bool test003_length_float4()
{
  const float4 Cx1 = { 1.0f, 2.0f, 3.0f, 4.0f };

  const float   dot1 = length3f(Cx1);
  const float4 dot2  = length3v(Cx1);
  const float   dot3 = length4f(Cx1);
  const float4 dot4  = length4v(Cx1);

  CVEX_ALIGNED(16) float result1[4];
  CVEX_ALIGNED(16) float result2[4];
  store(result1, dot2);
  store(result2, dot4);

  const float ref_dp3 = sqrtf(1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f);
  const float ref_dp4 = sqrtf(1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f + 4.0f*4.0f);

  const bool b1 = fabs(dot1 - ref_dp3) < 1e-6f;
  const bool b2 = fabs(result1[0] - ref_dp3) < 1e-6f && 
                  fabs(result1[1] - ref_dp3) < 1e-6f && 
                  fabs(result1[2] - ref_dp3) < 1e-6f &&
                  fabs(result1[3] - ref_dp3) < 1e-6f;

  const bool b3 = fabs(dot3 - ref_dp4) < 1e-6f;
  const bool b4 = fabs(result2[0] - ref_dp4) < 1e-6f &&
                  fabs(result2[1] - ref_dp4) < 1e-6f &&
                  fabs(result2[2] - ref_dp4) < 1e-6f &&
                  fabs(result2[3] - ref_dp4) < 1e-6f;
                  
  return (b1 && b2 && b3 && b4);
}

bool test004_colpack_f4x4()
{
  const float4 Cx1 = { 0.25f, 0.5f, 0.0, 1.0f };

  const unsigned int packed_rgba = color_pack_rgba(Cx1);
  const unsigned int packed_bgra = color_pack_bgra(Cx1);

  const bool passed = ((packed_bgra == 0xFF408000) || (packed_bgra == 0xFF3f7f00)) && 
                      ((packed_rgba == 0xFF008040) || (packed_rgba == 0xFF007f3f));

  if(!passed)
  {
    std::cout << std::hex << "bgra_res: " << packed_bgra << std::endl;
    std::cout << std::hex << "bgra_ref: " << 0xFF408000  << std::endl;

    std::cout << std::hex << "rgba_res: " << packed_rgba << std::endl;
    std::cout << std::hex << "rgba_ref: " << 0xFF008040  << std::endl;
  }

  return passed;
}

bool test005_matrix_elems()
{
  float4x4 m;
  m(1,2) = 3.0f;
  m(3,1) = 4.0f; 
  return (m[1][2] == 3.0f) && (m[3][1] == 4.0f) && (m[0][0] == 1.0f) && (m[1][1] == 1.0f) && (m[0][3] == 0.0f);
}

bool test006_any_all()
{
  const float4 Cx1 = { 1.0f, 2.0f, 7.0f, 2.0f };
  const float4 Cx2 = { 5.0f, 6.0f, 7.0f, 4.0f };
  const float4 Cx3 = { 5.0f, 6.0f, 8.0f, -4.0f };

  const auto cmp1 = (Cx1 > Cx2);
  const auto cmp2 = (Cx1 <= Cx3);

  const bool b1 = any_of(cmp1);
  const bool b2 = all_of(cmp1);

  const bool b3 = any_of(cmp2);
  const bool b4 = all_of(cmp2);

  return (!b1 && !b2 && b3 && !b4);
}

bool test007_reflect()
{
  const float4 dir4(1.0f, -1.0f, 0.0f, 0.0f);
  const float3 dir3(1.0f, -1.0f, 0.0f);
  const float2 dir2(1.0f, -1.0f);

  const float4 n4(0.0f, 1.0f, 0.0f, 0.0f);
  const float3 n3(0.0f, 1.0f, 0.0f);
  const float2 n2(0.0f, 1.0f);

  auto  r4 = reflect(dir4, n4);
  auto  r3 = reflect(dir3, n3);
  auto  r2 = reflect(dir2, n2);

  const float cos41 = dot3f(dir4, n4);
  const float cos42 = dot3f(r4, n4);

  const float cos31 = dot(dir3, n3);
  const float cos32 = dot(r3, n3);

  const float cos21 = dot(dir2, n2);
  const float cos22 = dot(r2, n2);

  return abs(cos41+cos42) < 1e-6f && abs(cos31+cos32) < 1e-6f && abs(cos21+cos22) < 1e-6f;
}

bool test008_normalize()
{
  const float4 dir4 = { 1.0f, 2.0f, 3.0f, 4.0f };
  const float3 dir3 = { 1.0f, 2.0f, 3.0f};
  const float2 dir2 = { 1.0f, 2.0f };
  
  const float4 n5  = normalize(dir4);
  const float4 n4  = normalize3(dir4);
  const float3 n3  = normalize(dir3);
  const float2 n2  = normalize(dir2);

  bool ok5 = length3(n5-n4) > 0.15f;
  bool ok4 = abs(length3f(n4) - 1.0f) < 1e-6f && abs( dot3f(n4, dir4/length3f(dir4)) - 1.0f) < 1e-6f;
  bool ok3 = abs(length(n3)   - 1.0f) < 1e-6f && abs( dot  (n3, dir3/length(dir3))   - 1.0f) < 1e-6f;
  bool ok2 = abs(length(n2)   - 1.0f) < 1e-6f && abs( dot  (n2, dir2/length(dir2))   - 1.0f) < 1e-6f;

  return ok5 && ok4 && ok3 && ok2;
}

bool test009_refract()
{
  float4 dir4 = { 10.0f, -1.0f, 0.0f, 0.0f };
  float3 dir3 = { 10.0f, -1.0f, 0.0f};
  float2 dir2 = { 10.0f, -1.0f };

  dir4 = normalize(dir4);
  dir3 = normalize(dir3);
  dir2 = normalize(dir2);

  const float4 n4  = { 0.0f, 1.0f, 0.0f, 0.0f };
  const float3 n3  = { 0.0f, 1.0f, 0.0f };
  const float2 n2  = { 0.0f, 1.0f };
  
  auto  r41 = refract(dir4, n4, 1.0f);
  auto  r31 = refract(dir3, n3, 1.0f);
  auto  r21 = refract(dir2, n2, 1.0f);
  
  auto  r42 = refract(dir4, n4, 10.0f);
  auto  r32 = refract(dir3, n3, 10.0f);
  auto  r22 = refract(dir2, n2, 10.0f);

  bool ok4 = length3f(r41-dir4) < 1e-6f && length3f(r42) == 0.0f;
  bool ok3 = length  (r31-dir3) < 1e-6f && length  (r32) == 0.0f;
  bool ok2 = length  (r21-dir2) < 1e-6f && length  (r22) == 0.0f;

  return ok4 && ok3 && ok2;
}

bool test010_faceforward()
{
  const float4 dir4 = { 1.0f, -1.0f, 0.0f, 0.0f };
  const float3 dir3(1.0f, -1.0f, 0.0f);
  const float2 dir2 = { 1.0f, -1.0f };

  const float4 n4  = { 0.0f, -1.0f, 0.0f, 0.0f };
  const float3 n3  = { 0.0f, -1.0f, 0.0f };
  const float2 n2  = { 0.0f, -1.0f };

  auto  r41 = faceforward(n4, dir4, n4);
  auto  r31 = faceforward(n3, dir3, n3);
  auto  r21 = faceforward(n2, dir2, n2);
  
  bool ok4 = abs(dot3f(r41, n4) + 1.0f) < 1e-6f;
  bool ok3 = abs(dot  (r31, n3) + 1.0f) < 1e-6f;
  bool ok2 = abs(dot  (r21, n2) + 1.0f) < 1e-6f;

  return ok4 && ok3 && ok2;
}

bool test011_mattranspose()
{
  float4x4 m(1.0f,2.0f,3.0f,4.0f,
             5.0f,6.0f,7.0f,8.0f,
             9.0f,10.0f,11.0f,12.0f,
             13.0f,14.0f,15.0f,16.0f);

  float4x4 m2 = transpose(m);

  float4x4 mrotX = rotate4x4X(M_PI*0.5f);
  float4x4 mrotY = rotate4x4Y(M_PI*0.5f);
  float4x4 mrotZ = rotate4x4Z(M_PI*0.5f);

  float4x4 mrotI = inverse4x4(mrotX);
  float4x4 check = mrotI*mrotX;
  float4x4 identity;
  
  float3 p1(0,1,0);
  float3 p2 = to_float3( mrotX*to_float4(p1,1.0f));
  float3 p3 = to_float3( mrotZ*to_float4(p1,1.0f));
  float3 p4 = to_float3( mul(mrotY, mrotX*to_float4(p1,1.0f)));
 
  double error = 0.0;
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      error += std::abs( m(i,j) - m2(j,i));
  
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      error += std::abs( check[i][j] - identity[i][j]);
  
  error += length(p2 - float3(0,0,1));
  error += length(p3 - float3(-1,0,0));
  error += length(p4 - float3(1,0,0));

  return error < 1e-6f;
}

bool test012_mat_double3x3()
{
  double3x3 m(1.0, 2.0, 3.0,
              3.0, 2.0, 1.0,
              2.0, 1.0, 3.0);

  double3x3 m2 = transpose(m);

  double3x3 m3 = inverse3x3(m);
  double3x3 check_inv(-0.41666667, 0.25000000, 0.33333333,
                       0.58333333, 0.25000000, -0.66666667,
                       0.083333333, -0.25000000, 0.33333333);

  double3x3 mrotX = rotate3x3X(M_PI*0.5);
  double3x3 mrotY = rotate3x3Y(M_PI*0.5);
  double3x3 mrotZ = rotate3x3Z(M_PI*0.5);

  double3x3 mrotI = inverse3x3(mrotX);
  double3x3 check = mrotI*mrotX;
  double3x3 identity;
  
  double3 p1(0,1,0);
  double3 p2 = mrotX*p1;
  double3 p3 = mrotZ*p1;
  double3 p4 = mul(mrotY, mrotX*p1);
 
  double error = 0.0;
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      error += fabs( m(i,j) - m2(j,i));
  
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      error += fabs( check[i][j] - identity[i][j]);

  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      error += fabs( m3[i][j] - check_inv[i][j]);
  
  error += length(p2 - double3(0,0,1));
  error += length(p3 - double3(-1,0,0));
  error += length(p4 - double3(1,0,0));

  return error < 1e-6f;
}

bool test013_bitcount_scalar()
{
  bool passed = true;
  passed = passed && (bitCount16(ushort(0xF0F0)) == 8) && (bitCount32(0xFFFFFFFFu) == 32);
  passed = passed && (bitCount(0x00000101u) == 2) && (bitCount64(0xFFFFFFFF00000001ull) == 33);
  passed = passed && (dot(3.0f, -2.0f) == -6.0f) && (SQR(3.0f) == 9.0f);
  passed = passed && (bit_cast<uint32_t>(1.0f) == 0x3f800000u) && (bit_cast<float>(0x3f800000u) == 1.0f);
  passed = passed && (sign(0.0) == 0.0) && (sign(-2.0) == -1.0) && (sign(0) == 0) && (sign(-2) == -1) && (sign(2) == 1);

  // second call of color_pack_* (after test004) uses already initialized static constant
  passed = passed && (color_pack_rgba(float4(1.0f)) == 0xFFFFFFFFu) && (color_pack_bgra(float4(0.0f)) == 0u);
  return passed;
}

bool test014_short_char_vectors()
{
  const ushort sData[4] = { 1, 2, 3, 4 };
  const ushort4 s0;
  const ushort4 s1(1, 2, 3, 4);
  const ushort4 s2(ushort(7));
  const ushort4 s3(sData);
  ushort4 s4 = make_ushort4(1, 2, 3, 4);
  s4[3] = 9;

  const ushort2 h0;
  const ushort2 h1(1, 2);
  const ushort2 h2(ushort(7));
  const ushort2 h3(sData);
  ushort2 h4(5, 6);
  h4[1] = 9;

  const uchar cData[4] = { 10, 20, 30, 40 };
  const uchar4 c0;
  const uchar4 c1(10, 20, 30, 40);
  const uchar4 c2(uchar(7));
  const uchar4 c3(cData);
  uchar4 c4 = make_uchar4(10, 20, 30, 40);
  c4[3] = 9;

  bool passed = true;
  for(int i=0;i<4;i++)
  {
    if(s0[i] != 0 || s1[i] != sData[i] || s2[i] != 7 || s3[i] != sData[i])
      passed = false;
    if(c0[i] != 0 || c1[i] != cData[i] || c2[i] != 7 || c3[i] != cData[i])
      passed = false;
  }
  for(int i=0;i<2;i++)
  {
    if(h0[i] != 0 || h1[i] != sData[i] || h2[i] != 7 || h3[i] != sData[i])
      passed = false;
  }
  passed = passed && (s4[0] == 1) && (s4[3] == 9) && (h4[0] == 5) && (h4[1] == 9) && (c4[0] == 10) && (c4[3] == 9);

  // uchar4 arithmetics
  const uchar4 a(10, 20, 30, 40);
  const uchar4 b(2, 4, 5, 6);
  const uchar4 c(20, 40, 60, 80);
  const uchar4 r[14] = { a*2.0f, a/2.0f, a+1.0f, a-1.0f, 
                         2.0f*a, 120.0f/b, 1.0f+a, 50.0f-a, 
                         a+b, a-b, a*b, a/b, 
                         lerp(a, c, 0.5f), uchar4() };
  const uchar ref[14][4] = { {20, 40, 60, 80}, {5, 10, 15, 20}, {11, 21, 31, 41}, {9, 19, 29, 39},
                             {20, 40, 60, 80}, {60, 30, 24, 20}, {11, 21, 31, 41}, {40, 30, 20, 10},
                             {12, 24, 35, 46}, {8, 16, 25, 34}, {20, 80, 150, 240}, {5, 5, 6, 6},
                             {15, 30, 45, 60}, {0, 0, 0, 0} };
  for(int j=0;j<14;j++)
  {
    for(int i=0;i<4;i++)
    {
      if(r[j][i] != ref[j][i])
      {
        std::cout << "uchar4 op " << j << ", comp " << i << ": " << int(r[j][i]) << " != " << int(ref[j][i]) << std::endl;
        passed = false;
      }
    }
  }

  passed = passed && (dot(a, b) == 10*2 + 20*4 + 30*5); // only xyz are used
  return passed;
}

bool test015_color_unpack()
{
  const float4 c1 = color_unpack_bgra(int(0xFF408000));
  const float4 c2 = color_unpack_rgba(int(0xFF008040));
  const float4 ref(64.0f/255.0f, 128.0f/255.0f, 0.0f, 1.0f);
  return length4f(c1 - ref) < 1e-6f && length4f(c2 - ref) < 1e-6f;
}

bool test016_camera_matrices()
{
  bool passed = true;

  // look at
  const float3 eye(0, 0, 5), center(0, 0, 0), up(0, 1, 0);
  const float4x4 mView = lookAt(eye, center, up);
  passed = passed && length(mul4x3(mView, eye)) < 1e-6f;
  passed = passed && length(mul4x3(mView, center) - float3(0, 0, -5)) < 1e-6f;
  passed = passed && length(mul4x3(mView, float3(1, 2, 5)) - float3(1, 2, 0)) < 1e-6f;

  // perspective: near plane goes to -1, far plane goes to +1 in NDC
  const float4x4 mProj = perspectiveMatrix(90.0f, 1.0f, 1.0f, 100.0f);
  const float4 pNear = mProj*float4(1, 1, -1, 1);
  const float4 pFar  = mProj*float4(0, 0, -100, 1);
  passed = passed && std::abs(pNear.z/pNear.w + 1.0f) < 1e-5f && std::abs(pNear.x/pNear.w - 1.0f) < 1e-5f && std::abs(pNear.y/pNear.w - 1.0f) < 1e-5f;
  passed = passed && std::abs(pFar.z/pFar.w - 1.0f) < 1e-5f;

  // orthographic
  const float4x4 mOrto = ortoMatrix(-2.0f, 2.0f, -1.0f, 1.0f, 0.5f, 10.0f);
  passed = passed && length(mul4x3(mOrto, float3( 2,  1, -10.0f)) - float3( 1,  1,  1)) < 1e-6f;
  passed = passed && length(mul4x3(mOrto, float3(-2, -1, -0.5f)) - float3(-1, -1, -1)) < 1e-6f;
  
  // vulkan fix
  const float4x4 mFix = OpenglToVulkanProjectionMatrixFix();
  passed = passed && length(mul4x3(mFix, float3(1, 1, -1)) - float3(1, -1, 0)) < 1e-6f;
  passed = passed && length(mul4x3(mFix, float3(1, 1,  1)) - float3(1, -1, 1)) < 1e-6f;

  // eye rays
  const float4x4 mProjInv = inverse4x4(mProj);
  const float4 ray1 = EyeRayDir4f(0.5f, 0.5f, 2.0f, 2.0f, mProjInv); // center of 2x2 image
  const float4 ray2 = EyeRayDir4f(0.0f, 0.0f, 2.0f, 2.0f, mProjInv); // left-top pixel
  const float3 ref2 = normalize(float3(-0.5f, -0.5f, -1.0f));
  passed = passed && length3f(ray1 - float4(0, 0, -1, 0)) < 1e-5f && ray1.w == INF_POSITIVE;
  passed = passed && length(to_float3(ray2) - ref2) < 1e-5f && ray2.w == INF_POSITIVE;

  return passed;
}

bool test017_box4f()
{
  bool passed = true;

  Box4f box;
  passed = passed && (box.boxMin.x == INF_POSITIVE) && (box.boxMax.x == INF_NEGATIVE);
  box.include(float4( 1, 2, 3, 0));
  box.include(float4(-1, 5, 0, 0));
  passed = passed && length4f(box.boxMin - float4(-1, 2, 0, 0)) == 0.0f && length4f(box.boxMax - float4(1, 5, 3, 0)) == 0.0f;
  passed = passed && (box.surfaceArea() == 42.0f) && (box.volume() == 18.0f);

  Box4f box2(float4(0, 0, 0, 0), float4(2, 2, 2, 0));
  box2.include(box);
  passed = passed && length4f(box2.boxMin - float4(-1, 0, 0, 0)) == 0.0f && length4f(box2.boxMax - float4(2, 5, 3, 0)) == 0.0f;
  box2.intersect(Box4f(float4(0, 1, 1, 0), float4(4, 4, 4, 0)));
  passed = passed && length4f(box2.boxMin - float4(0, 1, 1, 0)) == 0.0f && length4f(box2.boxMax - float4(2, 4, 3, 0)) == 0.0f;

  box2.setStart(5);
  box2.setCount(7);
  passed = passed && (box2.getStart() == 5) && (box2.getCount() == 7) && (box2.boxMin.x == 0.0f) && (box2.boxMax.z == 3.0f);

  const float4 pI = packIntW(float4(1, 2, 3, 4), -3);
  const float4 pF = packFloatW(float4(1, 2, 3, 4), 0.5f);
  passed = passed && (extractIntW(pI) == -3) && (pI.z == 3.0f) && (pF.w == 0.5f) && (pF.x == 1.0f);

  const Box4f flat(float4(0, 0, 2, 0), float4(1, 1, 2, 0));
  passed = passed && flat.isAxisAligned(2, 2.0f) && !flat.isAxisAligned(2, 1.0f) && !flat.isAxisAligned(0, 0.0f);

  // overlap of boxes
  const Box4f b1(float4(0, 0, 0, 0), float4(2, 2, 2, 0));
  const Box4f b2(float4(1,-1, 1, 0), float4(3, 1, 3, 0));
  const Box4f b3(float4(5, 5, 5, 0), float4(6, 6, 6, 0));
  const Box4f o1 = BoxBoxOverlap(b1, b2);
  const Box4f o2 = BoxBoxOverlap(b1, b3);
  passed = passed && length3f(o1.boxMin - float4(1, 0, 1, 0)) == 0.0f && length3f(o1.boxMax - float4(2, 1, 2, 0)) == 0.0f;
  passed = passed && length3f(o2.boxMin - b1.boxMax) == 0.0f && length3f(o2.boxMax - b1.boxMax) == 0.0f;

  return passed;
}

bool test018_ray4f()
{
  bool passed = true;

  Ray4f r0;
  r0.posAndNear = float4(0, 0, 0, 0);
  r0.dirAndFar  = float4(0, 0, 1, 0);
  r0.setNear(0.25f);
  r0.setFar(50.0f);
  passed = passed && (r0.getNear() == 0.25f) && (r0.getFar() == 50.0f) && (r0.dirAndFar.z == 1.0f);

  const Ray4f r1(float4(1, 2, 3, 0), float4(0, 0, 1, 0));
  const Ray4f r2(float4(1, 2, 3, 0), float4(0, 0, 1, 0), 0.5f, 100.0f);
  const Ray4f r3(float3(1, 2, 3), float3(0, 0, 1), 1.0f, 2.0f);
  passed = passed && (r1.getNear() == 0.0f) && (r1.getFar() == 0.0f) && (r1.posAndNear.y == 2.0f);
  passed = passed && (r2.getNear() == 0.5f) && (r2.getFar() == 100.0f) && (r2.posAndNear.y == 2.0f);
  passed = passed && (r3.getNear() == 1.0f) && (r3.getFar() == 2.0f) && (r3.posAndNear.z == 3.0f);

  // ray-box
  const float4 boxMin(0, 0, 0, 0), boxMax(1, 1, 1, 0);
  const float4 dirInv = 1.0f/float4(1, 1, 1, 1);
  const float2 hit  = Ray4fBox4fIntersection(float4(-1, -1, -1, 0), dirInv, boxMin, boxMax);
  const float2 miss = Ray4fBox4fIntersection(float4(-1,  5, -1, 0), dirInv, boxMin, boxMax);
  passed = passed && (hit.x == 1.0f) && (hit.y == 2.0f) && (miss.x > miss.y);

  return passed;
}

bool test019_bbox3f()
{
  BBox3f box;
  box.boxMin = float3(0, 0, 0);
  box.boxMax = float3(1, 1, 1);
  
  struct TestCase { float3 pos; float3 dir; int face; float t1; float t2; };
  const TestCase cases[] = { { float3(-1.0f, 0.5f,  0.5f),  float3( 1.0f,   0.25f,   0.25f), 0, 1.0f, 2.0f },
                             { float3( 0.5f,-1.0f,  0.5f),  float3( 0.25f,  1.0f,    0.25f), 1, 1.0f, 2.0f },
                             { float3( 0.5f, 0.5f, -1.0f),  float3( 0.25f,  0.125f,  1.0f),  2, 1.0f, 2.0f },
                             { float3( 0.5f, 0.5f, -1.0f),  float3( 0.125f, 0.25f,   1.0f),  2, 1.0f, 2.0f },
                             { float3( 2.0f, 0.75f, 0.75f), float3(-1.0f,  -0.25f,  -0.25f), 0, 1.0f, 2.0f } };
  
  bool passed = true;
  for(const auto& tc : cases)
  {
    const auto res = box.Intersection(tc.pos, 1.0f/tc.dir, 0.0f, 100.0f);
    if(res.face != tc.face || std::abs(res.t1 - tc.t1) > 1e-5f || std::abs(res.t2 - tc.t2) > 1e-5f)
    {
      std::cout << "BBox3f::Intersection: face = " << res.face << ", t1 = " << res.t1 << ", t2 = " << res.t2 << std::endl;
      passed = false;
    }
  }

  // clip by allowed range
  const auto res = box.Intersection(float3(-1.0f, 0.5f, 0.5f), 1.0f/float3(1.0f, 0.25f, 0.25f), 1.5f, 1.75f);
  passed = passed && (res.t1 == 1.5f) && (res.t2 == 1.75f);

  return passed;
}

bool test020_interlocked_reduce()
{
  float  f = 1.0f, fOld = 0.0f;
  double d = 1.0,  dOld = 0.0;
  int    i = 1,    iOld = 0;
  uint   u = 1,    uOld = 0;

  InterlockedAdd(f, 2.0f); InterlockedAdd(f, 3.0f, fOld);
  InterlockedAdd(d, 2.0);  InterlockedAdd(d, 3.0,  dOld);
  InterlockedAdd(i, 2);    InterlockedAdd(i, 3,    iOld);
  InterlockedAdd(u, 2u);   InterlockedAdd(u, 3u,   uOld);

  float arr[4] = { 0, 1, 2, 3 };
  InterlockedAdd3f(arr, 1, float3(1, 2, 3));

  bool passed = true;
  passed = passed && (f == 6.0f) && (fOld == 3.0f) && (d == 6.0) && (dOld == 3.0);
  passed = passed && (i == 6) && (iOld == 3) && (u == 6) && (uOld == 3);
  passed = passed && (arr[0] == 0.0f) && (arr[1] == 2.0f) && (arr[2] == 4.0f) && (arr[3] == 6.0f);
  passed = passed && (omp_get_num_threads() >= 1) && (omp_get_max_threads() >= 1) && (omp_get_thread_num() == 0);

  passed = passed && (align(13, 8) == 16) && (align(16, 8) == 16);

  std::vector<float> vec;
  const size_t sizeAligned = ReduceAddInit(vec, 10);
  ReduceAdd(vec, 3, 2.0f);
  ReduceAdd(vec, 3, 1.0f);
  ReduceAdd(vec, 5, INF_POSITIVE);                     // non finite values are ignored
  ReduceAdd(vec, size_t(4), sizeAligned, 5.0f);
  ReduceAdd(vec, size_t(6), sizeAligned, INF_NEGATIVE); // non finite values are ignored
  ReduceAddComplete(vec);
  passed = passed && (vec.size() == 10) && (sizeAligned >= 10) && (vec[3] == 3.0f) && (vec[4] == 5.0f) && (vec[5] == 0.0f) && (vec[6] == 0.0f);

  return passed;
}
