#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include "LiteMath.h"
using namespace LiteMath;

template<typename T>
static void PrintRR(const char* name1, const char* name2, T res[], T ref[], int size = 4)
{
  std::cout << name1 << ": ";
  for(int i=0;i<size;i++)
    std::cout << res[i] << " ";
  std::cout << std::endl;
  std::cout << name2 << ": "; 
   for(int i=0;i<size;i++)
    std::cout << ref[i] << " ";
  std::cout << std::endl;
  std::cout << std::endl;
}

bool test160_basev_double4()
{
  const double4 Cx1( double(1),  double(2),  double(-3),  double(4));
  const double4 Cx2( double(5),  double(-5),  double(6),  double(4));

  const auto Cx3 = Cx1 - Cx2;
  const auto Cx4 = (Cx1 + Cx2)*Cx1;
  const auto Cx5 = (Cx2 - Cx1)/Cx1;

  double result1[4];
  double result2[4];
  double result3[4];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  
  double expr1[4], expr2[4], expr3[4];
  bool passed = true;
  for(int i=0;i<4;i++)
  {
    expr1[i] = Cx1[i] - Cx2[i];
    expr2[i] = (Cx1[i] + Cx2[i])*Cx1[i];
    expr3[i] = (Cx2[i] - Cx1[i])/Cx1[i];
    

    if(fabs(result1[i] - expr1[i]) > 1e-6f || fabs(result2[i] - expr2[i]) > 1e-6f || fabs(result3[i] - expr3[i]) > 1e-6f) 

      passed = false;
  }

  if(!passed)
  {
    PrintRR("exp1_res", "exp2_res", result1, expr1, 4);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 4); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 4);
  }
  
  return passed;
}

bool test161_basek_double4()
{
  const double4 Cx1( double(1),  double(2),  double(-3),  double(4));
  const double Cx2 = double(5);

  const double4 Cx3 = Cx2*(Cx2 + Cx1) - double(2);
  const double4 Cx4 = double(1) + (Cx1 + Cx2)*Cx2;
  
  const double4 Cx5 = double(3) - Cx2/(Cx2 - Cx1);
  const double4 Cx6 = (Cx2 + Cx1)/Cx2 + double(5)/Cx1;

  CVEX_ALIGNED(16) double result1[4]; 
  CVEX_ALIGNED(16) double result2[4];
  CVEX_ALIGNED(16) double result3[4];
  CVEX_ALIGNED(16) double result4[4];

  store(result1, Cx3);
  store(result2, Cx4);
  store(result3, Cx5);
  store(result4, Cx6);
  
  bool passed = true;
  for(int i=0;i<4;i++)
  {
    const double expr1 = Cx2*(Cx2 + Cx1[i]) - double(2);
    const double expr2 = double(1) + (Cx1[i] + Cx2)*Cx2;
    const double expr3 = double(3) - Cx2/(Cx2 - Cx1[i]);
    const double expr4 = (Cx2 + Cx1[i])/Cx2 + double(5)/Cx1[i];
    

    if(fabs(result1[i] - expr1) > 1e-6f || fabs(result2[i] - expr2) > 1e-6f || fabs(result3[i] - expr3) > 1e-6f || fabs(result4[i] - expr4) > 1e-6f )

      passed = false;
  }

  return passed;
}

bool test162_unaryv_double4()
{
  const double4 Cx1( double(1),  double(2),  double(-3),  double(4));
  const double4 Cx2( double(5),  double(-5),  double(6),  double(4));

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  double result1[4];
  double result2[4];
  double result3[4];
  double result4[4];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  double expr1[4], expr2[4], expr3[4], expr4[4];
  bool passed = true;
  for(int i=0;i<4;i++)
  {
    expr1[i] = Cx1[i] + Cx2[i];
    expr2[i] = Cx1[i] - Cx2[i];
    expr3[i] = Cx1[i] * Cx2[i];
    expr4[i] = Cx1[i] / Cx2[i];
    

    if(fabs(result1[i] - expr1[i]) > 1e-6f || fabs(result2[i] - expr2[i]) > 1e-6f || fabs(result3[i] - expr3[i]) > 1e-6f) 

      passed = false;
  }

  if(!passed)
  {
    PrintRR("exp1_res", "exp2_res", result1, expr1, 4);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 4); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 4);
    PrintRR("exp4_res", "exp4_res", result4, expr4, 4);
  }
  
  return passed;
}

bool test162_unaryk_double4()
{
  const double4 Cx1( double(1),  double(2),  double(-3),  double(4));
  const double Cx2 = double(5);

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  double result1[4];
  double result2[4];
  double result3[4];
  double result4[4];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  double expr1[4], expr2[4], expr3[4], expr4[4];
  bool passed = true;
  for(int i=0;i<4;i++)
  {
    expr1[i] = Cx1[i] + Cx2;
    expr2[i] = Cx1[i] - Cx2;
    expr3[i] = Cx1[i] * Cx2;
    expr4[i] = Cx1[i] / Cx2;
    

    if(std::abs(result1[i] - expr1[i]) > 1e-6f || std::abs(result2[i] - expr2[i]) > 1e-6f || std::abs(result3[i] - expr3[i]) > 1e-6f) 
      passed = false;

  }

  if(!passed)
  {
    PrintRR("exp1_res", "exp2_res", result1, expr1, 4);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 4); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 4);
    PrintRR("exp4_res", "exp4_res", result4, expr4, 4);
  }
  
  return passed;
}

bool test163_cmpv_double4()
{
  const double4 Cx1( double(1),  double(2),  double(-3),  double(4));
  const double4 Cx2( double(5),  double(-5),  double(6),  double(4));

  auto Cx3 = (Cx1 < Cx2 );
  auto Cx4 = (Cx1 > Cx2 );
  auto Cx5 = (Cx1 <= Cx2);
  auto Cx6 = (Cx1 >= Cx2);
  auto Cx7 = (Cx1 == Cx2);
  auto Cx8 = (Cx1 != Cx2);

  const auto Cx9  = blend(Cx1, Cx2, Cx3);
  const auto Cx10 = blend(Cx1, Cx2, Cx6);

  uint32_t result1[4];
  uint32_t result2[4];
  uint32_t result3[4];
  uint32_t result4[4];
  uint32_t result5[4];
  uint32_t result6[4];
  double result7[4];
  double result8[4];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  store_u(result5, Cx7);
  store_u(result6, Cx8);
  store_u(result7, Cx9);
  store_u(result8, Cx10);
  
  uint32_t expr1[4], expr2[4], expr3[4], expr4[4], expr5[4], expr6[4];
  double expr7[4],  expr8[4];
  bool passed = true;
  for(int i=0;i<4;i++)
  {
    expr1[i] = Cx1[i] <  Cx2[i] ? 0xFFFFFFFF : 0;
    expr2[i] = Cx1[i] >  Cx2[i] ? 0xFFFFFFFF : 0;
    expr3[i] = Cx1[i] <= Cx2[i] ? 0xFFFFFFFF : 0;
    expr4[i] = Cx1[i] >= Cx2[i] ? 0xFFFFFFFF : 0;
    expr5[i] = Cx1[i] == Cx2[i] ? 0xFFFFFFFF : 0;
    expr6[i] = Cx1[i] != Cx2[i] ? 0xFFFFFFFF : 0;
    expr7[i] = Cx1[i] <  Cx2[i] ? Cx1[i] : Cx2[i];
    expr8[i] = Cx1[i] >= Cx2[i] ? Cx1[i] : Cx2[i];
    
    if(result1[i] != expr1[i] || result2[i] != expr2[i] || result3[i] != expr3[i] || result4[i] != expr4[i] || 
       result5[i] != expr5[i] || result6[i] != expr6[i] || result7[i] != expr7[i] || result8[i] != expr8[i]) 
      passed = false;
  }

  if(!passed)
  {
    PrintRR("exp1_res", "exp2_res", result1, expr1, 4);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 4); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 4);
    PrintRR("exp4_res", "exp4_res", result4, expr4, 4);
    PrintRR("exp5_res", "exp5_res", result5, expr5, 4);
    PrintRR("exp6_res", "exp6_res", result6, expr6, 4);
    PrintRR("exp7_res", "exp7_res", result7, expr7, 4);
    PrintRR("exp8_res", "exp8_res", result8, expr8, 4);
  }
  
  return passed;
}

bool test164_shuffle_double4()
{ 
  const double4 Cx1( double(1),  double(2),  double(-3),  double(4));


  const double4 Cr1 = shuffle_zyxw(Cx1);
  const double4 Cr2 = shuffle_zxyw(Cx1);
  const double4 Cr3 = shuffle_yzxw(Cx1);
  const double4 Cr6 = shuffle_yxzw(Cx1);
  const double4 Cr7 = shuffle_xzyw(Cx1);
  
  const double4 Cr4 = shuffle_xyxy(Cx1);
  const double4 Cr5 = shuffle_zwzw(Cx1);

  CVEX_ALIGNED(16) double result1[4];
  CVEX_ALIGNED(16) double result2[4];
  CVEX_ALIGNED(16) double result3[4];
  CVEX_ALIGNED(16) double result4[4];
  CVEX_ALIGNED(16) double result5[4];
  CVEX_ALIGNED(16) double result6[4];
  CVEX_ALIGNED(16) double result7[4];

  store(result1, Cr1);
  store(result2, Cr2);
  store(result3, Cr3);
  store(result4, Cr4);
  store(result5, Cr5);
  store(result6, Cr6);
  store(result7, Cr7);

  const bool b1 = (result1[0] == Cx1[2]) && (result1[1] == Cx1[1]) && (result1[2] == Cx1[0]) && (result1[3] == Cx1[3]);
  const bool b2 = (result2[0] == Cx1[2]) && (result2[1] == Cx1[0]) && (result2[2] == Cx1[1]) && (result2[3] == Cx1[3]);
  const bool b3 = (result3[0] == Cx1[1]) && (result3[1] == Cx1[2]) && (result3[2] == Cx1[0]) && (result3[3] == Cx1[3]);
  const bool b6 = (result6[0] == Cx1[1]) && (result6[1] == Cx1[0]) && (result6[2] == Cx1[2]) && (result6[3] == Cx1[3]);
  const bool b7 = (result7[0] == Cx1[0]) && (result7[1] == Cx1[2]) && (result7[2] == Cx1[1]) && (result7[3] == Cx1[3]);
  
  const bool b4 = (result4[0] == Cx1[0]) && (result4[1] == Cx1[1]) && (result4[2] == Cx1[0]) && (result4[3] == Cx1[1]);
  const bool b5 = (result5[0] == Cx1[2]) && (result5[1] == Cx1[3]) && (result5[2] == Cx1[2]) && (result5[3] == Cx1[3]);
 
  const bool passed = (b1 && b2 && b3 && b4 && b5 && b6 && b7);

  if(!passed)
  {
    std::cout << result1[0] << " " << result1[1] << " " << result1[2] << " " << result1[3] << std::endl;
    std::cout << result2[0] << " " << result2[1] << " " << result2[2] << " " << result2[3] << std::endl;
    std::cout << result3[0] << " " << result3[1] << " " << result3[2] << " " << result3[3] << std::endl;
    std::cout << result4[0] << " " << result4[1] << " " << result4[2] << " " << result4[3] << std::endl;
    std::cout << result5[0] << " " << result5[1] << " " << result5[2] << " " << result5[3] << std::endl;
    std::cout << result6[0] << " " << result6[1] << " " << result6[2] << " " << result6[3] << std::endl;
    std::cout << result7[0] << " " << result7[1] << " " << result7[2] << " " << result7[3] << std::endl;
  }

  return passed;


}

bool test165_exsplat_double4()
{
  const double4 Cx1( double(1),  double(2),  double(-3),  double(4));

  const double4 Cr0 = splat_0(Cx1);
  const double4 Cr1 = splat_1(Cx1);
  const double4 Cr2 = splat_2(Cx1);
  const double4 Cr3 = splat_3(Cx1);

  const double s0 = extract_0(Cx1);
  const double s1 = extract_1(Cx1);
  const double s2 = extract_2(Cx1);
  const double s3 = extract_3(Cx1);

  double result0[4];
  double result1[4];
  double result2[4];
  double result3[4];

  store_u(result0, Cr0);
  store_u(result1, Cr1);
  store_u(result2, Cr2);
  store_u(result3, Cr3);
  
  bool passed = true;
  for (int i = 0; i<4; i++)
  {

    if((result0[i] != Cx1[0]))
      passed = false;
    if((result1[i] != Cx1[1]))
      passed = false;
    if((result2[i] != Cx1[2]))
      passed = false;
    if((result3[i] != Cx1[3]))
      passed = false;
  }

  if(s0 != Cx1[0])
    passed = false;
  if(s1 != Cx1[1])
    passed = false;
  if(s2 != Cx1[2])
    passed = false;
  if(s3 != Cx1[3])
    passed = false;
  return passed;
}

bool test166_misc_double4()
{
  const double4 Cx1( double(1),  double(2),  double(-3),  double(4));
  const double4 Cx0;
  const double4 Cx2 = -Cx1;
  const double4 Cx3 = make_double4( double(1),  double(2),  double(-3),  double(4));

  bool passed = true;
  for(int i=0;i<4;i++)
  {
    if(Cx0[i] != double(0) || Cx2[i] != double(-Cx1[i]) || Cx3[i] != Cx1[i])
      passed = false;
  }

  // conversion constructors from other vector types

  {
    const float4 src( float(1),  float(2),  float(3),  float(4));
    const double4 dst(src);
    for(int i=0;i<4;i++)
      if(dst[i] != double(src[i]))
        passed = false;
  }

  {
    const int4 src( int(1),  int(2),  int(3),  int(4));
    const double4 dst(src);
    for(int i=0;i<4;i++)
      if(dst[i] != double(src[i]))
        passed = false;
  }



  const double3 Cr3 = to_double3(Cx1);
  const double4  Cr4 = to_double4(Cr3, double(7));
  passed = passed && (Cr3.x == Cx1.x) && (Cr3.y == Cx1.y) && (Cr3.z == Cx1.z);
  passed = passed && (Cr4.x == Cx1.x) && (Cr4.y == Cx1.y) && (Cr4.z == Cx1.z) && (Cr4.w == double(7));



  // geometric functions: n is unit normal along x, I is unit grazing direction with dot(I,n) < 0
  double4 n(double(0)), t(double(0));
  n[0] = 1;
  t[1] = 1;
  const double4  I   = normalize(t - n*double(0.1));
  const double IdN = dot(I, n);
  passed = passed && (std::abs(length(I) - double(1)) < 1e-6f) && (IdN < double(0));

  const double4 r  = reflect(I, n);
  const double4 r1 = refract(I, n, double(1));  // same media, ray goes straight
  const double4 r2 = refract(I, n, double(10)); // total internal reflection, returns zero
  const double4 f1 = faceforward(n, I, n);
  const double4 f2 = faceforward(n, -I, n);
  for(int i=0;i<4;i++)
  {
    if(std::abs(r[i] - (I[i] - double(2)*IdN*n[i])) > 1e-6f)
      passed = false;
    if(std::abs(r1[i] - I[i]) > 1e-6f || r2[i] != double(0))
      passed = false;
    if(f1[i] != n[i] || f2[i] != -n[i])
      passed = false;
  }


  const double4 Cx4( double(5),  double(-5),  double(6),  double(4));
  const double d3 = Cx1.x*Cx4.x + Cx1.y*Cx4.y + Cx1.z*Cx4.z;
  const double d4 = d3 + Cx1.w*Cx4.w;
  const double l3 = std::sqrt(Cx1.x*Cx1.x + Cx1.y*Cx1.y + Cx1.z*Cx1.z);
  const double l4 = std::sqrt(Cx1.x*Cx1.x + Cx1.y*Cx1.y + Cx1.z*Cx1.z + Cx1.w*Cx1.w);
  const double4 dv3 = dot3v(Cx1, Cx4);
  const double4 dv4 = dot4v(Cx1, Cx4);
  const double4 lv3 = length3v(Cx1);
  const double4 lv4 = length4v(Cx1);
  const double4 n3  = normalize3(Cx1);
  passed = passed && (std::abs(dot3f(Cx1, Cx4) - d3) < 1e-6f) && (std::abs(dot4f(Cx1, Cx4) - d4) < 1e-6f);
  passed = passed && (std::abs(length3(Cx1) - l3) < 1e-6f) && (std::abs(length3f(Cx1) - l3) < 1e-6f);
  passed = passed && (std::abs(length4(Cx1) - l4) < 1e-6f) && (std::abs(length4f(Cx1) - l4) < 1e-6f);
  for(int i=0;i<4;i++)
  {
    if(std::abs(dv3[i] - d3) > 1e-6f || std::abs(dv4[i] - d4) > 1e-6f || std::abs(lv3[i] - l3) > 1e-6f || std::abs(lv4[i] - l4) > 1e-6f)
      passed = false;
    if(std::abs(n3[i] - Cx1[i]/l3) > 1e-6f)
      passed = false;
  }



  return passed;
}

bool test167_funcv_double4()
{
  const double4 Cx1( double(1),  double(2),  double(-3),  double(4));
  const double4 Cx2( double(5),  double(-5),  double(6),  double(4));
  const double4 Cx9( double(2),  double(2),  double(2),  double(2));
  const double4 Cx0( double(0),  double(0),  double(0),  double(0));

  
  auto Cx3 = sign(Cx1);
  auto Cx4 = abs(Cx1);

  auto Cx5 = clamp(Cx1, double(2), double(3) );
  auto Cx6 = min(Cx1, Cx2);
  auto Cx7 = max(Cx1, Cx2);
  auto Cx8 = clamp(Cx1, Cx0, Cx9);

  double Cm = hmin(Cx1);
  double CM = hmax(Cx1);
  double horMinRef = Cx1[0];
  double horMaxRef = Cx1[0];
  

  double Cm3 = hmin3(Cx1);
  double CM3 = hmax3(Cx1);
  double horMinRef3 = Cx1[0];
  double horMaxRef3 = Cx1[0];

  
  for(int i=0;i<4;i++)
  {
    horMinRef = std::min(horMinRef, Cx1[i]);
    horMaxRef = std::max(horMaxRef, Cx1[i]);

    if(i<3)
    {
      horMinRef3 = std::min(horMinRef3, Cx1[i]);
      horMaxRef3 = std::max(horMaxRef3, Cx1[i]);
    }

  }

  bool passed = true;
  for(int i=0;i<4;i++)
  {
  
    if(Cx3[i] != sign(Cx1[i]))
      passed = false;
    if(Cx4[i] != abs(Cx1[i]))
      passed = false;

    if(Cx5[i] != clamp(Cx1[i], double(2), double(3) ))
      passed = false;
    if(Cx6[i] != min(Cx1[i], Cx2[i]))
      passed = false;
    if(Cx7[i] != max(Cx1[i], Cx2[i]))
      passed = false;
    if(Cx8[i] != clamp(Cx1[i], Cx0[i], Cx9[i]))
      passed = false;
  }

  if(horMinRef != Cm)
    passed = false;
  if(horMaxRef != CM)
    passed = false;

  if(horMinRef3 != Cm3)
    passed = false;
  if(horMaxRef3 != CM3)
    passed = false;


  return passed;
}




bool test168_funcfv_double4()
{
  const double4 Cx1( double(1),  double(2),  double(-3),  double(4));
  const double4 Cx2( double(5),  double(-5),  double(6),  double(4));
  
  auto Cx3 = mod(Cx1, Cx2);
  auto Cx4 = fract(Cx1);
  auto Cx5 = ceil(Cx1);
  auto Cx6 = floor(Cx1);
  auto Cx7 = sign(Cx1);
  auto Cx8 = abs(Cx1);

  auto Cx9  = clamp(Cx1, -2.0f, 2.0f);
  auto Cx10 = min(Cx1, Cx2);
  auto Cx11 = max(Cx1, Cx2);

  auto Cx12 = mix (Cx1, Cx2, 0.5f);
  auto Cx13 = lerp(Cx1, Cx2, 0.5f);
  
  auto Cx14 = sqrt(Cx8); 
  auto Cx15 = inversesqrt(Cx8);

  auto Cx18 = rcp(Cx1);

  double ref[19][4];
  double res[19][4];
  memset(ref, 0, 19*sizeof(double)*4);
  memset(res, 0, 19*sizeof(double)*4);

  store_u(res[3],  Cx3);
  store_u(res[4],  Cx4);
  store_u(res[5],  Cx5);
  store_u(res[6],  Cx6);
  store_u(res[7],  Cx7);
  store_u(res[8],  Cx8);
  store_u(res[9],  Cx9);
  store_u(res[10], Cx10);
  store_u(res[11], Cx11);
  store_u(res[12], Cx12);
  store_u(res[13], Cx13);
  store_u(res[14], Cx14);
  store_u(res[15], Cx15);
  store_u(res[18], Cx18);

  for(int i=0;i<4;i++)
  {
    ref[3][i] = mod(Cx1[i], Cx2[i]);
    ref[4][i] = fract(Cx1[i]);
    ref[5][i] = ceil(Cx1[i]);
    ref[6][i] = floor(Cx1[i]);
    ref[7][i] = sign(Cx1[i]);
    ref[8][i] = abs(Cx1[i]);

    ref[9][i]  = clamp(Cx1[i], double(-2), double(2) );
    ref[10][i] = min(Cx1[i], Cx2[i]);
    ref[11][i] = max(Cx1[i], Cx2[i]);

    ref[12][i] = mix (Cx1[i], Cx2[i], 0.5f);
    ref[13][i] = lerp(Cx1[i], Cx2[i], 0.5f);

    ref[14][i] = sqrt(Cx8[i]);
    ref[15][i] = inversesqrt(Cx8[i]);
    ref[18][i] = rcp(Cx1[i]);
  }
  
  bool passed = true;
  for(int i=0;i<4;i++)
  {
    for(int j=3;j<=18;j++)
    {
      if(abs( res[j][i]-ref[j][i]) > 1e-6f)
      {
        if(j == 15 || j == 18)
        {
          if(abs(res[j][i]-ref[j][i]) > 5e-4f)
          {
            passed = false;
            break;
          }
        }
        else
        {
          passed = false;
          break;
        }
      }
    }
  }

  return passed;
}

bool test169_cstcnv_double4()
{
  const double4 Cx1( double(1),  double(2),  double(-3),  double(4));
  
  const int4  Cr1 = to_int32(Cx1);
  const uint4 Cr2 = to_uint32(Cx1);
  const int4  Cr3 = as_int32(Cx1);
  const uint4 Cr4 = as_uint32(Cx1);

  int          result1[4];
  unsigned int result2[4];
  int          result3[4];
  unsigned int result4[4];

  store_u(result1, Cr1);
  store_u(result2, Cr2);
  store_u(result3, Cr3);
  store_u(result4, Cr4);

  int          ref1[4];
  unsigned int ref2[4];
  for(int i=0;i<4;i++)
  {
    ref1[i] = int(Cx1[i]);
    ref2[i] = (unsigned int)(Cx1[i]); 
  }

  int          ref3[4];
  unsigned int ref4[4];

  memcpy(ref3, &Cr3, sizeof(int)*4);
  memcpy(ref4, &Cr4, sizeof(uint)*4);
  
  bool passed = true;
  for (int i=0; i<4; i++)
  {
    if (result1[i] != ref1[i] || result2[i] != ref2[i] || result3[i] != ref3[i] || result4[i] != ref4[i])
    {
      passed = false;
      break;
    }
  }
  return passed;
}




bool test170_other_double4() // dummy test
{
  const double CxData[4] = {  double(1),  double(2),  double(-3),  double(4)};
  const double4  Cx1(CxData);
  const double4  Cx2(double4(1));
 
  const double4  Cx3 = Cx1 + Cx2;
  double result1[4];
  double result2[4];
  double result3[4];
  store_u(result1, Cx1);
  store_u(result2, Cx2);
  store_u(result3, Cx3);

  bool passed = true;
  for (int i=0; i<4; i++)
  {

    if (std::abs(result1[i] + double(1) - result3[i]) > 1e-10f || std::abs(result2[i] - double(1) > 1e-10f) )

    {
      passed = false;
      break;
    }
  }


  const double  dat3 = dot3(Cx1, Cx2);
  const double  dat4 = dot4(Cx1, Cx2);
  const double4   crs4 = cross3(Cx1, Cx2);

  const double  dat5 = dot  (Cx1, Cx2);

  const double4   crs3 = cross(Cx1, Cx2);
  const double crs_ref[3] = { Cx1[1]*Cx2[2] - Cx1[2]*Cx2[1], 
                                      Cx1[2]*Cx2[0] - Cx1[0]*Cx2[2], 
                                      Cx1[0]*Cx2[1] - Cx1[1]*Cx2[0] };



  passed = passed && (std::abs(dat3 - (Cx1.x*Cx2.x + Cx1.y*Cx2.y + Cx1.z*Cx2.z)) < 1e-6f);
  passed = passed && (std::abs(dat4 - (Cx1.x*Cx2.x + Cx1.y*Cx2.y + Cx1.z*Cx2.z + Cx1.w*Cx2.w)) < 1e-6f);
  passed = passed && (std::abs(dot(crs4-crs3, crs4-crs3)) < 1e-6f);

  {
    double sum = double(0);
    for(int i=0;i<4;i++)
      sum += Cx1[i]*Cx2[i];
    passed = passed && (std::abs(sum - dat5) < 1e-6f);

    for(int i=0;i<3;i++)
      passed = passed && (std::abs(crs3[i] - crs_ref[i]) < 1e-6f);

  }


  return passed;
}

bool test171_any_all_double4() // dummy test
{
  const double CxData[4] = {  double(1),  double(2),  double(3),  double(4)};
  const double4  Cx1(CxData);
  const double4  Cx2(double4(1));
 
  const double4  Cx3 = Cx1 + Cx2;
  

  auto cmp1 = (Cx1 < Cx3);
  auto cmp2 = (Cx1 < Cx2);
  auto cmp3 = (Cx1 <= Cx2);
  auto cmp4 = (Cx1 > Cx3);


  const bool a1 = all_of(cmp1);
  const bool a2 = all_of(cmp2);
  const bool a3 = any_of(cmp3);
  const bool a4 = any_of(cmp4);

  return a1 && !a2 && a3 && !a4;
}


