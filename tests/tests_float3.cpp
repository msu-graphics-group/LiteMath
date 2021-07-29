#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include "LiteMath.h"
using namespace LiteMath;

template<typename T>
static void PrintRR(const char* name1, const char* name2, T res[], T ref[], int size)
{
  std::cout << name1 << ": ";
  for(int i=0;i<4;i++)
    std::cout << res[i] << " ";
  std::cout << std::endl;
  std::cout << name2 << ": "; 
   for(int i=0;i<4;i++)
    std::cout << ref[i] << " ";
  std::cout << std::endl;
  std::cout << std::endl;
}

bool test150_basev_float3()
{
  const float3 Cx1( float(-1),  float(2),  float(-3));
  const float3 Cx2( float(3),  float(-4),  float(4));

  const auto Cx3 = Cx1 - Cx2;
  const auto Cx4 = (Cx1 + Cx2)*Cx1;
  const auto Cx5 = (Cx2 - Cx1)/Cx1;

  float result1[3];
  float result2[3];
  float result3[3];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  
  float expr1[3], expr2[3], expr3[3];
  bool passed = true;
  for(int i=0;i<3;i++)
  {
    expr1[i] = Cx1[i] - Cx2[i];
    expr2[i] = (Cx1[i] + Cx2[i])*Cx1[i];
    expr3[i] = (Cx2[i] - Cx1[i])/Cx1[i];
    

    if(fabs(result1[i] - expr1[i]) > 1e-6f || fabs(result2[i] - expr2[i]) > 1e-6f || fabs(result3[i] - expr3[i]) > 1e-6f) 

      passed = false;
  }

  if(!passed)
  {
    PrintRR("exp1_res", "exp2_res", result1, expr1, 3);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 3); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 3);
  }
  
  return passed;
}

bool test151_basek_float3()
{
  const float3 Cx1( float(-1),  float(2),  float(-3));
  const float Cx2 = float(3);

  const float3 Cx3 = Cx2*(Cx2 + Cx1) - float(2);
  const float3 Cx4 = float(1) + (Cx1 + Cx2)*Cx2;
  
  const float3 Cx5 = float(3) - Cx2/(Cx2 - Cx1);
  const float3 Cx6 = (Cx2 + Cx1)/Cx2 + float(5)/Cx1;

  CVEX_ALIGNED(16) float result1[4]; 
  CVEX_ALIGNED(16) float result2[4];
  CVEX_ALIGNED(16) float result3[4];
  CVEX_ALIGNED(16) float result4[4];

  store(result1, Cx3);
  store(result2, Cx4);
  store(result3, Cx5);
  store(result4, Cx6);
  
  bool passed = true;
  for(int i=0;i<3;i++)
  {
    const float expr1 = Cx2*(Cx2 + Cx1[i]) - float(2);
    const float expr2 = float(1) + (Cx1[i] + Cx2)*Cx2;
    const float expr3 = float(3) - Cx2/(Cx2 - Cx1[i]);
    const float expr4 = (Cx2 + Cx1[i])/Cx2 + float(5)/Cx1[i];
    

    if(fabs(result1[i] - expr1) > 1e-6f || fabs(result2[i] - expr2) > 1e-6f || fabs(result3[i] - expr3) > 1e-6f || fabs(result4[i] - expr4) > 1e-6f )

      passed = false;
  }

  return passed;
}

bool test152_unaryv_float3()
{
  const float3 Cx1( float(-1),  float(2),  float(-3));
  const float3 Cx2( float(3),  float(-4),  float(4));

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  float result1[3];
  float result2[3];
  float result3[3];
  float result4[3];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  float expr1[3], expr2[3], expr3[3], expr4[3];
  bool passed = true;
  for(int i=0;i<3;i++)
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
    PrintRR("exp1_res", "exp2_res", result1, expr1, 3);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 3); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 3);
    PrintRR("exp4_res", "exp4_res", result4, expr4, 3);
  }
  
  return passed;
}

bool test152_unaryk_float3()
{
  const float3 Cx1( float(-1),  float(2),  float(-3));
  const float Cx2 = float(3);

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  float result1[3];
  float result2[3];
  float result3[3];
  float result4[3];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  float expr1[3], expr2[3], expr3[3], expr4[3];
  bool passed = true;
  for(int i=0;i<3;i++)
  {
    expr1[i] = Cx1[i] + Cx2;
    expr2[i] = Cx1[i] - Cx2;
    expr3[i] = Cx1[i] * Cx2;
    expr4[i] = Cx1[i] / Cx2;
    

    if(fabs(result1[i] - expr1[i]) > 1e-6f || fabs(result2[i] - expr2[i]) > 1e-6f || fabs(result3[i] - expr3[i]) > 1e-6f) 

      passed = false;
  }

  if(!passed)
  {
    PrintRR("exp1_res", "exp2_res", result1, expr1, 3);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 3); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 3);
    PrintRR("exp4_res", "exp4_res", result4, expr4, 3);
  }
  
  return passed;
}

bool test153_cmpv_float3()
{
  const float3 Cx1( float(-1),  float(2),  float(-3));
  const float3 Cx2( float(3),  float(-4),  float(4));

  auto Cx3 = (Cx1 < Cx2 );
  auto Cx4 = (Cx1 > Cx2 );
  auto Cx5 = (Cx1 <= Cx2);
  auto Cx6 = (Cx1 >= Cx2);
  auto Cx7 = (Cx1 == Cx2);
  auto Cx8 = (Cx1 != Cx2);

  const auto Cx9  = blend(Cx1, Cx2, Cx3);
  const auto Cx10 = blend(Cx1, Cx2, Cx6);

  uint32_t result1[3];
  uint32_t result2[3];
  uint32_t result3[3];
  uint32_t result4[3];
  uint32_t result5[3];
  uint32_t result6[3];
  float result7[3];
  float result8[3];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  store_u(result5, Cx7);
  store_u(result6, Cx8);
  store_u(result7, Cx9);
  store_u(result8, Cx10);
  
  uint32_t expr1[3], expr2[3], expr3[3], expr4[3], expr5[3], expr6[3];
  float expr7[3],  expr8[3];
  bool passed = true;
  for(int i=0;i<3;i++)
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
    PrintRR("exp1_res", "exp2_res", result1, expr1, 3);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 3); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 3);
    PrintRR("exp4_res", "exp4_res", result4, expr4, 3);
    PrintRR("exp5_res", "exp5_res", result5, expr5, 3);
    PrintRR("exp6_res", "exp6_res", result6, expr6, 3);
    PrintRR("exp7_res", "exp7_res", result7, expr7, 3);
    PrintRR("exp8_res", "exp8_res", result8, expr8, 3);
  }
  
  return passed;
}

bool test154_shuffle_float3()
{ 
  const float3 Cx1( float(-1),  float(2),  float(-3));


  const float3 Cr1 = shuffle_zyx(Cx1);
  const float3 Cr2 = shuffle_zxy(Cx1);
  const float3 Cr3 = shuffle_yzx(Cx1);
  const float3 Cr4 = shuffle_yxz(Cx1);
  const float3 Cr5 = shuffle_xzy(Cx1);

  CVEX_ALIGNED(16) float result1[4];
  CVEX_ALIGNED(16) float result2[4];
  CVEX_ALIGNED(16) float result3[4];
  CVEX_ALIGNED(16) float result4[4];
  CVEX_ALIGNED(16) float result5[4];

  store(result1, Cr1);
  store(result2, Cr2);
  store(result3, Cr3);
  store(result4, Cr4);
  store(result5, Cr5);

  const bool b1 = (result1[0] == Cx1[2]) && (result1[1] == Cx1[1]) && (result1[2] == Cx1[0]);
  const bool b2 = (result2[0] == Cx1[2]) && (result2[1] == Cx1[0]) && (result2[2] == Cx1[1]);
  const bool b3 = (result3[0] == Cx1[1]) && (result3[1] == Cx1[2]) && (result3[2] == Cx1[0]);
  const bool b4 = (result4[0] == Cx1[1]) && (result4[1] == Cx1[0]) && (result4[2] == Cx1[2]);
  const bool b5 = (result5[0] == Cx1[0]) && (result5[1] == Cx1[2]) && (result5[2] == Cx1[1]);
  
  const bool passed = (b1 && b2 && b3 && b4 && b5);
  if(!passed)
  {
    std::cout << result1[0] << " " << result1[1] << " " << result1[2] << std::endl;
    std::cout << result2[0] << " " << result2[1] << " " << result2[2] << std::endl;
    std::cout << result3[0] << " " << result3[1] << " " << result3[2] << std::endl;
    std::cout << result4[0] << " " << result4[1] << " " << result4[2] << std::endl;
    std::cout << result5[0] << " " << result5[1] << " " << result5[2] << std::endl;
  }
  return passed;

}

bool test155_exsplat_float3()
{
  const float3 Cx1( float(-1),  float(2),  float(-3));

  const float3 Cr0 = splat_0(Cx1);
  const float3 Cr1 = splat_1(Cx1);
  const float3 Cr2 = splat_2(Cx1);

  const float s0 = extract_0(Cx1);
  const float s1 = extract_1(Cx1);
  const float s2 = extract_2(Cx1);

  float result0[3];
  float result1[3];
  float result2[3];

  store_u(result0, Cr0);
  store_u(result1, Cr1);
  store_u(result2, Cr2);
  
  bool passed = true;
  for (int i = 0; i<3; i++)
  {

    if((result0[i] != Cx1[0]))
      passed = false;
    if((result1[i] != Cx1[1]))
      passed = false;
    if((result2[i] != Cx1[2]))
      passed = false;
  }

  if(s0 != Cx1[0])
    passed = false;
  if(s1 != Cx1[1])
    passed = false;
  if(s2 != Cx1[2])
    passed = false;
  return passed;
}

bool test157_funcv_float3()
{
  const float3 Cx1( float(-1),  float(2),  float(-3));
  const float3 Cx2( float(3),  float(-4),  float(4));

  
  auto Cx3 = sign(Cx1);
  auto Cx4 = abs(Cx1);

  auto Cx5 = clamp(Cx1, float(2), float(3) );
  auto Cx6 = min(Cx1, Cx2);
  auto Cx7 = max(Cx1, Cx2);

  float Cm = hmin(Cx1);
  float CM = hmax(Cx1);
  float horMinRef = Cx1[0];
  float horMaxRef = Cx1[0];
  

  
  for(int i=0;i<3;i++)
  {
    horMinRef = std::min(horMinRef, Cx1[i]);
    horMaxRef = std::max(horMaxRef, Cx1[i]);

  }

  bool passed = true;
  for(int i=0;i<3;i++)
  {
  
    if(Cx3[i] != sign(Cx1[i]))
      passed = false;
    if(Cx4[i] != abs(Cx1[i]))
      passed = false;

    if(Cx5[i] != clamp(Cx1[i], float(2), float(3) ))
      passed = false;
    if(Cx6[i] != min(Cx1[i], Cx2[i]))
      passed = false;
    if(Cx7[i] != max(Cx1[i], Cx2[i]))
      passed = false;
  }

  if(horMinRef != Cm)
    passed = false;
  if(horMaxRef != CM)
    passed = false;


  return passed;
}




bool test158_funcfv_float3()
{
  const float3 Cx1( float(-1),  float(2),  float(-3));
  const float3 Cx2( float(3),  float(-4),  float(4));
  
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

  float ref[19][3];
  float res[19][3];
  memset(ref, 0, 19*sizeof(float)*3);
  memset(res, 0, 19*sizeof(float)*3);

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

  for(int i=0;i<3;i++)
  {
    ref[3][i] = mod(Cx1[i], Cx2[i]);
    ref[4][i] = fract(Cx1[i]);
    ref[5][i] = ceil(Cx1[i]);
    ref[6][i] = floor(Cx1[i]);
    ref[7][i] = sign(Cx1[i]);
    ref[8][i] = abs(Cx1[i]);

    ref[9][i]  = clamp(Cx1[i], float(-2), float(2) );
    ref[10][i] = min(Cx1[i], Cx2[i]);
    ref[11][i] = max(Cx1[i], Cx2[i]);

    ref[12][i] = mix (Cx1[i], Cx2[i], 0.5f);
    ref[13][i] = lerp(Cx1[i], Cx2[i], 0.5f);

    ref[14][i] = sqrt(Cx8[i]);
    ref[15][i] = inversesqrt(Cx8[i]);
    ref[18][i] = rcp(Cx1[i]);
  }
  
  bool passed = true;
  for(int i=0;i<3;i++)
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

bool test159_cstcnv_float3()
{
  const float3 Cx1( float(-1),  float(2),  float(-3));
  
  const int3  Cr1 = to_int32(Cx1);
  const uint3 Cr2 = to_uint32(Cx1);
  const int3  Cr3 = as_int32(Cx1);
  const uint3 Cr4 = as_uint32(Cx1);

  int          result1[3];
  unsigned int result2[3];
  int          result3[3];
  unsigned int result4[3];

  store_u(result1, Cr1);
  store_u(result2, Cr2);
  store_u(result3, Cr3);
  store_u(result4, Cr4);

  int          ref1[3];
  unsigned int ref2[3];
  for(int i=0;i<3;i++)
  {
    ref1[i] = int(Cx1[i]);
    ref2[i] = (unsigned int)(Cx1[i]); 
  }

  int          ref3[3];
  unsigned int ref4[3];

  memcpy(ref3, &Cr3, sizeof(int)*3);
  memcpy(ref4, &Cr4, sizeof(uint)*3);
  
  bool passed = true;
  for (int i=0; i<3; i++)
  {
    if (result1[i] != ref1[i] || result2[i] != ref2[i] || result3[i] != ref3[i] || result4[i] != ref4[i])
    {
      passed = false;
      break;
    }
  }
  return passed;
}




bool test160_other_float3() // dummy test
{
  const float CxData[3] = {  float(-1),  float(2),  float(-3)};
  const float3  Cx1(CxData);
  const float3  Cx2(float3(1));
 
  const float3  Cx3 = Cx1 + Cx2;
  float result1[3];
  float result2[3];
  float result3[3];
  store_u(result1, Cx1);
  store_u(result2, Cx2);
  store_u(result3, Cx3);

  bool passed = true;
  for (int i=0; i<3; i++)
  {

    if (fabs(result1[i] + float(1) - result3[i]) > 1e-10f || fabs(result2[i] - float(1) > 1e-10f) )

    {
      passed = false;
      break;
    }
  }


  const float  dat5 = dot  (Cx1, Cx2);

  const float3   crs3 = cross(Cx1, Cx2);
  const float crs_ref[3] = { Cx1[1]*Cx2[2] - Cx1[2]*Cx2[1], 
                                      Cx1[2]*Cx2[0] - Cx1[0]*Cx2[2], 
                                      Cx1[0]*Cx2[1] - Cx1[1]*Cx2[0] };



  {
    float sum = float(0);
    for(int i=0;i<3;i++)
      sum += Cx1[i]*Cx2[i];
    passed = passed && (std::abs(sum - dat5) < 1e-6f);

    for(int i=0;i<3;i++)
      passed = passed && (std::abs(crs3[i] - crs_ref[i]) < 1e-6f);

  }


  return passed;
}


