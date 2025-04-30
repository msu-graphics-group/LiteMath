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

bool test320_basev_double2()
{
  const double2 Cx1( double(-1),  double(2));
  const double2 Cx2( double(3),  double(-4));

  const auto Cx3 = Cx1 - Cx2;
  const auto Cx4 = (Cx1 + Cx2)*Cx1;
  const auto Cx5 = (Cx2 - Cx1)/Cx1;

  double result1[2];
  double result2[2];
  double result3[2];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  
  double expr1[2], expr2[2], expr3[2];
  bool passed = true;
  for(int i=0;i<2;i++)
  {
    expr1[i] = Cx1[i] - Cx2[i];
    expr2[i] = (Cx1[i] + Cx2[i])*Cx1[i];
    expr3[i] = (Cx2[i] - Cx1[i])/Cx1[i];
    

    if(fabs(result1[i] - expr1[i]) > 1e-6f || fabs(result2[i] - expr2[i]) > 1e-6f || fabs(result3[i] - expr3[i]) > 1e-6f) 

      passed = false;
  }

  if(!passed)
  {
    PrintRR("exp1_res", "exp2_res", result1, expr1, 2);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 2); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 2);
  }
  
  return passed;
}

bool test321_basek_double2()
{
  const double2 Cx1( double(-1),  double(2));
  const double Cx2 = double(3);

  const double2 Cx3 = Cx2*(Cx2 + Cx1) - double(2);
  const double2 Cx4 = double(1) + (Cx1 + Cx2)*Cx2;
  
  const double2 Cx5 = double(3) - Cx2/(Cx2 - Cx1);
  const double2 Cx6 = (Cx2 + Cx1)/Cx2 + double(5)/Cx1;

  CVEX_ALIGNED(16) double result1[4]; 
  CVEX_ALIGNED(16) double result2[4];
  CVEX_ALIGNED(16) double result3[4];
  CVEX_ALIGNED(16) double result4[4];

  store(result1, Cx3);
  store(result2, Cx4);
  store(result3, Cx5);
  store(result4, Cx6);
  
  bool passed = true;
  for(int i=0;i<2;i++)
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

bool test322_unaryv_double2()
{
  const double2 Cx1( double(-1),  double(2));
  const double2 Cx2( double(3),  double(-4));

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  double result1[2];
  double result2[2];
  double result3[2];
  double result4[2];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  double expr1[2], expr2[2], expr3[2], expr4[2];
  bool passed = true;
  for(int i=0;i<2;i++)
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
    PrintRR("exp1_res", "exp2_res", result1, expr1, 2);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 2); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 2);
    PrintRR("exp4_res", "exp4_res", result4, expr4, 2);
  }
  
  return passed;
}

bool test322_unaryk_double2()
{
  const double2 Cx1( double(-1),  double(2));
  const double Cx2 = double(3);

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  double result1[2];
  double result2[2];
  double result3[2];
  double result4[2];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  double expr1[2], expr2[2], expr3[2], expr4[2];
  bool passed = true;
  for(int i=0;i<2;i++)
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
    PrintRR("exp1_res", "exp2_res", result1, expr1, 2);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 2); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 2);
    PrintRR("exp4_res", "exp4_res", result4, expr4, 2);
  }
  
  return passed;
}

bool test323_cmpv_double2()
{
  const double2 Cx1( double(-1),  double(2));
  const double2 Cx2( double(3),  double(-4));

  auto Cx3 = (Cx1 < Cx2 );
  auto Cx4 = (Cx1 > Cx2 );
  auto Cx5 = (Cx1 <= Cx2);
  auto Cx6 = (Cx1 >= Cx2);
  auto Cx7 = (Cx1 == Cx2);
  auto Cx8 = (Cx1 != Cx2);

  const auto Cx9  = blend(Cx1, Cx2, Cx3);
  const auto Cx10 = blend(Cx1, Cx2, Cx6);

  uint32_t result1[2];
  uint32_t result2[2];
  uint32_t result3[2];
  uint32_t result4[2];
  uint32_t result5[2];
  uint32_t result6[2];
  double result7[2];
  double result8[2];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  store_u(result5, Cx7);
  store_u(result6, Cx8);
  store_u(result7, Cx9);
  store_u(result8, Cx10);
  
  uint32_t expr1[2], expr2[2], expr3[2], expr4[2], expr5[2], expr6[2];
  double expr7[2],  expr8[2];
  bool passed = true;
  for(int i=0;i<2;i++)
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
    PrintRR("exp1_res", "exp2_res", result1, expr1, 2);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 2); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 2);
    PrintRR("exp4_res", "exp4_res", result4, expr4, 2);
    PrintRR("exp5_res", "exp5_res", result5, expr5, 2);
    PrintRR("exp6_res", "exp6_res", result6, expr6, 2);
    PrintRR("exp7_res", "exp7_res", result7, expr7, 2);
    PrintRR("exp8_res", "exp8_res", result8, expr8, 2);
  }
  
  return passed;
}

bool test324_shuffle_double2()
{ 
  const double2 Cx1( double(-1),  double(2));


  return true;

}

bool test325_exsplat_double2()
{
  const double2 Cx1( double(-1),  double(2));

  const double2 Cr0 = splat_0(Cx1);
  const double2 Cr1 = splat_1(Cx1);

  const double s0 = extract_0(Cx1);
  const double s1 = extract_1(Cx1);

  double result0[2];
  double result1[2];

  store_u(result0, Cr0);
  store_u(result1, Cr1);
  
  bool passed = true;
  for (int i = 0; i<2; i++)
  {

    if((result0[i] != Cx1[0]))
      passed = false;
    if((result1[i] != Cx1[1]))
      passed = false;
  }

  if(s0 != Cx1[0])
    passed = false;
  if(s1 != Cx1[1])
    passed = false;
  return passed;
}

bool test327_funcv_double2()
{
  const double2 Cx1( double(-1),  double(2));
  const double2 Cx2( double(3),  double(-4));
  const double2 Cx9( double(2),  double(2));
  const double2 Cx0( double(0),  double(0));

  
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
  

  
  for(int i=0;i<2;i++)
  {
    horMinRef = std::min(horMinRef, Cx1[i]);
    horMaxRef = std::max(horMaxRef, Cx1[i]);

  }

  bool passed = true;
  for(int i=0;i<2;i++)
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


  return passed;
}




bool test328_funcfv_double2()
{
  const double2 Cx1( double(-1),  double(2));
  const double2 Cx2( double(3),  double(-4));
  
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

  double ref[19][2];
  double res[19][2];
  memset(ref, 0, 19*sizeof(double)*2);
  memset(res, 0, 19*sizeof(double)*2);

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

  for(int i=0;i<2;i++)
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
  for(int i=0;i<2;i++)
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

bool test329_cstcnv_double2()
{
  const double2 Cx1( double(-1),  double(2));
  
  const int2  Cr1 = to_int32(Cx1);
  const uint2 Cr2 = to_uint32(Cx1);
  const int2  Cr3 = as_int32(Cx1);
  const uint2 Cr4 = as_uint32(Cx1);

  int          result1[2];
  unsigned int result2[2];
  int          result3[2];
  unsigned int result4[2];

  store_u(result1, Cr1);
  store_u(result2, Cr2);
  store_u(result3, Cr3);
  store_u(result4, Cr4);

  int          ref1[2];
  unsigned int ref2[2];
  for(int i=0;i<2;i++)
  {
    ref1[i] = int(Cx1[i]);
    ref2[i] = (unsigned int)(Cx1[i]); 
  }

  int          ref3[2];
  unsigned int ref4[2];

  memcpy(ref3, &Cr3, sizeof(int)*2);
  memcpy(ref4, &Cr4, sizeof(uint)*2);
  
  bool passed = true;
  for (int i=0; i<2; i++)
  {
    if (result1[i] != ref1[i] || result2[i] != ref2[i] || result3[i] != ref3[i] || result4[i] != ref4[i])
    {
      passed = false;
      break;
    }
  }
  return passed;
}




bool test330_other_double2() // dummy test
{
  const double CxData[2] = {  double(-1),  double(2)};
  const double2  Cx1(CxData);
  const double2  Cx2(double2(1));
 
  const double2  Cx3 = Cx1 + Cx2;
  double result1[2];
  double result2[2];
  double result3[2];
  store_u(result1, Cx1);
  store_u(result2, Cx2);
  store_u(result3, Cx3);

  bool passed = true;
  for (int i=0; i<2; i++)
  {

    if (std::abs(result1[i] + double(1) - result3[i]) > 1e-10f || std::abs(result2[i] - double(1) > 1e-10f) )

    {
      passed = false;
      break;
    }
  }


  const double  dat5 = dot  (Cx1, Cx2);



  {
    double sum = double(0);
    for(int i=0;i<2;i++)
      sum += Cx1[i]*Cx2[i];
    passed = passed && (std::abs(sum - dat5) < 1e-6f);

  }


  return passed;
}

bool test331_any_all_double2() // dummy test
{
  const double CxData[2] = {  double(1),  double(2)};
  const double2  Cx1(CxData);
  const double2  Cx2(double2(1));
 
  const double2  Cx3 = Cx1 + Cx2;
  

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


