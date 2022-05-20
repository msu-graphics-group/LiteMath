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

bool test240_basev_int2()
{
  const int2 Cx1( int(1),  int(2));
  const int2 Cx2( int(5),  int(-5));

  const auto Cx3 = Cx1 - Cx2;
  const auto Cx4 = (Cx1 + Cx2)*Cx1;
  const auto Cx5 = (Cx2 - Cx1)/Cx1;

  int result1[2];
  int result2[2];
  int result3[2];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  
  int expr1[2], expr2[2], expr3[2];
  bool passed = true;
  for(int i=0;i<2;i++)
  {
    expr1[i] = Cx1[i] - Cx2[i];
    expr2[i] = (Cx1[i] + Cx2[i])*Cx1[i];
    expr3[i] = (Cx2[i] - Cx1[i])/Cx1[i];
    

    if(result1[i] != expr1[i] || result2[i] != expr2[i] || result3[i] != expr3[i]) 

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

bool test241_basek_int2()
{
  const int2 Cx1( int(1),  int(2));
  const int Cx2 = int(5);

  const int2 Cx3 = Cx2*(Cx2 + Cx1) - int(2);
  const int2 Cx4 = int(1) + (Cx1 + Cx2)*Cx2;
  
  const int2 Cx5 = int(3) - Cx2/(Cx2 - Cx1);
  const int2 Cx6 = (Cx2 + Cx1)/Cx2 + int(5)/Cx1;

  CVEX_ALIGNED(16) int result1[4]; 
  CVEX_ALIGNED(16) int result2[4];
  CVEX_ALIGNED(16) int result3[4];
  CVEX_ALIGNED(16) int result4[4];

  store(result1, Cx3);
  store(result2, Cx4);
  store(result3, Cx5);
  store(result4, Cx6);
  
  bool passed = true;
  for(int i=0;i<2;i++)
  {
    const int expr1 = Cx2*(Cx2 + Cx1[i]) - int(2);
    const int expr2 = int(1) + (Cx1[i] + Cx2)*Cx2;
    const int expr3 = int(3) - Cx2/(Cx2 - Cx1[i]);
    const int expr4 = (Cx2 + Cx1[i])/Cx2 + int(5)/Cx1[i];
    

    if(result1[i] != expr1 || result2[i] != expr2 || result3[i] != expr3 || result4[i] != expr4) 

      passed = false;
  }

  return passed;
}

bool test242_unaryv_int2()
{
  const int2 Cx1( int(1),  int(2));
  const int2 Cx2( int(5),  int(-5));

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  int result1[2];
  int result2[2];
  int result3[2];
  int result4[2];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  int expr1[2], expr2[2], expr3[2], expr4[2];
  bool passed = true;
  for(int i=0;i<2;i++)
  {
    expr1[i] = Cx1[i] + Cx2[i];
    expr2[i] = Cx1[i] - Cx2[i];
    expr3[i] = Cx1[i] * Cx2[i];
    expr4[i] = Cx1[i] / Cx2[i];
    

    if(result1[i] != expr1[i] || result2[i] != expr2[i] || result3[i] != expr3[i]) 

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

bool test242_unaryk_int2()
{
  const int2 Cx1( int(1),  int(2));
  const int Cx2 = int(5);

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  int result1[2];
  int result2[2];
  int result3[2];
  int result4[2];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  int expr1[2], expr2[2], expr3[2], expr4[2];
  bool passed = true;
  for(int i=0;i<2;i++)
  {
    expr1[i] = Cx1[i] + Cx2;
    expr2[i] = Cx1[i] - Cx2;
    expr3[i] = Cx1[i] * Cx2;
    expr4[i] = Cx1[i] / Cx2;
    

    if(result1[i] != expr1[i] || result2[i] != expr2[i] || result3[i] != expr3[i]) 
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

bool test243_cmpv_int2()
{
  const int2 Cx1( int(1),  int(2));
  const int2 Cx2( int(5),  int(-5));

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
  int result7[2];
  int result8[2];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  store_u(result5, Cx7);
  store_u(result6, Cx8);
  store_u(result7, Cx9);
  store_u(result8, Cx10);
  
  uint32_t expr1[2], expr2[2], expr3[2], expr4[2], expr5[2], expr6[2];
  int expr7[2],  expr8[2];
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

bool test244_shuffle_int2()
{ 
  const int2 Cx1( int(1),  int(2));


  return true;

}

bool test245_exsplat_int2()
{
  const int2 Cx1( int(1),  int(2));

  const int2 Cr0 = splat_0(Cx1);
  const int2 Cr1 = splat_1(Cx1);

  const int s0 = extract_0(Cx1);
  const int s1 = extract_1(Cx1);

  int result0[2];
  int result1[2];

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

bool test247_funcv_int2()
{
  const int2 Cx1( int(1),  int(2));
  const int2 Cx2( int(5),  int(-5));
  const int2 Cx9( int(2),  int(2));
  const int2 Cx0( int(0),  int(0));

  
  auto Cx3 = sign(Cx1);
  auto Cx4 = abs(Cx1);

  auto Cx5 = clamp(Cx1, int(2), int(3) );
  auto Cx6 = min(Cx1, Cx2);
  auto Cx7 = max(Cx1, Cx2);
  auto Cx8 = clamp(Cx1, Cx0, Cx9);

  int Cm = hmin(Cx1);
  int CM = hmax(Cx1);
  int horMinRef = Cx1[0];
  int horMaxRef = Cx1[0];
  

  
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

    if(Cx5[i] != clamp(Cx1[i], int(2), int(3) ))
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




bool test248_logicv_int2()
{
  const int2 Cx1( int(1),  int(2));
  const int2 Cx2( int(5),  int(-5));
  const int2 Cx3( int(4294967295),  int(4042260480));

  const auto Cr0 = (Cx1 & (~Cx3)) | Cx2;
  const auto Cr1 = (Cx2 & Cx3)    | Cx1;
  const auto Cr2 = (Cx1 << 8); 
  const auto Cr3 = (Cx3 >> 9); 
  const auto Cr4 = (Cx1 << 8) | (Cx2 >> 17); 
  const auto Cr5 = (Cx3 << 9) | (Cx3 >> 4); 

  int ref[6][2];
  int res[6][2];
  store_u(res[0],  Cr0);
  store_u(res[1],  Cr1);
  store_u(res[2],  Cr2);
  store_u(res[3],  Cr3);
  store_u(res[4],  Cr4);
  store_u(res[5],  Cr5);
  
  for(int i=0;i<2;i++)
  {
    ref[0][i] = (Cx1[i] & (~Cx3[i])) | Cx2[i];
    ref[1][i] = (Cx2[i] & Cx3[i])    | Cx1[i];
    ref[2][i] = (Cx1[i] << 8); 
    ref[3][i] = (Cx3[i] >> 9); 
    ref[4][i] = (Cx1[i] << 8) | (Cx2[i] >> 17); 
    ref[5][i] = (Cx3[i] << 9) | (Cx3[i] >> 4); 
  }
  
  bool passed = true;
  for(int i=0;i<2;i++)
  {
    for(int j=0;j<=5;j++)
      if(res[j][i] != ref[j][i])
        passed = false;
  }
  return passed;
}

bool test249_cstcnv_int2()
{
  const int2 Cx1( int(1),  int(2));
  
  const float2 Cr1 = to_float32(Cx1);
  const float2 Cr2 = as_float32(Cx1);

  float result1[4];
  float result2[4];
  store_u(result1, Cr1);
  store_u(result2, Cr2);

  float ref1[2];
  for(int i=0;i<2;i++)
    ref1[i] = float(Cx1[i]);
  float ref2[2];
  memcpy(ref2, &Cx1, sizeof(float)*2);
  
  bool passed = true;
  for (int i=0; i<2; i++)
  {
    if (result1[i] != ref1[i] || memcmp(result2, ref2, sizeof(int)*2) != 0)
    {
      passed = false;
      break;
    }
  }
  return passed;
}



bool test250_other_int2() // dummy test
{
  const int CxData[2] = {  int(1),  int(2)};
  const int2  Cx1(CxData);
  const int2  Cx2(int2(1));
 
  const int2  Cx3 = Cx1 + Cx2;
  int result1[2];
  int result2[2];
  int result3[2];
  store_u(result1, Cx1);
  store_u(result2, Cx2);
  store_u(result3, Cx3);

  bool passed = true;
  for (int i=0; i<2; i++)
  {

    if (result1[i] + int(1) != result3[i] || result2[i] != int(1))

    {
      passed = false;
      break;
    }
  }


  const int  dat5 = dot  (Cx1, Cx2);



  {
    int sum = int(0);
    for(int i=0;i<2;i++)
      sum += Cx1[i]*Cx2[i];
    passed = passed && (sum == dat5);

  }


  return passed;
}

bool test251_any_all_int2() // dummy test
{
  const int CxData[2] = {  int(1),  int(2)};
  const int2  Cx1(CxData);
  const int2  Cx2(int2(1));
 
  const int2  Cx3 = Cx1 + Cx2;

  const bool a1 = all_of(Cx1 < Cx3);
  const bool a2 = all_of(Cx1 < Cx2);
  const bool a3 = any_of(Cx1 <= Cx2);
  const bool a4 = any_of(Cx1 > Cx3);

  return a1 && !a2 && a3 && !a4;
}


