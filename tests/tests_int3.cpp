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

bool test140_basev_int3()
{
  const int3 Cx1( int(1),  int(2),  int(-3));
  const int3 Cx2( int(5),  int(-5),  int(6));

  const auto Cx3 = Cx1 - Cx2;
  const auto Cx4 = (Cx1 + Cx2)*Cx1;
  const auto Cx5 = (Cx2 - Cx1)/Cx1;

  int result1[3];
  int result2[3];
  int result3[3];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  
  int expr1[3], expr2[3], expr3[3];
  bool passed = true;
  for(int i=0;i<3;i++)
  {
    expr1[i] = Cx1[i] - Cx2[i];
    expr2[i] = (Cx1[i] + Cx2[i])*Cx1[i];
    expr3[i] = (Cx2[i] - Cx1[i])/Cx1[i];
    

    if(result1[i] != expr1[i] || result2[i] != expr2[i] || result3[i] != expr3[i]) 

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

bool test141_basek_int3()
{
  const int3 Cx1( int(1),  int(2),  int(-3));
  const int Cx2 = int(5);

  const int3 Cx3 = Cx2*(Cx2 + Cx1) - int(2);
  const int3 Cx4 = int(1) + (Cx1 + Cx2)*Cx2;
  
  const int3 Cx5 = int(3) - Cx2/(Cx2 - Cx1);
  const int3 Cx6 = (Cx2 + Cx1)/Cx2 + int(5)/Cx1;

  CVEX_ALIGNED(16) int result1[4]; 
  CVEX_ALIGNED(16) int result2[4];
  CVEX_ALIGNED(16) int result3[4];
  CVEX_ALIGNED(16) int result4[4];

  store(result1, Cx3);
  store(result2, Cx4);
  store(result3, Cx5);
  store(result4, Cx6);
  
  bool passed = true;
  for(int i=0;i<3;i++)
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

bool test142_unaryv_int3()
{
  const int3 Cx1( int(1),  int(2),  int(-3));
  const int3 Cx2( int(5),  int(-5),  int(6));

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  int result1[3];
  int result2[3];
  int result3[3];
  int result4[3];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  int expr1[3], expr2[3], expr3[3], expr4[3];
  bool passed = true;
  for(int i=0;i<3;i++)
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
    PrintRR("exp1_res", "exp2_res", result1, expr1, 3);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 3); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 3);
    PrintRR("exp4_res", "exp4_res", result4, expr4, 3);
  }
  
  return passed;
}

bool test142_unaryk_int3()
{
  const int3 Cx1( int(1),  int(2),  int(-3));
  const int Cx2 = int(5);

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  int result1[3];
  int result2[3];
  int result3[3];
  int result4[3];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  int expr1[3], expr2[3], expr3[3], expr4[3];
  bool passed = true;
  for(int i=0;i<3;i++)
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
    PrintRR("exp1_res", "exp2_res", result1, expr1, 3);
    PrintRR("exp2_res", "exp2_res", result2, expr2, 3); 
    PrintRR("exp3_res", "exp3_res", result3, expr3, 3);
    PrintRR("exp4_res", "exp4_res", result4, expr4, 3);
  }
  
  return passed;
}

bool test143_cmpv_int3()
{
  const int3 Cx1( int(1),  int(2),  int(-3));
  const int3 Cx2( int(5),  int(-5),  int(6));

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
  int result7[3];
  int result8[3];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  store_u(result5, Cx7);
  store_u(result6, Cx8);
  store_u(result7, Cx9);
  store_u(result8, Cx10);
  
  uint32_t expr1[3], expr2[3], expr3[3], expr4[3], expr5[3], expr6[3];
  int expr7[3],  expr8[3];
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

bool test144_shuffle_int3()
{ 
  const int3 Cx1( int(1),  int(2),  int(-3));


  const int3 Cr1 = shuffle_zyx(Cx1);
  const int3 Cr2 = shuffle_zxy(Cx1);
  const int3 Cr3 = shuffle_yzx(Cx1);
  const int3 Cr4 = shuffle_yxz(Cx1);
  const int3 Cr5 = shuffle_xzy(Cx1);

  CVEX_ALIGNED(16) int result1[4];
  CVEX_ALIGNED(16) int result2[4];
  CVEX_ALIGNED(16) int result3[4];
  CVEX_ALIGNED(16) int result4[4];
  CVEX_ALIGNED(16) int result5[4];

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

bool test145_exsplat_int3()
{
  const int3 Cx1( int(1),  int(2),  int(-3));

  const int3 Cr0 = splat_0(Cx1);
  const int3 Cr1 = splat_1(Cx1);
  const int3 Cr2 = splat_2(Cx1);

  const int s0 = extract_0(Cx1);
  const int s1 = extract_1(Cx1);
  const int s2 = extract_2(Cx1);

  int result0[3];
  int result1[3];
  int result2[3];

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

bool test147_funcv_int3()
{
  const int3 Cx1( int(1),  int(2),  int(-3));
  const int3 Cx2( int(5),  int(-5),  int(6));

  
  auto Cx3 = sign(Cx1);
  auto Cx4 = abs(Cx1);

  auto Cx5 = clamp(Cx1, int(2), int(3) );
  auto Cx6 = min(Cx1, Cx2);
  auto Cx7 = max(Cx1, Cx2);

  int Cm = hmin(Cx1);
  int CM = hmax(Cx1);
  int horMinRef = Cx1[0];
  int horMaxRef = Cx1[0];
  

  
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

    if(Cx5[i] != clamp(Cx1[i], int(2), int(3) ))
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




bool test148_logicv_int3()
{
  const int3 Cx1( int(1),  int(2),  int(-3));
  const int3 Cx2( int(5),  int(-5),  int(6));
  const int3 Cx3( int(4294967295),  int(4294967295),  int(4042260480));

  const auto Cr0 = (Cx1 & (~Cx3)) | Cx2;
  const auto Cr1 = (Cx2 & Cx3)    | Cx1;
  const auto Cr2 = (Cx1 << 8); 
  const auto Cr3 = (Cx3 >> 9); 
  const auto Cr4 = (Cx1 << 8) | (Cx2 >> 17); 
  const auto Cr5 = (Cx3 << 9) | (Cx3 >> 4); 

  int ref[6][3];
  int res[6][3];
  store_u(res[0],  Cr0);
  store_u(res[1],  Cr1);
  store_u(res[2],  Cr2);
  store_u(res[3],  Cr3);
  store_u(res[4],  Cr4);
  store_u(res[5],  Cr5);
  
  for(int i=0;i<3;i++)
  {
    ref[0][i] = (Cx1[i] & (~Cx3[i])) | Cx2[i];
    ref[1][i] = (Cx2[i] & Cx3[i])    | Cx1[i];
    ref[2][i] = (Cx1[i] << 8); 
    ref[3][i] = (Cx3[i] >> 9); 
    ref[4][i] = (Cx1[i] << 8) | (Cx2[i] >> 17); 
    ref[5][i] = (Cx3[i] << 9) | (Cx3[i] >> 4); 
  }
  
  bool passed = true;
  for(int i=0;i<3;i++)
  {
    for(int j=0;j<=5;j++)
      if(res[j][i] != ref[j][i])
        passed = false;
  }
  return passed;
}

bool test149_cstcnv_int3()
{
  const int3 Cx1( int(1),  int(2),  int(-3));
  
  const float3 Cr1 = to_float32(Cx1);
  const float3 Cr2 = as_float32(Cx1);

  float result1[4];
  float result2[4];
  store_u(result1, Cr1);
  store_u(result2, Cr2);

  float ref1[3];
  for(int i=0;i<3;i++)
    ref1[i] = float(Cx1[i]);
  float ref2[3];
  memcpy(ref2, &Cx1, sizeof(float)*3);
  
  bool passed = true;
  for (int i=0; i<3; i++)
  {
    if (result1[i] != ref1[i] || memcmp(result2, ref2, sizeof(int)*3) != 0)
    {
      passed = false;
      break;
    }
  }
  return passed;
}



bool test150_other_int3() // dummy test
{
  return true;
}


