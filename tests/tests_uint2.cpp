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

bool test160_basev_uint2()
{
  const uint2 Cx1( uint(1),  uint(2));
  const uint2 Cx2( uint(5),  uint(4294967291));

  const auto Cx3 = Cx1 - Cx2;
  const auto Cx4 = (Cx1 + Cx2)*Cx1;
  const auto Cx5 = (Cx2 - Cx1)/Cx1;

  uint result1[2];
  uint result2[2];
  uint result3[2];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  
  uint expr1[2], expr2[2], expr3[2];
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

bool test161_basek_uint2()
{
  const uint2 Cx1( uint(1),  uint(2));
  const uint Cx2 = uint(5);

  const uint2 Cx3 = Cx2*(Cx2 + Cx1) - uint(2);
  const uint2 Cx4 = uint(1) + (Cx1 + Cx2)*Cx2;
  
  const uint2 Cx5 = uint(3) - Cx2/(Cx2 - Cx1);
  const uint2 Cx6 = (Cx2 + Cx1)/Cx2 + uint(5)/Cx1;

  CVEX_ALIGNED(16) uint result1[4]; 
  CVEX_ALIGNED(16) uint result2[4];
  CVEX_ALIGNED(16) uint result3[4];
  CVEX_ALIGNED(16) uint result4[4];

  store(result1, Cx3);
  store(result2, Cx4);
  store(result3, Cx5);
  store(result4, Cx6);
  
  bool passed = true;
  for(int i=0;i<2;i++)
  {
    const uint expr1 = Cx2*(Cx2 + Cx1[i]) - uint(2);
    const uint expr2 = uint(1) + (Cx1[i] + Cx2)*Cx2;
    const uint expr3 = uint(3) - Cx2/(Cx2 - Cx1[i]);
    const uint expr4 = (Cx2 + Cx1[i])/Cx2 + uint(5)/Cx1[i];
    

    if(result1[i] != expr1 || result2[i] != expr2 || result3[i] != expr3 || result4[i] != expr4) 

      passed = false;
  }

  return passed;
}

bool test162_unaryv_uint2()
{
  const uint2 Cx1( uint(1),  uint(2));
  const uint2 Cx2( uint(5),  uint(4294967291));

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  uint result1[2];
  uint result2[2];
  uint result3[2];
  uint result4[2];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  uint expr1[2], expr2[2], expr3[2], expr4[2];
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

bool test162_unaryk_uint2()
{
  const uint2 Cx1( uint(1),  uint(2));
  const uint Cx2 = uint(5);

  auto Cx3 = Cx1;
  auto Cx4 = Cx1;
  auto Cx5 = Cx1;
  auto Cx6 = Cx1;

  Cx3 += Cx2;
  Cx4 -= Cx2;
  Cx5 *= Cx2;
  Cx6 /= Cx2;

  uint result1[2];
  uint result2[2];
  uint result3[2];
  uint result4[2];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  
  uint expr1[2], expr2[2], expr3[2], expr4[2];
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

bool test163_cmpv_uint2()
{
  const uint2 Cx1( uint(1),  uint(2));
  const uint2 Cx2( uint(5),  uint(4294967291));

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
  uint result7[2];
  uint result8[2];

  store_u(result1, Cx3);
  store_u(result2, Cx4);
  store_u(result3, Cx5);
  store_u(result4, Cx6);
  store_u(result5, Cx7);
  store_u(result6, Cx8);
  store_u(result7, Cx9);
  store_u(result8, Cx10);
  
  uint32_t expr1[2], expr2[2], expr3[2], expr4[2], expr5[2], expr6[2];
  uint expr7[2],  expr8[2];
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

bool test164_shuffle_uint2()
{ 
  const uint2 Cx1( uint(1),  uint(2));


  return true;

}

bool test165_exsplat_uint2()
{
  const uint2 Cx1( uint(1),  uint(2));

  const uint2 Cr0 = splat_0(Cx1);
  const uint2 Cr1 = splat_1(Cx1);

  const uint s0 = extract_0(Cx1);
  const uint s1 = extract_1(Cx1);

  uint result0[2];
  uint result1[2];

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

bool test167_funcv_uint2()
{
  const uint2 Cx1( uint(1),  uint(2));
  const uint2 Cx2( uint(5),  uint(4294967291));
  const uint2 Cx9( uint(2),  uint(2));
  const uint2 Cx0( uint(0),  uint(0));


  auto Cx5 = clamp(Cx1, uint(2), uint(3) );
  auto Cx6 = min(Cx1, Cx2);
  auto Cx7 = max(Cx1, Cx2);
  auto Cx8 = clamp(Cx1, Cx0, Cx9);

  uint Cm = hmin(Cx1);
  uint CM = hmax(Cx1);
  uint horMinRef = Cx1[0];
  uint horMaxRef = Cx1[0];
  

  
  for(int i=0;i<2;i++)
  {
    horMinRef = std::min(horMinRef, Cx1[i]);
    horMaxRef = std::max(horMaxRef, Cx1[i]);

  }

  bool passed = true;
  for(int i=0;i<2;i++)
  {

    if(Cx5[i] != clamp(Cx1[i], uint(2), uint(3) ))
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




bool test168_logicv_uint2()
{
  const uint2 Cx1( uint(1),  uint(2));
  const uint2 Cx2( uint(5),  uint(4294967291));
  const uint2 Cx3( uint(4294967295),  uint(4042260480));

  const auto Cr0 = (Cx1 & (~Cx3)) | Cx2;
  const auto Cr1 = (Cx2 & Cx3)    | Cx1;
  const auto Cr2 = (Cx1 << 8); 
  const auto Cr3 = (Cx3 >> 9); 
  const auto Cr4 = (Cx1 << 8) | (Cx2 >> 17); 
  const auto Cr5 = (Cx3 << 9) | (Cx3 >> 4); 

  uint ref[6][2];
  uint res[6][2];
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

bool test169_cstcnv_uint2()
{
  const uint2 Cx1( uint(1),  uint(2));
  
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
    if (result1[i] != ref1[i] || memcmp(result2, ref2, sizeof(uint)*2) != 0)
    {
      passed = false;
      break;
    }
  }
  return passed;
}



bool test170_other_uint2() // dummy test
{
  const uint CxData[2] = {  uint(1),  uint(2)};
  const uint2  Cx1(CxData);
  const uint2  Cx2(uint2(1));
 
  const uint2  Cx3 = Cx1 + Cx2;
  uint result1[2];
  uint result2[2];
  uint result3[2];
  store_u(result1, Cx1);
  store_u(result2, Cx2);
  store_u(result3, Cx3);

  bool passed = true;
  for (int i=0; i<2; i++)
  {

    if (result1[i] + uint(1) != result3[i] || result2[i] != uint(1))

    {
      passed = false;
      break;
    }
  }


  const uint  dat5 = dot  (Cx1, Cx2);



  {
    uint sum = uint(0);
    for(int i=0;i<2;i++)
      sum += Cx1[i]*Cx2[i];
    passed = passed && (sum == dat5);

  }


  return passed;
}


