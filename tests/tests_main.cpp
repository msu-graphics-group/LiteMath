#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include "LiteMath.h"
#include "tests/tests.h"

using TestFuncType = bool (*)();
void tests_all_images();

struct TestRun
{
  TestFuncType pTest;
  const char*  pTestName;
};

int main(int argc, const char** argv)
{ 
  if(argc > 1)
  {
    for(int i=0;i<argc;i++)
      std::cout << argv[i] << " ";
    std::cout << std::endl;
  }

  TestRun tests[] = { {test000_scalar_funcs,  "test000_scalar_funcs"},
                      {test001_dot_cross_f4,  "test001_dot_cross_f4"},
                      {test002_dot_cross_f3,  "test002_dot_cross_f3"},
                      {test003_length_float4, "test003_length_float4"},
                      {test004_colpack_f4x4,  "test004_colpack_f4x4"},
                      {test005_matrix_elems,  "test005_matrix_elems"},
                      {test006_any_all,       "test006_any_all"},
                      {test007_reflect,       "test007_reflect"},
                      {test008_normalize,     "test008_normalize"},
                      {test009_refract,       "test009_refract"},
                      {test010_faceforward,   "test010_faceforward"},
                      {test011_mattranspose,  "test011_mattranspose"},


                      {test100_basev_uint4,         "test100_basev_uint4"},
                      {test101_basek_uint4,         "test101_basek_uint4"},
                      {test102_unaryv_uint4,        "test102_unaryv_uint4"},
                      {test102_unaryk_uint4,        "test102_unaryk_uint4"}, 
                      {test103_cmpv_uint4,          "test103_cmpv_uint4"}, 
                      {test104_shuffle_uint4,       "test104_shuffle_uint4"},
                      {test105_exsplat_uint4,       "test105_exsplat_uint4"},
                      {test107_funcv_uint4,         "test107_funcv_uint4"},

                      {test108_logicv_uint4,        "test108_logicv_uint4"},
                      {test109_cstcnv_uint4,        "test109_cstcnv_uint4"},

                      {test110_other_uint4,        "test110_other_uint4"},
                      {test111_any_all_uint4,      "test111_any_all_uint4"},



                      {test120_basev_int4,         "test120_basev_int4"},
                      {test121_basek_int4,         "test121_basek_int4"},
                      {test122_unaryv_int4,        "test122_unaryv_int4"},
                      {test122_unaryk_int4,        "test122_unaryk_int4"}, 
                      {test123_cmpv_int4,          "test123_cmpv_int4"}, 
                      {test124_shuffle_int4,       "test124_shuffle_int4"},
                      {test125_exsplat_int4,       "test125_exsplat_int4"},
                      {test127_funcv_int4,         "test127_funcv_int4"},

                      {test128_logicv_int4,        "test128_logicv_int4"},
                      {test129_cstcnv_int4,        "test129_cstcnv_int4"},

                      {test130_other_int4,        "test130_other_int4"},
                      {test131_any_all_int4,      "test131_any_all_int4"},



                      {test140_basev_float4,         "test140_basev_float4"},
                      {test141_basek_float4,         "test141_basek_float4"},
                      {test142_unaryv_float4,        "test142_unaryv_float4"},
                      {test142_unaryk_float4,        "test142_unaryk_float4"}, 
                      {test143_cmpv_float4,          "test143_cmpv_float4"}, 
                      {test144_shuffle_float4,       "test144_shuffle_float4"},
                      {test145_exsplat_float4,       "test145_exsplat_float4"},
                      {test147_funcv_float4,         "test147_funcv_float4"},

                      {test148_funcfv_float4,        "test148_funcfv_float4"},
                      {test149_cstcnv_float4,        "test149_cstcnv_float4"},

                      {test150_other_float4,        "test150_other_float4"},
                      {test151_any_all_float4,      "test151_any_all_float4"},



                      {test160_basev_uint3,         "test160_basev_uint3"},
                      {test161_basek_uint3,         "test161_basek_uint3"},
                      {test162_unaryv_uint3,        "test162_unaryv_uint3"},
                      {test162_unaryk_uint3,        "test162_unaryk_uint3"}, 
                      {test163_cmpv_uint3,          "test163_cmpv_uint3"}, 
                      {test164_shuffle_uint3,       "test164_shuffle_uint3"},
                      {test165_exsplat_uint3,       "test165_exsplat_uint3"},
                      {test167_funcv_uint3,         "test167_funcv_uint3"},

                      {test168_logicv_uint3,        "test168_logicv_uint3"},
                      {test169_cstcnv_uint3,        "test169_cstcnv_uint3"},

                      {test170_other_uint3,        "test170_other_uint3"},
                      {test171_any_all_uint3,      "test171_any_all_uint3"},



                      {test180_basev_int3,         "test180_basev_int3"},
                      {test181_basek_int3,         "test181_basek_int3"},
                      {test182_unaryv_int3,        "test182_unaryv_int3"},
                      {test182_unaryk_int3,        "test182_unaryk_int3"}, 
                      {test183_cmpv_int3,          "test183_cmpv_int3"}, 
                      {test184_shuffle_int3,       "test184_shuffle_int3"},
                      {test185_exsplat_int3,       "test185_exsplat_int3"},
                      {test187_funcv_int3,         "test187_funcv_int3"},

                      {test188_logicv_int3,        "test188_logicv_int3"},
                      {test189_cstcnv_int3,        "test189_cstcnv_int3"},

                      {test190_other_int3,        "test190_other_int3"},
                      {test191_any_all_int3,      "test191_any_all_int3"},



                      {test200_basev_float3,         "test200_basev_float3"},
                      {test201_basek_float3,         "test201_basek_float3"},
                      {test202_unaryv_float3,        "test202_unaryv_float3"},
                      {test202_unaryk_float3,        "test202_unaryk_float3"}, 
                      {test203_cmpv_float3,          "test203_cmpv_float3"}, 
                      {test204_shuffle_float3,       "test204_shuffle_float3"},
                      {test205_exsplat_float3,       "test205_exsplat_float3"},
                      {test207_funcv_float3,         "test207_funcv_float3"},

                      {test208_funcfv_float3,        "test208_funcfv_float3"},
                      {test209_cstcnv_float3,        "test209_cstcnv_float3"},

                      {test210_other_float3,        "test210_other_float3"},
                      {test211_any_all_float3,      "test211_any_all_float3"},



                      {test220_basev_double3,         "test220_basev_double3"},
                      {test221_basek_double3,         "test221_basek_double3"},
                      {test222_unaryv_double3,        "test222_unaryv_double3"},
                      {test222_unaryk_double3,        "test222_unaryk_double3"}, 
                      {test223_cmpv_double3,          "test223_cmpv_double3"}, 
                      {test224_shuffle_double3,       "test224_shuffle_double3"},
                      {test225_exsplat_double3,       "test225_exsplat_double3"},
                      {test227_funcv_double3,         "test227_funcv_double3"},

                      {test228_funcfv_double3,        "test228_funcfv_double3"},
                      {test229_cstcnv_double3,        "test229_cstcnv_double3"},

                      {test230_other_double3,        "test230_other_double3"},
                      {test231_any_all_double3,      "test231_any_all_double3"},



                      {test240_basev_uint2,         "test240_basev_uint2"},
                      {test241_basek_uint2,         "test241_basek_uint2"},
                      {test242_unaryv_uint2,        "test242_unaryv_uint2"},
                      {test242_unaryk_uint2,        "test242_unaryk_uint2"}, 
                      {test243_cmpv_uint2,          "test243_cmpv_uint2"}, 
                      {test244_shuffle_uint2,       "test244_shuffle_uint2"},
                      {test245_exsplat_uint2,       "test245_exsplat_uint2"},
                      {test247_funcv_uint2,         "test247_funcv_uint2"},

                      {test248_logicv_uint2,        "test248_logicv_uint2"},
                      {test249_cstcnv_uint2,        "test249_cstcnv_uint2"},

                      {test250_other_uint2,        "test250_other_uint2"},
                      {test251_any_all_uint2,      "test251_any_all_uint2"},



                      {test260_basev_int2,         "test260_basev_int2"},
                      {test261_basek_int2,         "test261_basek_int2"},
                      {test262_unaryv_int2,        "test262_unaryv_int2"},
                      {test262_unaryk_int2,        "test262_unaryk_int2"}, 
                      {test263_cmpv_int2,          "test263_cmpv_int2"}, 
                      {test264_shuffle_int2,       "test264_shuffle_int2"},
                      {test265_exsplat_int2,       "test265_exsplat_int2"},
                      {test267_funcv_int2,         "test267_funcv_int2"},

                      {test268_logicv_int2,        "test268_logicv_int2"},
                      {test269_cstcnv_int2,        "test269_cstcnv_int2"},

                      {test270_other_int2,        "test270_other_int2"},
                      {test271_any_all_int2,      "test271_any_all_int2"},



                      {test280_basev_float2,         "test280_basev_float2"},
                      {test281_basek_float2,         "test281_basek_float2"},
                      {test282_unaryv_float2,        "test282_unaryv_float2"},
                      {test282_unaryk_float2,        "test282_unaryk_float2"}, 
                      {test283_cmpv_float2,          "test283_cmpv_float2"}, 
                      {test284_shuffle_float2,       "test284_shuffle_float2"},
                      {test285_exsplat_float2,       "test285_exsplat_float2"},
                      {test287_funcv_float2,         "test287_funcv_float2"},

                      {test288_funcfv_float2,        "test288_funcfv_float2"},
                      {test289_cstcnv_float2,        "test289_cstcnv_float2"},

                      {test290_other_float2,        "test290_other_float2"},
                      {test291_any_all_float2,      "test291_any_all_float2"},


                      };
  
  const auto arraySize = sizeof(tests)/sizeof(TestRun);
  
  for(int i=0;i<int(arraySize);i++)
  {
    const bool res = tests[i].pTest();
    std::cout << "test " << std::setfill('0') << std::setw(3) << i << " " << tests[i].pTestName << "\t";
    if(res)
      std::cout << "PASSED!";
    else 
      std::cout << "FAILED!" << "\t(!!!)";
    std::cout << std::endl;
    std::cout.flush();
  }
  
  return 0;
}
