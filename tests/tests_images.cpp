#include "Image2d.h"

#include <iostream>
#include <fstream>
#include <iomanip>      // std::setfill, std::setw

using LiteMath::float3;
using LiteMath::float4;
using LiteMath::uchar4;

using LiteMath::uint2;
using LiteMath::int2;
using LiteMath::float2;
using LiteImage::Image2D;

#ifdef WIN32
  #include <direct.h>     // for windows mkdir
#else
  #include <sys/stat.h>   // for linux mkdir
  #include <sys/types.h>
  #include <unistd.h>
#endif

bool test01_float3_save()
{
  Image2D<float3> imgRus(300,200, float3{1,1,1});

  for(unsigned y=0; y<imgRus.height();y++)
  {
    float3 color;
    if(y < imgRus.height()/3)
      color = float3{1,1,1};
    else if(y < 2*imgRus.height()/3)
      color = float3{0,0,1};
    else 
      color = float3{1,0,0};

    for(unsigned x=0;x<imgRus.width();x++)
      imgRus[uint2(x,y)] = color;
  }
  
  LiteImage::SaveImage("flags/rus.ppm", imgRus);
  LiteImage::SaveImage("flags/rus.bmp", imgRus);
  LiteImage::SaveImage("flags/rus.jpg", imgRus);

  LiteImage::SaveImage("flags/rus.image3f", imgRus);
  auto img2 = LiteImage::LoadImage<float3>("flags/rus.image3f");
  LiteImage::SaveImage("flags/rus.png", img2);

  auto img3 = LiteImage::LoadImage<float3>("flags/rus.ppm");
  auto img4 = LiteImage::LoadImage<float3>("flags/rus.bmp");
  auto img5 = LiteImage::LoadImage<float3>("flags/rus.jpg");
  auto img6 = LiteImage::LoadImage<float3>("flags/rus.png");
  //LiteImage::SaveImage("flags/rus6.png", img6);

  const float err2 = LiteImage::MSE(imgRus, img2);
  const float err3 = LiteImage::MSE(imgRus, img3);
  const float err4 = LiteImage::MSE(imgRus, img4);
  const float err5 = LiteImage::MSE(imgRus, img5);
  const float err6 = LiteImage::MSE(imgRus, img6);

  return (err2 < 1e-5f) && (err3 < 1e-5f) && (err4 < 1e-5f) && (err5 < 1e-4f) && (err6 < 1e-5f);
}

bool test02_float4_save()
{
  Image2D<float4> imgGer(300,200);

  for(unsigned y=0; y<imgGer.height();y++)
  {
    float4 color{0,0,0,0};
    if(y < imgGer.height()/3)
      color = float4{0,0,0,0};
    else if(y < 2*imgGer.height()/3)
      color = float4{1,0,0,0};
    else 
      color = float4{1,1,0,0};

    for(unsigned x=0;x<imgGer.width();x++)
      imgGer[uint2(x,y)] = color;
  }
  
  LiteImage::SaveImage("flags/ger.ppm", imgGer);
  LiteImage::SaveImage("flags/ger.bmp", imgGer);
  LiteImage::SaveImage("flags/ger.jpg", imgGer);

  LiteImage::SaveImage("flags/ger.image4f", imgGer);
  auto img2 = LiteImage::LoadImage<float4>("flags/ger.image4f");
  LiteImage::SaveImage("flags/ger.png", img2);

  auto img3 = LiteImage::LoadImage<float4>("flags/ger.ppm");
  auto img4 = LiteImage::LoadImage<float4>("flags/ger.bmp");
  auto img5 = LiteImage::LoadImage<float4>("flags/ger.jpg");
  auto img6 = LiteImage::LoadImage<float4>("flags/ger.png");

  const float err2 = LiteImage::MSE(imgGer, img2);
  const float err3 = LiteImage::MSE(imgGer, img3);
  const float err4 = LiteImage::MSE(imgGer, img4);
  const float err5 = LiteImage::MSE(imgGer, img5);
  const float err6 = LiteImage::MSE(imgGer, img6);

  return (err2 < 1e-5f) && (err3 < 1e-5f) && (err4 < 1e-5f) && (err5 < 1e-4f) && (err6 < 1e-5f);
}

bool test03_float1_save()
{
  Image2D<float> imgGradient(300,200);

  for(unsigned y=0; y<imgGradient.height();y++)
  {
    float color = float(y)/float(imgGradient.height()-1);
    for(unsigned x=0;x<imgGradient.width();x++)
      imgGradient[uint2(x,y)] = color;
  }
  
  LiteImage::SaveImage("flags/grad_22.ppm", imgGradient);
  LiteImage::SaveImage("flags/grad_22.bmp", imgGradient);
  LiteImage::SaveImage("flags/grad_22.jpg", imgGradient);

  LiteImage::SaveImage("flags/grad_10.ppm", imgGradient, 1.0f);
  LiteImage::SaveImage("flags/grad_10.bmp", imgGradient, 1.0f);
  LiteImage::SaveImage("flags/grad_10.jpg", imgGradient, 1.0f);

  LiteImage::SaveImage("flags/grad_10.image1f", imgGradient);
  auto img2 = LiteImage::LoadImage<float>("flags/grad_10.image1f");
  LiteImage::SaveImage("flags/grad_10.png", img2, 1.0f);

  auto img3 = LiteImage::LoadImage<float>("flags/grad_10.ppm", 1.0f);
  auto img4 = LiteImage::LoadImage<float>("flags/grad_10.bmp", 1.0f);
  auto img5 = LiteImage::LoadImage<float>("flags/grad_10.jpg", 1.0f);
  auto img6 = LiteImage::LoadImage<float>("flags/grad_10.png", 1.0f);
  
  const float err2 = LiteImage::MSE(imgGradient, img2);
  const float err3 = LiteImage::MSE(imgGradient, img3);
  const float err4 = LiteImage::MSE(imgGradient, img4);
  const float err5 = LiteImage::MSE(imgGradient, img5);
  const float err6 = LiteImage::MSE(imgGradient, img6);
  
  return (err2 < 1e-5f) && (err3 < 1e-5f) && (err4 < 1e-5f) && (err5 < 1e-4f) && (err6 < 1e-5f);
}

bool test04_uint1_save()
{
  Image2D<uint32_t> imgEst(300,200);

  for(unsigned y=0; y<imgEst.height();y++)
  {
    uint32_t color = 0;
    if(y < imgEst.height()/3)
      color = 0xFFCD7100;
    else if(y < 2*imgEst.height()/3)
      color = 0;
    else 
      color = 0xFFFFFFFF;

    for(unsigned x=0;x<imgEst.width();x++)
      imgEst[uint2(x,y)] = color;
  }
  
  LiteImage::SaveImage("flags/est.ppm", imgEst);
  LiteImage::SaveImage("flags/est.bmp", imgEst);
  LiteImage::SaveImage("flags/est.jpg", imgEst);
  LiteImage::SaveImage("flags/est.image4ub", imgEst);

  auto img2 = LiteImage::LoadImage<float4>("flags/est.ppm");
  auto img3 = LiteImage::LoadImage<float4>("flags/est.bmp");
  auto img4 = LiteImage::LoadImage<uint32_t>("flags/est.image4ub");
  LiteImage::SaveImage("flags/est.png", img2);
  LiteImage::SaveImage("flags/est2.png", img3);
  
  auto img12 = LiteImage::LoadImage<uint32_t>("flags/est.ppm");
  auto img13 = LiteImage::LoadImage<uint32_t>("flags/est.bmp");
  auto img14 = LiteImage::LoadImage<uint32_t>("flags/est.png");
  auto img15 = LiteImage::LoadImage<uint32_t>("flags/est.jpg");
  auto img16 = LiteImage::LoadImage<uint32_t>("flags/est2.png");
  //LiteImage::SaveImage("flags/est_12.bmp", img12);
  
  const float err2 = LiteImage::MSE(imgEst, img12);
  const float err3 = LiteImage::MSE(imgEst, img13);
  const float err4 = LiteImage::MSE(imgEst, img14);
  const float err5 = LiteImage::MSE(imgEst, img15);
  const float err6 = LiteImage::MSE(imgEst, img16);
  const float err7 = LiteImage::MSE(imgEst, img4);
  
  return (err2 < 1e-5f) && (err3 < 1e-5f) && (err4 < 1e-5f) && (err5 < 1e-4f) && (err6 < 1e-5f) && (err7 < 1e-5f);
}

bool test05_uchar4_save()
{
  Image2D<uchar4> imgUkr(300,200);

  for(unsigned y=0; y<imgUkr.height();y++)
  {
    uchar4 color;
    if(y < imgUkr.height()/2)
      color = uchar4(0,90,200,255);
    else 
      color = uchar4(255,210,0,255);

    for(unsigned x=0;x<imgUkr.width();x++)
      imgUkr[uint2(x,y)] = color;
  }
  
  LiteImage::SaveImage("flags/ukr.ppm", imgUkr);
  LiteImage::SaveImage("flags/ukr.bmp", imgUkr);
  LiteImage::SaveImage("flags/ukr.jpg", imgUkr);
  
  auto img2 = LiteImage::LoadImage<float3>("flags/ukr.ppm");
  auto img3 = LiteImage::LoadImage<float3>("flags/ukr.bmp");
  LiteImage::SaveImage("flags/ukr.png", img2);
  LiteImage::SaveImage("flags/ukr2.png", img3);
  LiteImage::SaveImage("flags/ukr.image4ub", imgUkr);
  
  auto img12 = LiteImage::LoadImage<uchar4>("flags/ukr.ppm");
  auto img13 = LiteImage::LoadImage<uchar4>("flags/ukr.bmp");
  auto img14 = LiteImage::LoadImage<uchar4>("flags/ukr.png");
  auto img15 = LiteImage::LoadImage<uchar4>("flags/ukr.jpg");
  auto img16 = LiteImage::LoadImage<uchar4>("flags/ukr2.png");
  auto img17 = LiteImage::LoadImage<uchar4>("flags/ukr.image4ub");
  LiteImage::SaveImage("flags/ukr_15.bmp", img15);
  
  const float err2 = LiteImage::MSE(imgUkr, img12);
  const float err3 = LiteImage::MSE(imgUkr, img13);
  const float err4 = LiteImage::MSE(imgUkr, img14);
  const float err5 = LiteImage::MSE(imgUkr, img15);
  const float err6 = LiteImage::MSE(imgUkr, img16);
  const float err7 = LiteImage::MSE(imgUkr, img17);
  
  return (err2 < 1e-5f) && (err3 < 1e-5f) && (err4 < 1e-5f) && (err5 < 1e-2f) && (err6 < 1e-5f) && (err7 < 1e-5f);
}

bool test06_textures()
{
  auto img0 = LiteImage::LoadImage<uchar4>("data/texture1.bmp");
  auto img1 = LiteImage::LoadImage<uint32_t>("data/texture1.bmp");
  auto img2 = LiteImage::LoadImage<float4>("data/texture1.bmp");

  Image2D<float4> img3(img1.width()*2, img1.height()*2);
  Image2D<float4> img4(img1.width()*2, img1.height()*2);
  Image2D<float4> img5(img1.width()*2, img1.height()*2);
  Image2D<float4> img6(img1.width()*2, img1.height()*2);
  Image2D<float4> img7(img1.width()*2, img1.height()*2);
  Image2D<float4> img8(img1.width()*2, img1.height()*2);
  Image2D<float4> img9(img1.width()*2, img1.height()*2);
  Image2D<float4> img10(img1.width()*2, img1.height()*2);
  Image2D<float4> img11(img1.width()*2, img1.height()*2);
  Image2D<float4> img12(img1.width()*2, img1.height()*2);
  Image2D<float4> img13(img1.width()*2, img1.height()*2);

  LiteImage::Sampler sam1, sam2, sam3, sam4;

  sam1.addressU = LiteImage::Sampler::AddressMode::CLAMP;
  sam1.addressV = LiteImage::Sampler::AddressMode::CLAMP;
  sam1.filter   = LiteImage::Sampler::Filter::LINEAR;

  sam2.addressU = LiteImage::Sampler::AddressMode::CLAMP;
  sam2.addressV = LiteImage::Sampler::AddressMode::CLAMP;
  sam2.filter   = LiteImage::Sampler::Filter::NEAREST;

  sam3.addressU = LiteImage::Sampler::AddressMode::WRAP;
  sam3.addressV = LiteImage::Sampler::AddressMode::WRAP;
  sam3.filter   = LiteImage::Sampler::Filter::LINEAR;

  sam4.addressU = LiteImage::Sampler::AddressMode::WRAP;
  sam4.addressV = LiteImage::Sampler::AddressMode::WRAP;
  sam4.filter   = LiteImage::Sampler::Filter::NEAREST;

  for(int y=0; y < int(img3.height()); y++)
  {
    float texCoordY = (float(y) + 0.5f)/float(img3.height());
    for(int x=0; x < int(img3.width()); x++)
    {
      float texCoordX = (float(x) + 0.5f)/float(img3.width());
      
      img3[int2(x,y)] = img1.sample(sam1, float2(texCoordX,texCoordY));
      img4[int2(x,y)] = img2.sample(sam2, float2(texCoordX,texCoordY));
      img5[int2(x,y)] = img2.sample(sam3, 2.0f*float2(texCoordX,texCoordY));
      img6[int2(x,y)] = img1.sample(sam3, 2.0f*float2(texCoordX,texCoordY));
      img7[int2(x,y)] = img0.sample(sam3, 2.0f*float2(texCoordX,texCoordY));
      img8[int2(x,y)] = img0.sample(sam1, 2.0f*float2(texCoordX,texCoordY) - float2(0.5f));

      img9 [int2(x,y)] = img0.sample(sam4, 4.0f*float2(texCoordX,texCoordY) - float2(0.25f));
      img10[int2(x,y)] = img1.sample(sam4, 4.0f*float2(texCoordX,texCoordY) - float2(0.25f));
      img11[int2(x,y)] = img2.sample(sam4, 4.0f*float2(texCoordX,texCoordY) - float2(0.25f));

      img12[int2(x,y)] = img0.sample(sam2, 4.0f*float2(texCoordX,texCoordY) - float2(0.25f));
      img13[int2(x,y)] = img1.sample(sam2, 4.0f*float2(texCoordX,texCoordY) - float2(0.25f));
    }
  }

  LiteImage::SaveImage("flags/tex1_linear_512.bmp", img3);
  LiteImage::SaveImage("flags/tex1_point_512.bmp",  img4);
  LiteImage::SaveImage("flags/tex1_wrap1_512.bmp",  img5);
  LiteImage::SaveImage("flags/tex1_wrap2_512.bmp",  img6);
  LiteImage::SaveImage("flags/tex1_wrap3_512.bmp",  img7);
  LiteImage::SaveImage("flags/tex1_clamp_512.bmp",  img8);
  LiteImage::SaveImage("flags/tex1_near_wrap1.bmp", img9);
  LiteImage::SaveImage("flags/tex1_near_wrap2.bmp", img10);
  LiteImage::SaveImage("flags/tex1_near_wrap3.bmp", img11);
  LiteImage::SaveImage("flags/tex1_near_clamp1.bmp", img12);
  LiteImage::SaveImage("flags/tex1_near_clamp2.bmp", img13);

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using TestFuncType2 = bool (*)();
struct TestRun2
{
  TestFuncType2 pTest;
  const char*   pTestName;
};

void tests_all_images()
{
  Image2D<float3> img1(500,200, float3{1,1,1});
  Image2D<float3> img2(500,200, float3{1,1,1});

  auto img3 = (img1 + img2)*0.5f;
  auto img4 = (img1 - img2)/2.0f;

  #ifdef WIN32
  mkdir("flags");
  chdir("..");
  #else
  mkdir("flags", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  std::ifstream test("data/texture1.bmp");
  if(!test.is_open())
    chdir("..");
  else
    test.close();
  #endif

  std::cout << std::endl;
  std::cout << "run images tests: " << std::endl;

  TestRun2 tests[] = { {test01_float3_save,  "test01_float3_save"},
                       {test02_float4_save,  "test02_float4_save"},
                       {test03_float1_save,  "test03_float1_save"},
                       {test04_uint1_save,   "test04_uint1_save"},
                       {test05_uchar4_save,  "test05_uchar4_save"},
                       {test06_textures,     "test06_textures"},
                     };

  

  const auto arraySize = sizeof(tests)/sizeof(TestRun2);
  
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

}