#include "Image2d.h"

using LiteMath::float3;
using LiteMath::float4;
using LiteMath::uchar4;

using LiteMath::uint2;
using LiteImage::Image2D;

#ifdef WIN32
  #include <direct.h>     // for windows mkdir
#else
  #include <sys/stat.h>   // for linux mkdir
  #include <sys/types.h>
#endif

void test01_float3_save()
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

  LiteImage::SaveImage("flags/rus.image3f", imgRus);
  auto img2 = LiteImage::LoadImage<float3>("flags/rus.image3f");
  LiteImage::SaveImage("flags/rus.png", img2);
}

void test02_float4_save()
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

  LiteImage::SaveImage("flags/ger.image4f", imgGer);
  auto img2 = LiteImage::LoadImage<float4>("flags/ger.image4f");
  LiteImage::SaveImage("flags/ger.png", img2);
}

void test03_float1_save()
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

  LiteImage::SaveImage("flags/grad_10.ppm", imgGradient, 1.0f);
  LiteImage::SaveImage("flags/grad_10.bmp", imgGradient, 1.0f);

  //#TODO: save float image, then load it to test both functions

}

void test04_uint1_save()
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

  //#TODO: save float4 image, then load it to test both functions

}

void test05_uchar4_save()
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

  //#TODO: save float4 image, then load it to test both functions

}

void tests_all_images()
{
  Image2D<float3> img1(500,200, float3{1,1,1});
  Image2D<float3> img2(500,200, float3{1,1,1});

  auto img3 = (img1 + img2)*0.5f;
  auto img4 = (img1 - img2)/2.0f;

  #ifdef WIN32
  mkdir("flags");
  #else
  mkdir("flags", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  #endif

  test01_float3_save();
  test02_float4_save();
  test03_float1_save();
  test04_uint1_save();
  test05_uchar4_save();
  //img1.load("data/texture1.bmp");

}