#include "LiteMath.h"
#include "Image2d.h"
#include "Bitmap.h"

#ifdef USE_STB_IMAGE
  #define STB_IMAGE_IMPLEMENTATION
  #include "stb_image.h"
  #define STB_IMAGE_WRITE_IMPLEMENTATION
  #include "stb_image_write.h"
#endif

#include <iostream>
#include <fstream>

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::uint4;
using LiteMath::int4;
using LiteMath::uint3;
using LiteMath::int3;
using LiteMath::int2;
using LiteMath::uint2;
using LiteMath::ushort4;
using LiteMath::uchar4;
using LiteMath::clamp;

static inline uint pitch(uint x, uint y, uint pitch) { return y * pitch + x; }  

static inline float4 read_array_uchar4(const uchar4* a_data, int offset)
{
  const float mult = 0.003921568f; // (1.0f/255.0f);
  const uchar4 c0  = a_data[offset];
  return mult*float4((float)c0.x, (float)c0.y, (float)c0.z, (float)c0.w);
}

static inline float4 read_array_uchar4(const uint32_t* a_data, int offset)
{
  return read_array_uchar4((const uchar4*)a_data, offset);
}

static inline int4 bilinearOffsets(const float ffx, const float ffy, const LiteImage::Sampler& a_sampler, const int w, const int h)
{
	const int sx = (ffx > 0.0f) ? 1 : -1;
	const int sy = (ffy > 0.0f) ? 1 : -1;

	const int px = (int)(ffx);
	const int py = (int)(ffy);

	int px_w0, px_w1, py_w0, py_w1;

	if (a_sampler.addressU == LiteImage::Sampler::AddressMode::CLAMP)
	{
		px_w0 = (px     >= w) ? w - 1 : px;
		px_w1 = (px + 1 >= w) ? w - 1 : px + 1;

		px_w0 = (px_w0 < 0) ? 0 : px_w0;
		px_w1 = (px_w1 < 0) ? 0 : px_w1;
	}
	else
	{
		px_w0 = px        % w;
		px_w1 = (px + sx) % w;

		px_w0 = (px_w0 < 0) ? px_w0 + w : px_w0;
		px_w1 = (px_w1 < 0) ? px_w1 + w : px_w1;
	}

	if (a_sampler.addressV == LiteImage::Sampler::AddressMode::CLAMP)
	{
		py_w0 = (py     >= h) ? h - 1 : py;
		py_w1 = (py + 1 >= h) ? h - 1 : py + 1;

		py_w0 = (py_w0 < 0) ? 0 : py_w0;
		py_w1 = (py_w1 < 0) ? 0 : py_w1;
	}
	else
	{
		py_w0 = py        % h;
		py_w1 = (py + sy) % h;

		py_w0 = (py_w0 < 0) ? py_w0 + h : py_w0;
		py_w1 = (py_w1 < 0) ? py_w1 + h : py_w1;
	}

	const int offset0 = py_w0*w + px_w0;
	const int offset1 = py_w0*w + px_w1;
	const int offset2 = py_w1*w + px_w0;
	const int offset3 = py_w1*w + px_w1;

	return int4(offset0, offset1, offset2, offset3);
}

///////////////////////////////////////////////////////////////////////

template<> 
float4 LiteImage::Image2D<float4>::sample(const LiteImage::Sampler& a_sampler, float2 a_uv) const
{
  float ffx = a_uv.x * m_fw - 0.5f; // a_texCoord should not be very large, so that the float does not overflow later. 
  float ffy = a_uv.y * m_fh - 0.5f; // This is left to the responsibility of the top level.

  if ((a_sampler.addressU == Sampler::AddressMode::CLAMP) != 0 && ffx < 0) ffx = 0.0f;
  if ((a_sampler.addressV == Sampler::AddressMode::CLAMP) != 0 && ffy < 0) ffy = 0.0f;
  
  float4 res;
  switch (a_sampler.filter)
  {
    case Sampler::Filter::LINEAR:
    {
      // Calculate the weights for each pixel
      //
      const int   px = (int)(ffx);
      const int   py = (int)(ffy);
  
      const float fx  = std::abs(ffx - (float)px);
      const float fy  = std::abs(ffy - (float)py);
      const float fx1 = 1.0f - fx;
      const float fy1 = 1.0f - fy;
  
      const float w1 = fx1 * fy1;
      const float w2 = fx  * fy1;
      const float w3 = fx1 * fy;
      const float w4 = fx  * fy;
  
      const int4 offsets = bilinearOffsets(ffx, ffy, a_sampler, m_width, m_height);
      const float4 f1    = m_data[offsets.x];
      const float4 f2    = m_data[offsets.y];
      const float4 f3    = m_data[offsets.z];
      const float4 f4    = m_data[offsets.w];

      // Calculate the weighted sum of pixels (for each color channel)
      //
      const float outr = f1.x * w1 + f2.x * w2 + f3.x * w3 + f4.x * w4;
      const float outg = f1.y * w1 + f2.y * w2 + f3.y * w3 + f4.y * w4;
      const float outb = f1.z * w1 + f2.z * w2 + f3.z * w3 + f4.z * w4;
      const float outa = f1.w * w1 + f2.w * w2 + f3.w * w3 + f4.w * w4;
  
      res = float4(outr, outg, outb, outa);
    }
    break;

    case Sampler::Filter::NEAREST:
    default:
    {
      int px = (int)(ffx + 0.5f);
      int py = (int)(ffy + 0.5f);

      if (a_sampler.addressU == Sampler::AddressMode::CLAMP)
      {
        px = (px >= int(m_width)) ? m_width - 1 : px;
        px = (px < 0) ? 0 : px;
      }
      else
      {
        px = px % m_width;
        px = (px < 0) ? px + m_width : px;
      }
  
      if (a_sampler.addressV == Sampler::AddressMode::CLAMP)
      {
        py = (py >= int(m_height)) ? m_height - 1 : py;
        py = (py < 0) ? 0 : py;
      }
      else
      {
        py = py % m_height;
        py = (py < 0) ? py + m_height : py;
      }
      res = m_data[py*m_width + px];
    }
    break;
  };
  
  return res;
}

// https://www.shadertoy.com/view/WlG3zG
inline float4 exp2m1(float4 v) { return float4(std::exp2(v.x), std::exp2(v.y), std::exp2(v.z), std::exp2(v.w)) - float4(1.0f); }
inline float4 pow_22(float4 x) { return (exp2m1(0.718151f*x)-0.503456f*x)*7.07342f; }

//inline float4 pow_22(float4 x) { x*x*(float4(0.75f) + 0.25f*x); }

template<> 
float4 LiteImage::Image2D<uchar4>::sample(const LiteImage::Sampler& a_sampler, float2 a_uv) const
{
  float ffx = a_uv.x * m_fw - 0.5f; // a_texCoord should not be very large, so that the float does not overflow later. 
  float ffy = a_uv.y * m_fh - 0.5f; // This is left to the responsibility of the top level.

  if ((a_sampler.addressU == Sampler::AddressMode::CLAMP) != 0 && ffx < 0) ffx = 0.0f;
  if ((a_sampler.addressV == Sampler::AddressMode::CLAMP) != 0 && ffy < 0) ffy = 0.0f;
  
  float4 res;
  switch (a_sampler.filter)
  {
    case Sampler::Filter::LINEAR:
    {
      // Calculate the weights for each pixel
      //
      const int   px = (int)(ffx);
      const int   py = (int)(ffy);
  
      const float fx  = std::abs(ffx - (float)px);
      const float fy  = std::abs(ffy - (float)py);
      const float fx1 = 1.0f - fx;
      const float fy1 = 1.0f - fy;
  
      const float w1 = fx1 * fy1;
      const float w2 = fx  * fy1;
      const float w3 = fx1 * fy;
      const float w4 = fx  * fy;
  
      const int4 offsets = bilinearOffsets(ffx, ffy, a_sampler, m_width, m_height);
      const float4 f1    = read_array_uchar4(m_data.data(), offsets.x);
      const float4 f2    = read_array_uchar4(m_data.data(), offsets.y);
      const float4 f3    = read_array_uchar4(m_data.data(), offsets.z);
      const float4 f4    = read_array_uchar4(m_data.data(), offsets.w);

      // Calculate the weighted sum of pixels (for each color channel)
      //
      const float outr = f1.x * w1 + f2.x * w2 + f3.x * w3 + f4.x * w4;
      const float outg = f1.y * w1 + f2.y * w2 + f3.y * w3 + f4.y * w4;
      const float outb = f1.z * w1 + f2.z * w2 + f3.z * w3 + f4.z * w4;
      const float outa = f1.w * w1 + f2.w * w2 + f3.w * w3 + f4.w * w4;
  
      res = float4(outr, outg, outb, outa);
    }
    break;

    case Sampler::Filter::NEAREST:
    default:
    {
      int px = (int)(ffx + 0.5f);
      int py = (int)(ffy + 0.5f);

      if (a_sampler.addressU == Sampler::AddressMode::CLAMP)
      {
        px = (px >= int(m_width)) ? m_width - 1 : px;
        px = (px < 0) ? 0 : px;
      }
      else
      {
        px = px % m_width;
        px = (px < 0) ? px + m_width : px;
      }
  
      if (a_sampler.addressV == Sampler::AddressMode::CLAMP)
      {
        py = (py >= int(m_height)) ? m_height - 1 : py;
        py = (py < 0) ? 0 : py;
      }
      else
      {
        py = py % m_height;
        py = (py < 0) ? py + m_height : py;
      }
      res = read_array_uchar4(m_data.data(), py*m_width + px);
    }
    break;
  };

  if(m_srgb)
    res = pow_22(res);
  
  return res;
}


template<> 
float4 LiteImage::Image2D<uint32_t>::sample(const LiteImage::Sampler& a_sampler, float2 a_uv) const
{
  float ffx = a_uv.x * m_fw - 0.5f; // a_texCoord should not be very large, so that the float does not overflow later. 
  float ffy = a_uv.y * m_fh - 0.5f; // This is left to the responsibility of the top level.

  if ((a_sampler.addressU == Sampler::AddressMode::CLAMP) != 0 && ffx < 0) ffx = 0.0f;
  if ((a_sampler.addressV == Sampler::AddressMode::CLAMP) != 0 && ffy < 0) ffy = 0.0f;
  
  float4 res;
  switch (a_sampler.filter)
  {
    case Sampler::Filter::LINEAR:
    {
      // Calculate the weights for each pixel
      //
      const int   px = (int)(ffx);
      const int   py = (int)(ffy);
  
      const float fx  = std::abs(ffx - (float)px);
      const float fy  = std::abs(ffy - (float)py);
      const float fx1 = 1.0f - fx;
      const float fy1 = 1.0f - fy;
  
      const float w1 = fx1 * fy1;
      const float w2 = fx  * fy1;
      const float w3 = fx1 * fy;
      const float w4 = fx  * fy;
  
      const int4 offsets = bilinearOffsets(ffx, ffy, a_sampler, m_width, m_height);
      const float4 f1    = read_array_uchar4(m_data.data(), offsets.x);
      const float4 f2    = read_array_uchar4(m_data.data(), offsets.y);
      const float4 f3    = read_array_uchar4(m_data.data(), offsets.z);
      const float4 f4    = read_array_uchar4(m_data.data(), offsets.w);

      // Calculate the weighted sum of pixels (for each color channel)
      //
      const float outr = f1.x * w1 + f2.x * w2 + f3.x * w3 + f4.x * w4;
      const float outg = f1.y * w1 + f2.y * w2 + f3.y * w3 + f4.y * w4;
      const float outb = f1.z * w1 + f2.z * w2 + f3.z * w3 + f4.z * w4;
      const float outa = f1.w * w1 + f2.w * w2 + f3.w * w3 + f4.w * w4;
  
      res = float4(outr, outg, outb, outa);
    }
    break;

    case Sampler::Filter::NEAREST:
    default:
    {
      int px = (int)(ffx + 0.5f);
      int py = (int)(ffy + 0.5f);

      if (a_sampler.addressU == Sampler::AddressMode::CLAMP)
      {
        px = (px >= int(m_width)) ? m_width - 1 : px;
        px = (px < 0) ? 0 : px;
      }
      else
      {
        px = px % m_width;
        px = (px < 0) ? px + m_width : px;
      }
  
      if (a_sampler.addressV == Sampler::AddressMode::CLAMP)
      {
        py = (py >= int(m_height)) ? m_height - 1 : py;
        py = (py < 0) ? 0 : py;
      }
      else
      {
        py = py % m_height;
        py = (py < 0) ? py + m_height : py;
      }
      res = read_array_uchar4(m_data.data(), py*m_width + px);
    }
    break;
  };
  
  if(m_srgb)
    res = pow_22(res);

  return res;

  // unfortunately this doesn not works correctly for bilinear sampling .... 

  /*bool useBorderColor = false;
  a_uv = process_coord(a_sampler.addressU, a_uv, &useBorderColor);
    
  if (useBorderColor) {
    return a_sampler.borderColor;
  }

  const float2 textureSize = make_float2(m_width, m_height);
  const float2 scaledUV    = textureSize * a_uv;
  const int2   baseTexel   = make_int2(scaledUV.x, scaledUV.y);
  const int    stride      = m_width;

  const uchar4* data2      = (const uchar4*)m_data.data();

  switch (a_sampler.filter)
  {
  
  case Sampler::Filter::LINEAR:
  {
    const int2 cornerTexel  = make_int2(baseTexel.x < m_width  - 1 ? baseTexel.x + 1 : baseTexel.x,
                                        baseTexel.y < m_height - 1 ? baseTexel.y + 1 : baseTexel.y);

    const int offset0       = pitch(baseTexel.x  , baseTexel.y  , stride);
    const int offset1       = pitch(cornerTexel.x, baseTexel.y  , stride);
    const int offset2       = pitch(baseTexel.x  , cornerTexel.y, stride);
    const int offset3       = pitch(cornerTexel.x, cornerTexel.y, stride);

    const float2 lerpCoefs  = scaledUV - float2(baseTexel.x, baseTexel.y);
    const uchar4 uData1     = lerp(data2[offset0], data2[offset1], lerpCoefs.x);
    const uchar4 uData2     = lerp(data2[offset2], data2[offset3], lerpCoefs.x);
    const float4 line1Color = (1.0f/255.0f)*float4(uData1.x, uData1.y, uData1.z, uData1.w);
    const float4 line2Color = (1.0f/255.0f)*float4(uData2.x, uData2.y, uData2.z, uData2.w);

    return lerp(line1Color, line2Color, lerpCoefs.y);
  }     
  case Sampler::Filter::NEAREST:
  default:
  {
    const uchar4 uData = data2[pitch(baseTexel.x, baseTexel.y, stride)];
    return (1.0f/255.0f)*float4(uData.x, uData.y, uData.z, uData.w);
  }

  //default:
  //  fprintf(stderr, "Unsupported filter is used.");
  //  break;
  }

  return make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  */
}

/*
template<typename Type>
float2 Image2D<Type>::process_coord(const Sampler::AddressMode mode, const float2 coord, bool* use_border_color) const
{ 
  float2 res = coord;

  switch (mode)
  {
    case Sampler::AddressMode::CLAMP: 
      //res = coord;            
      break;
    case Sampler::AddressMode::WRAP:  
    {
      res = make_float2(std::fmod(coord.x, 1.0), std::fmod(coord.y, 1.0));            
      if (res.x < 0.0F) res.x += 1.0F;
      if (res.y < 0.0F) res.y += 1.0F;      
      break;
    }
    case Sampler::AddressMode::MIRROR:
    {
      const float u = static_cast<int>(coord.x) % 2 ? 1.0f - std::fmod(coord.x, 1.0f) : std::fmod(coord.x, 1.0f);
      const float v = static_cast<int>(coord.y) % 2 ? 1.0f - std::fmod(coord.y, 1.0f) : std::fmod(coord.y, 1.0f);
      res = make_float2(u, v);            
      break;
    }
    case Sampler::AddressMode::MIRROR_ONCE:
      res = make_float2(std::abs(coord.x), std::abs(coord.y));
      break;
    case Sampler::AddressMode::BORDER:
      *use_border_color = *use_border_color || coord.x < 0.0f || coord.x > 1.0f || coord.y < 0.0f || coord.y > 1.0f;
      break;      
    default:
      break;
  }

  return clamp(res, 0.0f, 1.0F);
} 
*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace myvulkan 
{
  // Provided by VK_VERSION_1_0
  typedef enum VkFormat {
    VK_FORMAT_UNDEFINED = 0,
    VK_FORMAT_R8_UNORM = 9,               // ***
    VK_FORMAT_R8_SNORM = 10,              // ***
    VK_FORMAT_R8_SRGB = 15,               // ***
    VK_FORMAT_R8G8B8A8_UNORM = 37,        // ****
    VK_FORMAT_R8G8B8A8_SRGB = 43,         // ***
    VK_FORMAT_R16_UNORM = 70,             // ***
    VK_FORMAT_R16_SNORM = 71,             // ***
    VK_FORMAT_R16G16B16A16_UNORM = 91,    // ***
    VK_FORMAT_R32_SFLOAT = 100,           // ***
    VK_FORMAT_R32G32_SFLOAT = 103,        // ***
    VK_FORMAT_R32G32B32A32_SFLOAT = 109,  // ***
  } VkFormat;
};

template<> uint32_t LiteImage::GetVulkanFormat<uint32_t>(bool a_gamma22) { return a_gamma22 ? uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_SRGB) : uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_UNORM); } // SRGB, UNORM 
template<> uint32_t LiteImage::GetVulkanFormat<uchar4>(bool a_gamma22)   { return a_gamma22 ? uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_SRGB) : uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_UNORM); }

template<> uint32_t LiteImage::GetVulkanFormat<uint64_t>(bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R16G16B16A16_UNORM); }
template<> uint32_t LiteImage::GetVulkanFormat<ushort4>(bool a_gamma22)  { return uint32_t(myvulkan::VK_FORMAT_R16G16B16A16_UNORM); }

template<> uint32_t LiteImage::GetVulkanFormat<uint16_t>(bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R16_UNORM); }
template<> uint32_t LiteImage::GetVulkanFormat<uint8_t>(bool a_gamma22)  { return a_gamma22 ? uint32_t(myvulkan::VK_FORMAT_R8_SRGB) : uint32_t(myvulkan::VK_FORMAT_R8_UNORM); }

template<> uint32_t LiteImage::GetVulkanFormat<float4>(bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R32G32B32A32_SFLOAT); }
template<> uint32_t LiteImage::GetVulkanFormat<float2>(bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R32G32_SFLOAT); }
template<> uint32_t LiteImage::GetVulkanFormat<float> (bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R32_SFLOAT); }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline int tonemap(float x, float a_gammaInv) 
{ 
  const int colorLDR = int( std::pow(x, a_gammaInv)*255.0f + float(.5f) );
  if(colorLDR < 0)        return 0;
  else if(colorLDR > 255) return 255;
  else                    return colorLDR;
}

static inline unsigned IntColorUint32(int r, int g, int b) { return unsigned(r | (g << 8) | (b << 16) | 0xFF000000); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
bool LiteImage::SaveImage<float4>(const char* a_fileName, const LiteImage::Image2D<float4>& a_image, float a_gamma) 
{
  const float gammaInv = 1.0f/a_gamma;
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ofstream fout(a_fileName, std::fstream::out);
    if(!fout.is_open())
      return false;
    fout << "P3" << std::endl << a_image.width() << " " << a_image.height() << " 255" << std::endl;
    for (auto c : a_image.vector()) 
      fout << tonemap(c[0], gammaInv) << " " << tonemap(c[1], gammaInv) << " " << tonemap(c[2], gammaInv) << " ";
    fout.close();
    return true;
  }
  else if(fileExt == ".image4f")
  {
    unsigned wh[2] = { a_image.width(), a_image.height() };
    std::ofstream fout(a_fileName, std::fstream::out | std::ios::binary);
    if(!fout.is_open())
      return false;
    fout.write((char*)wh, sizeof(unsigned)* 2);
    fout.write((char*)a_image.data(), size_t(wh[0]*wh[1]*4)*sizeof(float));
    fout.close();
    return true;
  }
  
  if(fileExt == ".bmp" || fileExt == ".BMP" || fileExt == ".png" || fileExt == ".PNG")
  {
    std::vector<unsigned> flipedYData(a_image.width()*a_image.height());
    for (unsigned y = 0; y < a_image.height(); y++)
    {
      const unsigned offset1 = (a_image.height() - y - 1)*a_image.width(); 
      const unsigned offset2 = y*a_image.width();
      for(unsigned x=0; x<a_image.width(); x++)
      {
        auto c = a_image.data()[offset1 + x];
        flipedYData[offset2+x] = IntColorUint32(tonemap(c[0], gammaInv), tonemap(c[1], gammaInv), tonemap(c[2], gammaInv));
      }
    }

    if(fileExt == ".bmp" || fileExt == ".BMP")
      return SaveBMP(a_fileName, flipedYData.data(), a_image.width(), a_image.height());
    #ifdef USE_STB_IMAGE
    else if(fileExt == ".png" || fileExt == ".PNG") 
      return stbi_write_png(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)flipedYData.data(), width * 4);
    #endif
  }
  
  std::cout << "[SaveImage<float4>]: unsupported extension '" << fileExt.c_str() << "'" << std::endl;
  return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
bool LiteImage::SaveImage<float3>(const char* a_fileName, const LiteImage::Image2D<float3>& a_image, float a_gamma) 
{
  const float gammaInv = 1.0f/a_gamma;
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ofstream fout(a_fileName, std::fstream::out);
    if(!fout.is_open())
      return false;
    fout << "P3" << std::endl << a_image.width() << " " << a_image.height() << " 255" << std::endl;
    for (auto c : a_image.vector()) 
      fout << tonemap(c[0], gammaInv) << " " << tonemap(c[1], gammaInv) << " " << tonemap(c[2], gammaInv) << " ";
    fout.close();
    return true;
  }
  else if(fileExt == ".image3f")
  {
    unsigned wh[2] = { a_image.width(), a_image.height() };
    std::ofstream fout(a_fileName, std::fstream::out | std::ios::binary);
    if(!fout.is_open())
      return false;
    fout.write((char*)wh, sizeof(unsigned)* 2);
    fout.write((char*)a_image.data(), size_t(wh[0]*wh[1]*3)*sizeof(float));
    fout.close();
    return true;
  }
  
  if(fileExt == ".bmp" || fileExt == ".BMP" || fileExt == ".png" || fileExt == ".PNG")
  {
    std::vector<unsigned> flipedYData(a_image.width()*a_image.height());
    for (unsigned y = 0; y < a_image.height(); y++)
    {
      const unsigned offset1 = (a_image.height() - y - 1)*a_image.width(); 
      const unsigned offset2 = y*a_image.width();
      for(unsigned x=0; x<a_image.width(); x++)
      {
        auto c = a_image.data()[offset1 + x];
        flipedYData[offset2+x] = IntColorUint32(tonemap(c[0], gammaInv), tonemap(c[1], gammaInv), tonemap(c[2], gammaInv));
      }
    }

    if(fileExt == ".bmp" || fileExt == ".BMP")
      return SaveBMP(a_fileName, flipedYData.data(), a_image.width(), a_image.height());
    #ifdef USE_STB_IMAGE
    else if(fileExt == ".png" || fileExt == ".PNG") 
      return stbi_write_png(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)flipedYData.data(), width * 4);
    #endif
  }

  std::cout << "[SaveImage<float3>]: unsupported extension '" << fileExt.c_str() << "'" << std::endl;
  return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
bool LiteImage::SaveImage<float>(const char* a_fileName, const LiteImage::Image2D<float>& a_image, float a_gamma) 
{
  const float gammaInv = 1.0f/a_gamma;
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ofstream fout(a_fileName, std::fstream::out);
    if(!fout.is_open())
      return false;
    fout << "P3" << std::endl << a_image.width() << " " << a_image.height() << " 255" << std::endl;
    for (auto c : a_image.vector()) {
      auto val = tonemap(c, gammaInv);
      fout << val << " " << val << " " << val << " ";
    }
    fout.close();
    return true;
  }
  else if(fileExt == ".image1f")
  {
    unsigned wh[2] = { a_image.width(), a_image.height() };
    std::ofstream fout(a_fileName, std::fstream::out | std::ios::binary);
    if(!fout.is_open())
      return false;
    fout.write((char*)wh, sizeof(unsigned)* 2);
    fout.write((char*)a_image.data(), size_t(wh[0]*wh[1])*sizeof(float));
    fout.close();
    return true;
  }
  
  if(fileExt == ".bmp" || fileExt == ".BMP" || fileExt == ".png" || fileExt == ".PNG")
  {
    std::vector<unsigned> flipedYData(a_image.width()*a_image.height());
    for (unsigned y = 0; y < a_image.height(); y++)
    {
      const unsigned offset1 = (a_image.height() - y - 1)*a_image.width(); 
      const unsigned offset2 = y*a_image.width();
      for(unsigned x=0; x<a_image.width(); x++)
      {
        auto c   = a_image.data()[offset1 + x];
        auto val = tonemap(c, gammaInv);
        flipedYData[offset2+x] = IntColorUint32(val, val, val);
      }
    }

    if(fileExt == ".bmp" || fileExt == ".BMP")
      return SaveBMP(a_fileName, flipedYData.data(), a_image.width(), a_image.height());
    #ifdef USE_STB_IMAGE
    else if(fileExt == ".png" || fileExt == ".PNG") 
      return stbi_write_png(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)flipedYData.data(), width * 4);
    #endif
  }

  std::cout << "[SaveImage<float>]: unsupported extension '" << fileExt.c_str() << "'" << std::endl;
  return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
bool LiteImage::SaveImage<uint32_t>(const char* a_fileName, const LiteImage::Image2D<uint32_t>& a_image, float a_gamma) 
{
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ofstream fout(a_fileName, std::fstream::out);
    if(!fout.is_open())
      return false;
    fout << "P3" << std::endl << a_image.width() << " " << a_image.height() << " 255" << std::endl;
    for (auto c : a_image.vector()) {
      auto r = (c & 0x000000FF);
      auto g = (c & 0x0000FF00) >> 8;
      auto b = (c & 0x00FF0000) >> 16;
      fout << r << " " << g << " " << b << " ";
    }
    fout.close();
    return true;
  }
  else if(fileExt == ".image1ui" || fileExt == ".image4ub")
  {
    unsigned wh[2] = { a_image.width(), a_image.height() };
    std::ofstream fout(a_fileName, std::fstream::out | std::ios::binary);
    if(!fout.is_open())
      return false;
    fout.write((char*)wh, sizeof(unsigned)* 2);
    fout.write((char*)a_image.data(), size_t(wh[0]*wh[1])*sizeof(uint32_t));
    fout.close();
    return true;
  }
  
  if(fileExt == ".bmp" || fileExt == ".BMP" || fileExt == ".png" || fileExt == ".PNG")
  {
    std::vector<unsigned> flipedYData(a_image.width()*a_image.height());
    for (unsigned y = 0; y < a_image.height(); y++)
    {
      const unsigned offset1 = (a_image.height() - y - 1)*a_image.width(); 
      const unsigned offset2 = y*a_image.width();
      for(unsigned x=0; x<a_image.width(); x++)
      {
        auto c   = a_image.data()[offset1 + x];
        flipedYData[offset2+x] = c;
      }
    }

    if(fileExt == ".bmp" || fileExt == ".BMP")
      return SaveBMP(a_fileName, flipedYData.data(), a_image.width(), a_image.height());
    #ifdef USE_STB_IMAGE
    else if(fileExt == ".png" || fileExt == ".PNG") 
      return stbi_write_png(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)flipedYData.data(), width * 4);
    #endif
  }

  std::cout << "[SaveImage<uint32_t>]: unsupported extension '" << fileExt.c_str() << "'" << std::endl;
  return false;
}
