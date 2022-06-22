#ifndef TEXTURE2D_H
#define TEXTURE2D_H

#include <vector>
#include <string>
#include <memory>
#include <cassert>
//#include <type_traits>

#include "LiteMath.h"

namespace LiteImage 
{
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

  struct Sampler 
  {
    enum class AddressMode {
      WRAP        = 0,
      MIRROR      = 1,
      CLAMP       = 2,
      BORDER      = 3,
      MIRROR_ONCE = 4,
    };
  
    enum class Filter {
      NEAREST = 0,
      LINEAR  = 1,
    };
  
    // sampler state
    //
    AddressMode addressU      = AddressMode::WRAP;
    AddressMode addressV      = AddressMode::WRAP;
    AddressMode addressW      = AddressMode::WRAP;
    float4      borderColor   = float4(0.0f, 0.0f, 0.0f, 0.0f);
    Filter      filter        = Filter::NEAREST;
    //uint32_t    maxAnisotropy = 1;
    //uint32_t    maxLOD        = 32;
    //uint32_t    minLOD        = 0;
    //uint32_t    mipLODBias    = 0;
  };

  template<typename T> uint32_t GetVulkanFormat(bool a_gamma22);

  template<typename Type>
  struct Image2D
  {
    Image2D() : m_width(0), m_height(0) {}
    Image2D(unsigned int w, unsigned int h, const Type* a_data) : m_width(w), m_height(h) 
    {
      resize(width, height);
      memcpy(m_data.data(), a_data, w*h*sizeof(Type));
    }
    
    Image2D(unsigned int w, unsigned int h) : m_width(w), m_height(h) { resize(w, h); }
    Image2D(unsigned int w, unsigned int h, const Type val)           { resize(w, h); clear(val); }
  
    Image2D(const Image2D& rhs) = default;            // may cause crash on some old compilers
    Image2D(Image2D&& lhs)      = default;            // may cause crash on some old compilers
  
    Image2D& operator=(Image2D&& rhs)      = default; // may cause crash on some old compilers
    Image2D& operator=(const Image2D& rhs) = default; // may cause crash on some old compilers
    
    void resize(unsigned int width, unsigned int height) { m_width = width; m_height = height; m_data.resize(width*height); m_fw = float(width); m_fh = float(height); }
    void clear(const Type val = Type(0))                 { for(auto& pixel : m_data) pixel = val; }  
  
    float4 sample(const Sampler& a_sampler, float2 a_uv) const;    
    
    Type&  operator[](const int2 coord)        { return m_data[coord.y * m_width + coord.x]; }
    Type   operator[](const int2 coord) const  { return m_data[coord.y * m_width + coord.x]; }
    Type&  operator[](const uint2 coord)       { return m_data[coord.y * m_width + coord.x]; }
    Type   operator[](const uint2 coord) const { return m_data[coord.y * m_width + coord.x]; }
  
    unsigned int width()  const { return m_width; }
    unsigned int height() const { return m_height; }  
    unsigned int bpp()    const { return sizeof(Type); }
   
    const Type*              data()   const { return m_data.data(); }
    const std::vector<Type>& vector() const { return m_data; }
    unsigned int             format() const { return GetVulkanFormat<Type>(m_srgb); } 
  
    void setSRGB(bool a_val) { m_srgb = a_val; }
  
    // #TODO: void resampleTo(Image2D& a_outImage);
    // #TODO: void gaussBlur(int BLUR_RADIUS2, float a_sigma = ...);
    // #TODO: MSE, MSEImage
  
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
    Image2D operator+(const Image2D& rhs) const
    {
      assert(width() == rhs.width());
      assert(height() == rhs.height());
      Image2D res(width(), height());
      for(size_t i=0;i<m_data.size();i++)
        res.m_data[i] = m_data[i] + rhs.m_data[i];
      return res;
    }
  
    Image2D operator-(const Image2D& rhs) const
    {
      assert(width() == rhs.width());
      assert(height() == rhs.height());
      Image2D res(width(), height());
      for(size_t i=0;i<m_data.size();i++)
        res.m_data[i] = m_data[i] - rhs.m_data[i];
      return res;
    }
  
    Image2D operator*(const float a_mult) const
    {
      Image2D res(width(), height());
      for(size_t i=0;i<m_data.size();i++)
        res.m_data[i] = m_data[i]*a_mult;
      return res;
    }
  
    Image2D operator/(const float a_div) const
    {
      const float invDiv = 1.0f/a_div;
      Image2D res(width(), height());
      for(size_t i=0;i<m_data.size();i++)
        res.m_data[i] = m_data[i]*invDiv;
      return res;
    }
  
  protected: 
  
    unsigned int m_width, m_height;
    float        m_fw,    m_fh;
    std::vector<Type> m_data; 
    bool m_srgb = false;
  
  };

  template<typename Type> bool          SaveImage(const char* a_fileName, const Image2D<Type>& a_image, float a_gamma = 2.2f);
  template<typename Type> Image2D<Type> LoadImage(const char* a_fileName);

}; // end namespace


#endif