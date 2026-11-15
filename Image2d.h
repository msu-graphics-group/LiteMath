#ifndef IMAGE2D_H
#define IMAGE2D_H

#include <vector>
#include <string>
#include <memory>
#include <cassert>

#include "LiteMath.h"

#ifdef LoadImage
#undef LoadImage
#endif

namespace LiteImage 
{ 
  #ifndef CUDA_MATH
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
  using LiteMath::dot;
  using LiteMath::make_float2;
  using LiteMath::make_float3;
  using LiteMath::make_float4;
  using LiteMath::make_int4;
  using LiteMath::make_uint4;
  using LiteMath::make_uchar4;
  #else
  static inline float  dot (float a, float b) { return a*b;  } 
  #endif

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
      CUBIC   = 1000015000, // VK_FILTER_CUBIC_EXT = 1000015000, not supported on CPU
    };
  
    // sampler state
    //
    AddressMode addressU      = AddressMode::WRAP;
    AddressMode addressV      = AddressMode::WRAP;
    AddressMode addressW      = AddressMode::WRAP;
    float4      borderColor   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    Filter      filter        = Filter::NEAREST;
    uint32_t    maxAnisotropy = 1;
    uint32_t    maxLOD        = 32;
    uint32_t    minLOD        = 0;
    uint32_t    mipLODBias    = 0;
  };

  template<typename T> uint32_t GetVulkanFormat(bool a_gamma22);

  /**
  \brief Memory layout of image pixels on CPU, analogue of VkImageTiling (GPU always uses its own optimal tiling).
  */
  enum class Layout : uint32_t
  {
    LINEAR   = 0, ///< row by row
    TILED4x4 = 1, ///< tiles of 4x4 pixels stored as 16 subsequent pixels (row by row inside a tile), tiles are stored row by row;
                  ///< storage is padded to a multiple of 4 in both dimensions
    TILED4x4_BINS = 2, ///< two levels: bins of B x B pixels (B = binSize(), a power of 2 >= 4) are stored one after another row by row,
                       ///< inside a bin tiles 4x4 are stored as in TILED4x4; storage is padded to a multiple of B in both dimensions
  };

  /// offset of pixel (x,y) in storage of an image with layout a_layout; a_pitch is width for LINEAR, number of tiles in a row for
  /// TILED4x4 and number of bins in a row for TILED4x4_BINS; a_binShift is log2 of bin size (TILED4x4_BINS only)
  static inline unsigned int PixelOffset(Layout a_layout, unsigned int a_pitch, unsigned int x, unsigned int y, unsigned int a_binShift = 6)
  {
    if(a_layout == Layout::LINEAR)
      return y*a_pitch + x;
    const unsigned int inTile = ((y & 3) << 2) + (x & 3);
    if(a_layout == Layout::TILED4x4)
      return (((y >> 2)*a_pitch + (x >> 2)) << 4) + inTile;
    const unsigned int mask = (1u << a_binShift) - 1u;
    const unsigned int bin  = (y >> a_binShift)*a_pitch + (x >> a_binShift);
    const unsigned int tile = ((y & mask) >> 2)*((mask + 1u) >> 2) + ((x & mask) >> 2);
    return (bin << (2*a_binShift)) + (tile << 4) + inTile;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  template<typename Type>
  struct Image2D
  {
    Image2D() : m_width(0), m_height(0), m_fw(0.0f), m_fh(0.0f) {}
    Image2D(unsigned int w, unsigned int h, const Type* a_data) : m_width(w), m_height(h), m_fw(float(w)), m_fh(float(h))
    {
      resize(w, h);
      memcpy(m_data.data(), a_data, w*h*sizeof(Type));
    }
    
    Image2D(unsigned int w, unsigned int h) : m_width(w), m_height(h) { resize(w, h); }
    Image2D(unsigned int w, unsigned int h, const Type val)           { resize(w, h); clear(val); }
  
    Image2D(const Image2D& rhs) = default;            // may cause crash on some old compilers
    Image2D(Image2D&& lhs)      = default;            // may cause crash on some old compilers
  
    Image2D& operator=(Image2D&& rhs)      = default; // may cause crash on some old compilers
    Image2D& operator=(const Image2D& rhs) = default; // may cause crash on some old compilers
    
    void resize(unsigned int w, unsigned int h) { resize(w, h, m_layout, binSize()); }
    /// a_binSize is a size of bins for TILED4x4_BINS, a power of 2 from 4 to 1024
    void resize(unsigned int w, unsigned int h, Layout a_layout, unsigned int a_binSize = 64)
    {
      m_width    = w;
      m_height   = h;
      m_layout   = a_layout;
      m_binShift = 2;
      while((1u << m_binShift) < a_binSize && m_binShift < 10)
        m_binShift++;
      const unsigned int bs = binSize();
      if(a_layout == Layout::LINEAR)
      {
        m_pitch = w;
        m_data.resize(size_t(w)*size_t(h));
      }
      else if(a_layout == Layout::TILED4x4)
      {
        m_pitch = (w + 3)/4;
        m_data.resize(size_t(m_pitch)*size_t((h + 3)/4)*16);
      }
      else
      {
        m_pitch = (w + bs - 1)/bs;
        m_data.resize(size_t(m_pitch)*size_t((h + bs - 1)/bs)*size_t(bs)*size_t(bs));
      }
      m_fw = float(w);
      m_fh = float(h);
    }
    void clear(const Type val)                  { for(auto& pixel : m_data) pixel = val; }

    /// changes layout keeping pixel values
    void setLayout(Layout a_layout, unsigned int a_binSize = 64)
    {
      if(a_layout == m_layout && (a_layout != Layout::TILED4x4_BINS || a_binSize == binSize()))
        return;
      Image2D res;
      res.resize(m_width, m_height, a_layout, a_binSize);
      for(unsigned int y = 0; y < m_height; y++)
        for(unsigned int x = 0; x < m_width; x++)
          res.m_data[res.offset(x, y)] = m_data[offset(x, y)];
      res.m_srgb = m_srgb;
      *this = res;
    }

    float4 sample(const Sampler& a_sampler, float2 a_uv) const;

    unsigned int offset(unsigned int x, unsigned int y) const { return PixelOffset(m_layout, m_pitch, x, y, m_binShift); } ///< offset of pixel (x,y) in data()

    Type&  operator[](const int2 coord)        { return m_data[offset(coord.x, coord.y)]; }
    Type   operator[](const int2 coord) const  { return m_data[offset(coord.x, coord.y)]; }
    Type&  operator[](const uint2 coord)       { return m_data[offset(coord.x, coord.y)]; }
    Type   operator[](const uint2 coord) const { return m_data[offset(coord.x, coord.y)]; }

    unsigned int width()  const { return m_width; }
    unsigned int height() const { return m_height; }
    unsigned int bpp()    const { return sizeof(Type); }
    Layout       layout() const { return m_layout; }
    unsigned int pitch()  const { return m_pitch; } ///< width for LINEAR, number of tiles in a row for TILED4x4, number of bins in a row for TILED4x4_BINS
    unsigned int binSize()  const { return 1u << m_binShift; } ///< size of bins of TILED4x4_BINS
    unsigned int binShift() const { return m_binShift; }       ///< log2(binSize())
   
          Type*              data()         { return m_data.data(); }
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
      assert(layout() == rhs.layout());
      Image2D res;
      res.resize(width(), height(), m_layout);
      for(size_t i=0;i<m_data.size();i++)
        res.m_data[i] = m_data[i] + rhs.m_data[i];
      return res;
    }
  
    Image2D operator-(const Image2D& rhs) const
    {
      assert(width() == rhs.width());
      assert(height() == rhs.height());
      assert(layout() == rhs.layout());
      Image2D res;
      res.resize(width(), height(), m_layout);
      for(size_t i=0;i<m_data.size();i++)
        res.m_data[i] = m_data[i] - rhs.m_data[i];
      return res;
    }
  
    Image2D operator*(const float a_mult) const
    {
      Image2D res;
      res.resize(width(), height(), m_layout);
      for(size_t i=0;i<m_data.size();i++)
        res.m_data[i] = m_data[i]*a_mult;
      return res;
    }
  
    Image2D operator/(const float a_div) const
    {
      const float invDiv = 1.0f/a_div;
      Image2D res;
      res.resize(width(), height(), m_layout);
      for(size_t i=0;i<m_data.size();i++)
        res.m_data[i] = m_data[i]*invDiv;
      return res;
    }
  
  protected: 
  
    unsigned int m_width = 0, m_height = 0;
    float        m_fw = 0.0f,    m_fh = 0.0f;
    std::vector<Type> m_data; 
    bool m_srgb = false;
    Layout m_layout = Layout::LINEAR;
    unsigned int m_pitch = 0;
    unsigned int m_binShift = 6;
  
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
  \brief combined image of unknown type (could be any) and sampler
  */
  struct ICombinedImageSampler 
  {
    virtual ~ICombinedImageSampler() {}
    virtual float4       sample(float2 a_uv) const = 0;  
    
    virtual unsigned int width()             const = 0;
    virtual unsigned int height()            const = 0;
    virtual unsigned int depth()             const = 0;
    virtual unsigned int bpp()               const = 0;
    virtual unsigned int format()            const = 0; ///<! return uint(VkFormat) value 
    
    virtual Sampler      sampler()           const = 0;
    virtual const void*  data()              const = 0;
  };

  /**
    \brief Can have specific effitient implementations for various textures and samplers
  */
  template<typename Type>
  std::shared_ptr<ICombinedImageSampler> MakeCombinedTexture2D(std::shared_ptr< Image2D<Type> > a_texture, Sampler a_sampler) 
  { 
    // general implementation
    //
    struct CobminedImageSamplerGeneral : public ICombinedImageSampler
    {
      ~CobminedImageSamplerGeneral(){}
      float4 sample(float2 a_uv) const override { return m_pTexture->sample(m_sampler, a_uv); }  
    
      unsigned int width()       const override { return m_pTexture->width();  }
      unsigned int height()      const override { return m_pTexture->height(); }
      unsigned int depth()       const override { return 1; }
      unsigned int bpp()         const override { return m_pTexture->bpp();    }
      unsigned int format()      const override { return m_pTexture->format();  } 

      Sampler      sampler()     const override { return m_sampler; }
      const void*  data()        const override { return m_pTexture->data(); }
  
      std::shared_ptr<Image2D<Type> > m_pTexture;
      Sampler                         m_sampler;
    };
  
    std::shared_ptr<CobminedImageSamplerGeneral> pObj = std::make_shared<CobminedImageSamplerGeneral>();
    pObj->m_pTexture = a_texture;
    pObj->m_sampler  = a_sampler;
    return pObj;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  template<typename Type> bool          SaveImage(const char* a_fileName, const Image2D<Type>& a_image, float a_gamma = 2.2f);
  template<typename Type> Image2D<Type> LoadImage(const char* a_fileName, float a_gamma = 2.2f);
  
  template<typename Type> float extDotProd(Type a, Type b) { return dot(a,b); }
  template<> inline       float extDotProd<uint32_t>(uint32_t a, uint32_t b) 
  { 
    unsigned r1 = (a & 0x000000FF);
    unsigned g1 = (a & 0x0000FF00) >> 8;
    unsigned b1 = (a & 0x00FF0000) >> 16;
    unsigned r2 = (b & 0x000000FF);
    unsigned g2 = (b & 0x0000FF00) >> 8;
    unsigned b2 = (b & 0x00FF0000) >> 16;
    return float(r1*r2 + g1*g2 + b1*b2); 
  }
  
  template<typename T>    float MSE(const std::vector<T> image1, const std::vector<T> image2)
  {
    if (image1.size() != image2.size())
      return std::max<float>(100000.0f, float(image1.size()+image2.size()));

    double summ = 0;
    //#pragma omp parallel for reduction(+:summ)
    for (size_t i = 0; i < image1.size(); i++)
    {
      auto di = (image2[i] - image1[i]);
      summ += double(extDotProd(di,di));
    }
    return float(summ/double(image1.size()));
  }

  /// pixels of an image in LINEAR layout (copy is made only for other layouts)
  template<typename T> std::vector<T> LinearPixels(const Image2D<T>& a_image)
  {
    if(a_image.layout() == Layout::LINEAR)
      return a_image.vector();
    Image2D<T> linear = a_image;
    linear.setLayout(Layout::LINEAR);
    return linear.vector();
  }

  template<typename T> float MSE(const Image2D<T>& a, const Image2D<T>& b) { return MSE(LinearPixels(a), LinearPixels(b))/3.0f; }
  template<> inline    float MSE<float>(const Image2D<float>& a, const Image2D<float>& b) { return MSE(LinearPixels(a), LinearPixels(b)); }
  
  /**
  \brief save 24 bit RGB bitmap images.
  \param a      - input image1 (I)
  \param b      - input image2 (I_target)
  \return per pixel loss function: (I[x,y] - I_target[x,y])^2
  */
  template<typename T> Image2D<T> MSEImage(const Image2D<T>& a, const Image2D<T>& b) 
  { 
    assert(a.width()*a.height() == b.width()*b.height());
    Image2D<T> result(a.width(), a.height());
    const size_t imgSize = a.width()*a.height();
    for(size_t i=0;i<imgSize;i++) {
      const auto av = a.data()[i];
      const auto bv = b.data()[i];
      result.data()[i] = (bv - av)*(bv - av);
    }
    return result;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  /**
  \brief save 24 bit RGB bitmap images.
  \param fname  - file name
  \param w      - input image width
  \param h      - input image height
  \param pixels - R8G8B8A8 data, 4 bytes per pixel, one byte for channel.
  */
  bool SaveBMP(const char* fname, const unsigned int* pixels, int w, int h);

  /**
  \brief load 24 bit RGB bitmap images.
  \param fname - file name
  \param pW    - out image width
  \param pH    - out image height
  \return R8G8B8A8 data, 4 bytes per pixel, one byte for channel.
          Note that this function in this sample works correctly _ONLY_ for 24 bit RGB ".bmp" images.
          If you want to support gray-scale images or images with palette, please upgrade its implementation.
  */
  std::vector<unsigned int> LoadBMP(const char* filename, int* pW, int* pH);

}; // end namespace

#endif
