#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace LiteMathExtended
{
  template<typename T, typename a_size_type = uint32_t>
  class device_vector 
  {
  public:

    using size_type = a_size_type;
    
    inline __host__ __device__ const T* data() const { return m_data; }
    inline __host__ __device__ T* data()             { return m_data; }
    inline __host__ __device__ size_type size()     const { return m_size; }
    inline __host__ __device__ size_type capacity() const { return m_capacity; }
    
    inline __host__ __device__ T& operator[](size_t idx)       { return m_data[idx]; }
    inline __host__ __device__ T  operator[](size_t idx) const { return m_data[idx]; }

    template<typename It>
    void assign(It first, It last)
    {
      size_t actualSize = last - first;
      if(m_size > m_capacity) 
      {
        if(m_data != nullptr)
          cudaFree(m_data); 
        
        cudaMalloc((void**)&m_data, actualSize*sizeof(T));
        m_size     = actualSize;
        m_capacity = actualSize;
      }
      
      if(actualSize != 0)
      {
        const T* dataHost = &(*first);
        cudaMemcpy(m_data, dataHost, actualSize*sizeof(T), cudaMemcpyHostToDevice);
      }

      m_size = actualSize;
    }
    
    inline __host__ __device__ void resize(size_t a_size) { m_size = a_size; }
    inline __host__            void reserve(size_t capacity)
    {
      if(m_data != nullptr)
        cudaFree(m_data); 

      if(m_capacity < capacity)
      {
        cudaMalloc((void**)&m_data, capacity*sizeof(T));
        m_capacity = size_type(capacity);
      }
    }

    void shrink_to_fit()
    {
      if(m_size == 0)
      {
        cudaFree(m_data); 
        m_data = nullptr;
        m_capacity = 0;
      }
      else // todo: implement
      {

      }  
    }

    inline __host__ __device__ void push_back(const T& t)
    {
      #ifdef __CUDA_ARCH__
      auto oldSize = atomicAdd(&m_size, 1);
      if(oldSize < m_capacity)
        m_data[oldSize] = t;
      #else
      if(m_size < m_capacity)
      {
        #pragma omp critical
        {
          m_data[m_size] = t;
          m_size++;
        }
      }
      #endif
    }

    //device_vector(const device_vector& other);
    //device_vector(device_vector&& other);
    //device_vector(size_t size);
    //device_vector(size_t size, const T& value);
    //device_vector(const T* first, const T* last);
    //device_vector(initializer_list<T> l);

    //device_vector& operator=(const device_vector& other);
    //device_vector& operator=(vector&& other);
   
    //bool empty() const;
    //T& operator[](size_t idx);
    //const T& operator[](size_t idx) const;
    //const T& front() const;
    //T& front();
    //const T& back() const;
    //T& back();
    //void resize(size_t size, const T& value);
    //void clear();
    //void pop_back();
    //void emplace_back();
    //template<typename Param>
    //void emplace_back(const Param& param);
    
    //typedef T value_type;
    //typedef T* iterator;
    //iterator begin();
    //iterator end();
    //typedef const T* const_iterator;
    //const_iterator begin() const;
    //const_iterator end() const;
    //void insert(iterator where);
    //void insert(iterator where, const T& value);
    //void insert(iterator where, const T* first, const T* last);
    //template<typename Param>
    //void emplace(iterator where, const Param& param);
    //iterator erase(iterator where);
    //iterator erase(iterator first, iterator last);
    //iterator erase_unordered(iterator where);
    //iterator erase_unordered(iterator first, iterator last);
    
    #ifdef __CUDA_ARCH__
    T* m_data;
    size_type m_size;
    size_type m_capacity;
    #else
    T* m_data            = nullptr;
    size_type m_size     = 0;
    size_type m_capacity = 0;
    #endif
  };

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

};
