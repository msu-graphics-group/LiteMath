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

    template<typename It>
    void assign(It first, It last)
    {
      size_t actualSize = last - first;
      if(m_size != actualSize) 
      {
        if(m_data != nullptr)
          cudaFree(m_data); 
        
        cudaMalloc((void**)&m_data, actualSize*sizeof(T));
        m_size     = actualSize;
        m_capacity = actualSize;
      }

      const T* dataHost = &(*first);
      cudaMemcpy(m_data, dataHost, actualSize*sizeof(T), cudaMemcpyHostToDevice);
    }
    
    inline __host__ __device__ const T* data() const { return m_data; }
    inline __host__ __device__ T* data()             { return m_data; }
    inline __host__ __device__ size_type size()     const { return m_size; }
    inline __host__ __device__ size_type capacity() const { return m_capacity; }

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
    //void resize(size_t size);
    //void resize(size_t size, const T& value);
    //void clear();
    //void reserve(size_t capacity);
    //void push_back(const T& t);
    //void pop_back();
    //void emplace_back();
    //template<typename Param>
    //void emplace_back(const Param& param);
    //void shrink_to_fit();
    //void swap(vector& other);
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
    
    T* m_data;
    size_type m_size;
    size_type m_capacity;
  };

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

};
