#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include "LiteMath.h"
using namespace LiteMath;

template<typename T>
static bool IsClose(T a, T b, T eps = T(1e-5)) { return std::abs(a - b) <= eps; }

bool test400_mat4x4_float()
{
  const float A[16] = { 1, 2, 3, 4,
                       5, 6, 7, 8,
                       9, 1, 2, 3,
                       4, 5, 9, 7 }; // row-major

  const float4x4 m1(A);
  const float4x4 m2(1, 2, 3, 4,
                    5, 6, 7, 8,
                    9, 1, 2, 3,
                    4, 5, 9, 7);
  const float4x4 m3 = make_float4x4_from_rows(float4(1,2,3,4), float4(5,6,7,8), float4(9,1,2,3), float4(4,5,9,7));
  const float4x4 m4 = make_float4x4_from_cols(float4(1,5,9,4), float4(2,6,1,5), float4(3,7,2,9), float4(4,8,3,7));
  const float4x4 mI;
  const float4x4 mT = transpose(m1);
  const float4x4 mS = m1 + mT;
  const float4x4 mD = m1 - mT;
  const float4x4 mM = m1*mT;
  const float4x4 mM2 = mul(m1, mT);

  float4x4 m5;
  m5.identity();
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      m5(i,j) = A[i*4+j];

  float4x4 m6;
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      m6[i][j] = A[i*4+j];

  bool passed = true;
  for(int i=0;i<4;i++)
  {
    const float4 row = m1.get_row(i);
    const float4 col = m1.get_col(i);
    const float4 cl2 = m1.col(i);
    const auto rowAccess = m6[i]; // const RowTmp
    for(int j=0;j<4;j++)
    {
      const float ref = A[i*4+j];
      if(m1(i,j) != ref || m2(i,j) != ref || m3(i,j) != ref || m4(i,j) != ref || m5(i,j) != ref || rowAccess[j] != ref)
        passed = false;
      if(row[j] != ref || col[j] != A[j*4+i] || cl2[j] != A[j*4+i])
        passed = false;
      if(mI(i,j) != (i == j ? float(1) : float(0)))
        passed = false;
      if(mT(i,j) != A[j*4+i] || mS(i,j) != A[i*4+j] + A[j*4+i] || mD(i,j) != A[i*4+j] - A[j*4+i])
        passed = false;
      float sum = 0;
      for(int k=0;k<4;k++)
        sum += A[i*4+k]*A[j*4+k];
      if(mM(i,j) != sum || mM2(i,j) != sum)
        passed = false;
    }
  }

  // raw row-major matrix product
  float AA[16];
  mat4_rowmajor_mul_mat4(AA, A, A);
  for(int i=0;i<4;i++)
  {
    for(int j=0;j<4;j++)
    {
      float sum = 0;
      for(int k=0;k<4;k++)
        sum += A[i*4+k]*A[k*4+j];
      if(AA[i*4+j] != sum)
        passed = false;
    }
  }

  // matrix-vector
  const float4 v(1, -2, 3, -4);
  const float4 r1 = m1*v;
  const float4 r2 = mul(m1, v);
  const float4 r3 = mul4x4x4(m1, v);
  for(int i=0;i<4;i++)
  {
    const float ref = A[i*4+0]*v.x + A[i*4+1]*v.y + A[i*4+2]*v.z + A[i*4+3]*v.w;
    if(r1[i] != ref || r2[i] != ref || r3[i] != ref)
      passed = false;
  }

  // outer product
  const float4x4 mO = outerProduct(v, float4(1,2,3,4));
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      if(mO(i,j) != v[i]*float(j+1))
        passed = false;

  // affine transforms
  const float3 p(1, 2, 3);
  const float4x4 mTr = translate4x4(float3(10, 20, 30));
  const float4x4 mSc = scale4x4(float3(2, 3, 4));
  const float3 p1 = mul4x3(mTr, p);
  const float3 p2 = mul3x3(mTr, p);
  const float3 p3 = mul4x3(mSc, p);
  passed = passed && (p1.x == 11) && (p1.y == 22) && (p1.z == 33);
  passed = passed && (p2.x == 1)  && (p2.y == 2)  && (p2.z == 3);
  passed = passed && (p3.x == 2)  && (p3.y == 6)  && (p3.z == 12);

  // rotations
  const float angle = float(M_PI)*float(0.5);
  const float3 rx = mul3x3(rotate4x4X(angle), float3(0,1,0));
  const float3 ry = mul3x3(rotate4x4Y(angle), float3(0,0,1));
  const float3 rz = mul3x3(rotate4x4Z(angle), float3(0,1,0));
  passed = passed && length(rx - float3(0,0,1))  < 1e-6f;
  passed = passed && length(ry - float3(1,0,0))  < 1e-6f;
  passed = passed && length(rz - float3(-1,0,0)) < 1e-6f;

  // inverse
  const float4x4 mC = mTr*rotate4x4Y(float(0.3))*mSc;
  const float4x4 mCI = inverse4x4(mC);
  const float4x4 mCheck1 = mCI*mC;
  const float4x4 mCheck2 = inverse4x4(m1)*m1;
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      if(!IsClose(mCheck1(i,j), mI(i,j)) || !IsClose(mCheck2(i,j), mI(i,j)))
        passed = false;

  return passed;
}

bool test401_mat3x3_float()
{
  const float A[9] = { 1, 2, 3,
                      0, 1, 4,
                      5, 6, 0 }; // row-major, det = 1

  const float AInv[9] = { -24,  18,  5,
                          20, -15, -4,
                          -5,   4,  1 };

  const float3x3 m1(A);
  const float3x3 m2(1, 2, 3,
                    0, 1, 4,
                    5, 6, 0);
  const float3x3 m3 = make_float3x3_from_rows(float3(1,2,3), float3(0,1,4), float3(5,6,0));
  const float3x3 m4 = make_float3x3_from_cols(float3(1,0,5), float3(2,1,6), float3(3,4,0));
  const float3x3 m5 = make_float3x3(float3(1,2,3), float3(0,1,4), float3(5,6,0));
  const float3x3 m6 = make_float3x3_by_columns(float3(1,0,5), float3(2,1,6), float3(3,4,0));
  const float3x3 m7(m1);
  const float3x3 mI;
  const float3x3 mK(float(2));

  float3x3 mZ;
  mZ.zero();
  float3x3 mI2(mK);
  mI2.identity();

  float3x3 m8;
  for(int i=0;i<3;i++)
  {
    m8.set_col(i, float3(float(0)));
    m8.col(i).x = A[0*3+i];
    for(int j=1;j<3;j++)
      m8(j,i) = A[j*3+i];
  }

  float3x3 m9;
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      m9[i][j] = A[i*3+j];

  const float3x3 mT   = transpose(m1);
  const float3x3 mS   = m1 + mT;
  const float3x3 mD   = m1 - mT;
  const float3x3 mM   = m1*mT;
  const float3x3 mM2  = mul(m1, mT);
  const float3x3 mK1  = m1*float(3);
  const float3x3 mK2  = float(3)*m1;
  const float3x3 mInv = inverse3x3(m1);
  const float    det  = determinant(m1);

  bool passed = IsClose(det, float(1));
  for(int i=0;i<3;i++)
  {
    const float3 row = m1.get_row(i);
    const float3 col = m1.get_col(i);
    const float3 cl2 = m1.col(i);
    const auto rowAccess = m9[i]; // const RowTmp
    for(int j=0;j<3;j++)
    {
      const float ref = A[i*3+j];
      if(m1(i,j) != ref || m2(i,j) != ref || m3(i,j) != ref || m4(i,j) != ref || m5(i,j) != ref || m6(i,j) != ref || m7(i,j) != ref || m8(i,j) != ref || rowAccess[j] != ref)
        passed = false;
      if(row[j] != ref || col[j] != A[j*3+i] || cl2[j] != A[j*3+i])
        passed = false;
      const float id = (i == j) ? float(1) : float(0);
      if(mI(i,j) != id || mI2(i,j) != id || mZ(i,j) != 0 || mK(i,j) != 2)
        passed = false;
      if(mT(i,j) != A[j*3+i] || mS(i,j) != A[i*3+j] + A[j*3+i] || mD(i,j) != A[i*3+j] - A[j*3+i])
        passed = false;
      if(mK1(i,j) != 3*ref || mK2(i,j) != 3*ref || !IsClose(mInv(i,j), AInv[i*3+j], float(1e-4)))
        passed = false;
      float sum = 0;
      for(int k=0;k<3;k++)
        sum += A[i*3+k]*A[j*3+k];
      if(mM(i,j) != sum || mM2(i,j) != sum)
        passed = false;
    }
  }

  // matrix-vector
  const float3 v(1, -2, 3);
  const float3 r1 = m1*v;
  const float3 r2 = mul(m1, v);
  for(int i=0;i<3;i++)
  {
    const float ref = A[i*3+0]*v.x + A[i*3+1]*v.y + A[i*3+2]*v.z;
    if(r1[i] != ref || r2[i] != ref)
      passed = false;
  }

  // outer product
  const float3x3 mO = outerProduct(v, float3(1,2,3));
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      if(mO(i,j) != v[i]*float(j+1))
        passed = false;

  // scale and rotations
  const float3 p1 = scale3x3(float3(2,3,4))*float3(1,2,3);
  passed = passed && (p1.x == 2) && (p1.y == 6) && (p1.z == 12);

  const float angle = float(M_PI)*float(0.5);
  const float3 rx = rotate3x3X(angle)*float3(0,1,0);
  const float3 ry = rotate3x3Y(angle)*float3(0,0,1);
  const float3 rz = rotate3x3Z(angle)*float3(0,1,0);
  passed = passed && length(rx - float3(0,0,1))  < 1e-6f;
  passed = passed && length(ry - float3(1,0,0))  < 1e-6f;
  passed = passed && length(rz - float3(-1,0,0)) < 1e-6f;

  return passed;
}

bool test402_complex_float()
{
  const complex z0;
  const complex z1(float(3));
  const complex z2(1, 2);
  const complex z3(3, 4);

  const complex c1 = -z2;
  const complex c2 = z2 + z3;
  const complex c3 = z2 - z3;
  const complex c4 = z2 * z3;  // -5 + 10i
  const complex c5 = c4 / z3;  //  1 +  2i
  const complex c6 = float(2) + z2;
  const complex c7 = float(2) - z2;
  const complex c8 = float(2) * z2;
  const complex c9 = float(5) / z2; // 1 - 2i

  bool passed = true;
  passed = passed && (real(z0) == 0)  && (imag(z0) == 0);
  passed = passed && (real(z1) == 3)  && (imag(z1) == 0);
  passed = passed && (real(c1) == -1) && (imag(c1) == -2);
  passed = passed && (real(c2) == 4)  && (imag(c2) == 6);
  passed = passed && (real(c3) == -2) && (imag(c3) == -2);
  passed = passed && (real(c4) == -5) && (imag(c4) == 10);
  passed = passed && IsClose(real(c5), float(1)) && IsClose(imag(c5), float(2));
  passed = passed && (real(c6) == 3)  && (imag(c6) == 2);
  passed = passed && (real(c7) == 1)  && (imag(c7) == -2);
  passed = passed && (real(c8) == 2)  && (imag(c8) == 4);
  passed = passed && IsClose(real(c9), float(1)) && IsClose(imag(c9), float(-2));

  passed = passed && (complex_norm(z3) == 25) && IsClose(complex_abs(z3), float(5));

  const complex s0 = complex_sqrt(z0);
  const complex s1 = complex_sqrt(z3);                         // 2 + i
  const complex s2 = complex_sqrt(complex(-3,  4)); // 1 + 2i
  const complex s3 = complex_sqrt(complex(-3, -4)); // 1 - 2i
  passed = passed && (real(s0) == 0) && (imag(s0) == 0);
  passed = passed && IsClose(real(s1), float(2)) && IsClose(imag(s1), float(1));
  passed = passed && IsClose(real(s2), float(1)) && IsClose(imag(s2), float(2));
  passed = passed && IsClose(real(s3), float(1)) && IsClose(imag(s3), float(-2));

  return passed;
}

bool test410_mat4x4_double()
{
  const double A[16] = { 1, 2, 3, 4,
                       5, 6, 7, 8,
                       9, 1, 2, 3,
                       4, 5, 9, 7 }; // row-major

  const double4x4 m1(A);
  const double4x4 m2(1, 2, 3, 4,
                    5, 6, 7, 8,
                    9, 1, 2, 3,
                    4, 5, 9, 7);
  const double4x4 m3 = make_double4x4_from_rows(double4(1,2,3,4), double4(5,6,7,8), double4(9,1,2,3), double4(4,5,9,7));
  const double4x4 m4 = make_double4x4_from_cols(double4(1,5,9,4), double4(2,6,1,5), double4(3,7,2,9), double4(4,8,3,7));
  const double4x4 mI;
  const double4x4 mT = transpose(m1);
  const double4x4 mS = m1 + mT;
  const double4x4 mD = m1 - mT;
  const double4x4 mM = m1*mT;
  const double4x4 mM2 = mul(m1, mT);

  double4x4 m5;
  m5.identity();
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      m5(i,j) = A[i*4+j];

  double4x4 m6;
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      m6[i][j] = A[i*4+j];

  bool passed = true;
  for(int i=0;i<4;i++)
  {
    const double4 row = m1.get_row(i);
    const double4 col = m1.get_col(i);
    const double4 cl2 = m1.col(i);
    const auto rowAccess = m6[i]; // const RowTmp
    for(int j=0;j<4;j++)
    {
      const double ref = A[i*4+j];
      if(m1(i,j) != ref || m2(i,j) != ref || m3(i,j) != ref || m4(i,j) != ref || m5(i,j) != ref || rowAccess[j] != ref)
        passed = false;
      if(row[j] != ref || col[j] != A[j*4+i] || cl2[j] != A[j*4+i])
        passed = false;
      if(mI(i,j) != (i == j ? double(1) : double(0)))
        passed = false;
      if(mT(i,j) != A[j*4+i] || mS(i,j) != A[i*4+j] + A[j*4+i] || mD(i,j) != A[i*4+j] - A[j*4+i])
        passed = false;
      double sum = 0;
      for(int k=0;k<4;k++)
        sum += A[i*4+k]*A[j*4+k];
      if(mM(i,j) != sum || mM2(i,j) != sum)
        passed = false;
    }
  }

  // raw row-major matrix product
  double AA[16];
  mat4_rowmajor_mul_mat4(AA, A, A);
  for(int i=0;i<4;i++)
  {
    for(int j=0;j<4;j++)
    {
      double sum = 0;
      for(int k=0;k<4;k++)
        sum += A[i*4+k]*A[k*4+j];
      if(AA[i*4+j] != sum)
        passed = false;
    }
  }

  // matrix-vector
  const double4 v(1, -2, 3, -4);
  const double4 r1 = m1*v;
  const double4 r2 = mul(m1, v);
  const double4 r3 = mul4x4x4(m1, v);
  for(int i=0;i<4;i++)
  {
    const double ref = A[i*4+0]*v.x + A[i*4+1]*v.y + A[i*4+2]*v.z + A[i*4+3]*v.w;
    if(r1[i] != ref || r2[i] != ref || r3[i] != ref)
      passed = false;
  }

  // outer product
  const double4x4 mO = outerProduct(v, double4(1,2,3,4));
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      if(mO(i,j) != v[i]*double(j+1))
        passed = false;

  // affine transforms
  const double3 p(1, 2, 3);
  const double4x4 mTr = translate4x4(double3(10, 20, 30));
  const double4x4 mSc = scale4x4(double3(2, 3, 4));
  const double3 p1 = mul4x3(mTr, p);
  const double3 p2 = mul3x3(mTr, p);
  const double3 p3 = mul4x3(mSc, p);
  passed = passed && (p1.x == 11) && (p1.y == 22) && (p1.z == 33);
  passed = passed && (p2.x == 1)  && (p2.y == 2)  && (p2.z == 3);
  passed = passed && (p3.x == 2)  && (p3.y == 6)  && (p3.z == 12);

  // rotations
  const double angle = double(M_PI)*double(0.5);
  const double3 rx = mul3x3(rotate4x4X(angle), double3(0,1,0));
  const double3 ry = mul3x3(rotate4x4Y(angle), double3(0,0,1));
  const double3 rz = mul3x3(rotate4x4Z(angle), double3(0,1,0));
  passed = passed && length(rx - double3(0,0,1))  < 1e-6f;
  passed = passed && length(ry - double3(1,0,0))  < 1e-6f;
  passed = passed && length(rz - double3(-1,0,0)) < 1e-6f;

  // inverse
  const double4x4 mC = mTr*rotate4x4Y(double(0.3))*mSc;
  const double4x4 mCI = inverse4x4(mC);
  const double4x4 mCheck1 = mCI*mC;
  const double4x4 mCheck2 = inverse4x4(m1)*m1;
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      if(!IsClose(mCheck1(i,j), mI(i,j)) || !IsClose(mCheck2(i,j), mI(i,j)))
        passed = false;

  return passed;
}

bool test411_mat3x3_double()
{
  const double A[9] = { 1, 2, 3,
                      0, 1, 4,
                      5, 6, 0 }; // row-major, det = 1

  const double AInv[9] = { -24,  18,  5,
                          20, -15, -4,
                          -5,   4,  1 };

  const double3x3 m1(A);
  const double3x3 m2(1, 2, 3,
                    0, 1, 4,
                    5, 6, 0);
  const double3x3 m3 = make_double3x3_from_rows(double3(1,2,3), double3(0,1,4), double3(5,6,0));
  const double3x3 m4 = make_double3x3_from_cols(double3(1,0,5), double3(2,1,6), double3(3,4,0));
  const double3x3 m5 = make_double3x3(double3(1,2,3), double3(0,1,4), double3(5,6,0));
  const double3x3 m6 = make_double3x3_by_columns(double3(1,0,5), double3(2,1,6), double3(3,4,0));
  const double3x3 m7(m1);
  const double3x3 mI;
  const double3x3 mK(double(2));

  double3x3 mZ;
  mZ.zero();
  double3x3 mI2(mK);
  mI2.identity();

  double3x3 m8;
  for(int i=0;i<3;i++)
  {
    m8.set_col(i, double3(double(0)));
    m8.col(i).x = A[0*3+i];
    for(int j=1;j<3;j++)
      m8(j,i) = A[j*3+i];
  }

  double3x3 m9;
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      m9[i][j] = A[i*3+j];

  const double3x3 mT   = transpose(m1);
  const double3x3 mS   = m1 + mT;
  const double3x3 mD   = m1 - mT;
  const double3x3 mM   = m1*mT;
  const double3x3 mM2  = mul(m1, mT);
  const double3x3 mK1  = m1*double(3);
  const double3x3 mK2  = double(3)*m1;
  const double3x3 mInv = inverse3x3(m1);
  const double    det  = determinant(m1);

  bool passed = IsClose(det, double(1));
  for(int i=0;i<3;i++)
  {
    const double3 row = m1.get_row(i);
    const double3 col = m1.get_col(i);
    const double3 cl2 = m1.col(i);
    const auto rowAccess = m9[i]; // const RowTmp
    for(int j=0;j<3;j++)
    {
      const double ref = A[i*3+j];
      if(m1(i,j) != ref || m2(i,j) != ref || m3(i,j) != ref || m4(i,j) != ref || m5(i,j) != ref || m6(i,j) != ref || m7(i,j) != ref || m8(i,j) != ref || rowAccess[j] != ref)
        passed = false;
      if(row[j] != ref || col[j] != A[j*3+i] || cl2[j] != A[j*3+i])
        passed = false;
      const double id = (i == j) ? double(1) : double(0);
      if(mI(i,j) != id || mI2(i,j) != id || mZ(i,j) != 0 || mK(i,j) != 2)
        passed = false;
      if(mT(i,j) != A[j*3+i] || mS(i,j) != A[i*3+j] + A[j*3+i] || mD(i,j) != A[i*3+j] - A[j*3+i])
        passed = false;
      if(mK1(i,j) != 3*ref || mK2(i,j) != 3*ref || !IsClose(mInv(i,j), AInv[i*3+j], double(1e-4)))
        passed = false;
      double sum = 0;
      for(int k=0;k<3;k++)
        sum += A[i*3+k]*A[j*3+k];
      if(mM(i,j) != sum || mM2(i,j) != sum)
        passed = false;
    }
  }

  // matrix-vector
  const double3 v(1, -2, 3);
  const double3 r1 = m1*v;
  const double3 r2 = mul(m1, v);
  for(int i=0;i<3;i++)
  {
    const double ref = A[i*3+0]*v.x + A[i*3+1]*v.y + A[i*3+2]*v.z;
    if(r1[i] != ref || r2[i] != ref)
      passed = false;
  }

  // outer product
  const double3x3 mO = outerProduct(v, double3(1,2,3));
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      if(mO(i,j) != v[i]*double(j+1))
        passed = false;

  // scale and rotations
  const double3 p1 = scale3x3(double3(2,3,4))*double3(1,2,3);
  passed = passed && (p1.x == 2) && (p1.y == 6) && (p1.z == 12);

  const double angle = double(M_PI)*double(0.5);
  const double3 rx = rotate3x3X(angle)*double3(0,1,0);
  const double3 ry = rotate3x3Y(angle)*double3(0,0,1);
  const double3 rz = rotate3x3Z(angle)*double3(0,1,0);
  passed = passed && length(rx - double3(0,0,1))  < 1e-6f;
  passed = passed && length(ry - double3(1,0,0))  < 1e-6f;
  passed = passed && length(rz - double3(-1,0,0)) < 1e-6f;

  return passed;
}

bool test412_complex_double()
{
  const complexd z0;
  const complexd z1(double(3));
  const complexd z2(1, 2);
  const complexd z3(3, 4);

  const complexd c1 = -z2;
  const complexd c2 = z2 + z3;
  const complexd c3 = z2 - z3;
  const complexd c4 = z2 * z3;  // -5 + 10i
  const complexd c5 = c4 / z3;  //  1 +  2i
  const complexd c6 = double(2) + z2;
  const complexd c7 = double(2) - z2;
  const complexd c8 = double(2) * z2;
  const complexd c9 = double(5) / z2; // 1 - 2i

  bool passed = true;
  passed = passed && (real(z0) == 0)  && (imag(z0) == 0);
  passed = passed && (real(z1) == 3)  && (imag(z1) == 0);
  passed = passed && (real(c1) == -1) && (imag(c1) == -2);
  passed = passed && (real(c2) == 4)  && (imag(c2) == 6);
  passed = passed && (real(c3) == -2) && (imag(c3) == -2);
  passed = passed && (real(c4) == -5) && (imag(c4) == 10);
  passed = passed && IsClose(real(c5), double(1)) && IsClose(imag(c5), double(2));
  passed = passed && (real(c6) == 3)  && (imag(c6) == 2);
  passed = passed && (real(c7) == 1)  && (imag(c7) == -2);
  passed = passed && (real(c8) == 2)  && (imag(c8) == 4);
  passed = passed && IsClose(real(c9), double(1)) && IsClose(imag(c9), double(-2));

  passed = passed && (complex_norm(z3) == 25) && IsClose(complex_abs(z3), double(5));

  const complexd s0 = complex_sqrt(z0);
  const complexd s1 = complex_sqrt(z3);                         // 2 + i
  const complexd s2 = complex_sqrt(complexd(-3,  4)); // 1 + 2i
  const complexd s3 = complex_sqrt(complexd(-3, -4)); // 1 - 2i
  passed = passed && (real(s0) == 0) && (imag(s0) == 0);
  passed = passed && IsClose(real(s1), double(2)) && IsClose(imag(s1), double(1));
  passed = passed && IsClose(real(s2), double(1)) && IsClose(imag(s2), double(2));
  passed = passed && IsClose(real(s3), double(1)) && IsClose(imag(s3), double(-2));

  return passed;
}


