#include <iostream>
#include <iomanip>      // std::setfill, std::setw

#include "LiteMath.h"
using namespace LiteMath;

template<typename T>
static bool IsClose(T a, T b, T eps = T(1e-5)) { return std::abs(a - b) <= eps; }

## for FType in MatTypes
bool test{{FType.Number}}_mat4x4_{{FType.Name}}()
{
  const {{FType.Name}} A[16] = { 1, 2, 3, 4,
                       5, 6, 7, 8,
                       9, 1, 2, 3,
                       4, 5, 9, 7 }; // row-major

  const {{FType.Name}}4x4 m1(A);
  const {{FType.Name}}4x4 m2(1, 2, 3, 4,
                    5, 6, 7, 8,
                    9, 1, 2, 3,
                    4, 5, 9, 7);
  const {{FType.Name}}4x4 m3 = make_{{FType.Name}}4x4_from_rows({{FType.Name}}4(1,2,3,4), {{FType.Name}}4(5,6,7,8), {{FType.Name}}4(9,1,2,3), {{FType.Name}}4(4,5,9,7));
  const {{FType.Name}}4x4 m4 = make_{{FType.Name}}4x4_from_cols({{FType.Name}}4(1,5,9,4), {{FType.Name}}4(2,6,1,5), {{FType.Name}}4(3,7,2,9), {{FType.Name}}4(4,8,3,7));
  const {{FType.Name}}4x4 mI;
  const {{FType.Name}}4x4 mT = transpose(m1);
  const {{FType.Name}}4x4 mS = m1 + mT;
  const {{FType.Name}}4x4 mD = m1 - mT;
  const {{FType.Name}}4x4 mM = m1*mT;
  const {{FType.Name}}4x4 mM2 = mul(m1, mT);

  {{FType.Name}}4x4 m5;
  m5.identity();
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      m5(i,j) = A[i*4+j];

  {{FType.Name}}4x4 m6;
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      m6[i][j] = A[i*4+j];

  bool passed = true;
  for(int i=0;i<4;i++)
  {
    const {{FType.Name}}4 row = m1.get_row(i);
    const {{FType.Name}}4 col = m1.get_col(i);
    const {{FType.Name}}4 cl2 = m1.col(i);
    const auto rowAccess = m6[i]; // const RowTmp
    for(int j=0;j<4;j++)
    {
      const {{FType.Name}} ref = A[i*4+j];
      if(m1(i,j) != ref || m2(i,j) != ref || m3(i,j) != ref || m4(i,j) != ref || m5(i,j) != ref || rowAccess[j] != ref)
        passed = false;
      if(row[j] != ref || col[j] != A[j*4+i] || cl2[j] != A[j*4+i])
        passed = false;
      if(mI(i,j) != (i == j ? {{FType.Name}}(1) : {{FType.Name}}(0)))
        passed = false;
      if(mT(i,j) != A[j*4+i] || mS(i,j) != A[i*4+j] + A[j*4+i] || mD(i,j) != A[i*4+j] - A[j*4+i])
        passed = false;
      {{FType.Name}} sum = 0;
      for(int k=0;k<4;k++)
        sum += A[i*4+k]*A[j*4+k];
      if(mM(i,j) != sum || mM2(i,j) != sum)
        passed = false;
    }
  }

  // raw row-major matrix product
  {{FType.Name}} AA[16];
  mat4_rowmajor_mul_mat4(AA, A, A);
  for(int i=0;i<4;i++)
  {
    for(int j=0;j<4;j++)
    {
      {{FType.Name}} sum = 0;
      for(int k=0;k<4;k++)
        sum += A[i*4+k]*A[k*4+j];
      if(AA[i*4+j] != sum)
        passed = false;
    }
  }

  // matrix-vector
  const {{FType.Name}}4 v(1, -2, 3, -4);
  const {{FType.Name}}4 r1 = m1*v;
  const {{FType.Name}}4 r2 = mul(m1, v);
  const {{FType.Name}}4 r3 = mul4x4x4(m1, v);
  for(int i=0;i<4;i++)
  {
    const {{FType.Name}} ref = A[i*4+0]*v.x + A[i*4+1]*v.y + A[i*4+2]*v.z + A[i*4+3]*v.w;
    if(r1[i] != ref || r2[i] != ref || r3[i] != ref)
      passed = false;
  }

  // outer product
  const {{FType.Name}}4x4 mO = outerProduct(v, {{FType.Name}}4(1,2,3,4));
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      if(mO(i,j) != v[i]*{{FType.Name}}(j+1))
        passed = false;

  // affine transforms
  const {{FType.Name}}3 p(1, 2, 3);
  const {{FType.Name}}4x4 mTr = translate4x4({{FType.Name}}3(10, 20, 30));
  const {{FType.Name}}4x4 mSc = scale4x4({{FType.Name}}3(2, 3, 4));
  const {{FType.Name}}3 p1 = mul4x3(mTr, p);
  const {{FType.Name}}3 p2 = mul3x3(mTr, p);
  const {{FType.Name}}3 p3 = mul4x3(mSc, p);
  passed = passed && (p1.x == 11) && (p1.y == 22) && (p1.z == 33);
  passed = passed && (p2.x == 1)  && (p2.y == 2)  && (p2.z == 3);
  passed = passed && (p3.x == 2)  && (p3.y == 6)  && (p3.z == 12);

  // rotations
  const {{FType.Name}} angle = {{FType.Name}}(M_PI)*{{FType.Name}}(0.5);
  const {{FType.Name}}3 rx = mul3x3(rotate4x4X(angle), {{FType.Name}}3(0,1,0));
  const {{FType.Name}}3 ry = mul3x3(rotate4x4Y(angle), {{FType.Name}}3(0,0,1));
  const {{FType.Name}}3 rz = mul3x3(rotate4x4Z(angle), {{FType.Name}}3(0,1,0));
  passed = passed && length(rx - {{FType.Name}}3(0,0,1))  < 1e-6f;
  passed = passed && length(ry - {{FType.Name}}3(1,0,0))  < 1e-6f;
  passed = passed && length(rz - {{FType.Name}}3(-1,0,0)) < 1e-6f;

  // inverse
  const {{FType.Name}}4x4 mC = mTr*rotate4x4Y({{FType.Name}}(0.3))*mSc;
  const {{FType.Name}}4x4 mCI = inverse4x4(mC);
  const {{FType.Name}}4x4 mCheck1 = mCI*mC;
  const {{FType.Name}}4x4 mCheck2 = inverse4x4(m1)*m1;
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      if(!IsClose(mCheck1(i,j), mI(i,j)) || !IsClose(mCheck2(i,j), mI(i,j)))
        passed = false;

  return passed;
}

bool test{{FType.Number+1}}_mat3x3_{{FType.Name}}()
{
  const {{FType.Name}} A[9] = { 1, 2, 3,
                      0, 1, 4,
                      5, 6, 0 }; // row-major, det = 1

  const {{FType.Name}} AInv[9] = { -24,  18,  5,
                          20, -15, -4,
                          -5,   4,  1 };

  const {{FType.Name}}3x3 m1(A);
  const {{FType.Name}}3x3 m2(1, 2, 3,
                    0, 1, 4,
                    5, 6, 0);
  const {{FType.Name}}3x3 m3 = make_{{FType.Name}}3x3_from_rows({{FType.Name}}3(1,2,3), {{FType.Name}}3(0,1,4), {{FType.Name}}3(5,6,0));
  const {{FType.Name}}3x3 m4 = make_{{FType.Name}}3x3_from_cols({{FType.Name}}3(1,0,5), {{FType.Name}}3(2,1,6), {{FType.Name}}3(3,4,0));
  const {{FType.Name}}3x3 m5 = make_{{FType.Name}}3x3({{FType.Name}}3(1,2,3), {{FType.Name}}3(0,1,4), {{FType.Name}}3(5,6,0));
  const {{FType.Name}}3x3 m6 = make_{{FType.Name}}3x3_by_columns({{FType.Name}}3(1,0,5), {{FType.Name}}3(2,1,6), {{FType.Name}}3(3,4,0));
  const {{FType.Name}}3x3 m7(m1);
  const {{FType.Name}}3x3 mI;
  const {{FType.Name}}3x3 mK({{FType.Name}}(2));

  {{FType.Name}}3x3 mZ;
  mZ.zero();
  {{FType.Name}}3x3 mI2(mK);
  mI2.identity();

  {{FType.Name}}3x3 m8;
  for(int i=0;i<3;i++)
  {
    m8.set_col(i, {{FType.Name}}3({{FType.Name}}(0)));
    m8.col(i).x = A[0*3+i];
    for(int j=1;j<3;j++)
      m8(j,i) = A[j*3+i];
  }

  {{FType.Name}}3x3 m9;
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      m9[i][j] = A[i*3+j];

  const {{FType.Name}}3x3 mT   = transpose(m1);
  const {{FType.Name}}3x3 mS   = m1 + mT;
  const {{FType.Name}}3x3 mD   = m1 - mT;
  const {{FType.Name}}3x3 mM   = m1*mT;
  const {{FType.Name}}3x3 mM2  = mul(m1, mT);
  const {{FType.Name}}3x3 mK1  = m1*{{FType.Name}}(3);
  const {{FType.Name}}3x3 mK2  = {{FType.Name}}(3)*m1;
  const {{FType.Name}}3x3 mInv = inverse3x3(m1);
  const {{FType.Name}}    det  = determinant(m1);

  bool passed = IsClose(det, {{FType.Name}}(1));
  for(int i=0;i<3;i++)
  {
    const {{FType.Name}}3 row = m1.get_row(i);
    const {{FType.Name}}3 col = m1.get_col(i);
    const {{FType.Name}}3 cl2 = m1.col(i);
    const auto rowAccess = m9[i]; // const RowTmp
    for(int j=0;j<3;j++)
    {
      const {{FType.Name}} ref = A[i*3+j];
      if(m1(i,j) != ref || m2(i,j) != ref || m3(i,j) != ref || m4(i,j) != ref || m5(i,j) != ref || m6(i,j) != ref || m7(i,j) != ref || m8(i,j) != ref || rowAccess[j] != ref)
        passed = false;
      if(row[j] != ref || col[j] != A[j*3+i] || cl2[j] != A[j*3+i])
        passed = false;
      const {{FType.Name}} id = (i == j) ? {{FType.Name}}(1) : {{FType.Name}}(0);
      if(mI(i,j) != id || mI2(i,j) != id || mZ(i,j) != 0 || mK(i,j) != 2)
        passed = false;
      if(mT(i,j) != A[j*3+i] || mS(i,j) != A[i*3+j] + A[j*3+i] || mD(i,j) != A[i*3+j] - A[j*3+i])
        passed = false;
      if(mK1(i,j) != 3*ref || mK2(i,j) != 3*ref || !IsClose(mInv(i,j), AInv[i*3+j], {{FType.Name}}(1e-4)))
        passed = false;
      {{FType.Name}} sum = 0;
      for(int k=0;k<3;k++)
        sum += A[i*3+k]*A[j*3+k];
      if(mM(i,j) != sum || mM2(i,j) != sum)
        passed = false;
    }
  }

  // matrix-vector
  const {{FType.Name}}3 v(1, -2, 3);
  const {{FType.Name}}3 r1 = m1*v;
  const {{FType.Name}}3 r2 = mul(m1, v);
  for(int i=0;i<3;i++)
  {
    const {{FType.Name}} ref = A[i*3+0]*v.x + A[i*3+1]*v.y + A[i*3+2]*v.z;
    if(r1[i] != ref || r2[i] != ref)
      passed = false;
  }

  // outer product
  const {{FType.Name}}3x3 mO = outerProduct(v, {{FType.Name}}3(1,2,3));
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      if(mO(i,j) != v[i]*{{FType.Name}}(j+1))
        passed = false;

  // scale and rotations
  const {{FType.Name}}3 p1 = scale3x3({{FType.Name}}3(2,3,4))*{{FType.Name}}3(1,2,3);
  passed = passed && (p1.x == 2) && (p1.y == 6) && (p1.z == 12);

  const {{FType.Name}} angle = {{FType.Name}}(M_PI)*{{FType.Name}}(0.5);
  const {{FType.Name}}3 rx = rotate3x3X(angle)*{{FType.Name}}3(0,1,0);
  const {{FType.Name}}3 ry = rotate3x3Y(angle)*{{FType.Name}}3(0,0,1);
  const {{FType.Name}}3 rz = rotate3x3Z(angle)*{{FType.Name}}3(0,1,0);
  passed = passed && length(rx - {{FType.Name}}3(0,0,1))  < 1e-6f;
  passed = passed && length(ry - {{FType.Name}}3(1,0,0))  < 1e-6f;
  passed = passed && length(rz - {{FType.Name}}3(-1,0,0)) < 1e-6f;

  return passed;
}

bool test{{FType.Number+2}}_complex_{{FType.Name}}()
{
  const complex{{FType.Suffix}} z0;
  const complex{{FType.Suffix}} z1({{FType.Name}}(3));
  const complex{{FType.Suffix}} z2(1, 2);
  const complex{{FType.Suffix}} z3(3, 4);

  const complex{{FType.Suffix}} c1 = -z2;
  const complex{{FType.Suffix}} c2 = z2 + z3;
  const complex{{FType.Suffix}} c3 = z2 - z3;
  const complex{{FType.Suffix}} c4 = z2 * z3;  // -5 + 10i
  const complex{{FType.Suffix}} c5 = c4 / z3;  //  1 +  2i
  const complex{{FType.Suffix}} c6 = {{FType.Name}}(2) + z2;
  const complex{{FType.Suffix}} c7 = {{FType.Name}}(2) - z2;
  const complex{{FType.Suffix}} c8 = {{FType.Name}}(2) * z2;
  const complex{{FType.Suffix}} c9 = {{FType.Name}}(5) / z2; // 1 - 2i

  bool passed = true;
  passed = passed && (real(z0) == 0)  && (imag(z0) == 0);
  passed = passed && (real(z1) == 3)  && (imag(z1) == 0);
  passed = passed && (real(c1) == -1) && (imag(c1) == -2);
  passed = passed && (real(c2) == 4)  && (imag(c2) == 6);
  passed = passed && (real(c3) == -2) && (imag(c3) == -2);
  passed = passed && (real(c4) == -5) && (imag(c4) == 10);
  passed = passed && IsClose(real(c5), {{FType.Name}}(1)) && IsClose(imag(c5), {{FType.Name}}(2));
  passed = passed && (real(c6) == 3)  && (imag(c6) == 2);
  passed = passed && (real(c7) == 1)  && (imag(c7) == -2);
  passed = passed && (real(c8) == 2)  && (imag(c8) == 4);
  passed = passed && IsClose(real(c9), {{FType.Name}}(1)) && IsClose(imag(c9), {{FType.Name}}(-2));

  passed = passed && (complex_norm(z3) == 25) && IsClose(complex_abs(z3), {{FType.Name}}(5));

  const complex{{FType.Suffix}} s0 = complex_sqrt(z0);
  const complex{{FType.Suffix}} s1 = complex_sqrt(z3);                         // 2 + i
  const complex{{FType.Suffix}} s2 = complex_sqrt(complex{{FType.Suffix}}(-3,  4)); // 1 + 2i
  const complex{{FType.Suffix}} s3 = complex_sqrt(complex{{FType.Suffix}}(-3, -4)); // 1 - 2i
  passed = passed && (real(s0) == 0) && (imag(s0) == 0);
  passed = passed && IsClose(real(s1), {{FType.Name}}(2)) && IsClose(imag(s1), {{FType.Name}}(1));
  passed = passed && IsClose(real(s2), {{FType.Name}}(1)) && IsClose(imag(s2), {{FType.Name}}(2));
  passed = passed && IsClose(real(s3), {{FType.Name}}(1)) && IsClose(imag(s3), {{FType.Name}}(-2));

  return passed;
}

## endfor
