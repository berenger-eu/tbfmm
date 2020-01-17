// ===================================================================================
// Copyright ScalFmm 2011 INRIA, Olivier Coulaud, Berenger Bramas, Matthias Messner
// olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.  
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info". 
// "http://www.gnu.org/licenses".
// ===================================================================================
#ifndef FBLAS_HPP
#define FBLAS_HPP

namespace FBlas {

template <class Type>
inline void copy(const unsigned n, const Type* orig, Type* dest)
{ memcpy(dest, orig, n*sizeof(Type)); }

template <class Type>
inline void c_copy(const unsigned n, const Type* orig, Type* dest)
{ memcpy(dest, orig, 2*n*sizeof(Type)); }

template <class Type>
inline void setzero(const unsigned n, Type* dest)
{ memset(dest, 0, n*sizeof(Type)); }

template <class Type>
inline void c_setzero(const unsigned n, Type* dest)
{ memset(dest, 0, 2*n*sizeof(Type)); }

template <class Type>
inline void scal(const unsigned n, const Type d, Type* const x, const unsigned incd = 1)
{
    for(unsigned idx = 0 ; idx < n ; ++idx){
        x[idx*incd] *= d;
    }
}

// y += x
template <class Type>
inline void add(const unsigned n, const Type* const x, Type* const y)
{
    for(unsigned idx = 0 ; idx < n ; ++idx){
        y[idx] += x[idx];
    }
}

// y += d Ax
template <class Type>
inline void gemva(const unsigned m, const unsigned n, Type d, const Type* A, const Type *x, Type *y){
    for(unsigned idxRow = 0 ; idxRow < m ; ++idxRow){
        y[idxRow] = 0;
        for(unsigned idxCol = 0 ; idxCol < n ; ++idxCol){
            y[idxRow] += d * A[idxRow+n*idxCol] * x[idxCol];
        }
    }
}


// y += d A^T x
template <class Type>
inline void gemtva(const unsigned m, const unsigned n, Type d, const Type* A, const Type *x, Type *y){
    for(unsigned idxRow = 0 ; idxRow < n ; ++idxRow){
        y[idxRow] = 0;
        for(unsigned idxCol = 0 ; idxCol < m ; ++idxCol){
            y[idxRow] += d * A[idxCol*n+idxRow] * x[idxCol];
        }
    }
}


// C = d A B, A is m x p, B is p x n
template <class Type>
inline void gemm(unsigned m, unsigned p, unsigned n, Type d,
                 const Type* A, unsigned ldA, const Type* B, unsigned ldB, Type* C, unsigned ldC)
{
    for(unsigned idxRow = 0 ; idxRow < m ; ++idxRow){
        for(unsigned idxCol = 0 ; idxCol < n ; ++idxCol){
            C[idxRow+ldC*idxCol] = 0;
            for(unsigned idxK = 0 ; idxK < p ; ++idxK){
                C[idxRow+ldC*idxCol] += d * A[idxRow+ldA*idxK] * B[idxK+ldB*idxCol];
            }
        }
    }
}


// C = d A^T B, A is m x p, B is m x n
template <class Type>
inline void gemtm(unsigned m, unsigned p, unsigned n, Type d,
                  const Type* A, unsigned ldA, const Type *B, unsigned ldB, Type* C, unsigned ldC)
{
    for(unsigned idxRow = 0 ; idxRow < m ; ++idxRow){
        for(unsigned idxCol = 0 ; idxCol < n ; ++idxCol){
            C[idxRow+ldC*idxCol] = 0;
            for(unsigned idxK = 0 ; idxK < p ; ++idxK){
                C[idxRow+ldC*idxCol] += d * A[idxK+ldA*idxRow] * B[idxK+ldB*idxCol];
            }
        }
    }
}


} // end namespace FCBlas



#endif //FBLAS_HPP

