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
#ifndef FMATH_HPP
#define FMATH_HPP


#include <cmath>
#include <limits>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
    #define M_PI_2 1.57079632679489661923
#endif

/**
 * @author Berenger Bramas (berenger.bramas@inria.fr)
 * @class
 * Please read the license
 *
 * Propose basic math functions or indirections to std math.
 */
struct FMath{
    template <class FReal>
    constexpr static FReal FPi(){ return FReal(M_PI); }
    template <class FReal>
    constexpr static FReal FTwoPi(){ return FReal(2.0*M_PI); }
    template <class FReal>
    constexpr static FReal FPiDiv2(){ return FReal(M_PI_2); }
    template <class FReal>
    constexpr static FReal Epsilon(){ return FReal(0.00000000000000000001); }

    /** To get absolute value */
    template <class NumType>
    static NumType Abs(const NumType inV){
        return (inV < 0 ? -inV : inV);
    }

    /** To get max between 2 values */
    template <class NumType>
    static NumType Max(const NumType inV1, const NumType inV2){
        return (inV1 > inV2 ? inV1 : inV2);
    }

    /** To get min between 2 values */
    template <class NumType>
    static NumType Min(const NumType inV1, const NumType inV2){
        return (inV1 < inV2 ? inV1 : inV2);
    }


    /** To know if 2 values seems to be equal */
    template <class NumType>
    static bool LookEqual(const NumType inV1, const NumType inV2){
        return (Abs(inV1-inV2) < std::numeric_limits<NumType>::epsilon());
        //const FReal relTol = FReal(0.00001);
        //const FReal absTol = FReal(0.00001);
        //return (Abs(inV1 - inV2) <= Max(absTol, relTol * Max(Abs(inV1), Abs(inV2))));
    }

    /** To know if 2 values seems to be equal */
    template <class NumType>
    static NumType RelatifDiff(const NumType inV1, const NumType inV2){
        return Abs(inV1 - inV2)*Abs(inV1 - inV2)/Max(Abs(inV1*inV1), Abs(inV2*inV2));
    }

    /** To get floor of a FReal */
    static float dfloor(const float inValue){
        return floorf(inValue);
    }
    static double dfloor(const double inValue){
        return floor(inValue);
    }

    /** To get ceil of a FReal */
    static float Ceil(const float inValue){
        return ceilf(inValue);
    }
    static double Ceil(const double inValue){
        return ceil(inValue);
    }


    /** To get pow */
    static double pow(double x, double y){
        return ::pow(x,y);
    }
    static float pow(float x, float y){
        return ::powf(x,y);
    }
    template <class NumType>
    static NumType pow(const NumType inValue, int power){
        NumType result = 1;
        while(power-- > 0) result *= inValue;
        return result;
    }

    /** To get pow of 2 */
    static int pow2(const int power){
        return (1 << power);
    }

    /** To get factorial */
    template <class NumType>
    static double factorial(int inValue){
        if(inValue==0) return NumType(1.);
        else {
            NumType result = NumType(inValue);
            while(--inValue > 1) result *= NumType(inValue);
            return result;
        }
    }

    /** To get exponential */
    static double Exp(double x){
        return ::exp(x);
    }
    static float Exp(float x){
        return ::expf(x);
    }

    /** To know if a value is between two others */
    template <class NumType>
    static bool Between(const NumType inValue, const NumType inMin, const NumType inMax){
        return ( inMin <= inValue && inValue < inMax );
    }
    /** To compute fmadd operations **/
    template <class NumType>
    static NumType FMAdd(const NumType a, const NumType b, const NumType c){
	return a * b + c;
    }

    /** To get sqrt of a FReal */
    static float Sqrt(const float inValue){
        return sqrtf(inValue);
    }
    static double Sqrt(const double inValue){
        return sqrt(inValue);
    }
    static float Rsqrt(const float inValue){
        return float(1.0)/sqrtf(inValue);
    }
    static double Rsqrt(const double inValue){
        return 1.0/sqrt(inValue);
    }

    /** To get Log of a FReal */
    static float Log(const float inValue){
        return logf(inValue);
    }
    static double Log(const double inValue){
        return log(inValue);
    }

    /** To get Log2 of a FReal */
    static float Log2(const float inValue){
        return log2f(inValue);
    }
    static double Log2(const double inValue){
        return log2(inValue);
    }

    /** To get atan2 of a 2 FReal,  The return value is given in radians and is in the
      range -pi to pi, inclusive.  */
    static float Atan2(const float inValue1,const float inValue2){
        return atan2f(inValue1,inValue2);
    }
    static double Atan2(const double inValue1,const double inValue2){
        return atan2(inValue1,inValue2);
    }

    /** To get sin of a FReal */
    static float Sin(const float inValue){
        return sinf(inValue);
    }
    static double Sin(const double inValue){
        return sin(inValue);
    }

    /** To get asinf of a float. The result is in the range [0, pi]*/
    static float ASin(const float inValue){
        return asinf(inValue);
    }
    /** To get asinf of a double. The result is in the range [0, pi]*/
    static double ASin(const double inValue){
        return asin(inValue);
    }

    /** To get cos of a FReal */
    static float Cos(const float inValue){
        return cosf(inValue);
    }
    static double Cos(const double inValue){
        return cos(inValue);
    }

    /** To get arccos of a float. The result is in the range [0, pi]*/
    static float ACos(const float inValue){
        return acosf(inValue);
    }
    /** To get arccos of a double. The result is in the range [0, pi]*/
    static double ACos(const double inValue){
        return acos(inValue);
    }

    /** To get atan2 of a 2 FReal */
    static float Fmod(const float inValue1,const float inValue2){
        return fmodf(inValue1,inValue2);
    }
    /** return the floating-point remainder of inValue1  / inValue2 */
    static double Fmod(const double inValue1,const double inValue2){
        return fmod(inValue1,inValue2);
    }

    /** To know if a variable is nan, based on the C++0x */
    template <class TestClass>
    static bool IsNan(const TestClass& value){
        //volatile const TestClass* const pvalue = &value;
        //return (*pvalue) != value;
        return std::isnan(value);
    }

    /** To know if a variable is not inf, based on the C++0x */
    template <class TestClass>
    static bool IsFinite(const TestClass& value){
        // return !(value <= std::numeric_limits<T>::min()) && !(std::numeric_limits<T>::max() <= value);
        return std::isfinite(value);
    }

    template <class NumType>
    static NumType Zero();

    template <class NumType>
    static NumType One();

    template <class DestType, class SrcType>
    static DestType ConvertTo(const SrcType val);


    /** A class to compute accuracy */
    template <class FReal, class IndexType = long int>
    class FAccurater {
        IndexType    nbElements;
        FReal l2Dot;
        FReal l2Diff;
        FReal max;
        FReal maxDiff;
    public:
        FAccurater() : nbElements(0),l2Dot(0.0), l2Diff(0.0), max(0.0), maxDiff(0.0) {
        }
        /** with inital values */
        FAccurater(const FReal inGood[], const FReal inBad[], const IndexType nbValues)
            :  nbElements(0),l2Dot(0.0), l2Diff(0.0), max(0.0), maxDiff(0.0)  {
            add(inGood, inBad, nbValues);
        }


        /** Add value to the current list */
        void add(const FReal inGood, const FReal inBad){
            l2Diff          += (inBad - inGood) * (inBad - inGood);
            l2Dot          += inGood * inGood;
            max               = Max(max , Abs(inGood));
            maxDiff         = Max(maxDiff, Abs(inGood-inBad));
            nbElements += 1 ;
        }
        /** Add array of values */
        void add(const FReal inGood[], const FReal inBad[], const IndexType nbValues){
            for(IndexType idx = 0 ; idx < nbValues ; ++idx){
                add(inGood[idx],inBad[idx]);
            }
            nbElements += nbValues ;
        }

        /** Add an accurater*/
        void add(const FAccurater& inAcc){
            l2Diff += inAcc.getl2Diff();
            l2Dot +=  inAcc.getl2Dot();
            max = Max(max,inAcc.getmax());
            maxDiff = Max(maxDiff,inAcc.getInfNorm());
            nbElements += inAcc.getNbElements();
        }

        FReal getl2Diff() const{
            return l2Diff;
        }
        FReal getl2Dot() const{
            return l2Dot;
        }
        FReal getmax() const{
            return max;
        }
        IndexType getNbElements() const{
            return nbElements;
        }
        void  setNbElements(const IndexType & n) {
            nbElements = n;
        }

        /** Get the root mean squared error*/
        FReal getL2Norm() const{
            return Sqrt(l2Diff );
        }
        /** Get the L2 norm */
        FReal getRMSError() const{
            return Sqrt(l2Diff /static_cast<FReal>(nbElements));
        }

        /** Get the inf norm */
        FReal getInfNorm() const{
            return maxDiff;
        }
        /** Get the L2 norm */
        FReal getRelativeL2Norm() const{
            return Sqrt(l2Diff / l2Dot);
        }
        /** Get the inf norm */
        FReal getRelativeInfNorm() const{
            return maxDiff / max;
        }
        /** Print */
        template <class StreamClass>
        friend StreamClass& operator<<(StreamClass& output, const FAccurater& inAccurater){
            output << "[Error] Relative L2Norm = " << inAccurater.getRelativeL2Norm() << " \t RMS Norm = " << inAccurater.getRMSError() << " \t Relative Inf = " << inAccurater.getRelativeInfNorm();
            return output;
        }

        void reset()
        {
            l2Dot          = FReal(0);
            l2Diff          = FReal(0);;
            max            = FReal(0);
            maxDiff      = FReal(0);
            nbElements = 0 ;
        }
    };
};

template <>
inline float FMath::Zero<float>(){
    return float(0.0);
}

template <>
inline double FMath::Zero<double>(){
    return double(0.0);
}

template <>
inline float FMath::One<float>(){
    return float(1.0);
}

template <>
inline double FMath::One<double>(){
    return double(1.0);
}

template <>
inline float FMath::ConvertTo<float,float>(const float val){
    return val;
}

template <>
inline double FMath::ConvertTo<double,double>(const double val){
    return val;
}

template <>
inline float FMath::ConvertTo<float,const float*>(const float* val){
    return *val;
}

template <>
inline double FMath::ConvertTo<double,const double*>(const double* val){
    return *val;
}



#endif //FMATH_HPP
