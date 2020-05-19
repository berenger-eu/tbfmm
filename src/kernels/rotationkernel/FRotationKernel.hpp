// See LICENCE file at project root
#ifndef FROTATIONKERNEL_HPP
#define FROTATIONKERNEL_HPP

#include "tbfglobal.hpp"
#include <complex>

#include "FSpherical.hpp"
#include "FMemUtils.hpp"
#include "kernels/unifkernel/FP2PR.hpp"

#include "utils/tbfperiodicshifter.hpp"

/** This is a recursion to get the minimal size of the matrix dlmk
  */
template<int N> struct NumberOfValuesInDlmk{
    enum {Value = (N*2+1)*(N*2+1) + NumberOfValuesInDlmk<N - 1>::Value};
};
template<> struct NumberOfValuesInDlmk<0>{
    enum {Value = 1};
};

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* @class FRotationKernel
* @brief
*
* This kernels is a complete rotation based kernel with spherical
* harmonic.
*
* Here is the optimizated kernel, please refer to FRotationOriginalKernel
* to see the non optimized easy to understand kernel.
*/
template<class RealType_T, int P, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class FRotationKernel {
public:
    static_assert (SpaceIndexType_T::Dim == 3, "Must be 3");

    using RealType = RealType_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

private:
    const RealType PI = RealType(3.14159265358979323846264338327950288419716939937510582097494459230781640628620899863L);
    const RealType PIDiv2 = RealType(3.14159265358979323846264338327950288419716939937510582097494459230781640628620899863L/2);
    const RealType PI2 = RealType(3.14159265358979323846264338327950288419716939937510582097494459230781640628620899863L*2);

    //< Size of the data array computed using a suite relation
    static const int SizeArray = ((P+2)*(P+1))/2;
    //< To have P*2 where needed
    static const int P2 = P*2;

    ///////////////////////////////////////////////////////
    // Object attributes
    ///////////////////////////////////////////////////////

    const SpaceIndexType spaceIndexSystem;

    const RealType boxWidth;               //< the box width at leaf level
    const int   treeHeight;             //< The height of the tree
    const RealType widthAtLeafLevel;       //< width of box at leaf level
    const RealType widthAtLeafLevelDiv2;   //< width of box at leaf leve div 2
    const std::array<RealType,3> boxCorner;             //< position of the box corner

    RealType factorials[P2+1];             //< This contains the factorial until 2*P+1

    ///////////// Translation /////////////////////////////
 #ifdef __NEC__
    std::shared_ptr<RealType>      M2MTranslationCoef;  //< This contains some precalculated values for M2M translation
    std::shared_ptr<RealType> M2LTranslationCoef;  //< This contains some precalculated values for M2L translation
    std::shared_ptr<RealType>      L2LTranslationCoef;  //< This contains some precalculated values for L2L translation
 #else
    std::shared_ptr<RealType[][P+1]>      M2MTranslationCoef;  //< This contains some precalculated values for M2M translation
    std::shared_ptr<RealType[][343][P+1]> M2LTranslationCoef;  //< This contains some precalculated values for M2L translation
    std::shared_ptr<RealType[][P+1]>      L2LTranslationCoef;  //< This contains some precalculated values for L2L translation
#endif
    ///////////// Rotation    /////////////////////////////
    std::complex<RealType> rotationExpMinusImPhi[8][SizeArray];  //< This is the vector use for the rotation around z for the M2M (multipole)
    std::complex<RealType> rotationExpImPhi[8][SizeArray];       //< This is the vector use for the rotation around z for the L2L (taylor)

    std::complex<RealType> rotationM2LExpMinusImPhi[343][SizeArray]; //< This is the vector use for the rotation around z for the M2L (multipole)
    std::complex<RealType> rotationM2LExpImPhi[343][SizeArray];      //< This is the vector use for the rotation around z for the M2L (taylor)

    ///////////// Rotation    /////////////////////////////
    // First we compute the size of the d{l,m,k} matrix.

    static const int SizeDlmkMatrix = NumberOfValuesInDlmk<P>::Value;

    RealType DlmkCoefOTheta[8][SizeDlmkMatrix];        //< d_lmk for Multipole rotation
    RealType DlmkCoefOMinusTheta[8][SizeDlmkMatrix];   //< d_lmk for Multipole reverse rotation

    RealType DlmkCoefMTheta[8][SizeDlmkMatrix];        //< d_lmk for Local rotation
    RealType DlmkCoefMMinusTheta[8][SizeDlmkMatrix];   //< d_lmk for Local reverse rotation

    RealType DlmkCoefM2LOTheta[343][SizeDlmkMatrix];       //< d_lmk for Multipole rotation
    RealType DlmkCoefM2LMMinusTheta[343][SizeDlmkMatrix];  //< d_lmk for Local reverse rotation

    ///////////////////////////////////////////////////////
    // Precomputation
    ///////////////////////////////////////////////////////

    /** Compute the factorial from 0 to P*2
      * Then the data is accessible in factorials array:
      * factorials[n] = n! with n <= 2*P
      */
    void precomputeFactorials(){
        factorials[0] = 1;
        RealType fidx = 1;
        for(int idx = 1 ; idx <= P2 ; ++idx, ++fidx){
            factorials[idx] = fidx * factorials[idx-1];
        }
    }

    /** This function precompute the translation coef.
      * Translation are independant of the angle between both cells.
      * So in the M2M/L2L the translation is the same for all children.
      * In the M2L the translation depend on the distance between the
      * source and the target (so a few number of possibilities exist)
      *
      * The number of possible translation depend of the tree height,
      * so the memory is allocated dynamically with a smart pointer to share
      * the data between threads.
      */
    void precomputeTranslationCoef(){
        {// M2M & L2L
            // Allocate
#ifdef __NEC__
            M2MTranslationCoef.reset((RealType*)(new RealType[(treeHeight-1)*(P+1)]));
            L2LTranslationCoef.reset((RealType*)(new RealType[(treeHeight-1)*(P+1)]));
#else
            M2MTranslationCoef.reset(new RealType[treeHeight-1][P+1]);
            L2LTranslationCoef.reset(new RealType[treeHeight-1][P+1]);
#endif
            // widthAtLevel represents half of the size of a box
            RealType widthAtLevel = boxWidth/4;
            // we go from the root to the leaf-1
            for( int idxLevel = 0 ; idxLevel < treeHeight - 1 ; ++idxLevel){
                // b is the parent-child distance = norm( vec(widthAtLevel,widthAtLevel,widthAtLevel))
                const RealType b = std::sqrt(widthAtLevel*widthAtLevel*3);
                // we compute b^idx iteratively
                RealType bPowIdx = 1.0;
                // we compute -1^idx iteratively
                RealType minus_1_pow_idx = 1.0;
                for(int idx = 0 ; idx <= P ; ++idx){
#ifdef __NEC__
                    // coef m2m = (-b)^j/j!
                    ((RealType*)M2MTranslationCoef.get())[idxLevel*(P+1)+idx] = minus_1_pow_idx * bPowIdx / factorials[idx];
                    // coef l2l = b^j/j!
                    ((RealType*)L2LTranslationCoef.get())[idxLevel*(P+1)+idx] = bPowIdx / factorials[idx];

#else
                    // coef m2m = (-b)^j/j!
                    M2MTranslationCoef[idxLevel][idx] = minus_1_pow_idx * bPowIdx / factorials[idx];
                    // coef l2l = b^j/j!
                    L2LTranslationCoef[idxLevel][idx] = bPowIdx / factorials[idx];
#endif
                    // increase
                    bPowIdx *= b;
                    minus_1_pow_idx = -minus_1_pow_idx;
                }
                // divide by two per level
                widthAtLevel /= 2;
            }
        }
        {// M2L
            // Allocate
#ifdef __NEC__
            M2LTranslationCoef.reset((RealType*)(new RealType[treeHeight*343*(P+1)]));
#else
            M2LTranslationCoef.reset(new RealType[treeHeight][343][P+1]);
#endif
            // This is the width of a box at each level
            RealType boxWidthAtLevel = widthAtLeafLevel;
            // from leaf level to the root
            for(int idxLevel = treeHeight-1 ; idxLevel > 0 ; --idxLevel){
                // we compute all possibilities
                for(int idxX = -3 ; idxX <= 3 ; ++idxX ){
                    for(int idxY = -3 ; idxY <= 3 ; ++idxY ){
                        for(int idxZ = -3 ; idxZ <= 3 ; ++idxZ ){
                            // if this is not a neighbour
                            if( idxX < -1 || 1 < idxX || idxY < -1 || 1 < idxY || idxZ < -1 || 1 < idxZ ){
                                // compute the relative position
                                const std::array<RealType,3> relativePosition{{ -RealType(idxX)*boxWidthAtLevel,
                                                                      -RealType(idxY)*boxWidthAtLevel,
                                                                      -RealType(idxZ)*boxWidthAtLevel}};
                                // this is the position in the index system from 0 to 343
                                const int position = ((( (idxX+3) * 7) + (idxY+3))) * 7 + idxZ + 3;
                                // b is the distance between the two cells
                                const RealType b = std::sqrt( (relativePosition[0] * relativePosition[0]) +
                                                             (relativePosition[1] * relativePosition[1]) +
                                                             (relativePosition[2] * relativePosition[2]));
                                // compute b^idx+1 iteratively
                                RealType bPowIdx1 = b;
                                for(int idx = 0 ; idx <= P ; ++idx){
                                    // factorials[j+l] / FMath::pow(b,j+l+1)
#ifdef __NEC__
                                    ((RealType*)M2LTranslationCoef.get())[(idxLevel*343+position)*(P+1)+idx] = factorials[idx] / bPowIdx1;
#else
                                    M2LTranslationCoef[idxLevel][position][idx] = factorials[idx] / bPowIdx1;
#endif
                                    bPowIdx1 *= b;
                                }
                            }
                        }
                    }
                }
                // multiply per two at each level
                boxWidthAtLevel *= RealType(2.0);
            }
        }
    }

    ///////////////////////////////////////////////////////
    // Precomputation rotation vector
    // This is a all in one function
    // First we compute the d_lmk needed,
    // then we compute vectors for M2M/L2L
    // finally we compute the vectors for M2L
    ///////////////////////////////////////////////////////


    /** The following comments include formula taken from the original vectors
      *
      *
      * This function rotate a multipole vector by an angle azimuth phi
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1
      * \f[
      * O_{l,m}( \alpha, \beta + \phi ) = e^{-i \phi m} O_{l,m}( \alpha, \beta )
      * \f]
      * The computation is simply a multiplication per a complex number \f$ e^{-i \phi m} \f$
      * Phi should be in [0,2pi]
      *
      * This function rotate a local vector by an angle azimuth phi
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1
      * \f[
      * M_{l,m}( \alpha, \beta + \phi ) = e^{i \phi m} M_{l,m}( \alpha, \beta )
      * \f]
      * The computation is simply a multiplication per a complex number \f$ e^{i \phi m} \f$
      * Phi should be in [0,2pi]
      *
      * This function rotate a multipole vector by an angle inclination \theta
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1
      * \f[
      * O_{l,m}( \alpha + \theta, \beta ) = \sum_{k=-l}^l{ \sqrt{ \frac{(l-k)!(l+k)!}{(l-|m|)!(l+|m|)!} }
      *                                     d^l_{km}( \theta ) O_{l,k}( \alpha, \beta ) }
      * \f]
      * Because we store only P_lm for l >= 0 and m >= 0 we use the relation of symetrie as:
      * \f$ O_{l,-m} = \bar{ O_{l,m} } (-1)^m \f$
      * Theta should be in [0,pi]
      *
      * This function rotate a local vector by an angle inclination \theta
      * The formula used is present in several paper, but we refer to
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1
      * \f[
      * M_{l,m}( \alpha + \theta, \beta ) = \sum_{k=-l}^l{ \sqrt{ \frac{(l-|m|)!(l+|m|)!}{(l-k)!(l+k)!} }
      *                                     d^l_{km}( \theta ) M_{l,k}( \alpha, \beta ) }
      * \f]
      * Because we store only P_lm for l >= 0 and m >= 0 we use the relation of symetrie as:
      * \f$ M_{l,-m} = \bar{ M_{l,m} } (-1)^m \f$
      * Theta should be in [0,pi]
      *
      * Remark about the structure of the structure of the matrixes DlmkCoef[O/M](Minus)Theta.
      * It is composed of "P" small matrix.
      * The matrix M(l) (0 <= l <= P) has a size of (l*2+1)
      * It means indexes are going from -l to l for column and row.
      * l = 0: ( -0 <= m <= 0 ; -0 <= k <= 0)
      * [X]
      * l = 1: ( -1 <= m <= 1 ; -1 <= k <= 1)
      * [X X X]
      * [X X X]
      * [X X X]
      * etc.
      * The real size of such matrix is :
      * 1x1 + 3x3 + ... + (2P+1)x(2P+1)
      */
    void precomputeRotationVectors(){
        /////////////////////////////////////////////////////////////////
        // We will need a Sqrt(factorial[x-y]*factorial[x+y])
        // so we precompute it
        RealType sqrtDoubleFactorials[P+1][P+1];
        for(int l = 0 ; l <= P ; ++l ){
            for(int m = 0 ; m <= l ; ++m ){
                sqrtDoubleFactorials[l][m] = std::sqrt(factorials[l-m]*factorials[l+m]);
            }
        }

        /////////////////////////////////////////////////////////////////
        // We compute the rotation matrix, we do not need 343 matrix
        // We will compute only a part of the since we compute the inclinaison
        // angle. inclinaison(+/-x,+/-y,z) = inclinaison(+/-y,+/-x,z)
        // we put the negative (-theta) with a negative x
        typedef RealType (*pMatrixDlmk) /*[P+1]*/[P2+1][P2+1];
        pMatrixDlmk dlmkMatrix[7][4][7];
        // Allocate matrix
        for(int idxX = 0 ; idxX < 7 ; ++idxX)
            for(int idxY = 0 ; idxY < 4 ; ++idxY)
                for(int idxZ = 0 ; idxZ < 7 ; ++idxZ) {
                    dlmkMatrix[idxX][idxY][idxZ] = new RealType[P+1][P2+1][P2+1];
                }

        // First we compute special vectors:
        DlmkBuild0(dlmkMatrix[0+3][0][1+3]);    // theta = 0
        DlmkBuildPi(dlmkMatrix[0+3][0][-1+3]);  // theta = Pi
        DlmkBuild(dlmkMatrix[1+3][0][0+3],PIDiv2);              // theta = Pi/2
        DlmkInverse(dlmkMatrix[-1+3][0][0+3],dlmkMatrix[1+3][0][0+3]);  // theta = -Pi/2
        // Then other angle
        for(int x = 1 ; x <= 3 ; ++x){
            for(int y = 0 ; y <= x ; ++y){
                for(int z = 1 ; z <= 3 ; ++z){
                    const RealType inclinaison = FSpherical<RealType>(std::array<RealType,3>{{RealType(x),RealType(y),RealType(z)}}).getInclination();
                    DlmkBuild(dlmkMatrix[x+3][y][z+3],inclinaison);
                    // For inclinaison between ]pi/2;pi[
                    DlmkZNegative(dlmkMatrix[x+3][y][(-z)+3],dlmkMatrix[x+3][y][z+3]);
                    // For inclinaison between ]pi;3pi/2[
                    DlmkInverseZNegative(dlmkMatrix[(-x)+3][y][(-z)+3],dlmkMatrix[x+3][y][z+3]);
                    // For inclinaison between ]3pi/2;2pi[
                    DlmkInverse(dlmkMatrix[(-x)+3][y][z+3],dlmkMatrix[x+3][y][z+3]);
                }
            }
        }

        /////////////////////////////////////////////////////////////////
        // Manage angle for M2M/L2L

        const int index_P0 = atLm(P,0);
        // For all possible child (morton indexing from 0 to 7)
        for(int idxChild = 0 ; idxChild < 8 ; ++idxChild){
            // Retrieve relative position of child to parent
            const RealType x = RealType((idxChild&4)? -boxWidth : boxWidth);
            const RealType y = RealType((idxChild&2)? -boxWidth : boxWidth);
            const RealType z = RealType((idxChild&1)? -boxWidth : boxWidth);
            const std::array<RealType,3> relativePosition{{x , y , z }};
            // compute azimuth
            const FSpherical<RealType> sph(relativePosition);

            // First compute azimuth rotation
            // compute the last part with l == P
            {
                int index_lm = index_P0;
                for(int m = 0 ; m <= P ; ++m, ++index_lm ){
                    const RealType mphi = (sph.getPhiZero2Pi() + PIDiv2) * RealType(m);
                    // O_{l,m}( \alpha, \beta + \phi ) = e^{-i \phi m} O_{l,m}( \alpha, \beta )
                    rotationExpMinusImPhi[idxChild][index_lm] = std::complex<RealType>(std::cos(-mphi), std::sin(-mphi));
                    // M_{l,m}( \alpha, \beta + \phi ) = e^{i \phi m} M_{l,m}( \alpha, \beta )
                    rotationExpImPhi[idxChild][index_lm] = std::complex<RealType>(std::cos(mphi), std::sin(mphi));
                }
            }
            // Then for l < P it just a copy of the previous computed vector
            {
                int index_lm = 0;
                // for l < P
                for(int l = 0 ; l < P ; ++l){
                    // take the l + 1 numbers from the vector with l' = P
                    FMemUtils::copyall(&rotationExpMinusImPhi[idxChild][0] + index_lm,
                                       &rotationExpMinusImPhi[idxChild][0] + index_P0,
                                       l + 1);
                    FMemUtils::copyall(&rotationExpImPhi[idxChild][0] + index_lm,
                                       &rotationExpImPhi[idxChild][0] + index_P0,
                                       l + 1);
                    // index(l+1,0) = index(l,0) + l + 1
                    index_lm += l + 1;
                }
            }
            { // Then compute the inclinaison rotation
                // For the child parent relation we always have a inclinaison
                // for (1,1,1) or (1,1,-1)
                const int dx = 1;
                const int dy = 1;
                const int dz = (idxChild&1)?-1:1;

                //
                int index_lmk = 0;
                for(int l = 0 ; l <= P ; ++l){
                    for(int m = 0 ; m <= l ; ++m ){
                        { // for k == 0
                            const RealType d_lmk_minusTheta = dlmkMatrix[-dx+3][dy][dz+3][l][m+P][0+P];
                            const RealType d_lmk            = dlmkMatrix[dx+3][dy][dz+3][l][m+P][0+P];
                            // \sqrt{ \frac{(l-k)!(l+k)!}{(l-|m|)!(l+|m|)!} }
                            const RealType Ofactor = sqrtDoubleFactorials[l][0]/sqrtDoubleFactorials[l][m];
                            const RealType Mfactor = sqrtDoubleFactorials[l][m]/sqrtDoubleFactorials[l][0];

                            DlmkCoefOTheta[idxChild][index_lmk]      = Ofactor * d_lmk;
                            DlmkCoefMTheta[idxChild][index_lmk]      = Mfactor * d_lmk;
                            DlmkCoefOMinusTheta[idxChild][index_lmk] = Ofactor * d_lmk_minusTheta;
                            DlmkCoefMMinusTheta[idxChild][index_lmk] = Mfactor * d_lmk_minusTheta;

                            ++index_lmk;
                        }
                        // for 0 < k
                        RealType minus_1_pow_k = -1.0;
                        for(int k = 1 ; k <= l ; ++k){
                            const RealType d_lm_minus_k            = dlmkMatrix[dx+3][dy][dz+3][l][m+P][-k+P];
                            const RealType d_lmk                   = dlmkMatrix[dx+3][dy][dz+3][l][m+P][k+P];
                            const RealType d_lm_minus_k_minusTheta = dlmkMatrix[-dx+3][dy][dz+3][l][m+P][-k+P];
                            const RealType d_lmk_minusTheta        = dlmkMatrix[-dx+3][dy][dz+3][l][m+P][k+P];

                            const RealType Ofactor = sqrtDoubleFactorials[l][k]/sqrtDoubleFactorials[l][m];
                            const RealType Mfactor = sqrtDoubleFactorials[l][m]/sqrtDoubleFactorials[l][k];

                            // for k negatif
                            DlmkCoefOTheta[idxChild][index_lmk]      = Ofactor * (d_lmk + minus_1_pow_k * d_lm_minus_k);
                            DlmkCoefMTheta[idxChild][index_lmk]      = Mfactor * (d_lmk + minus_1_pow_k * d_lm_minus_k);
                            DlmkCoefOMinusTheta[idxChild][index_lmk] = Ofactor * (d_lmk_minusTheta + minus_1_pow_k * d_lm_minus_k_minusTheta);
                            DlmkCoefMMinusTheta[idxChild][index_lmk] = Mfactor * (d_lmk_minusTheta + minus_1_pow_k * d_lm_minus_k_minusTheta);
                            ++index_lmk;
                            // for k positif
                            DlmkCoefOTheta[idxChild][index_lmk]      = Ofactor * (d_lmk - minus_1_pow_k * d_lm_minus_k);
                            DlmkCoefMTheta[idxChild][index_lmk]      = Mfactor * (d_lmk - minus_1_pow_k * d_lm_minus_k);
                            DlmkCoefOMinusTheta[idxChild][index_lmk] = Ofactor * (d_lmk_minusTheta - minus_1_pow_k * d_lm_minus_k_minusTheta);
                            DlmkCoefMMinusTheta[idxChild][index_lmk] = Mfactor * (d_lmk_minusTheta - minus_1_pow_k * d_lm_minus_k_minusTheta);
                            ++index_lmk;

                            minus_1_pow_k = -minus_1_pow_k;
                        }
                    }
                }
            }
        }

        /////////////////////////////////////////////////////////////////
        // Manage angle for M2L
        // For all possible cases
        for(int idxX = -3 ; idxX <= 3 ; ++idxX ){
            for(int idxY = -3 ; idxY <= 3 ; ++idxY ){
                for(int idxZ = -3 ; idxZ <= 3 ; ++idxZ ){
                    // Test if it is not a neighbors
                    if( idxX < -1 || 1 < idxX || idxY < -1 || 1 < idxY || idxZ < -1 || 1 < idxZ ){
                        // Build relative position between target and source
                        const std::array<RealType,3> relativePosition{{ -RealType(idxX)*boxWidth,
                                                              -RealType(idxY)*boxWidth,
                                                              -RealType(idxZ)*boxWidth}};
                        const int position = ((( (idxX+3) * 7) + (idxY+3))) * 7 + idxZ + 3;
                        const FSpherical<RealType> sph(relativePosition);

                        // Compute azimuth rotation vector
                        // first compute the last part with l == P
                        {
                            int index_lm = index_P0;
                            for(int m = 0 ; m <= P ; ++m, ++index_lm ){
                                const RealType mphi = (sph.getPhiZero2Pi() + PIDiv2) * RealType(m);
                                // O_{l,m}( \alpha, \beta + \phi ) = e^{-i \phi m} O_{l,m}( \alpha, \beta )
                                rotationM2LExpMinusImPhi[position][index_lm] = std::complex<RealType>(std::cos(-mphi), std::sin(-mphi));
                                // M_{l,m}( \alpha, \beta + \phi ) = e^{i \phi m} M_{l,m}( \alpha, \beta )
                                rotationM2LExpImPhi[position][index_lm] = std::complex<RealType>(std::cos(mphi), std::sin(mphi));
                            }
                        }
                        // Then for l < P copy the subpart of the previous vector
                        {
                            int index_lm = 0;
                            for(int l = 0 ; l < P ; ++l){
                                FMemUtils::copyall(rotationM2LExpMinusImPhi[position] + index_lm,
                                                   rotationM2LExpMinusImPhi[position] + index_P0,
                                                   l + 1);
                                FMemUtils::copyall(rotationM2LExpImPhi[position] + index_lm,
                                                   rotationM2LExpImPhi[position] + index_P0,
                                                   l + 1);
                                index_lm += l + 1;
                            }
                        }
                        // Compute inclination vector
                        {
                            // We have to find the right d_lmk matrix
                            int dx = 0 , dy = 0, dz = 0;
                            // if x == 0 && y == 0 it means we have an inclination of 0 or PI
                            if(idxX == 0 && idxY == 0){
                                dx = 0;
                                dy = 0;
                                // no matter if z is big, we want [0][0][1] or [0][0][-1]
                                if( idxZ < 0 ) dz = 1;
                                else dz = -1;
                            }
                            // if z == 0 we have an inclination of Pi/2
                            else if ( idxZ == 0){
                                dx = 1;
                                dy = 0;
                                dz = 0;
                            }
                            // else we take the right indexes
                            else {
                                dx = std::max(std::abs(idxX),std::abs(idxY));
                                dy = std::min(std::abs(idxX),std::abs(idxY));
                                dz = -idxZ;
                            }

                            int index_lmk = 0;
                            for(int l = 0 ; l <= P ; ++l){
                                for(int m = 0 ; m <= l ; ++m ){
                                    { // k == 0
                                        const RealType d_lmk            = dlmkMatrix[dx+3][dy][dz+3][l][m+P][0+P];
                                        const RealType d_lmk_minusTheta = dlmkMatrix[-dx+3][dy][dz+3][l][m+P][0+P];

                                        // \sqrt{ \frac{(l-k)!(l+k)!}{(l-|m|)!(l+|m|)!} }
                                        const RealType Ofactor = sqrtDoubleFactorials[l][0]/sqrtDoubleFactorials[l][m];
                                        const RealType Mfactor = sqrtDoubleFactorials[l][m]/sqrtDoubleFactorials[l][0];

                                        DlmkCoefM2LOTheta[position][index_lmk]      = Ofactor * d_lmk;
                                        DlmkCoefM2LMMinusTheta[position][index_lmk] = Mfactor * d_lmk_minusTheta;
                                        ++index_lmk;
                                    }
                                    RealType minus_1_pow_k = -1.0;
                                    for(int k = 1 ; k <= l ; ++k){
                                        const RealType d_lm_minus_k            = dlmkMatrix[dx+3][dy][dz+3][l][m+P][-k+P];
                                        const RealType d_lmk                   = dlmkMatrix[dx+3][dy][dz+3][l][m+P][k+P];

                                        const RealType d_lm_minus_k_minusTheta = dlmkMatrix[-dx+3][dy][dz+3][l][m+P][-k+P];
                                        const RealType d_lmk_minusTheta        = dlmkMatrix[-dx+3][dy][dz+3][l][m+P][k+P];

                                        const RealType Ofactor = sqrtDoubleFactorials[l][k]/sqrtDoubleFactorials[l][m];
                                        const RealType Mfactor = sqrtDoubleFactorials[l][m]/sqrtDoubleFactorials[l][k];

                                        DlmkCoefM2LOTheta[position][index_lmk]      = Ofactor * (d_lmk + minus_1_pow_k * d_lm_minus_k);
                                        DlmkCoefM2LMMinusTheta[position][index_lmk] = Mfactor * (d_lmk_minusTheta + minus_1_pow_k * d_lm_minus_k_minusTheta);
                                        ++index_lmk;

                                        DlmkCoefM2LOTheta[position][index_lmk]      = Ofactor * (d_lmk - minus_1_pow_k * d_lm_minus_k);
                                        DlmkCoefM2LMMinusTheta[position][index_lmk] = Mfactor * (d_lmk_minusTheta - minus_1_pow_k * d_lm_minus_k_minusTheta);
                                        ++index_lmk;

                                        minus_1_pow_k = -minus_1_pow_k;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Deallocate matrix
        for(int idxX = 0 ; idxX < 7 ; ++idxX)
            for(int idxY = 0 ; idxY < 4 ; ++idxY)
                for(int idxZ = 0 ; idxZ < 7 ; ++idxZ) {
                    delete[] dlmkMatrix[idxX][idxY][idxZ];
                }
    }



    ///////////////////////////////////////////////////////
    // d_lmk computation
    // This part is constitued of 6 functions :
    // DlmkBuild computes the matrix from a angle ]0;pi/2]
    // DlmkBuild0 computes the matrix for angle 0
    // DlmkBuildPi computes the matrix for angle pi
    // Then, others use the d_lmk to build a rotated matrix:
    // DlmkZNegative computes for angle \theta ]pi/2;pi[ using d_lmk(pi- \theta)
    // DlmkInverseZNegative computes for angle \theta ]pi;3pi/2[ using d_lmk(\theta-pi)
    // DlmkInverse computes for angle \theta ]3pi/2;2pi[ using d_lmk(2pi- \theta)
    ///////////////////////////////////////////////////////

    /** Compute d_mlk for \theta = 0
      * \f[
      * d^l_{m,k}( \theta ) = \delta_{m,k,} \,\, \mbox{\textrm{ $\delta$ Kronecker symbol }}
      * \f]
      */
    void DlmkBuild0(RealType dlmk[P+1][P2+1][P2+1]) const {
        for(int l = 0 ; l <= P ; ++l){
            for(int m = -l ; m <= l ; ++m){
                // first put 0 every where
                for(int k = -l ; k <= l ; ++k){
                    dlmk[l][P+m][P+k] = RealType(0.0);
                }
                // then replace per 1 for m == k
                dlmk[l][P+m][P+m] = RealType(1.0);
            }
        }
    }

    /** Compute d_mlk for \theta = PI
      * \f[
      * d^l_{m,k}( \theta ) = (-1)^{l+k} \delta_{m,k},\,\, \mbox{\textrm{ $\delta$ Kronecker delta } }
      * \f]
      */
    void DlmkBuildPi(RealType dlmk[P+1][P2+1][P2+1]) const {
        for(int l = 0 ; l <= P ; ++l){
            for(int m = -l ; m <= l ; ++m){
                // put 0 every where
                for(int k = -l ; k <= l ; ++k){
                    dlmk[l][P+m][P+k] = RealType(0.0);
                }
                // -1^l+k * 1 where m == k
                dlmk[l][P+m][P-m] = ((l+m)&0x1 ? RealType(-1) : RealType(1));
            }
        }
    }

    /** Compute d_mlk for \theta = ]PI/2;PI[
      * \f[
      * d^l_{m,k}( \theta ) = (-1)^{l+m} d^l_{m,-k}( \Pi - \theta )
      * \f]
      */
    void DlmkZNegative(RealType dlmk[P+1][P2+1][P2+1], const RealType dlmkZPositif[P+1][P2+1][P2+1]) const {
        for(int l = 0 ; l <= P ; ++l){
            for(int m = -l ; m <= l ; ++m){
                // if l+m is odd
                if( (l+m)&0x1 ){
                    // put -1 every where
                    for(int k = -l ; k <= l ; ++k){
                        dlmk[l][P+m][P+k] = -dlmkZPositif[l][P+m][P-k];
                    }
                }
                else{
                    // else just copy
                    for(int k = -l ; k <= l ; ++k){
                        dlmk[l][P+m][P+k] = dlmkZPositif[l][P+m][P-k];
                    }
                }
            }
        }
    }

    /** Compute d_mlk for \theta = ]PI;3PI/2[
      * \f[
      * d^l_{m,k}( \theta ) = (-1)^{l+m} d^l_{-m,k}( \theta - \Pi )
      * \f]
      */
    void DlmkInverseZNegative(RealType dlmk[P+1][P2+1][P2+1], const RealType dlmkZPositif[P+1][P2+1][P2+1]) const {
        for(int l = 0 ; l <= P ; ++l){
            for(int m = -l ; m <= l ; ++m){
                if( (l+m)&0x1 ){
                    for(int k = -l ; k <= l ; ++k){
                        dlmk[l][P+m][P+k] = -dlmkZPositif[l][P-m][P+k];
                    }
                }
                else{
                    for(int k = -l ; k <= l ; ++k){
                        dlmk[l][P+m][P+k] = dlmkZPositif[l][P-m][P+k];
                    }
                }
            }
        }
    }

    /** Compute d_mlk for \theta = ]3PI/2;2PI[
      * \f[
      * d^l_{m,k}( \theta ) = (-1)^{m+k} d^l_{m,k}( 2 \Pi - \theta )
      * \f]
      */
    void DlmkInverse(RealType dlmk[P+1][P2+1][P2+1], const RealType dlmkZPositif[P+1][P2+1][P2+1]) const {
        for(int l = 0 ; l <= P ; ++l){
            for(int m = -l ; m <= l ; ++m){
                // we start with k == -l, so if (k+m) is odd
                if( (l+m)&0x1 ){
                    // then we start per (-1)
                    for(int k = -l ; k < l ; k+=2){
                        dlmk[l][P+m][P+k] = -dlmkZPositif[l][P+m][P+k];
                        dlmk[l][P+m][P+k+1] = dlmkZPositif[l][P+m][P+k+1];
                    }
                    // l is always odd
                    dlmk[l][P+m][P+l] = -dlmkZPositif[l][P+m][P+l];
                }
                else{
                    // else we start per (+1)
                    for(int k = -l ; k < l ; k+=2){
                        dlmk[l][P+m][P+k] = dlmkZPositif[l][P+m][P+k];
                        dlmk[l][P+m][P+k+1] = -dlmkZPositif[l][P+m][P+k+1];
                    }
                    // l is always odd
                    dlmk[l][P+m][P+l] = dlmkZPositif[l][P+m][P+l];
                }
            }
        }
    }


    /** Compute d_mlk for \theta = ]0;PI/2[
      * This used the second formula from the paper:
      * Fast and accurate determination of the wigner rotation matrices in FMM
      *
      * We use formula 28,29 to compute "g"
      * Then 25,26,27 for the recurrence.
      */
    // theta should be between [0;pi] as the inclinaison angle
    void DlmkBuild(RealType dlmk[P+1][P2+1][P2+1], const RealType inTheta) const {
        assert(0 <= inTheta && inTheta < PI2);
        // To have constants for very used values
        const RealType F0 = RealType(0.0);
        const RealType F1 = RealType(1.0);
        const RealType F2 = RealType(2.0);

        const RealType cosTheta = std::cos(inTheta);
        const RealType sinTheta = std::sin(inTheta);

        // First compute g
        RealType g[SizeArray];
        {// Equ 29
            // g{0,0} = 1
            g[0] = F1;

            // g{l,0} = sqrt( (2l - 1) / 2l) g{l-1,0}  for l > 0
            {
                int index_l0 = 1;
                RealType fl = F1;
                for(int l = 1; l <= P ; ++l, ++fl ){
                    g[index_l0] = std::sqrt((fl*F2-F1)/(fl*F2)) * g[index_l0-l];
                    index_l0 += l + 1;
                }
            }
            // g{l,m} = sqrt( (l - m + 1) / (l+m)) g{l,m-1}  for l > 0, 0 < m <= l
            {
                int index_lm = 2;
                RealType fl = F1;
                for(int l = 1; l <= P ; ++l, ++fl ){
                    RealType fm = F1;
                    for(int m = 1; m <= l ; ++m, ++index_lm, ++fm ){
                        g[index_lm] = std::sqrt((fl-fm+F1)/(fl+fm)) * g[index_lm-1];
                    }
                    ++index_lm;
                }
            }
        }
        { // initial condition
            // Equ 28
            // d{l,m,l} = -1^(l+m) g{l,m} (1+cos(theta))^m sin(theta)^(l-m) , For l > 0, 0 <= m <= l
            int index_lm = 0;
            RealType sinTheta_pow_l = F1;
            for(int l = 0 ; l <= P ; ++l){
                // build variable iteratively
                RealType minus_1_pow_lm = l&0x1 ? RealType(-1) : RealType(1);
                RealType cosTheta_1_pow_m = F1;
                RealType sinTheta_pow_l_minus_m = sinTheta_pow_l;
                for(int m = 0 ; m <= l ; ++m, ++index_lm){
                    dlmk[l][P+m][P+l] = minus_1_pow_lm * g[index_lm] * cosTheta_1_pow_m * sinTheta_pow_l_minus_m;
                    // update
                    minus_1_pow_lm = -minus_1_pow_lm;
                    cosTheta_1_pow_m *= F1 + cosTheta;
                    sinTheta_pow_l_minus_m /= sinTheta;
                }
                // update
                sinTheta_pow_l *= sinTheta;
            }
        }
        { // build the rest of the matrix
            RealType fl = F1;
            for(int l = 1 ; l <= P ; ++l, ++fl){
                RealType fk = fl;
                for(int k = l ; k > -l ; --k, --fk){
                    // Equ 25
                    // For l > 0, 0 <= m < l, -l < k <= l, cos(theta) >= 0
                    // d{l,m,k-1} = sqrt( l(l+1) - m(m+1) / l(l+1) - k(k-1)) d{l,m+1,k}
                    //            + (m+k) sin(theta) d{l,m,k} / sqrt(l(l+1) - k(k-1)) (1+cos(theta))
                    RealType fm = F0;
                    for(int m = 0 ; m < l ; ++m, ++fm){
                        dlmk[l][P+m][P+k-1] =
                                (std::sqrt((fl*(fl+F1)-fm*(fm+F1))/(fl*(fl+F1)-fk*(fk-F1))) * dlmk[l][P+m+1][P+k])
                                + ((fm+fk)*sinTheta*dlmk[l][P+m][P+k]/(std::sqrt(fl*(fl+F1)-fk*(fk-F1))*(F1+cosTheta)));
                    }
                    // Equ 26
                    // For l > 0, -l < k <= l, cos(theta) >= 0
                    // d{l,l,k-1} = (l+k) sin(theta) d{l,l,k}
                    //             / sqrt(l(l+1)-k(k-1)) (1+cos(theta))
                    dlmk[l][P+l][P+k-1] = (fl+fk)*sinTheta*dlmk[l][P+l][P+k]/(std::sqrt(fl*(fl+F1)-fk*(fk-F1))*(F1+cosTheta));
                }
                // Equ 27
                // d{l,m,k} = -1^(m+k) d{l,-m,-k}  , For l > 0, -l <= m < 0, -l <= k <= l
                for(int m = -l ; m < 0 ; ++m){
                    RealType minus_1_pow_mk = (m-l)&0x1 ? RealType(-1) : RealType(1);
                    for(int k = -l ; k <= l ; ++k){
                        dlmk[l][P+m][P+k] = minus_1_pow_mk * dlmk[l][P-m][P-k];
                        minus_1_pow_mk = -minus_1_pow_mk;
                    }
                }
            }
        }
    }

    /** Compute the legendre polynomial from {0,0} to {P,P}
      * the computation is made by recurence (P cannot be equal to 0)
      *
      * The formula has been taken from:
      * Fast and accurate determination of the wigner rotation matrices in the fast multipole method
      * Formula number (22)
      * \f[
      * P_{0,0} = 1
      * P_{l,l} = (2l-1) sin( \theta ) P_{l-1,l-1} ,l \ge 0
      * P_{l,l-1} = (2l-1) cos( \theta ) P_{l-1,l-1} ,l \ge 0
      * P_{l,m} = \frac{(2l-1) cos( \theta ) P_{l-1,m} - (l+m-1) P_{l-2,m}x}{(l-k)} ,l \ge 1, 0 \leq m \le l-1
      * \f]
      */
    void computeLegendre(RealType legendre[], const RealType inCosTheta, const RealType inSinTheta) const {
        const RealType invSinTheta = -inSinTheta;

        legendre[0] = 1.0;             // P_0,0(1) = 1

        legendre[1] = inCosTheta;      // P_1,0 = cos(theta)
        legendre[2] = invSinTheta;     // P_1,1 = -sin(theta)

        // work with pointers
        RealType* legendre_l1_m1 = legendre;     // P{l-2,m} starts with P_{0,0}
        RealType* legendre_l1_m  = legendre + 1; // P{l-1,m} starts with P_{1,0}
        RealType* legendre_lm  = legendre + 3;   // P{l,m} starts with P_{2,0}

        // Compute using recurrence
        RealType l2_minus_1 = 3; // 2 * l - 1
        RealType fl = RealType(2.0);// To get 'l' as a float
        for(int l = 2; l <= P ; ++l, ++fl ){
            RealType lm_minus_1 = fl - RealType(1.0); // l + m - 1
            RealType l_minus_m = fl;               // l - m
            for( int m = 0; m < l - 1 ; ++m ){
                // P_{l,m} = \frac{(2l-1) cos( \theta ) P_{l-1,m} - (l+m-1) P_{l-2,m}x}{(l-m)}
                *(legendre_lm++) = (l2_minus_1 * inCosTheta * (*legendre_l1_m++) - (lm_minus_1++) * (*legendre_l1_m1++) )
                        / (l_minus_m--);
            }
            // P_{l,l-1} = (2l-1) cos( \theta ) P_{l-1,l-1}
            *(legendre_lm++) = l2_minus_1 * inCosTheta * (*legendre_l1_m);
            // P_{l,l} = (2l-1) sin( \theta ) P_{l-1,l-1}
            *(legendre_lm++) = l2_minus_1 * invSinTheta * (*legendre_l1_m);
            // goto P_{l-1,0}
            ++legendre_l1_m;
            l2_minus_1 += RealType(2.0); // 2 * l - 1 => progress by two
        }
    }

    ///////////////////////////////////////////////////////
    // Multiplication for rotation
    // Here we have two function that are optimized
    // to compute the rotation fast!
    ///////////////////////////////////////////////////////

    /** This function use a d_lmk vector to rotate the vec
      * multipole or local vector.
      * The result is copyed in vec.
      * Please see the structure of dlmk to understand this function.
      * Warning we cast the vec std::complex<RealType> array into a RealType array
      */
    static void RotationYWithDlmk(std::complex<RealType> vec[], const RealType* dlmkCoef){
        RealType originalVec[2*SizeArray];
        FMemUtils::copyall(originalVec,reinterpret_cast<const RealType*>(vec),2*SizeArray);
        // index_lm == atLm(l,m) but progress iteratively to write the result
        int index_lm = 0;
        for(int l = 0 ; l <= P ; ++l){
            const RealType*const originalVecAtL0 = originalVec + (index_lm * 2);
            for(int m = 0 ; m <= l ; ++m, ++index_lm ){
                RealType res_lkm_real = 0.0;
                RealType res_lkm_imag = 0.0;
                // To read all "m" value for current "l"
                const RealType* iterOrignalVec = originalVecAtL0;
                { // for k == 0
                    // same coef for real and imaginary
                    res_lkm_real += (*dlmkCoef) * (*iterOrignalVec++);
                    res_lkm_imag += (*dlmkCoef++) * (*iterOrignalVec++);
                }
                for(int k = 1 ; k <= l ; ++k){
                    // coef contains first real value
                    res_lkm_real += (*dlmkCoef++) * (*iterOrignalVec++);
                    // then imaginary
                    res_lkm_imag += (*dlmkCoef++) * (*iterOrignalVec++);
                }
                // save the result
                vec[index_lm] = std::complex<RealType>(res_lkm_real, res_lkm_imag);
            }
        }
    }

    /** This function is computing dest[:] *= src[:]
      * it computes inSize std::complex<RealType> multiplication
      * to do so we first proceed per 4 and the the inSize%4 rest
      */
    static void RotationZVectorsMul(std::complex<RealType>* dest, const std::complex<RealType>* src, const int inSize = SizeArray){
        const std::complex<RealType>*const lastElement = dest + inSize;
        const std::complex<RealType>*const intermediateLastElement = dest + (inSize & ~0x3);
        // first the inSize - inSize%4 elements
        for(; dest != intermediateLastElement ;) {
            (*dest++) *= (*src++);
            (*dest++) *= (*src++);
            (*dest++) *= (*src++);
            (*dest++) *= (*src++);
        }
        // then the rest
        for(; dest != lastElement ;) {
            (*dest++) *= (*src++);
        }
    }

    ///////////////////////////////////////////////////////
    // Utils
    ///////////////////////////////////////////////////////


    /** Return the position of a leaf from its tree coordinate
      * This is used only for the leaf
      */
    std::array<RealType,3> getLeafCenter(const std::array<long int, 3>& coordinate) const {
        return std::array<RealType, 3>{boxCorner[0] + (RealType(coordinate[0]) + RealType(.5)) * widthAtLeafLevel,
                      boxCorner[1] + (RealType(coordinate[1]) + RealType(.5)) * widthAtLeafLevel,
                      boxCorner[2] + (RealType(coordinate[2]) + RealType(.5)) * widthAtLeafLevel};
    }

    std::array<RealType, 3> getLeafCenter(const typename SpaceIndexType::IndexType& inIndex) const{
        return getLeafCenter(spaceIndexSystem.getBoxPosFromIndex(inIndex));
    }

    /** Return position in the array of the l/m couple
      * P[atLm(l,m)] => P{l,m}
      * 0
      * 1 2
      * 3 4 5
      * 6 7 8 9 ...
      */
    int atLm(const int l, const int m) const {
        // summation series over l + m => (l*(l+1))/2 + m
        return ((l*(l+1))>>1) + m;
    }

public:

    /** Constructor, needs system information */
    FRotationKernel(const SpacialConfiguration& inConfiguration) :
        spaceIndexSystem(inConfiguration),
        boxWidth(inConfiguration.getBoxWidths()[0]),
        treeHeight(int(inConfiguration.getTreeHeight())),
        widthAtLeafLevel(inConfiguration.getLeafWidths()[0]),
        widthAtLeafLevelDiv2(widthAtLeafLevel/2),
        boxCorner(inConfiguration.getBoxCorner())
    {
        // simply does the precomputation
        precomputeFactorials();
        precomputeTranslationCoef();
        precomputeRotationVectors();
    }

    /** Copy Constructor */
    FRotationKernel(const FRotationKernel& other) :
        spaceIndexSystem(other.spaceIndexSystem),
        boxWidth(other.boxWidth),
        treeHeight(other.treeHeight),
        widthAtLeafLevel(other.widthAtLeafLevel),
        widthAtLeafLevelDiv2(other.widthAtLeafLevelDiv2),
        boxCorner(other.boxCorner)
    {
        // simply does the precomputation
        precomputeFactorials();
        precomputeTranslationCoef();
        precomputeRotationVectors();
    }

    /** Default destructor */
    virtual ~FRotationKernel(){
    }

    /** P2M
      * The computation is based on the paper :
      * Parallelization of the fast multipole method
      * Formula number 10, page 3
      * \f[
      * \omega (q,a) = q \frac{a^{l}}{(l+|m|)!} P_{lm}(cos( \alpha ) )e^{-im \beta}
      * \f]
      */
    template <class CellSymbolicData, class ParticlesClass, class LeafClass>
    void P2M(const CellSymbolicData& LeafIndex,  const long int /*particlesIndexes*/[],
             const ParticlesClass& SourceParticles, const long int inNbParticles, LeafClass& LeafCell)
    {
        const RealType i_pow_m[4] = {0, PIDiv2, PI, -PIDiv2};
        // w is the multipole moment
        std::complex<RealType>* const w = &LeafCell[0];

        // Copying the position is faster than using cell position
        const std::array<RealType,3> cellPosition = getLeafCenter(LeafIndex.boxCoord);

        // We need a legendre array
        RealType legendre[SizeArray];
        RealType angles[P+1][2];

        // For all particles in the leaf box
        const RealType*const physicalValues = SourceParticles[3];
        const RealType*const positionsX = SourceParticles[0];
        const RealType*const positionsY = SourceParticles[1];
        const RealType*const positionsZ = SourceParticles[2];

        for(long int idxPart = 0 ; idxPart < inNbParticles ; ++ idxPart){
            // P2M
            const std::array<RealType,3> relativePosition{{positionsX[idxPart] - cellPosition[0],
                                                   positionsY[idxPart] - cellPosition[1],
                                                   positionsZ[idxPart] - cellPosition[2]}};
            const FSpherical<RealType> sph(relativePosition);

            // The physical value (charge, mass)
            const RealType q = physicalValues[idxPart];
            // The distance between the SH and the particle
            const RealType a = sph.getR();

            // Compute the legendre polynomial
            computeLegendre(legendre, sph.getCosTheta(), sph.getSinTheta());

            // w{l,m}(q,a) = q a^l/(l+|m|)! P{l,m}(cos(alpha)) exp(-i m Beta)
            RealType q_aPowL = q; // To consutrct q*a^l continously
            int index_l_m = 0; // To construct the index of (l,m) continously
            RealType fl = 0.0;
            for(int l = 0 ; l <= P ; ++l, ++fl ){
                { // We need to compute the angles to use in the "m" loop
                    // So we can compute only the one needed after "l" inc
                    const RealType angle = fl * sph.getPhi() + i_pow_m[l & 0x3];
                    angles[l][0] = std::cos(angle);
                    angles[l][1] = std::sin(angle);
                }
                for(int m = 0 ; m <= l ; ++m, ++index_l_m){
                    const RealType magnitude = q_aPowL * legendre[index_l_m] / factorials[l+m];
                    w[index_l_m].real(w[index_l_m].real() + magnitude * angles[m][0]);
                    w[index_l_m].imag(w[index_l_m].imag() + magnitude * angles[m][1]);
                }
                q_aPowL *= a;
            }
        }
    }

    /** M2M
      * The operator A has been taken from :
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1 as the operator A
      * \f[
      * O_{l,m}(a+b') = \sum_{j=|m|}^l{ \frac{ b^{l-j} }{ (l-j)! } O_{j,m}(a) }
      * \f]
      * As describe in the paper, when need first to rotate the SH
      * then transfer using the formula
      * and finaly rotate back.
      */
    template <class CellSymbolicData, class CellClassContainer, class CellClass>
    void M2M(const CellSymbolicData& /*inParentIndex*/,
             const long int inLevel, const CellClassContainer& inLowerCell, CellClass& inOutUpperCell,
             const long int childrenPos[], const long int inNbChildren) const {
        // Get the translation coef for this level (same for all child)
        const RealType (&coef)[P+1] = ((RealType(*)[P+1])M2MTranslationCoef.get())[inLevel];
        // A buffer to copy the source w allocated once
        std::complex<RealType> source_w[SizeArray];
        // For all children
        for(int idxChild = 0 ; idxChild < inNbChildren ; ++idxChild){
            // Copy the source
            FMemUtils::copyall(source_w, inLowerCell[idxChild].get(), SizeArray);

            // rotate it forward
            RotationZVectorsMul(source_w,rotationExpMinusImPhi[childrenPos[idxChild]]);
            RotationYWithDlmk(source_w,DlmkCoefOTheta[childrenPos[idxChild]]);

            // Translate it
            std::complex<RealType> target_w[SizeArray];
            int index_lm = 0;
            for(int l = 0 ; l <= P ; ++l ){
                for(int m = 0 ; m <= l ; ++m, ++index_lm ){
                    // w{l,m}(a+b) = sum(j=m:l, b^(l-j)/(l-j)! w{j,m}(a)
                    RealType w_lm_real = 0.0;
                    RealType w_lm_imag = 0.0;
                    int index_jm = atLm(m,m);   // get atLm(l,m)
                    int index_l_minus_j = l-m;  // get l-j continuously
                    for(int j = m ; j <= l ; ++j, --index_l_minus_j, index_jm += j ){
                        //const coef = (b^l-j) / (l-j)!;
                        w_lm_real += coef[index_l_minus_j] * source_w[index_jm].real();
                        w_lm_imag += coef[index_l_minus_j] * source_w[index_jm].imag();
                    }
                    target_w[index_lm] = std::complex<RealType>(w_lm_real,w_lm_imag);
                }
            }

            // Rotate it back
            RotationYWithDlmk(target_w,DlmkCoefOMinusTheta[childrenPos[idxChild]]);
            RotationZVectorsMul(target_w,rotationExpImPhi[childrenPos[idxChild]]);

            // Sum the result
            FMemUtils::addall( inOutUpperCell, target_w, SizeArray);
        }
    }

    /** M2L
      * The operator B has been taken from :
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1 as the operator B
      * \f[
      * M_{l,m}(a-b') = \sum_{j=|m|}^{\infty}{ \frac{ (j+l)! } { b^{j+l+1} } O_{j,-m}(a) } , \mbox{\textrm{ j bounded by P-l } }
      * \f]
      * As describe in the paper, when need first to rotate the SH
      * then transfer using the formula
      * and finaly rotate back.
      */
    template <class CellSymbolicData, class CellClassContainer, class CellClass>
    void M2L(const CellSymbolicData& /*inTargetIndex*/,
             const long int inLevel, const CellClassContainer& inInteractingCells, const long int neighPos[], const long int inNbNeighbors,
             CellClass& inOutCell) {
        // To copy the multipole data allocated once
        std::complex<RealType> source_w[SizeArray];
        // For all children
        for(int idxNeigh = 0 ; idxNeigh < inNbNeighbors ; ++idxNeigh){
            const RealType (&coef)[P+1] = ((RealType(*)[343][P+1])M2LTranslationCoef.get())[inLevel][neighPos[idxNeigh]];
            // Copy multipole data into buffer
            FMemUtils::copyall(source_w, inInteractingCells[idxNeigh].get(), SizeArray);

            // Rotate
            RotationZVectorsMul(source_w,rotationM2LExpMinusImPhi[neighPos[idxNeigh]]);
            RotationYWithDlmk(source_w,DlmkCoefM2LOTheta[neighPos[idxNeigh]]);

            // Transfer to u
            std::complex<RealType> target_u[SizeArray];
            int index_lm = 0;
            for(int l = 0 ; l <= P ; ++l ){
                RealType minus_1_pow_m = 1.0;
                for(int m = 0 ; m <= l ; ++m, ++index_lm ){
                    // u{l,m}(a-b) = sum(j=|m|:P-l, (j+l)!/b^(j+l+1) w{j,-m}(a)
                    RealType u_lm_real = 0.0;
                    RealType u_lm_imag = 0.0;
                    int index_jl = m + l;       // get j+l
                    int index_jm = atLm(m,m);   // get atLm(l,m)
                    for(int j = m ; j <= P-l ; ++j, ++index_jl, index_jm += j ){
                        // coef = (j+l)!/b^(j+l+1)
                        // because {l,-m} => {l,m} conjugate -1^m with -i
                        u_lm_real += minus_1_pow_m * coef[index_jl] * source_w[index_jm].real();
                        u_lm_imag -= minus_1_pow_m * coef[index_jl] * source_w[index_jm].imag();
                    }
                    target_u[index_lm] = std::complex<RealType>(u_lm_real,u_lm_imag);
                    minus_1_pow_m = -minus_1_pow_m;
                }
            }

            // Rotate it back
            RotationYWithDlmk(target_u,DlmkCoefM2LMMinusTheta[neighPos[idxNeigh]]);
            RotationZVectorsMul(target_u,rotationM2LExpMinusImPhi[neighPos[idxNeigh]]);

            // Sum
            FMemUtils::addall(inOutCell, target_u, SizeArray);
        }
    }

    /** L2L
      * The operator C has been taken from :
      * Implementation of rotation-based operators for Fast Multipole Method in X10
      * At page 5 .1 as the operator C
      * \f[
      * M_{l,m}(a-b') = \sum_{j=l}^{\infty}{ \frac{ b^{j-l} }{ (j-l)! } M_{j,m}(a) } , \textrm{j bounded by P}
      * \f]
      * As describe in the paper, when need first to rotate the SH
      * then transfer using the formula
      * and finaly rotate back.
      */
    template <class CellSymbolicData, class CellClass, class CellClassContainer>
    void L2L(const CellSymbolicData& /*inParentIndex*/,
             const long int inLevel, const CellClass& inUpperCell, CellClassContainer& inOutLowerCell,
             const long int childrenPos[], const long int inNbChildren) {
        // Get the translation coef for this level (same for all chidl)
        const RealType (&coef)[P+1] = ((RealType(*)[P+1])L2LTranslationCoef.get())[inLevel];
        // To copy the source local allocated once
        std::complex<RealType> source_u[SizeArray];
        // For all children
        for(int idxChild = 0 ; idxChild < inNbChildren ; ++idxChild){
            // Copy the local data into the buffer
            FMemUtils::copyall(source_u, inUpperCell, SizeArray);

            // Rotate
            RotationZVectorsMul(source_u,rotationExpImPhi[childrenPos[idxChild]]);
            RotationYWithDlmk(source_u,DlmkCoefMTheta[childrenPos[idxChild]]);

            // Translate
            std::complex<RealType> target_u[SizeArray];
            for(int l = 0 ; l <= P ; ++l ){
                for(int m = 0 ; m <= l ; ++m ){
                    // u{l,m}(r-b) = sum(j=0:P, b^(j-l)/(j-l)! u{j,m}(r);
                    RealType u_lm_real = 0.0;
                    RealType u_lm_imag = 0.0;
                    int index_jm = atLm(l,m);   // get atLm(j,m)
                    int index_j_minus_l = 0;    // get l-j continously
                    for(int j = l ; j <= P ; ++j, ++index_j_minus_l, index_jm += j){
                        // coef = b^j-l/j-l!
                        u_lm_real += coef[index_j_minus_l] * source_u[index_jm].real();
                        u_lm_imag += coef[index_j_minus_l] * source_u[index_jm].imag();
                    }
                    target_u[atLm(l,m)] = std::complex<RealType>(u_lm_real,u_lm_imag);
                }
            }

            // Rotate
            RotationYWithDlmk(target_u,DlmkCoefMMinusTheta[childrenPos[idxChild]]);
            RotationZVectorsMul(target_u,rotationExpMinusImPhi[childrenPos[idxChild]]);

            // Sum in child
            FMemUtils::addall(inOutLowerCell[idxChild].get(), target_u, SizeArray);
        }
    }

    /** L2P
      * Equation are coming from the PhD report of Pierre Fortin.
      * We have two different computations, one for the potential (end of function)
      * the other for the forces.
      *
      * The potential use the fallowing formula, page 36, formula 2.14 + 1:
      * \f[
      *  \Phi = \sum_{j=0}^P{\left( u_{j,0} I_{j,0}(r, \theta, \phi) + \sum_{k=1}^j{2 Re(u_{j,k} I_{j,k}(r, \theta, \phi))} \right)},
      *  \textrm{since } u_{l,-m} = (-1)^m \overline{ u_{l,m} }
      * \f]
      *
      * The forces are coming form the formulas, page 37, formulas 2.14 + 3:
      * \f[
      * F_r = -\frac{1}{r} \left( \sum_{j=1}^P{j u_{j,0} I_{j,0}(r, \theta, \phi) } + \sum_{k=1}^j{2 j Re(u_{j,k} I_{j,k}(r, \theta, \phi))} \right)
      * F_{ \theta } = -\frac{1}{r} \left( \sum_{j=0}^P{j u_{j,0} \frac{ \partial I_{j,0}(r, \theta, \phi) }{ \partial \theta } } + \sum_{k=1}^j{2 Re(u_{j,k} \frac{ \partial I_{j,k}(r, \theta, \phi) }{ \partial \theta })} \right)
      * F_{ \phi } = -\frac{1}{r sin \phi} \sum_{j=0}^P \sum_{k=1}^j{(-2k) Im(u_{j,k} I_{j,k}(r, \theta, \phi)) }
      * \f]
      */
    template <class CellSymbolicData, class LeafClass, class ParticlesClass, class ParticlesClassRhs>
    void L2P(const CellSymbolicData& LeafIndex,
             const LeafClass& LeafCell, const long int /*particlesIndexes*/[],
             const ParticlesClass& inOutParticles, ParticlesClassRhs& inOutParticlesRhs,
             const long int inNbParticles) {
        const RealType i_pow_m[4] = {0, PIDiv2, PI, -PIDiv2};
        // Take the local value from the cell
        const std::complex<RealType>* const u = &LeafCell[0];

        // Copying the position is faster than using cell position
        const std::array<RealType,3> cellPosition = getLeafCenter(LeafIndex.boxCoord);

        // For all particles in the leaf box
        const RealType*const physicalValues = inOutParticles[3];
        const RealType*const positionsX = inOutParticles[0];
        const RealType*const positionsY = inOutParticles[1];
        const RealType*const positionsZ = inOutParticles[2];
        RealType*const forcesX = inOutParticlesRhs[0];
        RealType*const forcesY = inOutParticlesRhs[1];
        RealType*const forcesZ = inOutParticlesRhs[2];
        RealType*const potentials = inOutParticlesRhs[3];

        for(long int idxPart = 0 ; idxPart < inNbParticles ; ++ idxPart){
            // L2P
            const std::array<RealType,3> relativePosition{{positionsX[idxPart]-cellPosition[0],
                                                   positionsY[idxPart]-cellPosition[1],
                                                   positionsZ[idxPart]-cellPosition[2]}};
            const FSpherical<RealType> sph(relativePosition);

            // The distance between the SH and the particle
            const RealType r = sph.getR();

            // Compute the legendre polynomial
            RealType legendre[SizeArray];
            computeLegendre(legendre, sph.getCosTheta(), sph.getSinTheta());

            // pre compute what is used more than once
            RealType minus_r_pow_l_div_fact_lm[SizeArray];
            RealType minus_r_pow_l_legendre_div_fact_lm[SizeArray];
            {
                int index_lm = 0;
                RealType minus_r_pow_l = 1.0;  // To get (-1*r)^l
                for(int l = 0 ; l <= P ; ++l){
                    for(int m = 0 ; m <= l ; ++m, ++index_lm){
                        minus_r_pow_l_div_fact_lm[index_lm] = minus_r_pow_l / factorials[l+m];
                        minus_r_pow_l_legendre_div_fact_lm[index_lm] = minus_r_pow_l_div_fact_lm[index_lm] * legendre[index_lm];
                    }
                    minus_r_pow_l *= -r;
                }
            }
            // pre compute what is use more than once
            RealType cos_m_phi_i_pow_m[P+1];
            RealType sin_m_phi_i_pow_m[P+1];
            {
                for(int m = 0 ; m <= P ; ++m){
                    const RealType m_phi_i_pow_m = RealType(m) * sph.getPhi() + i_pow_m[m & 0x3];
                    cos_m_phi_i_pow_m[m] = std::cos(m_phi_i_pow_m);
                    sin_m_phi_i_pow_m[m] = std::sin(m_phi_i_pow_m);
                }
            }

            // compute the forces
            {
                RealType Fr = 0;
                RealType FO = 0;
                RealType Fp = 0;

                int index_lm = 1;          // To get atLm(l,m), warning starts with l = 1
                RealType fl = 1.0;            // To get "l" as a float

                for(int l = 1 ; l <= P ; ++l, ++fl){
                    // first m == 0
                    {
                        Fr += fl * u[index_lm].real() * minus_r_pow_l_legendre_div_fact_lm[index_lm];
                    }
                    {
                        const RealType coef = minus_r_pow_l_div_fact_lm[index_lm] * (fl * (sph.getCosTheta()*legendre[index_lm]
                                                                                        - legendre[index_lm-l]) / sph.getSinTheta());
                        const RealType dI_real = coef;
                        // F(O) += 2 * Real(L dI/dO)
                        FO += u[index_lm].real() * dI_real;
                    }
                    ++index_lm;
                    // then 0 < m
                    for(int m = 1 ; m <= l ; ++m, ++index_lm){
                        {
                            const RealType coef = minus_r_pow_l_legendre_div_fact_lm[index_lm];
                            const RealType I_real = coef * cos_m_phi_i_pow_m[m];
                            const RealType I_imag = coef * sin_m_phi_i_pow_m[m];
                            // F(r) += 2 x l x Real(LI)
                            Fr += 2 * fl * (u[index_lm].real() * I_real - u[index_lm].imag() * I_imag);
                            // F(p) += -2 x m x Imag(LI)
                            Fp -= 2 * RealType(m) * (u[index_lm].real() * I_imag + u[index_lm].imag() * I_real);
                        }
                        {
                            const RealType legendre_l_minus_1 = (m == l) ? RealType(0.0) : RealType(l+m)*legendre[index_lm-l];
                            const RealType coef = minus_r_pow_l_div_fact_lm[index_lm] * ((fl * sph.getCosTheta()*legendre[index_lm]
                                                                                       - legendre_l_minus_1) / sph.getSinTheta());
                            const RealType dI_real = coef * cos_m_phi_i_pow_m[m];
                            const RealType dI_imag = coef * sin_m_phi_i_pow_m[m];
                            // F(O) += 2 * Real(L dI/dO)
                            FO += RealType(2.0) * (u[index_lm].real() * dI_real - u[index_lm].imag() * dI_imag);
                        }
                    }
                }
                // div by r
                Fr /= sph.getR();
                FO /= sph.getR();
                Fp /= sph.getR() * sph.getSinTheta();

                // copy variable from spherical position
                const RealType cosPhi     = std::cos(sph.getPhi());
                const RealType sinPhi     = std::sin(sph.getPhi());
                const RealType physicalValue = physicalValues[idxPart];

                // compute forces
                const RealType forceX = (
                            cosPhi * sph.getSinTheta() * Fr  +
                            cosPhi * sph.getCosTheta() * FO +
                            (-sinPhi) * Fp) * physicalValue;

                const RealType forceY = (
                            sinPhi * sph.getSinTheta() * Fr  +
                            sinPhi * sph.getCosTheta() * FO +
                            cosPhi * Fp) * physicalValue;

                const RealType forceZ = (
                            sph.getCosTheta() * Fr +
                            (-sph.getSinTheta()) * FO) * physicalValue;

                // inc particles forces
                forcesX[idxPart] += forceX;
                forcesY[idxPart] += forceY;
                forcesZ[idxPart] += forceZ;
            }
            // compute the potential
            {
                RealType magnitude = 0;
                // E = sum( l = 0:P, sum(m = -l:l, u{l,m} ))
                int index_lm = 0;
                for(int l = 0 ; l <= P ; ++l ){
                    {//for m == 0
                        // (l-|m|)! * P{l,0} / r^(l+1)
                        magnitude += u[index_lm].real() * minus_r_pow_l_legendre_div_fact_lm[index_lm];
                        ++index_lm;
                    }
                    for(int m = 1 ; m <= l ; ++m, ++index_lm ){
                        const RealType coef = minus_r_pow_l_legendre_div_fact_lm[index_lm];
                        const RealType I_real = coef * cos_m_phi_i_pow_m[m];
                        const RealType I_imag = coef * sin_m_phi_i_pow_m[m];
                        magnitude += RealType(2.0) * ( u[index_lm].real() * I_real - u[index_lm].imag() * I_imag );
                    }
                }
                // inc potential
                potentials[idxPart] += magnitude;
            }
        }
    }


    template <class LeafSymbolicData, class ParticlesClassValues, class ParticlesClassRhs>
    void P2P(const LeafSymbolicData& inNeighborIndex, const long int /*neighborsIndexes*/[],
             const ParticlesClassValues& inNeighbors, ParticlesClassRhs& inNeighborsRhs, const long int inNbParticlesNeighbors,
             const LeafSymbolicData& inTargetIndex,  const long int /*targetIndexes*/[],
             const ParticlesClassValues& inTargets,
             ParticlesClassRhs& inTargetsRhs, const long int inNbOutParticles,
             [[maybe_unused]] const long arrayIndexSrc) const {
        if constexpr(SpaceIndexType::IsPeriodic){
            using PeriodicShifter = typename TbfPeriodicShifter<RealType, SpaceIndexType>::Neighbor;
            if(PeriodicShifter::NeedToShift(inNeighborIndex, inTargetIndex, spaceIndexSystem, arrayIndexSrc)){
                const auto duplicateSources = PeriodicShifter::DuplicatePositionsAndApplyShift(inNeighborIndex, inTargetIndex, spaceIndexSystem, arrayIndexSrc,
                                                                            inNeighbors, inNbParticlesNeighbors);
                FP2PR::template FullMutual<RealType> ((duplicateSources),(inNeighborsRhs), inNbParticlesNeighbors,
                                                             (inTargets), (inTargetsRhs), inNbOutParticles);
                PeriodicShifter::FreePositions(duplicateSources);
            }
            else{
                FP2PR::template FullMutual<RealType> ((inNeighbors),(inNeighborsRhs), inNbParticlesNeighbors,
                                                             (inTargets), (inTargetsRhs), inNbOutParticles);
            }
        }
        else{
            FP2PR::template FullMutual<RealType> ((inNeighbors),(inNeighborsRhs), inNbParticlesNeighbors,
                                                                                       (inTargets), (inTargetsRhs), inNbOutParticles);
        }
    }

    template <class LeafSymbolicDataSource, class ParticlesClassValuesSource, class LeafSymbolicDataTarget, class ParticlesClassValuesTarget, class ParticlesClassRhs>
    void P2PTsm(const LeafSymbolicDataSource& inNeighborIndex, const long int /*neighborsIndexes*/[],
             const ParticlesClassValuesSource& inNeighbors,
             const long int inNbParticlesNeighbors,
             const LeafSymbolicDataTarget& inTargetIndex, const long int /*targetIndexes*/[],
             const ParticlesClassValuesTarget& inTargets,
             ParticlesClassRhs& inTargetsRhs, const long int inNbOutParticles,
             [[maybe_unused]] const long arrayIndexSrc) const {
        if constexpr(SpaceIndexType::IsPeriodic){
            using PeriodicShifter = typename TbfPeriodicShifter<RealType, SpaceIndexType>::Neighbor;
            if(PeriodicShifter::NeedToShift(inNeighborIndex, inTargetIndex, spaceIndexSystem, arrayIndexSrc)){
                const auto duplicateSources = PeriodicShifter::DuplicatePositionsAndApplyShift(inNeighborIndex, inTargetIndex, spaceIndexSystem, arrayIndexSrc,
                                                                            inNeighbors, inNbParticlesNeighbors);
                FP2PR::template GenericFullRemote<RealType> ((duplicateSources), inNbParticlesNeighbors,
                                                             (inTargets), (inTargetsRhs), inNbOutParticles);
                PeriodicShifter::FreePositions(duplicateSources);
            }
            else{
                FP2PR::template GenericFullRemote<RealType> ((inNeighbors), inNbParticlesNeighbors,
                                                             (inTargets), (inTargetsRhs), inNbOutParticles);
            }
        }
        else{
            FP2PR::template GenericFullRemote<RealType> ((inNeighbors), inNbParticlesNeighbors,
                                                                                       (inTargets), (inTargetsRhs), inNbOutParticles);
        }
    }

    template <class LeafSymbolicData, class ParticlesClassValues, class ParticlesClassRhs>
    void P2PInner(const LeafSymbolicData& /*inIndex*/, const long int /*indexes*/[],
                  const ParticlesClassValues& inTargets,
                  ParticlesClassRhs& inTargetsRhs, const long int inNbOutParticles) const {
        FP2PR::template GenericInner<RealType>((inTargets),(inTargetsRhs), inNbOutParticles);
    }
};


#endif // FROTATIONKERNEL_HPP
