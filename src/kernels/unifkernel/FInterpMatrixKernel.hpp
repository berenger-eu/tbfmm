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
#ifndef FINTERPMATRIXKERNEL_HPP
#define FINTERPMATRIXKERNEL_HPP

#include <stdexcept>

#include "FMath.hpp"


// probably not extendable :)
enum KERNEL_FUNCTION_TYPE {HOMOGENEOUS, NON_HOMOGENEOUS};


/**
 * @author Matthias Messner (matthias.messner@inria.fr)
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FInterpMatrixKernels
 * Please read the license
 *
 * This class provides the evaluators and scaling functions of the matrix
 * kernels. A matrix kernel should be understood in the sense of a kernel
 * of interaction (or the fundamental solution of a given equation).
 * It can either be scalar (NCMP=1) or tensorial (NCMP>1) depending on the
 * dimension of the equation considered. NCMP denotes the number of components
 * that are actually stored (e.g. 6 for a \f$3\times3\f$ symmetric tensor).
 * Notes on application scheme:
 * Let there be a kernel \f$K\f$ such that \f$X_i=K_{ij}Y_j\f$
 * with \f$X\f$ the lhs of size NLHS and \f$Y\f$ the rhs of size NRHS.
 * The table applyTab provides the indices in the reduced storage table
 * corresponding to the application scheme depicted earlier.
 *
 * PB: BEWARE! Homogeneous matrix kernels do not support cell width extension
 * yet. Is it possible to find a reference width and a scale factor such that
 * only 1 set of M2L ops can be used for all levels??
 *
 */
template <class FReal>
struct FInterpAbstractMatrixKernel
{ 
    FInterpAbstractMatrixKernel(){}
    FInterpAbstractMatrixKernel(const FInterpAbstractMatrixKernel&) = delete;
    FInterpAbstractMatrixKernel& operator=(const FInterpAbstractMatrixKernel&) = delete;

    virtual ~FInterpAbstractMatrixKernel(){} // to remove warning
    //virtual FReal evaluate(const std::array<FReal,Dim>&, const std::array<FReal,Dim>&) const = 0;
    // I need both functions because required arguments are not always given
    virtual FReal getScaleFactor(const FReal, const int) const = 0;
    virtual FReal getScaleFactor(const FReal) const = 0;
};


/// One over r
template <class FReal>
struct FInterpMatrixKernelR : FInterpAbstractMatrixKernel<FReal>
{
    static const int Dim = 3;

    static const KERNEL_FUNCTION_TYPE Type = HOMOGENEOUS;
    static const unsigned int NCMP = 1; //< number of components
    static const unsigned int NPV  = 1; //< dim of physical values
    static const unsigned int NPOT = 1; //< dim of potentials
    static const unsigned int NRHS = 1; //< dim of mult exp
    static const unsigned int NLHS = 1; //< dim of loc exp

    FInterpMatrixKernelR() {}

    // copy ctor
    FInterpMatrixKernelR(const FInterpMatrixKernelR& /*other*/)
    {}

  static const char* getID() { return "ONE_OVER_R"; }

    // returns position in reduced storage
    int getPosition(const unsigned int) const
    {return 0;}

    // evaluate interaction
    template <class ValueClass>
    ValueClass evaluate(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                   const ValueClass& x2, const ValueClass& y2, const ValueClass& z2) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        return FMath::One<ValueClass>() / FMath::Sqrt(diffx*diffx + diffy*diffy + diffz*diffz);
    }

    // evaluate interaction (blockwise)
    template <class ValueClass>
    void evaluateBlock(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                       const ValueClass& x2, const ValueClass& y2, const ValueClass& z2, ValueClass* block) const
    {
        block[0] = this->evaluate(x1,y1,z1,x2,y2,z2);
    }

    // evaluate interaction and derivative (blockwise)
    template <class ValueClass>
    void evaluateBlockAndDerivative(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                                    const ValueClass& x2, const ValueClass& y2, const ValueClass& z2,
                                    ValueClass block[1], ValueClass blockDerivative[3]) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        const ValueClass one_over_r = FMath::One<ValueClass>() / FMath::Sqrt(diffx*diffx + diffy*diffy + diffz*diffz);

        const ValueClass one_over_r3 = one_over_r*one_over_r*one_over_r;

        block[0] = one_over_r;

        blockDerivative[0] = one_over_r3 * diffx;
        blockDerivative[1] = one_over_r3 * diffy;
        blockDerivative[2] = one_over_r3 * diffz;
    }

    FReal getScaleFactor(const FReal RootCellWidth, const int TreeLevel) const
    {
        const FReal CellWidth(RootCellWidth / FReal(FMath::pow(2, TreeLevel)));
        return getScaleFactor(CellWidth);
    }

    FReal getScaleFactor(const FReal CellWidth) const
    {
        return FReal(2.) / CellWidth;
    }

    FReal evaluate(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2) const {
        return evaluate<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
    }
    void evaluateBlock(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2, FReal* block) const{
        evaluateBlock<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], block);
    }
    void evaluateBlockAndDerivative(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2,
                                    FReal block[1], FReal blockDerivative[3]) const {
        evaluateBlockAndDerivative<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], block, blockDerivative);
    }
};

/// One over r when the box size is rescaled to 1
template <class FReal>
struct FInterpMatrixKernelRH :FInterpMatrixKernelR<FReal>{
    static const int Dim = 3;

    static const KERNEL_FUNCTION_TYPE Type = HOMOGENEOUS;
    static const unsigned int NCMP = 1; //< number of components
    static const unsigned int NPV  = 1; //< dim of physical values
    static const unsigned int NPOT = 1; //< dim of potentials
    static const unsigned int NRHS = 1; //< dim of mult exp
    static const unsigned int NLHS = 1; //< dim of loc exp
    FReal LX,LY,LZ ;

    FInterpMatrixKernelRH() : LX(1.0),LY(1.0),LZ(1.0)
    {	 }

    // copy ctor
    FInterpMatrixKernelRH(const FInterpMatrixKernelRH& other)
        :FInterpMatrixKernelR<FReal>(other), LX(other.LX), LY(other.LY), LZ(other.LZ)
    {}

  static const char* getID() { return "ONE_OVER_RH"; }

    // evaluate interaction
    template <class ValueClass>
    ValueClass evaluate(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                   const ValueClass& x2, const ValueClass& y2, const ValueClass& z2) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        return FMath::One<ValueClass>() / FMath::Sqrt(FMath::ConvertTo<ValueClass,FReal>(LX)*diffx*diffx +
                                       FMath::ConvertTo<ValueClass,FReal>(LY)*diffy*diffy +
                                       FMath::ConvertTo<ValueClass,FReal>(LZ)*diffz*diffz);
    }
    void setCoeff(const FReal& a,  const FReal& b, const FReal& c)
    {LX= a*a ; LY = b*b ; LZ = c *c;}
    // returns position in reduced storage
    int getPosition(const unsigned int) const
    {return 0;}

    template <class ValueClass>
    void evaluateBlock(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                       const ValueClass& x2, const ValueClass& y2, const ValueClass& z2, ValueClass* block) const
    {
        block[0]=this->evaluate(x1,y1,z1,x2,y2,z2);
    }

    // evaluate interaction and derivative (blockwise)
    template <class ValueClass>
    void evaluateBlockAndDerivative(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                                    const ValueClass& x2, const ValueClass& y2, const ValueClass& z2,
                                    ValueClass block[1], ValueClass blockDerivative[3]) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        const ValueClass one_over_rL = FMath::One<ValueClass>() / FMath::Sqrt(FMath::ConvertTo<ValueClass,FReal>(LX)*diffx*diffx +
                                                          FMath::ConvertTo<ValueClass,FReal>(LY)*diffy*diffy +
                                                          FMath::ConvertTo<ValueClass,FReal>(LZ)*diffz*diffz);
        const ValueClass one_over_rL3 = one_over_rL*one_over_rL*one_over_rL;

        block[0] = one_over_rL;

        blockDerivative[0] = FMath::ConvertTo<ValueClass,FReal>(LX) * one_over_rL3 * diffx;
        blockDerivative[1] = FMath::ConvertTo<ValueClass,FReal>(LY)* one_over_rL3 * diffy;
        blockDerivative[2] = FMath::ConvertTo<ValueClass,FReal>(LZ)* one_over_rL3 * diffz;

    }

    FReal getScaleFactor(const FReal RootCellWidth, const int TreeLevel) const
    {
        const FReal CellWidth(RootCellWidth / FReal(FMath::pow(2, TreeLevel)));
        return getScaleFactor(CellWidth);
    }

    FReal getScaleFactor(const FReal CellWidth) const
    {
        return FReal(2.) / CellWidth;
    }

    FReal evaluate(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2) const{
        return evaluate<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
    }
    void evaluateBlock(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2, FReal* block) const{
        evaluateBlock<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], block);
    }
    void evaluateBlockAndDerivative(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2,
                                    FReal block[1], FReal blockDerivative[3]) const {
        evaluateBlockAndDerivative<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], block, blockDerivative);
    }
};


/// One over r^2
template <class FReal>
struct FInterpMatrixKernelRR : FInterpAbstractMatrixKernel<FReal>
{
    static const int Dim = 3;

    static const KERNEL_FUNCTION_TYPE Type = HOMOGENEOUS;
    static const unsigned int NCMP = 1; //< number of components
    static const unsigned int NPV  = 1; //< dim of physical values
    static const unsigned int NPOT = 1; //< dim of potentials
    static const unsigned int NRHS = 1; //< dim of mult exp
    static const unsigned int NLHS = 1; //< dim of loc exp

    FInterpMatrixKernelRR() {}

    // copy ctor
    FInterpMatrixKernelRR(const FInterpMatrixKernelRR& /*other*/)
    {}

  static const char* getID() { return "ONE_OVER_R_SQUARED"; }

    // returns position in reduced storage
    int getPosition(const unsigned int) const
    {return 0;}

    // evaluate interaction
    template <class ValueClass>
    ValueClass evaluate(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                   const ValueClass& x2, const ValueClass& y2, const ValueClass& z2) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        return FMath::One<ValueClass>() / FReal(diffx*diffx +
                                 diffy*diffy +
                                 diffz*diffz);
    }

    // evaluate interaction (blockwise)
    template <class ValueClass>
    void evaluateBlock(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                       const ValueClass& x2, const ValueClass& y2, const ValueClass& z2, ValueClass* block) const
    {
        block[0]=this->evaluate(x1,y1,z1,x2,y2,z2);
    }

    // evaluate interaction and derivative (blockwise)
    template <class ValueClass>
    void evaluateBlockAndDerivative(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                                    const ValueClass& x2, const ValueClass& y2, const ValueClass& z2,
                                    ValueClass block[1], ValueClass blockDerivative[3]) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        const ValueClass r2 = (diffx*diffx +
                               diffy*diffy +
                               diffz*diffz);
        const ValueClass one_over_r2 = FMath::One<ValueClass>() / (r2);
        const ValueClass one_over_r4 = one_over_r2*one_over_r2;

        block[0] = one_over_r2;

        const ValueClass coef = FMath::ConvertTo<ValueClass,FReal>(-2.) * one_over_r4;
        blockDerivative[0] = coef * diffx;
        blockDerivative[1] = coef * diffy;
        blockDerivative[2] = coef * diffz;

    }

    FReal getScaleFactor(const FReal RootCellWidth, const int TreeLevel) const
    {
        const FReal CellWidth(RootCellWidth / FReal(FMath::pow(2, TreeLevel)));
        return getScaleFactor(CellWidth);
    }

    FReal getScaleFactor(const FReal CellWidth) const
    {
        return FReal(4.) / (CellWidth*CellWidth);
    }

    FReal evaluate(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2) const{
        return evaluate<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
    }
    void evaluateBlock(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2, FReal* block) const{
        evaluateBlock<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], block);
    }
    void evaluateBlockAndDerivative(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2,
                                    FReal block[1], FReal blockDerivative[3]) const {
        evaluateBlockAndDerivative<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], block, blockDerivative);
    }
};



/// One over r^12 - One over r^6
template <class FReal>
struct FInterpMatrixKernelLJ : FInterpAbstractMatrixKernel<FReal>
{
    static const int Dim = 3;

    static const KERNEL_FUNCTION_TYPE Type = NON_HOMOGENEOUS;
    static const unsigned int NCMP = 1; //< number of components
    static const unsigned int NPV  = 1; //< dim of physical values
    static const unsigned int NPOT = 1; //< dim of potentials
    static const unsigned int NRHS = 1; //< dim of mult exp
    static const unsigned int NLHS = 1; //< dim of loc exp

    FInterpMatrixKernelLJ() {}

    // copy ctor
    FInterpMatrixKernelLJ(const FInterpMatrixKernelLJ& /*other*/)
    {}

  static const char* getID() { return "LENNARD_JONES_POTENTIAL"; }

    // returns position in reduced storage
    int getPosition(const unsigned int) const
    {return 0;}

    // evaluate interaction
    template <class ValueClass>
    ValueClass evaluate(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                   const ValueClass& x2, const ValueClass& y2, const ValueClass& z2) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        const ValueClass r = FMath::Sqrt(diffx*diffx +
                                   diffy*diffy +
                                   diffz*diffz);
        const ValueClass r3 = r*r*r;
        const ValueClass one_over_r6 = FMath::One<ValueClass>() / (r3*r3);
        //return one_over_r6 * one_over_r6;
        //return one_over_r6;
        return one_over_r6 * one_over_r6 - one_over_r6;
    }

    // evaluate interaction (blockwise)
    template <class ValueClass>
    void evaluateBlock(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                       const ValueClass& x2, const ValueClass& y2, const ValueClass& z2, ValueClass* block) const
    {
        block[0]=this->evaluate(x1,y1,z1,x2,y2,z2);
    }

    // evaluate interaction and derivative (blockwise)
    template <class ValueClass>
    void evaluateBlockAndDerivative(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                                    const ValueClass& x2, const ValueClass& y2, const ValueClass& z2,
                                    ValueClass block[1], ValueClass blockDerivative[3]) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        const ValueClass r = FMath::Sqrt(diffx*diffx +
                                    diffy*diffy +
                                    diffz*diffz);
        const ValueClass r2 = r*r;
        const ValueClass r3 = r2*r;
        const ValueClass one_over_r6 = FMath::One<ValueClass>() / (r3*r3);
        const ValueClass one_over_r8 = one_over_r6 / (r2);

        block[0] = one_over_r6 * one_over_r6 - one_over_r6;

        const FReal coef = FMath::ConvertTo<ValueClass,FReal>(12.0)*one_over_r6*one_over_r8 - FMath::ConvertTo<ValueClass,FReal>(6.0)*one_over_r8;
        blockDerivative[0]= coef * diffx;
        blockDerivative[1]= coef * diffy;
        blockDerivative[2]= coef * diffz;

    }

    FReal getScaleFactor(const FReal, const int) const
    {
        // return 1 because non homogeneous kernel functions cannot be scaled!!!
        return FReal(1.0);
    }

    FReal getScaleFactor(const FReal) const
    {
        // return 1 because non homogeneous kernel functions cannot be scaled!!!
        return FReal(1.0);
    }



    FReal evaluate(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2) const{
        return evaluate<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
    }
    void evaluateBlock(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2, FReal* block) const{
        evaluateBlock<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], block);
    }
    void evaluateBlockAndDerivative(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2,
                                    FReal block[1], FReal blockDerivative[3]) const {
        evaluateBlockAndDerivative<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], block, blockDerivative);
    }
};


/// One over (a+r^2)
template <class FReal>
struct FInterpMatrixKernelAPLUSRR : FInterpAbstractMatrixKernel<FReal>
{
    static const int Dim = 3;

    static const KERNEL_FUNCTION_TYPE Type = NON_HOMOGENEOUS;
    static const unsigned int NCMP = 1; //< number of components
    static const unsigned int NPV  = 1; //< dim of physical values
    static const unsigned int NPOT = 1; //< dim of potentials
    static const unsigned int NRHS = 1; //< dim of mult exp
    static const unsigned int NLHS = 1; //< dim of loc exp

    const FReal CoreWidth;

    FInterpMatrixKernelAPLUSRR(const FReal inCoreWidth = .25)
    : CoreWidth(inCoreWidth)
     {}

    // copy ctor
    FInterpMatrixKernelAPLUSRR(const FInterpMatrixKernelAPLUSRR& other)
    : CoreWidth(other.CoreWidth)
    {}

  static const char* getID() { return "ONE_OVER_A_PLUS_RR"; }

    // returns position in reduced storage
    int getPosition(const unsigned int) const
    {return 0;}

    // evaluate interaction
    template <class ValueClass>
    ValueClass evaluate(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                   const ValueClass& x2, const ValueClass& y2, const ValueClass& z2) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        return FMath::One<ValueClass>() / (FMath::ConvertTo<ValueClass,FReal>(CoreWidth) + // WHY FReal??
                                           diffx*diffx +
                                           diffy*diffy +
                                           diffz*diffz);
    }

    // evaluate interaction (blockwise)
    template <class ValueClass>
    void evaluateBlock(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                       const ValueClass& x2, const ValueClass& y2, const ValueClass& z2, ValueClass* block) const
    {
        block[0]=this->evaluate(x1,y1,z1,x2,y2,z2);
    }

    // evaluate interaction and derivative (blockwise)
    template <class ValueClass>
    void evaluateBlockAndDerivative(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                                    const ValueClass& x2, const ValueClass& y2, const ValueClass& z2,
                                    ValueClass block[1], ValueClass blockDerivative[3]) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        const ValueClass r2 = (diffx*diffx +
                               diffy*diffy +
                               diffz*diffz);
        const ValueClass one_over_a_plus_r2 = FMath::One<ValueClass>() / (FMath::ConvertTo<ValueClass,FReal>(CoreWidth)+r2);
        const ValueClass one_over_a_plus_r2_squared = one_over_a_plus_r2*one_over_a_plus_r2;

        block[0] = one_over_a_plus_r2;

        // TODO Fix derivative
        const ValueClass coef = FMath::ConvertTo<ValueClass,FReal>(-2.) * one_over_a_plus_r2_squared;
        blockDerivative[0] = coef * diffx;
        blockDerivative[1] = coef * diffy;
        blockDerivative[2] = coef * diffz;

    }

    FReal getScaleFactor(const FReal, const int) const
    {
        // return 1 because non homogeneous kernel functions cannot be scaled!!!
        return FReal(1.0);
    }

    FReal getScaleFactor(const FReal) const
    {
        // return 1 because non homogeneous kernel functions cannot be scaled!!!
        return FReal(1.0);    
    }

    FReal evaluate(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2) const{
        return evaluate<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
    }
    void evaluateBlock(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2, FReal* block) const{
        evaluateBlock<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], block);
    }
    void evaluateBlockAndDerivative(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2,
                                    FReal block[1], FReal blockDerivative[3]) const {
        evaluateBlockAndDerivative<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], block, blockDerivative);
    }
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// Tensorial Matrix Kernels (NCMP>1)
//
// The definition of the potential p and force f are extended to the case
// of tensorial interaction kernels:
// p_i(x) = K_{ip}(x,y)w_p(y), \forall i=1..NPOT, p=1..NPV
// f_{ik}= w_p(x)K_{ip,k}(x,y)w_p(y) "
//
// Since the interpolation scheme is such that
// p_i(x) \approx S^m(x) L^{m}_{ip}
// f_{ik}= w_p(x) \nabla_k S^m(x) L^{m}_{ip}
// with
// L^{m}_{ip} = K^{mn}_{ip} S^n(y) w_p(y) (local expansion)
// M^{m}_{p} = S^n(y) w_p(y) (multipole expansion)
// then the multipole exp have NPV components and the local exp NPOT*NPV.
//
// NB1: Only the computation of forces requires that the sum over p is 
// performed at L2P step. It could be done at M2L step for the potential.
//
// NB2: An efficient application of the matrix kernel is highly kernel 
// dependent, we recommand overriding the P2M/M2L/L2P function of the kernel 
// you are using in order to have optimal performances + set your own NRHS/NLHS.
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/// R_{,ij}
// PB: IMPORTANT! This matrix kernel does not present the symmetries 
// required by ChebSym kernel => only suited for Unif interpolation
template <class FReal>
struct FInterpMatrixKernel_R_IJ : FInterpAbstractMatrixKernel<FReal>
{
    static const int Dim = 3;

    static const KERNEL_FUNCTION_TYPE Type = NON_HOMOGENEOUS;
    static const unsigned int NK   = 3*3; //< total number of components
    static const unsigned int NCMP = 6;   //< number of components
    static const unsigned int NPV  = 3;   //< dim of physical values
    static const unsigned int NPOT = 3;   //< dim of potentials
    static const unsigned int NRHS = NPV; //< dim of mult exp
    static const unsigned int NLHS = NPOT*NRHS; //< dim of loc exp

    // store indices (i,j) corresponding to sym idx
    static const unsigned int indexTab[/*2*NCMP=12*/];

    // store positions in sym tensor (when looping over NRHSxNLHS)
    static const unsigned int applyTab[/*NK=9*/];

    // indices to be set at construction if component-wise evaluation is performed
    const unsigned int _i,_j;

    // Material Parameters
    const FReal _CoreWidth2; // if >0 then kernel is NON homogeneous

    FInterpMatrixKernel_R_IJ(const FReal CoreWidth = 0.0, const unsigned int d = 0)
        : _i(indexTab[d]), _j(indexTab[d+NCMP]), _CoreWidth2(CoreWidth*CoreWidth)
    {}

    // copy ctor
    FInterpMatrixKernel_R_IJ(const FInterpMatrixKernel_R_IJ& other)
        : _i(other._i), _j(other._j), _CoreWidth2(other._CoreWidth2)
    {}

  static const char* getID() { return "R_IJ"; }

    // returns position in reduced storage from position in full 3x3 matrix
    unsigned  int getPosition(const unsigned int n) const
    {return applyTab[n];}

    // returns Core Width squared
    FReal getCoreWidth2() const
    {return _CoreWidth2;}

    // evaluate interaction
    template <class ValueClass>
    FReal evaluate(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                   const ValueClass& x2, const ValueClass& y2, const ValueClass& z2) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        const ValueClass one_over_r = FMath::One<ValueClass>()/FMath::Sqrt(diffx*diffx +
                                                diffy*diffy +
                                                diffz*diffz + FMath::ConvertTo<ValueClass,FReal>(_CoreWidth2));
        const ValueClass one_over_r3 = one_over_r*one_over_r*one_over_r;
        ValueClass ri,rj;

        if(_i==0) ri=diffx;
        else if(_i==1) ri=diffy;
        else if(_i==2) ri=diffz;
        else throw std::runtime_error("Update i!");

        if(_j==0) rj=diffx;
        else if(_j==1) rj=diffy;
        else if(_j==2) rj=diffz;
        else throw std::runtime_error("Update j!");

        if(_i==_j)
            return one_over_r - ri * ri * one_over_r3;
        else
            return - ri * rj * one_over_r3;

    }

    // evaluate interaction (blockwise)
    template <class ValueClass>
    void evaluateBlock(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                       const ValueClass& x2, const ValueClass& y2, const ValueClass& z2, ValueClass* block) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        const ValueClass one_over_r = FMath::One<ValueClass>()/FMath::Sqrt(diffx*diffx +
                                                diffy*diffy +
                                                diffz*diffz + FMath::ConvertTo<ValueClass,FReal>(_CoreWidth2));
        const ValueClass one_over_r3 = one_over_r*one_over_r*one_over_r;
        const ValueClass r[3] = {diffx,diffy,diffz};

        for(unsigned int d=0;d<NCMP;++d){
            unsigned int i = indexTab[d];
            unsigned int j = indexTab[d+NCMP];

            if(i==j)
                block[d] = one_over_r - r[i] * r[i] * one_over_r3;
            else
                block[d] = - r[i] * r[j] * one_over_r3;
        }
    }

    // evaluate interaction and derivative (blockwise)
    template <class ValueClass>
    void evaluateBlockAndDerivative(const ValueClass& x1, const ValueClass& y1, const ValueClass& z1,
                                    const ValueClass& x2, const ValueClass& y2, const ValueClass& z2,
                                    ValueClass block[NCMP], ValueClass blockDerivative[NCMP][3]) const
    {
        const ValueClass diffx = (x1-x2);
        const ValueClass diffy = (y1-y2);
        const ValueClass diffz = (z1-z2);
        const ValueClass r2[3] = {diffx*diffx,diffy*diffy,diffz*diffz};
        const ValueClass one_over_r2 = FMath::One<ValueClass>() / (r2[0] + r2[1] + r2[2]);
        const ValueClass one_over_r = FMath::Sqrt(one_over_r2);
        const ValueClass one_over_r3 = one_over_r2*one_over_r;

        const ValueClass r[3] = {diffx,diffy,diffz};

        const ValueClass Three = FMath::ConvertTo<ValueClass,FReal>(3.);
        const ValueClass MinusOne = -FMath::One<ValueClass>();

        for(unsigned int d=0;d<NCMP;++d){
            unsigned int i = indexTab[d];
            unsigned int j = indexTab[d+NCMP];

            // evaluate kernel
            if(i==j)
                block[d] = one_over_r - r2[i] * one_over_r3;
            else
                block[d] = - r[i] * r[j] * one_over_r3;

            // evaluate derivative
            for(unsigned int k = 0 ; k < 3 ; ++k){
              if(i==j){
                if(j==k) //i=j=k
                  blockDerivative[d][k] = Three * ( MinusOne + r2[i] * one_over_r2 ) * r[i] * one_over_r3;
                else //i=j!=k
                  blockDerivative[d][k] = ( MinusOne + Three * r2[i] * one_over_r2 ) * r[k] * one_over_r3;
              }
              else{ //(i!=j)
                if(i==k) //i=k!=j
                  blockDerivative[d][k] = ( MinusOne + Three * r2[i] * one_over_r2 ) * r[j] * one_over_r3;
                else if(j==k) //i!=k=j
                  blockDerivative[d][k] = ( MinusOne + Three * r2[j] * one_over_r2 ) * r[i] * one_over_r3;
                else //i!=k!=j
                  blockDerivative[d][k] = Three * r[i] * r[j] * r[k] * one_over_r2 * one_over_r3;
              }
            }// k

        }// NCMP
    }

    FReal getScaleFactor(const FReal RootCellWidth, const int TreeLevel) const
    {
        const FReal CellWidth(RootCellWidth / FReal(FMath::pow(2, TreeLevel)));
        return getScaleFactor(CellWidth);
    }

    // R_{,ij} is homogeneous to [L]/[L]^{-2}=[L]^{-1}
    // => scales like ONE_OVER_R
    FReal getScaleFactor(const FReal CellWidth) const
    {
        return FReal(2.) / CellWidth;
    }



    FReal evaluate(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2) const{
        return evaluate<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
    }
    void evaluateBlock(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2, FReal* block) const{
        evaluateBlock<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], block);
    }
    void evaluateBlockAndDerivative(const std::array<FReal,Dim>& p1, const std::array<FReal,Dim>& p2,
                                    FReal block[NCMP], FReal blockDerivative[NCMP][3]) const {
        evaluateBlockAndDerivative<FReal>(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], block, blockDerivative);
    }
};

/// R_IJ
template <class FReal>
const unsigned int FInterpMatrixKernel_R_IJ<FReal>::indexTab[]={0,0,0,1,1,2,
                                                         0,1,2,1,2,2};

template <class FReal>
const unsigned int FInterpMatrixKernel_R_IJ<FReal>::applyTab[]={0,1,2,
                                                         1,3,4,
                                                         2,4,5};



/*!  Functor which provides the interface to assemble a matrix based on the
  number of rows and cols and on the coordinates x and y and the type of the
  generating matrix-kernel function.
*/
template <class FReal, typename MatrixKernelClass>
class EntryComputer
{
    static const int Dim = 3;

    const MatrixKernelClass *const MatrixKernel;

    const unsigned int nx, ny;
    const std::array<FReal,Dim> *const px, *const py;

    const FReal *const weights;

public:
    explicit EntryComputer(const MatrixKernelClass *const inMatrixKernel,
                           const unsigned int _nx, const std::array<FReal,Dim> *const _px,
                           const unsigned int _ny, const std::array<FReal,Dim> *const _py,
                           const FReal *const _weights = NULL)
        : MatrixKernel(inMatrixKernel),	nx(_nx), ny(_ny), px(_px), py(_py), weights(_weights) {}

    void operator()(const unsigned int xbeg, const unsigned int xend,
                    const unsigned int ybeg, const unsigned int yend,
                    FReal *const data) const
    {
        unsigned int idx = 0;
        if (weights) {
            for (unsigned int j=ybeg; j<yend; ++j)
                for (unsigned int i=xbeg; i<xend; ++i)
                    data[idx++] = weights[i] * weights[j] * MatrixKernel->evaluate(px[i], py[j]);
        } else {
            for (unsigned int j=ybeg; j<yend; ++j)
                for (unsigned int i=xbeg; i<xend; ++i)
                    data[idx++] = MatrixKernel->evaluate(px[i], py[j]);
        }

        /*
    // apply weighting matrices
    if (weights) {
    if ((xend-xbeg) == (yend-ybeg) && (xend-xbeg) == nx)
    for (unsigned int n=0; n<nx; ++n) {
    FBlas::scal(nx, weights[n], data + n,  nx); // scale rows
    FBlas::scal(nx, weights[n], data + n * nx); // scale cols
    }
    else if ((xend-xbeg) == 1 && (yend-ybeg) == ny)
    for (unsigned int j=0; j<ny; ++j)	data[j] *= weights[j];
    else if ((yend-ybeg) == 1 && (xend-xbeg) == nx)
    for (unsigned int i=0; i<nx; ++i)	data[i] *= weights[i];
    }
    */

    }
};





#endif // FINTERPMATRIXKERNEL_HPP

// [--END--]
