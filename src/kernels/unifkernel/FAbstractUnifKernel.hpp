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
// Keep in private GIT
// @SCALFMM_PRIVATE

#ifndef FABSTRACTUNIFKERNEL_HPP
#define FABSTRACTUNIFKERNEL_HPP

#include <memory>

#include "FInterpP2PKernels.hpp"
#include "FUnifInterpolator.hpp"

/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FAbstractUnifKernel
 * @brief
 * This kernels implement the Lagrange interpolation based FMM operators. It
 * implements all interfaces (P2P, P2M, M2M, M2L, L2L, L2P) which are required by
 * the FFmmAlgorithm and FFmmAlgorithmThread.
 *
 * @tparam CellClass Type of cell
 * @tparam ContainerClass Type of container to store particles
 * @tparam MatrixKernelClass Type of matrix kernel function
 * @tparam ORDER Lagrange interpolation order
 */
template < class FReal,	class MatrixKernelClass, int ORDER, int Dim = 3, int NVALS = 1>
class FAbstractUnifKernel
{
protected:
  enum {nnodes = TensorTraits<ORDER>::nnodes};
  typedef FUnifInterpolator<FReal, ORDER,MatrixKernelClass,NVALS> InterpolatorClass;

  /// Needed for P2M, M2M, L2L and L2P operators
  const std::shared_ptr<InterpolatorClass> Interpolator;
  /// Height of the entire oct-tree
  const unsigned int TreeHeight;
  /// Corner of oct-tree box
  const std::array<FReal, Dim> BoxCorner;
  /// Width of oct-tree box
  const FReal BoxWidth;
  /// Width of a leaf cell box
  const FReal BoxWidthLeaf;
  /// Extension of the box width ( same for all level! )
  const FReal BoxWidthExtension;

  /**
   * Compute center of leaf cell from its tree coordinate.
   * @param[in] Coordinate tree coordinate
   * @return center of leaf cell
   */
  std::array<FReal, Dim> getLeafCellCenter(const std::array<long int, Dim>& Coordinate) const
  {
      if constexpr (Dim == 3){
          return std::array<FReal, Dim>{BoxCorner[0] + (FReal(Coordinate[0]) + FReal(.5)) * BoxWidthLeaf,
                        BoxCorner[1] + (FReal(Coordinate[1]) + FReal(.5)) * BoxWidthLeaf,
                        BoxCorner[2] + (FReal(Coordinate[2]) + FReal(.5)) * BoxWidthLeaf};
      }
      else{
          std::array<FReal, Dim> res;
          for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
              res[idxDim] = BoxCorner[idxDim] + (FReal(Coordinate[idxDim]) + FReal(.5)) * BoxWidthLeaf;
          }
          return res;
      }
  }

  /** 
   * @brief Return the position of the center of a cell from its tree
   *  coordinate 
   * @param std::array<long int, Dim>
   * @param inLevel the current level of Cell
   */
  std::array<FReal, Dim> getCellCenter(const std::array<long int, Dim> coordinate, int inLevel)
  {

    //Set the boxes width needed
    FReal widthAtCurrentLevel = BoxWidthLeaf*FReal(1 << (TreeHeight-(inLevel+1)));   
    FReal widthAtCurrentLevelDiv2 = widthAtCurrentLevel/FReal(2.);

    //Set the center real coordinates from box corner and widths.
    FReal X = BoxCorner[0] + FReal(coordinate[0])*widthAtCurrentLevel + widthAtCurrentLevelDiv2;
    FReal Y = BoxCorner[1] + FReal(coordinate[1])*widthAtCurrentLevel + widthAtCurrentLevelDiv2;
    FReal Z = BoxCorner[2] + FReal(coordinate[2])*widthAtCurrentLevel + widthAtCurrentLevelDiv2;
    
    return std::array<FReal, Dim>(X,Y,Z);
  }

public:
  /**
   * The constructor initializes all constant attributes and it reads the
   * precomputed and compressed M2L operators from a binary file (an
   * runtime_error is thrown if the required file is not valid).
   */
  FAbstractUnifKernel(const int inTreeHeight,
                      const FReal inBoxWidth,
                      const std::array<FReal, Dim>& inBoxCenter,
                      const FReal inBoxWidthExtension = 0.0)
    : Interpolator(new InterpolatorClass(inTreeHeight,
                                         inBoxWidth,
                                         inBoxWidthExtension)),
      TreeHeight(inTreeHeight),
      BoxCorner(inBoxCenter - inBoxWidth / FReal(2.)),
      BoxWidth(inBoxWidth),
      BoxWidthLeaf(BoxWidth / FReal(FMath::pow(2, inTreeHeight - 1))),
      BoxWidthExtension(inBoxWidthExtension)
  {
    /* empty */
  }

  virtual ~FAbstractUnifKernel(){
    // should not be used
  }

  const InterpolatorClass * getPtrToInterpolator() const
  { return Interpolator.get(); }
};





#endif //FABSTRACTUNIFKERNEL_HPP

// [--END--]
