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
template < class RealType_T, class MatrixKernelClass, int ORDER,
           int Dim = 3,  class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class FAbstractUnifKernel
{
public:
    using RealType = RealType_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

protected:
  enum {nnodes = TensorTraits<ORDER>::nnodes};
  typedef FUnifInterpolator<RealType, ORDER,MatrixKernelClass> InterpolatorClass;

  /// Needed for P2M, M2M, L2L and L2P operators
  const std::shared_ptr<InterpolatorClass> Interpolator;

  const SpaceIndexType spaceIndexSystem;

  /// Height of the entire oct-tree
  const unsigned int TreeHeight;
  /// Corner of oct-tree box
  const std::array<RealType, Dim> BoxCorner;
  /// Width of oct-tree box
  const RealType BoxWidth;
  /// Width of a leaf cell box
  const RealType BoxWidthLeaf;
  /// Extension of the box width ( same for all level! )
  const RealType BoxWidthExtension;

  /**
   * Compute center of leaf cell from its tree coordinate.
   * @param[in] Coordinate tree coordinate
   * @return center of leaf cell
   */
  std::array<RealType, Dim> getLeafCellCenter(const std::array<long int, Dim>& Coordinate) const
  {
      if constexpr (Dim == 3){
          return std::array<RealType, Dim>{BoxCorner[0] + (RealType(Coordinate[0]) + RealType(.5)) * BoxWidthLeaf,
                        BoxCorner[1] + (RealType(Coordinate[1]) + RealType(.5)) * BoxWidthLeaf,
                        BoxCorner[2] + (RealType(Coordinate[2]) + RealType(.5)) * BoxWidthLeaf};
      }
      else{
          std::array<RealType, Dim> res;
          for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
              res[idxDim] = BoxCorner[idxDim] + (RealType(Coordinate[idxDim]) + RealType(.5)) * BoxWidthLeaf;
          }
          return res;
      }
  }

  std::array<RealType, Dim> getLeafCellCenter(const typename SpaceIndexType::IndexType& inIndex) const{
      return getLeafCellCenter(spaceIndexSystem.getBoxPosFromIndex(inIndex));
  }

  /** 
   * @brief Return the position of the center of a cell from its tree
   *  coordinate 
   * @param std::array<long int, Dim>
   * @param inLevel the current level of Cell
   */
  std::array<RealType, Dim> getCellCenter(const std::array<long int, Dim> coordinate, int inLevel)
  {

    //Set the boxes width needed
    RealType widthAtCurrentLevel = BoxWidthLeaf*RealType(1 << (TreeHeight-(inLevel+1)));
    RealType widthAtCurrentLevelDiv2 = widthAtCurrentLevel/RealType(2.);

    //Set the center real coordinates from box corner and widths.
    RealType X = BoxCorner[0] + RealType(coordinate[0])*widthAtCurrentLevel + widthAtCurrentLevelDiv2;
    RealType Y = BoxCorner[1] + RealType(coordinate[1])*widthAtCurrentLevel + widthAtCurrentLevelDiv2;
    RealType Z = BoxCorner[2] + RealType(coordinate[2])*widthAtCurrentLevel + widthAtCurrentLevelDiv2;
    
    return std::array<RealType, Dim>(X,Y,Z);
  }

  static auto BoxCornerFromCenterAndWidth(const std::array<RealType, Dim>& inCenter, const RealType width){
      std::array<RealType, Dim> corner;
      for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
          corner[idxDim] = inCenter[idxDim] - width/2;
      }
      return corner;
  }

public:
  /**
   * The constructor initializes all constant attributes and it reads the
   * precomputed and compressed M2L operators from a binary file (an
   * runtime_error is thrown if the required file is not valid).
   */
  FAbstractUnifKernel(const SpacialConfiguration& inConfiguration,
                      const RealType inBoxWidthExtension = 0.0)
    : Interpolator(new InterpolatorClass(int(inConfiguration.getTreeHeight()),
                                         inConfiguration.getBoxWidths()[0],
                                         inBoxWidthExtension)),
      spaceIndexSystem(inConfiguration),
      TreeHeight(int(inConfiguration.getTreeHeight())),
      BoxCorner(BoxCornerFromCenterAndWidth(inConfiguration.getBoxCenter(), inConfiguration.getBoxWidths()[0])),
      BoxWidth(inConfiguration.getBoxWidths()[0]),
      BoxWidthLeaf(inConfiguration.getBoxWidths()[0] / RealType(FMath::pow(2, int(inConfiguration.getTreeHeight()) - 1))),
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
