// ===================================================================================
// Copyright ScalFmm 2011 INRIA, Olivier Coulaud, Bérenger Bramas, Matthias Messner
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
#ifndef FINTERPTENSOR_HPP
#define FINTERPTENSOR_HPP

#include "./FInterpMapping.hpp"


/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * Please read the license
 */

/**
 * @class TensorTraits
 *
 * The class @p TensorTraits gives the number of interpolation nodes per
 * cluster in 3D, depending on the interpolation order.
 *
 * @tparam ORDER interpolation order
 */
template <int ORDER> struct TensorTraits
{
	enum {nnodes = ORDER*ORDER*ORDER};
};


/**
 * @class FInterpTensor
 *
 * The class FInterpTensor provides function considering the tensor product
 * interpolation.
 *
 * @tparam ORDER interpolation order \f$\ell\f$
 * @tparam RootsClass class containing the roots choosen for the interpolation 
 * (e.g. FChebRoots, FUnifRoots...)

 */
template <class FReal, int ORDER, typename RootsClass>
class FInterpTensor
{
  static const int Dim = 3;
  enum {nnodes = TensorTraits<ORDER>::nnodes};
  typedef RootsClass BasisType;

public:
  FInterpTensor(const FInterpTensor&) = delete;
  FInterpTensor& operator=(const FInterpTensor&) = delete;

  /**
   * Sets the ids of the coordinates of all \f$\ell^3\f$ interpolation
   * nodes
   *
   * @param[out] NodeIds ids of coordinates of interpolation nodes
   */
  static
  void setNodeIds(unsigned int NodeIds[nnodes][3])
  {
    for (unsigned int n=0; n<nnodes; ++n) {
      NodeIds[n][0] =  n         % ORDER;
      NodeIds[n][1] = (n/ ORDER) % ORDER;
      NodeIds[n][2] =  n/(ORDER  * ORDER);
    }
  }


  /**
   * Sets the interpolation points in the cluster with @p center and @p width
   *
   * PB: tensorial version
   *
   * @param[in] center of cluster
   * @param[in] width of cluster
   * @param[out] rootPositions coordinates of interpolation points
   */
  static
  void setRoots(const std::array<FReal, Dim>& center, const FReal width, std::array<FReal, Dim> rootPositions[nnodes])
  {
    unsigned int node_ids[nnodes][3];
    setNodeIds(node_ids);
    const map_loc_glob<FReal> map(center, width);
    std::array<FReal, Dim> localPosition;
    for (unsigned int n=0; n<nnodes; ++n) {
      localPosition[0] = (FReal(BasisType::roots[node_ids[n][0]]));
      localPosition[1] = (FReal(BasisType::roots[node_ids[n][1]]));
      localPosition[2] = (FReal(BasisType::roots[node_ids[n][2]]));
      map(localPosition, rootPositions[n]);
    }
  }

  /**
   * Sets the equispaced roots in the cluster with @p center and @p width
   *
   * @param[in] center of cluster
   * @param[in] width of cluster
   * @param[out] roots coordinates of equispaced roots
   */
  static
  void setPolynomialsRoots(const std::array<FReal, Dim>& center, const FReal width, FReal roots[3][ORDER])
  {
    const map_loc_glob<FReal> map(center, width);
    std::array<FReal, Dim> lPos, gPos;
    for (unsigned int n=0; n<ORDER; ++n) {
      lPos[0] = (FReal(BasisType::roots[n]));
      lPos[1] = (FReal(BasisType::roots[n]));
      lPos[2] = (FReal(BasisType::roots[n]));
      map(lPos, gPos);
      roots[0][n] = gPos[0];
      roots[1][n] = gPos[1];
      roots[2][n] = gPos[2];
    }
  }

  /**
   * Set the relative child (width = 1) center according to the Morton index.
   *
   * @param[in] ChildIndex index of child according to Morton index
   * @param[out] center
   * @param[in] ExtendedCellRatio ratio between extended child and parent widths
   */
  static
  void setRelativeChildCenter(const unsigned int ChildIndex,
                              std::array<FReal, Dim>& ChildCenter,
                              const FReal ExtendedCellRatio=FReal(.5))
  {
    const int RelativeChildPositions[][3] = { {-1, -1, -1},
                                              {-1, -1,  1},
                                              {-1,  1, -1},
                                              {-1,  1,  1},
                                              { 1, -1, -1},
                                              { 1, -1,  1},
                                              { 1,  1, -1},
                                              { 1,  1,  1} };
    // Translate center if cell widths are extended
    const FReal frac = (FReal(1.) - ExtendedCellRatio); 

    ChildCenter[0] = (FReal(RelativeChildPositions[ChildIndex][0]) * frac);
    ChildCenter[1] = (FReal(RelativeChildPositions[ChildIndex][1]) * frac);
    ChildCenter[2] = (FReal(RelativeChildPositions[ChildIndex][2]) * frac);
  }
};





#endif /*FINTERPTENSOR_HPP*/
