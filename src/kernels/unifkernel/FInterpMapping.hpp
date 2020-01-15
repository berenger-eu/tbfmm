// ===================================================================================
// Copyright ScalFmm 2011 INRIA
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
#ifndef FINTERPMAPPING_HPP
#define FINTERPMAPPING_HPP

#include <limits>

/**
 * @author Matthias Messner (matthias.matthias@inria.fr)
 * Please read the license
 */

/**
 * @class FInterpMapping
 *
 * The class @p FInterpMapping is the base class for the affine mapping
 * \f$\Phi:[-1,1]\rightarrow[a,b] \f$ and the inverse affine mapping
 * \f$\Phi^{-1}:[a,b]\rightarrow[-1,1]\f$.
 */
template <class FReal, int Dim = 3>
class FInterpMapping
{
protected:
    std::array<FReal, Dim> a;
    std::array<FReal, Dim> b;

    explicit FInterpMapping(const std::array<FReal, Dim>& center,
                            const FReal width){
        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            a[idxDim] = center[idxDim] - width / FReal(2.);
            b[idxDim] = center[idxDim] + width / FReal(2.);
        }
    }

    virtual void operator()(const std::array<FReal, Dim>&, std::array<FReal, Dim>&) const = 0;

public:
    FInterpMapping(const FInterpMapping&) = delete;
    FInterpMapping& operator=(const FInterpMapping&) = delete;
    virtual ~FInterpMapping(){}

    /**
     * Checks wheter @p position is within cluster, ie within @p a and @p b, or
     * not.
     *
     * @param[in] position position (eg of a particle)
     * @return @p true if position is in cluster else @p false
     */
    bool is_in_cluster(const std::array<FReal, Dim>& position) const
    {
        // Set numerical limit
        const FReal epsilon = FReal(10.) * std::numeric_limits<FReal>::epsilon();

        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            if (a[idxDim]-position[idxDim]>epsilon ||	position[idxDim]-b[idxDim]>epsilon) {
                std::cout << a[idxDim]-position[idxDim] << "\t"
                          << position[idxDim]-b[idxDim] << "\t"	<< epsilon << std::endl;
                return false;
            }
        }
        // Position is in cluster, return true
        return true;
    }

};


/**
 * @class map_glob_loc
 *
 * This class defines the inverse affine mapping
 * \f$\Phi^{-1}:[a,b]\rightarrow[-1,1]\f$. It maps from global coordinates to
 * local ones.
 */
template <class FReal, int Dim = 3>
class map_glob_loc : public FInterpMapping<FReal>
{
    using FInterpMapping<FReal>::a;
    using FInterpMapping<FReal>::b;
public:
    explicit map_glob_loc(const std::array<FReal, Dim>& center, const FReal width)
        : FInterpMapping<FReal>(center, width) {}

    /**
     * Maps from a global position to its local position: \f$\Phi^{-1}(x) =
     * \frac{2x-b-a}{b-a}\f$.
     *
     * @param[in] globPos global position
     * @param[out] loclPos local position
     */
    void operator()(const std::array<FReal, Dim>& globPos, std::array<FReal, Dim>& loclPos) const
    {
        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            loclPos[idxDim] = ((FReal(2.)*globPos[idxDim]-b[idxDim]-a[idxDim]) / (b[idxDim]-a[idxDim])); // 5 flops
        }
    }

    // jacobian = 2 / (b - a);
    void computeJacobian(std::array<FReal, Dim>& jacobian) const
    {
        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            jacobian[idxDim] = (FReal(2.) / (b[idxDim] - a[idxDim])); // 2 flops
        }
    }
};


/**
 * @class map_loc_glob
 *
 * This class defines the affine mapping \f$\Phi:[-1,1]\rightarrow[a,b]\f$. It
 * maps from local coordinates to global ones.
 */
template <class FReal, int Dim = 3>
class map_loc_glob : public FInterpMapping<FReal>
{
    using FInterpMapping<FReal>::a;
    using FInterpMapping<FReal>::b;
public:
    explicit map_loc_glob(const std::array<FReal, Dim>& center, const FReal width)
        : FInterpMapping<FReal>(center, width) {}

    // globPos = (a + b) / 2 + (b - a) * loclPos / 2;
    /**
     * Maps from a local position to its global position: \f$\Phi(\xi) =
     * \frac{1}{2}(a+b)+ \frac{1}{2}(b-a)\xi\f$.
     *
     * @param[in] loclPos local position
     * @param[out] globPos global position
     */
    void operator()(const std::array<FReal, Dim>& loclPos, std::array<FReal, Dim>& globPos) const
    {
        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            globPos[idxDim] = ((a[idxDim]+b[idxDim])/FReal(2.)+
                     (b[idxDim]-a[idxDim])*loclPos[idxDim]/FReal(2.));
        }
    }
};



#endif /*FUNIFTENSOR_HPP*/
