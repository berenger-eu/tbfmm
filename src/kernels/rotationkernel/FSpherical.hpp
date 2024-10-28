// See LICENCE file at project root
#ifndef FSPHERICAL_HPP
#define FSPHERICAL_HPP

#include <cmath>

/**
* This class is a Spherical position
*
* @brief Spherical coordinate system
*
* We consider the spherical coordinate system \f$(r, \theta, \varphi)\f$ commonly used in physics. r is the radial distance, \f$\theta\f$ the polar/inclination angle and \f$\varphi\f$ the azimuthal angle.<br>
* The <b>radial distance</b> is the Euclidean distance from the origin O to P.<br>
* The <b>inclination (or polar angle) </b>is the angle between the zenith direction and the line segment OP.<br>
* The <b>azimuth (or azimuthal angle)</b> is the signed angle measured from the azimuth reference direction to the orthogonal projection of the line segment OP on the reference plane.<br>
*
* The spherical coordinates of a point can be obtained from its Cartesian coordinates (x, y, z) by the formulae
* \f$ \displaystyle r = \sqrt{x^2 + y^2 + z^2}\f$<br>
* \f$ \displaystyle \theta = \displaystyle\arccos\left(\frac{z}{r}\right) \f$<br>
* \f$ \displaystyle \varphi = \displaystyle\arctan\left(\frac{y}{x}\right) \f$<br>
*and \f$\varphi\in[0,2\pi[ \f$ \f$ \theta\in[0,\pi]\f$<br>
*
* The spherical coordinate system  is retrieved from the the spherical coordinates by <br>
* \f$x = r \sin(\theta) \cos(\varphi)\f$ <br>
* \f$y = r \sin(\theta) \sin(\varphi)\f$ <br>
* \f$z = r \cos(\theta) \f$<br>
* with  \f$\varphi\in[-\pi,\pi[ \f$ \f$ \theta\in[0,\pi]\f$<br>
*
* This system is defined in p 872 of the paper of Epton and Dembart, SIAM J Sci Comput 1995.<br>
*
* Even if it can look different from usual expression (where theta and phi are inversed),
* such expression is used to match the SH expression.
*  See http://en.wikipedia.org/wiki/Spherical_coordinate_system
*/
template <class FReal>
class FSpherical {
    const FReal PI = FReal(3.14159265358979323846264338327950288419716939937510582097494459230781640628620899863L);
    const FReal PI2 = FReal(3.14159265358979323846264338327950288419716939937510582097494459230781640628620899863L*2);
    // The attributes of a sphere
    FReal r;         //!< the radial distance
    FReal theta;     //!< the inclination angle [0, pi] - colatitude, polar angle
    FReal phi;       //!< the azimuth angle [-pi,pi] - longitude - around z axis
    FReal cosTheta;
    FReal sinTheta;
public:
    /** Default Constructor, set attributes to 0 */
    FSpherical()
        : r(0.0), theta(0.0), phi(0.0), cosTheta(0.0), sinTheta(0.0) {
    }

    /** From now, we just need a constructor based on a 3D position */
    explicit FSpherical(const std::array<FReal, 3>& inVector){
        const FReal x2y2 = (inVector[0] * inVector[0]) + (inVector[1] * inVector[1]);
        this->r          = std::sqrt( x2y2 + (inVector[2] * inVector[2]));

        this->phi        = std::atan2(inVector[1],inVector[0]);

        this->cosTheta = inVector[2] / r;
        this->sinTheta = std::sqrt(x2y2) / r;
        this->theta    = std::acos(this->cosTheta);
    }

    /** Get the radius */
    FReal getR() const{
        return r;
    }

    /** Get the inclination angle theta = acos(z/r) [0, pi] */
    FReal getTheta() const{
        return theta;
    }
    /** Get the azimuth angle phi = atan2(y,x) [-pi,pi] */
    FReal getPhi() const{
        return phi;
    }

    /** Get the inclination angle [0, pi] */
    FReal getInclination() const{
        return theta;
    }
    /** Get the azimuth angle [0,2pi]. You should use this method in order to obtain (x,y,z)*/
    FReal getPhiZero2Pi() const{
        return (phi < 0 ? PI2 + phi : phi);
    }

    /** Get the cos of theta = z / r */
    FReal getCosTheta() const{
        return cosTheta;
    }

    /** Get the sin of theta = sqrt(x2 + y2) / r */
    FReal getSinTheta() const{
        return sinTheta;
    }
};

#endif // FSPHERICAL_HPP
