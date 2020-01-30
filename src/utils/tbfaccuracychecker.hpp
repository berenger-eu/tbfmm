#ifndef TBFACCURACYCHECKER_HPP
#define TBFACCURACYCHECKER_HPP

#include <cmath>
#include <limits>

template <class RealType_T>
class TbfAccuracyChecker{
public:
    using RealType = RealType_T;

private:
    RealType maxDiff = 0;
    RealType maxValue = std::numeric_limits<RealType>::min();
    RealType l2Diff = 0;
    RealType l2Dot = 0;
    RealType maxRelativeDiff = 0;

    long int nbElements = 0;

public:
    TbfAccuracyChecker() = default;

    void reset(){
        (*this) = TbfAccuracyChecker<RealType>();
    }

    void addValues(const RealType inGood, const RealType inBad){
        maxDiff = std::max(maxDiff, std::abs(inGood - inBad));
        maxValue = std::max(maxValue, inGood);
        l2Diff += ((inBad - inGood) * (inBad - inGood));
        l2Dot += (inGood * inGood);
        maxRelativeDiff = std::max(maxRelativeDiff, (inGood != 0 ? std::abs(inGood - inBad)/std::abs(inGood) : inBad));
        nbElements += 1;
    }

    void addAccuracyChecker(const TbfAccuracyChecker<RealType>& inOther){
        maxDiff = std::max(maxDiff, inOther.maxDiff);
        maxValue = std::max(maxValue, inOther.maxValue);
        l2Diff += inOther.l2Diff;
        l2Dot += inOther.l2Dot;
        maxRelativeDiff = std::max(maxRelativeDiff, inOther.maxRelativeDiff);
        nbElements += inOther.nbElements;
    }

    template <class ItemContainer>
    void addManyValues(ItemContainer&& inContainerGood,
                       ItemContainer&& inContainerBad,
                       const long int inSize){
        for(long int idxVal = 0 ; idxVal < inSize ; ++idxVal){
            addValues(inContainerGood[idxVal], inContainerBad[idxVal]);
        }
    }

    RealType getl2Diff() const{
        return l2Diff;
    }

    RealType getl2Dot() const{
        return l2Dot;
    }

    RealType getmax() const{
        return maxValue;
    }

    long int getNbElements() const{
        return nbElements;
    }

    RealType getL2Norm() const{
        return std::sqrt(l2Diff );
    }

    RealType getRMSError() const{
        if(nbElements){
            return std::sqrt(l2Diff /static_cast<RealType>(nbElements));
        }
        else{
            return 0;
        }
    }

    RealType getInfNorm() const{
        return maxDiff;
    }

    RealType getRelativeL2Norm() const{
        if(l2Dot != 0 && l2Diff != 0){
            return std::sqrt(l2Diff / l2Dot);
        }
        else{
            return l2Diff;
        }
    }

    RealType getRelativeInfNorm() const{
        return maxDiff / maxValue;
    }

    RealType getMaxRelativeDiff() const{
        return maxRelativeDiff;
    }

    template <class StreamClass>
    friend StreamClass& operator<<(StreamClass& output, const TbfAccuracyChecker<RealType>& inAccurater){
        output << "Relative L2Norm = " << inAccurater.getRelativeL2Norm()
               << " \t RMS Norm = " << inAccurater.getRMSError()
               << " \t Relative Inf = " << inAccurater.getRelativeInfNorm()
               << " \t Max Relative diff = " << inAccurater.getMaxRelativeDiff();
        return output;
    }
};

#endif
