#ifndef TBFSPACIALCONFIGURATION_HPP
#define TBFSPACIALCONFIGURATION_HPP

#include <array>

template <class RealType_T, long int Dim_T>
class TbfSpacialConfiguration {
public:
    using RealType = RealType_T;
    constexpr static long int Dim = Dim_T;

private:
    const long int treeHeight;
    const std::array<RealType, Dim> boxCenter;
    const std::array<RealType, Dim> boxCorner;
    const std::array<RealType, Dim> boxWidths;
    const std::array<RealType, Dim> boxWidthsAtLeafLevel;

public:
    TbfSpacialConfiguration(const long int inTreeHeight, const std::array<RealType, Dim>& inBoxWidths, const std::array<RealType, Dim>& inBoxCenter)
        : treeHeight(inTreeHeight),
          boxCenter(inBoxCenter),
          boxCorner(TbfUtils::AddVecToVec(inBoxCenter, TbfUtils::MulToVec(inBoxWidths, -RealType(1)/RealType(2)))),
          boxWidths(inBoxWidths),
          boxWidthsAtLeafLevel(TbfUtils::MulToVec(inBoxWidths, RealType(1)/RealType(1<<(inTreeHeight-1)))){
    }

    TbfSpacialConfiguration(const TbfSpacialConfiguration&) = default;
    TbfSpacialConfiguration& operator=(const TbfSpacialConfiguration&) = default;

    TbfSpacialConfiguration(TbfSpacialConfiguration&&) = delete;
    TbfSpacialConfiguration& operator=(TbfSpacialConfiguration&&) = delete;

    long int getTreeHeight() const{
        return treeHeight;
    }

    const std::array<RealType, Dim>& getBoxCenter() const{
        return boxCenter;
    }

    const std::array<RealType, Dim>& getBoxCorner() const{
        return boxCorner;
    }

    const std::array<RealType, Dim>& getBoxWidths() const{
        return boxWidths;
    }

    const std::array<RealType, Dim>& getLeafWidths() const{
        return boxWidthsAtLeafLevel;
    }

    // With C++20 bool operator==(const TbfSpacialConfiguration&) const = default;
    bool operator==(const TbfSpacialConfiguration& other) const{
        return treeHeight == other.treeHeight
                && boxCenter == other.boxCenter
                && boxCorner == other.boxCorner
                && boxWidths == other.boxWidths
                && boxWidthsAtLeafLevel == other.boxWidthsAtLeafLevel;
    }


    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfSpacialConfiguration& inConfiguration) {
        inStream << "TbfSpacialConfiguration @ " << &inConfiguration << "\n";
        inStream << " - Dim " << Dim << "\n";
        inStream << " - treeHeight " << inConfiguration.getTreeHeight() << "\n";
        inStream << " - box center " << TbfUtils::ArrayPrinter(inConfiguration.getBoxCenter()) << "\n";
        inStream << " - box widths " << TbfUtils::ArrayPrinter(inConfiguration.getBoxWidths()) << "\n";
        return inStream;
    }
};

#endif

