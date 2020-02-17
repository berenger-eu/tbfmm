#ifndef TBFPERIODICSHIFTER_HPP
#define TBFPERIODICSHIFTER_HPP

template <class RealType, class SpacialConfiguration>
class TbfPeriodicShifter {
    static constexpr long int Dim = SpacialConfiguration::Dim;

public:
    struct Neighbor{
        template <class SymbDataSource, class SymbDataTarget>
        static bool NeedToShift(const SymbDataSource& inSymSrc, const SymbDataTarget& inSymTgt,
                                const SpacialConfiguration& inConfig, const long int inIndexArray){
            [[maybe_unused]] const auto& symSrcPos = inSymSrc.boxCoord;
            const auto& symTgtPos = inSymTgt.boxCoord;
            const auto& indexPos = inConfig.getRelativePosFromNeighborIndex(inIndexArray);
            const long int boxLimit = inConfig.getBoxLimitAtLeafLevel();

            bool needShift = false;

            for(long int idxDim = 0 ; !needShift && idxDim < Dim ; ++idxDim){
                if(symTgtPos[idxDim] + indexPos[idxDim] < 0){
                    assert(symSrcPos[idxDim] == boxLimit-1);
                    needShift = true;
                }
                else if (boxLimit <= symTgtPos[idxDim] + indexPos[idxDim]){
                    assert(symSrcPos[idxDim] == 0);
                    needShift = true;
                }
            }

            return needShift;
        }

        template <class SymbDataSource, class SymbDataTarget>
        static std::array<RealType, Dim> GetShiftCoef(const SymbDataSource& inSymSrc, const SymbDataTarget& inSymTgt,
                                                      const SpacialConfiguration& inConfig, const long int inIndexArray){
            std::array<RealType, Dim> shiftValues;

            [[maybe_unused]] const auto& symSrcPos = inSymSrc.boxCoord;
            const auto& symTgtPos = inSymTgt.boxCoord;
            const auto& indexPos = inConfig.getRelativePosFromNeighborIndex(inIndexArray);
            const long int boxLimit = inConfig.getBoxLimitAtLeafLevel();
            const auto& boxWidths = inConfig.getConfiguration().getBoxWidths();

            bool needShift = false;

            for(long int idxDim = 0 ; !needShift && idxDim < Dim ; ++idxDim){
                if(symTgtPos[idxDim] + indexPos[idxDim] < 0){
                    assert(symSrcPos[idxDim] == boxLimit-1);
                    shiftValues[idxDim] = -boxWidths[idxDim];
                }
                else if (boxLimit <= symTgtPos[idxDim] + indexPos[idxDim]){
                    assert(symSrcPos[idxDim] == 0);
                    shiftValues[idxDim] = boxWidths[idxDim];
                }
                else{
                    shiftValues[idxDim] = 0;
                }
            }

            return shiftValues;
        }

        template <class SymbDataSource, class SymbDataTarget, class PositionsData>
        static auto DuplicatePositionsAndApplyShift(const SymbDataSource& inSymSrc, const SymbDataTarget& inSymTgt,
                               const SpacialConfiguration& inConfig, const long int inIndexArray,
                               PositionsData& inPositions, const long int inNbParticles){
            const std::array<RealType, Dim> shiftValues = GetShiftCoef(inSymSrc, inSymTgt, inConfig, inIndexArray);

            constexpr long int NbValues = std::tuple_size<PositionsData>::value;
            static_assert (Dim <= NbValues, "Should be at least Dim");
            std::array<const RealType*, NbValues> copyPositions;

            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                RealType* ptr = (new RealType[inNbParticles]);

                for(long int idxPart = 0 ; idxPart < inNbParticles ; ++idxPart){
                    ptr[idxPart] = inPositions[idxDim][idxPart] + shiftValues[idxDim];
                }

                copyPositions[idxDim] = ptr;
            }

            for(long int idxValue = Dim ; idxValue < NbValues ; ++idxValue){
                RealType* ptr = (new RealType[inNbParticles]);

                for(long int idxPart = 0 ; idxPart < inNbParticles ; ++idxPart){
                    ptr[idxPart] = inPositions[idxValue][idxPart];
                }

                copyPositions[idxValue] = ptr;
            }

            return copyPositions;
        }

        template <const long unsigned int NbValues>
        static auto FreePositions(const std::array<const RealType*, NbValues>& copiedPositions){
            for(long unsigned int idxDim = 0 ; idxDim < NbValues ; ++idxDim){
                delete[] copiedPositions[idxDim];
            }

        }
    };
};


#endif
