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
            const auto& indexPos = inConfig.getPosFromNeighborIndex(inIndexArray);
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
            const auto& indexPos = inConfig.getPosFromNeighborIndex(inIndexArray);
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
        static void ApplyShift(const SymbDataSource& inSymSrc, const SymbDataTarget& inSymTgt,
                               const SpacialConfiguration& inConfig, const long int inIndexArray,
                               PositionsData& inPositions, const long int inNbParticles){
            std::array<RealType, Dim> shiftValues = GetShiftCoef(inSymSrc, inSymTgt, inConfig, inIndexArray);

            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                for(long int idxPart = 0 ; idxPart < inNbParticles ; ++idxPart){
                    inPositions[idxDim][idxPart] += shiftValues[idxDim];
                }
            }
        }

        template <class PositionsData>
        static auto DuplicatePositions(const PositionsData& inPositions, const long int inNbParticles){
            constexpr long int NbValues = std::tuple_size<PositionsData>::value;
            static_assert (Dim <= NbValues, "Should be at least Dim");
            std::array<std::unique_ptr<RealType[]>, NbValues> copyPositions;
            for(long int idxDim = 0 ; idxDim < NbValues ; ++idxDim){
                copyPositions[idxDim].reset(new RealType[inNbParticles]);

                for(long int idxPart = 0 ; idxPart < inNbParticles ; ++idxPart){
                    copyPositions[idxDim][idxPart] = inPositions[idxDim][idxPart];
                }
            }
            return copyPositions;
        }
    };
};


#endif
