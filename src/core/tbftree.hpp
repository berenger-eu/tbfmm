#ifndef TBFTREE_HPP
#define TBFTREE_HPP

#include "tbfglobal.hpp"
#include "tbfparticlescontainer.hpp"
#include "tbfinteraction.hpp"
#include "tbfcellscontainer.hpp"

#include "algorithms/tbfblocksizefinder.hpp"

#include <vector>
#include <array>

template <class RealType, class DataType, long int NbDataValuesPerParticle, class RhsType, long int NbRhsValuesPerParticle,
          class MultipoleClass, class LocalClass, class SpaceIndexType = TbfDefaultSpaceIndexType<RealType>>
class TbfTree {
public:
    using LeafGroupClass = TbfParticlesContainer<RealType, DataType, NbDataValuesPerParticle, RhsType, NbRhsValuesPerParticle, SpaceIndexType>;
    using CellGroupClass = TbfCellsContainer<RealType, MultipoleClass, LocalClass, SpaceIndexType>;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;
    using IndexType = typename TbfDefaultSpaceIndexType<RealType>::IndexType;

protected:
    const SpacialConfiguration configuration;
    const SpaceIndexType spaceSystem;
    const long int nbElementsPerBlock;
    const bool oneGroupPerParent;

    std::vector<std::vector<CellGroupClass>> cellBlocks;
    std::vector<LeafGroupClass> particleGroups;

    long int nbParticles;

public:

    template<class ParticleContainer>
    TbfTree(const SpacialConfiguration& inConfiguration,
               const ParticleContainer& inParticlePositions,
               const long int inNbElementsPerBlock = -1,
               const bool inOneGroupPerParent = false)
        : configuration(inConfiguration), spaceSystem(configuration),
          nbElementsPerBlock(inNbElementsPerBlock == -1 ? TbfBlockSizeFinder::Estimate<RealType, ParticleContainer, SpaceIndexType>(inParticlePositions,
                                                                                                 inConfiguration):
                                                          inNbElementsPerBlock),
          oneGroupPerParent(inOneGroupPerParent), nbParticles(static_cast<long int>(std::size(inParticlePositions))){

        cellBlocks.resize(configuration.getTreeHeight());
        if(std::size(inParticlePositions) == 0){
            return;
        }

        {
            TbfParticleSorter<RealType, SpaceIndexType> partSorter(spaceSystem, inParticlePositions);
            const auto groupProperties = partSorter.splitInGroups(nbElementsPerBlock);
            particleGroups.reserve(std::size(groupProperties));

            for(const auto& groupProperty : groupProperties){
                particleGroups.emplace_back(groupProperty, inParticlePositions, spaceSystem);
            }
        }

        if(configuration.getTreeHeight() <= 0){
            return;
        }

        {
            std::vector<IndexType> leafIndexes;

            cellBlocks[configuration.getTreeHeight()-1].reserve(particleGroups.size());
            for(const auto& particleGroup : particleGroups){
                leafIndexes.resize(particleGroup.getNbLeaves());

                for(long int idxLeaf = 0 ; idxLeaf < particleGroup.getNbLeaves() ; ++idxLeaf){
                    leafIndexes[idxLeaf] = particleGroup.getLeafSpacialIndex(idxLeaf);
                }

                cellBlocks[configuration.getTreeHeight()-1].emplace_back(leafIndexes, spaceSystem);
            }
        }

        std::vector<IndexType> cellIndexes;
        cellIndexes.reserve(nbElementsPerBlock);

        for(long int idxLevel = configuration.getTreeHeight()-2 ; idxLevel >= 0 ; --idxLevel){
            if(oneGroupPerParent){
                cellBlocks[idxLevel].reserve(cellBlocks[idxLevel+1].size());

                for(const auto& lowerCellGroup : cellBlocks[idxLevel+1]){
                    cellIndexes.clear();
                    long int idxCell = 0;

                    if(cellBlocks[idxLevel].size()){
                        while(idxCell < lowerCellGroup.getNbCells()
                              && spaceSystem.getParentIndex(lowerCellGroup.getCellSpacialIndex(idxCell)) <= cellBlocks[idxLevel].back().getEndingSpacialIndex()){
                            idxCell += 1;
                        }
                    }

                    for( ; idxCell < lowerCellGroup.getNbCells() ; ++idxCell){
                        if(cellIndexes.size() == 0 || cellIndexes.back() != spaceSystem.getParentIndex(lowerCellGroup.getCellSpacialIndex(idxCell))){
                            cellIndexes.push_back(spaceSystem.getParentIndex(lowerCellGroup.getCellSpacialIndex(idxCell)));
                        }
                    }

                    if(cellIndexes.size()){
                        cellBlocks[idxLevel].emplace_back(cellIndexes, spaceSystem);
                    }
                }
            }
            else{
                cellBlocks[idxLevel].reserve(cellBlocks[idxLevel+1].size()/8);

                cellIndexes.clear();
                IndexType previousIndex = -1;

                for(const auto& lowerCellGroup : cellBlocks[idxLevel+1]){
                    for(long int idxCell = 0; idxCell < lowerCellGroup.getNbCells() ; ++idxCell){
                        if(previousIndex != spaceSystem.getParentIndex(lowerCellGroup.getCellSpacialIndex(idxCell))){
                            cellIndexes.push_back(spaceSystem.getParentIndex(lowerCellGroup.getCellSpacialIndex(idxCell)));
                            previousIndex = spaceSystem.getParentIndex(lowerCellGroup.getCellSpacialIndex(idxCell));

                            if(static_cast<long int>(cellIndexes.size()) == nbElementsPerBlock){
                                cellBlocks[idxLevel].emplace_back(cellIndexes, spaceSystem);
                                cellIndexes.clear();
                            }
                        }
                    }
                }

                if(cellIndexes.size()){
                    cellBlocks[idxLevel].emplace_back(cellIndexes, spaceSystem);
                    cellIndexes.clear();
                }
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////

    long int getNbParticles() const{
        return nbParticles;
    }

    long int getNbElementsPerGroup() const{
        return nbElementsPerBlock;
    }

    const SpacialConfiguration& getSpacialConfiguration() const{
        return configuration;
    }

    const SpaceIndexType& getSpacialSystem() const{
        return spaceSystem;
    }

    long int getHeight() const{
        return configuration.getTreeHeight();
    }

    long int getNbCellGroupsAtLevel(const long int inIdxLevel) const{
        return static_cast<long int>(cellBlocks[inIdxLevel].size());
    }

    std::vector<CellGroupClass>& getCellGroupsAtLevel(const long int inIdxLevel){
        return cellBlocks[inIdxLevel];
    }

    const std::vector<CellGroupClass>& getCellGroupsAtLevel(const long int inIdxLevel) const {
        return cellBlocks[inIdxLevel];
    }

    std::vector<CellGroupClass>& getLeafGroups(){
        return cellBlocks[cellBlocks.size()-1];
    }

    const std::vector<CellGroupClass>& getLeafGroups() const {
        return cellBlocks[cellBlocks.size()-1];
    }

    long int getNbParticleGroups() const{
        return static_cast<long int>(particleGroups.size());
    }

    std::vector<LeafGroupClass>& getParticleGroups(){
        return particleGroups;
    }

    const std::vector<LeafGroupClass>& getParticleGroups() const {
        return particleGroups;
    }

    //////////////////////////////////////////////////////////////////////////////

    auto findGroupWithCell(const long int inLevel, const IndexType inMIndex){
        assert(inLevel < configuration.getTreeHeight());
        auto iterCells = cellBlocks[inLevel].begin();
        auto endCells = cellBlocks[inLevel].end();

        // for(auto& cellGroup : cellBlocks[inLevel]){
        const auto cellGroupIter = std::lower_bound( iterCells, endCells, inMIndex, [](const auto& cellsToTest, const auto& mindex){
            return cellsToTest.getEndingSpacialIndex() < mindex;
        });

        if(cellGroupIter != endCells){
            auto& cellGroup = (*cellGroupIter);
            if(cellGroup.getStartingSpacialIndex() <= inMIndex && inMIndex <= cellGroup.getEndingSpacialIndex()){
                auto foundCell = cellGroup.getElementFromSpacialIndex(inMIndex);
                if(foundCell){
                    return std::optional<std::pair<std::reference_wrapper<CellGroupClass>,long int>>(std::make_pair(std::ref(cellGroup), *foundCell));
                }
            }
        }

        return std::optional<std::pair<std::reference_wrapper<CellGroupClass>,long int>>();
    }


    auto findGroupWithLeaf(const IndexType inMIndex){
        auto iterLeaves = particleGroups.begin();
        auto endLeaves = particleGroups.end();

        //for(auto& leafGroup : particleGroups){
        const auto leafGroupIter = std::lower_bound( iterLeaves, endLeaves, inMIndex, [](const auto& leavesToTest, const auto& mindex){
            return  leavesToTest.getEndingSpacialIndex() < mindex;
        });

        if(leafGroupIter != endLeaves){
            auto& leafGroup = *leafGroupIter;
            if(leafGroup.getStartingSpacialIndex() <= inMIndex && inMIndex <= leafGroup.getEndingSpacialIndex()){
                auto foundLeaf = leafGroup.getElementFromSpacialIndex(inMIndex);
                if(foundLeaf){
                    return std::optional<std::pair<std::reference_wrapper<LeafGroupClass>,long int>>(std::make_pair(std::ref(leafGroup), *foundLeaf));
                }
            }
        }
        return std::optional<std::pair<std::reference_wrapper<LeafGroupClass>,long int>>();
    }

    //////////////////////////////////////////////////////////////////////////////

    template <class FuncClass>
    void applyToAllCells(FuncClass&& inFunc){
        for (long int idxLevel = 0 ; idxLevel < configuration.getTreeHeight() ; ++idxLevel) {
            for(auto& cellGroup : cellBlocks[idxLevel]){
                cellGroup.applyToAllCells(idxLevel, inFunc);
            }
        }
    }

    template <class FuncClass>
    void applyToAllLeaves(FuncClass&& inFunc){
        for(auto& leafGroup : particleGroups){
            leafGroup.applyToAllLeaves(inFunc);
        }
    }

    template <class FuncClass>
    void applyToAllCells(FuncClass&& inFunc) const {
        for (long int idxLevel = 0 ; idxLevel < configuration.getTreeHeight() ; ++idxLevel) {
            for(auto& cellGroup : cellBlocks[idxLevel]){
                cellGroup.applyToAllCells(idxLevel, inFunc);
            }
        }
    }

    template <class FuncClass>
    void applyToAllLeaves(FuncClass&& inFunc) const {
        for(auto& leafGroup : particleGroups){
            leafGroup.applyToAllLeaves(inFunc);
        }
    }

    //////////////////////////////////////////////////////////////////////////////

    auto getAllParticlesData(){
        std::unique_ptr<std::array<RealType, NbDataValuesPerParticle>[]> data(new std::array<RealType, NbDataValuesPerParticle>[nbParticles]());

        applyToAllLeaves([&data](auto&& leafHeader, const long int* particleIndexes,
                             const std::array<DataType*, NbDataValuesPerParticle> particleDataPtr,
                             const std::array<RhsType*, NbRhsValuesPerParticle> /*particleRhsPtr*/){
            for(int idxValue = 0 ; idxValue < NbDataValuesPerParticle ; ++idxValue){
                for(long int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
                    data[idxValue][particleIndexes[idxPart]] = particleDataPtr[idxValue][idxPart];
                }
            }
        });

        return data;
    }

    auto getAllParticlesRhs(){
        std::unique_ptr<std::array<RhsType, NbRhsValuesPerParticle>[]> rhs(new std::array<RhsType, NbRhsValuesPerParticle>[nbParticles]());

        applyToAllLeaves([&rhs](auto&& leafHeader, const long int* particleIndexes,
                             const std::array<DataType*, NbDataValuesPerParticle> /*particleDataPtr*/,
                             const std::array<RhsType*, NbRhsValuesPerParticle> particleRhsPtr){
            for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
                for(long int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
                    rhs[idxValue][particleIndexes[idxPart]] = particleRhsPtr[idxValue][idxPart];
                }
            }
        });

        return rhs;
    }

    void rebuild(){
        std::vector<std::array<RealType, NbDataValuesPerParticle>> data(nbParticles);
        std::vector<std::array<RhsType, NbRhsValuesPerParticle>> rhs(nbParticles);

        applyToAllLeaves([&data, &rhs](auto&& leafHeader, const long int* particleIndexes,
                             const std::array<DataType*, NbDataValuesPerParticle> particleDataPtr,
                             const std::array<RhsType*, NbRhsValuesPerParticle> particleRhsPtr){

            for(int idxValue = 0 ; idxValue < NbDataValuesPerParticle ; ++idxValue){
                for(long int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
                    data[particleIndexes[idxPart]][idxValue] = particleDataPtr[idxValue][idxPart];
                }
            }
            for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
                for(long int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
                    rhs[particleIndexes[idxPart]][idxValue] = particleRhsPtr[idxValue][idxPart];
                }
            }
        });


        cellBlocks.clear();
        particleGroups.clear();

        cellBlocks.resize(configuration.getTreeHeight());
        if(std::size(data) == 0){
            return;
        }

        {
            TbfParticleSorter<RealType> partSorter(spaceSystem, data);
            const auto groupProperties = partSorter.splitInGroups(nbElementsPerBlock);
            particleGroups.reserve(std::size(groupProperties));

            for(const auto& groupProperty : groupProperties){
                particleGroups.emplace_back(groupProperty, data, spaceSystem);
            }
        }

        if(configuration.getTreeHeight() <= 0){
            return;
        }

        {
            std::vector<IndexType> leafIndexes;

            cellBlocks[configuration.getTreeHeight()-1].reserve(particleGroups.size());
            for(const auto& particleGroup : particleGroups){
                leafIndexes.resize(particleGroup.getNbLeaves());

                for(long int idxLeaf = 0 ; idxLeaf < particleGroup.getNbLeaves() ; ++idxLeaf){
                    leafIndexes[idxLeaf] = particleGroup.getLeafSpacialIndex(idxLeaf);
                }

                cellBlocks[configuration.getTreeHeight()-1].emplace_back(leafIndexes, spaceSystem);
            }
        }

        std::vector<IndexType> cellIndexes;
        cellIndexes.reserve(nbElementsPerBlock);

        for(long int idxLevel = configuration.getTreeHeight()-2 ; idxLevel >= 0 ; --idxLevel){
            if(oneGroupPerParent){
                cellBlocks[idxLevel].reserve(cellBlocks[idxLevel+1].size());

                for(const auto& lowerCellGroup : cellBlocks[idxLevel+1]){
                    cellIndexes.clear();
                    long int idxCell = 0;

                    if(cellBlocks[idxLevel].size()){
                        while(idxCell < lowerCellGroup.getNbCells()
                              && spaceSystem.getParentIndex(lowerCellGroup.getCellSpacialIndex(idxCell)) <= cellBlocks[idxLevel].back().getEndingSpacialIndex()){
                            idxCell += 1;
                        }
                    }

                    for( ; idxCell < lowerCellGroup.getNbCells() ; ++idxCell){
                        if(cellIndexes.size() == 0 || cellIndexes.back() != spaceSystem.getParentIndex(lowerCellGroup.getCellSpacialIndex(idxCell))){
                            cellIndexes.push_back(spaceSystem.getParentIndex(lowerCellGroup.getCellSpacialIndex(idxCell)));
                        }
                    }

                    if(cellIndexes.size()){
                        cellBlocks[idxLevel].emplace_back(cellIndexes, spaceSystem);
                    }
                }
            }
            else{
                cellBlocks[idxLevel].reserve(cellBlocks[idxLevel+1].size()/8);

                cellIndexes.clear();
                IndexType previousIndex = -1;

                for(const auto& lowerCellGroup : cellBlocks[idxLevel+1]){
                    for(long int idxCell = 0; idxCell < lowerCellGroup.getNbCells() ; ++idxCell){
                        if(previousIndex != spaceSystem.getParentIndex(lowerCellGroup.getCellSpacialIndex(idxCell))){
                            cellIndexes.push_back(spaceSystem.getParentIndex(lowerCellGroup.getCellSpacialIndex(idxCell)));
                            previousIndex = spaceSystem.getParentIndex(lowerCellGroup.getCellSpacialIndex(idxCell));

                            if(static_cast<long int>(cellIndexes.size()) == nbElementsPerBlock){
                                cellBlocks[idxLevel].emplace_back(cellIndexes, spaceSystem);
                                cellIndexes.clear();
                            }
                        }
                    }
                }

                if(cellIndexes.size()){
                    cellBlocks[idxLevel].emplace_back(cellIndexes, spaceSystem);
                    cellIndexes.clear();
                }
            }
        }

        applyToAllLeaves([&rhs](auto&& leafHeader, const long int* particleIndexes,
                                  const std::array<DataType*, NbDataValuesPerParticle> /*particleDataPtr*/,
                                  const std::array<RhsType*, NbRhsValuesPerParticle> particleRhsPtr){
             for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
                 for(long int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
                     particleRhsPtr[idxValue][idxPart] = rhs[particleIndexes[idxPart]][idxValue];
                 }
             }
         });

    }


    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfTree& inAlgo) {
        inStream << "TbfTree @ " << &inAlgo << "\n";
        inStream << " - Configuration: " << "\n";
        inStream << inAlgo.configuration << "\n";
        inStream << " - Space system: " << "\n";
        inStream << inAlgo.spaceSystem << "\n";
        inStream << " - Number of elements per block: " << inAlgo.nbElementsPerBlock << "\n";
        inStream << " - One group per element: " << inAlgo.oneGroupPerParent << "\n";
        inStream << " - Number of particles: " << inAlgo.nbParticles << "\n";

        inStream << " -- Cell groups:" << "\n";
        for (long int idxLevel = 0 ; idxLevel < inAlgo.configuration.getTreeHeight() ; ++idxLevel) {
            inStream << " -- level:" << idxLevel << "\n";
            for(const auto& cellGroup : inAlgo.cellBlocks[idxLevel]){
                inStream << (cellGroup) << "\n";
            }
        }

        inStream << " -- Leaf groups:" << "\n";
        for(const auto& leafGroup : inAlgo.particleGroups){
            inStream << (leafGroup) << "\n";
        }
        return inStream;
    }
};

#endif
