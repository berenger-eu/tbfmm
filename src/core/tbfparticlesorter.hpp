#ifndef TBFPARTICLESORTER_HPP
#define TBFPARTICLESORTER_HPP

#include "tbfglobal.hpp"

#include <vector>
#include <algorithm>
#include <cassert>

template <class RealType_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class TbfParticleSorter {
public:
    using RealType = RealType_T;
    using SpaceIndexType = SpaceIndexType_T;
    using IndexType = typename SpaceIndexType::IndexType;
private:
    std::vector<std::pair<IndexType, long int>> leaves;
    std::vector<std::pair<IndexType, long int>> particleIndexes;

public:
    template <class ContainerClass>
    explicit TbfParticleSorter(const SpaceIndexType& inSpaceSystem, const ContainerClass& inParticlePositions){
        particleIndexes.reserve(inParticlePositions.size());

        for(std::size_t idxPart = 0ul ; idxPart < static_cast<std::size_t>(inParticlePositions.size()) ; ++idxPart){
            particleIndexes.emplace_back(inSpaceSystem.getIndexFromPosition(inParticlePositions[idxPart]), idxPart);
        }

        std::sort(particleIndexes.begin(), particleIndexes.end(),[](auto& p1, auto& p2){
            return p1.first < p2.first;
        });

        leaves.reserve(inParticlePositions.size()/100);

        for(long int idxPart = 0 ; idxPart < static_cast<long int>(inParticlePositions.size()) ; ++idxPart){
            if(leaves.empty() || leaves.back().first != particleIndexes[idxPart].first){
                leaves.emplace_back();
                leaves.back().first = particleIndexes[idxPart].first;
                leaves.back().second = 0;
            }
            leaves.back().second += 1;
        }
    }

    TbfParticleSorter(const TbfParticleSorter&) = delete;
    TbfParticleSorter& operator=(const TbfParticleSorter&) = delete;

    TbfParticleSorter(TbfParticleSorter&&) = default;
    TbfParticleSorter& operator=(TbfParticleSorter&&) = default;

    long int getNbLeaves() const{
        return static_cast<long int>(leaves.size());
    }

    long int getNbParticles() const{
        return static_cast<long int>(particleIndexes.size());
    }

    IndexType getSpacialIndexForLeaf(const long int inLeafIndex) const{
        return leaves[inLeafIndex].first;
    }

    long int getNbParticlesInLeaf(const long int inLeafIndex) const{
        return leaves[inLeafIndex].second;
    }

    long int getSpacialIndexForParticle(const long int inSortedIndex) const{
        return particleIndexes[inSortedIndex].first;
    }

    long int getParticleIndex(const long int inSortedIndex) const{
        return particleIndexes[inSortedIndex].second;
    }

    ////////////////////////////////////////////////////////////////////////

    struct GroupProperty{
        const TbfParticleSorter& parent;

        long int firstCell;
        long int nbCells;
        long int firstParticle;
        long int nbParticles;
    public:
        GroupProperty(const TbfParticleSorter& inParent)
            : parent(inParent), firstCell(-1), nbCells(0), firstParticle(-1), nbParticles(0){}

        void setFirstCell(const long int inFirstCell){
            firstCell = inFirstCell;
        }

        void setNbCells(const long int inNbCells){
            nbCells = inNbCells;
        }

        void setFirstParticle(const long int inFirstParticle){
            firstParticle = inFirstParticle;
        }

        void setNbParticles(const long int inNbParticles){
            nbParticles = inNbParticles;
        }

        ////////////////////////////////////////////////////////////////////////
        long int getNbLeaves() const{
            return nbCells;
        }

        long int getNbParticles() const{
            return nbParticles;
        }

        IndexType getSpacialIndexForLeaf(const long int inLeafIndex) const{
            assert(inLeafIndex < nbCells);
            return parent.getSpacialIndexForLeaf(firstCell + inLeafIndex);
        }

        long int getNbParticlesInLeaf(const long int inLeafIndex) const{
            assert(inLeafIndex < nbCells);
            return parent.getNbParticlesInLeaf(firstCell + inLeafIndex);
        }

        long int getSpacialIndexForParticle(const long int inSortedIndex) const{
            assert(inSortedIndex < nbParticles);
            return parent.getSpacialIndexForParticle(firstParticle + inSortedIndex);
        }

        long int getParticleIndex(const long int inSortedIndex) const{
            assert(inSortedIndex < nbParticles);
            return parent.getParticleIndex(firstParticle + inSortedIndex);
        }
    };

    std::vector<GroupProperty> splitInGroups(const long int inGroupSize) const {
        if(inGroupSize <= 0){
            return std::vector<GroupProperty>();
        }

        const long int nbGroups = (getNbLeaves() + inGroupSize - 1)/inGroupSize;

        std::vector<GroupProperty> groups;
        groups.reserve(nbGroups);

        for(long int idxGroup = 0 ; idxGroup < nbGroups ; ++idxGroup){
            groups.emplace_back(*this);

            groups.back().setFirstCell(idxGroup*inGroupSize);
            groups.back().setNbCells(std::min((idxGroup+1)*inGroupSize, getNbLeaves()) - groups.back().firstCell);
            if(idxGroup == 0){
                groups.back().setFirstParticle(0);
            }
            else{
                groups.back().setFirstParticle(groups[idxGroup-1].firstParticle + groups[idxGroup-1].nbParticles);
            }

            long int currentNbParticles = 0;

            const auto groupSpacialLimit = getSpacialIndexForLeaf(groups.back().firstCell + groups.back().nbCells - 1) + 1;

            while(groups.back().firstParticle + currentNbParticles < static_cast<long int>(particleIndexes.size())
                  && getSpacialIndexForParticle(groups.back().firstParticle + currentNbParticles) < groupSpacialLimit){
                currentNbParticles += 1;
            }

            groups.back().setNbParticles(currentNbParticles);
        }

        return groups;
    }
};

#endif
