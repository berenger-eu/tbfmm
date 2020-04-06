#ifndef TBFTREETSM_HPP
#define TBFTREETSM_HPP

#include "tbfglobal.hpp"
#include "tbfparticlescontainer.hpp"
#include "tbfinteraction.hpp"
#include "tbfcellscontainer.hpp"
#include "tbftree.hpp"

#include "algorithms/tbfblocksizefinder.hpp"

#include <vector>
#include <array>

template <class RealType, class DataType, long int NbDataValuesPerParticle, class RhsType, long int NbRhsValuesPerParticle,
          class MultipoleClass, class LocalClass, class SpaceIndexType = TbfDefaultSpaceIndexType<RealType>>
class TbfTreeTsm {
public:
    using TreeClassSource = TbfTree<RealType, DataType, NbDataValuesPerParticle, void_data, 0, MultipoleClass, void_data, SpaceIndexType>;
    using TreeClassTarget = TbfTree<RealType, DataType, NbDataValuesPerParticle, RhsType, NbRhsValuesPerParticle, void_data, LocalClass, SpaceIndexType>;

    using LeafGroupClassSource = typename TreeClassSource::LeafGroupClass;
    using CellGroupClassSource = typename TreeClassSource::CellGroupClass;

    using LeafGroupClassTarget = typename TreeClassTarget::LeafGroupClass;
    using CellGroupClassTarget = typename TreeClassTarget::CellGroupClass;

    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;
    using IndexType = typename TbfDefaultSpaceIndexType<RealType>::IndexType;

protected:
    const SpacialConfiguration configuration;
    const SpaceIndexType spaceSystem;

    TreeClassSource treeSource;
    TreeClassTarget treeTarget;

public:

    template<class ParticleContainer>
    TbfTreeTsm(const SpacialConfiguration& inConfiguration,
               const ParticleContainer& inParticleSourcePositions,
               const ParticleContainer& inParticleTargetPositions,
               const long int inNbElementsPerBlock = -1,
               const bool inOneGroupPerParent = false)
        : configuration(inConfiguration), spaceSystem(configuration),
          treeSource(inConfiguration, inParticleSourcePositions,
                     inNbElementsPerBlock == -1 ?
                         TbfBlockSizeFinder::EstimateTsm<RealType>(inParticleSourcePositions, inParticleTargetPositions, configuration):
                         inNbElementsPerBlock,
                     inOneGroupPerParent),
          treeTarget(inConfiguration, inParticleTargetPositions,
                     inNbElementsPerBlock == -1 ?
                         TbfBlockSizeFinder::EstimateTsm<RealType>(inParticleSourcePositions, inParticleTargetPositions, configuration):
                         inNbElementsPerBlock,
                     inOneGroupPerParent){
    }

    //////////////////////////////////////////////////////////////////////////////

    long int getNbParticles() const{
        return treeSource.getNbParticles() + treeTarget.getNbParticles();
    }

    long int getNbElementsPerGroupSource() const{
        return treeSource.getNbElementsPerGroup();
    }

    long int getNbElementsPerGroupTarget() const{
        return treeTarget.getNbElementsPerGroup();
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

    //////////////////////////////////////////////////////////////////////////////

    long int getNbCellGroupsAtLevelSource(const long int inIdxLevel) const{
        return treeSource.getNbCellGroupsAtLevel(inIdxLevel);
    }

    auto& getCellGroupsAtLevelSource(const long int inIdxLevel){
        return treeSource.getCellGroupsAtLevel(inIdxLevel);
    }

    const auto& getCellGroupsAtLevelSource(const long int inIdxLevel) const {
        return treeSource.getCellGroupsAtLevel(inIdxLevel);
    }

    auto& getLeafGroupsSource(){
        return treeSource.getLeafGroups();
    }

    const auto& getLeafGroupsSource() const {
        return treeSource.getLeafGroups();
    }

    long int getNbParticleGroupsSource() const{
        return treeSource.getNbParticleGroups();
    }

    auto& getParticleGroupsSource(){
        return treeSource.getParticleGroups();
    }

    const auto& getParticleGroupsSource() const {
        return treeSource.getParticleGroups();
    }

    long int getNbCellGroupsAtLevelTarget(const long int inIdxLevel) const{
        return treeTarget.getNbCellGroupsAtLevel(inIdxLevel);
    }

    auto& getCellGroupsAtLevelTarget(const long int inIdxLevel){
        return treeTarget.getCellGroupsAtLevel(inIdxLevel);
    }

    const auto& getCellGroupsAtLevelTarget(const long int inIdxLevel) const {
        return treeTarget.getCellGroupsAtLevel(inIdxLevel);
    }

    auto& getLeafGroupsTarget(){
        return treeTarget.getLeafGroups();
    }

    const auto& getLeafGroupsTarget() const {
        return treeTarget.getLeafGroups();
    }

    long int getNbParticleGroupsTarget() const{
        return treeTarget.getNbParticleGroups();
    }

    auto& getParticleGroupsTarget(){
        return treeTarget.getParticleGroups();
    }

    const auto& getParticleGroupsTarget() const {
        return treeTarget.getParticleGroups();
    }

    //////////////////////////////////////////////////////////////////////////////

    auto findGroupWithCellSource(const long int inLevel, const IndexType inMIndex){
        return treeSource.findGroupWithCell(inLevel, inMIndex);
    }

    auto findGroupWithCellTarget(const long int inLevel, const IndexType inMIndex){
        return treeTarget.findGroupWithCell(inLevel, inMIndex);
    }

    auto findGroupWithLeafSource(const IndexType inMIndex){
        return treeSource.findGroupWithLeaf(inMIndex);
    }

    auto findGroupWithLeafTarget(const IndexType inMIndex){
        return treeTarget.findGroupWithLeaf(inMIndex);
    }

    //////////////////////////////////////////////////////////////////////////////

    template <class FuncClass>
    void applyToAllCells(FuncClass&& inFunc){
        treeSource.applyToAllCells(inFunc);
        treeTarget.applyToAllCells(inFunc);
    }

    template <class FuncClass>
    void applyToAllCellsSource(FuncClass&& inFunc){
        treeSource.applyToAllCells(inFunc);
    }

    template <class FuncClass>
    void applyToAllCellsTarget(FuncClass&& inFunc){
        treeTarget.applyToAllCells(inFunc);
    }

    template <class FuncClass>
    void applyToAllLeaves(FuncClass&& inFunc){
        treeSource.applyToAllLeaves(inFunc);
        treeTarget.applyToAllLeaves(inFunc);
    }

    template <class FuncClass>
    void applyToAllLeavesSource(FuncClass&& inFunc){
        treeSource.applyToAllLeaves(inFunc);
    }

    template <class FuncClass>
    void applyToAllLeavesTarget(FuncClass&& inFunc){
        treeTarget.applyToAllLeaves(inFunc);
    }

    template <class FuncClass>
    void applyToAllCells(FuncClass&& inFunc) const {
        treeSource.applyToAllCells(inFunc);
        treeTarget.applyToAllCells(inFunc);
    }

    template <class FuncClass>
    void applyToAllCellsSource(FuncClass&& inFunc) const {
        treeSource.applyToAllCells(inFunc);
    }

    template <class FuncClass>
    void applyToAllCellsTarget(FuncClass&& inFunc) const {
        treeTarget.applyToAllCells(inFunc);
    }

    template <class FuncClass>
    void applyToAllLeaves(FuncClass&& inFunc) const {
        treeSource.applyToAllLeaves(inFunc);
        treeTarget.applyToAllLeaves(inFunc);
    }

    template <class FuncClass>
    void applyToAllLeavesSource(FuncClass&& inFunc) const {
        treeSource.applyToAllLeaves(inFunc);
    }

    template <class FuncClass>
    void applyToAllLeavesTarget(FuncClass&& inFunc) const {
        treeTarget.applyToAllLeaves(inFunc);
    }

    //////////////////////////////////////////////////////////////////////////////

    auto getAllParticlesDataSource(){
        return treeSource.getAllParticlesData();
    }

    auto getAllParticlesDataTarget(){
        return treeTarget.getAllParticlesData();
    }

    auto getAllParticlesRhsTarget(){
        return treeTarget.getAllParticlesRhs();
    }

    void rebuild(){
        treeSource.rebuild();
        treeTarget.rebuild();
    }


    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfTreeTsm& inAlgo) {
        inStream << "TbfTreeTsm @ " << &inAlgo << "\n";
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
