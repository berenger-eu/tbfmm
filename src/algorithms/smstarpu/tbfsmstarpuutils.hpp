#ifndef TBFSMSTARPUUTILS_HPP
#define TBFSMSTARPUUTILS_HPP

#include <vector>
#include <array>
#include <functional>

#include <starpu.h>

class TbStarPUUtils{
protected:
    static void ExecOnWorkersBind(void* ptr){
        std::function<void(void)>* func = static_cast<std::function<void(void)>*>(ptr);
        (*func)();
    }

public:
    static void ExecOnWorkers(const unsigned int inWorkersType, std::function<void(void)> func){
        starpu_execute_on_each_worker(ExecOnWorkersBind, &func, inWorkersType);
    }
};



class TbfStarPUHandleBuilder{
public:
    using CellHandleContainer = std::vector<std::vector<std::array<starpu_data_handle_t, 3>>>;
    using ParticleHandleContainer = std::vector<std::array<starpu_data_handle_t,2>>;

    template <class TreeClass, class ConfigClass>
    static auto GetCellHandles(TreeClass& inTree, ConfigClass& inConfiguration) {
        CellHandleContainer allCellHandles(inConfiguration.getTreeHeight());

        for(long int idxLevel = 0 ; idxLevel < inConfiguration.getTreeHeight() ; ++idxLevel){
            auto& cellGroups = inTree.getCellGroupsAtLevel(idxLevel);

            auto currentCellGroup = cellGroups.begin();
            const auto endCellGroup = cellGroups.end();

            while(currentCellGroup != endCellGroup){
                starpu_data_handle_t handleData;
                starpu_variable_data_register(&handleData, STARPU_MAIN_RAM,
                                              uintptr_t(currentCellGroup->getDataPtr()),
                                              uint32_t(currentCellGroup->getDataSize()));

                starpu_data_handle_t handleMultipole;
                starpu_variable_data_register(&handleMultipole, STARPU_MAIN_RAM,
                                              uintptr_t(currentCellGroup->getMultipolePtr()),
                                              uint32_t(currentCellGroup->getMultipoleSize()));

                starpu_data_handle_t handleLocal;
                starpu_variable_data_register(&handleLocal, STARPU_MAIN_RAM,
                                              uintptr_t(currentCellGroup->getLocalPtr()),
                                              uint32_t(currentCellGroup->getLocalSize()));

                std::array<starpu_data_handle_t, 3> cellHandles{handleData, handleMultipole, handleLocal};
                allCellHandles[idxLevel].push_back(cellHandles);

                ++currentCellGroup;
            }
        }
        return allCellHandles;
    }


    template <class TreeClass>
    static auto GetParticleHandles(TreeClass& inTree) {
        ParticleHandleContainer allParticlesHandles;

        auto& particleGroups = inTree.getParticleGroups();

        auto currentParticleGroup = particleGroups.begin();
        const auto endParticleGroup = particleGroups.end();

        while(currentParticleGroup != endParticleGroup){
            starpu_data_handle_t handleData;
            starpu_variable_data_register(&handleData, STARPU_MAIN_RAM,
                                          uintptr_t(currentParticleGroup->getDataPtr()),
                                          uint32_t(currentParticleGroup->getDataSize()));

            starpu_data_handle_t handleRhs;
            starpu_variable_data_register(&handleRhs, STARPU_MAIN_RAM,
                                          uintptr_t(currentParticleGroup->getRhsPtr()),
                                          uint32_t(currentParticleGroup->getRhsSize()));

            std::array<starpu_data_handle_t,2> particlesHandles{handleData, handleRhs};
            allParticlesHandles.push_back(particlesHandles);

            ++currentParticleGroup;
        }
        return allParticlesHandles;
    }


    static void CleanCellHandles(CellHandleContainer& inCellHandles) {
        for(auto& handlePerLevel : inCellHandles){
            for(auto& handleGroup : handlePerLevel){
                for(auto& handle : handleGroup){
                    starpu_data_acquire(handle, STARPU_R);
                    starpu_data_release(handle);
                    starpu_data_unregister(handle);
                }
            }
        }
    }


    static void CleanParticleHandles(ParticleHandleContainer& inParticleHandles) {
        for(auto& handleGroup : inParticleHandles){
            for(auto& handle : handleGroup){
                starpu_data_acquire(handle, STARPU_R);
                starpu_data_release(handle);
                starpu_data_unregister(handle);
            }
        }
    }
};


class TbfStarPUHandleBuilderTsm{
public:
    using CellSrcHandleContainer = std::vector<std::vector<std::array<starpu_data_handle_t, 2>>>;
    using ParticleSrcHandleContainer = std::vector<std::array<starpu_data_handle_t,1>>;

    using CellTgtHandleContainer = std::vector<std::vector<std::array<starpu_data_handle_t, 2>>>;
    using ParticleTgtHandleContainer = std::vector<std::array<starpu_data_handle_t,2>>;

    template <class TreeClass, class ConfigClass>
    static auto GetCellSrcHandles(TreeClass& inTree, ConfigClass& inConfiguration) {
        CellSrcHandleContainer allCellHandles(inConfiguration.getTreeHeight());

        for(long int idxLevel = 0 ; idxLevel < inConfiguration.getTreeHeight() ; ++idxLevel){
            auto& cellGroups = inTree.getCellGroupsAtLevelSource(idxLevel);

            auto currentCellGroup = cellGroups.begin();
            const auto endCellGroup = cellGroups.end();

            while(currentCellGroup != endCellGroup){
                starpu_data_handle_t handleData;
                starpu_variable_data_register(&handleData, STARPU_MAIN_RAM,
                                              uintptr_t(currentCellGroup->getDataPtr()),
                                              uint32_t(currentCellGroup->getDataSize()));

                starpu_data_handle_t handleMultipole;
                starpu_variable_data_register(&handleMultipole, STARPU_MAIN_RAM,
                                              uintptr_t(currentCellGroup->getMultipolePtr()),
                                              uint32_t(currentCellGroup->getMultipoleSize()));

                std::array<starpu_data_handle_t, 2> cellHandles{handleData, handleMultipole};
                allCellHandles[idxLevel].push_back(cellHandles);

                ++currentCellGroup;
            }
        }
        return allCellHandles;
    }

    template <class TreeClass, class ConfigClass>
    static auto GetCellTgtHandles(TreeClass& inTree, ConfigClass& inConfiguration) {
        CellTgtHandleContainer allCellHandles(inConfiguration.getTreeHeight());

        for(long int idxLevel = 0 ; idxLevel < inConfiguration.getTreeHeight() ; ++idxLevel){
            auto& cellGroups = inTree.getCellGroupsAtLevelTarget(idxLevel);

            auto currentCellGroup = cellGroups.begin();
            const auto endCellGroup = cellGroups.end();

            while(currentCellGroup != endCellGroup){
                starpu_data_handle_t handleData;
                starpu_variable_data_register(&handleData, STARPU_MAIN_RAM,
                                              uintptr_t(currentCellGroup->getDataPtr()),
                                              uint32_t(currentCellGroup->getDataSize()));

                starpu_data_handle_t handleLocal;
                starpu_variable_data_register(&handleLocal, STARPU_MAIN_RAM,
                                              uintptr_t(currentCellGroup->getLocalPtr()),
                                              uint32_t(currentCellGroup->getLocalSize()));

                std::array<starpu_data_handle_t, 2> cellHandles{handleData, handleLocal};
                allCellHandles[idxLevel].push_back(cellHandles);

                ++currentCellGroup;
            }
        }
        return allCellHandles;
    }


    template <class TreeClass>
    static auto GetParticleSrcHandles(TreeClass& inTree) {
        ParticleSrcHandleContainer allParticlesHandles;

        auto& particleGroups = inTree.getParticleGroupsSource();

        auto currentParticleGroup = particleGroups.begin();
        const auto endParticleGroup = particleGroups.end();

        while(currentParticleGroup != endParticleGroup){
            starpu_data_handle_t handleData;
            starpu_variable_data_register(&handleData, STARPU_MAIN_RAM,
                                          uintptr_t(currentParticleGroup->getDataPtr()),
                                          uint32_t(currentParticleGroup->getDataSize()));

            std::array<starpu_data_handle_t,1> particlesHandles{handleData};
            allParticlesHandles.push_back(particlesHandles);

            ++currentParticleGroup;
        }
        return allParticlesHandles;
    }

    template <class TreeClass>
    static auto GetParticleTgtHandles(TreeClass& inTree) {
        ParticleTgtHandleContainer allParticlesHandles;

        auto& particleGroups = inTree.getParticleGroupsTarget();

        auto currentParticleGroup = particleGroups.begin();
        const auto endParticleGroup = particleGroups.end();

        while(currentParticleGroup != endParticleGroup){
            starpu_data_handle_t handleData;
            starpu_variable_data_register(&handleData, STARPU_MAIN_RAM,
                                          uintptr_t(currentParticleGroup->getDataPtr()),
                                          uint32_t(currentParticleGroup->getDataSize()));

            starpu_data_handle_t handleRhs;
            starpu_variable_data_register(&handleRhs, STARPU_MAIN_RAM,
                                          uintptr_t(currentParticleGroup->getRhsPtr()),
                                          uint32_t(currentParticleGroup->getRhsSize()));

            std::array<starpu_data_handle_t,2> particlesHandles{handleData, handleRhs};
            allParticlesHandles.push_back(particlesHandles);

            ++currentParticleGroup;
        }
        return allParticlesHandles;
    }

    template <class AnyCellHandleContainer>
    static void CleanCellHandles(AnyCellHandleContainer& inCellHandles) {
        for(auto& handlePerLevel : inCellHandles){
            for(auto& handleGroup : handlePerLevel){
                for(auto& handle : handleGroup){
                    starpu_data_unregister(handle);
                }
            }
        }
    }

    template <class AnyParticleHandleContainer>
    static void CleanParticleHandles(AnyParticleHandleContainer& inParticleHandles) {
        for(auto& handleGroup : inParticleHandles){
            for(auto& handle : handleGroup){
                starpu_data_unregister(handle);
            }
        }
    }
};


#endif
