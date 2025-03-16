#include "core/tbftree.hpp"

#include "utils/tbfparams.hpp"

#include "utils/tbfutils.hpp"
#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"

#include <set>
#include <array>

#include <iostream>

int main(int argc, char **argv)
{
    if (TbfParams::ExistParameter(argc, argv, {"-h", "--help"}))
    {
        std::cout << "[HELP] Command " << argv[0] << " [params]" << std::endl;
        std::cout << "[HELP] where params are:" << std::endl;
        std::cout << "[HELP]   -h, --help: to get the current text" << std::endl;
        std::cout << "[HELP]   -th, --tree-height: the height of the tree" << std::endl;
        std::cout << "[HELP]   -nb, --nb-particles: specify the number of particles (when no file are given)" << std::endl;
        return 1;
    }

    const int Dim = 2;
    const long int TreeHeight = 5;

    using RealType = double;

    const std::array<RealType, Dim> BoxWidths{{1, 1}};
    const std::array<RealType, Dim> BoxCenter{{0.5, 0.5}};
    const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);
    const TbfMortonSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim>> morton(configuration);

    using IndexType = typename TbfMortonSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim>>::IndexType;

    std::cout << "Morton upper bound of level " << 0 << ": "
              << morton.getUpperBound(0) << " is equal to " << IndexType(1)
              << std::endl;
    std::cout << "Morton upper bound of level " << 1 << ": "
              << morton.getUpperBound(1) << " is equal to " << IndexType(morton.getNbChildrenPerCell())
              << std::endl;
    std::cout << "Morton upper bound of level " << 2 << ": "
              << morton.getUpperBound(2) << " is equal to " << IndexType(morton.getNbChildrenPerCell() * morton.getNbChildrenPerCell())
              << std::endl;

    for (long int idxLevel = 0; idxLevel < TreeHeight - 1; ++idxLevel)
    {
        std::set<std::array<long int, Dim>> generatedPositions;

        for (IndexType idx = 0; idx < morton.getUpperBound(idxLevel); ++idx)
        {
            auto pos = morton.getBoxPosFromIndex(idx);

            if(generatedPositions.find(pos) == generatedPositions.end())
            {
                // std::cout << "CORRECT:\n";
                std::cout << "New generated position {" << pos[0] << ", " << pos[1] << "} is not in the set-container yet" 
                << std::endl;
                std::cout << "Corresponding index is " << idx 
                << std::endl;
                
                generatedPositions.insert(pos);
            }
            else
            {
                std::cout << "WRONG:\n";
                // std::cout << "New generated position {" << pos[0] << ", " << pos[1] << "} is in the set-container yet"
                // << std::endl;
            }
        }
    }

    return 0;
}
