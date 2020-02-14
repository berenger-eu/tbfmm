#ifndef TBFFMALOADER_HPP
#define TBFFMALOADER_HPP

#include "tbfglobal.hpp"

#include <vector>
#include <array>
#include <cassert>
#include <string>
#include <istream>
#include <fstream>

template <class RealType, const long int Dim = 3, const long int DimArray = Dim>
class TbfFmaLoader {
    static_assert(Dim <= DimArray);

    const std::string filename;
    std::fstream file;

    long int nbDataPerParticle;
    std::array<RealType, Dim> centerOfBox;
    std::array<RealType, Dim> boxWidths;
    long int nbParticles;

public:
    TbfFmaLoader(std::string inFilename) :
        filename(std::move(inFilename)),
        file(filename.c_str(),std::ifstream::in),
        nbParticles(0){

        if(file.is_open()){
            unsigned int tmp1;
            file >> tmp1 >> nbDataPerParticle;
            file >> nbParticles;

            file >> boxWidths[0];
            boxWidths[0] *= 2;
            for(long int idxDim = 1 ; idxDim < Dim ; ++idxDim){
                boxWidths[idxDim] = boxWidths[0];
            }

            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                file >> centerOfBox[idxDim];
            }
        }
    }

    bool isOpen() const{
        return file.is_open();
    }

    const std::array<RealType, Dim>& getBoxCenter() const{
        return centerOfBox;
    }

    const std::array<RealType, Dim>& getBoxWidths() const{
        return boxWidths;
    }

    long int getNbParticles() const{
        return nbParticles;
    }

    auto loadOneParticle(){
        std::array<RealType, DimArray> particle;
        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            file >> particle[idxDim];
        }
        for(long int idxVal = Dim ; idxVal < std::min(DimArray,nbDataPerParticle) ; ++idxVal){
            file >> particle[idxVal];
        }
        for(long int idxVal = 0 ; idxVal < nbDataPerParticle-DimArray ; ++idxVal){
            RealType tmp;
            file >> tmp;
        }
        return particle;
    }

    auto loadAllParticles(){
        std::vector<std::array<RealType, DimArray>> particlePositions(nbParticles);
        for(long int idxPart = 0 ; idxPart < nbParticles ; ++idxPart){
            particlePositions[idxPart] = loadOneParticle();
        }

        return particlePositions;
    }

    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfFmaLoader& inLoader) {
        inStream << "TbfFmaLoader @ " << &inLoader << "\n";
        inStream << " - Filename: " << inLoader.filename << "\n";
        inStream << " - Number of particles: " << inLoader.nbParticles << "\n";
        inStream << " - Box widths: " << TbfUtils::ArrayPrinter(inLoader.boxWidths) << "\n";
        inStream << " - Center of box: " << TbfUtils::ArrayPrinter(inLoader.centerOfBox) << "\n";
        return inStream;
    }
};

#endif
