#ifndef TBFINTERACTION_HPP
#define TBFINTERACTION_HPP

#include "tbfglobal.hpp"

template <class IndexType_T>
struct TbfXtoXInteraction{
    using IndexType = IndexType_T;

    IndexType indexTarget;
    IndexType indexSrc;

    long int globalTargetPos;
    long int arrayIndexSrc;


    static bool SrcFirst(const TbfXtoXInteraction& i1, const TbfXtoXInteraction i2){
        return i1.indexSrc < i2.indexSrc
                || (i1.indexSrc == i2.indexSrc && i1.indexTarget < i2.indexTarget);
    }
};

#endif
