#ifndef TBFTEMPLATE_HPP
#define TBFTEMPLATE_HPP

namespace TbfTemplate
{

template <long int index_t>
struct Index{
    static constexpr long int index = index_t;

    constexpr operator long int() const{
        return index_t;
    }
};

template <long int index_t>
auto make_index(){
    return Index<index_t>();
}

template <long int startIndex, long int limiteIndex, long int step, class Func>
auto For(Func&& func){
    if constexpr (startIndex < limiteIndex){
        func(make_index<startIndex>());
        For<startIndex+step , limiteIndex, step>(std::forward<Func>(func));
    }
}

template <long int startIndex, long int limiteIndex, long int step, class Func>
auto If(const long int testIndex, Func&& func){
    if constexpr (startIndex < limiteIndex){
        if(startIndex == testIndex){
            func(make_index<startIndex>());
        }
        else{
            If<startIndex+step , limiteIndex, step>(testIndex, std::forward<Func>(func));
        }
    }
}

}
#endif
