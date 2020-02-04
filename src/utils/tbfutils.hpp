#ifndef TBFUTILS_HPP
#define TBFUTILS_HPP

#include <cassert>
#include <type_traits>
#include <utility>
#include <iterator>
#include <array>

namespace TbfUtils {
template <bool...>
struct all_true; // UNDEFINED

template <>
struct all_true<>
  : std::true_type {};

template <bool... Conds>
struct all_true<false, Conds...>
  : std::false_type {};

template <bool... Conds>
struct all_true<true, Conds...>
  : all_true<Conds...> {};

// See; https://blog.tartanllama.xyz/exploding-tuples-fold-expressions/

template <class BlockType>
struct for_each_type{
    using BlockTypeT = BlockType;
};

template <std::size_t... Idx>
inline auto make_index_dispatcher(std::index_sequence<Idx...>) {
    return [] (auto&& f) { (f(std::integral_constant<std::size_t,Idx>{}), ...); };
}

template <typename Tuple, typename Func>
inline void for_each(Func&& f) {
    auto dispatcher = make_index_dispatcher(std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
    dispatcher([&f](auto idx) { f(for_each_type<typename std::tuple_element<idx, Tuple>::type>(), idx); });
}

template <class DataType>
inline constexpr long int GetLeadingDim(const long int inNbItems, const long int MemoryAlignementBytes){
    const long int size = sizeof(DataType)*inNbItems;
    return ((size + MemoryAlignementBytes - 1)/MemoryAlignementBytes)*MemoryAlignementBytes;
}


template <class VecClass, class ScalarClass>
inline typename std::decay<VecClass>::type AddToVec(VecClass&& inVec, ScalarClass&& inScalar){
    typename std::decay<VecClass>::type dest = std::forward<VecClass>(inVec);

    for(auto& val : dest){
        val += inScalar;
    }

    return dest;
}

template <class VecClass1, class VecClass2>
inline typename std::decay<VecClass1>::type AddVecToVec(VecClass1&& inVec1, VecClass2&& inVec2){
    assert(std::size(inVec1) == std::size(inVec2));

    typename std::decay<VecClass1>::type dest = std::forward<VecClass1>(inVec1);

    auto iterDest = std::begin(dest);
    const auto endDest = std::end(dest);
    auto iter2 = std::begin(inVec2);

    while(iterDest != endDest){
        (*iterDest) += (*iter2);
        ++iterDest;
        ++iter2;
    }

    return dest;
}

template <class VecClass, class ScalarClass>
inline typename std::decay<VecClass>::type MulToVec(VecClass&& inVec, ScalarClass&& inScalar){
    typename std::decay<VecClass>::type dest = std::forward<VecClass>(inVec);

    for(auto& val : dest){
        val *= inScalar;
    }

    return dest;
}

inline long int lipow(long int val, const long int power){
    long int res = 1;

    long incPower = val;

    for(long int idx = 1 ; idx <= power ; idx <<= 1){
        if(idx & power){
            res *= incPower;
        }
        incPower *= incPower;
    }

    return res;
}

template <class ObjectType>
static const ObjectType& make_const(ObjectType& obj){
    return obj;
}

template <class ArrayType>
class ArrayPrinterCore{
    const ArrayType& array;

public:
    ArrayPrinterCore(const ArrayType& inArray) : array(inArray){}

    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const ArrayPrinterCore<ArrayType>& inArray) {
        inStream << "std::" << typeid(ArrayType).name() << " @ " << &inArray;
        inStream << " - Size " << std::size(inArray.array);
        inStream << " - Data { ";
        for(long int idx = 0 ; idx < static_cast<long int>(std::size(inArray.array)) ; ++idx){
            inStream << inArray.array[idx];
            if(idx != static_cast<long int>(std::size(inArray.array))-1){
                inStream << ",";
            }
        }
        inStream << "}";
        return inStream;
    }
};

template <class ArrayType>
inline ArrayPrinterCore<ArrayType> ArrayPrinter(const ArrayType& inArray){
    return ArrayPrinterCore<ArrayType>(inArray);
}


template <class Type>
inline Type* CreateNew(Type&& inObject){
    return new Type(std::forward<Type>(inObject));
}


template <class Type, size_t Size, size_t... OtherSizes>
struct marray_core
{
  using SubArray = typename marray_core<Type, OtherSizes...>::type;
  using type = std::array<SubArray, Size>;
};

template <class Type, size_t Size>
struct marray_core<Type, Size>
{
  using type = std::array<Type, Size>;
};

template <class Type, size_t... AllSizes>
using marray = typename marray_core<Type, AllSizes...>::type;


template<class T, class Compare>
long int lower_bound_indexes(long int first, const long int last, const T& value, Compare comp)
{
    long int count = (first - last);

    while (count > 0) {
        long int it = first;
        long int step = count / 2;
        it += step;
        if (comp(it, value)) {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;
    }
    return first;
}

}

#endif
