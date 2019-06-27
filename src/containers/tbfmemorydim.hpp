#ifndef TBFMEMORYDIM_HPP
#define TBFMEMORYDIM_HPP

class TbfMemoryDim{
public:
    struct Scalar{};
    struct Unused{};
    struct Variable{};
    template <long int Size>
    struct Fixed{};
    struct RowMajor{};
    struct ColMajor{};
};

#endif

