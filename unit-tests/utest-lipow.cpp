#include "UTester.hpp"

#include "utils/tbfutils.hpp"

class TestLipow : public UTester< TestLipow > {
    using Parent = UTester< TestLipow >;
    
    void TestBasic() {
      UASSERTEEQUAL(TbfUtils::lipow(1,0), 1L);
      UASSERTEEQUAL(TbfUtils::lipow(1,1), 1L);
      UASSERTEEQUAL(TbfUtils::lipow(1,2), 1L);

      UASSERTEEQUAL(TbfUtils::lipow(3,0), 1L);
      UASSERTEEQUAL(TbfUtils::lipow(3,1), 3L);
      UASSERTEEQUAL(TbfUtils::lipow(3,2), 3L*3);
      UASSERTEEQUAL(TbfUtils::lipow(3,3), 3L*3*3);
    }

    void TestLoop() {
        for(long int idxValue = 0 ; idxValue < 10 ; ++idxValue){
            for(long int idxPow = 0 ; idxPow < 99 ; ++idxPow){
                long int res = 1;
                for(long int idx = 0 ; idx < idxPow ; ++idx){
                    res *= idxValue;
                }

                UASSERTEEQUAL(TbfUtils::lipow(idxValue,idxPow), res);
            }
        }
    }
    
    void SetTests() {
        Parent::AddTest(&TestLipow::TestBasic, "Basic test for lipow");
        Parent::AddTest(&TestLipow::TestLoop, "Loop test for lipow");
    }
};

// You must do this
TestClass(TestLipow)


