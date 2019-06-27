#include "UTester.hpp"

class TestContent : public UTester< TestContent > {
    using Parent = UTester< TestContent >;
    
    void TestBasic() {
      UASSERTEEQUAL(0, 0);
    }
    
    void SetTests() {
        Parent::AddTest(&TestContent::TestBasic, "Basic test for vector");
    }
};

// You must do this
TestClass(TestContent)


