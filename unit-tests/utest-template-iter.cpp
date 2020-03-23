#include "UTester.hpp"

#include "utils/tbfutils.hpp"
#include "utils/tbftemplate.hpp"


class TestTemplate : public UTester< TestTemplate > {
    using Parent = UTester< TestTemplate >;

    void TestBasic() {
        {
            int counter = 0;
            TbfTemplate::For<0, 5, 1>([&counter](const auto index){
                static_assert (0 <= static_cast<long int>(index)
                        && static_cast<long int>(index) < 5, "Test index");
                static_assert (0 <= index.index
                        && index.index < 5, "Test index");
                counter += 1;
            });
            UASSERTETRUE(counter == 5);
        }
        {
            int counter = 0;
            const int testIndex = 1;
            TbfTemplate::If<0, 5, 1>(testIndex, [&counter, this, testIndex](const auto index){
                static_assert (0 <= static_cast<long int>(index)
                        && static_cast<long int>(index) < 5, "Test index");
                static_assert (0 <= index.index
                        && index.index < 5, "Test index");
                UASSERTETRUE(testIndex == index.index);
                counter += 1;
            });
            UASSERTETRUE(counter == 1);
        }
    }


    void SetTests() {
        Parent::AddTest(&TestTemplate::TestBasic, "Basic test for template feature");
    }
};

// You must do this
TestClass(TestTemplate)


