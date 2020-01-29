#ifndef TBFTIMER_HPP
#define TBFTIMER_HPP

#include <chrono>

class TbfTimer {
    using double_second_time = std::chrono::duration<double, std::ratio<1, 1>>;

    std::chrono::high_resolution_clock::time_point startTime;  ///< startTime time (start)
    std::chrono::high_resolution_clock::time_point endTime;  ///< stop time (stop)
    std::chrono::nanoseconds cumulateTime;  ///< the cumulateTime time

public:
    /// Constructor
    TbfTimer() : cumulateTime(std::chrono::nanoseconds::zero()) { start(); }

    /// Copy constructor
    TbfTimer(const TbfTimer& other) = default;
    /// Copies an other timer
    TbfTimer& operator=(const TbfTimer& other) = default;
    /// Move constructor
    TbfTimer(TbfTimer&& other) = delete;
    /// Copies an other timer
    TbfTimer& operator=(TbfTimer&& other) = delete;

    /** Rest all the values, and apply start */
    void reset() {
        startTime = std::chrono::high_resolution_clock::time_point();
        endTime = std::chrono::high_resolution_clock::time_point();
        cumulateTime = std::chrono::nanoseconds::zero();
        start();
    }

    /** Start the timer */
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    /** Stop the current timer */
    void stop() {
        endTime = std::chrono::high_resolution_clock::now();
        cumulateTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
    }

    /** Return the elapsed time between start and stop (in second) */
    double getElapsed() const {
        return std::chrono::duration_cast<double_second_time>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime)).count();
    }

    /** Return the total counted time */
    double getCumulated() const {
        return std::chrono::duration_cast<double_second_time>(cumulateTime).count();
    }

    /** End the current counter (stop) and return the elapsed time */
    double stopAndGetElapsed() {
        stop();
        return getElapsed();
    }

    /** Add a timer to another one */
    void merge(const TbfTimer& inOther) {
        cumulateTime = inOther.cumulateTime;
    }
};

#endif
