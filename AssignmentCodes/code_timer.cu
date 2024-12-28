#include <stdio.h>
#include <sys/time.h>

// Global variables to store start and end times
struct timeval startTime, endTime;

/**
 * @brief Starts the timer by recording the current time.
 *
 * This function captures the current time and stores it in a global variable for later
 * use in calculating the elapsed time. It should be called at the beginning of the
 * sequence of code you wish to time.
 */
void startTimer() {
    gettimeofday(&startTime, NULL);
}

/**
 * @brief Stops the timer by recording the current time.
 *
 * Similar to startTimer, this function captures the current time at the point where the
 * timer should stop. This end time is used in conjunction with the start time to calculate
 * the elapsed time.
 */
void endTimer() {
    gettimeofday(&endTime, NULL);
}

/**
 * @brief Prints the elapsed time since the timer was started.
 *
 * After startTimer and endTimer have been called, this function calculates and prints
 * the elapsed time in seconds and microseconds. It uses the global variables set by
 * the previous two functions.
 */
void printTime() {
    long seconds = (endTime.tv_sec - startTime.tv_sec);
    long micros = ((seconds * 1000000) + endTime.tv_usec) - (startTime.tv_usec);
    
    // Adjusting the calculation to correctly handle the boundary case
    if (endTime.tv_usec < startTime.tv_usec) {
        seconds -= 1;
        micros = (seconds * 1000000) + (1000000 + endTime.tv_usec) - startTime.tv_usec;
    }

    printf("Time elapsed is %ld seconds and %ld microseconds\n", seconds, micros);
}
