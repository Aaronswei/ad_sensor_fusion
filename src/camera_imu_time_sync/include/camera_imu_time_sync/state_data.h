#ifndef STATE_DATA_TIME_AUTOSYNC_H
#define STATE_DATA_TIME_AUTOSYNC_H

#include <Eigen/Eigen>

enum StateElements : size_t {
  STATE_TIMESTAMP,
  DELTA_T,
  OFFSET,
  NUM_STATE_ELEMENTS
};

enum MeasurementElements : size_t {
  MEASURED_TIMESTAMP,
  ANGULAR_VELOCITY,
  NUM_MEASUREMENT_ELEMENTS
};

constexpr size_t kStateElementSize = 1;
constexpr size_t kMeasurementElementSize = 1;

constexpr size_t kStateSize = kStateElementSize * NUM_STATE_ELEMENTS;
constexpr size_t kMeasurementSize =
    kMeasurementElementSize * NUM_MEASUREMENT_ELEMENTS;

#endif  // STATE_DATA_TIME_AUTOSYNC_H
