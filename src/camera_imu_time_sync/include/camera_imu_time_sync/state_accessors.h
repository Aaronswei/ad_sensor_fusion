#ifndef STATE_ACCESSORS_TIME_AUTOSYNC_H
#define STATE_ACCESSORS_TIME_AUTOSYNC_H

#include <Eigen/Eigen>

#include "camera_imu_time_sync/state_data.h"

// convince functions for referring to elements in vectors and matrices
template <size_t element_height, size_t element_width, typename Derived>
Eigen::Block<Derived, element_height, element_width> access(
    Eigen::MatrixBase<Derived>& mat, const size_t element_y,
    const size_t element_x = 0) {
  const size_t y = element_y * element_height;
  const size_t x = element_x * element_width;

  assert((y + element_height) < mat.rows());
  assert((x + element_width) < mat.cols());

  return Eigen::Block<Derived, element_height, element_width>(mat.derived(), y,
                                                              x);
}

template <size_t element_height, size_t element_width, typename Derived>
const Eigen::Block<const Derived, element_height, element_width> access(
    const Eigen::MatrixBase<Derived>& mat, const size_t element_y,
    const size_t element_x = 0) {
  const size_t y = element_y * element_height;
  const size_t x = element_x * element_width;

  assert((y + element_height) < mat.rows());
  assert((x + element_width) < mat.cols());

  return Eigen::Block<const Derived, element_height, element_width>(
      mat.derived(), y, x);
}

template <typename Derived>
Eigen::Block<Derived, kMeasurementElementSize, 1> accessM(
    Eigen::MatrixBase<Derived>& mat, const size_t element) {
  return access<kMeasurementElementSize, 1, Derived>(mat, element, 0);
}

template <typename Derived>
Eigen::Block<const Derived, kMeasurementElementSize, 1> accessM(
    const Eigen::MatrixBase<Derived>& mat, const size_t element) {
  return access<kMeasurementElementSize, 1, Derived>(mat, element, 0);
}

template <typename Derived>
Eigen::Block<Derived, kMeasurementElementSize, kMeasurementElementSize> accessM(
    Eigen::MatrixBase<Derived>& mat, const size_t element_y,
    const size_t element_x) {
  return access<kMeasurementElementSize, kMeasurementElementSize, Derived>(
      mat, element_y, element_x);
}

template <typename Derived>
Eigen::Block<const Derived, kMeasurementElementSize, kMeasurementElementSize>
accessM(const Eigen::MatrixBase<Derived>& mat, const size_t element_y,
        const size_t element_x) {
  return access<kMeasurementElementSize, kMeasurementElementSize, Derived>(
      mat, element_y, element_x);
}

template <typename Derived>
Eigen::Block<Derived, kStateElementSize, 1> accessS(
    Eigen::MatrixBase<Derived>& mat, const size_t element) {
  return access<kStateElementSize, 1, Derived>(mat, element, 0);
}

template <typename Derived>
Eigen::Block<const Derived, kStateElementSize, 1> accessS(
    const Eigen::MatrixBase<Derived>& mat, const size_t element) {
  return access<kStateElementSize, 1, Derived>(mat, element, 0);
}

template <typename Derived>
Eigen::Block<Derived, kStateElementSize, kStateElementSize> accessS(
    Eigen::MatrixBase<Derived>& mat, const size_t element_y,
    const size_t element_x) {
  return access<kStateElementSize, kStateElementSize, Derived>(mat, element_y,
                                                               element_x);
}

template <typename Derived>
Eigen::Block<const Derived, kStateElementSize, kStateElementSize> accessS(
    const Eigen::MatrixBase<Derived>& mat, const size_t element_y,
    const size_t element_x) {
  return access<kStateElementSize, kStateElementSize, Derived>(mat, element_y,
                                                               element_x);
}

#endif  // STATE_ACCESSORS_TIME_AUTOSYNC_H
