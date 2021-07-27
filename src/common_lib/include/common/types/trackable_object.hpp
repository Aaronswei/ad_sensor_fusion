/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#ifndef COMMON_LIBS_INCLUDE_COMMON_TYPES_TRACKABLE_OBJECT_HPP_
#define COMMON_LIBS_INCLUDE_COMMON_TYPES_TRACKABLE_OBJECT_HPP_

#include <memory>
#include "common/geometry.hpp"
#include "common/object.hpp"
#include "common/type.h"

namespace autosense {

// TODO(gary): 引入“锚点”观测
struct TrackableObject {
    /* NEED TO NOTICE: All the states of track would be collected mainly based
     * on
     * the states of tracked object. Thus, update tracked object's state when
     * you
     * update the state of track !!! */
    TrackableObject() = default;

    /**
     * @brief init Trackable Object from built Object
     *  inherit built object's ground center/size/direction
     *  compute barycenter/anchor_point
     *  init velocity/acceleration/velocity_uncertainty
     * @param obj_ptr
     */
    explicit TrackableObject(ObjectPtr obj_ptr) : object_ptr(obj_ptr) {
        if (object_ptr != nullptr) {
            ground_center = object_ptr->ground_center.cast<float>();
            size = Eigen::Vector3f(object_ptr->length, object_ptr->width,
                                   object_ptr->height);
            direction = object_ptr->direction.cast<float>();

            // 初始化重心
            barycenter =
                common::geometry::getCloudBarycenter<PointI>(object_ptr->cloud)
                    .cast<float>();

            // TODO(gary): need HD Map
            // lane_direction = Eigen::Vector3f::Zero();

            /**
             * @brief initial state
             * @note bary center as anchor point
             */
            // 重心作为锚点
            anchor_point = barycenter;
            velocity = Eigen::Vector3f::Zero();
            acceleration = Eigen::Vector3f::Zero();

            velocity_uncertainty = Eigen::Matrix3d::Identity() * 5;

            type = object_ptr->type;
        }
    }

    /**
     * @brief deep copy of Trackable object
     * @param rhs
     */
    void clone(const TrackableObject& rhs) {
        *this = rhs;
        object_ptr.reset(new Object());
        object_ptr->clone(*rhs.object_ptr);
    }

    // store transformed object before tracking, cloud...
    ObjectPtr object_ptr;
    // ground center and size
    Eigen::Vector3f ground_center;
    Eigen::Vector3f size;
    Eigen::Vector3f direction;
    // 重心
    Eigen::Vector3f barycenter;
    // TODO(gary): lane direction needs HD Map
    // Eigen::Vector3f lane_direction;
    // states 每个追踪器追踪的物体状态估计(锚点+速度+加速度)
    Eigen::Vector3f anchor_point;
    Eigen::Vector3f velocity;
    Eigen::Matrix3d velocity_uncertainty;
    Eigen::Vector3f acceleration;
    // class type
    ObjectType type;
    // association distance, range from 0 to association_score_maximum
    float association_score = 0.0f;
};  // struct TrackableObject

// 追踪物体
typedef std::shared_ptr<TrackableObject> TrackableObjectPtr;
typedef std::shared_ptr<const TrackableObject> TrackableObjectConstPtr;

}  // namespace autosense

#endif  // COMMON_LIBS_INCLUDE_COMMON_TYPES_TRACKABLE_OBJECT_HPP_
