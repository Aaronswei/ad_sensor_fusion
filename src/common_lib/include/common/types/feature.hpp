/*
 * Copyright (C) 2019 by AutoSense Organization. All rights reserved.
 * Gary Chan <chenshj35@mail2.sysu.edu.cn>
 */
#ifndef COMMON_LIBS_INCLUDE_COMMON_TYPES_FEATURE_HPP_
#define COMMON_LIBS_INCLUDE_COMMON_TYPES_FEATURE_HPP_

#include <ros/ros.h>
#include <string>
#include <vector>

namespace autosense {

typedef double FeatureElementType;
typedef double Label;

struct FeatureElement {
    FeatureElement(std::string feature_name, FeatureElementType feature_value)
        : name(feature_name), value(feature_value) {}

    std::string name = "";
    FeatureElementType value = 0.0;
};

/// \brief A feature can be composed of any number of values, each with its own
/// name.
class Feature {
 public:
    Feature() {}

    ~Feature() {}

    explicit Feature(const FeatureElement& element) { push_back(element); }

    explicit Feature(const std::string& name, const FeatureElementType& value) {
        push_back(FeatureElement(name, value));
    }

    void push_back(const FeatureElement& element) {
        if (findValueByName(element.name, NULL)) {
            ROS_WARN_STREAM(
                "Adding several FeatureValues of same name to Feature is not "
                "recommended.");
        }
        feature_elements_.push_back(element);
    }

    bool setValueById(unsigned int dimesion_id,
                      const FeatureElementType& value) {
        feature_elements_.at(dimesion_id).value = value;
    }

    size_t size() const { return feature_elements_.size(); }

    bool empty() { return feature_elements_.empty(); }

    const FeatureElement& at(const size_t& index) const {
        return feature_elements_.at(index);
    }

    void clear() { feature_elements_.clear(); }

 protected:
    bool findValueByName(const std::string& name, FeatureElement* value) const {
        for (size_t i = 0u; i < feature_elements_.size(); ++i) {
            if (feature_elements_.at(i).name == name) {
                if (value != NULL) {
                    *value = feature_elements_.at(i);
                }
                return true;
            }
        }
        return false;
    }

 private:
    std::vector<FeatureElement> feature_elements_;
};

}  // namespace autosense

#endif  // COMMON_LIBS_INCLUDE_COMMON_TYPES_FEATURE_HPP_
