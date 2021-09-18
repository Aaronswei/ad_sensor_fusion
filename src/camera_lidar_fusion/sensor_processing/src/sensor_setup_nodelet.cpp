#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <sensor_processing_lib/sensor_fusion.h>

namespace sensor_processing{

class SensorSetupNodelet: public nodelet::Nodelet{
public:
  SensorSetupNodelet() {}
  ~SensorSetupNodelet() {}

private:
  virtual void onInit()
  {
    sensor_fusion_.reset(
      new SensorFusion(getNodeHandle(), getPrivateNodeHandle()));
  }

  boost::shared_ptr<SensorFusion> sensor_fusion_;
};

} // namespace sensor