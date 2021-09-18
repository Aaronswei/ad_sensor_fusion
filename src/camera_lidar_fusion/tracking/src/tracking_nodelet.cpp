

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <tracking_lib/ukf.h>

namespace tracking{

class TrackingNodelet: public nodelet::Nodelet{

public:
	TrackingNodelet() {}
	~TrackingNodelet() {}

private:
	virtual void onInit()
	{
		tracker_.reset(
			new UnscentedKF(getNodeHandle(), getPrivateNodeHandle()));
	}

	boost::shared_ptr<UnscentedKF> tracker_;
};

} // namespace sensor