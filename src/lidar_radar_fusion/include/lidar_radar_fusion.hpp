#ifndef __LIDAR_RADAR_FUSION_HPP__
#define __LIDAR_RADAR_FUSION_HPP__
#include "utils.hpp"
#include "ekfInterface.hpp"


namespace lidar_radar_fusion {
    class LidarRadarFusion {
    public:    
        LidarRadarFusion();
        ~LidarRadarFusion();

        bool Init(const std::map<TimeStamp, std::vector<ObjectInfo> >& lidar_objects,
                  const std::map<TimeStamp, std::vector<ObjectInfo> >& radar_objects);
        bool Run();
        std::vector<TrackObject> GetTrackedObjectPoints();
        std::vector<ObjectInfo> GetFilteredObjectPoints();

    private:
        float CalcuProjectScore(const ObjectInfo& lidar_object,
                                const ObjectInfo& radar_object);
        void MatchWith3D(const std::vector<ObjectInfo>& lidar_objects,
                         const std::vector<ObjectInfo>& radar_objects);
        void Match(const std::vector<ObjectInfo>& lidar_objects,
                             const std::vector<ObjectInfo>& radar_objects,
                             std::vector<int>& lidar2radar, 
                             std::vector<int>& radar2lidar);
        void HungarianMatcher(Eigen::MatrixXd &scores, 
                            std::vector<int> &t2d, std::vector<int> &d2t,
					        double thres = 0.2);

        bool _is_initialized = false;
        std::shared_ptr<EKF_API> _ekf_api;// = new EKF_API();

        std::map<TimeStamp, std::vector<ObjectInfo> > _lidar_objects;
        std::map<TimeStamp, std::vector<ObjectInfo> > _radar_objects;
       
        std::map<int , int > _lidar_radar_match;
        std::vector<TrackObject> _tracked_list;
        std::vector<ObjectInfo> _filtered_list;
        int _object_id = -1;
        

    };
}

#endif // __LIDAR_RADAR_FUSION_HPP__