#include "lidar_radar_fusion.hpp"
#include "ekfInterface.hpp"

using namespace lidar_radar_fusion;

LidarRadarFusion::LidarRadarFusion() {
    _lidar_objects.clear();
    _radar_objects.clear();
    _lidar_radar_match.clear();
    _ekf_api = std::make_shared<EKF_API>();
    
}

LidarRadarFusion::~LidarRadarFusion() {

}

bool LidarRadarFusion::Init(const std::map<TimeStamp, std::vector<ObjectInfo> >& lidar_objects,
                            const std::map<TimeStamp, std::vector<ObjectInfo> >& radar_objects) {
    if(lidar_objects.size() == radar_objects.size()) {
        return false;
    }

    _lidar_objects = lidar_objects;
    _radar_objects = radar_objects;
    _is_initialized = true;
    return true;
}

bool LidarRadarFusion::Run() {
    if(!_is_initialized) {
        return false;
    }

    std::map<TimeStamp, std::vector<ObjectInfo> >::const_iterator lidar_iter = _lidar_objects.begin();
    std::map<TimeStamp, std::vector<ObjectInfo> >::const_iterator radar_iter = _radar_objects.begin();
    for(; lidar_iter != _lidar_objects.end(), radar_iter != _radar_objects.end();lidar_iter++,radar_iter++) {
        MatchWith3D(lidar_iter->second,radar_iter->second);
    }

    std::vector<ObjectInfo> filtered_list_temp;
    std::vector<Measurement> measurement_pack_list;

    for(const auto &point : _filtered_list)
    {

        Measurement temp;
		temp.raw_measurements_ = VectorXd(2);
        temp.raw_measurements_ << point.position[1], point.position[0];
        temp.timestamp_ = 100; //ms
        measurement_pack_list.push_back(temp); 
    }
    _ekf_api->process(measurement_pack_list);

    return true;
}

void LidarRadarFusion::MatchWith3D(const std::vector<ObjectInfo>& lidar_objects,
                                   const std::vector<ObjectInfo>& radar_objects) {
	std::vector<int> lidar2radar(lidar_objects.size(), -1);
    std::vector<int> radar2lidar(radar_objects.size(), -1);
    Match(lidar_objects,radar_objects,lidar2radar,radar2lidar);

    std::map<int,int>::const_iterator iter = _lidar_radar_match.begin();

    for(;iter != _lidar_radar_match.end();iter++) {
        ObjectInfo fuse_object;
        fuse_object.time_stamp = lidar_objects[iter->second].time_stamp;
        fuse_object.position = 0.5 * lidar_objects[iter->second].position + 0.5 * radar_objects[iter->first].position;
        _filtered_list.emplace_back(fuse_object);
    }
}

void LidarRadarFusion::Match(const std::vector<ObjectInfo>& lidar_objects,
                             const std::vector<ObjectInfo>& radar_objects,
                             std::vector<int>& lidar2radar, 
                             std::vector<int>& radar2lidar) {
    if(lidar_objects.empty() || radar_objects.empty()) {
        return;
    }

    Eigen::MatrixXd match_scores_matrix = Eigen::MatrixXd::Zero( lidar_objects.size(),radar_objects.size());
    for(size_t i = 0; i<lidar_objects.size(); i++) {
        for(size_t j = 0; j<radar_objects.size(); j++) {
            match_scores_matrix(i,j) = CalcuProjectScore(radar_objects[j], lidar_objects[i]);
        }
    }

    HungarianMatcher(match_scores_matrix, lidar2radar, radar2lidar);
}

void LidarRadarFusion::HungarianMatcher(Eigen::MatrixXd &scores,
                                        std::vector<int> &lidar2radar, 
                                        std::vector<int> &radar2lidar,
					                    double thres) {
    Eigen::MatrixXd::Index row;
	Eigen::MatrixXd::Index col;
    while (scores.maxCoeff(&row, &col) >= thres) {
        int idx1 = row;
        int idx2 = col;
        scores.row(idx1).setZero();
        scores.col(idx2).setZero();
        lidar2radar[idx1] = idx2;
        radar2lidar[idx2] = idx1;
        _lidar_radar_match.insert(std::make_pair(idx2,idx1));
	}

}


float LidarRadarFusion::CalcuProjectScore(const ObjectInfo& lidar_object, const ObjectInfo& radar_object) {
	double longitudinal_dis = abs(lidar_object.position[0] - radar_object.position[0]);
	double longitudinal_dis_procent = longitudinal_dis / std::max(lidar_object.position[0], radar_object.position[0]);
	if (longitudinal_dis > 3.5 && longitudinal_dis_procent > 0.25) { //3.5m,25%
		return 0.0;
	}

	double proj_dist = (lidar_object.position.normalized() * lidar_object.position.normalized().dot(radar_object.position) - radar_object.position).norm();
	if (proj_dist < 3.0) {
		return (3.0 - proj_dist) / 3.0;
	}
    return 0.0;
}

std::vector<ObjectInfo> LidarRadarFusion::GetFilteredObjectPoints() {
    return _filtered_list;
}
std::vector<TrackObject> LidarRadarFusion::GetTrackedObjectPoints() {
    return _tracked_list;
}