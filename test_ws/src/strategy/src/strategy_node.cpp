#include "ros/ros.h"
#include <geometry_msgs/PoseStamped.h> //modify
#include <strategy/mobilenetv2ssd.h>

// rosrun ORB_SLAM3 Mono Vocabulary/ORBvoc.txt Examples_old/Monocular/EuRoC.yaml
static bool pose_msg_initialized = false;
static bool obj_msg_initialized = false;
static geometry_msgs::PoseStamped* pose_msg= new geometry_msgs::PoseStamped();
static strategy::mobilenetv2ssd* mobilenetv2ssd_msg= new strategy::mobilenetv2ssd();


void chatterCallback(const geometry_msgs::PoseStamped& msg)
{
  std::cout<<msg;
  
  if (!pose_msg_initialized) {
    pose_msg = new geometry_msgs::PoseStamped();
    pose_msg_initialized = true;
  }

   *pose_msg = msg;
}

void Objectdetection_Callback(const strategy::mobilenetv2ssd& msg)
{
  std::cout<<msg;
  
  if (!obj_msg_initialized) {
    mobilenetv2ssd_msg = new strategy::mobilenetv2ssd();
    obj_msg_initialized = true;
  }

   *mobilenetv2ssd_msg = msg;
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "strategy_node");
  ros::NodeHandle n;

  geometry_msgs::PoseStamped* pose_msg;

  // 確認 pose_msg 已經被正確初始化
  
  ros::Subscriber sub = n.subscribe("ORB_SLAM3/pose", 100, chatterCallback);
  ros::Subscriber sub_mobilenetv2_ssd = n.subscribe("/mobilenetv2_ssd", 100, Objectdetection_Callback);

  ROS_INFO("aaaa");
  while(ros::ok() && pose_msg_initialized)
  {
    ROS_INFO("pose_msg.position.x = %f\n",pose_msg->pose.position.x);
  }
  ROS_INFO("vvvvv");
  ros::spin();
  return 0;

}
