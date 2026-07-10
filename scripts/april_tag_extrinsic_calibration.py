#!/usr/bin/env python3

"""
measure_cam2_extrinsic_apriltag.py

Compute stereo camera extrinsic calibration using an AprilTag.

Assumptions:
- Both cameras see the same AprilTag.
- AprilTag detector publishes PoseStamped:
    /cam1/apriltag_pose
    /cam2/apriltag_pose

The script computes:

    T_cam1_cam2 = T_cam1_tag * inv(T_cam2_tag)

which gives camera 2 pose relative to camera 1.

Tag:
    family: tag36h11
    id: 0

"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped

from scipy.spatial.transform import Rotation as R

import numpy as np



def pose_to_matrix(pose):

    """
    Convert geometry_msgs/Pose to homogeneous transform matrix.
    """

    t = np.array([
        pose.position.x,
        pose.position.y,
        pose.position.z
    ])

    q = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ]

    rot = R.from_quat(q).as_matrix()

    T = np.eye(4)

    T[:3, :3] = rot
    T[:3, 3] = t

    return T



class ExtrinsicMeasurer(Node):

    def __init__(self):

        super().__init__('apriltag_extrinsic_calibrator')


        self.declare_parameter(
            'n_samples',
            200
        )


        self.n_samples = (
            self.get_parameter('n_samples')
            .get_parameter_value()
            .integer_value
        )


        self.samples = []


        self.latest_cam1 = None
        self.latest_cam2 = None



        self.sub_cam1 = self.create_subscription(
            PoseStamped,
            '/cam1/apriltag_pose',
            self.cam1_callback,
            10
        )


        self.sub_cam2 = self.create_subscription(
            PoseStamped,
            '/cam2/apriltag_pose',
            self.cam2_callback,
            10
        )


        self.timer = self.create_timer(
            0.1,
            self.collect_samples
        )


        self.get_logger().info(
            f"""
AprilTag Stereo Extrinsic Calibration

Tag:
    family : tag36h11
    id     : 0

Collecting:
    {self.n_samples} samples

Move the tag slightly or keep it stable.
Both cameras must see the tag.
"""
        )



    def cam1_callback(self,msg):

        self.latest_cam1 = msg



    def cam2_callback(self,msg):

        self.latest_cam2 = msg




    def collect_samples(self):

        if self.latest_cam1 is None:
            return

        if self.latest_cam2 is None:
            return


        T_cam1_tag = pose_to_matrix(
            self.latest_cam1.pose
        )

        T_cam2_tag = pose_to_matrix(
            self.latest_cam2.pose
        )


        # Camera 2 transform relative to camera 1

        T_cam1_cam2 = (
            T_cam1_tag @
            np.linalg.inv(T_cam2_tag)
        )


        self.samples.append(
            T_cam1_cam2
        )


        n = len(self.samples)


        self.get_logger().info(
            f"Sample {n}/{self.n_samples}"
        )


        if n >= self.n_samples:

            self.compute_result()

            rclpy.shutdown()



    def compute_result(self):


        translations = []
        rotations = []


        for T in self.samples:

            translations.append(
                T[:3,3]
            )

            rotations.append(
                R.from_matrix(
                    T[:3,:3]
                ).as_quat()
            )


        translations = np.array(
            translations
        )

        rotations = np.array(
            rotations
        )


        # Translation averaging

        t_mean = np.mean(
            translations,
            axis=0
        )


        t_std = np.std(
            translations,
            axis=0
        )


        # Quaternion averaging

        q_ref = rotations[0]


        for i in range(len(rotations)):

            if np.dot(
                q_ref,
                rotations[i]
            ) < 0:

                rotations[i] *= -1


        q_mean = np.mean(
            rotations,
            axis=0
        )

        q_mean /= np.linalg.norm(
            q_mean
        )



        baseline = np.linalg.norm(
            t_mean
        )


        print("\n" + "="*60)

        print(
            "AprilTag Stereo Extrinsic Calibration Result"
        )

        print("="*60)


        print(
            f"""
Translation cam2 relative cam1:

x = {t_mean[0]:.6f}
y = {t_mean[1]:.6f}
z = {t_mean[2]:.6f}


Translation std:

x = {t_std[0]:.6f}
y = {t_std[1]:.6f}
z = {t_std[2]:.6f}


Quaternion:

qx = {q_mean[0]:.6f}
qy = {q_mean[1]:.6f}
qz = {q_mean[2]:.6f}
qw = {q_mean[3]:.6f}


Camera baseline:

{baseline:.6f} meters

"""
        )


        print(
            "static_transform_publisher command:"
        )


        print(
            f"""
ros2 run tf2_ros static_transform_publisher \\
{t_mean[0]:.6f} {t_mean[1]:.6f} {t_mean[2]:.6f} \\
{q_mean[0]:.6f} {q_mean[1]:.6f} {q_mean[2]:.6f} {q_mean[3]:.6f} \\
cam1_optical_frame cam2_optical_frame
"""
        )


        print("="*60)




def main():

    rclpy.init()

    node = ExtrinsicMeasurer()

    rclpy.spin(node)



if __name__ == '__main__':

    main()
