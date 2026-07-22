"""
dual_apriltag_triangulation.launch.py

Launches:
  - usb_cam × 2
  - apriltag_ros apriltag_node × 2  (official detector)
  - apriltag_adapter_node × 2       (TF → PoseStamped bridge)
  - static_transform_publisher × 2  (world → cam1, world → cam2)
  - apriltag_triangulation_node × 1

Usage:
  ros2 launch apriltag_triangulation dual_apriltag_triangulation.launch.py \
      cam1_device:=/dev/video0 \
      cam2_device:=/dev/video2 \
      cam1_calib:=file:///home/$USER/camera_calib/ost.yaml \
      cam2_calib:=file:///home/$USER/camera_calib/ost.yaml \
      tag_size:=0.1 \
      tag_id:=0 \
      cam2_tx:='0.4040' cam2_ty:='0.0485' cam2_tz:='0.1163' \
      cam2_qx:='-0.0205' cam2_qy:='-0.2605' cam2_qz:='0.1163' cam2_qw:='0.9582'
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ── Arguments ─────────────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('cam1_device', default_value='/dev/video0'),
        DeclareLaunchArgument('cam2_device', default_value='/dev/video2'),
        DeclareLaunchArgument('cam1_calib',
            default_value='file:///home/user/camera_calib/ost.yaml'),
        DeclareLaunchArgument('cam2_calib',
            default_value='file:///home/user/camera_calib/ost.yaml'),
        DeclareLaunchArgument('tag_id',      default_value='0'),
        DeclareLaunchArgument('tag_size',    default_value='0.1'),    # metres
        DeclareLaunchArgument('tag_family',  default_value='36h11'),  # AprilTag family
        DeclareLaunchArgument('world_frame', default_value='world'),

        # cam2 extrinsic — from measure_cam2_extrinsic.py
        DeclareLaunchArgument('cam2_tx', default_value='1.0'),
        DeclareLaunchArgument('cam2_ty', default_value='0.0'),
        DeclareLaunchArgument('cam2_tz', default_value='0.0'),
        DeclareLaunchArgument('cam2_qx', default_value='0.0'),
        DeclareLaunchArgument('cam2_qy', default_value='0.0'),
        DeclareLaunchArgument('cam2_qz', default_value='0.707'),
        DeclareLaunchArgument('cam2_qw', default_value='0.707'),
    ]

    # Tag config yaml — shared by both apriltag_node instances
    tags_cfg = PathJoinSubstitution([
        FindPackageShare('apriltag_triangulation'), 'config', 'tags.yaml'])

    # ── Camera 1 ───────────────────────────────────────────────────────────────
    cam1_group = GroupAction([
        PushRosNamespace('cam1'),

        # USB camera driver
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            parameters=[{
                'video_device':        LaunchConfiguration('cam1_device'),
                'camera_info_url':     LaunchConfiguration('cam1_calib'),
                'publish_camera_info': True,
                'camera_name':         'cam1',
                'frame_id':            'cam1_optical_frame',
                'pixel_format':        'mjpeg2rgb',
            }],
            remappings=[
                ('image_raw',   '/cam1/image_raw'),
                ('camera_info', '/cam1/camera_info'),
            ],
        ),

        # apriltag_ros detection node
        # NOTE: apriltag_ros expects rectified images on image_rect
        # For uncalibrated/already-undistorted cameras use image_raw
        # IMPORTANT: the dict AFTER tags_cfg overrides 'size' from it —
        # this is what actually makes tag_size:=X on the command line work.
        # Without this second dict, apriltag_node silently keeps using
        # whatever 'size' is hardcoded in tags.yaml regardless of tag_size.
        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='apriltag_node',
            remappings=[
                ('image_rect',   '/cam1/image_raw'),
                ('camera_info',  '/cam1/camera_info'),
                ('detections',   '/cam1/apriltag/detections'),
            ],
            # CRITICAL: unique tag frame per camera. Both cameras publishing
            # the same child frame 'tag0' makes its TF parent flip-flop, and
            # lookups then silently route through world → the static launch
            # extrinsic, contaminating "raw" detections with launch values.
            # NOTE: tag.ids is hardcoded [0]; change together with tag_id.
            parameters=[tags_cfg, {
                'size': LaunchConfiguration('tag_size'),
                'tag.ids': [0],
                'tag.frames': ['tag0_cam1'],
            }],
        ),

        # Adapter: TF detections → PoseStamped topic
        Node(
            package='apriltag_triangulation',
            executable='apriltag_adapter_node',
            name='apriltag_adapter',
            parameters=[{
                'tag_id':           LaunchConfiguration('tag_id'),
                'tag_family':       LaunchConfiguration('tag_family'),
                'camera_frame':     'cam1_optical_frame',
                'detections_topic': '/cam1/apriltag/detections',
                'tag_frame':        'tag0_cam1',
            }],
            remappings=[
                ('apriltag_pose',     '/cam1/apriltag_pose'),
                ('apriltag_detected', '/cam1/apriltag_detected'),
            ],
        ),
    ])

    # ── Camera 2 (delayed 3 s) ─────────────────────────────────────────────────
    cam2_group = GroupAction([
        PushRosNamespace('cam2'),

        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            parameters=[{
                'video_device':        LaunchConfiguration('cam2_device'),
                'camera_info_url':     LaunchConfiguration('cam2_calib'),
                'publish_camera_info': True,
                'camera_name':         'cam2',
                'frame_id':            'cam2_optical_frame',
                'pixel_format':        'mjpeg2rgb',
            }],
            remappings=[
                ('image_raw',   '/cam2/image_raw'),
                ('camera_info', '/cam2/camera_info'),
            ],
        ),

        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='apriltag_node',
            remappings=[
                ('image_rect',  '/cam2/image_raw'),
                ('camera_info', '/cam2/camera_info'),
                ('detections',  '/cam2/apriltag/detections'),
            ],
            # Unique frame — see cam1 comment. NOTE: tag.ids hardcoded [0].
            parameters=[tags_cfg, {
                'size': LaunchConfiguration('tag_size'),
                'tag.ids': [0],
                'tag.frames': ['tag0_cam2'],
            }],
        ),

        Node(
            package='apriltag_triangulation',
            executable='apriltag_adapter_node',
            name='apriltag_adapter',
            parameters=[{
                'tag_id':           LaunchConfiguration('tag_id'),
                'tag_family':       LaunchConfiguration('tag_family'),
                'camera_frame':     'cam2_optical_frame',
                'detections_topic': '/cam2/apriltag/detections',
                'tag_frame':        'tag0_cam2',
            }],
            remappings=[
                ('apriltag_pose',     '/cam2/apriltag_pose'),
                ('apriltag_detected', '/cam2/apriltag_detected'),
            ],
        ),
    ])

    # ── Static TFs ─────────────────────────────────────────────────────────────
    tf_cam1 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_world_to_cam1',
        arguments=['0', '0', '0', '0', '0', '0', '1',
                   'world', 'cam1_optical_frame'],
    )

    tf_cam2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_world_to_cam2',
        arguments=[
            LaunchConfiguration('cam2_tx'), LaunchConfiguration('cam2_ty'),
            LaunchConfiguration('cam2_tz'), LaunchConfiguration('cam2_qx'),
            LaunchConfiguration('cam2_qy'), LaunchConfiguration('cam2_qz'),
            LaunchConfiguration('cam2_qw'),
            'world', 'cam2_optical_frame',
        ],
    )

    # ── Triangulation node (delayed 6 s) ───────────────────────────────────────
    triangulation_node = Node(
        package='apriltag_triangulation',
        executable='apriltag_triangulation_node',
        name='apriltag_triangulation_node',
        parameters=[{
            'world_frame':       LaunchConfiguration('world_frame'),
            'cam1_frame':        'cam1_optical_frame',
            'cam2_frame':        'cam2_optical_frame',
            'max_discrepancy_m': 0.15,
            'cam1_topic':        '/cam1/apriltag_pose',
            'cam2_topic':        '/cam2/apriltag_pose',
            'fusion_rate_hz':    20.0,
        }],
    )

    return LaunchDescription(
        args + [
            cam1_group,
            tf_cam1,
            tf_cam2,
            TimerAction(period=3.0, actions=[cam2_group]),
            TimerAction(period=6.0, actions=[triangulation_node]),
        ]
    )
