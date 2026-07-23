"""
triple_apriltag_triangulation.launch.py

Three-camera version. cam1 = world origin; cam2 and cam3 each carry an
extrinsic (from measure_extrinsics_multi). Cameras start staggered
(0 s / 3 s / 6 s) to avoid USB stream-open collisions; computation nodes
start at 9 s.

NOTE on USB bandwidth: three identical webcams on ONE USB controller may
not fit even with mjpeg2rgb. If cam3 dies with "Unable to start stream",
move it to a different controller (different physical port group / add-in
card) or lower its resolution/framerate.

Usage:
  ros2 launch apriltag_triangulation triple_apriltag_triangulation.launch.py \
      cam1_device:=/dev/video0 cam2_device:=/dev/video2 cam3_device:=/dev/video4 \
      cam1_calib:=file:///.../cam1/ost.yaml \
      cam2_calib:=file:///.../cam2/ost.yaml \
      cam3_calib:=file:///.../cam3/ost.yaml \
      tag_size:=0.15 tag_id:=0 \
      cam2_tx:=... cam2_qw:=...  cam3_tx:=... cam3_qw:=... \
      geo_baseline_override:=0.59
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def make_cam_group(idx: int, tags_cfg):
    """usb_cam + apriltag_node + adapter for camera <idx>."""
    ns = f'cam{idx}'
    frame = f'cam{idx}_optical_frame'
    tag_frame = f'tag0_cam{idx}'
    return GroupAction([
        PushRosNamespace(ns),
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            parameters=[{
                'video_device':        LaunchConfiguration(f'cam{idx}_device'),
                'camera_info_url':     LaunchConfiguration(f'cam{idx}_calib'),
                'publish_camera_info': True,
                'camera_name':         ns,
                'frame_id':            frame,
                'pixel_format':        'mjpeg2rgb',
            }],
            remappings=[
                ('image_raw',   f'/{ns}/image_raw'),
                ('camera_info', f'/{ns}/camera_info'),
            ],
        ),
        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='apriltag_node',
            remappings=[
                ('image_rect',  f'/{ns}/image_raw'),
                ('camera_info', f'/{ns}/camera_info'),
                ('detections',  f'/{ns}/apriltag/detections'),
            ],
            # Unique tag frame per camera — shared frames make the tag's
            # TF parent flip-flop and leak the launch extrinsics into raw
            # measurements. tag.ids hardcoded [0]; change with tag_id.
            parameters=[tags_cfg, {
                'size': LaunchConfiguration('tag_size'),
                'tag.ids': [0],
                'tag.frames': [tag_frame],
            }],
        ),
        Node(
            package='apriltag_triangulation',
            executable='apriltag_adapter_node',
            name='apriltag_adapter',
            parameters=[{
                'tag_id':           LaunchConfiguration('tag_id'),
                'camera_frame':     frame,
                'detections_topic': f'/{ns}/apriltag/detections',
                'tag_frame':        tag_frame,
            }],
            remappings=[
                ('apriltag_pose',     f'/{ns}/apriltag_pose'),
                ('apriltag_detected', f'/{ns}/apriltag_detected'),
            ],
        ),
    ])


def static_tf(name, args_prefix, child):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=name,
        arguments=[
            LaunchConfiguration(f'{args_prefix}_tx'),
            LaunchConfiguration(f'{args_prefix}_ty'),
            LaunchConfiguration(f'{args_prefix}_tz'),
            LaunchConfiguration(f'{args_prefix}_qx'),
            LaunchConfiguration(f'{args_prefix}_qy'),
            LaunchConfiguration(f'{args_prefix}_qz'),
            LaunchConfiguration(f'{args_prefix}_qw'),
            'world', child,
        ],
    )


def generate_launch_description():

    args = [
        DeclareLaunchArgument('cam1_device', default_value='/dev/video0'),
        DeclareLaunchArgument('cam2_device', default_value='/dev/video2'),
        DeclareLaunchArgument('cam3_device', default_value='/dev/video4'),
        DeclareLaunchArgument('cam1_calib',
            default_value='file:///home/ros/ws/src/camera_calibrations/camera_calib/ost.yaml'),
        DeclareLaunchArgument('cam2_calib',
            default_value='file:///home/ros/ws/src/camera_calibrations/camera_calib2/ost.yaml'),
        DeclareLaunchArgument('cam3_calib',
            default_value='file:///home/ros/ws/src/camera_calibrations/camera_calib3/ost.yaml'),
        DeclareLaunchArgument('tag_id',   default_value='0'),
        DeclareLaunchArgument('tag_size', default_value='0.15'),
        DeclareLaunchArgument('world_frame', default_value='world'),
        DeclareLaunchArgument('max_age_sec', default_value='1.0'),
        DeclareLaunchArgument('geo_baseline_override', default_value='0.0'),

        # cam2 extrinsic
        DeclareLaunchArgument('cam2_tx', default_value='0.5'),
        DeclareLaunchArgument('cam2_ty', default_value='0.0'),
        DeclareLaunchArgument('cam2_tz', default_value='0.0'),
        DeclareLaunchArgument('cam2_qx', default_value='0.0'),
        DeclareLaunchArgument('cam2_qy', default_value='0.0'),
        DeclareLaunchArgument('cam2_qz', default_value='0.0'),
        DeclareLaunchArgument('cam2_qw', default_value='1.0'),
        # cam3 extrinsic
        DeclareLaunchArgument('cam3_tx', default_value='-0.5'),
        DeclareLaunchArgument('cam3_ty', default_value='0.0'),
        DeclareLaunchArgument('cam3_tz', default_value='0.0'),
        DeclareLaunchArgument('cam3_qx', default_value='0.0'),
        DeclareLaunchArgument('cam3_qy', default_value='0.0'),
        DeclareLaunchArgument('cam3_qz', default_value='0.0'),
        DeclareLaunchArgument('cam3_qw', default_value='1.0'),
    ]

    tags_cfg = PathJoinSubstitution([
        FindPackageShare('apriltag_triangulation'), 'config', 'tags.yaml'])

    cam1 = make_cam_group(1, tags_cfg)
    cam2 = make_cam_group(2, tags_cfg)
    cam3 = make_cam_group(3, tags_cfg)

    tf_cam1 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_world_to_cam1',
        arguments=['0', '0', '0', '0', '0', '0', '1',
                   'world', 'cam1_optical_frame'],
    )
    tf_cam2 = static_tf('tf_world_to_cam2', 'cam2', 'cam2_optical_frame')
    tf_cam3 = static_tf('tf_world_to_cam3', 'cam3', 'cam3_optical_frame')

    fusion = Node(
        package='apriltag_triangulation',
        executable='multi_apriltag_fusion_node',
        name='multi_apriltag_fusion_node',
        parameters=[{
            'n_cams':            3,
            'world_frame':       LaunchConfiguration('world_frame'),
            'max_discrepancy_m': 0.08,
            'fusion_rate_hz':    20.0,
            'max_age_sec':       LaunchConfiguration('max_age_sec'),
        }],
    )

    geometric = Node(
        package='apriltag_triangulation',
        executable='multi_geometric_triangulation_node',
        name='multi_geometric_triangulation_node',
        parameters=[{
            'n_cams':            3,
            'tag_id':            LaunchConfiguration('tag_id'),
            'world_frame':       LaunchConfiguration('world_frame'),
            'max_age_sec':       LaunchConfiguration('max_age_sec'),
            'baseline_override': LaunchConfiguration('geo_baseline_override'),
        }],
    )

    return LaunchDescription(
        args + [
            cam1,
            tf_cam1, tf_cam2, tf_cam3,
            TimerAction(period=3.0, actions=[cam2]),
            TimerAction(period=6.0, actions=[cam3]),
            TimerAction(period=9.0, actions=[fusion, geometric]),
        ]
    )
