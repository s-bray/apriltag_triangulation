from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'apriltag_triangulation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='AprilTag dual-camera triangulation for ground-truth tracking',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'apriltag_adapter_node = apriltag_triangulation.apriltag_adapter_node:main',
            'apriltag_triangulation_node = apriltag_triangulation.apriltag_triangulation_node:main',
        ],
    },
)
