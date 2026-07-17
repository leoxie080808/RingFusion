from setuptools import setup

package_name = 'ringfusion_drivers'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Leo Xie',
    maintainer_email='ly4xie@uwaterloo.ca',
    description='ToF hub and Arducam camera driver nodes',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tof_driver = ringfusion_drivers.tof_driver_node:main',
            'camera = ringfusion_drivers.camera_node:main',
        ],
    },
)
