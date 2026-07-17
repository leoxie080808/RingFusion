from setuptools import setup

package_name = 'ringfusion_perception'

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
    description='Mono depth + ToF anchoring perception',
    license='MIT',
    entry_points={
        'console_scripts': [
            'perception = ringfusion_perception.perception_node:main',
        ],
    },
)
