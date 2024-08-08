import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hf_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob(os.path.join('share', package_name, "*.json"))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nate',
    maintainer_email='nathanbcorral@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'maskformer = models.segmentation.maskformer.Maskformer:main',
            'detr = models.obj_det.detr.DETR:main',
            'viz = viz_utils.VizNode:main',

            'model_node = models.ModelNode:main',

            'german_traffic_signs_dataset = models.BBoxImageDatasetPublisher:main',
        ],
    },
)
