from setuptools import setup, find_packages

setup(
    name='dfxdemo',
    version='0.20.0',
    packages=find_packages(),
    install_requires=[
        'dfx-apiv2-client>=0.13,<0.14',
        'libdfx>=4.9',
        'opencv-python>=4.5,<4.7',
    ],
    extras_require={
        "dlib": ["dlib>=19,!=19.22,!=19.22.1"],
        "mediapipe": ["mediapipe>=0.9,<0.10"],
        "visage": ["libvisage"],
    },
    setup_requires=['cmake', 'wheel'],
    description='dfxdemo is a commandline demo for NuraLogix DeepAffex™ technologies.',
    entry_points={
        'console_scripts': [
            'dfxdemo = dfxdemo.dfxdemo:cmdline',
        ],
    },
)
