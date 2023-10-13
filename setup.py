from setuptools import setup, find_packages

setup(
    name='dfxdemo',
    version='0.21.0',
    packages=find_packages(),
    install_requires=[
        'dfx-apiv2-client>=0.13,<0.14',
        'libdfx>=4.12',
        'opencv-python>=4.6,<4.9',
    ],
    extras_require={
        "dlib": ["dlib>=19,!=19.22,!=19.22.1"],
        "mediapipe": ["mediapipe>=0.9,<0.11"],
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
