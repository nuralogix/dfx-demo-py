from setuptools import setup, find_packages

setup(
    name='dfxdemo',
    version='0.25.0',
    packages=find_packages(),
    install_requires=[
        'dfx-apiv2-client>=0.15,<0.16',
        'libdfx>=4.14',
    ],
    extras_require={
        "dlib": ["dlib>=19,!=19.22,!=19.22.1", "opencv-python>=4.6,<5"],
        "mediapipe": ["mediapipe>=0.9,<0.11"],
        "visage": ["libvisage", "opencv-python>=4.6,<5"],
    },
    setup_requires=['cmake', 'wheel'],
    description='dfxdemo is a commandline demo for NuraLogix DeepAffex™ technologies.',
    entry_points={
        'console_scripts': [
            'dfxdemo = dfxdemo.dfxdemo:cmdline',
            "dfxextract = dfxdemo.dfxextract:cmdline"
        ],
    },
)
