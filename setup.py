from setuptools import setup, find_packages

setup(
    name='dfxdemo',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'dfx-apiv2-client @ https://github.com/nuralogix/dfx-apiv2-client-py/tarball/master',
        'libdfx>=4.9',
        'dlib>=19,!=19.22,!=19.22.1',
        'opencv-python>=4.5,<4.6'
    ],
    setup_requires=['cmake', 'wheel'],
    description='dfxdemo is a commandline demo for NuraLogix DeepAffex™ technologies.',
    entry_points={
        'console_scripts': [
            'dfxdemo = dfxdemo.dfxdemo:cmdline',
        ],
    },
)
