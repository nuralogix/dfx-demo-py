from setuptools import setup, find_packages

setup(
    name='dfxdemo',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'dfx-apiv2-client @ https://github.com/nuralogix/dfx-apiv2-client-py/tarball/master',
        'libdfx>=4.4',
        'cmake',
        'dlib>=19',
        'opencv-python>=4',
        'pymediainfo',
    ],
    setup_requires=['wheel'],
    description='dfxdemo is a commandline demo for NuraLogix DeepAffex™ technologies.',
    entry_points={
        'console_scripts': [
            'dfxdemo = dfxdemo:cmdline',
        ],
    },
)
