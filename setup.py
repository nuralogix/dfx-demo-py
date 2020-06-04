from setuptools import setup, find_packages

setup(
    name='dfxpydemo',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'dfx-apiv2-client @ https://github.com/nuralogix/dfx-apiv2-client-py/tarball/master',
        'libdfx>=4',
        'cmake',
        'dlib>=19',
        'opencv-python>=4',
    ],
    setup_requires=['wheel'],
    description='dfxpydemo is a commandline demo for NuraLogix DeepAffex technologies.',
    entry_points={
        'console_scripts': [
            'dfxpydemo = dfxpydemo:cmdline',
        ],
    },
)
