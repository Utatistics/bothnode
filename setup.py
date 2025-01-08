import os
import json
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension

with open('config.json') as f:
    jf = json.load(f)
    version = jf['CLI']['version']

class CMakeBuild(build_ext):
    def run(self):
        subprocess.check_call(['cmake', '.'])
        subprocess.check_call(['make'])
        super().run()
        
setup(
    name='bothnode',
    version=version,
    install_requires=[
        'blessed==1.20.0'
        ,'colorlog'
        ,'fastapi'
        ,'uvicorn'
        ,'pymongo'
        ,'PyYAML'
        ,'eth-abi'
        ,'pandas'
        ,'matplotlib'
        ,'torch==2.4'  # Ensure torch 2.4 is installed
        ,'dgl @ https://data.dgl.ai/wheels-test/torch-2.4/repo.html' # Adding the specific DGL install link
        ,'pyod'
        ,'pybind11'
    ],
    ext_modules=[],  # You no longer need this since CMake handles the build
    cmdclass={'build_ext': CMakeBuild},  # Use the custom CMake build command
    entry_points={
        "console_scripts": [
            "bothnode = cli:main",
        ]
    },
)
