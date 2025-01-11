import os
import json
import subprocess
from pathlib import Path
from setuptools import setup
from setuptools.command.build_ext import build_ext

import logging
from logging import getLogger

logger = getLogger(__name__)

# Load version from config.json
with open('config.json') as f:
    jf = json.load(f)
    version = jf['CLI']['version']

class CMakeBuild(build_ext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.path_to_cmake = Path('./backend/cpp')

    def run(self):
        """invoked by build_ext. 
        """
        subprocess.check_call(['cmake', self.path_to_cmake.__str__()])

        try:
            subprocess.check_call(['make', '-j', '4'])
        except subprocess.CalledProcessError as e:
            logger.error(f"Error occurred during make: {e}")
            raise

        super().run()
        self.cleanup_files()

    def cleanup_files(self):
        """Cleans up temporary files and directories created during the build
        """
        logger.info("Deleting the intermediary file objects.")
        paths = ['CMakeCache.txt', 'Makefile', 'cmake_install.cmake', 'CMakeFiles']
        for path in paths:
            if os.path.exists(path):
                logger.info(f"Removing {'directory' if os.path.isdir(path) else 'file'}: {path}")
                subprocess.check_call(['rm', '-rf', path])

setup(
    name='bothnode',
    version=version,
    install_requires=[
        'blessed==1.20.0',
        'colorlog',
        'fastapi',
        'uvicorn',
        'pymongo',
        'PyYAML',
        'eth-abi',
        'pandas',
        'matplotlib',
        'torch==2.4',
        'dgl @ https://data.dgl.ai/wheels-test/torch-2.4/repo.html',
        'pyod',
        'pybind11',
    ],
    cmdclass={'build_ext': CMakeBuild},  # Custom CMake build command
    entry_points={
        "console_scripts": [
            "bothnode = cli:main",
        ]
    },
)
