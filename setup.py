from setuptools import setup
import json

with open('config.json') as f:
    jf = json.load(f)
    version = jf['CLI']['version']

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
    ],
    entry_points={
        "console_scripts": [
            "bothnode = cli:main",
        ]
    }
)
