from setuptools import setup
import json

with open('config.json') as f:
    jf = json.load(f)
    version = jf['CLI']['version']

setup(
    name='bothnode',
    version=version,
    install_requires=['blessed==1.20.0'],  # web3==6.16.0',   
    entry_points={
        "console_scripts": [
            "bothnode = cli:main",
        ]
    }
)
