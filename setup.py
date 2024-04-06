from setuptools import setup
import json

with open('config.json') as f:
    jf = json.load(f)
    version = jf['CLI']['version']

setup(
    name='bothnode',
    version=version,
    install_requires=[
        'blessed',
        'web3'
        ],
    entry_points={
        "console_scripts": [
            "bothnode = cli:main",

        ]
    }
)
