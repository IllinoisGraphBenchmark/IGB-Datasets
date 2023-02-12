from setuptools import setup

setup(
    name='igb',
    version='0.1.0',    
    description='IGB Graph Dataset Collection',
    url='https://github.com/IllinoisGraphBenchmark/IGB-Datasets',
    author='Arpandeep Khatua',
    author_email='akhatua2@illinois.edu',
    license='ODC-By-1.0',
    packages=['igb'],
    install_requires=['numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
    ],
)