from setuptools import setup, find_packages

setup(
    name='geoconformal',
    version='0.1.2',
    description='Geographically Weighted Conformal Prediction Methods',
    author='Peng Luo',
    author_email='pengluo@mit.edu',
    url='https://github.com/pengluo/geoconformal',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
