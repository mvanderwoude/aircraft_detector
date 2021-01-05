import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
    "torch>=1.3.0",
    "matplotlib",
    "numpy",
    "pandas",
    "paramiko",
    "sklearn",
    "librosa",
]

setuptools.setup(
    name="aircraft_detector",
    version="0.0.1",
    author="Mark van der Woude",
    author_email="mark_woude@hotmail.com",
    description="Package for sound-based aircraft detection using MAVs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mvanderwoude/aircraft_detector",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires=">=3.6",
)
