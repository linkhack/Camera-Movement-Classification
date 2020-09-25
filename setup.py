import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="camclassifier-cvl", # Replace with your own username
    version="0.0.1",
    author="Patrick Link",
    author_email="e11728332@student.tuwien.ac.at",
    description="Package to classify camera movement in historical videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/linkhack/Camera-Movement-Classification",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        'tensorflow-gpu==2.0.3',
        'keras',
        'pyyaml',
        'scikit-image',
        'opencv-python',
    ],
    python_requires='>=3.6',

)