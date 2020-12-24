import setuptools

#with open("./README.md", "r") as fh:
 #   long_description = fh.read()

setuptools.setup(
    name="axa.rpa.dti",
    version="0.0.1",
    author="AXA DTI Team",
    author_email="eikden.yeoh@asia-assistance.com",
    description="An AI engine package to simplify business process.",
    long_description="AI Automation",
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

