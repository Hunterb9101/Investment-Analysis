import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='invest',
    version='0.0.1',
    author='Hunter Boles',
    author_email='hunterb9101@gmail.com',
    description='An investment metric package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 1-Planning"
    ],
    python_requires='>=3.6'
)
