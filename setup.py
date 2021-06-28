import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = ['numpy', 'scikit-learn==0.24.2'] #==0.22.0

setuptools.setup(
    # required
    name="ImbalancedEstimator.refael-kohen", # only contains letters, numbers, _ , and - (Does not have to be the same as package name)
    version="0.0.2", # The format: N(.N)* for example: 0.9, 0.9.1 or X.YbN for beta version
    # optional
    author="Refael Kohen",
    author_email="refael.kohen@gmail.com",
    description="An estimator for imbalanced data",
    long_description=long_description, #  This is shown on the package detail package on the Pypi
    long_description_content_type="text/x-rst", #text/plain, text/x-rst (for reStructuredText), text/markdown
    url="https://github.com/refael-kohen/ImbalancedEstimator",

    classifiers=[ # Meta-data. Examples: https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requires, # Install dependencies
    scripts=[
        'scripts/test_package.py',
    ],

    # packages=['ImbalancedEstimator'], # You can find the packages in the directory with setuptools.find_packages()
    # Find packages under the current directory (you can specify other directory as paramter -
    # find_packages('dir_name')
    packages = setuptools.find_packages(),  # ['ImbalancedEstimator']

    # data_files=[('shell_scripts',['shell_scripts/my_shell1.sh', 'shell_scripts/my_shell2.sh']),
	# 			('docs',['docs/index.rst']),],
    #
    # package_data={
    #     # If any package contains *.txt or *.rst files, include them:
    #     "": ["*.txt", "*.rst"],
    #     # And include any *.txt files found in the "ImbalancedEstimator.sub_package1" package, too:
    #     "ImbalancedEstimator.sub_package1": ["*.txt"],
    # },
    #
    # # For sdist only:
    # # True: include only files from MANIFEST.in file (not from package_data).
    # # False: include files from MANIFEST.in file AND from package_data
    # include_package_data=False,
    #
    # tests_require=requires + ['nose2'], # Install requirements when you run: python setup.py test
    # test_suite='nose2.collector.collector',
)


