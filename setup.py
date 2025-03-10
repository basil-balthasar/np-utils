from setuptools import setup, find_packages

setup(
    name="np_utils",
    version="0.1.0",
    author="Cyril Achard",
    author_email="cyril.achard@finalspark.com",
    description="A collection of utilities for the FinalSpark Neuroplatform",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "numba",
    ],
    extras_require={
        "all": ["matplotlib", "seaborn", "scipy", "scikit-learn", "numba", "tqdm", "joblib", "h5py", "Pillow", "plotly", "nbformat"],
        "SPL": ["plotly", "Pillow", "tqdm", "nbformat"],
        "SSG": ["scikit-learn", "matplotlib", "seaborn", "tqdm", "joblib"],
        "RRL": ["h5py"],
        "CCM": ["matplotlib", "seaborn", "numba", "tqdm"],
        "SSN": ["matplotlib", "tqdm", "seaborn", "np_utils[SPL] @ git+https://github.com/FinalSpark-np/np-utils.git"],
    },
    package_data={"np_utils": ["src/np_utils/MEA_schema.png"]},
    project_urls={
        "Homepage": "https://github.com/FinalSpark-np/np-utils",
        "Documentation": "https://finalspark-np.github.io/np-docs/welcome.html#navigation",
        "Source": "https://github.com/FinalSpark-np/np-utils",
        "Tracker": "https://github.com/FinalSpark-np/np-utils/issues",
    },
)