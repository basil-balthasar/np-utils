from setuptools import setup, find_packages
import toml

# Read pyproject.toml
with open("pyproject.toml", "r") as f:
    pyproject = toml.load(f)

project = pyproject["project"]

setup(
    name=project["name"],
    version=project["version"],
    author=project["authors"][0]["name"],
    author_email=project["authors"][0]["email"],
    description=project["description"],
    long_description=open(project["readme"]).read(),
    long_description_content_type="text/markdown",
    license=project["license"]["file"],
    classifiers=project["classifiers"],
    python_requires=project["requires-python"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=project["dependencies"],
    package_data={
        "np_utils": ["src/np_utils/MEA_schema.png"],
    },
    extras_require=project["optional-dependencies"],
    project_urls=project["urls"],
)