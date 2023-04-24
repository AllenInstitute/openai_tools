from setuptools import setup,find_packages    

with open("README.md", encoding="utf-8") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
    name="papers_extractor",
    description="Extract summaries and reviews from papers using AI",
    long_description_content_type="text/x-rst",
    long_description=readme,
    author="Jerome Lecoq",
    version='0.1.0',
    author_email="jeromel@alleninstitute.org",
    url="https://github.com/AllenInstitute/openai_tools",
    license=license,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,
)