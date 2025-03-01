import setuptools

__version__       =  "0.0.1"

REPO_NAME         =  "CompanyReviewGlassdoorProject"
AUTHOR_NAME       =  "KemoAI"
SRC_REPO          =  "CompanyReview"
AUTHOR_EMAIL      =  "kemo_attlanta@hotmail.com"

with open("README.md" , "r" , encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name             = SRC_REPO ,
    version          = __version__ ,
    author           = AUTHOR_NAME ,
    author_email     = AUTHOR_EMAIL ,
    description      = "Review Companyies Using Employee Sentiments Project",
    long_description = long_description , 
    long_description_content = "text/markdown" ,
    url = f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}" ,
    project_urls = {
        "Bug Tracker": f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}/issues",
    },
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where = "src")

     )