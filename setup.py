from setuptools import setup, find_packages

setup(
    name='WaiterTipsPrediction',
    version='0.1.0',
    author='Igor Chęciński, Marcin Ciećwierz, Yasin Ozdemir',
    author_email='s24605@pjwstk.edu.pl, s25116@pjwstk.edu.pl, s22252@pejot.edu.pl',
    description='Simple project which predicts waiter tips',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["requirements.txt"]
)
