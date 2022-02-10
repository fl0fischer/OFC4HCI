from setuptools import setup

setup(name='ofc4hci',
      version='0.1.0',
      author='Florian Fischer',
      author_email='florian.j.fischer@uni-bayreuth.de',
      url='https://github.com/fl0fischer/OFC4HCI',
      license='LICENSE',
      description='This python script includes all OFC methods described in the ToCHI paper "Optimal Feedback Control for Modeling Human-Computer Interaction" (2OL-Eq, MinJerk, LQR, LQG, and E-LQG).',
      long_description=open('README.md').read(),
      python_requires='>= 3.8',
      install_requires=['numpy', 'scipy', 'pandas', 'matplotlib']
)
