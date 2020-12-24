from setuptools import find_packages, setup
import os

setup(
  name='RPA 3.0',
  version='3.0.0',
  packages=find_packages(),
  include_package_data=True,
  zip_safe=False,
  install_requires=[
    'Flask',
    'Flask-SQLAlchemy',
    'flask-swagger-ui==3.25.0',
    'pika==1.1.0',
    'psycopg2==2.8.4',
    'PyJWT==1.7.1'
  ],
)
