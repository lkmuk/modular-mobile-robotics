from setuptools import setup, find_packages

setup(
    name="mrobotics",
    fullname="Mobile Robotics Development Kit",
    author="Lai-Kan Muk",
    author_email="lkmuk2017@gmail.com",
    version="0.1",
    description="modeling for state estimation and control, data structures and algorithms",
    packages=['mrobotics.'+pkg_name for pkg_name in find_packages('mrobotics')],
    license="BSD",
    install_requires=["numpy","matplotlib"]
)
