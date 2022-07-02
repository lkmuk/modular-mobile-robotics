from setuptools import setup

setup(
    name="mrobotics",
    fullname="Mobile Robotics Development Kit",
    author="Lai-Kan Muk",
    author_email="lkmuk2017@gmail.com",
    version="0.1",
    description="modeling for state estimation and control, data structures and algorithms",
    packages=["mrobotics","mrobotics.piecewise."],
    license="BSD",
    install_requires=["numpy","matplotlib"]
)
