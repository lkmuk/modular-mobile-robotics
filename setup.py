from setuptools import setup, find_packages

pkg_name_main = "mrobotics"


setup(
    name="mrobotics",
    fullname="Mobile Robotics Development Kit",
    author="Lai-Kan Muk",
    author_email="lkmuk2017@gmail.com",
    version="0.1",
    description="modeling for state estimation and control, data structures and algorithms",
    packages=[pkg_name_main, *[pkg_name_main+"."+pkg_name for pkg_name in find_packages(pkg_name_main)]],
    license="BSD",
    install_requires=["numpy","matplotlib"]
)
