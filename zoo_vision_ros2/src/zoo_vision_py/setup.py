from setuptools import find_packages, setup

package_name = "zoo_vision_py"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "rclpy",
        "image_transport",
        "image_transport_py",
        "rerun-sdk",
    ],
    zip_safe=True,
    maintainer="dherrera",
    maintainer_email="daniel.herrera.castro@gmail.com",
    description="TODO: Package description",
    license="GPL-3.0-only",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["rerun_forwarder = zoo_vision_py.rerun_forwarder:main"],
    },
)
