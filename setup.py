import os
from glob import glob
from setuptools import find_packages, setup

package_name = "san_navistar_ros2"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join(
                "share",
                package_name,
                "san_navistar_ros2",
                "data",
                "navigation",
                "star",
                "checkpoints",
            ),
            glob(
                os.path.join(
                    "san_navistar_ros2/data/navigation/star/checkpoints", "*.pt"
                )
            ),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sasm",
    maintainer_email="sasilva1998@gmail.com",
    description="TODO: Package description",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "san_navistar_node = san_navistar_ros2.san_navistar_node:main"
        ],
    },
)
