from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['ssd'],#这里你要引用的pkg目录
    package_dir={'': 'src'},#包含这个文件夹的目录比如这里我是放到src目录下
)
setup(**setup_args)

