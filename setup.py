from setuptools import setup, find_packages

# Python >= 3.12 is required by pandas-ta 0.4.71b0, which pins numba 0.61.2.
# Without this, installing on 3.11 fails with a numpy resolution error that
# points nowhere near the real cause.
setup(
    name="trading_agent",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.12",
)
