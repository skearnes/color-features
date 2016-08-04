from setuptools import setup, find_packages


def main():
    setup(
        name='color-features',
        version='0.1',
        license='3-clause BSD',
        url='https://github.com/skearnes/color-features',
        description='Utilities for using OpenEye applications and toolkits',
        packages=find_packages(),
    )

if __name__ == '__main__':
    main()
