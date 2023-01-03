from setuptools import setup, find_packages


if __name__ == '__main__':
    packages = find_packages(
        exclude=('images', 'results', 'scripts'),
        include=('lvae')
    )

    setup(
        name='lvae',
        version='0.1',
        description='Lossy image/video compression using variational autoencoders',
        author='Zhihao Duan',
        author_email='duan90@purdue.edu',
        url='https://github.com/duanzhiihao/lossy-vae',
        packages=packages,
    )