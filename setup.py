from distutils.core import setup

setup(
    name='confmap',
    version='0.0.1',
    author='Ruslan Guseinov',
    description='Conformal mapping algorithms CETM and BFF.',
    url='https://github.com/russelmann/confmap',
    python_requires='>=3.8',
    install_requires=['numpy', 'scipy'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    py_modules=[],
    test_suite='tests',
)
