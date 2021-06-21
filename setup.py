from setuptools import setup, find_packages

VERSION = '0.1'

requirements = [
        "requests==2.25.1"

        # "absl-py==0.12.0",
        # "astunparse==1.6.3",
        # "cachetools==4.2.2",
        # "certifi==2021.5.30",
        # "chardet==4.0.0",
        # "click==8.0.1",
        # "cycler==0.10.0",
        # "Flask==2.0.1",
        # "flatbuffers==1.12",
        # "gast==0.4.0",
        # "google-auth==1.31.0",
        # "google-auth-oauthlib==0.4.4",
        # "google-pasta==0.2.0",
        # "grpcio==1.34.1",
        # "h5py==3.1.0",
        # "idna==2.10",
        # "itsdangerous==2.0.1",
        # "Jinja2==3.0.1",
        # "joblib==1.0.1",
        # "keras-nightly==2.5.0.dev2021032900",
        # "Keras-Preprocessing==1.1.2",
        # "kiwisolver==1.3.1",
        # "Markdown==3.3.4",
        # "MarkupSafe==2.0.1",
        # "matplotlib==3.4.2",
        # "nltk==3.6.2",
        # "numpy==1.19.5",
        # "oauthlib==3.1.1",
        # "opt-einsum==3.3.0",
        # "pandas==1.2.4",
        # "Pillow==8.2.0",
        # "protobuf==3.17.3",
        # "pyasn1==0.4.8",
        # "pyasn1-modules==0.2.8",
        # "pyparsing==2.4.7",
        # "python-dateutil==2.8.1",
        # "pytz==2021.1",
        # "regex==2021.4.4",
        # "requests==2.25.1",
        # "requests-oauthlib==1.3.0",
        # "rsa==4.7.2",
        # "scikit-learn==0.24.2",
        # "scipy==1.6.3",
        # "six==1.15.0",
        # "tensorboard==2.5.0",
        # "tensorboard-data-server==0.6.1",
        # "tensorboard-plugin-wit==1.8.0",
        # "tensorflow==2.5.0",
        # "tensorflow-estimator==2.5.0",
        # "termcolor==1.1.0",
        # "threadpoolctl==2.1.0",
        # "tqdm==4.61.0",
        # "typing-extensions==3.7.4.3",
        # "urllib3==1.26.5",
        # "Werkzeug==2.0.1",
        # "wrapt==1.12.1",
    ]


def release_version():

    setup(
        name="movie_classifier",
        packages=find_packages(),
        package_data={
            'movie_classifier.resources': ['*']},
        entry_points={
            'console_scripts': ['movie_classifier=movie_classifier.main:main'],
        },
        include_package_data=True,
        scripts=['movie_classifier/init.sh'],
        version=VERSION,
        install_requires=requirements,
        description="Movie Classifier",
        long_description_content_type="text/markdown",
        author="Daniele Moltisanti",
        author_email="danielemoltisanti@live.it",
        license="MIT Licence"
    )


if __name__ == '__main__':
    release_version()
