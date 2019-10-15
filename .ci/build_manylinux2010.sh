python -m pip install --upgrade pip setuptools

pip install -e "/io/.[tests,doc]"
pip uninstall -y giotto-learn
pip install wheel twine

pytest --cov /io/giotto/ --cov-report xml
flake8 --exit-zero /io/
displayName: 'Test with pytest and flake8'

python /io/setup.py sdist bdist_wheel
