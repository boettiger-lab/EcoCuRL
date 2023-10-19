import setuptools

setuptools.setup(
	name="ecocurl",
	version="0.0.1",
	license="MIT",
	description="Curriculum learning tools for ecological decision problems under model uncertainty",
	author="Felipe Montealegre-Mora",
	author_email="felimomouni@gmail.com",
	url="https://github.com/boettiger-lab/EcoCuRL",
	keywords=[
		"adaptive management",
		"reinforcement learning",
		"curriculum learning",
		"theoretical ecology",
	],
	packages=setuptools.find_packages(exclude=["docs", "scripts", "tests"]),
	install_requires=[
		"numpy", "pandas", "polars", "torch", "ray[rllib,tune]", "gymnasium", 
	],
	extras_require={
		"tests": [
			# Run tests and coverage
			"pytest",
			"pytest-cov",
			"pytest-env",
			"pytest-xdist",
			# Type check
			"pytype",
			# Lint code
			"flake8>=3.8",
			# Sort imports
			"isort>=5.0",
			# Reformat
			"black",
		],
		"docs": [
			"sphinx",
			"sphinx-autobuild",
			"sphinx-rtd-theme",
			# For spelling
			"sphinxcontrib.spelling",
			# Type hints support
			"sphinx-autodoc-typehints",
		],
		"extra": ["twine"],
	},
	# Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
	classifiers=[
		"Development Status :: 3 - Alpha",
		"Intended Audience :: Developers",
		"License :: OSI Approved :: MIT License",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
	],
)