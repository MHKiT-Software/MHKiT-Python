# ![](logo.png) MHKiT-Python

<p align="left">
    <a href=https://github.com/MHKiT-Software/MHKiT-Python/actions/workflows/main.yml>
        <img src="https://github.com/MHKiT-Software/MHKiT-Python/actions/workflows/main.yml/badge.svg">
    </a>
    <a href=https://coveralls.io/github/MHKiT-Software/MHKiT-Python?branch=main>
        <img src="https://coveralls.io/repos/github/MHKiT-Software/MHKiT-Python/badge.svg?branch=main">
    </a>
    <a href=https://pepy.tech/project/mhkit>
        <img src="https://pepy.tech/badge/mhkit">
    </a>
    <a href=https://doi.org/10.5281/zenodo.3924683>
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3924683.svg">
    </a>
</p>

MHKiT-Python is a Python package designed for marine energy applications to assist in
data processing and visualization. The software package include functionality for:

- Data processing
- Data visualization
- Data quality control
- Resource assessment
- Device performance
- Device loads

## Documentation

MHKiT-Python documentation includes overview information, installation instructions, API documentation, and examples.
See the [MHKiT documentation](https://mhkit-software.github.io/MHKiT) for more information.

## Installation

[MHKiT-Python](https://github.com/MHKiT-Software/MHKiT-Python) requires [Python (3.10-3.12)](https://www.python.org/).
It is recommended to use the [Anaconda Python Distribution](https://www.anaconda.com/distribution/) (a fully featured Python installer with a GUI)
or [Miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install) (a lightweight installer with the `conda` command line utility).  
Both will include most of MHKiT-Python's package dependencies.

MHKiT can be installed several ways:

### Option 1: Install With Anaconda/Miniconda

This option is recommended for most MHKiT-Python users.
To install MHKiT-Python using `conda`, in an Anaconda Prompt:

```bash
conda env create mhkit-env --python=3.11
```

```bash
conda activate mhkit-env
```

```bash
conda install -c conda-forge mhkit
```

Optional: Installing dependencies to run mhkit examples and development dependencies:

```bash
pip install mhkit["examples"]
```

Note: To use the above installed version of MHKiT-Python users must activate the `mhkit-env` environment each time, using `conda activate
mhkit-env` in each new shell/terminal to use MHKiT.
To avoid this, users can install MHKiT into their base conda environment, but this is not
recommended as it may cause dependency conflicts with other software.

Visual Studio Code has [instructions for using Python environments in VS Code](https://code.visualstudio.com/docs/python/environments) that support conda environment discovery.

### Option 2: Clone Repository from GitHub

This option is recommended for MHKiT-Python users who want access to example notebooks and developers.
Download and install your preferred version of [git](https://git-scm.com/).
To clone MHKiT-Python:

```bash
git clone https://github.com/MHKiT-Software/MHKiT-Python
```

```bash
cd MHKiT-Python
```

#### Virtual Environment Setup

A virtual environment is a self-contained directory that contains a Python installation for a
particular version of Python, plus a number of additional packages. Using a virtual environment
allows you to manage dependencies for different projects separately, avoiding conflicts between
packages and ensuring that your project has access to the specific versions of packages it needs.

Use of a virtual environment is recommended to avoid dependency conflicts with other python
packages (this environment must be activated in each new shell/terminal before using MHKiT).:

##### Python `venv` (built into python)

Python venv (built into python)

Windows

Note: A supported version of Python ([see installation for supported versions](#installation)) must be installed and added to the system PATH for venv to work in Git Bash or WSL. Use of Git Bash or WSL is recommended for Windows users to avoid issues with activating the virtual environment in Command Prompt or PowerShell.

```sh
python -m venv mhkit-env
```

```sh
.\mhkit-env\Scripts\activate
```

Linux/MacOS:

```bash
python -m venv mhkit-env
```

```bash
source mhkit-env/bin/activate
```

##### Conda (requires separate installation of Anaconda or Miniconda):

```bash
conda create -n mhkit-env python=3.11
```

```bash
conda activate mhkit-env
```

#### Install MHKiT-Python with pip

To install a local, editable version of MHKiT-Python using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install -e .["all"]
```

An [environment YAML file](https://github.com/MHKiT-Software/MHKiT-Python/blob/main/environment.yml) is also provided that can create the base environment required by MHKiT.
MHKiT can then be installed into that environment using any of the provided methods.

### Option 3: Module-specific Install from Python

A slim version of MHKiT-Python can be installed to reduce the number of dependencies and potential conflicts with other software.
This installation utilizes pip's optional dependencies installation.

Note: Use of a virtual environment is recommended to avoid dependency conflicts with other python.
See Option 2 installation instructions for virtual environment setup.

To install a single MHKiT module, e.g. the wave module, and its dependencies, use:

```
pip install mhkit["wave"]
```

Note that `pip install mhkit` only installs the base MHKiT dependencies and not the entire software.
To install all MHKiT dependencies use:

```
pip install mhkit["all"]
```

See [installation instructions](https://mhkit-software.github.io/MHKiT/installation.html) for more information.

### Development Installation

For developers contributing to MHKiT, there are three development installation strategies after
cloning the repository locally:

Note: Use of a virtual environment is recommended to avoid dependency conflicts with other python.
See Option 2 installation instructions for virtual environment setup.

**Pip development** (no conda):

```bash
pip install -e ".[all,dev]"
```

**Conda development** (minimal conda + pip resolves deps):

```bash
conda env create -f environment.yml
```

```bash
conda activate mhkit-env
```

```bash
pip install -e ".[all,dev]"
```

**Conda-forge development** (all deps from conda-forge, mirrors production deployment):

```bash
conda env create -f environment-dev.yml
conda activate mhkit-env
pip install -e ".[all,dev]" --no-deps
```

The conda-forge option mirrors how users install MHKiT via `conda install -c conda-forge mhkit`, ensuring all dependencies come from the conda-forge channel. The `--no-deps` flag prevents pip from resolving dependencies, relying entirely on the conda-forge packages for dependencies. The conda-forge build and deployment happens in separate repository: [https://github.com/conda-forge/mhkit-feedstock] which is updated with each MHKiT release.

## Copyright and license

MHKiT-Python is copyright through the National Laboratory of the Rockies,
Pacific Northwest National Laboratory, and Sandia National Laboratories.
The software is distributed under the Revised BSD License.
See [copyright and license](LICENSE.md) for more information.

## Issues

The GitHub platform has the Issues feature that is used to track ideas, feedback, tasks, and/or bugs. To submit an Issue, follow the steps below. More information about GitHub Issues can be found [here](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues)

1. Navigate to the [MHKiT-Python main page](https://github.com/MHKiT-Software/MHKiT-Python)
2. 2.Under the repository name (upper left), click **Issues**.
3. Click **New Issue**.
4. If the Issue is a bug, use the **Bug report** template and click **Get started**, otherwise click on the **Open a blank issue** link.
5. Provide a **Title** and **description** for the issue. Be sure the title is relevant to the issue and that the description is clear and provided with sufficient detail.
6. When you're finished, click **Submit new issue**. The developers will follow-up once the issue is addressed.

## Creating a fork

The GitHub platform has the Fork feature that facilitates code modification and contributions. A fork is a new repository that shares code and visibility settings with the original upstream repository. To fork MHKiT-Python, follow the steps below. More information about GitHub Forks can be found [here](https://docs.github.com/en/get-started/quickstart/fork-a-repo)

1. Navigate to the [MHKiT-Python main page](https://github.com/MHKiT-Software/MHKiT-Python)
2. Under the repository name (upper left), click **Fork**.
3. Select an owner for the forked repository.
4. Specify a name for the fork. By default, forks are named the same as their upstream repositories.
5. Add a description of your fork (optional).
6. Choose whether to copy only the default branch or all branches to the new fork. You will only need copy the default branch to contribute to MHKiT-Python.
7. When you're finished, click **Create fork**. You will now have a fork of the MHKiT-Python repository.

## Creating a branch

The GitHub platform has the branch feature that facilitates code contributions and collaboration amongst developers. A branch isolates development work without affecting other branches in the repository. Each repository has one default branch, and can have multiple other branches. To create a branch of your forked MHKiT-Python repository, follow the steps below. More information about GitHub branches can be found [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches)

1. Navigate to your fork of MHKiT-Python (see instructions above)
2. Above the list of files, click **Branches**.
3. Click **New Branch**.
4. Enter a name for the branch. Be sure to select **MHKiT-Software/MHKiT-Python:main** as the source.
5. Click **Create branch**. You will now have a branch on your fork of MHKiT-Python that you can use to work with the code base.

## Creating a pull request

The GitHub platform has the pull request feature that allows you to propose changes to a repository such as MHKiT-Python. The pull request will allow the repository administrators to evaluate the pull request. To create a pull request for MHKiT-Python repository, follow the steps below. More information about GitHub pull requests can be found [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)

1. Navigate to the [MHKiT-Python main page](https://github.com/MHKiT-Software/MHKiT-Python)
2. Above the list of files, click **Pull request**.
3. On the compare page, click **Compare accross forks**.
4. In the "base branch" drop-down menu, select the branch of the upstream repository you'd like to merge changes into.
5. In the "head fork" drop-down menu, select your fork, then use the "compare branch" drop-down menu to select the branch you made your changes in.
6. Type a title and description for your pull request.
7. If you want to allow anyone with push access to the upstream repository to make changes to your pull request, select **Allow edits from maintainers**.
8. To create a pull request that is ready for review, click **Create Pull Request**. To create a draft pull request, use the drop-down and select **Create Draft Pull Request**, then click **Draft Pull Request**. More information about draft pull requests can be found [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests)
9. MHKiT-Python adminstrators will review your pull request and contact you if needed.

## Code Formatting in MHKiT

MHKiT adheres to the "black" code formatting standard to maintain a consistent and readable code style. Developers contributing to MHKiT have several options to ensure their code meets this standard:

1. **Manual Formatting with Black**: Install the 'black' formatter and run it manually from the terminal to format your code. This can be done by executing a command like `black [file or directory]`.

2. **IDE Extension**: If you are using an Integrated Development Environment (IDE) like Visual Studio Code (VS Code), you can install the 'black' formatter as an extension. This allows for automatic formatting of code within the IDE.

3. **Pre-Commit Hook**: Enable the pre-commit hook in your development environment. This automatically formats your code with 'black' each time you make a commit, ensuring that all committed code conforms to the formatting standard.

For detailed instructions on installing and using 'black', please refer to the [Black Documentation](https://black.readthedocs.io/en/stable/). This resource provides comprehensive guidance on installation, usage, and configuration of the formatter.
