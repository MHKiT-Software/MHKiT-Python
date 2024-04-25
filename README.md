# ![](logo.png) MHKiT-Python

<p align="left">
    <a href=https://github.com/MHKiT-Software/MHKiT-Python/actions/workflows/main.yml>
        <img src="https://github.com/MHKiT-Software/MHKiT-Python/actions/workflows/main.yml/badge.svg">
    </a>
    <a href=https://coveralls.io/github/MHKiT-Software/MHKiT-Python?branch=master>
        <img src="https://coveralls.io/repos/github/MHKiT-Software/MHKiT-Python/badge.svg?branch=master">
    </a>
    <a href=https://pepy.tech/project/mhkit>
        <img src="https://pepy.tech/badge/mhkit">
    </a>
    <a href=https://doi.org/10.5281/zenodo.3924683>
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3924683.svg">
    </a>
</p>

MHKiT-Python is a Python package designed for marine renewable energy applications to assist in
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

MHKiT-Python requires Python (3.8, 3.9, 3.10, 3.11) along with several Python
package dependencies. MHKiT-Python can be installed from PyPI using the command:

`pip install mhkit`

See [installation instructions](https://mhkit-software.github.io/MHKiT/installation.html) for more information.

## Copyright and license

MHKiT-Python is copyright through the National Renewable Energy Laboratory,
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
4. Enter a name for the branch. Be sure to select **MHKiT-Software/MHKiT-Python:master** as the source.
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
