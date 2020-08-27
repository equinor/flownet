# Contributing

The following is a set of guidelines for contributing to FlowNet.

## Ground Rules

1. We use PEP8
1. We use Black code formatting
1. We use Pylint
1. We document our code
1. We use type annotations

## Pull Request Process

1. Work on your own fork of the main repo
1. Push your commits and make a draft pull request using the pull request template; follow the
   instructions in the template.
1. Check that your pull request passes all tests.
1. When all tests have passed and your are happy with your changes, change your pull request to "ready for review"
   and ask for a code review.
1. When your code has been approved - merge your changes.

### Test data

Test data is available at https://github.com/equinor/flownet-testdata.

Note that if you do changes to the source code that needs corresponding
changes in the configuration file, you should make a pull request to the
`flownet-testdata` repository as well, and temporarily change the environment variables
in the [CI GitHub workflow](./.github/workflows/flownet.yml) to make sure
CI downloads your branch during initial automatic testing.

Good practice is to merge the PR on `flownet-testdata` first, and then change the workflow
back to official `equinor/master` in the main FlowNet PR.

### Build documentation

You can build the documentation after installation by running
```bash
cd ./docs
make html
```
and then open the generated `./docs/_build/html/index.html` in a browser.
