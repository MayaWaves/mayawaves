.. Mayawaves documentation master file, created by
   sphinx-quickstart on Thu May  6 10:28:27 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Contributor's Guide
=====================================

`Mayawaves` is intended to be a library that will grow and expand with additional analysis tools. We welcome the
input of the community to both request new features and help in implementing them. In order to make this process smooth, we use the fork-merge
workflow.

In this workflow, you will create a fork of the repository.
You will then create a feature branch and make all of your changes on that branch.
Do not make any changes to your main branch.
You may then create a merge request to merge your feature branch into the primary repository.
This branch can then be merged into main.

Make a fork
-----------
The steps in this section only need to be done once.

Click the "fork" button at https://github.com/MayaWaves/mayawaves

Then clone your fork to your machine using::

    git clone git@github.com:<your_workspace>/mayawaves.git

Link your clone to the primary repository upstream::

    cd mayawaves
    git remote add upstream git@github.com:MayaWaves/mayawaves.git

Setup main branch to track main branch of the primary repository::

    git checkout main
    git fetch upstream
    git branch --set-upstream-to upstream/main
    git pull

Keeping your fork up to date
----------------------------
To keep the main branch of your fork up to date, regularly do the following::

    git checkout main
    git pull
    git push origin main


Making changes
--------------
Make all your changes in a branch on your fork.

Create a new branch with the following::

    git checkout main
    git pull
    git checkout -b new-feature-name upstream/main

Make any desired changes, committing frequently and using descriptive commit messages.
When you are ready to do your first push to your branch on your fork, run::

    git push --set-upstream origin new-feature-name

All future pushes can be done with::

    git push

To keep your branch up to date with the main branch, periodically do the following::

    git checkout main
    git pull
    git checkout new-feature-name
    git rebase upstream/main
    git push -f origin new-feature-name

Running tests
^^^^^^^^^^^^^
To ensure that all existing functionality is not broken, any changes need to pass the test suite.
To run the tests, run the following from the main directory::

    python -m unittest discover tests

Please add tests to the suite for any features you add and ensure they also pass.

Updating the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^
To update the documentation with any changes you've made, do the following from the main directory::

    cd docs
    rm -r html doctrees
    make html

Submitting a merge request
^^^^^^^^^^^^^^^^^^^^^^^^^^
Once you are happy with your changes, you can create a merge request that will be reviewed by the MAYA developer team.
