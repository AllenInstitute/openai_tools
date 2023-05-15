Before committing to the repo
========================

1. Install the local dev packages using :

```pip install -r dev_requirements.txt```

2. Check for linting by calling at the root of the repo: 

```flake8```

3. If there are any linting errors, try: 

```autopep8 --in-place --aggressive --aggressive --recursive ./```

4. If that does not fixes the errors, fix the styling manually. 

5. Run tests using :

```pytest```

6. If errors, fix them. 

Branch Policy
========================

Naming Branches
========================

All branches should be named clearly and succinctly. The name should reflect the purpose of the work being done on the branch.

Here are the naming conventions we follow:

Feature branches: feature/short-feature-description
Bugfix branches: bugfix/short-bug-description

Branch Lifespan
========================

Branches should be kept as short-lived as possible. This means you should merge branches into the main or master branch as soon as the work is complete, tested, and reviewed.

After a branch has been merged, it should be deleted to keep the repository tidy.

Merging and Pull Requests
========================

When you're ready to merge your branch, open a pull request. All code should go through a pull request review process before being merged into the main branch.

Pull requests should be reviewed by at least one other developer. This ensures that we maintain a high standard of code quality and that more than one person understands each part of our codebase.

Regularly Update from Main
========================

To avoid complex merge conflicts, regularly merge the main branch into your feature branches. This ensures that you're working with the most recent version of our codebase.

Branch Protection
========================

The main branch is protected. This means you cannot push to it directly. All changes must go through a pull request.
