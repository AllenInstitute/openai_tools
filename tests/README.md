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
