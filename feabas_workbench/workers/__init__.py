"""
Subprocess workers. Each module has a ``main()`` that reads a JSON spec file
(``--spec``), prints ``##PROGRESS {...}`` lines and, on success, one
``##RESULT {...}`` line. They run in the FEABAS or deep-learning environment,
so they must only import feabas_workbench.core modules that have light
dependencies (numpy, tifffile, cv2, yaml).
"""
