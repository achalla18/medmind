# NOTE: this module was renamed to model_pipeline.py -- a module literally
# named "pipeline.py" collides with sklearn/imblearn's internal "pipeline"
# submodule resolution and breaks the sklearn.experimental import guard.
# Use model_pipeline.py instead. This stub is kept only because files in the
# shared outputs folder cannot be deleted.
from model_pipeline import make_pipeline, make_preprocessor  # noqa: F401
