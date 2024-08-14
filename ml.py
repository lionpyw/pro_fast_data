import os, joblib
from sklearn.pipeline import Pipeline
from schema import PredictionInput, PredictionOutput


class NewsgroupsModel:
    model: Pipeline | None = None
    targets: list[str] | None = None

    def load_model(self) -> None:
        """Loads the model"""
        model_file = os.path.join(os.path.dirname(__file__), "Dataset/newsgroups_model.joblib")
        loaded_model: tuple[Pipeline, list[str]] = joblib.load(model_file)
        model, targets = loaded_model
        self.model = model
        self.targets = targets

    async def predict(self, input: PredictionInput) -> PredictionOutput:
        """Runs a prediction"""
        if not self.model or not self.targets:
            raise RuntimeError("Model is not loaded")
        prediction = self.model.predict([input.text])
        category = self.targets[prediction[0]]
        return PredictionOutput(category=category)