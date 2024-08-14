import contextlib
from fastapi import Depends, FastAPI
from ml import NewsgroupsModel
from schema import PredictionOutput

newgroups_model = NewsgroupsModel()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    newgroups_model.load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/prediction")
async def prediction(
    output: PredictionOutput = Depends(newgroups_model.predict),
) -> PredictionOutput:
    return output
