"""API endpoints of the hiring challenge."""

from io import BytesIO
from typing import Any, AnyStr, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi.applications import FastAPI
from fastapi.param_functions import File
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertModel, BertTokenizer

# Use GPU when possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pd.set_option("display.max_columns", None)
app = FastAPI()
# Stable Parameters
BATCH_SIZE = 64
K = 5
BERT_TRAIN = True
BERT_TEST = True


def generate_embeddings(textBatch: List[AnyStr]) -> Any:
    """Generate Bert embeddings for the batch."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    # Tokenize and encode the batch of text
    encodedBatch = tokenizer.batch_encode_plus(
        textBatch,
        add_special_tokens=True,
        max_length=768,
        padding='longest',
        return_attention_mask=True,
        return_tensors='pt',
    )
    # Move tensors to the appropriate device
    inputIds = encodedBatch['input_ids'].to(device)
    attentionMask = encodedBatch['attention_mask'].to(device)
    # Generate embeddings
    with torch.no_grad():
        outputs = model(inputIds, attention_mask=attentionMask)
    # Convert to numpy and return
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def create_json(npArray: NDArray[np.float64], mlb: MultiLabelBinarizer) -> Dict[Any, Any]:
    """Create a dict to be parsed as a json to the server for evaluation."""
    result = {}
    for i, row in enumerate(npArray):
        # Get indices of the top 5 probabilities in descending order
        top5Indices = np.argsort(row)[::-1][:5]
        # Convert indices to labels using the trained MultiLabelBinarizer
        top5Labels = mlb.classes_[top5Indices]
        # Create a dictionary for each row with rankings
        result[f"{i}"] = {rank: label for rank, label in enumerate(top5Labels)}
    return result


@app.post("/genres/train")
def train(file: bytes = File(...)) -> None:
    """Train a predictive model to rank movie genres based on their synopsis."""
    dfTrain = pd.read_csv(BytesIO(file))
    # Remove line with no genres
    dfTrain = dfTrain[dfTrain['genres'].str.len() > 0]
    # Train Bert if you want....
    if BERT_TRAIN:
        print("Creating Training Embeddings")
        # Initialize an empty array to store embeddings
        allEmbeddings = np.array([]).reshape(0, 768)  # 768 is the size of BERT-base embeddings
        # Process in batches
        for i in range(0, len(dfTrain), BATCH_SIZE):
            print(f"Processing batch {(i+1) // BATCH_SIZE} of {len(dfTrain) // BATCH_SIZE + 2}")
            # Create batch
            batchTexts = dfTrain['synopsis'].iloc[i:i + BATCH_SIZE].tolist()
            # Process batch
            batchEmbeddings = generate_embeddings(batchTexts)
            # Postprocess to get correect dims
            allEmbeddings = np.vstack((allEmbeddings, batchEmbeddings))
        # Add embeddings to the DataFrame
        dfTrain['bert'] = list(allEmbeddings)
        # Save model
        np.save('trainBert.npy', dfTrain['bert'].to_numpy())
        print("Finished training data")
    # ... or load the weights
    else:
        print("Load Training Embeddings")
        dfTrain['bert'] = np.load('trainBert.npy')
    # Split targets to list
    dfTrain["genres"] = dfTrain["genres"].apply(lambda x: x.split(" "))
    # Apply multilabel binarizer
    mlb = MultiLabelBinarizer()
    genres = mlb.fit_transform(dfTrain["genres"].to_list())
    # Save binarizer model
    joblib.dump(mlb, "mlb.model")
    # Train a classifier for each label
    # with weighted targets and all workers
    lr = OneVsRestClassifier(LogisticRegression(
        class_weight='balanced',
        solver='newton-cholesky'),
        verbose=10, n_jobs=-1)
    lr.fit(np.stack(dfTrain['bert'].values), genres)
    print("FInished training")
    # Save ML model
    joblib.dump(lr, "lr.joblib")


@app.post("/genres/predict")
def predict(file: bytes = File(...)) -> Dict[Any, Any]:
    """Train a predictive model to rank movie genres based on their synopsis."""
    dfPredict = pd.read_csv(BytesIO(file))
    # Load the models from the file
    lr = joblib.load("lr.joblib")
    mlb = joblib.load("mlb.model")
    # Create Bert embeddings....
    if BERT_TEST:
        print("Creating Testing Embeddings")
        # Initialize an empty array to store embeddings
        allEmbeddings = np.array([]).reshape(0, 768)
        # Process in batches
        for i in range(0, len(dfPredict), BATCH_SIZE):
            print(f"Processing batch {(i+1) // BATCH_SIZE} of {len(dfPredict) // BATCH_SIZE + 2}")
            # Create batch
            batchTexts = dfPredict['synopsis'].iloc[i:i + BATCH_SIZE].tolist()
            # Process
            batchEmbeddings = generate_embeddings(batchTexts)
            # Postprocess for right dims
            allEmbeddings = np.vstack((allEmbeddings, batchEmbeddings))
        # Add embeddings to the DataFrame
        dfPredict['bert'] = list(allEmbeddings)
        # Save embeddings
        np.save('testBert.npy', dfPredict['bert'].to_numpy())
        print("Finished Predicting data")
    # ... or load ready ones
    else:
        print("Load Testing Embeddings")
        dfPredict['bert'] = np.load('testBert.npy')
    # Create predictions
    yPred = lr.predict_proba(np.stack(dfPredict['bert'].values))
    # Create and return json
    jsonEnd = create_json(yPred, mlb)
    return jsonEnd
