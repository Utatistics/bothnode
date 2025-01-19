import torch
import numpy as np

from typing import Tuple

from backend.object.network import Network
from backend.object.block import Block

from pyod.models.ocsvm import OCSVM
from sklearn.preprocessing import StandardScaler

from logging import getLogger

logger = getLogger(__name__)

def call_node_embeddings():
    """
    """
    pass

def call_one_class_SVM(array: np.ndarray) -> Tuple:
    """train one-class SVM on the given array
    
    Args
    ----
    array : np.ndarray
        input array
        
    Retunrs
    -------
    anomoly_score : np.array
        anomoly score caclulated by the algorithm
    anomolies : list 
        indices for the detected anomolies
    """
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(array)

    # Step 2: Initialize the One-Class SVM
    nu = 0.1
    kernel = 'rbf'
    gamma = 'auto'
    
    ocsvm = OCSVM(nu=nu, kernel=kernel, gamma=gamma)  # Adjust 'nu' and 'gamma' as needed

    ocsvm.fit(embeddings_scaled)
    predictions = ocsvm.predict(embeddings_scaled)
    logger.info(f"Predictions: {predictions}")
    
    anomaly_scores = ocsvm.decision_function(embeddings_scaled).tolist()
    anomalies = np.where(predictions == 1)[0].tolist()
    anomaly_dict = dict(zip(anomalies, anomaly_scores))
    
    logger.debug(f"Anomaly scores: {anomaly_scores}")
    logger.info(f"Anomalies detected at indices: {anomaly_dict}")
    
    return anomaly_scores, anomaly_dict

def call_performance_metrics(label_address: list, anomaly_address: list) -> None:
    """calculate basic metrics for model validation
    
    Args
    ----
    label_address : list
        list of anamoly addresses (as lables)
    anamoly_address : list
        list of predicted anamoly addresses
        
    """
    logger.info(f'{None}')