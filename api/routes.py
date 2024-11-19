import datetime
from typing import List
from fastapi import APIRouter, HTTPException

from api.models import Transaction, TransactionCreate, Contract
from backend.util.config import Config
from backend.object.db import MongoDBClient, add_auth_to_mongo_connection_string
from api.models import ObjectId

from logging import getLogger

logger = getLogger(__name__)

# init router instance, which will be included in FastAPI application instance
router = APIRouter()

# Database setup 
config = Config()
db_config = config.DB_CONFIG
db_map = db_config['database']

connection_string = add_auth_to_mongo_connection_string(connection_string=db_config['connection_string'], username=db_config['init_username'], password=db_config['init_password'])

'''Transaction db
'''
@router.get("/transactions/{transaction_id}", response_model=Transaction)
async def get_transaction(transaction_id: str):
    db_client = MongoDBClient(uri=connection_string, database_name=db_map['transaction'])
    transaction = db_client.find_document('transactions', {"_id": ObjectId(transaction_id)})
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    transaction['id'] = str(transaction['_id'])
    return transaction

@router.get("/transactions/", response_model=List[Transaction])
async def get_transactions():
    db_client = MongoDBClient(uri=connection_string, database_name=db_map['transaction'])
    transactions = list(db_client.find_documents('transactions'))
    for transaction in transactions:
        transaction['id'] = str(transaction['_id'])
    return transactions

@router.post("/transactions/", response_model=Transaction)
async def create_transaction(transaction: TransactionCreate):
    db_client = MongoDBClient(uri=connection_string, database_name=db_map['transaction'])
    new_transaction = transaction.dict()
    new_transaction['timestamp'] = str(datetime.datetime.now())
    result = db_client.insert_document('transactions', new_transaction)
    new_transaction['id'] = str(result)
    return new_transaction

@router.delete("/transactions/{transaction_id}", response_model=dict)
async def delete_transaction(transaction_id: str):
    db_client = MongoDBClient(uri=connection_string, database_name=db_map['transaction'])
    result = db_client.delete_document('transactions', {"_id": ObjectId(transaction_id)})
    if result == 0:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return {"message": "Transaction deleted"}

'''contract_db
'''
@router.get("/contracts/", response_model=List[Contract])
async def get_transactions():
    db_client = MongoDBClient(uri=connection_string, database_name=db_map['contract'])
    deployments = list(db_client.find_documents('deployment'))
    for deployment in deployments:
        deployment['id'] = str(deployment['_id'])
    return deployments