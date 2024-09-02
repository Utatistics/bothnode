import datetime
from typing import List
from fastapi import APIRouter, HTTPException

from api.models import Transaction, TransactionCreate
from backend.util.config import Config
from backend.object.db import MongoDBClient, add_auth_to_mongo_connection_string
from api.models import ObjectId

from logging import getLogger

logger = getLogger(__name__)

router = APIRouter()

# Database setup 
config = Config()
db_config = config.DB_CONFIG
connection_string = add_auth_to_mongo_connection_string(connection_string=db_config['connection_string'], username=db_config['init_username'], password=db_config['init_password'])
db_client = MongoDBClient(uri=connection_string, database_name='transaction_db')

@router.get("/transactions/{transaction_id}", response_model=Transaction)
async def get_transaction(transaction_id: str):
    transaction = db_client.find_document('transactions', {"_id": ObjectId(transaction_id)})
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    transaction['id'] = str(transaction['_id'])
    return transaction

@router.get("/transactions/", response_model=List[Transaction])
async def get_transactions():
    transactions = list(db_client.find_documents('transactions'))
    for transaction in transactions:
        transaction['id'] = str(transaction['_id'])
    return transactions

@router.post("/transactions/", response_model=Transaction)
async def create_transaction(transaction: TransactionCreate):
    new_transaction = transaction.dict()
    new_transaction['timestamp'] = str(datetime.datetime.now())
    result = db_client.insert_document('transactions', new_transaction)
    new_transaction['id'] = str(result)
    return new_transaction

@router.delete("/transactions/{transaction_id}", response_model=dict)
async def delete_transaction(transaction_id: str):
    result = db_client.delete_document('transactions', {"_id": ObjectId(transaction_id)})
    if result == 0:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return {"message": "Transaction deleted"}
