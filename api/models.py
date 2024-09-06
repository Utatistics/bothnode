from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId
from datetime import datetime

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, context=None):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid ObjectId')
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, schema):
        schema.update(type="string")
        return schema

class Transaction(BaseModel):
    id: str
    sender_address: str
    target_transaction: Optional[str]  # Allow target_transaction to be None
    payload: Optional[dict]  # Allow payload to be None
    timestamp: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

class TransactionCreate(BaseModel):
    sender_address: str
    target_transaction: Optional[str] = None
    payload: Optional[dict] = None
    timestamp: datetime