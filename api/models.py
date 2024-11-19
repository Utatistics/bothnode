from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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
            
class Contract(BaseModel):
    id: str 
    address: str
    abi: Optional[List[Dict[str, Any]]]  # Represents an array of dictionaries for ABI
    bytecode: Optional[str]  # Optional, can be None
    contractName: str
    network: str
    sourcePath: str
    timestamp: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),  # Converts datetime to ISO format in JSON
            ObjectId: str  # Ensures ObjectId is serialized as a string
        }
        allow_population_by_field_name = True  # Allows using Pydantic field names or MongoDB keys
