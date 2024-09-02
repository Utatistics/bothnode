from pydantic import BaseModel
from typing import Optional, List
from bson import ObjectId

# Utility class to handle ObjectId in Pydantic models
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid ObjectId')
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class Transaction(BaseModel):
    id: Optional[PyObjectId] = None
    sender_address: str
    target_transaction: str
    payload: dict
    timestamp: str

    class Config:
        json_encoders = {
            ObjectId: str
        }

class TransactionCreate(BaseModel):
    sender_address: str
    target_transaction: str
    payload: dict
