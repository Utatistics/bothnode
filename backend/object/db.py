from pymongo import MongoClient, ReturnDocument
from pymongo.errors import ConnectionFailure, OperationFailure

from logging import getLogger

logger = getLogger(__name__)

def add_auth_to_mongo_connection_string(connection_string: str, username: str, password: str) -> str:
    if not connection_string.startswith("mongodb://"):
        raise ValueError("Invalid MongoDB connection string. Must start with 'mongodb://'.")

    protocol, rest = connection_string.split("://")
    
    auth_part = f"{username}:{password}@"
    new_connection_string = f"{protocol}://{auth_part}{rest}"

    return new_connection_string

class MongoDBClient:
    def __init__(self, uri: str, database_name: str):
        """Initialize the MongoDB client and select the database.
        
        Args
        ----
        uri: 
            MongoDB connection string
        database_name:
            Name of the database to use
        """
        try:
            self.client = MongoClient(uri)
            self.database = self.client[database_name]
        except ConnectionError as e:
            logger.error(f"Connection error: {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def insert_document(self, collection_name: str, document: dict) -> None:
        """Insert a single document into a collection.
        
        Args
        ----
        collection_name:
            Name of the collection
        document:
            The document to insert
        
        Returns
        -------
        inserted_id : 
            
        """
        try:
            collection = self.database[collection_name]
            result = collection.insert_one(document)
            return result.inserted_id
        except OperationFailure as e:
            logger.error(f"Operation failed: {e}")

    def find_document(self, collection_name: str, filter: dict=None, projection: dict=None, sort: list=None) -> dict:
        """Find a single document in a collection.
        
        Args
        ----
        collection_name: str
            Name of the collection
        filter: dict
            specifies the criteria for selecting documents
        projection : dict
            specifies which fields of the selected documents

        sort : list
        
        Returns
        -------
            The found document or None
        """
        try:
            collection = self.database[collection_name]
            document = collection.find_one(filter=filter, projection=projection, sort=sort)
            return document
        except OperationFailure as e:
            logger.error(f"Operation failed: {e}")

    def find_documents(self, collection_name: str, filter: dict=None, projection: dict=None) -> dict:
        """Find multiple documents in a collection.
        
        Args
        ----
        collection_name:
            Name of the collection
        filter:
            Query to filter the documents
        projection:
            Fields to include or exclude in the results
        Returns
        -------
            Cursor to iterate over the results
        """
        try:
            collection = self.database[collection_name]
            cursor = collection.find(filter=filter, projection=projection)
            return cursor
        except OperationFailure as e:
            logger.error(f"Operation failed: {e}")

    def update_document(self, collection_name: str, filter: dict=None, update: dict=None, upsert: bool=False):
        """Update a single document in a collection.
        
        ARgs
        ----
        collection_name:
            Name of the collection
        filter:
            Query to match the document
        update:
            The update operations to apply
        upsert:
            Whether to insert the document if it does not exist
        
        Returns
        -------
        The updated document or None
        """
        try:
            collection = self.database[collection_name]
            result = collection.find_one_and_update(filter-filter, update=update, upsert=upsert, return_document=ReturnDocument.AFTER)
            return result
        except OperationFailure as e:
            logger.error(f"Operation failed: {e}")

    def delete_document(self, collection_name: str, filter: dict=None):
        """Delete a single document from a collection.
        
        Args
        ----
        collection_name:
            Name of the collection
        filter:
            Query to match the document
        Returns
        -------
        The number of documents deleted
        """
        try:
            collection = self.database[collection_name]
            result = collection.delete_one(filter=filter)
            return result.deleted_count
        except OperationFailure as e:
            logger.error(f"Operation failed: {e}")

    def close(self):
        """
        Close the MongoDB connection.
        """
        self.client.close()

class GraphQLClient(object):
        def __init__(self) -> None:
            pass
        
class MerkelTree(object):
    def __init__(self) -> None:
        pass 
    
        pass
