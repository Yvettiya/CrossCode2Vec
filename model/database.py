import sqlite3
import os

class MySQLiteDB:
    def __init__(self, db_name):
        self.db_name = os.path.join('./dataset/',db_name)
        self.conn = None
        self.cursor = None
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
    
    def close(self):
        self.conn.close()
    
    def execute(self, query, data=None):
        if data is None:
            self.cursor.execute(query)
        else:
            self.cursor.execute(query, data)
        self.conn.commit()
    
    def executemany(self, query, data):
        self.cursor.executemany(query, data)
        self.conn.commit()
    
    def fetchall(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def fetchone(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchone()
    
    def delete_all(self, table_name):
        self.execute(f'DELETE FROM {table_name}')
    
    def insert(self, table_name, data):
        placeholders = ','.join(['?' for _ in range(len(data))])
        query = f'INSERT INTO {table_name} VALUES ({placeholders})'
        self.execute(query, data)
    
    def select_all(self, table_name):
        query = f'SELECT * FROM {table_name}'
        return self.fetchall(query)
    
    def select_columns_all(self, table_name,columns):
        query = f'SELECT {columns} FROM {table_name}'
        return self.fetchall(query)
    
    def select_by_id(self, table_name, id):
        query = f'SELECT * FROM {table_name} WHERE id={id}'
        return self.fetchone(query)
    
    def select_columns_by_condition(self, table_name, columns, condition):
        query = f'SELECT {columns} FROM {table_name} WHERE {condition}'
        return self.fetchall(query)
    
    def select_columns_withoutwhere(self, table_name, columns, notwherecondition):
        query = f'SELECT {columns} FROM {table_name} {notwherecondition}'
        return self.fetchall(query)

    def update_by_id(self, table_name, data, id):
        placeholders = ','.join([f'{column}=?' for column in data.keys()])
        query = f'UPDATE {table_name} SET {placeholders} WHERE id={id}'
        self.execute(query, tuple(data.values()))

    def get_datanum(self,table_name,column,condition=''):
        if condition == '':
            return self.select_columns_all(table_name=table_name,columns=f'count({column})')[0][0]
        else:
        
            return self.select_columns_by_condition(table_name=table_name,columns=f'count({column})',condition=condition)[0][0]
