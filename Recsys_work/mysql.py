import pymysql
import yaml
import pandas as pd

with open("conf.yaml") as ya:
    cfg = yaml.safe_load(ya)

mysql_host = cfg["mysql"]["host"]
mysql_user = cfg["mysql"]["user"]
mysql_password = cfg["mysql"]["passwd"]
mysql_port = cfg["mysql"]["port"]
mysql_db = cfg["mysql"]["database"]
mysql_charset = cfg["mysql"]["charset"]

def connect():
    '''Connect to Mysql database'''
    try:
        db = pymysql.connect(
            host=mysql_host,
            port=mysql_port,
            user=mysql_user,
            passwd=mysql_password,
            db=mysql_db,
            charset=mysql_charset
        )
        return db
    except Exception:
        raise Exception("Database connection failed")

def implement(sql):
    '''Execute SQL statement'''
    db = connect()
    cursor = db.cursor()
    for i in range(1):
        if sql== '':
            sql = """SELECT count(datetime) FROM dl_hash """
        try:
            cursor.execute(sql)
            result = cursor.fetchone()
            db.commit()
            print('Search result：', result)
        except Exception:
            db.rollback()
            print("Query failed")

    cursor.close()
    db.close()
