import redis
from sqlalchemy import text

from app import app
from flask_sqlalchemy import SQLAlchemy

# load db
db = SQLAlchemy(app)
conn_pool = redis.ConnectionPool(host='localhost',port=6379)

def connect_redis():
    try:
        connect = redis.Redis(connection_pool=conn_pool)
        # connect = redis.StrictRedis(host='localhost', port=6379)
        print('Redis连接成功')
        return connect
    except Exception:
        print('Redis连接失败')
        return None
    

def parse_row_to_dict(row):
    """
    : 提取从mysql数据集中读取来的数据, 然后包裹成指定的数据结构
    :param row: query result from mysql: d_hits, d_hits_result
    :param dict_template: hit_dict, hit_result_dict
    :return:
    """
    return {
        'id': row[0],  # int(11) unsigned
        'projectId': row[1],  # varchar(36)
        'data': row[4],   # text
        'extras': row[3],  # text
        'status': row[4],  # varchar(50)
        'evaluation': row[5],  # varchar(50)
        'isGoldenHIT': row[6],  # tinyint(1)
        'goldenHITResultId': row[7],  # int(11) unsigned
        'notes': row[8],  # text
        'created_timestamp': row[9],  # timestamp, sql result 直接是 datetime
        'updated_timestamp': row[10],  # timestamp
        'isURL': row[11]  # tinyint(4)
    }

def query_d_hits(projectId=None):
    """
    :param project_name: voc, Retail Project Dataset
    :param status: 'done', 'notDone'
    :return:
    """
    sql = text("select * from d_hits where projectId='{}' and isnull(d_hits.correctResult)".format(projectId))
    res = db.session.execute(sql)
    db.session.commit()
    db.session.close()
    return res

def valid_userId(userId):
    sql = "select count(*) from d_users where oAuthId='{}'".format(userId)
    res = db.session.execute(sql)
    db.session.commit()
    db.session.close()
    return res.fetchone()[0] > 0    # exist user