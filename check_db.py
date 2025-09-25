import sqlite3
import sys
import os

# データベースパスを設定
db_path = os.path.join(os.path.dirname(__file__), 'development_datas', 'database.db')

if not os.path.exists(db_path):
    print(f"Database not found: {db_path}")
    sys.exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# evaluation_result_tableの構造を確認
cursor.execute('PRAGMA table_info(evaluation_result_table)')
print('evaluation_result_table columns:')
for row in cursor.fetchall():
    print(row)

# サンプルデータを確認
cursor.execute('SELECT * FROM evaluation_result_table LIMIT 3')
print('\nSample rows from evaluation_result_table:')
for row in cursor.fetchall():
    print(row)

# algorithm_output_tableとの関連を確認
cursor.execute('PRAGMA table_info(algorithm_output_table)')
print('\nalgorithm_output_table columns:')
for row in cursor.fetchall():
    print(row)

# 外部キー制約を確認
cursor.execute('PRAGMA foreign_key_list(evaluation_result_table)')
print('\nForeign keys for evaluation_result_table:')
for row in cursor.fetchall():
    print(row)

conn.close()
