import sqlite3
import os

# データベースパス
db_path = os.path.join(os.path.dirname(__file__), '..', 'development_datas', 'database.db')

if not os.path.exists(db_path):
    print(f"Database not found: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=== EVALUATION RESULT ID = 3 ===")

# evaluation_result_table の内容を確認
cursor.execute("SELECT * FROM evaluation_result_table WHERE evaluation_result_ID = 3")
eval_result = cursor.fetchone()
if eval_result:
    cursor.execute("PRAGMA table_info(evaluation_result_table)")
    columns = [row[1] for row in cursor.fetchall()]
    print("evaluation_result_table columns:", columns)
    print("evaluation_result_table data:", dict(zip(columns, eval_result)))
else:
    print("No data found in evaluation_result_table for ID=3")

# 関連するテーブルを確認
print("\n=== ALGORITHM OUTPUT TABLE ===")
cursor.execute("SELECT * FROM algorithm_output_table LIMIT 5")
algo_outputs = cursor.fetchall()
if algo_outputs:
    cursor.execute("PRAGMA table_info(algorithm_output_table)")
    columns = [row[1] for row in cursor.fetchall()]
    print("algorithm_output_table columns:", columns)
    for i, row in enumerate(algo_outputs):
        print(f"Row {i+1}:", dict(zip(columns, row)))
else:
    print("No data in algorithm_output_table")

# 外部キー関係を確認
print("\n=== FOREIGN KEY RELATIONSHIPS ===")
cursor.execute("PRAGMA foreign_key_list(evaluation_result_table)")
fk_list = cursor.fetchall()
if fk_list:
    for fk in fk_list:
        print(f"Foreign key: {fk}")
else:
    print("No foreign keys defined")

conn.close()
