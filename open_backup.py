import sqlite3
import sys

if __name__ == '__main__':
    # arguments: backup_path
    backup_path = sys.argv[1]

    # Open backup read-only to avoid creating WAL/SHM alongside the snapshot
    ro_uri = f"file:{backup_path}?mode=ro&immutable=1"
    backup_max_id_conn = sqlite3.connect(ro_uri, uri=True)
    cursor = backup_max_id_conn.execute("SELECT MAX(id) FROM speech")
    result = cursor.fetchone()
    print(result)
    backup_max_id_conn.close()