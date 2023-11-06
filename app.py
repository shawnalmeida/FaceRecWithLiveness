from flask import Flask, render_template

import mysql.connector

app = Flask(__name__)

# MySQL configuration
db_config = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "Savageniggas@1",
    "database": "attendance_system"
}

conn = mysql.connector.connect(**db_config)

@app.route('/')
def display_data():
    cursor = conn.cursor()
    cursor.execute("SELECT employee_id, employee_name, attendance_count, clock_in FROM emp_attendance")
    data = cursor.fetchall()
    cursor.close()


    return render_template('display_dbdata.html', data= data)

if __name__ == '__main__':
    app.run()
